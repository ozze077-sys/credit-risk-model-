"""
data_loader.py
==============
Finds, loads, cleans and validates the UCI "Default of Credit Card Clients" dataset.

Key ideas for beginners
-----------------------
* We never modify the original file. We only *read* it into a pandas DataFrame.
* The UCI Excel file has an extra header row (X1, X2, ... Y) above the real
  column names (ID, LIMIT_BAL, ...). We detect the real header row automatically
  instead of hard-coding it.
* Column names are standardised (e.g. "default payment next month" ->
  "default.payment.next.month") so the rest of the project can rely on them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = (".xls", ".xlsx", ".csv")

# ---------------------------------------------------------------------------
# Column definitions (the "data dictionary")
# ---------------------------------------------------------------------------
ID_COL = "ID"
TARGET_COL = "default.payment.next.month"

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Numerical features: real quantities (money, years).
NUMERIC_FEATURES = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS
# Categorical features: codes that label a category, not a quantity.
# Repayment status is treated as categorical because -2, -1 and 0 are
# special codes (see PAY_STATUS_LABELS), not points on a numeric scale.
CATEGORICAL_FEATURES = DEMOGRAPHIC_CATEGORICAL_COLS + PAY_STATUS_COLS
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLS = [ID_COL] + FEATURE_COLS + [TARGET_COL]

# Normalised names that may refer to the target column in other copies of the data.
TARGET_ALIASES = {
    "DEFAULT_PAYMENT_NEXT_MONTH",
    "DEFAULT_PAYMENT",
    "DEFAULT_NEXT_MONTH",
    "DEFAULT",
    "Y",
}
# Other equivalent names seen in different copies of the dataset.
COLUMN_ALIASES = {"PAY_1": "PAY_0"}

# Human-readable labels (from the UCI documentation).
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Others"}
# Everyday words for the outcome ("default" = did not make the required payment).
TARGET_LABELS = {0: "Paid on time", 1: "Missed payment"}

# Months each column refers to (the data covers April-September 2005).
MONTHS = {1: "Sep 2005", 2: "Aug 2005", 3: "Jul 2005", 4: "Jun 2005", 5: "May 2005", 6: "Apr 2005"}
PAY_STATUS_MONTH = dict(zip(PAY_STATUS_COLS, MONTHS.values()))

# Plain-English names for the columns, used in charts, tables and forms.
_SHORT_MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
FRIENDLY_NAMES = {
    ID_COL: "Customer ID", "LIMIT_BAL": "Credit limit", "AGE": "Age", "SEX": "Gender",
    "EDUCATION": "Education", "MARRIAGE": "Marital status", TARGET_COL: "Missed payment (actual)",
    **{c: f"Payment status – {m}" for c, m in zip(PAY_STATUS_COLS, _SHORT_MONTHS)},
    **{c: f"Bill – {m}" for c, m in zip(BILL_COLS, _SHORT_MONTHS)},
    **{c: f"Amount paid – {m}" for c, m in zip(PAY_AMT_COLS, _SHORT_MONTHS)},
}


def friendly_name(column: str) -> str:
    """'PAY_0' -> 'Payment status – Sep'. Also handles model names like 'PAY_0 = 2 months late'."""
    column = str(column)
    if " = " in column:
        col, value = column.split(" = ", 1)
        return f"{FRIENDLY_NAMES.get(col, col)}: {value}"
    return FRIENDLY_NAMES.get(column, column)


def pay_status_label(code) -> str:
    """Translate a repayment-status code into plain English.

    UCI documentation: -1 = paid duly; 1..9 = payment delay of that many months.
    Codes -2 and 0 are not described in the original documentation. The
    interpretation widely used (based on clarification from the dataset donor) is:
    -2 = no credit used / no balance that month, 0 = revolving credit
    (the minimum payment was made but a balance was carried).
    """
    if pd.isna(code):
        return "Missing"
    code = int(code)
    if code == -2:
        return "No bill that month"
    if code == -1:
        return "Paid in full"
    if code == 0:
        return "Paid the minimum"
    if code >= 1:
        return f"{code} month{'s' if code > 1 else ''} late"
    return f"Unknown code {code}"


def category_label(column: str, code) -> str:
    """Readable label for SEX / EDUCATION / MARRIAGE / PAY_x codes."""
    if pd.isna(code):
        return "Missing"
    code = int(code)
    if column in PAY_STATUS_COLS:
        return pay_status_label(code)
    mapping = {"SEX": SEX_LABELS, "EDUCATION": EDUCATION_LABELS, "MARRIAGE": MARRIAGE_LABELS}.get(column, {})
    return mapping.get(code, f"Undocumented code {code}")


class DatasetError(Exception):
    """Raised with a beginner-friendly message when the data cannot be used."""


# ---------------------------------------------------------------------------
# Finding and reading the file
# ---------------------------------------------------------------------------
def _normalise(name) -> str:
    """'default payment next month' -> 'DEFAULT_PAYMENT_NEXT_MONTH'."""
    return re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")


KNOWN_NORMALISED = {_normalise(c) for c in REQUIRED_COLS} | TARGET_ALIASES | set(COLUMN_ALIASES)


def find_dataset_files(folder: Path | str = PROJECT_DIR) -> list[Path]:
    """Return EVERY spreadsheet/CSV in the project folder that looks like credit-card data.

    We don't assume filenames. Each supported file in the top-level folder is
    checked for a header row with the expected column names; files that don't
    match (e.g. unrelated spreadsheets) are skipped. Subfolders such as outputs/
    are not searched, so exported results are never re-used as training data.
    """
    folder = Path(folder)
    candidates = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                        and not p.name.startswith("~$"))  # skip Excel lock files
    if not candidates:
        raise DatasetError(
            f"No dataset file found in '{folder}'. Put the UCI file "
            f"({', '.join(SUPPORTED_EXTENSIONS)}) in the project folder."
        )
    found, problems = [], []
    for path in candidates:
        try:
            preview = _read_table(path, header=None, nrows=10)
            header = {_normalise(v) for v in preview.iloc[detect_header_row(preview)]}
        except DatasetError as exc:
            problems.append(f"- {path.name}: {exc}")
            continue
        if header & TARGET_ALIASES:
            found.append(path)
        else:  # e.g. new customers to score with predict_file.py - never used for training
            problems.append(f"- {path.name}: no target column (outcome unknown), so it is not used for training")
    if not found:
        raise DatasetError("None of the files look like the UCI credit-card dataset:\n" + "\n".join(problems))
    return found


def _read_table(path: Path, header, nrows=None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, header=header, nrows=nrows)
        if suffix in (".xls", ".xlsx"):
            # Use the first sheet. (The UCI file has a single sheet called "Data".)
            return pd.read_excel(path, header=header, nrows=nrows, sheet_name=0)
    except ImportError as exc:
        raise DatasetError(
            f"A package needed to read '{suffix}' files is missing ({exc}). "
            "Run: pip install -r requirements.txt"
        ) from exc
    except Exception as exc:  # corrupted / unreadable file
        raise DatasetError(f"Could not read the file: {exc}") from exc
    raise DatasetError(f"Unsupported file format '{suffix}'. Use .xls, .xlsx or .csv.")


def detect_header_row(raw: pd.DataFrame, max_rows: int = 10) -> int:
    """Return the row number (0-based) that contains the real column names.

    ``raw`` is the sheet read *without* a header. Each of the first rows is
    scored by how many cells match known column names; the best-scoring row wins.
    """
    preview = raw.head(max_rows)
    scores = [sum(_normalise(v) in KNOWN_NORMALISED for v in row) for row in preview.itertuples(index=False)]
    best = int(np.argmax(scores)) if scores else 0
    if not scores or scores[best] < 5:
        raise DatasetError("Could not find a header row with the expected column names (ID, LIMIT_BAL, ...).")
    return best


def _read_raw(path: Path, require_target: bool = True) -> tuple[pd.DataFrame, dict]:
    """Read the file with the right header row and standardise column names.

    The file is read only once, without a header; we then pick the real header
    row and use the rows below it as data. ``require_target=False`` is used for
    files of new customers whose outcome is not known yet.
    """
    raw = _read_table(path, header=None)
    header_row = detect_header_row(raw)
    names = [f"Unnamed: {i}" if pd.isna(v) else str(v).strip() for i, v in enumerate(raw.iloc[header_row])]
    df = raw.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = names
    info = {"file_name": path.name, "header_row_excel": header_row + 1, "original_columns": names}

    # Drop empty "Unnamed" index-like columns that sometimes appear in exports.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=unnamed)
    info["dropped_empty_columns"] = [str(c) for c in unnamed]

    # Standardise names.
    rename, target_original = {}, None
    for col in df.columns:
        norm = _normalise(col)
        norm = COLUMN_ALIASES.get(norm, norm)
        if norm in TARGET_ALIASES and target_original is None:
            rename[col], target_original = TARGET_COL, str(col)
        else:
            rename[col] = norm
    df = df.rename(columns=rename)
    if target_original is None and require_target:
        raise DatasetError(
            "The target column (e.g. 'default payment next month') was not found. "
            "Please check that this is the UCI Default of Credit Card Clients file."
        )
    info["target_original_name"] = target_original

    needed = REQUIRED_COLS if require_target else [ID_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DatasetError(f"Expected columns are missing: {', '.join(missing)}")
    return df, info


def file_fingerprint(path: Path) -> str:
    """MD5 hash of the file. Lets us check that a saved model matches the current data."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
SOURCE_COL = "Source_File"  # which file each row came from (never a model input)


def load_dataset(path: Path | str | None = None, folder: Path | str = PROJECT_DIR) -> tuple[pd.DataFrame, dict]:
    """Load, validate and lightly clean the data.

    * ``path`` given  -> load just that file.
    * no ``path``     -> load ALL dataset files found in the project folder and
      stack them into one table. A ``Source_File`` column records where each
      row came from. Customer IDs must be unique across files.

    Returns (df, info). ``info`` records what was found/changed so the
    Data Quality page can report it.

    Cleaning is deliberately minimal:
    * Values are converted to numbers; anything unreadable becomes missing (NaN)
      and is later filled in by the model pipeline's imputers.
    * Rows with a missing/invalid target are excluded (we can't learn from them).
    * Unusual codes (e.g. EDUCATION 0/5/6, PAY_x -2) are KEPT - they are real
      categories in the data, not errors we should delete.
    """
    paths = [Path(path)] if path else find_dataset_files(folder)
    parts, infos = [], []
    for p in paths:
        try:
            df, info = _load_one(p)
        except DatasetError as exc:
            raise DatasetError(f"{p.name}: {exc}") from exc
        parts.append(df.assign(**{SOURCE_COL: p.name}))
        infos.append(info)
    df = pd.concat(parts, ignore_index=True)

    clashes = df[ID_COL][df[ID_COL].duplicated()].unique()
    if len(paths) > 1 and len(clashes):
        raise DatasetError(f"The same customer ID appears in more than one file (e.g. {list(clashes[:5])}). "
                           "IDs must be unique across files so customers can be told apart.")

    info = {
        "files": infos,
        "file_name": " + ".join(i["file_name"] for i in infos),
        "header_row_excel": " / ".join(str(i["header_row_excel"]) for i in infos),
        "target_original_name": " / ".join(dict.fromkeys(i["target_original_name"] for i in infos)),
        "rows_read": sum(i["rows_read"] for i in infos),
        "rows_used": len(df),
        "rows_dropped_missing_target": sum(i["rows_dropped_missing_target"] for i in infos),
        # One fingerprint for the whole set of files: changes if any file is added, removed or edited.
        "fingerprint": hashlib.md5("|".join(sorted(f"{i['file_name']}:{i['fingerprint']}" for i in infos))
                                   .encode()).hexdigest(),
    }
    return df, info


def load_customers_to_score(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Read a file of customers whose outcome is NOT known yet (no target column needed).

    Columns are matched by name, so their order doesn't matter. Values that are not
    numbers become missing (NaN); the model's imputers fill them in, and the number
    of such values is reported in ``info``.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    raw, info = _read_raw(path, require_target=False)
    df = raw[[ID_COL] + FEATURE_COLS].copy()
    if df.empty:
        raise DatasetError("The file has no customer rows.")
    info["non_numeric_values_converted_to_missing"] = {}
    for col in df.columns:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        added = int(df[col].isna().sum() - before)
        if added:
            info["non_numeric_values_converted_to_missing"][col] = added
    info["has_target"] = info["target_original_name"] is not None
    return df, info


def _load_one(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate a single file."""
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    df, info = _read_raw(path)
    info["file_path"] = str(path)
    info["fingerprint"] = file_fingerprint(path)

    df = df[REQUIRED_COLS].copy()
    if df.empty:
        raise DatasetError("The dataset has no rows.")
    info["rows_read"] = len(df)

    # Convert every column to a number. Text that isn't a number becomes NaN.
    non_numeric = {}
    for col in REQUIRED_COLS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        added = int(df[col].isna().sum() - before)
        if added:
            non_numeric[col] = added
    info["non_numeric_values_converted_to_missing"] = non_numeric

    # Validate target encoding: must be 0 (no default) / 1 (default).
    target_values = set(df[TARGET_COL].dropna().unique())
    if not target_values <= {0, 1}:
        raise DatasetError(
            f"The target column should contain only 0 and 1, but found: {sorted(target_values)[:10]}"
        )
    if len(target_values) < 2:
        raise DatasetError("The target column contains only one class, so a model cannot be trained.")
    bad_target = df[TARGET_COL].isna()
    info["rows_dropped_missing_target"] = int(bad_target.sum())
    df = df.loc[~bad_target].reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    info["rows_used"] = len(df)
    if df.empty:
        raise DatasetError("No rows have a valid target value.")
    return df, info


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------
DOCUMENTED_CODES = {
    "SEX": set(SEX_LABELS),
    "EDUCATION": set(EDUCATION_LABELS),
    "MARRIAGE": set(MARRIAGE_LABELS),
    **{c: set(range(-2, 10)) for c in PAY_STATUS_COLS},
}


def data_quality_report(df: pd.DataFrame, info: dict) -> dict:
    """Build the tables shown on the Data Quality page."""
    sources = df[SOURCE_COL] if SOURCE_COL in df else pd.Series(info.get("file_name"), index=df.index)
    file_infos = {i["file_name"]: i for i in info.get("files", [])}
    per_file = df.groupby(sources)[TARGET_COL].agg(["size", "sum", "mean"])
    sources_table = pd.DataFrame({
        "File": per_file.index,
        "Customers": per_file["size"].values,
        "Defaults": per_file["sum"].values,
        "Default rate %": (per_file["mean"] * 100).round(1).values,
        "Header row": [file_infos.get(f, {}).get("header_row_excel") for f in per_file.index],
        "Target column name": [file_infos.get(f, {}).get("target_original_name") for f in per_file.index],
    })
    df = df[REQUIRED_COLS]  # the checks below look at the data columns only
    columns = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique values": df.nunique().values,
    })

    unexpected = []
    for col, allowed in DOCUMENTED_CODES.items():
        counts = df[col].value_counts(dropna=True)
        for code, n in counts.items():
            if col in PAY_STATUS_COLS and code in (-2, 0):
                meaning = "'no balance / no consumption'" if code == -2 else "'revolving credit (minimum paid)'"
                note = f"Not in the original UCI documentation; commonly interpreted as {meaning}. Kept."
            elif code not in allowed:
                note = "Undocumented code; kept as its own category (not deleted)."
            else:
                continue
            unexpected.append({"Column": col, "Code": int(code), "Rows": int(n), "Note": note})
    unexpected = pd.DataFrame(unexpected, columns=["Column", "Code", "Rows", "Note"])

    unusual = []
    for col in BILL_COLS:
        n = int((df[col] < 0).sum())
        if n:
            unusual.append({"Check": f"{col} < 0", "Rows": n,
                            "Interpretation": "Negative bill = credit balance (customer overpaid). Valid, kept."})
    for col in PAY_AMT_COLS + ["LIMIT_BAL"]:
        n = int((df[col] < 0).sum())
        if n:
            unusual.append({"Check": f"{col} < 0", "Rows": n, "Interpretation": "Negative amount - unexpected."})
    n_age = int(((df["AGE"] < 18) | (df["AGE"] > 100)).sum())
    unusual.append({"Check": "AGE outside 18-100", "Rows": n_age, "Interpretation": "Implausible ages."})
    unusual = pd.DataFrame(unusual)

    target_counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    target = pd.DataFrame({
        "Outcome": [TARGET_LABELS[k] for k in target_counts.index],
        "Code": target_counts.index,
        "Customers": target_counts.values,
        "Share %": (target_counts / target_counts.sum() * 100).round(2).values,
    })

    summary = {
        "File": info.get("file_name"),
        "Source files": len(sources_table),
        "Real header found on Excel row": info.get("header_row_excel"),
        "Original target column name": info.get("target_original_name"),
        "Rows read": info.get("rows_read"),
        "Rows used (valid target)": info.get("rows_used"),
        "Columns used": df.shape[1],
        "Total missing values": int(df.isna().sum().sum()),
        "Fully duplicated rows": int(df.duplicated().sum()),
        "Rows identical except ID": int(df.drop(columns=[ID_COL]).duplicated().sum()),
        "Duplicate customer IDs": int(df[ID_COL].duplicated().sum()),
        "Missing customer IDs": int(df[ID_COL].isna().sum()),
    }
    describe = df[NUMERIC_FEATURES].describe().T.round(1)
    return {"summary": summary, "sources": sources_table, "columns": columns, "unexpected_codes": unexpected,
            "unusual_values": unusual, "target": target, "describe": describe}


def export_quality_report(report: dict, outputs_dir: Path = PROJECT_DIR / "outputs") -> Path:
    """Save the data-quality summary as one CSV: overall checks, then per-column details."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / "data_quality_summary.csv"
    overall = pd.DataFrame({"Section": "Overall", "Item": list(report["summary"]),
                            "Value": [str(v) for v in report["summary"].values()]})
    cols = report["columns"]
    per_col = pd.DataFrame({
        "Section": "Column",
        "Item": cols["Column"],
        "Value": [f"type={t}; missing={m} ({p}%); unique={u}" for t, m, p, u in
                  zip(cols["Data type"], cols["Missing"], cols["Missing %"], cols["Unique values"])],
    })
    codes = report["unexpected_codes"]
    code_rows = pd.DataFrame({"Section": "Unusual code", "Item": codes["Column"] + " = " + codes["Code"].astype(str),
                              "Value": codes["Rows"].astype(str) + " rows. " + codes["Note"]})
    src = report["sources"]
    source_rows = pd.DataFrame({"Section": "Source file", "Item": src["File"],
                                "Value": [f"customers={c}; defaults={d}; default rate={r}%" for c, d, r in
                                          zip(src["Customers"], src["Defaults"], src["Default rate %"])]})
    pd.concat([overall, source_rows, per_col, code_rows], ignore_index=True).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    data, meta = load_dataset()
    report = data_quality_report(data, meta)
    for k, v in report["summary"].items():
        print(f"{k}: {v}")
    print(report["sources"].to_string(index=False))
    print(report["target"].to_string(index=False))
