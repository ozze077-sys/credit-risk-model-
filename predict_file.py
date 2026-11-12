"""
predict_file.py
===============
Score a file of NEW customers (outcome unknown) with the saved model and write
the results to an Excel file in outputs/. Nothing is shown on the dashboard and
the file is never used for training.

    python predict_file.py "credit_risk_unknown_data (1).xlsx"
    python predict_file.py                      # uses the only file without a target column

The output workbook has two sheets:
* Predictions - the customer's original columns (same order as the input) plus
  the model's estimated probability of default, the 0/1 prediction, the risk
  segment, the inputs that pushed the estimate up most, and any data warnings.
* Summary     - the model used, the thresholds, and counts per risk segment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader import (FEATURE_COLS, ID_COL, PROJECT_DIR, SUPPORTED_EXTENSIONS, TARGET_ALIASES, DatasetError,
                         _normalise, _read_table, detect_header_row, load_customers_to_score)
from model_training import MODEL_PATH, OUTPUTS_DIR
from prediction import (DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD, SEGMENTS, ModelError,
                        assign_segments, explain_prediction, load_model, predict_probability,
                        validate_customer_input)


def find_files_to_score(folder: Path = PROJECT_DIR) -> list[Path]:
    """Customer files in the project folder that have the input columns but NO target column."""
    found = []
    for p in sorted(folder.iterdir()):
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS) or p.name.startswith("~$"):
            continue
        try:
            preview = _read_table(p, header=None, nrows=10)
            header = {_normalise(v) for v in preview.iloc[detect_header_row(preview)]}
        except DatasetError:
            continue
        if not header & TARGET_ALIASES:
            found.append(p)
    return found


def risk_drivers(bundle: dict, row: pd.DataFrame, n: int = 3) -> str:
    """The inputs that raise this customer's estimate most, compared with an average customer."""
    exp = explain_prediction(bundle, row, top_n=len(FEATURE_COLS))
    up = exp[exp["Contribution"] > 0].head(n)
    return "; ".join(f"{f} ({v})" for f, v in zip(up["Feature"], up["Value"])) or "none above average"


def score_file(path: Path, medium: float = DEFAULT_MEDIUM_THRESHOLD, high: float = DEFAULT_HIGH_THRESHOLD,
               outputs_dir: Path = OUTPUTS_DIR) -> tuple[Path, pd.DataFrame]:
    bundle = load_model(MODEL_PATH)
    customers, info = load_customers_to_score(path)

    # Data checks: we still score every row, but flag values outside the expected ranges/codes.
    warnings = [" | ".join(validate_customer_input({c: (None if pd.isna(v) else v) for c, v in row.items()}))
                for row in customers[FEATURE_COLS].to_dict("records")]

    proba = predict_probability(bundle, customers)
    threshold = bundle.get("decision_threshold", 0.5)

    # Keep the customer's original columns in the original order, then add the results.
    original = pd.read_excel(path, header=info["header_row_excel"] - 1) if path.suffix.lower() != ".csv" \
        else pd.read_csv(path, header=info["header_row_excel"] - 1)
    original = original.loc[:, [c for c in original.columns if not str(c).startswith("Unnamed")]]
    result = original.copy()
    result["Predicted_Default_Probability"] = proba.round(4)
    result["Predicted_Default"] = (proba >= threshold).astype(int)
    result["Risk_Segment"] = assign_segments(proba, medium, high)
    result["Top_Risk_Drivers"] = [risk_drivers(bundle, customers.iloc[[i]]) for i in range(len(customers))]
    result["Data_Warnings"] = warnings

    seg_counts = result["Risk_Segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    summary = pd.DataFrame([
        ("Input file", path.name),
        ("Customers scored", len(result)),
        ("Model", f"Logistic regression (class weight: {bundle['class_weight'] or 'none'})"),
        ("Model trained at", bundle["trained_at"].replace("T", " ")),
        ("Model trained on", ", ".join(bundle.get("dataset_files") or [bundle.get("dataset_file") or ""])),
        ("Model test ROC-AUC", round(bundle["metrics"]["roc_auc"], 3)),
        ("Predicted_Default = 1 when probability >=", f"{threshold:.0%}"),
        ("Risk segments", f"Low < {medium:.0%} <= Medium < {high:.0%} <= High (demonstration thresholds)"),
        *[(f"{s} customers", int(n)) for s, n in seg_counts.items()],
        ("Predicted defaults (1)", int(result["Predicted_Default"].sum())),
        ("Average predicted probability", f"{proba.mean():.1%}"),
        ("Rows with data warnings", int(sum(bool(w) for w in warnings))),
        ("Non-numeric values treated as missing", sum(info["non_numeric_values_converted_to_missing"].values())),
        ("Note", "Probabilities are model estimates, not guarantees. Top_Risk_Drivers describe how the model "
                 "reacts to the inputs, not causes of default."),
    ], columns=["Item", "Value"])

    outputs_dir.mkdir(parents=True, exist_ok=True)
    out = outputs_dir / f"{path.stem}_predictions.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        result.to_excel(xl, sheet_name="Predictions", index=False)
        summary.to_excel(xl, sheet_name="Summary", index=False)
        _format(xl, result)
    return out, result


def _format(xl: pd.ExcelWriter, result: pd.DataFrame) -> None:
    """Readable Excel: bold frozen header, sensible widths, probability as a percentage."""
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="0B1F3A")
    for name, ws in xl.sheets.items():
        ws.freeze_panes = "B2" if name == "Predictions" else "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
        for i, col in enumerate(ws.columns, 1):
            longest = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, longest + 2), 70)
    ws = xl.sheets["Predictions"]
    prob_col = result.columns.get_loc("Predicted_Default_Probability") + 1
    for (cell,) in ws.iter_rows(min_row=2, min_col=prob_col, max_col=prob_col):
        cell.number_format = "0.0%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score new customers with the saved credit-risk model.")
    parser.add_argument("file", nargs="?", help="Excel/CSV file of customers (no target column needed)")
    parser.add_argument("--medium", type=float, default=DEFAULT_MEDIUM_THRESHOLD * 100, help="Medium-risk threshold %%")
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH_THRESHOLD * 100, help="High-risk threshold %%")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        candidates = find_files_to_score()
        if len(candidates) != 1:
            names = ", ".join(p.name for p in candidates) or "none"
            raise SystemExit(f"Please name the file to score. Files without a target column found: {names}")
        path = candidates[0]
    if not 0 < args.medium < args.high < 100:
        raise SystemExit("Thresholds must satisfy 0 < medium < high < 100.")

    try:
        out, result = score_file(path, args.medium / 100, args.high / 100)
    except (DatasetError, ModelError) as exc:
        raise SystemExit(f"Could not score the file: {exc}")
    print(f"Scored {len(result)} customers from {path.name}")
    print(result["Risk_Segment"].value_counts().reindex(SEGMENTS, fill_value=0).to_string())
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
