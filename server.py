"""
server.py - Credit Risk dashboard as a plain Python web server + one HTML page
=============================================================================
Start with:   python server.py        then open http://127.0.0.1:8000

How it works (beginner summary)
-------------------------------
* Python's built-in ``http.server`` module answers browser requests. No web
  framework is needed.
* ``GET /`` returns ONE HTML page (web/index.html). That page contains its own
  CSS and JavaScript.
* The JavaScript asks this server for data with ``fetch('/api/...')``. Every
  ``/api/...`` endpoint answers with JSON (or a CSV/JSON file for downloads).
* Charts are built in Python (charts.py) and sent as Plotly JSON; the page draws
  them with Plotly.js, served locally from the installed plotly package (no
  internet needed).
* The dataset and model are loaded ONCE at start-up. The model is trained only
  when no valid saved model exists (or the dataset changed), or when you press
  "Retrain model".

API endpoints
-------------
GET  /api/status                         dataset + model info, default thresholds (or a load error)
GET  /api/overview?medium=30&high=50     KPI cards and Overview charts
GET  /api/exploration                    KPIs, group default-rate charts, limit charts, correlation
GET  /api/exploration/bill-payment?stat=Mean|Median
GET  /api/exploration/distribution?var=LIMIT_BAL
GET  /api/segmentation?medium&high       segment cards, histogram, summary, filter options
GET  /api/segmentation/customers?medium&high&segment=&age=&limit=&split=&top=25
GET  /api/segmentation/export.csv?(same filters)       download filtered predictions
POST /api/segmentation/save   {"medium":30,"high":50}  write outputs/*.csv
GET  /api/customer?id=1&medium&high      one existing customer: details, probability, explanation
GET  /api/form-defaults                  field list, medians and code options for the new-customer form
POST /api/predict   {"values":{...},"medium":30,"high":50}   probability for a new customer
GET  /api/performance                    test-set metrics, explanations, charts, report
GET  /api/performance/threshold?t=0.5    metrics at another decision threshold
GET  /api/quality                        data-quality report
GET  /api/download/metrics.json | test_predictions.csv?medium&high | data_quality.csv
POST /api/retrain   {"balanced": false}  retrain the model
"""

from __future__ import annotations

import json
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly
from plotly.utils import PlotlyJSONEncoder

import charts
from data_analysis import (AGE_LABELS, LIMIT_LABELS, add_groups, age_group, bill_payment_comparison,
                           correlation_matrix, credit_limit_stats, default_rate_by, limit_range)
from data_loader import (BILL_COLS, EDUCATION_LABELS, FEATURE_COLS, ID_COL, MARRIAGE_LABELS, MONTHS,
                         PAY_AMT_COLS, PAY_STATUS_COLS, PAY_STATUS_MONTH, PROJECT_DIR, SEX_LABELS,
                         SOURCE_COL, TARGET_COL, TARGET_LABELS, DatasetError, category_label, data_quality_report,
                         export_quality_report, friendly_name, load_dataset, pay_status_label)
from model_training import METRICS_PATH, MODEL_PATH, OUTPUTS_DIR, evaluate, save_bundle, train_model
from prediction import (DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD, SEGMENTS, ModelError,
                        assign_segments, build_prediction_table, explain_prediction, export_predictions,
                        load_model, model_matches_data, predict_probability, segment_summary,
                        validate_customer_input)

HOST, PORT = "127.0.0.1", 8000
INDEX_HTML = PROJECT_DIR / "web" / "index.html"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


class ApiError(Exception):
    """An error caused by the request (bad parameter etc.). Sent to the browser as JSON."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Data and model (loaded once)
# ---------------------------------------------------------------------------
def train_and_save(df: pd.DataFrame, info: dict, class_weight: str | None) -> dict:
    bundle = train_model(df, class_weight=class_weight, dataset_info=info)
    save_bundle(bundle)
    export_predictions(df, bundle)
    export_quality_report(data_quality_report(df, info))
    return bundle


def load_or_train(df: pd.DataFrame, info: dict) -> dict:
    """Reuse the saved model if it matches this dataset; otherwise train a new one."""
    try:
        bundle = load_model(MODEL_PATH)
        if model_matches_data(bundle, info):
            return bundle
        print("Saved model was trained on a different dataset version - retraining…")
    except ModelError as exc:
        print(f"{exc} Training a new model…")
    return train_and_save(df, info, class_weight=None)


class AppState:
    """Everything the endpoints need. A lock keeps retraining and reading from clashing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.error = None
        try:
            self.df, self.info = load_dataset()
            self.quality = data_quality_report(self.df, self.info)
            self.data = add_groups(self.df)
            self.overall = self.df[TARGET_COL].mean()
            self.set_bundle(load_or_train(self.df, self.info))
        except (DatasetError, ModelError) as exc:
            self.error = str(exc)
        except Exception as exc:  # e.g. model training failed
            self.error = f"The model could not be trained: {exc}"

    def set_bundle(self, bundle: dict) -> None:
        """Store a model and score every customer once (segments are re-assigned cheaply later)."""
        pred = build_prediction_table(self.df, bundle)
        pred["Age_Group"] = age_group(pred["AGE"])
        pred["Credit_Limit_Range"] = limit_range(pred["LIMIT_BAL"])
        self.bundle, self.base_pred = bundle, pred

    def predictions(self, medium: float, high: float) -> pd.DataFrame:
        pred = self.base_pred.copy()
        pred["Risk_Segment"] = assign_segments(pred["Predicted_Default_Probability"], medium, high)
        return pred


STATE: AppState | None = None


# ---------------------------------------------------------------------------
# Helpers that turn pandas objects into JSON-friendly structures
# ---------------------------------------------------------------------------
def table(df: pd.DataFrame, formats: dict | None = None) -> dict:
    """{"columns": [...], "rows": [[...], ...], "formats": {column: "pct"|"int"|"dec1"|"dec3"}}."""
    df = df.astype(object).where(df.notna(), None)
    return {"columns": [str(c) for c in df.columns], "rows": df.values.tolist(), "formats": formats or {}}


def kpi(label: str, value: str, note: str = "", kind: str = "hist") -> dict:
    """kind: "hist" (historical), "pred" (model) or "plain" (no tag)."""
    return {"label": label, "value": value, "note": note, "kind": kind}


def money(v) -> str:
    return f"NT$ {v:,.0f}"


def thresholds(q: dict) -> tuple[float, float]:
    """Read ?medium=30&high=50 (percent). Validates 0 < medium < high < 100."""
    medium = q_float(q, "medium", DEFAULT_MEDIUM_THRESHOLD * 100)
    high = q_float(q, "high", DEFAULT_HIGH_THRESHOLD * 100)
    if not 0 < medium < high < 100:
        raise ApiError("Thresholds must satisfy 0 < medium < high < 100.")
    return medium / 100, high / 100


def q_float(q: dict, key: str, default: float) -> float:
    try:
        return float(q.get(key, [default])[0])
    except ValueError:
        raise ApiError(f"'{key}' must be a number.")


def q_list(q: dict, key: str, allowed: list[str], sep: str = ",") -> list[str]:
    """Separated filter values (file names may contain commas, so they use "|"). Missing = everything selected."""
    if key not in q:
        return list(allowed)
    raw = q[key][0]
    return [v for v in raw.split(sep) if v in allowed] if raw else []


# ---------------------------------------------------------------------------
# Endpoint implementations
# ---------------------------------------------------------------------------
def api_status(_q) -> dict:
    if STATE.error:
        return {"ok": False, "error": STATE.error}
    b = STATE.bundle
    return {
        "ok": True,
        "dataset": {"file": STATE.info["file_name"], "rows": len(STATE.df),
                    "files": [i["file_name"] for i in STATE.info["files"]]},
        "model": {"class_weight": b["class_weight"], "trained_at": b["trained_at"], "n_train": b["n_train"],
                  "n_test": b["n_test"]},
        "defaults": {"medium": DEFAULT_MEDIUM_THRESHOLD * 100, "high": DEFAULT_HIGH_THRESHOLD * 100},
    }


def api_overview(q) -> dict:
    medium, high = thresholds(q)
    df, pred = STATE.df, STATE.predictions(medium, high)
    n_def = int(df[TARGET_COL].sum())
    counts = pred["Risk_Segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    n_high = int(counts["High risk"])
    late = df["PAY_0"] >= 2
    return {
        "low_share": float(counts["Low risk"] / len(pred)),
        "summary": [
            f"We looked at {len(df):,} credit-card customers. {n_def:,} of them ({one_in(STATE.overall)}) missed "
            "their payment the following month.",
            f"The model sorted every customer into a risk group: {counts['Low risk']:,} low, "
            f"{counts['Medium risk']:,} medium and {n_high:,} high risk.",
            f"The clearest warning sign is paying late recently: customers who were 2 or more months late in "
            f"September missed their next payment {df.loc[late, TARGET_COL].mean():.0%} of the time, compared with "
            f"{df.loc[~late, TARGET_COL].mean():.0%} for everyone else.",
            "These are estimates based on past behaviour. They show who is more likely to miss a payment, not who "
            "certainly will.",
        ],
        "kpis": [
            kpi("Customers", f"{len(df):,}", "in the data"),
            kpi("Missed a payment", f"{STATE.overall:.1%}", f"{one_in(STATE.overall)} customers"),
            kpi("Customers who missed", f"{n_def:,}", f"out of {len(df):,}"),
            kpi("Average credit limit", f"{df['LIMIT_BAL'].mean() / 1000:,.0f}K", "New Taiwan dollars (NT$)"),
            kpi("Flagged as high risk", f"{n_high:,}", f"{n_high / len(pred):.0%} of customers", "pred"),
            kpi("Average estimated risk", f"{pred['Predicted_Default_Probability'].mean():.0%}",
                "the model's average estimate", "pred"),
        ],
        "charts": {
            "outcome": charts.outcome_bar(df),
            "pay0": charts.rate_bar(default_rate_by(df, "PAY_0"), "Missed payments by September payment status",
                                    "Payment status in September", STATE.overall),
            "segments": charts.segment_bar(segment_summary(pred), medium, high),
            "probability": charts.probability_histogram(pred["Predicted_Default_Probability"], medium, high),
            "importance": charts.importance_bar(STATE.bundle["permutation_importance"].head(10),
                                                "What the model looks at most"),
        },
    }


def one_in(rate: float) -> str:
    """0.221 -> 'about 1 in 5'. Easier to picture than a percentage."""
    if rate <= 0:
        return "none"
    n = round(1 / rate)
    return "about half" if n <= 2 else f"about 1 in {n}"


def in_100(p: float) -> str:
    """0.656 -> '66 out of 100'."""
    return f"{round(p * 100)} out of 100"


def api_exploration(_q) -> dict:
    df, data, overall = STATE.df, STATE.data, STATE.overall
    lim = credit_limit_stats(df)
    n_def = int(df[TARGET_COL].sum())
    variables = [{"value": "LIMIT_BAL", "label": "Credit limit"}, {"value": "AGE", "label": "Age"}]
    variables += [{"value": c, "label": f"Bill – {MONTHS[i]}"} for i, c in enumerate(BILL_COLS, 1)]
    variables += [{"value": c, "label": f"Amount paid – {MONTHS[i]}"} for i, c in enumerate(PAY_AMT_COLS, 1)]
    by_age, by_limit = default_rate_by(data, "Age_Group"), default_rate_by(data, "Credit_Limit_Range")
    by_pay = default_rate_by(df, "PAY_0")
    lo_lim, hi_lim = by_limit.iloc[0], by_limit.iloc[-1]
    worst_age, best_age = by_age.loc[by_age["Default_Rate"].idxmax()], by_age.loc[by_age["Default_Rate"].idxmin()]
    paid_full = by_pay.loc[by_pay["Group"] == pay_status_label(-1), "Default_Rate"]
    two_late = by_pay.loc[by_pay["Group"] == pay_status_label(2), "Default_Rate"]
    summary = [
        f"Customers with smaller credit limits missed payments more often: {lo_lim['Default_Rate']:.0%} with a "
        f"limit {lo_lim['Group'].replace('Under', 'under')} vs {hi_lim['Default_Rate']:.0%} with {hi_lim['Group']}.",
        f"Age matters less: the rate ranges from {best_age['Default_Rate']:.0%} ({best_age['Group']}) to "
        f"{worst_age['Default_Rate']:.0%} ({worst_age['Group']}).",
    ]
    if len(paid_full) and len(two_late):
        summary.insert(0, f"Recent late payment is the strongest sign: {two_late.iloc[0]:.0%} of customers who were "
                          f"2 months late in September missed their next payment, vs {paid_full.iloc[0]:.0%} of those "
                          "who had paid in full.")
    summary.append("These are patterns in past data. They do not prove that one thing causes another.")
    return {
        "summary": summary,
        "kpis": [
            kpi("Paid on time", f"{len(df) - n_def:,}", "customers"), kpi("Missed payment", f"{n_def:,}", "customers"),
            kpi("Share who missed", f"{overall:.1%}", one_in(overall)),
            kpi("Average credit limit", f"{lim['mean'] / 1000:,.0f}K", f"NT$ · typical {lim['median'] / 1000:,.0f}K"),
            kpi("Lowest credit limit", f"{lim['min'] / 1000:,.0f}K", "NT$"),
            kpi("Highest credit limit", f"{lim['max'] / 1000:,.0f}K", "NT$"),
        ],
        "charts": {
            "age": charts.rate_bar(by_age, "Missed payments by age", "Age", overall),
            "limit": charts.rate_bar(by_limit, "Missed payments by credit limit", "Credit limit (NT$)", overall),
            "pay0": charts.rate_bar(by_pay, "Missed payments by September payment status",
                                    "Payment status in September", overall),
            "education": charts.rate_bar(default_rate_by(df, "EDUCATION"), "Missed payments by education",
                                         "Education", overall),
            "marriage": charts.rate_bar(default_rate_by(df, "MARRIAGE"), "Missed payments by marital status",
                                        "Marital status", overall),
            "limit_by_outcome": charts.limit_by_outcome(df),
            "limit_box": charts.limit_box(data),
            "correlation": charts.correlation_heatmap(correlation_matrix(df)),
        },
        "variables": variables,
    }


def api_bill_payment(q) -> dict:
    stat = q.get("stat", ["Mean"])[0]
    if stat not in ("Mean", "Median"):
        raise ApiError("stat must be Mean or Median.")
    cmp = bill_payment_comparison(STATE.df)
    return {"bill": charts.bill_payment_bars(cmp, "BILL_AMT", "Bill", stat),
            "pay": charts.bill_payment_bars(cmp, "PAY_AMT", "Amount paid", stat)}


def api_distribution(q) -> dict:
    col = q.get("var", ["LIMIT_BAL"])[0]
    if col not in ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS:
        raise ApiError(f"Unknown variable '{col}'.")
    label = friendly_name(col).lower()
    fig, hidden, lo, hi = charts.distribution_hist(STATE.data, col, label)
    text = ("Blue = customers who paid, orange = customers who missed. Each colour adds up to 100%, so you can "
            "compare the shapes even though the groups differ in size.")
    if hidden:
        text += f" A few very large values ({hidden:,} customers above NT$ {hi:,.0f}) are left out so the chart stays readable."
    return {"chart": fig, "caption": text}


def api_segmentation(q) -> dict:
    medium, high = thresholds(q)
    pred = STATE.predictions(medium, high)
    seg = segment_summary(pred)
    cards = [kpi(r["Risk_Segment"], f"{r['Customers']:,}",
                 f"{r['Share']:.0%} of customers · typical chance "
                 + ("–" if pd.isna(r["Avg_Probability"]) else f"{r['Avg_Probability']:.0%}"),
                 "pred") for _, r in seg.iterrows()]
    sentences = [f"<b>{r['Risk_Segment']}</b>: {r['Customers']:,} customers. The model gives them on average a "
                 f"{r['Avg_Probability']:.0%} chance of missing a payment, and {r['Actual_Default_Rate']:.0%} of them "
                 "really did." for _, r in seg.iterrows() if r["Customers"]]
    summary = seg.rename(columns={"Risk_Segment": "Risk group", "Avg_Probability": "Estimated chance (average)",
                                  "Actual_Default_Rate": "Actually missed"})[
        ["Risk group", "Customers", "Share", "Estimated chance (average)", "Actually missed"]]
    return {
        "summary_text": sentences + ["When the two numbers are close, the estimates can be trusted."],
        "cards": cards,
        "histogram": charts.probability_histogram(pred["Predicted_Default_Probability"], medium, high),
        "summary": table(summary, {"Customers": "int", "Share": "pct", "Estimated chance (average)": "pct",
                                   "Actually missed": "pct"}),
        "filters": {"segment": SEGMENTS, "age": AGE_LABELS, "limit": LIMIT_LABELS, "split": ["Train", "Test"],
                    "source": source_files()},
    }


def source_files() -> list[str]:
    return [i["file_name"] for i in STATE.info["files"]]


def filtered_predictions(q) -> pd.DataFrame:
    medium, high = thresholds(q)
    pred = STATE.predictions(medium, high)
    return pred[pred["Risk_Segment"].isin(q_list(q, "segment", SEGMENTS))
                & pred["Age_Group"].isin(q_list(q, "age", AGE_LABELS))
                & pred["Credit_Limit_Range"].isin(q_list(q, "limit", LIMIT_LABELS))
                & pred["Data_Split"].isin(q_list(q, "split", ["Train", "Test"]))
                & pred[SOURCE_COL].isin(q_list(q, "source", source_files(), sep="|"))]


def api_segment_customers(q) -> dict:
    view = filtered_predictions(q)
    top_n = int(min(max(q_float(q, "top", 25), 1), 500))
    cols = [ID_COL, "Predicted_Default_Probability", "Risk_Segment", "Actual_Default", "LIMIT_BAL", "AGE", "PAY_0",
            "BILL_AMT1", "PAY_AMT1", "Data_Split", SOURCE_COL]
    top = view.nlargest(top_n, "Predicted_Default_Probability")[cols].copy()
    top["Actual_Default"] = top["Actual_Default"].map({1: "Yes", 0: "No"})
    top["PAY_0"] = top["PAY_0"].map(pay_status_label)
    top["Data_Split"] = top["Data_Split"].map({"Train": "Learning", "Test": "Testing"})
    top = top.rename(columns={
        ID_COL: "Customer ID", "Predicted_Default_Probability": "Estimated chance", "Risk_Segment": "Risk group",
        "Actual_Default": "Actually missed?", "LIMIT_BAL": "Credit limit", "AGE": "Age",
        "PAY_0": "Payment status – Sep", "BILL_AMT1": "Bill – Sep", "PAY_AMT1": "Amount paid – Sep",
        "Data_Split": "Used for", SOURCE_COL: "Data file"})
    return {"count": len(view), "total": len(STATE.df),
            "table": table(top, {"Estimated chance": "pct", "Credit limit": "int", "Bill – Sep": "int",
                                 "Amount paid – Sep": "int"})}


def export_frame(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.drop(columns=["Age_Group", "Credit_Limit_Range"]).copy()
    out["Predicted_Default_Probability"] = out["Predicted_Default_Probability"].round(4)
    return out


def api_segment_save(body) -> dict:
    medium, high = thresholds({k: [v] for k, v in body.items()})
    paths = export_predictions(STATE.df, STATE.bundle, medium_threshold=medium, high_threshold=high)
    return {"saved": [str(p.relative_to(OUTPUTS_DIR.parent)) for p in paths]}


def customer_details(row: pd.Series) -> list[list[str]]:
    items = [("Credit limit", money(row["LIMIT_BAL"])), ("Age", f"{row['AGE']:.0f}"),
             ("Gender", category_label("SEX", row["SEX"])), ("Education", category_label("EDUCATION", row["EDUCATION"])),
             ("Marital status", category_label("MARRIAGE", row["MARRIAGE"]))]
    items += [(f"Payment status – {PAY_STATUS_MONTH[c]}", pay_status_label(row[c])) for c in PAY_STATUS_COLS]
    items += [(f"Bill / amount paid – {MONTHS[i]}", f"{money(row[b])} / {money(row[p])}")
              for i, (b, p) in enumerate(zip(BILL_COLS, PAY_AMT_COLS), 1)]
    return [list(i) for i in items]


def prediction_result(row: pd.DataFrame, proba: float, medium: float, high: float) -> dict:
    """Probability, risk group and a plain-English explanation of one prediction."""
    exp = explain_prediction(STATE.bundle, row)
    overall = STATE.overall
    reason = lambda r: f"{friendly_name(r['Feature'])} ({r['Value']})"  # noqa: E731
    up = [reason(r) for _, r in exp[exp["Contribution"] > 0].head(3).iterrows()]
    down = [reason(r) for _, r in exp[exp["Contribution"] < 0].head(3).iterrows()]
    ratio = proba / overall
    if ratio >= 1.15:
        compare = f"That is about {ratio:.1f} times the average customer ({in_100(overall)})."
    elif ratio <= 0.87:
        compare = f"That is lower than the average customer ({in_100(overall)})."
    else:
        compare = f"That is close to the average customer ({in_100(overall)})."
    return {
        "probability": proba,
        "segment": str(assign_segments([proba], medium, high)[0]),
        "segment_colors": charts.SEGMENT_COLORS,
        "overall_rate": overall,
        "headline": f"About {in_100(proba)} customers like this one missed their next payment.",
        "compare": compare,
        "summary": {"direction": "higher" if proba > overall else "lower", "up": up, "down": down},
        "explanation_chart": charts.contribution_bar(exp),
    }


def api_customer(q) -> dict:
    medium, high = thresholds(q)
    try:
        cid = int(q.get("id", [""])[0])
    except ValueError:
        raise ApiError("Customer ID must be a whole number.")
    match = STATE.base_pred[STATE.base_pred[ID_COL] == cid]
    if match.empty:
        raise ApiError(f"No customer with ID {cid} was found.", HTTPStatus.NOT_FOUND)
    r = match.iloc[0]
    result = prediction_result(match[FEATURE_COLS], float(r["Predicted_Default_Probability"]), medium, high)
    result.update({"id": cid, "details": [["Source file", r[SOURCE_COL]]] + customer_details(r),
                   "actual": TARGET_LABELS[int(r["Actual_Default"])], "split": r["Data_Split"]})
    return result


def api_form_defaults(_q) -> dict:
    med = STATE.df[FEATURE_COLS].median()
    codes = [{"value": c, "label": pay_status_label(c)} for c in range(-2, 9)]

    def options(labels):
        return [{"value": k, "label": v} for k, v in labels.items()]

    profile = [
        {"name": "LIMIT_BAL", "label": "Credit limit (NT$)", "type": "number", "min": 1, "max": 10_000_000,
         "step": 10_000, "value": int(med["LIMIT_BAL"])},
        {"name": "AGE", "label": "Age", "type": "number", "min": 18, "max": 100, "step": 1, "value": int(med["AGE"])},
        {"name": "SEX", "label": "Gender", "type": "select", "options": options(SEX_LABELS), "value": 2},
        {"name": "EDUCATION", "label": "Education", "type": "select", "options": options(EDUCATION_LABELS), "value": 2},
        {"name": "MARRIAGE", "label": "Marital status", "type": "select", "options": options(MARRIAGE_LABELS), "value": 2},
    ]
    months = [[
        {"name": s, "label": f"{MONTHS[i]}: how did they pay?", "type": "select", "options": codes, "value": 0},
        {"name": b, "label": f"{MONTHS[i]}: bill (NT$)", "type": "number", "min": -10_000_000, "max": 10_000_000,
         "step": 1000, "value": int(med[b])},
        {"name": p, "label": f"{MONTHS[i]}: amount paid (NT$)", "type": "number", "min": 0, "max": 10_000_000,
         "step": 500, "value": int(med[p])},
    ] for i, (s, b, p) in enumerate(zip(PAY_STATUS_COLS, BILL_COLS, PAY_AMT_COLS), 1)]
    return {"profile": profile, "months": months}


def api_predict(body) -> dict:
    medium, high = thresholds({k: [v] for k, v in body.items() if k in ("medium", "high")})
    raw = body.get("values") or {}
    values, errors = {}, []
    for col in FEATURE_COLS:  # convert text from the form into numbers
        try:
            values[col] = float(raw[col]) if raw.get(col) not in (None, "") else None
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number.")
    if not errors:  # range / code checks only make sense once everything is a number
        errors = validate_customer_input(values)
    if errors:
        # Show column codes with their plain names, e.g. "AGE must be ..." -> "Age must be ...".
        errors = [f"{friendly_name(e.split(' ', 1)[0])} {e.split(' ', 1)[1]}" if " " in e else e for e in errors]
        raise ApiError("Please fix the following: " + " | ".join(errors))
    row = pd.DataFrame([values])[FEATURE_COLS]
    proba = float(predict_probability(STATE.bundle, row)[0])
    return prediction_result(row, proba, medium, high)


METRIC_HELP = {
    "Missed payments caught": "Recall. Of the customers who really missed a payment, the share the model had "
                              "warned about. Missing them is usually the most expensive mistake.",
    "Warnings that were right": "Precision. Of the customers the model warned about, the share who really missed "
                                "a payment. The rest were false alarms.",
    "Overall correct calls": "Accuracy. Share of all customers the model called right. It looks good easily: saying "
                             "'everyone will pay' would already score {base:.0%}.",
    "Ranking quality": "ROC-AUC. How well the model puts risky customers above safe ones. 0.5 = coin flip, "
                       "1.0 = perfect.",
    "Balance score": "F1 score. One number combining 'caught' and 'right'. Higher is better.",
}


def quality_word(auc: float) -> str:
    return "good" if auc >= 0.8 else "fair to good" if auc >= 0.7 else "modest" if auc >= 0.6 else "weak"


def performance_cards(mt: dict, base: dict | None = None) -> list[dict]:
    """The five headline numbers in everyday words (technical name in the note)."""
    def delta(key):
        return "" if base is None else f"{(mt[key] - base[key]) * 100:+.0f} points vs the 50% level"
    return [
        kpi("Missed payments caught", f"{mt['recall']:.0%}", delta("recall") or "recall", "pred"),
        kpi("Warnings that were right", f"{mt['precision']:.0%}", delta("precision") or "precision", "pred"),
        kpi("Overall correct calls", f"{mt['accuracy']:.0%}", delta("accuracy") or "accuracy", "pred"),
        kpi("Balance score", f"{mt['f1']:.0%}", delta("f1") or "F1 score", "pred"),
    ]


def api_performance(_q) -> dict:
    b = STATE.bundle
    m = b["metrics"]
    values = {"Missed payments caught": m["recall"], "Warnings that were right": m["precision"],
              "Overall correct calls": m["accuracy"], "Ranking quality": m["roc_auc"], "Balance score": m["f1"]}
    rep = pd.DataFrame(m["classification_report"]).T.loc[["No default", "Default", "macro avg", "weighted avg"]]
    rep = rep.reset_index()
    rep.columns = ["Class", "Precision", "Recall", "F1 score", "Support"]
    rep["Class"] = rep["Class"].replace({"No default": "Paid on time", "Default": "Missed payment"})
    gap = m["test_mean_predicted_probability"] - m["test_actual_default_rate"]
    if abs(gap) > 0.05:
        cal = {"level": "warn", "text": f"Not quite. On average the estimates are {abs(gap) * 100:.0f} points too "
               f"{'high' if gap > 0 else 'low'}. " + (
                   "That is expected with the 'balanced' training option, which deliberately pushes estimates up. "
                   "Treat them as a ranking, not literal chances." if b["class_weight"] == "balanced"
                   else "Read the percentages with care.")}
    else:
        cal = {"level": "ok", "text": "Yes. When the model says, for example, 30%, about 30 in 100 of those "
                                      "customers really missed a payment."}
    plain = [
        f"The model was tested on {b['n_test']:,} customers it had never seen before.",
        f"Of every 100 customers who really missed a payment, it had warned about {round(m['recall'] * 100)} of them.",
        f"When it gave a warning, it was right about {round(m['precision'] * 100)} times out of 100.",
        f"Its percentages are realistic: on average it estimated {m['test_mean_predicted_probability']:.0%}, and "
        f"{m['test_actual_default_rate']:.0%} actually missed.",
        f"Ranking quality is {m['roc_auc']:.2f} (0.5 = coin flip, 1.0 = perfect): {quality_word(m['roc_auc'])} for "
        "this kind of data.",
        "Trade-off: it misses many customers who later miss a payment. To catch more, lower the warning level "
        "below. You will also get more false alarms.",
    ]
    by_source = pd.DataFrame(m.get("by_source", [])).rename(columns={
        "source": "Source file", "n_test": "Test customers", "actual_default_rate": "Actual default rate",
        "mean_predicted_probability": "Avg predicted probability", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1 score", "roc_auc": "ROC-AUC"})
    return {
        "by_source": table(by_source, {"Test customers": "int", "Actual default rate": "pct",
                                       "Avg predicted probability": "pct", "Accuracy": "dec3", "Precision": "dec3",
                                       "Recall": "dec3", "F1 score": "dec3", "ROC-AUC": "dec3"}),
        "n_test": b["n_test"], "seed": b["random_seed"], "decision_threshold": b["decision_threshold"],
        "plain": plain,
        "kpis": performance_cards(m) + [kpi("Ranking quality", f"{m['roc_auc']:.2f}",
                                            f"0.5 = coin flip, 1 = perfect · ROC-AUC", "pred")],
        "help": [{"name": k, "value": values[k], "text": t.format(base=m["baseline_accuracy_always_no_default"])}
                 for k, t in METRIC_HELP.items()],
        "test_default_rate": m["test_actual_default_rate"],
        "report": table(rep, {"Precision": "dec3", "Recall": "dec3", "F1 score": "dec3", "Support": "int"}),
        "calibration": {"mean_predicted": m["test_mean_predicted_probability"],
                        "actual": m["test_actual_default_rate"], "brier": m["brier_score"], "message": cal},
        "charts": {
            "confusion": charts.confusion_heatmap(m["confusion_matrix"]),
            "roc": charts.roc_chart(m), "pr": charts.pr_chart(m), "calibration": charts.calibration_chart(m),
            "coefficients": charts.coefficient_bar(b["coefficients"]),
            "importance": charts.importance_bar(b["permutation_importance"].head(15),
                                                "What the model looks at most", height=520),
        },
    }


def api_threshold(q) -> dict:
    t = q_float(q, "t", 0.5)
    if not 0 < t < 1:
        raise ApiError("t must be between 0 and 1.")
    b = STATE.bundle
    tr, base = b["test_results"], b["metrics"]
    mt = evaluate(tr["Actual_Default"].values, tr["Predicted_Default_Probability"].values, threshold=t)
    same = abs(t - b["decision_threshold"]) < 1e-9
    cards = performance_cards(mt, None if same else base)
    for c in cards:
        c["kind"] = "plain"
    return {"cards": cards, "confusion": mt["confusion_matrix"], "threshold": t}


def api_quality(_q) -> dict:
    q = STATE.quality
    s = q["summary"]
    describe = q["describe"].reset_index().rename(columns={"index": "Variable"})
    describe.insert(1, "Meaning", describe["Variable"].map(friendly_name))
    columns = q["columns"].copy()
    columns.insert(1, "Meaning", columns["Column"].map(friendly_name))
    missing, dup_ids = s["Total missing values"], s["Duplicate customer IDs"]
    summary = [
        f"The data has {s['Rows used (valid target)']:,} customers from {s['Source files']} file"
        f"{'s' if s['Source files'] != 1 else ''}, each with {s['Columns used'] - 2} pieces of information plus the "
        "outcome.",
        ("Nothing is missing" if missing == 0 else f"{missing:,} cells are empty (the model fills them with typical "
         "values)") + (" and every customer ID is unique." if dup_ids == 0 else f", and {dup_ids:,} customer IDs repeat."),
        "A few codes are not in the official description (for example education codes 0, 5 and 6). They were kept "
        "as their own groups, not deleted.",
        "Negative bill amounts mean the customer had paid more than they owed (a credit balance). That is normal.",
    ]
    return {
        "summary": summary,
        "sources": table(q["sources"].rename(columns={"File": "Data file", "Defaults": "Missed payment",
                                                      "Default rate %": "Missed %"}),
                         {"Customers": "int", "Missed payment": "int", "Missed %": "dec1"}),
        "kpis": [
            kpi("Customers", f"{s['Rows used (valid target)']:,}", f"read {s['Rows read']:,}", "plain"),
            kpi("Data files", f"{s['Source files']}", "combined for learning", "plain"),
            kpi("Empty cells", f"{missing:,}", "", "plain"),
            kpi("Repeated customer IDs", f"{dup_ids:,}", "", "plain"),
            kpi("Repeated rows", f"{s['Fully duplicated rows']:,}",
                f"{s['Rows identical except ID']} look alike apart from the ID", "plain"),
            kpi("Columns", f"{s['Columns used']}", "incl. ID and outcome", "plain"),
        ],
        "file": s["File"], "target_name": s["Original target column name"],
        "columns": table(columns, {"Missing": "int", "Missing %": "dec1", "Unique values": "int"}),
        "target": table(q["target"], {"Customers": "int", "Share %": "dec1"}),
        "unusual": table(q["unusual_values"], {"Rows": "int"}),
        "codes": table(q["unexpected_codes"], {"Rows": "int"}),
        "describe": table(describe, {c: "dec1" for c in q["describe"].columns}),
    }


def api_retrain(body) -> dict:
    class_weight = "balanced" if body.get("balanced") else None
    bundle = train_and_save(STATE.df, STATE.info, class_weight)
    STATE.set_bundle(bundle)
    m = bundle["metrics"]
    return {"message": f"Done. The new model catches {m['recall']:.0%} of missed payments "
                       f"(ranking quality {m['roc_auc']:.2f}).",
            "status": api_status({})}


GET_ROUTES = {
    "/api/status": api_status, "/api/overview": api_overview, "/api/exploration": api_exploration,
    "/api/exploration/bill-payment": api_bill_payment, "/api/exploration/distribution": api_distribution,
    "/api/segmentation": api_segmentation, "/api/segmentation/customers": api_segment_customers,
    "/api/customer": api_customer, "/api/form-defaults": api_form_defaults, "/api/performance": api_performance,
    "/api/performance/threshold": api_threshold, "/api/quality": api_quality,
}
POST_ROUTES = {"/api/segmentation/save": api_segment_save, "/api/predict": api_predict, "/api/retrain": api_retrain}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "CreditRisk/1.0"

    # --- sending responses ---
    def send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK,
                   filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Only the large, unchanging Plotly.js file may be cached; the page and data are always fresh.
        cacheable = content_type.startswith("application/javascript")
        self.send_header("Cache-Control", "max-age=3600" if cacheable else "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(data, cls=PlotlyJSONEncoder).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def send_csv(self, df: pd.DataFrame, filename: str) -> None:
        self.send_bytes(df.to_csv(index=False).encode("utf-8"), "text/csv; charset=utf-8", filename=filename)

    # --- request handling ---
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query, keep_blank_values=True)
        self.handle_safely(lambda: self.route_get(url.path, q))

    def do_POST(self):
        url = urlparse(self.path)

        def run():
            route = POST_ROUTES.get(url.path)
            if not route:
                raise ApiError("Unknown endpoint.", HTTPStatus.NOT_FOUND)
            self.require_data()
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise ApiError("The request body must be JSON.")
            with STATE.lock:
                self.send_json(route(body))

        self.handle_safely(run)

    def route_get(self, path: str, q: dict) -> None:
        if path in ("/", "/index.html"):
            return self.send_bytes(INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/static/plotly.min.js":
            return self.send_bytes(PLOTLY_JS.read_bytes(), "application/javascript; charset=utf-8")
        if path == "/api/status":
            return self.send_json(api_status(q))
        if path.startswith("/api/download/"):
            self.require_data()
            return self.download(path.rsplit("/", 1)[-1], q)
        if path == "/api/segmentation/export.csv":
            self.require_data()
            return self.send_csv(export_frame(filtered_predictions(q)), "customer_predictions_filtered.csv")
        route = GET_ROUTES.get(path)
        if not route:
            raise ApiError("Not found.", HTTPStatus.NOT_FOUND)
        self.require_data()
        with STATE.lock:
            self.send_json(route(q))

    def download(self, name: str, q: dict) -> None:
        if name == "metrics.json":
            content = (METRICS_PATH.read_bytes() if METRICS_PATH.exists()
                       else json.dumps(STATE.bundle["metrics"], cls=PlotlyJSONEncoder, indent=2).encode("utf-8"))
            return self.send_bytes(content, "application/json", filename="model_metrics.json")
        if name == "test_predictions.csv":
            medium, high = thresholds(q)
            test = export_frame(STATE.predictions(medium, high))
            test = test[test["Data_Split"] == "Test"]
            test["Correct_Prediction"] = (test["Predicted_Default"] == test["Actual_Default"]).astype(int)
            return self.send_csv(test, "test_predictions.csv")
        if name == "data_quality.csv":
            path = export_quality_report(STATE.quality)
            return self.send_bytes(path.read_bytes(), "text/csv; charset=utf-8", filename="data_quality_summary.csv")
        raise ApiError("Unknown download.", HTTPStatus.NOT_FOUND)

    def require_data(self) -> None:
        if STATE.error:
            raise ApiError(STATE.error, HTTPStatus.SERVICE_UNAVAILABLE)

    def handle_safely(self, fn) -> None:
        """Turn any error into a JSON message the page can show, instead of a broken connection."""
        try:
            fn()
        except ApiError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": f"Unexpected server error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main() -> None:
    global STATE
    print("Loading data and model…")
    STATE = AppState()
    if STATE.error:
        print(f"WARNING: {STATE.error}\nThe page will show this message.")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Dashboard running at http://{HOST}:{PORT}  (press Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
