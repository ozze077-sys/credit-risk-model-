"""
build_static_site.py - build the GitHub Pages version of the dashboard
======================================================================
Run with:   python build_static_site.py

GitHub Pages can only host static files, so it cannot run server.py. This script:

1. loads the data and the saved model exactly like server.py does;
2. computes every answer that never changes (exploration charts, model
   performance, data quality, form defaults) by calling server.py's own
   endpoint functions;
3. embeds the customer table, the model coefficients and those answers as JSON;
4. writes ONE self-contained ``index.html`` at the project root: the normal
   web/index.html page plus web/static_api.js, which answers the page's
   ``/api/...`` requests in the browser (thresholds, filters, customer lookup,
   new-customer estimates, warning-level slider, CSV downloads).

The output opens directly from disk and is deployed by .github/workflows/pages.yml.
Re-run it (and push) whenever the data or the model changes.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

import numpy as np
from plotly.utils import PlotlyJSONEncoder

import charts
import server
from data_analysis import AGE_LABELS, LIMIT_LABELS
from data_loader import (BILL_COLS, CATEGORICAL_FEATURES, EDUCATION_LABELS, FEATURE_COLS, FRIENDLY_NAMES, ID_COL,
                         MARRIAGE_LABELS, MONTHS, PAY_AMT_COLS, PAY_STATUS_COLS, PAY_STATUS_MONTH, PROJECT_DIR,
                         SEX_LABELS, SOURCE_COL, TARGET_LABELS, export_quality_report)
from model_training import METRICS_PATH
from prediction import ALLOWED_CODES, INPUT_RULES, SEGMENTS, explain_prediction

OUTPUT = PROJECT_DIR / "index.html"
PAGE = PROJECT_DIR / "web" / "index.html"
SHIM = PROJECT_DIR / "web" / "static_api.js"
# Same Plotly.js version as the plotly Python package bundles (checked to return HTTP 200).
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"

# Parts of the page that need the Python server are hidden online.
STATIC_CSS = """
<style>
  /* Online (GitHub Pages) version: retraining and saving to outputs/ need the Python server. */
  .side-card details, #save-seg { display: none !important; }
  /* Phones: keep the top bar inside the screen width. */
  @media (max-width: 600px) {
    .topbar { gap: 6px; }
    .topbar h1 { font-size: 1.1rem; white-space: normal; min-width: 0; }
    .topbar .icon-btn { width: 38px; height: 38px; flex: none; }
    .topbar .view-switch button { padding: 6px 10px; }
    #print-btn { display: none; }
    .grid > * { min-width: 0; }
  }
</style>"""


def to_json(obj) -> dict | list:
    """Round-trip through Plotly's encoder, so figures look exactly like server.py's answers."""
    return json.loads(json.dumps(obj, cls=PlotlyJSONEncoder))


def model_parameters(bundle: dict) -> dict:
    """Everything needed to repeat the scikit-learn pipeline in JavaScript."""
    pipe = bundle["pipeline"]
    pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    num, cat = pre.named_transformers_["num"], pre.named_transformers_["cat"]
    num_cols = [c for name, _, cols in pre.transformers_ if name == "num" for c in cols]
    cat_cols = [c for name, _, cols in pre.transformers_ if name == "cat" for c in cols]
    return {
        "num_features": num_cols,
        "num_median": num.named_steps["impute"].statistics_.tolist(),
        "num_mean": num.named_steps["scale"].mean_.tolist(),
        "num_scale": num.named_steps["scale"].scale_.tolist(),
        "cat_features": cat_cols,
        "cat_mode": cat.named_steps["impute"].statistics_.astype(float).tolist(),
        "categories": [c.astype(float).tolist() for c in cat.named_steps["onehot"].categories_],
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "sources": list(bundle["transformed_sources"]),
        "train_mean": np.asarray(bundle["train_transformed_mean"], dtype=float).tolist(),
    }


def customer_table(state) -> dict:
    """Columnar customer table (one array per column keeps the file small)."""
    pred = state.base_pred
    sources = [i["file_name"] for i in state.info["files"]]
    table = {c: pred[c].astype(int).tolist() for c in [ID_COL] + FEATURE_COLS}
    table.update({
        "source": pred[SOURCE_COL].map({s: i for i, s in enumerate(sources)}).astype(int).tolist(),
        "split": (pred["Data_Split"] == "Test").astype(int).tolist(),
        "actual": pred["Actual_Default"].astype(int).tolist(),
        "prob": pred["Predicted_Default_Probability"].astype(float).tolist(),
        "age_group": pred["Age_Group"].cat.codes.astype(int).tolist(),
        "limit_range": pred["Credit_Limit_Range"].cat.codes.astype(int).tolist(),
    })
    if min(table["age_group"]) < 0 or min(table["limit_range"]) < 0:
        raise ValueError("Some customers fall outside the age or credit-limit groups.")
    return table


def fixed_responses(state) -> dict:
    """Answers that do not depend on thresholds, filters or user input."""
    s = server
    out = {
        "/api/status": s.api_status({}),
        "/api/exploration": s.api_exploration({}),
        "/api/performance": s.api_performance({}),
        "/api/quality": s.api_quality({}),
        "/api/form-defaults": s.api_form_defaults({}),
    }
    for stat in ("Mean", "Median"):
        out[f"/api/exploration/bill-payment?{urlencode({'stat': stat})}"] = s.api_bill_payment({"stat": [stat]})
    for v in out["/api/exploration"]["variables"]:
        out[f"/api/exploration/distribution?{urlencode({'var': v['value']})}"] = s.api_distribution({"var": [v["value"]]})
    return out


def build_data() -> dict:
    server.STATE = state = server.AppState()
    if state.error:
        raise SystemExit(f"Cannot build the static site: {state.error}")
    bundle = state.bundle
    defaults = {"medium": [str(server.DEFAULT_MEDIUM_THRESHOLD * 100)], "high": [str(server.DEFAULT_HIGH_THRESHOLD * 100)]}
    overview = server.api_overview(defaults)
    first = state.base_pred.iloc[[0]]
    pred_cols = [c for c in state.base_pred.columns if c not in ("Age_Group", "Credit_Limit_Range")]
    metrics_json = (METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists()
                    else json.dumps(bundle["metrics"], cls=PlotlyJSONEncoder, indent=2))
    return {
        "responses": fixed_responses(state),
        "templates": {
            "overview": overview,
            "segment_bar": overview["charts"]["segments"],
            "probability": overview["charts"]["probability"],
            "contribution": charts.contribution_bar(explain_prediction(bundle, first[FEATURE_COLS])),
            "segmentation_filters": server.api_segmentation(defaults)["filters"],
        },
        "customers": customer_table(state),
        "model": model_parameters(bundle),
        "defaults": {"medium": server.DEFAULT_MEDIUM_THRESHOLD * 100, "high": server.DEFAULT_HIGH_THRESHOLD * 100},
        "decision_threshold": bundle["decision_threshold"],
        "base_metrics": {k: bundle["metrics"][k] for k in ("accuracy", "precision", "recall", "f1")},
        "overall": float(state.overall),
        "sources": [i["file_name"] for i in state.info["files"]],
        "age_labels": AGE_LABELS, "limit_labels": LIMIT_LABELS,
        "feature_cols": FEATURE_COLS, "categorical_features": CATEGORICAL_FEATURES,
        "pay_status_cols": PAY_STATUS_COLS, "bill_cols": BILL_COLS, "pay_amt_cols": PAY_AMT_COLS,
        "export_columns": pred_cols,
        "input_rules": INPUT_RULES,
        "allowed_codes": {k: sorted(v) for k, v in ALLOWED_CODES.items()},
        "segment_colors": charts.SEGMENT_COLORS,
        "colors": {"up": charts.UP_COLOR, "down": charts.DOWN_COLOR},
        "labels": {"friendly": FRIENDLY_NAMES, "sex": SEX_LABELS, "education": EDUCATION_LABELS,
                   "marriage": MARRIAGE_LABELS, "target": TARGET_LABELS, "months": MONTHS,
                   "pay_status_month": PAY_STATUS_MONTH},
        "downloads": {"metrics_json": metrics_json,
                      "quality_csv": export_quality_report(state.quality).read_text(encoding="utf-8")},
        "segments": SEGMENTS,
    }


def build_html(data: dict) -> str:
    html = PAGE.read_text(encoding="utf-8")
    payload = json.dumps(to_json(data), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    shim = SHIM.read_text(encoding="utf-8")
    plotly_tag = '<script src="/static/plotly.min.js"></script>'
    if plotly_tag not in html or "<script>" not in html:
        raise SystemExit("web/index.html changed: cannot find the Plotly script tag or the main <script>.")
    html = html.replace(plotly_tag, f'<script src="{PLOTLY_CDN}"></script>')
    html = html.replace("</head>", STATIC_CSS + "\n</head>", 1)
    injected = (f'<script id="static-data" type="application/json">{payload}</script>\n'
                f"<script>\n{shim}\n</script>\n<script>")
    html = html.replace("<script>", injected, 1)  # before the page's own script
    return html


def main() -> None:
    print("Loading data and model…")
    html = build_html(build_data())
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_DIR)} ({len(html.encode('utf-8')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
