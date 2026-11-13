"""
build_static_site.py - build the GitHub Pages version of the dashboard
======================================================================
Run with:   python build_static_site.py        (writes the site/ folder)
Preview:    python -m http.server 8001 -d site  then open http://127.0.0.1:8001

GitHub Pages can only host static files, so it cannot run server.py. This script:

1. loads the data and the saved model exactly like server.py does;
2. computes every answer that never changes (exploration charts, model
   performance, data quality, form defaults) by calling server.py's own
   endpoint functions;
3. writes the customer table, the model coefficients and those answers as JSON
   files in site/data/, split so each page fetches only what it needs
   (the first page needs only data/core.json, about 0.4 MB compressed);
4. writes site/index.html: the normal web/index.html page plus
   web/static_api.js, which answers the page's ``/api/...`` requests in the
   browser (thresholds, filters, customer lookup, new-customer estimates,
   warning-level slider, CSV downloads), and a small Plotly.js build
   (web/vendor, only the chart types this dashboard uses).

.github/workflows/pages.yml deploys site/. Re-run this (and push) whenever the
data or the model changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
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

SITE = PROJECT_DIR / "site"
PAGE = PROJECT_DIR / "web" / "index.html"
SHIM = PROJECT_DIR / "web" / "static_api.js"
# Plotly.js "cartesian" build (bar, box, heatmap, histogram, scatter): 1.5 MB instead of 4.6 MB.
# Same version as the plotly Python package bundles; downloaded from
# https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@4.1.1/plotly-cartesian.min.js
PLOTLY_JS = PROJECT_DIR / "web" / "vendor" / "plotly-cartesian-4.1.1.min.js"

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


def customer_core(state) -> dict:
    """The customer columns every page needs (columnar: one array per column keeps the file small)."""
    pred = state.base_pred
    sources = [i["file_name"] for i in state.info["files"]]
    table = {
        ID_COL: pred[ID_COL].astype(int).tolist(),
        # PAY_0, LIMIT_BAL, AGE, BILL_AMT1, PAY_AMT1 are shown in the Risk groups table.
        **{c: pred[c].astype(int).tolist() for c in ("LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1")},
        "source": pred[SOURCE_COL].map({s: i for i, s in enumerate(sources)}).astype(int).tolist(),
        "split": (pred["Data_Split"] == "Test").astype(int).tolist(),
        "actual": pred["Actual_Default"].astype(int).tolist(),
        # Full-precision probabilities as little-endian float64 in base64: half the size of decimal text.
        "prob": {"f8": base64.b64encode(pred["Predicted_Default_Probability"].to_numpy("<f8").tobytes()).decode("ascii")},
        "age_group": pred["Age_Group"].cat.codes.astype(int).tolist(),
        "limit_range": pred["Credit_Limit_Range"].cat.codes.astype(int).tolist(),
    }
    if min(table["age_group"]) < 0 or min(table["limit_range"]) < 0:
        raise ValueError("Some customers fall outside the age or credit-limit groups.")
    return table


def without_x(fig: dict) -> dict:
    """The probability histogram template: its 30,120 values are filled in by the browser from the customer table."""
    fig = json.loads(json.dumps(fig))
    fig["data"][0]["x"] = []
    return fig


def build_chunks() -> dict[str, dict]:
    """name -> JSON object, written to site/data/<name>.json."""
    s = server
    s.STATE = state = s.AppState()
    if state.error:
        raise SystemExit(f"Cannot build the static site: {state.error}")
    bundle = state.bundle
    defaults = {"medium": [str(s.DEFAULT_MEDIUM_THRESHOLD * 100)], "high": [str(s.DEFAULT_HIGH_THRESHOLD * 100)]}
    overview = to_json(s.api_overview(defaults))
    probability = without_x(overview["charts"]["probability"])
    overview["charts"]["probability"] = None  # rebuilt in the browser for the chosen thresholds
    first = state.base_pred.iloc[[0]]
    exploration = s.api_exploration({})
    variables = [v["value"] for v in exploration["variables"]]
    metrics_json = (METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists()
                    else json.dumps(bundle["metrics"], cls=PlotlyJSONEncoder, indent=2))

    core = {
        "responses": {"/api/status": s.api_status({}), "/api/quality": s.api_quality({}),
                      "/api/form-defaults": s.api_form_defaults({})},
        "templates": {
            "overview": overview,
            "segment_bar": overview["charts"]["segments"],
            "probability": probability,
            "contribution": charts.contribution_bar(explain_prediction(bundle, first[FEATURE_COLS])),
            "segmentation_filters": s.api_segmentation(defaults)["filters"],
        },
        "customers": customer_core(state),
        "model": model_parameters(bundle),
        "defaults": {"medium": s.DEFAULT_MEDIUM_THRESHOLD * 100, "high": s.DEFAULT_HIGH_THRESHOLD * 100},
        "decision_threshold": bundle["decision_threshold"],
        "base_metrics": {k: bundle["metrics"][k] for k in ("accuracy", "precision", "recall", "f1")},
        "overall": float(state.overall),
        "sources": [i["file_name"] for i in state.info["files"]],
        "age_labels": AGE_LABELS, "limit_labels": LIMIT_LABELS,
        "feature_cols": FEATURE_COLS, "categorical_features": CATEGORICAL_FEATURES,
        "pay_status_cols": PAY_STATUS_COLS, "bill_cols": BILL_COLS, "pay_amt_cols": PAY_AMT_COLS,
        "export_columns": [c for c in state.base_pred.columns if c not in ("Age_Group", "Credit_Limit_Range")],
        "distribution_vars": variables,
        "input_rules": INPUT_RULES,
        "allowed_codes": {k: sorted(v) for k, v in ALLOWED_CODES.items()},
        "segment_colors": charts.SEGMENT_COLORS,
        "colors": {"up": charts.UP_COLOR, "down": charts.DOWN_COLOR},
        "labels": {"friendly": FRIENDLY_NAMES, "sex": SEX_LABELS, "education": EDUCATION_LABELS,
                   "marriage": MARRIAGE_LABELS, "target": TARGET_LABELS, "months": MONTHS,
                   "pay_status_month": PAY_STATUS_MONTH},
        "segments": SEGMENTS,
    }
    chunks = {
        "core": core,
        # All 23 inputs per customer: needed for customer lookup and the CSV downloads.
        "features": {c: state.base_pred[c].astype(int).tolist() for c in FEATURE_COLS},
        "exploration": {
            "/api/exploration": exploration,
            **{f"/api/exploration/bill-payment?{urlencode({'stat': st})}": s.api_bill_payment({"stat": [st]})
               for st in ("Mean", "Median")},
        },
        "performance": {"/api/performance": s.api_performance({})},
        "downloads": {"metrics_json": metrics_json,
                      "quality_csv": export_quality_report(state.quality).read_text(encoding="utf-8")},
    }
    for v in variables:
        chunks[f"dist-{v}"] = {f"/api/exploration/distribution?{urlencode({'var': v})}": s.api_distribution({"var": [v]})}
    return {name: to_json(obj) for name, obj in chunks.items()}


def build_html(version: str) -> str:
    html = PAGE.read_text(encoding="utf-8")
    shim = SHIM.read_text(encoding="utf-8")
    plotly_tag = '<script src="/static/plotly.min.js"></script>'
    if plotly_tag not in html or "<script>" not in html:
        raise SystemExit("web/index.html changed: cannot find the Plotly script tag or the main <script>.")
    # Start downloading Plotly and the core data at once, while the page layout already shows.
    preload = (f'<link rel="preload" href="{PLOTLY_JS.name}" as="script">\n'
               f'<link rel="preload" href="data/core.json?v={version}" as="fetch" crossorigin="anonymous">')
    html = html.replace(plotly_tag, preload)
    html = html.replace("</head>", STATIC_CSS + "\n</head>", 1)
    injected = (f'<script src="{PLOTLY_JS.name}"></script>\n'
                f'<script>window.STATIC_DATA_VERSION = "{version}";</script>\n'
                f"<script>\n{shim}\n</script>\n<script>")
    html = html.replace("<script>", injected, 1)  # before the page's own script
    return html.replace("Connecting to the server…", "Loading the data…")


def main() -> None:
    print("Loading data and model…")
    chunks = {name: json.dumps(obj, separators=(",", ":"), ensure_ascii=False) for name, obj in build_chunks().items()}
    version = hashlib.sha256("".join(chunks[k] for k in sorted(chunks)).encode("utf-8")).hexdigest()[:12]
    if (SITE / "data").exists():
        shutil.rmtree(SITE / "data")  # remove data files from earlier builds
    (SITE / "data").mkdir(parents=True)
    for name, text in chunks.items():
        (SITE / "data" / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
    (SITE / "index.html").write_text(build_html(version), encoding="utf-8", newline="\n")
    shutil.copyfile(PLOTLY_JS, SITE / PLOTLY_JS.name)
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"Wrote site/ ({len(chunks)} data files, {total / 1e6:.1f} MB in total, "
          f"first page loads data/core.json: {(SITE / 'data' / 'core.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
