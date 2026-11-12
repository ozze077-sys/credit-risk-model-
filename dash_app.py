"""
dash_app.py - Credit Risk Analytics & Default Prediction Dashboard (Plotly Dash version)
=======================================================================================
Start with:   python dash_app.py        then open http://127.0.0.1:8050

How a Dash app works (beginner summary)
---------------------------------------
* ``app.layout`` describes the page as a tree of components (html.Div, dcc.Graph, ...).
* ``@callback`` functions connect components: when an *Input* changes (a slider,
  a dropdown, the URL), Dash calls the function and puts the return value into
  the *Output* component. *State* values are read but don't trigger the call.
* The URL path (/, /exploration, ...) decides which page is shown.

The data, model and prediction logic are the same modules used everywhere else
(data_loader.py, model_training.py, prediction.py, data_analysis.py); charts are
built in charts.py. This file only handles layout and interaction.

The dataset and model are loaded ONCE when the app starts. The model is trained
only if no valid saved model exists (or the dataset changed); otherwise use the
"Retrain model" button.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Group, Scheme

import charts
from data_analysis import (AGE_LABELS, LIMIT_LABELS, add_groups, age_group, bill_payment_comparison,
                           correlation_matrix, credit_limit_stats, default_rate_by, limit_range)
from data_loader import (BILL_COLS, EDUCATION_LABELS, FEATURE_COLS, ID_COL, MARRIAGE_LABELS, MONTHS,
                         PAY_AMT_COLS, PAY_STATUS_COLS, PAY_STATUS_MONTH, SEX_LABELS, TARGET_COL,
                         TARGET_LABELS, DatasetError, category_label, data_quality_report,
                         export_quality_report, load_dataset, pay_status_label)
from model_training import METRICS_PATH, MODEL_PATH, OUTPUTS_DIR, evaluate, save_bundle, train_model
from prediction import (DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD, SEGMENTS, ModelError,
                        assign_segments, build_prediction_table, explain_prediction, export_predictions,
                        load_model, model_matches_data, predict_probability, segment_summary,
                        validate_customer_input)

PAGES = {
    "/": "Overview",
    "/exploration": "Data Exploration",
    "/segmentation": "Risk Segmentation",
    "/individual": "Individual Risk Assessment",
    "/performance": "Model Performance",
    "/quality": "Data Quality",
}
DEFAULT_THRESHOLDS = [int(DEFAULT_MEDIUM_THRESHOLD * 100), int(DEFAULT_HIGH_THRESHOLD * 100)]


# ---------------------------------------------------------------------------
# Load data and model once at start-up
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


LOAD_ERROR = None
try:
    DF, INFO = load_dataset()
    QUALITY = data_quality_report(DF, INFO)
    DATA = add_groups(DF)
    OVERALL = DF[TARGET_COL].mean()
    STATE = {"bundle": load_or_train(DF, INFO), "version": 0}
except (DatasetError, ModelError) as exc:
    LOAD_ERROR = str(exc)
except Exception as exc:  # e.g. model training failed
    LOAD_ERROR = f"The model could not be trained: {exc}"


@lru_cache(maxsize=1)
def base_predictions(version: int) -> pd.DataFrame:
    """Probabilities for every customer (computed once per model version)."""
    pred = build_prediction_table(DF, STATE["bundle"])
    pred["Age_Group"] = age_group(pred["AGE"])
    pred["Credit_Limit_Range"] = limit_range(pred["LIMIT_BAL"])
    return pred


def predictions(thresholds) -> pd.DataFrame:
    """Predictions with risk segments for the current sidebar thresholds (cheap: no re-scoring)."""
    medium, high = thresholds_to_float(thresholds)
    pred = base_predictions(STATE["version"]).copy()
    pred["Risk_Segment"] = assign_segments(pred["Predicted_Default_Probability"], medium, high)
    return pred


def thresholds_to_float(thresholds) -> tuple[float, float]:
    t = thresholds or DEFAULT_THRESHOLDS
    return t[0] / 100, t[1] / 100


# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------
def kpi(label: str, value: str, note: str = "", predicted: bool = False, tag: bool = True) -> html.Div:
    badge = (html.Span("MODEL" if predicted else "HISTORICAL", className=f"tag {'pred' if predicted else 'hist'}")
             if tag else None)
    return html.Div(className=f"kpi {'pred' if predicted else ''}", children=[
        html.Div([label, badge], className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(note, className="kpi-note"),
    ])


def kpi_row(*cards) -> html.Div:
    return html.Div(list(cards), className="kpi-row")


def note(*children) -> html.Div:
    return html.Div(list(children), className="note")


def graph(fig, **kw) -> dcc.Graph:
    return dcc.Graph(figure=fig, config={"displaylogo": False}, className="card", **kw)


def grid(*children, cols: int = 2) -> html.Div:
    return html.Div(list(children), className=f"grid grid-{cols}")


def caption(text) -> html.Div:
    return html.Div(text, className="caption")


def money(v) -> str:
    return f"NT$ {v:,.0f}"


PCT = Format(precision=1, scheme=Scheme.percentage)
INT = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
DEC3 = Format(precision=3, scheme=Scheme.fixed)


def table(df: pd.DataFrame, formats: dict | None = None, page_size: int = 15, **kw) -> dash_table.DataTable:
    """A styled, read-only table. ``formats`` maps column -> dash_table Format."""
    formats = formats or {}
    kw.setdefault("sort_action", "native")
    columns = [{"name": c, "id": c, **({"type": "numeric", "format": formats[c]} if c in formats else {})}
               for c in df.columns]
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=columns, page_size=page_size,
        style_table={"overflowX": "auto"}, style_as_list_view=True,
        style_header={"backgroundColor": "#F4F6FA", "fontWeight": 600, "color": charts.MUTED,
                      "border": "none", "borderBottom": "1px solid #DDE3EC"},
        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 13, "padding": "6px 10px",
                    "color": charts.INK, "border": "none", "borderBottom": "1px solid #EEF1F5",
                    "textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": 520},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"} for c in formats],
        **kw)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(thresholds) -> list:
    medium, high = thresholds_to_float(thresholds)
    pred = predictions(thresholds)
    n_high = int((pred["Risk_Segment"] == "High risk").sum())
    bundle = STATE["bundle"]
    return [
        html.H2("Portfolio overview"),
        note(html.B("Historical"), " figures (blue) describe what actually happened in the data. ",
             html.B("Model"), " figures (violet) are estimates from the logistic-regression model. "
             "This dataset is a single snapshot (Apr–Sep 2005), so no time trend is shown."),
        kpi_row(
            kpi("Total customers", f"{len(DF):,}", "records with a valid outcome"),
            kpi("Actual default rate", f"{OVERALL:.1%}", "defaulted the following month"),
            kpi("Actual defaults", f"{int(DF[TARGET_COL].sum()):,}", f"of {len(DF):,} customers"),
            kpi("Avg credit limit", f"{DF['LIMIT_BAL'].mean() / 1000:,.1f}K",
                f"NT$ · median {money(DF['LIMIT_BAL'].median())}"),
            kpi("Predicted high risk", f"{n_high:,}", f"probability ≥ {high:.0%} ({n_high / len(pred):.1%})", True),
            kpi("Avg predicted probability", f"{pred['Predicted_Default_Probability'].mean():.1%}",
                "mean of model estimates", True),
        ),
        grid(graph(charts.outcome_bar(DF)),
             graph(charts.rate_bar(default_rate_by(DF, "PAY_0"),
                                   "Historical default rate by Sep 2005 repayment status (PAY_0)",
                                   "Repayment status", OVERALL))),
        grid(graph(charts.segment_bar(segment_summary(pred), medium, high)),
             graph(charts.probability_histogram(pred["Predicted_Default_Probability"], medium, high))),
        graph(charts.importance_bar(bundle["permutation_importance"].head(10),
                                    "Most important model inputs (permutation importance on the test set)")),
        caption("Importance shows how much the model relies on an input to rank customers, measured on unseen "
                "test data. It does not mean the input causes default."),
    ]


def page_exploration(_thresholds) -> list:
    lim = credit_limit_stats(DF)
    n_def = int(DF[TARGET_COL].sum())
    var_options = [{"label": "Credit limit (LIMIT_BAL)", "value": "LIMIT_BAL"}, {"label": "Age (AGE)", "value": "AGE"}]
    var_options += [{"label": f"Bill amount {MONTHS[i]} ({c})", "value": c} for i, c in enumerate(BILL_COLS, 1)]
    var_options += [{"label": f"Payment amount {MONTHS[i]} ({c})", "value": c} for i, c in enumerate(PAY_AMT_COLS, 1)]
    codes_box = html.Div(className="card pad", children=[
        html.B("What the repayment-status codes mean"),
        html.Ul([html.Li([html.B("-2"), ": no balance / no credit used that month"]),
                 html.Li([html.B("-1"), ": paid in full (\"pay duly\")"]),
                 html.Li([html.B("0"), ": revolving credit — minimum paid, balance carried over"]),
                 html.Li([html.B("1 – 8"), ": payment delayed by that many months"])]),
        html.P(["Codes -1 and 1–9 come from the UCI documentation. Codes -2 and 0 are not in the original "
                "documentation; the meanings above are the widely used interpretation. They are ",
                html.B("categories"), ", not negative amounts, so the model one-hot encodes them."]),
        caption("EDUCATION codes 0, 5, 6 and MARRIAGE code 0 are undocumented; they are kept as separate groups "
                "rather than deleted."),
    ])
    return [
        html.H2("Data exploration (historical outcomes)"),
        note("All rates on this page are ", html.B("historical"), ": the share of customers in a group who actually "
             "defaulted. They describe associations in this dataset, not causes, and they are not individual "
             "predictions. Hover over bars to see group sizes; small groups give unreliable rates."),
        kpi_row(
            kpi("No default", f"{len(DF) - n_def:,}"),
            kpi("Default", f"{n_def:,}"),
            kpi("Default rate", f"{OVERALL:.1%}"),
            kpi("Avg credit limit", f"{lim['mean'] / 1000:,.1f}K", f"NT$ · median {lim['median'] / 1000:,.0f}K"),
            kpi("Min credit limit", f"{lim['min'] / 1000:,.0f}K", "NT$"),
            kpi("Max credit limit", f"{lim['max'] / 1000:,.0f}K", "NT$"),
        ),
        dcc.Tabs(className="tabs", children=[
            dcc.Tab(label="Default rates by group", children=[
                grid(graph(charts.rate_bar(default_rate_by(DATA, "Age_Group"), "Default rate by age group",
                                           "Age group", OVERALL)),
                     graph(charts.rate_bar(default_rate_by(DATA, "Credit_Limit_Range"),
                                           "Default rate by credit-limit range", "Credit limit (NT$)", OVERALL))),
                grid(graph(charts.rate_bar(default_rate_by(DF, "PAY_0"),
                                           "Default rate by Sep 2005 repayment status (PAY_0)",
                                           "Repayment status", OVERALL)),
                     graph(charts.rate_bar(default_rate_by(DF, "EDUCATION"), "Default rate by education",
                                           "Education", OVERALL))),
                grid(graph(charts.rate_bar(default_rate_by(DF, "MARRIAGE"), "Default rate by marital status",
                                           "Marital status", OVERALL)), codes_box),
            ]),
            dcc.Tab(label="Defaulters vs non-defaulters", children=[
                grid(graph(charts.limit_by_outcome(DF)), graph(charts.limit_box(DATA))),
                html.Div(className="control-row", children=[
                    html.Label("Statistic"),
                    dcc.RadioItems(id="stat-radio", options=["Mean", "Median"], value="Mean", inline=True,
                                   className="radio"),
                    caption("Amounts are very skewed (a few huge values), so the median is often more typical."),
                ]),
                grid(dcc.Graph(id="bill-chart", className="card", config={"displaylogo": False}),
                     dcc.Graph(id="pay-chart", className="card", config={"displaylogo": False})),
            ]),
            dcc.Tab(label="Correlation", children=[
                graph(charts.correlation_heatmap(correlation_matrix(DF))),
                caption("Correlation measures linear association only (−1 to +1). It does not imply causation. "
                        "PAY_x codes are included as numbers here only for a quick overview."),
            ]),
            dcc.Tab(label="Distributions", children=[
                html.Div(className="control-row", children=[
                    html.Label("Variable"),
                    dcc.Dropdown(id="dist-var", options=var_options, value="LIMIT_BAL", clearable=False,
                                 className="dropdown-wide"),
                ]),
                dcc.Graph(id="dist-chart", className="card", config={"displaylogo": False}),
                html.Div(id="dist-caption", className="caption"),
            ]),
        ]),
    ]


def page_segmentation(thresholds) -> list:
    medium, high = thresholds_to_float(thresholds)
    pred = predictions(thresholds)
    seg = segment_summary(pred)
    cards = [kpi(r["Risk_Segment"], f"{r['Customers']:,} ({r['Share']:.1%})",
                 "avg predicted probability " + ("–" if pd.isna(r["Avg_Probability"]) else f"{r['Avg_Probability']:.1%}"),
                 True) for _, r in seg.iterrows()]
    summary = seg.rename(columns={"Risk_Segment": "Segment", "Avg_Probability": "Avg predicted probability",
                                  "Actual_Default_Rate": "Actual default rate (historical)"})[
        ["Segment", "Customers", "Share", "Avg predicted probability", "Actual default rate (historical)"]]

    def multi(id_, label, options):
        return html.Div([html.Label(label), dcc.Dropdown(id=id_, options=options, value=options, multi=True)],
                        className="filter")

    return [
        html.H2("Risk segmentation (model predictions)"),
        note("Customers are grouped by their ", html.B("predicted"), f" probability of default: Low < {medium:.0%} ≤ "
             f"Medium < {high:.0%} ≤ High. These are ", html.B("configurable demonstration thresholds"),
             " (change them in the sidebar), not established banking standards or validated lending decisions. "
             "Predictions for training-set customers are slightly optimistic because the model learned from them."),
        html.Div(cards, className="kpi-row three"),
        grid(graph(charts.probability_histogram(pred["Predicted_Default_Probability"], medium, high,
                                                "Predicted probability distribution with segment thresholds")),
             html.Div(className="card pad", children=[
                 html.B("Segment summary"),
                 table(summary, {"Customers": INT, "Share": PCT, "Avg predicted probability": PCT,
                                 "Actual default rate (historical)": PCT}, sort_action="none"),
                 caption("Comparing the average predicted probability with the actual historical default rate in "
                         "each segment is a quick sanity check of the model's probabilities. (Actual rates include "
                         "training customers, so they look slightly better than on new data.)"),
             ])),
        html.H3("Filter customers"),
        html.Div(className="filters", children=[
            multi("f-seg", "Risk segment", SEGMENTS),
            multi("f-age", "Age group", AGE_LABELS),
            multi("f-lim", "Credit-limit range", LIMIT_LABELS),
            multi("f-split", "Data split (Test = never seen in training)", ["Train", "Test"]),
        ]),
        html.Div(className="control-row", children=[
            html.Label("Show the highest-risk customers"),
            html.Div(dcc.Slider(id="f-top", min=10, max=200, step=5, value=25,
                                marks={10: "10", 50: "50", 100: "100", 150: "150", 200: "200"},
                                tooltip={"placement": "bottom"}), className="slider-wide"),
        ]),
        html.Div(id="seg-count", className="caption"),
        html.Div(id="seg-table"),
        html.Div(className="button-row", children=[
            html.Button("⬇ Download filtered predictions (CSV)", id="btn-dl-seg", className="btn"),
            html.Button("💾 Save all predictions to outputs/ folder", id="btn-save-seg", className="btn"),
        ]),
        html.Div(id="save-msg", className="caption"),
    ]


def customer_details(row: pd.Series) -> pd.DataFrame:
    items = [("Credit limit", money(row["LIMIT_BAL"])), ("Age", f"{row['AGE']:.0f}"),
             ("Sex", category_label("SEX", row["SEX"])), ("Education", category_label("EDUCATION", row["EDUCATION"])),
             ("Marital status", category_label("MARRIAGE", row["MARRIAGE"]))]
    items += [(f"Repayment status {PAY_STATUS_MONTH[c]}", pay_status_label(row[c])) for c in PAY_STATUS_COLS]
    items += [(f"Bill / paid {MONTHS[i]}", f"{money(row[b])} / {money(row[p])}")
              for i, (b, p) in enumerate(zip(BILL_COLS, PAY_AMT_COLS), 1)]
    return pd.DataFrame(items, columns=["Field", "Value"])


def page_individual(_thresholds) -> list:
    med = DF[FEATURE_COLS].median()
    codes = [{"label": pay_status_label(c), "value": c} for c in range(-2, 9)]

    def field(label, component):
        return html.Div([html.Label(label), component], className="field")

    def number(col, lo, hi, step):
        return dcc.Input(id=f"in-{col}", type="number", min=lo, max=hi, step=step, value=int(med[col]),
                         className="input")

    def select(col, labels, value):
        return dcc.Dropdown(id=f"in-{col}", options=[{"label": v, "value": k} for k, v in labels.items()],
                            value=value, clearable=False)

    month_rows = [html.Div(className="form-row month", children=[
        field(f"Status {MONTHS[i]} ({s})", dcc.Dropdown(id=f"in-{s}", options=codes, value=0, clearable=False)),
        field(f"Bill {MONTHS[i]} (NT$)", number(b, -10_000_000, 10_000_000, 1000)),
        field(f"Paid {MONTHS[i]} (NT$)", number(p, 0, 10_000_000, 500)),
    ]) for i, (s, b, p) in enumerate(zip(PAY_STATUS_COLS, BILL_COLS, PAY_AMT_COLS), 1)]

    return [
        html.H2("Individual risk assessment"),
        note("The probability comes from the model's ", html.Code("predict_proba()"), ", which combines ",
             html.B("all"), " inputs for this one customer. It is an ", html.B("estimate, not a guarantee"),
             ", and must not be used on its own to make a real credit decision."),
        dcc.RadioItems(id="ind-mode", options=[{"label": "Existing customer (by ID)", "value": "existing"},
                                               {"label": "Enter new customer details", "value": "new"}],
                       value="existing", inline=True, className="radio"),
        html.Div(id="existing-box", children=[
            field("Customer ID", dcc.Input(id="cust-id", type="number", min=int(DF[ID_COL].min()),
                                           max=int(DF[ID_COL].max()), step=1, value=int(DF[ID_COL].min()),
                                           debounce=True, className="input")),
            html.Div(id="existing-result"),
        ]),
        html.Div(id="new-box", style={"display": "none"}, children=[
            html.Div(className="card pad", children=[
                html.B("Customer profile"),
                html.Div(className="form-row", children=[
                    field("Credit limit (NT$)", number("LIMIT_BAL", 1, 10_000_000, 10_000)),
                    field("Age", number("AGE", 18, 100, 1)),
                    field("Sex", select("SEX", SEX_LABELS, 2)),
                    field("Education", select("EDUCATION", EDUCATION_LABELS, 2)),
                    field("Marital status", select("MARRIAGE", MARRIAGE_LABELS, 2)),
                ]),
                html.B("Repayment status, bill and payment by month (most recent first)"),
                *month_rows,
                html.Button("Estimate default probability", id="btn-estimate", className="btn primary"),
                caption("Fields are pre-filled with median values from the dataset."),
            ]),
            html.Div(id="new-result"),
        ]),
    ]


def probability_result(proba: float, thresholds, actual=None, split=None) -> html.Div:
    medium, high = thresholds_to_float(thresholds)
    segment = str(assign_segments([proba], medium, high)[0])
    parts = [
        html.B("Estimated probability of default next month"),
        html.Div(f"{proba:.1%}", className="big-number"),
        html.Div(html.Div(style={"width": f"{proba * 100:.1f}%"}, className="bar-fill"), className="bar"),
        html.Div(className="seg", children=[
            html.Span(className="dot", style={"background": charts.SEGMENT_COLORS[segment]}), segment,
            html.Span(f" (thresholds {medium:.0%} / {high:.0%})", className="muted")]),
        caption(f"Historical average default rate for comparison: {OVERALL:.1%}"),
    ]
    if actual is not None:
        split_note = ("This customer was in the training data, so the model has already seen this outcome; the "
                      "estimate may look better than it would for a brand-new customer." if split == "Train" else
                      "This customer was in the held-out test set: the model never saw this outcome.")
        parts.append(html.Div(className="card pad", children=[
            html.Div([html.B("Recorded historical outcome: "), TARGET_LABELS[int(actual)]]),
            caption("Shown for comparison only. It is never used as a model input."),
            caption(split_note)]))
    return html.Div(parts)


def explanation(row: pd.DataFrame, proba: float) -> html.Div:
    exp = explain_prediction(STATE["bundle"], row)
    up = exp[exp["Contribution"] > 0]["Feature"].tolist()[:3]
    down = exp[exp["Contribution"] < 0]["Feature"].tolist()[:3]
    text = f"The estimate is **{'higher' if proba > OVERALL else 'lower'}** than the historical average ({OVERALL:.1%})."
    if up:
        text += f" Inputs pushing it **up**: {', '.join(up)}."
    if down:
        text += f" Inputs pulling it **down**: {', '.join(down)}."
    return html.Div([
        dcc.Markdown(text),
        graph(charts.contribution_bar(exp)),
        caption("Contributions come from the model's coefficients: they describe how the model reacts to these "
                "inputs, not what causes default in real life."),
    ])


METRIC_HELP = {
    "Accuracy": "Share of all test customers classified correctly. Misleading on its own: predicting 'no default' "
                "for everyone would already score {base:.1%}.",
    "Precision": "Of the customers the model flags as defaulters, the share that really defaulted. Low precision = "
                 "many false alarms.",
    "Recall": "Of the customers who really defaulted, the share the model caught. Low recall = missed defaulters, "
              "which is usually the costliest error in credit risk.",
    "F1 score": "A single score balancing precision and recall (their harmonic mean).",
    "ROC-AUC": "Probability that a random defaulter gets a higher score than a random non-defaulter. 0.5 = random "
               "guessing, 1.0 = perfect ranking. Does not depend on the 50% cut-off.",
}


def page_performance(_thresholds) -> list:
    bundle = STATE["bundle"]
    m = bundle["metrics"]
    values = {"Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"], "F1 score": m["f1"],
              "ROC-AUC": m["roc_auc"]}
    rep = pd.DataFrame(m["classification_report"]).T.loc[["No default", "Default", "macro avg", "weighted avg"]]
    rep = rep.reset_index()
    rep.columns = ["Class", "Precision", "Recall", "F1 score", "Support"]
    gap = m["test_mean_predicted_probability"] - m["test_actual_default_rate"]
    if abs(gap) > 0.05:
        cal_msg = html.Div(f"Predicted probabilities are on average {gap:+.1%} away from the actual rate. " + (
            "This is expected with class_weight='balanced', which deliberately inflates the default class; treat "
            "the numbers as risk scores rather than literal probabilities." if bundle["class_weight"] == "balanced"
            else "Interpret the probabilities with care."), className="alert warn")
    else:
        cal_msg = html.Div("On average the probabilities match the actual default rate. Check the curve for groups "
                           "where the model over- or under-estimates risk.", className="alert ok")
    return [
        html.H2("Model performance (held-out test set)"),
        note(f"All numbers are calculated on the ", html.B(f"{bundle['n_test']:,} test customers"),
             f" (20%, stratified, random seed {bundle['random_seed']}) that were never used for training or "
             f"preprocessing. A customer is classified as a defaulter when the predicted probability is ≥ "
             f"{bundle['decision_threshold']:.0%}."),
        html.Div([kpi(k, f"{v:.3f}", "", True) for k, v in values.items()], className="kpi-row five"),
        html.Div(className="card pad", children=[
            html.B("What do these metrics mean?"),
            html.Ul([html.Li([html.B(k), f" ({values[k]:.3f}): ",
                              t.format(base=m["baseline_accuracy_always_no_default"])]) for k, t in METRIC_HELP.items()]),
            html.P([f"Only {m['test_actual_default_rate']:.1%} of test customers defaulted, so the classes are ",
                    html.B("imbalanced"), " — that is why recall, precision and ROC-AUC matter more than accuracy here."]),
        ]),
        grid(graph(charts.confusion_heatmap(m["confusion_matrix"])),
             html.Div(className="card pad", children=[
                 html.B("Classification report (test set)"),
                 table(rep, {"Precision": DEC3, "Recall": DEC3, "F1 score": DEC3, "Support": INT},
                       sort_action="none"),
                 caption("'Support' is the number of test customers in each actual class.")])),
        grid(graph(charts.roc_chart(m)), graph(charts.pr_chart(m))),
        html.H3("Explore the decision threshold"),
        html.Div(className="control-row", children=[
            html.Label("Classify as 'default' when predicted probability ≥"),
            html.Div(dcc.Slider(id="perf-threshold", min=0.05, max=0.95, step=0.05, value=0.5,
                                marks={v / 100: f"{v / 100:.2f}" for v in range(10, 100, 10)},
                                tooltip={"placement": "bottom"}), className="slider-wide"),
        ]),
        html.Div(id="threshold-metrics"),
        html.H3("Probability calibration"),
        grid(graph(charts.calibration_chart(m)), html.Div(className="card pad", children=[
            html.Ul([html.Li(["Average predicted probability: ", html.B(f"{m['test_mean_predicted_probability']:.1%}")]),
                     html.Li(["Actual test default rate: ", html.B(f"{m['test_actual_default_rate']:.1%}")]),
                     html.Li(["Brier score: ", html.B(f"{m['brier_score']:.3f}"),
                              " (mean squared error of the probabilities; lower is better)"])]),
            html.P("If the points lie close to the dashed line, a predicted 40% really does mean roughly 40 in 100 "
                   "such customers defaulted."),
            cal_msg,
            caption("Class imbalance: the standard model keeps probabilities calibrated. Retraining with "
                    "class_weight='balanced' (sidebar) increases recall at the 50% cut-off but makes probabilities "
                    "systematically too high."),
        ])),
        html.H3("What influences the model?"),
        grid(graph(charts.coefficient_bar(bundle["coefficients"])),
             graph(charts.importance_bar(bundle["permutation_importance"].head(15), "Permutation importance (test set)",
                                         height=520))),
        html.Div(className="card pad", children=html.Ul([
            html.Li([html.B("Coefficients"), ": numeric inputs are standardised, so a coefficient is the change in "
                     "log-odds for a one-standard-deviation increase. Categories (e.g. PAY_0 = 2 months late) are "
                     "compared with the model's other categories of the same variable. Odds ratio = e^coefficient."]),
            html.Li([html.B("Limitations"), ": correlated inputs (e.g. the six bill amounts) share and can swap their "
                     "weight, so individual coefficients can be unstable or have surprising signs. Rare categories "
                     "can get large coefficients from very few customers."]),
            html.Li([html.B("Permutation importance"), " shuffles one original column in the test data and measures "
                     "how much the ranking quality (ROC-AUC) drops. It shows reliance, not causation."]),
            html.Li([html.B("Fairness"), ": SEX, AGE, EDUCATION and MARRIAGE are demographic attributes. Using them (or "
                     "proxies for them) in real lending can be unlawful or discriminatory. Any real-world system "
                     "would need legal review, fairness testing across groups and possibly removal of these inputs."]),
        ])),
        html.Div(className="button-row", children=[
            html.Button("⬇ Model metrics (JSON)", id="btn-dl-metrics", className="btn"),
            html.Button("⬇ Actual vs predicted, test customers (CSV)", id="btn-dl-test", className="btn"),
        ]),
    ]


def page_quality(_thresholds) -> list:
    q = QUALITY
    s = q["summary"]
    return [
        html.H2("Data quality report"),
        kpi_row(
            kpi("Rows used", f"{s['Rows used (valid target)']:,}", f"read {s['Rows read']:,}", tag=False),
            kpi("Columns", f"{s['Columns used']}", "incl. ID and target", tag=False),
            kpi("Missing values", f"{s['Total missing values']:,}", tag=False),
            kpi("Duplicate IDs", f"{s['Duplicate customer IDs']:,}", tag=False),
            kpi("Duplicate rows", f"{s['Fully duplicated rows']:,}", f"{s['Rows identical except ID']} identical except ID",
                tag=False),
            kpi("Header row", f"Excel row {s['Real header found on Excel row']}", "detected automatically", tag=False),
        ),
        note("File: ", html.B(s["File"]), " · target column found as ", html.B(f"'{s['Original target column name']}'"),
             " (0 = no default, 1 = default; verified to contain only 0/1). The original file is only read, never "
             "modified. Rows identical except their ID are kept: different customers can share the same values."),
        grid(html.Div(className="card pad", children=[html.B("Columns, types and missing values"),
                                                      table(q["columns"], {"Missing": INT, "Missing %": DEC3,
                                                                           "Unique values": INT}, page_size=25)]),
             html.Div(className="card pad", children=[
                 html.B("Target distribution"), table(q["target"], {"Customers": INT}),
                 html.B("Unusual numeric values"), table(q["unusual_values"], {"Rows": INT})])),
        html.Div(className="card pad", children=[html.B("Unexpected or undocumented category codes (kept, not deleted)"),
                                                 table(q["unexpected_codes"], {"Rows": INT}, page_size=40)]),
        html.Div(className="card pad", children=[html.B("Descriptive statistics (numeric variables)"),
                                                 table(q["describe"].reset_index().rename(columns={"index": "Variable"}),
                                                       {c: INT for c in q["describe"].columns})]),
        html.Div(className="button-row", children=[
            html.Button("⬇ Data-quality summary (CSV)", id="btn-dl-quality", className="btn")]),
    ]


PAGE_BUILDERS = {"/": page_overview, "/exploration": page_exploration, "/segmentation": page_segmentation,
                 "/individual": page_individual, "/performance": page_performance, "/quality": page_quality}


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Credit Risk Dashboard", suppress_callback_exceptions=True)


def model_info_text() -> str:
    b = STATE["bundle"]
    return (f"Logistic regression · class weight: {b['class_weight'] or 'none (standard)'}. "
            f"Trained {b['trained_at'].replace('T', ' ')} on {b['n_train']:,} customers.")


if LOAD_ERROR:
    app.layout = html.Div(className="main solo", children=[
        html.Div(className="app-header", children=[html.Div("Credit Risk Analytics", className="title")]),
        html.Div(className="alert error", children=[html.B("The dashboard could not start. "), LOAD_ERROR]),
        note("Place the UCI 'Default of Credit Card Clients' file (.xls, .xlsx or .csv) in the project folder and "
             "restart the app. No synthetic data is generated."),
    ])
else:
    app.layout = html.Div(className="shell", children=[
        dcc.Location(id="url"),
        dcc.Store(id="model-version", data=0),
        dcc.Download(id="dl-seg"), dcc.Download(id="dl-metrics"), dcc.Download(id="dl-test"),
        dcc.Download(id="dl-quality"),
        html.Aside(className="sidebar", children=[
            html.Div("💳 Credit Risk", className="brand"),
            html.Nav(id="nav", className="nav"),
            html.Hr(),
            html.B("Risk-segment thresholds"),
            caption("Configurable demonstration thresholds, not banking standards. Left handle = Medium starts, "
                    "right handle = High starts."),
            dcc.RangeSlider(id="thresholds", min=5, max=95, step=1, value=DEFAULT_THRESHOLDS, pushable=5,
                            marks={v: f"{v}%" for v in (5, 25, 50, 75, 95)}, tooltip={"placement": "top"}),
            html.Div(id="threshold-text", className="caption"),
            html.Hr(),
            html.B("Model"),
            html.Div(id="model-info", className="caption"),
            html.Details(className="retrain", children=[
                html.Summary("Retrain model"),
                dcc.Checklist(id="balanced", options=[{"label": ' Use class_weight="balanced"', "value": "balanced"}],
                              value=["balanced"] if STATE["bundle"]["class_weight"] == "balanced" else []),
                caption("Balanced weighting raises recall at the 50% cut-off but makes probabilities inflated."),
                html.Button("Retrain now", id="btn-retrain", className="btn primary"),
                dcc.Loading(html.Div(id="retrain-msg", className="caption"), type="dot", color=charts.NAVY),
            ]),
        ]),
        html.Main(className="main", children=[
            html.Div(className="app-header", children=[
                html.Div("Credit Risk Analytics", className="title"),
                html.Div(id="header-sub", className="sub")]),
            dcc.Loading(html.Div(id="page"), type="circle", color=charts.NAVY, delay_show=300),
        ]),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(Output("page", "children"), Output("nav", "children"), Output("header-sub", "children"),
          Output("threshold-text", "children"), Output("model-info", "children"),
          Input("url", "pathname"), Input("thresholds", "value"), Input("model-version", "data"))
def render_page(path, thresholds, _version):
    path = path if path in PAGES else "/"
    nav = [dcc.Link(name, href=href, className="nav-link active" if href == path else "nav-link")
           for href, name in PAGES.items()]
    medium, high = thresholds_to_float(thresholds)
    t_text = f"Low < {medium:.0%} ≤ Medium < {high:.0%} ≤ High"
    sub = f"{PAGES[path]} · UCI Default of Credit Card Clients (Taiwan, 2005)"
    return PAGE_BUILDERS[path](thresholds), nav, sub, t_text, model_info_text()


@callback(Output("model-version", "data"), Output("retrain-msg", "children"),
          Input("btn-retrain", "n_clicks"), State("balanced", "value"), State("model-version", "data"),
          prevent_initial_call=True)
def retrain(_clicks, balanced, version):
    try:
        STATE["bundle"] = train_and_save(DF, INFO, "balanced" if balanced else None)
    except Exception as exc:
        return no_update, f"Training failed: {exc}. The previous model is still in use."
    STATE["version"] = version + 1
    m = STATE["bundle"]["metrics"]
    return STATE["version"], f"Retrained. Test ROC-AUC {m['roc_auc']:.3f}, recall {m['recall']:.3f}."


# --- Data Exploration ---
@callback(Output("bill-chart", "figure"), Output("pay-chart", "figure"), Input("stat-radio", "value"))
def update_bill_payment(stat):
    cmp = bill_payment_comparison(DF)
    return (charts.bill_payment_bars(cmp, "BILL_AMT", "Bill statement amount", stat),
            charts.bill_payment_bars(cmp, "PAY_AMT", "Amount paid", stat))


@callback(Output("dist-chart", "figure"), Output("dist-caption", "children"), Input("dist-var", "value"))
def update_distribution(col):
    label = {"LIMIT_BAL": "credit limit", "AGE": "age"}.get(col, col)
    fig, hidden, lo, hi = charts.distribution_hist(DATA, col, label)
    text = "Each outcome group is scaled to 100% so the two shapes can be compared fairly."
    if hidden:
        text += (f" To keep the chart readable, {hidden:,} extreme values outside NT$ {lo:,.0f} – {hi:,.0f} "
                 "(0.5th–99th percentile) are not drawn.")
    return fig, text


# --- Risk Segmentation ---
def filtered(thresholds, seg, age, lim, split) -> pd.DataFrame:
    pred = predictions(thresholds)
    return pred[pred["Risk_Segment"].isin(seg or []) & pred["Age_Group"].isin(age or [])
                & pred["Credit_Limit_Range"].isin(lim or []) & pred["Data_Split"].isin(split or [])]


SEG_FILTERS = [Input("f-seg", "value"), Input("f-age", "value"), Input("f-lim", "value"), Input("f-split", "value")]


@callback(Output("seg-table", "children"), Output("seg-count", "children"),
          *SEG_FILTERS, Input("f-top", "value"), State("thresholds", "value"))
def update_segment_table(seg, age, lim, split, top_n, thresholds):
    view = filtered(thresholds, seg, age, lim, split)
    count = f"{len(view):,} of {len(DF):,} customers match the filters."
    if view.empty:
        return html.Div("No customers match these filters.", className="alert info"), count
    cols = [ID_COL, "Predicted_Default_Probability", "Risk_Segment", "Actual_Default", "Data_Split",
            "LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
    top = view.nlargest(top_n or 25, "Predicted_Default_Probability")[cols].rename(columns={
        "Predicted_Default_Probability": "Predicted probability", "Actual_Default": "Actual default (historical)",
        "LIMIT_BAL": "Credit limit", "BILL_AMT1": "Bill Sep", "PAY_AMT1": "Paid Sep"})
    return table(top, {"Predicted probability": PCT, "Credit limit": INT, "Bill Sep": INT, "Paid Sep": INT},
                 page_size=25), count


@callback(Output("dl-seg", "data"), Input("btn-dl-seg", "n_clicks"),
          State("f-seg", "value"), State("f-age", "value"), State("f-lim", "value"), State("f-split", "value"),
          State("thresholds", "value"), prevent_initial_call=True)
def download_segments(_n, seg, age, lim, split, thresholds):
    view = filtered(thresholds, seg, age, lim, split).drop(columns=["Age_Group", "Credit_Limit_Range"])
    return dcc.send_data_frame(view.to_csv, "customer_predictions_filtered.csv", index=False)


@callback(Output("save-msg", "children"), Input("btn-save-seg", "n_clicks"), State("thresholds", "value"),
          prevent_initial_call=True)
def save_segments(_n, thresholds):
    medium, high = thresholds_to_float(thresholds)
    paths = export_predictions(DF, STATE["bundle"], medium_threshold=medium, high_threshold=high)
    return "Saved: " + ", ".join(str(p.relative_to(OUTPUTS_DIR.parent)) for p in paths)


# --- Individual Risk Assessment ---
@callback(Output("existing-box", "style"), Output("new-box", "style"), Input("ind-mode", "value"))
def toggle_mode(mode):
    show, hide = {"display": "block"}, {"display": "none"}
    return (show, hide) if mode == "existing" else (hide, show)


@callback(Output("existing-result", "children"), Input("cust-id", "value"), State("thresholds", "value"))
def show_existing(cid, thresholds):
    if cid is None:
        return html.Div("Enter a customer ID.", className="alert info")
    pred = predictions(thresholds)
    match = pred[pred[ID_COL] == int(cid)]
    if match.empty:
        return html.Div(f"No customer with ID {cid} was found.", className="alert warn")
    r = match.iloc[0]
    proba = float(r["Predicted_Default_Probability"])
    return html.Div(className="grid grid-2 individual", children=[
        html.Div(className="card pad", children=[html.B(f"Customer ID {int(cid)}"),
                                                 table(customer_details(r), page_size=20, sort_action="none")]),
        html.Div([probability_result(proba, thresholds, r["Actual_Default"], r["Data_Split"]),
                  explanation(match[FEATURE_COLS], proba)]),
    ])


@callback(Output("new-result", "children"), Input("btn-estimate", "n_clicks"),
          [State(f"in-{c}", "value") for c in FEATURE_COLS], State("thresholds", "value"),
          prevent_initial_call=True)
def estimate_new(_n, *args):
    *field_values, thresholds = args
    values = dict(zip(FEATURE_COLS, field_values))
    errors = validate_customer_input(values)
    if errors:
        return html.Div([html.B("Please fix the following:"), html.Ul([html.Li(e) for e in errors])],
                        className="alert error")
    row = pd.DataFrame([values])[FEATURE_COLS]
    try:
        proba = float(predict_probability(STATE["bundle"], row)[0])
    except Exception as exc:
        return html.Div(f"The prediction failed: {exc}", className="alert error")
    return html.Div(className="grid grid-2 individual", children=[
        html.Div(className="card pad", children=[probability_result(proba, thresholds)]),
        explanation(row, proba)])


# --- Model Performance ---
@callback(Output("threshold-metrics", "children"), Input("perf-threshold", "value"))
def update_threshold(t):
    bundle = STATE["bundle"]
    tr, base = bundle["test_results"], bundle["metrics"]
    mt = evaluate(tr["Actual_Default"].values, tr["Predicted_Default_Probability"].values, threshold=t)
    cards = []
    for name, key in [("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"), ("Accuracy", "accuracy")]:
        delta = mt[key] - base[key]
        note_ = "" if abs(t - bundle["decision_threshold"]) < 1e-9 else f"{delta:+.3f} vs 0.50"
        cards.append(kpi(name, f"{mt[key]:.3f}", note_, True, tag=False))
    cm = mt["confusion_matrix"]
    return html.Div([
        html.Div(cards, className="kpi-row four"),
        caption(f"At {t:.2f}: {cm['tp']:,} defaulters caught, {cm['fn']:,} missed, {cm['fp']:,} false alarms. "
                "Lowering the threshold raises recall but creates more false alarms. Choosing it is a business "
                "decision about which error costs more."),
    ])


@callback(Output("dl-metrics", "data"), Input("btn-dl-metrics", "n_clicks"), prevent_initial_call=True)
def download_metrics(_n):
    content = (METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists()
               else json.dumps(STATE["bundle"]["metrics"], default=float, indent=2))
    return dict(content=content, filename="model_metrics.json")


@callback(Output("dl-test", "data"), Input("btn-dl-test", "n_clicks"), State("thresholds", "value"),
          prevent_initial_call=True)
def download_test(_n, thresholds):
    test = predictions(thresholds)
    test = test[test["Data_Split"] == "Test"].drop(columns=["Age_Group", "Credit_Limit_Range"])
    test["Correct_Prediction"] = (test["Predicted_Default"] == test["Actual_Default"]).astype(int)
    return dcc.send_data_frame(test.to_csv, "test_predictions.csv", index=False)


# --- Data Quality ---
@callback(Output("dl-quality", "data"), Input("btn-dl-quality", "n_clicks"), prevent_initial_call=True)
def download_quality(_n):
    path = export_quality_report(QUALITY)
    return dcc.send_file(str(path))


if __name__ == "__main__":
    # debug=False keeps start-up fast and avoids loading the data twice.
    app.run(host="127.0.0.1", port=8050, debug=False)
