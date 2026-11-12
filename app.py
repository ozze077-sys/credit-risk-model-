"""
app.py - Credit Risk Analytics & Default Prediction Dashboard
=============================================================
Start with:   streamlit run app.py

How the app is organised
------------------------
* Data loading, training and prediction logic live in separate modules
  (data_loader.py, model_training.py, prediction.py, data_analysis.py).
  This file only handles the dashboard: layout, charts and user input.
* ``st.cache_data`` / ``st.cache_resource`` remember expensive results
  (reading the Excel file, loading the model, scoring 30,000 customers), so
  moving a slider doesn't reload or retrain anything.
* The model is trained automatically only when no valid model exists or the
  dataset file has changed. Otherwise use "Retrain model" in the sidebar.

Colour conventions
------------------
* Blue / orange  = HISTORICAL outcomes (what actually happened).
* Violet         = model PREDICTIONS (estimated probabilities).
* Green / amber / red = predicted risk segments (always shown with a text label).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_analysis import (AGE_LABELS, LIMIT_LABELS, add_groups, age_group, bill_payment_comparison,
                           correlation_matrix, credit_limit_stats, default_rate_by, limit_range)
from data_loader import (BILL_COLS, EDUCATION_LABELS, FEATURE_COLS, ID_COL,
                         MARRIAGE_LABELS, MONTHS, PAY_AMT_COLS, PAY_STATUS_COLS,
                         PAY_STATUS_MONTH, SEX_LABELS, TARGET_COL, TARGET_LABELS, DatasetError,
                         category_label, data_quality_report, export_quality_report, find_dataset_files,
                         load_dataset, pay_status_label)
from model_training import METRICS_PATH, MODEL_PATH, OUTPUTS_DIR, evaluate, save_bundle, train_model
from prediction import (DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD, SEGMENTS, ModelError,
                        assign_segments, build_prediction_table, explain_prediction, export_predictions,
                        load_model, model_matches_data, predict_probability, segment_summary,
                        validate_customer_input)

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
NAVY = "#0B1F3A"
INK = "#14213D"
MUTED = "#5B6780"
GRID = "#E6EAF0"
OUTCOME_COLORS = dict(zip(TARGET_LABELS.values(), ["#2a78d6", "#eb6834"]))
HIST_COLOR = "#2a78d6"
PRED_COLOR = "#6250d6"
SEGMENT_COLORS = {"Low risk": "#0ca30c", "Medium risk": "#fab219", "High risk": "#d03b3b"}
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}}
  header[data-testid="stHeader"] {{background: transparent;}}
  .app-header {{background:{NAVY}; color:#fff; padding:14px 22px; border-radius:8px; margin-bottom:6px;
               display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;}}
  .app-header .title {{font-size:1.25rem; font-weight:600; letter-spacing:.01em;}}
  .app-header .sub {{color:#B8C4D9; font-size:.82rem;}}
  .kpi {{border:1px solid #DDE3EC; border-left:4px solid {HIST_COLOR}; border-radius:8px; padding:12px 16px;
         background:#fff; min-height:118px; margin-bottom:12px;}}
  .kpi.pred {{border-left-color:{PRED_COLOR};}}
  .kpi-label {{font-size:.74rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.05em;}}
  .kpi-value {{font-size:1.55rem; font-weight:650; color:{NAVY}; margin-top:2px;}}
  .kpi-note {{font-size:.78rem; color:{MUTED};}}
  .tag {{display:inline-block; font-size:.66rem; padding:1px 6px; border-radius:4px; margin-left:4px;
         vertical-align:middle; font-weight:600;}}
  .tag.hist {{background:#E3EEFB; color:#1d5aa6;}}
  .tag.pred {{background:#ECEAFB; color:#4a3aa7;}}
  .note {{background:#F4F6FA; border:1px solid #DDE3EC; border-radius:8px; padding:10px 14px;
          font-size:.9rem; color:#33415C; margin:6px 0 12px 0;}}
  .seg {{display:inline-flex; align-items:center; gap:8px; font-weight:600; font-size:1.05rem; color:{INK};}}
  .seg .dot {{width:14px; height:14px; border-radius:50%; display:inline-block;}}
  section[data-testid="stSidebar"] {{background:#F4F6FA;}}
  h2, h3 {{color:{NAVY};}}
</style>
""", unsafe_allow_html=True)


def header(page: str) -> None:
    st.markdown(f'<div class="app-header"><div class="title">Credit Risk Analytics</div>'
                f'<div class="sub">{page} · UCI Default of Credit Card Clients (Taiwan, 2005)</div></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = "", predicted: bool = False, show_tag: bool = True) -> None:
    tag = '<span class="tag pred">MODEL</span>' if predicted else '<span class="tag hist">HISTORICAL</span>'
    tag = tag if show_tag else ""
    col.markdown(f'<div class="kpi {"pred" if predicted else ""}"><div class="kpi-label">{label}{tag}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
                 unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", pct_y: bool = False,
          height: int = 380, legend: bool = True) -> go.Figure:
    """Apply one consistent style to every chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=78 if "<br>" in title else 56, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK, size=12),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12), bargap=0.25,
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor="#C9D1DD")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=True, zerolinecolor="#C9D1DD", rangemode="tozero")
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def rate_bar(table: pd.DataFrame, title: str, x_title: str, overall: float | None = None) -> go.Figure:
    """Bar chart of historical default rate per group (hover shows group size).

    Groups with long labels (e.g. repayment status) are drawn as horizontal bars so labels stay readable.
    """
    horizontal = table["Group"].astype(str).str.len().max() > 16
    if overall is not None:
        title += f"<br><span style='font-size:12px;color:{MUTED};font-weight:normal'>Dotted line = all customers " \
                 f"({overall:.1%})</span>"
    top = min(1, table["Default_Rate"].max() * 1.2 + 0.02)
    common = dict(marker_color=HIST_COLOR, marker_cornerradius=4, customdata=table[["Customers", "Defaults"]].values,
                  text=[f"{v:.1%}" for v in table["Default_Rate"]], textposition="outside", cliponaxis=False)
    hover = ("Historical default rate: %{VAL:.1%}<br>Customers: %{customdata[0]:,}<br>"
             "Defaults: %{customdata[1]:,}<extra></extra>")
    if horizontal:
        fig = go.Figure(go.Bar(y=table["Group"].astype(str), x=table["Default_Rate"], orientation="h", **common,
                               hovertemplate="<b>%{y}</b><br>" + hover.replace("VAL", "x")))
        style(fig, title, "Historical default rate", "", legend=False, height=max(380, 34 * len(table) + 110))
        fig.update_xaxes(tickformat=".0%", range=[0, top], showgrid=True, gridcolor=GRID)
        fig.update_yaxes(autorange="reversed", rangemode="normal", showgrid=False)
        if overall is not None:
            fig.add_vline(x=overall, line_dash="dot", line_color=MUTED)
        return fig
    fig = go.Figure(go.Bar(x=table["Group"].astype(str), y=table["Default_Rate"], **common,
                           hovertemplate="<b>%{x}</b><br>" + hover.replace("VAL", "y")))
    style(fig, title, x_title, "Historical default rate", pct_y=True, legend=False)
    fig.update_yaxes(range=[0, top])
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dot", line_color=MUTED)
    return fig


def fmt_money(v) -> str:
    return f"NT$ {v:,.0f}"


# ---------------------------------------------------------------------------
# Cached data / model access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the dataset…")
def cached_dataset(files_key: tuple):
    # ``files_key`` (file names + modification times) is only the cache key:
    # if a data file is added, removed or changed, everything is reloaded.
    return load_dataset()


@st.cache_data(show_spinner=False)
def cached_quality(fingerprint: str, _df: pd.DataFrame, _info: dict):
    return data_quality_report(_df, _info)


@st.cache_resource(show_spinner="Loading model…")
def cached_model(path: str, modified: float):
    return load_model(Path(path))


@st.cache_data(show_spinner="Scoring customers…")
def cached_predictions(model_key: str, medium: float, high: float, _df: pd.DataFrame, _bundle: dict):
    pred = build_prediction_table(_df, _bundle, medium, high)
    pred["Age_Group"] = age_group(pred["AGE"])
    pred["Credit_Limit_Range"] = limit_range(pred["LIMIT_BAL"])
    return pred


def model_key(bundle: dict) -> str:
    return f"{bundle['trained_at']}|{bundle['class_weight']}|{bundle['dataset_fingerprint']}"


def retrain(df: pd.DataFrame, info: dict, class_weight: str | None) -> dict:
    """Train, save and export. Shows a friendly error instead of crashing."""
    try:
        with st.spinner("Training the logistic-regression model (about 30 seconds)…"):
            bundle = train_model(df, class_weight=class_weight, dataset_info=info)
            save_bundle(bundle)
            export_predictions(df, bundle)
            export_quality_report(cached_quality(info["fingerprint"], df, info))
    except Exception as exc:
        st.error(f"The model could not be trained: {exc}. No predictions are shown because "
                 "they would not be based on a valid model.")
        st.stop()
    cached_model.clear()
    cached_predictions.clear()
    return bundle


# ---------------------------------------------------------------------------
# Load data and model
# ---------------------------------------------------------------------------
try:
    data_files = find_dataset_files()
    df, info = cached_dataset(tuple((p.name, p.stat().st_mtime) for p in data_files))
except DatasetError as exc:
    header("Setup problem")
    st.error(f"**The dataset could not be loaded.**\n\n{exc}")
    st.info("Place the UCI 'Default of Credit Card Clients' file (.xls, .xlsx or .csv) in the project "
            "folder and refresh the page. No synthetic data is generated.")
    st.stop()

bundle, model_problem = None, None
if MODEL_PATH.exists():
    try:
        bundle = cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        if not model_matches_data(bundle, info):
            model_problem = "the saved model was trained on a different version of the dataset"
            bundle = None
    except ModelError as exc:
        model_problem = str(exc)
else:
    model_problem = "no saved model was found"
if bundle is None:
    st.toast(f"Training a new model because {model_problem}.")
    bundle = retrain(df, info, class_weight=None)

quality = cached_quality(info["fingerprint"], df, info)
data = add_groups(df)
overall_rate = df[TARGET_COL].mean()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Overview", "Data Exploration", "Risk Segmentation", "Individual Risk Assessment",
         "Model Performance", "Data Quality"]
with st.sidebar:
    st.markdown(f"### 💳 Credit Risk")
    # A link such as http://localhost:8501/?page=Data+Quality opens that page directly.
    requested = st.query_params.get("page")
    page = st.radio("Page", PAGES, index=PAGES.index(requested) if requested in PAGES else 0,
                    label_visibility="collapsed")

    st.divider()
    st.markdown("**Risk-segment thresholds**")
    st.caption("Configurable demonstration thresholds, not banking standards.")
    medium_pct = st.slider("Medium risk starts at (%)", 5, 90, int(DEFAULT_MEDIUM_THRESHOLD * 100), 1)
    high_pct = st.slider("High risk starts at (%)", 10, 95, int(DEFAULT_HIGH_THRESHOLD * 100), 1)
    if medium_pct >= high_pct:
        st.error("The medium threshold must be lower than the high threshold. Using 30% / 50%.")
        medium_pct, high_pct = int(DEFAULT_MEDIUM_THRESHOLD * 100), int(DEFAULT_HIGH_THRESHOLD * 100)
    medium_t, high_t = medium_pct / 100, high_pct / 100

    st.divider()
    st.markdown("**Model**")
    st.caption(f"Logistic regression · class weight: {bundle['class_weight'] or 'none (standard)'}\n\n"
               f"Trained {bundle['trained_at'].replace('T', ' ')} on {bundle['n_train']:,} customers.")
    with st.expander("Retrain model"):
        balanced = st.checkbox('Use class_weight="balanced"', value=bundle["class_weight"] == "balanced",
                               help="Gives defaulters more weight during training. Recall at the 50% cut-off "
                                    "rises, but probabilities become inflated (less well calibrated).")
        if st.button("Retrain now", type="primary", width="stretch"):
            bundle = retrain(df, info, "balanced" if balanced else None)
            st.rerun()

pred = cached_predictions(model_key(bundle), medium_t, high_t, df, bundle)
metrics = bundle["metrics"]
header(page)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.subheader("Portfolio overview")
    note("<b>Historical</b> figures (blue) describe what actually happened in the data. "
         "<b>Model</b> figures (violet) are estimates from the logistic-regression model. "
         "This dataset is a single snapshot (Apr–Sep 2005), so no time trend is shown.")
    c = st.columns(6)
    kpi(c[0], "Total customers", f"{len(df):,}", "records with a valid outcome")
    kpi(c[1], "Actual default rate", f"{overall_rate:.1%}", "defaulted the following month")
    kpi(c[2], "Actual defaults", f"{int(df[TARGET_COL].sum()):,}", f"of {len(df):,} customers")
    kpi(c[3], "Avg credit limit", f"{df['LIMIT_BAL'].mean() / 1000:,.1f}K",
        f"NT$ · median {fmt_money(df['LIMIT_BAL'].median())}")
    n_high = int((pred["Risk_Segment"] == "High risk").sum())
    kpi(c[4], "Predicted high risk", f"{n_high:,}", f"probability ≥ {high_pct}% ({n_high / len(pred):.1%})", True)
    kpi(c[5], "Avg predicted probability", f"{pred['Predicted_Default_Probability'].mean():.1%}",
        "mean of model estimates", True)

    left, right = st.columns(2)
    with left:
        counts = df[TARGET_COL].map(TARGET_LABELS).value_counts().reindex(TARGET_LABELS.values()).reset_index()
        counts.columns = ["Outcome", "Customers"]
        counts["Share"] = counts["Customers"] / counts["Customers"].sum()
        fig = px.bar(counts, x="Outcome", y="Customers", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                     text=[f"{n:,} ({s:.1%})" for n, s in zip(counts["Customers"], counts["Share"])],
                     custom_data=["Share"])
        fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br>Share: %{customdata[0]:.1%}<extra></extra>")
        show(style(fig, "Historical outcome: default vs no default", "", "Customers", legend=False))
    with right:
        show(rate_bar(default_rate_by(df, "PAY_0"), "Historical default rate by Sep 2005 repayment status (PAY_0)",
                      "Repayment status", overall_rate))

    left, right = st.columns(2)
    with left:
        seg = segment_summary(pred)
        fig = go.Figure(go.Bar(
            x=seg["Risk_Segment"], y=seg["Customers"], marker_color=[SEGMENT_COLORS[s] for s in seg["Risk_Segment"]],
            marker_cornerradius=4, text=[f"{n:,} ({s:.1%})" for n, s in zip(seg["Customers"], seg["Share"])],
            textposition="outside", cliponaxis=False, customdata=seg[["Share", "Avg_Probability"]].values,
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br>Share: %{customdata[0]:.1%}"
                          "<br>Avg predicted probability: %{customdata[1]:.1%}<extra></extra>"))
        show(style(fig, f"Predicted risk segments (thresholds {medium_pct}% / {high_pct}%)", "", "Customers",
                   legend=False))
    with right:
        show(probability_histogram(pred))

    imp = bundle["permutation_importance"].head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["AUC_Drop_Mean"], y=imp["Feature"], orientation="h", marker_color=PRED_COLOR,
                           marker_cornerradius=4, error_x=dict(type="data", array=imp["AUC_Drop_Std"], color=MUTED),
                           hovertemplate="<b>%{y}</b><br>ROC-AUC drop when shuffled: %{x:.4f}<extra></extra>"))
    style(fig, "Most important model inputs (permutation importance on the test set)",
          "Drop in ROC-AUC when the column is shuffled", "", legend=False, height=420)
    fig.update_yaxes(rangemode="normal")
    show(fig)
    st.caption("Importance shows how much the model *relies* on an input to rank customers, measured on unseen "
               "test data. It does not mean the input *causes* default.")


def probability_histogram(p: pd.DataFrame, title: str = "Predicted default probability distribution") -> go.Figure:
    fig = go.Figure(go.Histogram(x=p["Predicted_Default_Probability"], xbins=dict(start=0, end=1, size=0.025),
                                 marker_color=PRED_COLOR, marker_line=dict(color="white", width=1),
                                 hovertemplate="Probability %{x}<br>Customers: %{y:,}<extra></extra>"))
    style(fig, title, "Predicted probability of default", "Customers", legend=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    for t, label, pos in [(medium_t, "Medium", "top left"), (high_t, "High", "top right")]:
        fig.add_vline(x=t, line_dash="dash", line_color=NAVY, annotation_text=f"{label} ≥ {t:.0%}",
                      annotation_position=pos, annotation_font_color=NAVY)
    return fig


def page_exploration() -> None:
    st.subheader("Data exploration (historical outcomes)")
    note("All rates on this page are <b>historical</b>: the share of customers in a group who actually "
         "defaulted. They describe associations in this dataset, not causes, and they are not individual "
         "predictions. Hover over bars to see group sizes; small groups give unreliable rates.")
    lim = credit_limit_stats(df)
    n_def = int(df[TARGET_COL].sum())
    c = st.columns(6)
    kpi(c[0], "No default", f"{len(df) - n_def:,}")
    kpi(c[1], "Default", f"{n_def:,}")
    kpi(c[2], "Default rate", f"{overall_rate:.1%}")
    kpi(c[3], "Avg credit limit", f"{lim['mean'] / 1000:,.1f}K", f"NT$ · median {lim['median'] / 1000:,.0f}K")
    kpi(c[4], "Min credit limit", f"{lim['min'] / 1000:,.0f}K", "NT$")
    kpi(c[5], "Max credit limit", f"{lim['max'] / 1000:,.0f}K", "NT$")

    tabs = st.tabs(["Default rates by group", "Defaulters vs non-defaulters", "Correlation", "Distributions"])
    with tabs[0]:
        l, r = st.columns(2)
        with l:
            show(rate_bar(default_rate_by(data, "Age_Group"), "Default rate by age group", "Age group", overall_rate))
            show(rate_bar(default_rate_by(df, "PAY_0"), "Default rate by Sep 2005 repayment status (PAY_0)",
                          "Repayment status", overall_rate))
            show(rate_bar(default_rate_by(df, "MARRIAGE"), "Default rate by marital status", "Marital status",
                          overall_rate))
        with r:
            show(rate_bar(default_rate_by(data, "Credit_Limit_Range"), "Default rate by credit-limit range",
                          "Credit limit (NT$)", overall_rate))
            show(rate_bar(default_rate_by(df, "EDUCATION"), "Default rate by education", "Education", overall_rate))
            with st.container(border=True):
                st.markdown("**What the repayment-status codes mean**")
                st.markdown(
                    "- **-2**: no balance / no credit used that month\n"
                    "- **-1**: paid in full (\"pay duly\")\n"
                    "- **0**: revolving credit — minimum paid, balance carried over\n"
                    "- **1 – 8**: payment delayed by that many months\n\n"
                    "Codes -1 and 1–9 come from the UCI documentation. Codes -2 and 0 are not in the original "
                    "documentation; the meanings above are the widely used interpretation. They are "
                    "**categories**, not negative amounts, so the model one-hot encodes them.")
            st.caption("EDUCATION codes 0, 5, 6 and MARRIAGE code 0 are undocumented; they are kept as "
                       "separate groups rather than deleted.")

    with tabs[1]:
        l, r = st.columns(2)
        with l:
            lim_cmp = df.groupby(df[TARGET_COL].map(TARGET_LABELS))["LIMIT_BAL"].agg(["mean", "median"]) \
                        .reindex(TARGET_LABELS.values()).reset_index()
            lim_cmp.columns = ["Outcome", "Mean", "Median"]
            long = lim_cmp.melt(id_vars="Outcome", var_name="Statistic", value_name="Credit limit")
            fig = px.bar(long, x="Statistic", y="Credit limit", color="Outcome", barmode="group",
                         color_discrete_map=OUTCOME_COLORS, text=[f"{v / 1000:,.0f}K" for v in long["Credit limit"]])
            fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                              hovertemplate="%{fullData.name}<br>%{x}: NT$ %{y:,.0f}<extra></extra>")
            show(style(fig, "Credit limit: defaulters vs non-defaulters", "", "Credit limit (NT$)"))
        with r:
            fig = px.box(data, x="Outcome", y="LIMIT_BAL", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                         category_orders={"Outcome": list(TARGET_LABELS.values())})
            show(style(fig, "Credit limit spread by outcome (box plot)", "", "Credit limit (NT$)", legend=False))

        cmp = bill_payment_comparison(df)
        cmp["Month"] = cmp["Variable"].str[-1].astype(int).map(MONTHS)
        month_order = list(MONTHS.values())[::-1]
        stat = st.radio("Statistic", ["Mean", "Median"], horizontal=True,
                        help="Amounts are very skewed (a few huge values), so the median is often more typical.")
        l, r = st.columns(2)
        for col, prefix, label in [(l, "BILL_AMT", "Bill statement amount"), (r, "PAY_AMT", "Amount paid")]:
            part = cmp[cmp["Variable"].str.startswith(prefix)]
            fig = px.bar(part, x="Month", y=stat, color="Outcome", barmode="group", color_discrete_map=OUTCOME_COLORS,
                         category_orders={"Month": month_order}, custom_data=["Variable"])
            fig.update_traces(marker_cornerradius=4, hovertemplate="%{fullData.name} · %{x} (%{customdata[0]})"
                                                                  f"<br>{stat}: NT$ %{{y:,.0f}}<extra></extra>")
            with col:
                show(style(fig, f"{stat} {label.lower()} by month and outcome", "Month", f"{label} (NT$)"))

    with tabs[2]:
        corr = correlation_matrix(df).rename(index={TARGET_COL: "DEFAULT (target)"},
                                             columns={TARGET_COL: "DEFAULT (target)"})
        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING, text_auto=".2f", aspect="auto")
        fig.update_traces(hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>", textfont_size=9)
        style(fig, "Correlation between numeric variables (Pearson)", height=680, legend=False)
        fig.update_yaxes(rangemode="normal", gridcolor=None)
        show(fig)
        st.caption("Correlation measures linear association only (−1 to +1). It does not imply causation. "
                   "PAY_x codes are included as numbers here only for a quick overview.")

    with tabs[3]:
        options = {"Credit limit (LIMIT_BAL)": "LIMIT_BAL", "Age (AGE)": "AGE",
                   **{f"Bill amount {MONTHS[i]} ({c})": c for i, c in enumerate(BILL_COLS, 1)},
                   **{f"Payment amount {MONTHS[i]} ({c})": c for i, c in enumerate(PAY_AMT_COLS, 1)}}
        choice = st.selectbox("Variable", list(options))
        col = options[choice]
        lo, hi = data[col].quantile([0.005, 0.99]) if col not in ("AGE", "LIMIT_BAL") else (data[col].min(), data[col].max())
        shown = data[(data[col] >= lo) & (data[col] <= hi)]
        fig = px.histogram(shown, x=col, color="Outcome", barmode="overlay", histnorm="percent", nbins=60,
                           opacity=0.6, color_discrete_map=OUTCOME_COLORS,
                           category_orders={"Outcome": list(TARGET_LABELS.values())})
        fig.update_traces(hovertemplate="%{fullData.name}<br>Range: %{x}<br>%{y:.1f}% of this outcome group<extra></extra>")
        show(style(fig, f"Distribution of {choice} by outcome", choice, "% of customers in each outcome group"))
        hidden = len(data) - len(shown)
        st.caption("Each outcome group is scaled to 100% so the two shapes can be compared fairly. "
                   + (f"To keep the chart readable, {hidden:,} extreme values outside "
                      f"NT$ {lo:,.0f} – {hi:,.0f} (0.5th–99th percentile) are not drawn." if hidden else ""))


def page_segmentation() -> None:
    st.subheader("Risk segmentation (model predictions)")
    note(f"Customers are grouped by their <b>predicted</b> probability of default: Low &lt; {medium_pct}% ≤ "
         f"Medium &lt; {high_pct}% ≤ High. These are <b>configurable demonstration thresholds</b> (change them "
         "in the sidebar), not established banking standards or validated lending decisions. "
         "Predictions for training-set customers are slightly optimistic because the model learned from them.")

    seg = segment_summary(pred)
    c = st.columns(3)
    for col, (_, r) in zip(c, seg.iterrows()):
        avg = "–" if pd.isna(r["Avg_Probability"]) else f"{r['Avg_Probability']:.1%}"
        kpi(col, r["Risk_Segment"], f"{r['Customers']:,} ({r['Share']:.1%})",
            f"avg predicted probability {avg}", True)

    l, r = st.columns(2)
    with l:
        show(probability_histogram(pred, "Predicted probability distribution with segment thresholds"))
    with r:
        tbl = seg.rename(columns={"Risk_Segment": "Segment", "Avg_Probability": "Avg predicted",
                                  "Actual_Default_Rate": "Actual rate"})[
            ["Segment", "Customers", "Share", "Avg predicted", "Actual rate"]]
        st.markdown("**Segment summary**")
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            "Customers": st.column_config.NumberColumn(format="localized"),
            "Share": st.column_config.NumberColumn(format="percent"),
            "Avg predicted": st.column_config.NumberColumn("Avg predicted probability", format="percent"),
            "Actual rate": st.column_config.NumberColumn("Actual default rate (historical)", format="percent")})
        st.caption("Comparing the average predicted probability with the actual historical default rate in "
                   "each segment is a quick sanity check of the model's probabilities. (Actual rates include "
                   "training customers, so they look slightly better than on new data.)")

    st.markdown("#### Filter customers")
    f = st.columns(4)
    seg_f = f[0].multiselect("Risk segment", SEGMENTS, default=SEGMENTS)
    age_f = f[1].multiselect("Age group", AGE_LABELS, default=AGE_LABELS)
    lim_f = f[2].multiselect("Credit-limit range", LIMIT_LABELS, default=LIMIT_LABELS)
    split_f = f[3].multiselect("Data split", ["Train", "Test"], default=["Train", "Test"],
                               help="Test = held-out customers the model never saw during training.")
    view = pred[pred["Risk_Segment"].isin(seg_f) & pred["Age_Group"].isin(age_f)
                & pred["Credit_Limit_Range"].isin(lim_f) & pred["Data_Split"].isin(split_f)]
    st.caption(f"{len(view):,} of {len(pred):,} customers match the filters.")
    if view.empty:
        st.info("No customers match these filters.")
        return

    top_n = st.slider("Show the highest-risk customers", 10, 200, 25, 5)
    cols = [ID_COL, "Predicted_Default_Probability", "Risk_Segment", "Actual_Default", "Data_Split",
            "LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
    top = view.nlargest(top_n, "Predicted_Default_Probability")[cols]
    st.dataframe(top, hide_index=True, width="stretch", column_config={
        "Predicted_Default_Probability": st.column_config.ProgressColumn(
            "Predicted probability", format="percent", min_value=0, max_value=1),
        "Actual_Default": st.column_config.NumberColumn("Actual default (historical)"),
        "LIMIT_BAL": st.column_config.NumberColumn("Credit limit", format="localized"),
        "BILL_AMT1": st.column_config.NumberColumn("Bill Sep", format="localized"),
        "PAY_AMT1": st.column_config.NumberColumn("Paid Sep", format="localized")})

    export = view.drop(columns=["Age_Group", "Credit_Limit_Range"])
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download filtered predictions (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "customer_predictions_filtered.csv", "text/csv", width="stretch")
    if d2.button("💾 Save all predictions to outputs/ folder", width="stretch"):
        paths = export_predictions(df, bundle, medium_threshold=medium_t, high_threshold=high_t)
        st.success("Saved: " + ", ".join(str(p.relative_to(OUTPUTS_DIR.parent)) for p in paths))


def show_probability_result(proba: float, actual: int | None = None, split: str | None = None) -> None:
    segment = str(assign_segments([proba], medium_t, high_t)[0])
    st.markdown("**Estimated probability of default next month**")
    st.markdown(f'<div style="font-size:2.6rem;font-weight:700;color:{NAVY};line-height:1.1">{proba:.1%}</div>',
                unsafe_allow_html=True)
    st.progress(float(proba))
    st.markdown(f'<span class="seg"><span class="dot" style="background:{SEGMENT_COLORS[segment]}"></span>'
                f'{segment}</span> <span style="color:{MUTED};font-size:.85rem">(thresholds {medium_pct}% / '
                f'{high_pct}%)</span>', unsafe_allow_html=True)
    st.caption(f"Historical average default rate for comparison: {overall_rate:.1%}")
    if actual is not None:
        with st.container(border=True):
            st.markdown(f"**Recorded historical outcome:** {TARGET_LABELS[int(actual)]}  \n"
                        "<span style='font-size:.85rem;color:#5B6780'>Shown for comparison only. "
                        "It is never used as a model input.</span>", unsafe_allow_html=True)
            if split == "Train":
                st.caption("This customer was in the training data, so the model has already seen this outcome; "
                           "the estimate may look better than it would for a brand-new customer.")
            else:
                st.caption("This customer was in the held-out test set: the model never saw this outcome.")


def show_explanation(row: pd.DataFrame, proba: float) -> None:
    exp = explain_prediction(bundle, row)
    up = exp[exp["Contribution"] > 0]["Feature"].tolist()[:3]
    down = exp[exp["Contribution"] < 0]["Feature"].tolist()[:3]
    level = "higher" if proba > overall_rate else "lower"
    text = f"The estimate is **{level}** than the historical average ({overall_rate:.1%})."
    if up:
        text += f" Inputs pushing it **up**: {', '.join(up)}."
    if down:
        text += f" Inputs pulling it **down**: {', '.join(down)}."
    st.markdown(text)
    e = exp.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=e["Contribution"], y=[f"{f} = {v}" for f, v in zip(e["Feature"], e["Value"])], orientation="h",
        marker_color=["#e34948" if v > 0 else "#2a78d6" for v in e["Contribution"]], marker_cornerradius=4,
        hovertemplate="%{y}<br>Change in log-odds vs average customer: %{x:+.3f}<extra></extra>"))
    style(fig, "Main drivers of this estimate (vs an average customer)", "Effect on log-odds (red = raises risk)",
          "", legend=False, height=330)
    fig.update_yaxes(rangemode="normal")
    show(fig)
    st.caption("Contributions come from the model's coefficients: they describe how the *model* reacts to "
               "these inputs, not what causes default in real life.")


def customer_details(row: pd.Series) -> pd.DataFrame:
    items = [("Credit limit", fmt_money(row["LIMIT_BAL"])), ("Age", f"{row['AGE']:.0f}"),
             ("Sex", category_label("SEX", row["SEX"])), ("Education", category_label("EDUCATION", row["EDUCATION"])),
             ("Marital status", category_label("MARRIAGE", row["MARRIAGE"]))]
    items += [(f"Repayment status {PAY_STATUS_MONTH[c]}", pay_status_label(row[c])) for c in PAY_STATUS_COLS]
    items += [(f"Bill / paid {MONTHS[i]}", f"{fmt_money(row[b])} / {fmt_money(row[p])}")
              for i, (b, p) in enumerate(zip(BILL_COLS, PAY_AMT_COLS), 1)]
    return pd.DataFrame(items, columns=["Field", "Value"])


def page_individual() -> None:
    st.subheader("Individual risk assessment")
    note("The probability comes from the model's <code>predict_proba()</code>, which combines <b>all</b> inputs "
         "for this one customer. It is an <b>estimate, not a guarantee</b>, and must not be used on its own to "
         "make a real credit decision.")
    mode = st.radio("Choose a customer", ["Existing customer (by ID)", "Enter new customer details"], horizontal=True)

    if mode.startswith("Existing"):
        ids = df[ID_COL]
        cid = st.number_input("Customer ID", min_value=int(ids.min()), max_value=int(ids.max()),
                              value=int(ids.min()), step=1)
        match = pred[pred[ID_COL] == cid]
        if match.empty:
            st.warning(f"No customer with ID {cid} was found.")
            return
        row = match.iloc[[0]]
        r = row.iloc[0]
        l, rcol = st.columns([1, 1.2])
        with l:
            st.markdown(f"**Customer ID {cid}**")
            details = customer_details(r)
            st.dataframe(details, hide_index=True, width="stretch", height=(len(details) + 1) * 35 + 3)
        with rcol:
            proba = float(r["Predicted_Default_Probability"])
            show_probability_result(proba, actual=r["Actual_Default"], split=r["Data_Split"])
            show_explanation(row[FEATURE_COLS], proba)
        return

    med = df[FEATURE_COLS].median()
    with st.form("new_customer"):
        st.markdown("**Customer profile**")
        c = st.columns(5)
        values = {
            "LIMIT_BAL": c[0].number_input("Credit limit (NT$)", 1, 10_000_000, int(med["LIMIT_BAL"]), 10_000),
            "AGE": c[1].number_input("Age", 18, 100, int(med["AGE"]), 1),
            "SEX": c[2].selectbox("Sex", list(SEX_LABELS), format_func=SEX_LABELS.get),
            "EDUCATION": c[3].selectbox("Education", list(EDUCATION_LABELS), index=1, format_func=EDUCATION_LABELS.get),
            "MARRIAGE": c[4].selectbox("Marital status", list(MARRIAGE_LABELS), index=1,
                                       format_func=MARRIAGE_LABELS.get),
        }
        st.markdown("**Repayment status, bill and payment by month** (most recent first)")
        codes = list(range(-2, 9))
        for i, (s, b, p) in enumerate(zip(PAY_STATUS_COLS, BILL_COLS, PAY_AMT_COLS), 1):
            c = st.columns([1.4, 1, 1])
            values[s] = c[0].selectbox(f"Status {MONTHS[i]} ({s})", codes, index=codes.index(0),
                                       format_func=pay_status_label, key=s)
            values[b] = c[1].number_input(f"Bill {MONTHS[i]} (NT$)", -10_000_000, 10_000_000, int(med[b]), 1000, key=b)
            values[p] = c[2].number_input(f"Paid {MONTHS[i]} (NT$)", 0, 10_000_000, int(med[p]), 500, key=p)
        submitted = st.form_submit_button("Estimate default probability", type="primary")
    if not submitted:
        st.caption("Fields are pre-filled with median values from the dataset. Adjust them and press the button.")
        return
    errors = validate_customer_input(values)
    if errors:
        st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in errors))
        return
    row = pd.DataFrame([values])[FEATURE_COLS]
    try:
        proba = float(predict_probability(bundle, row)[0])
    except Exception as exc:
        st.error(f"The prediction failed: {exc}")
        return
    l, r = st.columns([1, 1.2])
    with l:
        show_probability_result(proba)
    with r:
        show_explanation(row, proba)


METRIC_HELP = {
    "Accuracy": "Share of all test customers classified correctly. Misleading on its own: predicting "
                "'no default' for everyone would already score {base:.1%}.",
    "Precision": "Of the customers the model flags as defaulters, the share that really defaulted. "
                 "Low precision = many false alarms.",
    "Recall": "Of the customers who really defaulted, the share the model caught. Low recall = missed "
              "defaulters, which is usually the costliest error in credit risk.",
    "F1 score": "A single score balancing precision and recall (their harmonic mean).",
    "ROC-AUC": "Probability that a random defaulter gets a higher score than a random non-defaulter. "
               "0.5 = random guessing, 1.0 = perfect ranking. Does not depend on the 50% cut-off.",
}


def page_performance() -> None:
    st.subheader("Model performance (held-out test set)")
    note(f"All numbers are calculated on the <b>{bundle['n_test']:,} test customers</b> (20%, stratified, "
         f"random seed {bundle['random_seed']}) that were never used for training or preprocessing. "
         f"A customer is classified as a defaulter when the predicted probability is ≥ "
         f"{bundle['decision_threshold']:.0%}.")
    m = metrics
    c = st.columns(5)
    values = {"Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
              "F1 score": m["f1"], "ROC-AUC": m["roc_auc"]}
    for col, (name, v) in zip(c, values.items()):
        kpi(col, name, f"{v:.3f}", "", True)
    with st.expander("What do these metrics mean?", expanded=True):
        for name, txt in METRIC_HELP.items():
            st.markdown(f"- **{name}** ({values[name]:.3f}): {txt.format(base=m['baseline_accuracy_always_no_default'])}")
        st.markdown(f"Only {m['test_actual_default_rate']:.1%} of test customers defaulted, so the classes are "
                    "**imbalanced** — that is why recall, precision and ROC-AUC matter more than accuracy here.")

    l, r = st.columns(2)
    with l:
        cm = m["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        labels = [["True negatives<br>(correctly no default)", "False positives<br>(false alarm)"],
                  ["False negatives<br>(missed defaulter)", "True positives<br>(caught defaulter)"]]
        text = [[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted: no default", "Predicted: default"],
                                   y=["Actual: no default", "Actual: default"], text=text, texttemplate="%{text}",
                                   colorscale=[[0, "#EEF3FB"], [1, "#2a78d6"]], showscale=False,
                                   hovertemplate="%{y} / %{x}<br>Customers: %{z:,}<extra></extra>"))
        style(fig, "Confusion matrix (test set)", legend=False)
        fig.update_yaxes(autorange="reversed", gridcolor=None)
        show(fig)
    with r:
        rep = pd.DataFrame(m["classification_report"]).T
        rep = rep.loc[["No default", "Default", "macro avg", "weighted avg"]].reset_index()
        rep.columns = ["Class", "Precision", "Recall", "F1 score", "Support"]
        rep["Support"] = rep["Support"].astype(int)
        st.markdown("**Classification report (test set)**")
        st.dataframe(rep.round(3), hide_index=True, width="stretch")
        st.caption("'Support' is the number of test customers in each actual class.")

    l, r = st.columns(2)
    with l:
        roc = m["roc_curve"]
        fig = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"Model (AUC {m['roc_auc']:.3f})",
                                    line=dict(color=PRED_COLOR, width=2),
                                    hovertemplate="False-positive rate %{x:.2f}<br>Recall %{y:.2f}<extra></extra>"),
                         go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                                    line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip")])
        style(fig, "ROC curve", "False-positive rate", "True-positive rate (recall)")
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.02])
        show(fig)
    with r:
        pr = m["pr_curve"]
        base = m["test_actual_default_rate"]
        fig = go.Figure([go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                    name=f"Model (AP {m['average_precision']:.3f})", line=dict(color=PRED_COLOR, width=2),
                                    hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"),
                         go.Scatter(x=[0, 1], y=[base, base], mode="lines", name=f"Random guessing ({base:.1%})",
                                    line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip")])
        style(fig, "Precision-recall curve", "Recall", "Precision")
        fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.02])
        show(fig)

    st.markdown("#### Explore the decision threshold")
    tr = bundle["test_results"]
    t = st.slider("Classify as 'default' when predicted probability ≥", 0.05, 0.95, 0.50, 0.05, format="%.2f")
    mt = evaluate(tr["Actual_Default"].values, tr["Predicted_Default_Probability"].values, threshold=t)
    c = st.columns(4)
    for col, (name, key) in zip(c, [("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"),
                                    ("Accuracy", "accuracy")]):
        delta = None if t == bundle["decision_threshold"] else f"{mt[key] - m[key]:+.3f} vs 0.50"
        col.metric(name, f"{mt[key]:.3f}", delta, border=True)
    st.caption(f"At {t:.2f}: {mt['confusion_matrix']['tp']:,} defaulters caught, {mt['confusion_matrix']['fn']:,} "
               f"missed, {mt['confusion_matrix']['fp']:,} false alarms. Lowering the threshold raises recall but "
               "creates more false alarms. Choosing it is a business decision about which error costs more.")

    st.markdown("#### Probability calibration")
    l, r = st.columns([1.2, 1])
    with l:
        cal = m["calibration"]
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                    line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"),
                         go.Scatter(x=cal["mean_predicted"], y=cal["observed_default_rate"], mode="lines+markers",
                                    name="Model", line=dict(color=PRED_COLOR, width=2), marker=dict(size=8),
                                    hovertemplate="Avg predicted %{x:.1%}<br>Actual default rate %{y:.1%}<extra></extra>")])
        style(fig, "Calibration: predicted vs actual default rate (10 groups of test customers)",
              "Average predicted probability", "Actual default rate", height=400)
        fig.update_xaxes(tickformat=".0%", range=[0, 1]); fig.update_yaxes(tickformat=".0%", range=[0, 1])
        show(fig)
    with r:
        gap = m["test_mean_predicted_probability"] - m["test_actual_default_rate"]
        st.markdown(
            f"- Average predicted probability: **{m['test_mean_predicted_probability']:.1%}**\n"
            f"- Actual test default rate: **{m['test_actual_default_rate']:.1%}**\n"
            f"- Brier score: **{m['brier_score']:.3f}** (mean squared error of the probabilities; lower is better)\n\n"
            "If the points lie close to the dashed line, a predicted 40% really does mean roughly 40 in 100 such "
            "customers defaulted. ")
        if abs(gap) > 0.05:
            st.warning(f"Predicted probabilities are on average {gap:+.1%} away from the actual rate. "
                       + ("This is expected with class_weight='balanced', which deliberately inflates the "
                          "default class; treat the numbers as risk *scores* rather than literal probabilities."
                          if bundle["class_weight"] == "balanced" else "Interpret the probabilities with care."))
        else:
            st.success("On average the probabilities match the actual default rate. Check the curve for "
                       "groups where the model over- or under-estimates risk.")
        st.caption("Class imbalance: the standard model keeps probabilities calibrated. Retraining with "
                   "class_weight='balanced' (sidebar) increases recall at the 50% cut-off but makes "
                   "probabilities systematically too high.")

    st.markdown("#### What influences the model?")
    l, r = st.columns(2)
    with l:
        coef = bundle["coefficients"].copy()
        coef = coef.reindex(coef["Coefficient"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=coef["Coefficient"], y=coef["Feature"], orientation="h", marker_cornerradius=4,
                               marker_color=["#e34948" if v > 0 else "#2a78d6" for v in coef["Coefficient"]],
                               customdata=coef["Odds_Ratio"],
                               hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+.3f}<br>Odds ratio: "
                                             "%{customdata:.2f}<extra></extra>"))
        style(fig, "Largest logistic-regression coefficients", "Coefficient (red = higher risk)", "",
              legend=False, height=520)
        fig.update_yaxes(rangemode="normal")
        show(fig)
    with r:
        imp = bundle["permutation_importance"].head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["AUC_Drop_Mean"], y=imp["Feature"], orientation="h", marker_color=PRED_COLOR,
                               marker_cornerradius=4,
                               error_x=dict(type="data", array=imp["AUC_Drop_Std"], color=MUTED),
                               hovertemplate="<b>%{y}</b><br>ROC-AUC drop: %{x:.4f}<extra></extra>"))
        style(fig, "Permutation importance (test set)", "Drop in ROC-AUC when shuffled", "", legend=False, height=520)
        fig.update_yaxes(rangemode="normal")
        show(fig)
    st.markdown(
        "- **Coefficients**: numeric inputs are standardised, so a coefficient is the change in log-odds for a "
        "one-standard-deviation increase. Categories (e.g. `PAY_0 = 2 months late`) are compared with the "
        "model's other categories of the same variable. The odds ratio = e^coefficient.\n"
        "- **Limitations**: correlated inputs (e.g. the six bill amounts) share and can swap their weight, so "
        "individual coefficients can be unstable or have surprising signs. Rare categories can get large "
        "coefficients from very few customers.\n"
        "- **Permutation importance** shuffles one original column in the test data and measures how much the "
        "ranking quality (ROC-AUC) drops. It shows reliance, not causation.\n"
        "- **Fairness**: SEX, AGE, EDUCATION and MARRIAGE are demographic attributes. Using them (or proxies "
        "for them) in real lending can be unlawful or discriminatory. Any real-world system would need legal "
        "review, fairness testing across groups and possibly removal of these inputs.")

    d = st.columns(2)
    d[0].download_button("⬇ Model metrics (JSON)",
                         METRICS_PATH.read_bytes() if METRICS_PATH.exists() else json.dumps(m, default=float),
                         "model_metrics.json", "application/json", width="stretch")
    test = pred[pred["Data_Split"] == "Test"].drop(columns=["Age_Group", "Credit_Limit_Range"]).copy()
    test["Correct_Prediction"] = (test["Predicted_Default"] == test["Actual_Default"]).astype(int)
    d[1].download_button("⬇ Actual vs predicted, test customers (CSV)", test.to_csv(index=False).encode("utf-8"),
                         "test_predictions.csv", "text/csv", width="stretch")


def page_quality() -> None:
    st.subheader("Data quality report")
    q = quality
    s = q["summary"]
    c = st.columns(6)
    cards = [("Rows used", f"{s['Rows used (valid target)']:,}", f"read {s['Rows read']:,}"),
             ("Columns", f"{s['Columns used']}", "incl. ID and target"),
             ("Missing values", f"{s['Total missing values']:,}", ""),
             ("Duplicate IDs", f"{s['Duplicate customer IDs']:,}", ""),
             ("Duplicate rows", f"{s['Fully duplicated rows']:,}", f"{s['Rows identical except ID']} identical except ID"),
             ("Header row", f"Excel row {s['Real header found on Excel row']}", "detected automatically")]
    for col, (label, value, sub) in zip(c, cards):
        kpi(col, label, value, sub, show_tag=False)
    note(f"File: <b>{s['File']}</b> · target column found as <b>'{s['Original target column name']}'</b> "
         f"(0 = no default, 1 = default; verified to contain only 0/1). The original file is only read, never "
         f"modified. Rows identical except their ID are kept: different customers can share the same values.")

    l, r = st.columns([1.3, 1])
    with l:
        st.markdown("**Columns, types and missing values**")
        st.dataframe(q["columns"], hide_index=True, width="stretch", height=420)
    with r:
        st.markdown("**Target distribution**")
        st.dataframe(q["target"], hide_index=True, width="stretch")
        st.markdown("**Unusual numeric values**")
        st.dataframe(q["unusual_values"], hide_index=True, width="stretch")
    st.markdown("**Unexpected or undocumented category codes** (kept, not deleted)")
    st.dataframe(q["unexpected_codes"], hide_index=True, width="stretch")
    st.markdown("**Descriptive statistics (numeric variables)**")
    st.dataframe(q["describe"], width="stretch")

    path = export_quality_report(q)
    st.download_button("⬇ Data-quality summary (CSV)", path.read_bytes(), "data_quality_summary.csv", "text/csv")


{"Overview": page_overview, "Data Exploration": page_exploration, "Risk Segmentation": page_segmentation,
 "Individual Risk Assessment": page_individual, "Model Performance": page_performance,
 "Data Quality": page_quality}[page]()
