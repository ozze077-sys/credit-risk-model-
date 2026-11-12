"""
charts.py
=========
Plotly figure builders used by the dashboards (server.py, dash_app.py).

Each function takes data and returns a ``plotly.graph_objects.Figure``.
Keeping the charts here (and not inside the page code) means the page layout
stays short and readable.

Wording is written for non-technical readers: "missed payment" instead of
"default", "chance" instead of "probability", and column codes such as PAY_0
are shown with plain names ("Payment status – Sep") via ``friendly_name``.

Colour convention
-----------------
* Blue / orange  = what ACTUALLY happened (paid on time / missed payment).
* Violet         = the model's ESTIMATES.
* Green / amber / red = estimated risk groups (always with a text label).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import MONTHS, TARGET_COL, TARGET_LABELS, friendly_name

# Palette of a modern banking dashboard: deep blue, bright blue, teal, pink, yellow, coral.
NAVY = "#343C6A"
INK = "#232323"
MUTED = "#718EBF"
GRID = "#EEF2F8"
OUTCOME_COLORS = dict(zip(TARGET_LABELS.values(), ["#16DBCC", "#FF82AC"]))  # paid = teal, missed = pink
HIST_COLOR = "#2D60FF"      # what happened (bright blue bars)
PRED_COLOR = "#1814F3"      # model estimates (deep blue)
UP_COLOR, DOWN_COLOR = "#FE5C73", "#2D60FF"  # raises / lowers risk
SEGMENT_COLORS = {"Low risk": "#16DBCC", "Medium risk": "#FFBB38", "High risk": "#FE5C73"}
DIVERGING = [[0, "#2D60FF"], [0.5, "#F5F7FA"], [1, "#FE5C73"]]


def style(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", pct_y: bool = False,
          height: int = 380, legend: bool = True, subtitle: str | None = None) -> go.Figure:
    """Apply one consistent style to every chart."""
    title = f"<b>{title}</b>"
    if subtitle:
        title += f"<br><span style='font-size:12px;color:{MUTED};font-weight:normal'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=78 if subtitle else 56, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK, size=12,
                                                               family="Inter, Segoe UI, sans-serif"),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12), bargap=0.4,
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor="#C9D1DD")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=True, zerolinecolor="#C9D1DD", rangemode="tozero")
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def _horizontal_bars(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(rangemode="normal", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# What actually happened (historical charts)
# ---------------------------------------------------------------------------
def rate_bar(table: pd.DataFrame, title: str, x_title: str, overall: float | None = None) -> go.Figure:
    """Share of customers in each group who missed a payment. Long labels -> horizontal bars."""
    horizontal = table["Group"].astype(str).str.len().max() > 16
    top = min(1, table["Default_Rate"].max() * 1.2 + 0.02)
    subtitle = f"Dotted line = average of all customers ({overall:.1%})" if overall is not None else None
    common = dict(marker_color=HIST_COLOR, marker_cornerradius=4, customdata=table[["Customers", "Defaults"]].values,
                  text=[f"{v:.0%}" for v in table["Default_Rate"]], textposition="outside", cliponaxis=False)
    hover = ("%{VAL:.1%} missed a payment<br>%{customdata[1]:,} of %{customdata[0]:,} customers"
             "<extra></extra>")
    if horizontal:
        fig = go.Figure(go.Bar(y=table["Group"].astype(str), x=table["Default_Rate"], orientation="h", **common,
                               hovertemplate="<b>%{y}</b><br>" + hover.replace("VAL", "x")))
        style(fig, title, "Share who missed a payment", "", legend=False, subtitle=subtitle,
              height=max(380, 34 * len(table) + 120))
        _horizontal_bars(fig)
        fig.update_xaxes(tickformat=".0%", range=[0, top])
        fig.update_yaxes(autorange="reversed")
        if overall is not None:
            fig.add_vline(x=overall, line_dash="dot", line_color=MUTED)
        return fig
    fig = go.Figure(go.Bar(x=table["Group"].astype(str), y=table["Default_Rate"], **common,
                           hovertemplate="<b>%{x}</b><br>" + hover.replace("VAL", "y")))
    style(fig, title, x_title, "Share who missed a payment", pct_y=True, legend=False, subtitle=subtitle)
    fig.update_yaxes(range=[0, top])
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dot", line_color=MUTED)
    return fig


def outcome_bar(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COL].map(TARGET_LABELS).value_counts().reindex(TARGET_LABELS.values(), fill_value=0)
    share = counts / counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=[OUTCOME_COLORS[k] for k in counts.index],
        marker_cornerradius=4, text=[f"{n:,} ({s:.0%})" for n, s in zip(counts, share)], textposition="outside",
        cliponaxis=False, customdata=share.values,
        hovertemplate="<b>%{x}</b><br>%{y:,} customers (%{customdata:.1%})<extra></extra>"))
    return style(fig, "What actually happened the following month", "", "Number of customers", legend=False)


def limit_by_outcome(df: pd.DataFrame) -> go.Figure:
    stats = (df.groupby(df[TARGET_COL].map(TARGET_LABELS))["LIMIT_BAL"].agg(["mean", "median"])
               .reindex(TARGET_LABELS.values()))
    fig = go.Figure()
    for outcome, row in stats.iterrows():
        fig.add_bar(name=outcome, x=["Average", "Typical (median)"], y=[row["mean"], row["median"]],
                    marker_color=OUTCOME_COLORS[outcome], marker_cornerradius=4,
                    text=[f"{v / 1000:,.0f}K" for v in (row["mean"], row["median"])], textposition="outside",
                    cliponaxis=False, hovertemplate=f"{outcome}<br>%{{x}}: NT$ %{{y:,.0f}}<extra></extra>")
    fig.update_layout(barmode="group")
    return style(fig, "Credit limit: customers who paid vs customers who missed", "", "Credit limit (NT$)")


def limit_box(data: pd.DataFrame) -> go.Figure:
    fig = px.box(data, x="Outcome", y="LIMIT_BAL", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                 category_orders={"Outcome": list(TARGET_LABELS.values())})
    return style(fig, "Range of credit limits in each group", "", "Credit limit (NT$)", legend=False,
                 subtitle="Box = middle half of customers; line inside = typical value")


def bill_payment_bars(cmp: pd.DataFrame, prefix: str, label: str, stat: str) -> go.Figure:
    """Average/typical bill or payment per month, for customers who paid vs missed."""
    part = cmp[cmp["Variable"].str.startswith(prefix)].copy()
    part["Month"] = part["Variable"].str[-1].astype(int).map(MONTHS)
    fig = px.bar(part, x="Month", y=stat, color="Outcome", barmode="group", color_discrete_map=OUTCOME_COLORS,
                 category_orders={"Month": list(MONTHS.values())[::-1], "Outcome": list(TARGET_LABELS.values())})
    word = "Average" if stat == "Mean" else "Typical"
    fig.update_traces(marker_cornerradius=4,
                      hovertemplate=f"%{{fullData.name}} · %{{x}}<br>{word}: NT$ %{{y:,.0f}}<extra></extra>")
    return style(fig, f"{word} {label.lower()} each month", "Month", f"{label} (NT$)")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    corr = corr.rename(index=friendly_name, columns=friendly_name)
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING, text_auto=".2f", aspect="auto")
    fig.update_traces(hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>", textfont_size=9)
    style(fig, "Which numbers move together (correlation)", height=680, legend=False,
          subtitle="Red = rise together, blue = one rises as the other falls, white = no link")
    fig.update_yaxes(rangemode="normal", showgrid=False)
    return fig


def distribution_hist(data: pd.DataFrame, col: str, label: str) -> tuple[go.Figure, int, float, float]:
    """Histogram by outcome, each group scaled to 100%. Extreme money values are clipped for readability."""
    if col in ("AGE", "LIMIT_BAL"):
        lo, hi = data[col].min(), data[col].max()
    else:
        lo, hi = data[col].quantile([0.005, 0.99])
    shown = data[(data[col] >= lo) & (data[col] <= hi)]
    fig = px.histogram(shown, x=col, color="Outcome", barmode="overlay", histnorm="percent", nbins=60,
                       opacity=0.6, color_discrete_map=OUTCOME_COLORS,
                       category_orders={"Outcome": list(TARGET_LABELS.values())})
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}<br>%{y:.1f}% of this group<extra></extra>")
    style(fig, f"How {label} is spread out", label, "% of customers in the group")
    return fig, len(data) - len(shown), lo, hi


# ---------------------------------------------------------------------------
# Model estimates
# ---------------------------------------------------------------------------
def segment_bar(seg: pd.DataFrame, medium: float, high: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=seg["Risk_Segment"], y=seg["Customers"], marker_color=[SEGMENT_COLORS[s] for s in seg["Risk_Segment"]],
        marker_cornerradius=4, text=[f"{n:,} ({s:.0%})" for n, s in zip(seg["Customers"], seg["Share"])],
        textposition="outside", cliponaxis=False, customdata=seg[["Share", "Avg_Probability"]].values,
        hovertemplate="<b>%{x}</b><br>%{y:,} customers (%{customdata[0]:.1%})"
                      "<br>Average estimated chance: %{customdata[1]:.0%}<extra></extra>"))
    return style(fig, "Customers in each risk group", "", "Number of customers", legend=False,
                 subtitle=f"Low below {medium:.0%} · Medium {medium:.0%}–{high:.0%} · High {high:.0%} and above")


def probability_histogram(proba: pd.Series, medium: float, high: float,
                          title: str = "How risky are the customers?") -> go.Figure:
    fig = go.Figure(go.Histogram(x=proba, xbins=dict(start=0, end=1, size=0.025), marker_color=PRED_COLOR,
                                 marker_line=dict(color="white", width=1),
                                 hovertemplate="Estimated chance around %{x}<br>%{y:,} customers<extra></extra>"))
    style(fig, title, "Estimated chance of missing next month's payment", "Number of customers", legend=False,
          subtitle="Most customers are on the left (low chance); the long tail on the right is the risky group")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    for t, label, pos in [(medium, "Medium", "top left"), (high, "High", "top right")]:
        fig.add_vline(x=t, line_dash="dash", line_color=NAVY, annotation_text=f"{label} from {t:.0%}",
                      annotation_position=pos, annotation_font_color=NAVY)
    return fig


def importance_bar(imp: pd.DataFrame, title: str, height: int = 420) -> go.Figure:
    imp = imp.iloc[::-1]
    names = [friendly_name(f) for f in imp["Feature"]]
    fig = go.Figure(go.Bar(x=imp["AUC_Drop_Mean"], y=names, orientation="h", marker_color=PRED_COLOR,
                           marker_cornerradius=4,
                           hovertemplate="<b>%{y}</b><br>Importance score: %{x:.3f}<extra></extra>"))
    style(fig, title, "How much the model relies on it (longer bar = more important)", "", legend=False,
          height=height)
    fig.update_xaxes(showticklabels=False)
    return _horizontal_bars(fig)


def coefficient_bar(coef: pd.DataFrame, top_n: int = 15) -> go.Figure:
    coef = coef.reindex(coef["Coefficient"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=coef["Coefficient"], y=[friendly_name(f) for f in coef["Feature"]], orientation="h",
                           marker_cornerradius=4,
                           marker_color=[UP_COLOR if v > 0 else DOWN_COLOR for v in coef["Coefficient"]],
                           customdata=coef["Odds_Ratio"],
                           hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+.3f}<br>Odds ratio: %{customdata:.2f}<extra></extra>"))
    style(fig, "Model weights (technical)", "← lowers risk · raises risk →", "", legend=False, height=520)
    return _horizontal_bars(fig)


def contribution_bar(exp: pd.DataFrame) -> go.Figure:
    e = exp.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=e["Contribution"], y=[f"{friendly_name(f)}: {v}" for f, v in zip(e["Feature"], e["Value"])],
        orientation="h", marker_color=[UP_COLOR if v > 0 else DOWN_COLOR for v in e["Contribution"]],
        marker_cornerradius=4,
        customdata=["Raises the risk" if v > 0 else "Lowers the risk" for v in e["Contribution"]],
        hovertemplate="%{y}<br>%{customdata} (strength %{x:+.2f})<extra></extra>"))
    style(fig, "Why this estimate? (compared with a typical customer)", "← lowers risk · raises risk →",
          "", legend=False, height=340)
    fig.update_xaxes(showticklabels=False, zeroline=True, zerolinecolor=MUTED, zerolinewidth=1)
    return _horizontal_bars(fig)


# ---------------------------------------------------------------------------
# Reliability charts
# ---------------------------------------------------------------------------
def confusion_heatmap(cm: dict) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    labels = [["Correctly said 'will pay'", "False alarm"], ["Missed (no warning)", "Correctly warned"]]
    text = [[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=["Model said: will pay", "Model warned: will miss"],
                               y=["Actually paid", "Actually missed"], text=text, texttemplate="%{text}",
                               colorscale=[[0, "#EDF1FF"], [1, "#2D60FF"]], showscale=False,
                               hovertemplate="%{y} / %{x}<br>%{z:,} customers<extra></extra>"))
    style(fig, "Right and wrong calls on the test customers", legend=False,
          subtitle="Rows = what really happened · columns = what the model said")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def _diagonal_chart(x, y, name: str, base_x, base_y, base_name: str, title: str, x_title: str, y_title: str,
                    hover: str) -> go.Figure:
    fig = go.Figure([go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=PRED_COLOR, width=3),
                                fill="tozeroy", fillcolor="rgba(45,96,255,0.10)", hovertemplate=hover),
                     go.Scatter(x=base_x, y=base_y, mode="lines", name=base_name,
                                line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip")])
    style(fig, title, x_title, y_title)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def roc_chart(m: dict) -> go.Figure:
    roc = m["roc_curve"]
    return _diagonal_chart(roc["fpr"], roc["tpr"], f"Model (AUC {m['roc_auc']:.3f})", [0, 1], [0, 1],
                           "Random guessing", "ROC curve (technical)", "False-positive rate",
                           "True-positive rate (recall)",
                           "False-positive rate %{x:.2f}<br>Recall %{y:.2f}<extra></extra>")


def pr_chart(m: dict) -> go.Figure:
    pr, base = m["pr_curve"], m["test_actual_default_rate"]
    return _diagonal_chart(pr["recall"], pr["precision"], f"Model (AP {m['average_precision']:.3f})", [0, 1],
                           [base, base], f"Random guessing ({base:.1%})", "Precision-recall curve (technical)",
                           "Recall", "Precision", "Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>")


def calibration_chart(m: dict) -> go.Figure:
    cal = m["calibration"]
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly realistic",
                                line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"),
                     go.Scatter(x=cal["mean_predicted"], y=cal["observed_default_rate"], mode="lines+markers",
                                name="This model", line=dict(color=PRED_COLOR, width=3, shape="spline"), marker=dict(size=9, color="white", line=dict(color=PRED_COLOR, width=3)),
                                fill="tozeroy", fillcolor="rgba(45,96,255,0.10)",
                                hovertemplate="Model said about %{x:.0%}<br>Actually missed: %{y:.0%}<extra></extra>")])
    style(fig, "Estimated chance vs what really happened", "What the model estimated", "What actually happened",
          height=420, subtitle="Each dot is a group of customers; dots on the dashed line = estimates came true")
    fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
                      margin=dict(b=60))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig
