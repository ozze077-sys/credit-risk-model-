"""
data_analysis.py
================
Helper functions for exploratory data analysis (EDA).

Everything here uses the *actual historical outcomes* (the target column).
A "default rate" for a group is simply:

    number of customers in the group who defaulted / number of customers in the group

This describes the past. It is NOT a prediction for an individual customer,
and a relationship between a variable and default is a correlation, not proof
of cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import (BILL_COLS, PAY_AMT_COLS, NUMERIC_FEATURES, TARGET_COL, TARGET_LABELS,
                         category_label)

# Bins used to group continuous variables. Edges are inclusive on the left.
AGE_BINS = [0, 25, 30, 35, 40, 45, 50, 60, 200]
AGE_LABELS = ["Under 25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-59", "60+"]

LIMIT_BINS = [0, 50_000, 100_000, 200_000, 300_000, 500_000, np.inf]
LIMIT_LABELS = ["Under 50K", "50K-99K", "100K-199K", "200K-299K", "300K-499K", "500K+"]


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def limit_range(limit: pd.Series) -> pd.Series:
    return pd.cut(limit, bins=LIMIT_BINS, labels=LIMIT_LABELS, right=False)


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable grouping columns used by charts and filters (does not modify the input)."""
    out = df.copy()
    out["Age_Group"] = age_group(out["AGE"])
    out["Credit_Limit_Range"] = limit_range(out["LIMIT_BAL"])
    if TARGET_COL in out:
        out["Outcome"] = out[TARGET_COL].map(TARGET_LABELS)
    return out


def default_rate_by(df: pd.DataFrame, column: str, use_labels: bool = True) -> pd.DataFrame:
    """Historical default rate for each category of ``column``.

    Returns columns: Group, Customers, Defaults, Default_Rate (0-1), Share_of_Customers (0-1).
    Missing values get their own "Missing" group instead of disappearing.
    """
    key = df[column]
    if isinstance(key.dtype, pd.CategoricalDtype):
        key = key.cat.add_categories("Missing").fillna("Missing")
    else:
        key = key.fillna(-999)  # placeholder so NaN rows are counted
    grouped = (df.assign(_key=key)
                 .groupby("_key", observed=True)[TARGET_COL]
                 .agg(Customers="size", Defaults="sum")
                 .reset_index())
    grouped = grouped[grouped["Customers"] > 0]
    if use_labels and not isinstance(df[column].dtype, pd.CategoricalDtype):
        grouped["Group"] = [("Missing" if k == -999 else category_label(column, k)) for k in grouped["_key"]]
    else:
        grouped["Group"] = grouped["_key"].astype(str)
    grouped["Default_Rate"] = grouped["Defaults"] / grouped["Customers"]
    grouped["Share_of_Customers"] = grouped["Customers"] / grouped["Customers"].sum()
    return grouped[["Group", "Customers", "Defaults", "Default_Rate", "Share_of_Customers"]]


def credit_limit_stats(df: pd.DataFrame) -> dict:
    s = df["LIMIT_BAL"]
    return {"mean": s.mean(), "median": s.median(), "min": s.min(), "max": s.max()}


def outcome_comparison(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean and median of each column for defaulting vs non-defaulting customers (long format)."""
    rows = []
    for outcome_code, label in TARGET_LABELS.items():
        part = df[df[TARGET_COL] == outcome_code]
        for col in columns:
            rows.append({"Outcome": label, "Variable": col,
                         "Mean": part[col].mean(), "Median": part[col].median()})
    return pd.DataFrame(rows)


def bill_payment_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return outcome_comparison(df, BILL_COLS + PAY_AMT_COLS)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between numeric variables, repayment status codes and the target.

    Note: correlation only measures *linear* association. PAY_x codes are
    included as numbers here purely for a quick overview.
    """
    cols = NUMERIC_FEATURES + ["PAY_0", "PAY_2", "PAY_3", TARGET_COL]
    return df[cols].corr().round(2)
