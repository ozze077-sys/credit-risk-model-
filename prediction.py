"""
prediction.py
=============
Uses a trained model bundle to:
* estimate each customer's probability of default (``predict_proba``),
* assign configurable risk segments,
* explain an individual prediction,
* validate manually entered customer details,
* export results to CSV.

Important: an individual probability comes from the *multivariable model*
(all features together). It is not the same as a historical group default rate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import (BILL_COLS, CATEGORICAL_FEATURES, FEATURE_COLS, ID_COL, PAY_AMT_COLS,
                         PAY_STATUS_COLS, SOURCE_COL, TARGET_COL, category_label)
from model_training import MODEL_PATH, OUTPUTS_DIR

DEFAULT_MEDIUM_THRESHOLD = 0.30
DEFAULT_HIGH_THRESHOLD = 0.50
SEGMENTS = ["Low risk", "Medium risk", "High risk"]

REQUIRED_BUNDLE_KEYS = {"pipeline", "feature_cols", "metrics", "test_ids", "transformed_sources"}


class ModelError(Exception):
    """Raised with a beginner-friendly message when the model can't be used."""


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the saved model bundle, checking that it is complete."""
    if not Path(path).exists():
        raise ModelError(f"No trained model found at '{path}'. Train the model first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelError(f"The model file could not be loaded (it may be corrupted or from another "
                         f"library version): {exc}. Please retrain the model.") from exc
    if not isinstance(bundle, dict) or not REQUIRED_BUNDLE_KEYS <= set(bundle):
        raise ModelError("The model file is incomplete or in an old format. Please retrain the model.")
    return bundle


def model_matches_data(bundle: dict, dataset_info: dict) -> bool:
    """True if the model was trained on exactly this dataset file (same content hash)."""
    return bundle.get("dataset_fingerprint") == dataset_info.get("fingerprint")


def predict_probability(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Probability of default (class 1) for each row of ``X``.

    Only the model's feature columns are passed in - never the ID or the target.
    """
    missing = [c for c in bundle["feature_cols"] if c not in X.columns]
    if missing:
        raise ModelError(f"Input is missing required columns: {', '.join(missing)}")
    return bundle["pipeline"].predict_proba(X[bundle["feature_cols"]])[:, 1]


def assign_segments(proba, medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
                    high_threshold: float = DEFAULT_HIGH_THRESHOLD) -> np.ndarray:
    """Low: p < medium; Medium: medium <= p < high; High: p >= high."""
    if not 0 < medium_threshold < high_threshold < 1:
        raise ValueError("Thresholds must satisfy 0 < medium < high < 1.")
    proba = np.asarray(proba)
    return np.select([proba >= high_threshold, proba >= medium_threshold], ["High risk", "Medium risk"],
                     default="Low risk")


def build_prediction_table(df: pd.DataFrame, bundle: dict,
                           medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
                           high_threshold: float = DEFAULT_HIGH_THRESHOLD) -> pd.DataFrame:
    """One row per customer: original inputs, actual outcome and model outputs.

    ``Data_Split`` shows whether the row was in the training set (the model has
    seen its outcome, so its prediction is optimistic) or the held-out test set.
    """
    proba = predict_probability(bundle, df)
    threshold = bundle.get("decision_threshold", 0.5)
    test_ids = set(bundle["test_ids"])
    out = df[[ID_COL] + ([SOURCE_COL] if SOURCE_COL in df else []) + FEATURE_COLS].copy()
    out["Data_Split"] = np.where(df[ID_COL].isin(test_ids), "Test", "Train")
    out["Actual_Default"] = df[TARGET_COL].values
    out["Predicted_Default_Probability"] = proba  # full precision; rounded only when exported
    out["Predicted_Default"] = (proba >= threshold).astype(int)
    out["Risk_Segment"] = assign_segments(proba, medium_threshold, high_threshold)
    return out


def segment_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """Customers, share and average probability per risk segment (all segments shown, even if empty)."""
    g = pred.groupby("Risk_Segment")["Predicted_Default_Probability"].agg(Customers="size", Avg_Probability="mean")
    g = g.reindex(SEGMENTS).fillna({"Customers": 0})
    g["Customers"] = g["Customers"].astype(int)
    g["Share"] = g["Customers"] / max(g["Customers"].sum(), 1)
    g["Actual_Default_Rate"] = pred.groupby("Risk_Segment")["Actual_Default"].mean().reindex(SEGMENTS)
    return g.reset_index().rename(columns={"index": "Risk_Segment"})


def explain_prediction(bundle: dict, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """How each original feature pushes this customer's log-odds up or down.

    For logistic regression:  log-odds = intercept + sum(coefficient x feature value).
    We compare the customer with the average training customer, so each
    contribution = coefficient x (customer's value - average value), summed over
    the one-hot columns that belong to the same original feature.
    Positive -> raises estimated risk, negative -> lowers it. This describes the
    model's arithmetic, not real-world causes.
    """
    pipe = bundle["pipeline"]
    transformed = np.asarray(pipe.named_steps["preprocess"].transform(row[bundle["feature_cols"]]))[0]
    coefs = pipe.named_steps["model"].coef_[0]
    contrib = coefs * (transformed - bundle["train_transformed_mean"])
    per_feature = (pd.DataFrame({"Feature": bundle["transformed_sources"], "Contribution": contrib})
                   .groupby("Feature", sort=False)["Contribution"].sum())
    out = per_feature.reindex(per_feature.abs().sort_values(ascending=False).index).head(top_n).reset_index()
    out["Value"] = [_display_value(f, row.iloc[0][f]) for f in out["Feature"]]
    out["Direction"] = np.where(out["Contribution"] > 0, "Raises risk", "Lowers risk")
    return out


def _display_value(feature: str, value) -> str:
    if feature in CATEGORICAL_FEATURES:
        return category_label(feature, value)
    if feature in ("LIMIT_BAL", *BILL_COLS, *PAY_AMT_COLS):
        return f"NT$ {value:,.0f}"
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Manual input validation
# ---------------------------------------------------------------------------
INPUT_RULES = {
    "LIMIT_BAL": (1, 10_000_000),
    "AGE": (18, 100),
    **{c: (-10_000_000, 10_000_000) for c in BILL_COLS},
    **{c: (0, 10_000_000) for c in PAY_AMT_COLS},
}
ALLOWED_CODES = {"SEX": {1, 2}, "EDUCATION": {1, 2, 3, 4}, "MARRIAGE": {1, 2, 3},
                 **{c: set(range(-2, 10)) for c in PAY_STATUS_COLS}}


def validate_customer_input(values: dict) -> list[str]:
    """Return a list of error messages (empty list = input is valid)."""
    errors = []
    for col in FEATURE_COLS:
        v = values.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{col} is required.")
            continue
        if col in INPUT_RULES:
            lo, hi = INPUT_RULES[col]
            if not lo <= v <= hi:
                errors.append(f"{col} must be between {lo:,} and {hi:,} (got {v:,}).")
        elif col in ALLOWED_CODES and int(v) not in ALLOWED_CODES[col]:
            errors.append(f"{col} has an invalid code {v}.")
    return errors


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def export_predictions(df: pd.DataFrame, bundle: dict, outputs_dir: Path = OUTPUTS_DIR,
                       medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
                       high_threshold: float = DEFAULT_HIGH_THRESHOLD) -> list[Path]:
    """Write customer_predictions.csv and test_predictions.csv to outputs/."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pred = build_prediction_table(df, bundle, medium_threshold, high_threshold)
    pred["Predicted_Default_Probability"] = pred["Predicted_Default_Probability"].round(4)
    all_path = outputs_dir / "customer_predictions.csv"
    test_path = outputs_dir / "test_predictions.csv"
    pred.to_csv(all_path, index=False)
    test = pred[pred["Data_Split"] == "Test"].copy()
    test["Correct_Prediction"] = (test["Predicted_Default"] == test["Actual_Default"]).astype(int)
    test.to_csv(test_path, index=False)
    return [all_path, test_path]
