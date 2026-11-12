"""
model_training.py
=================
Trains and evaluates a logistic-regression model that estimates the probability
that a customer defaults on next month's payment.

Run from a terminal to (re)train the model and refresh the files in outputs/:

    python model_training.py               # standard (unweighted) logistic regression
    python model_training.py --balanced    # use class_weight="balanced"

How the pieces fit together
---------------------------
1. Split the data into an 80% TRAINING set and a 20% TEST set (stratified,
   so both sets keep the ~22% default share).
2. Build a scikit-learn *Pipeline*: preprocessing (impute -> scale / one-hot
   encode) followed by LogisticRegression. Fitting the pipeline on the training
   set means all preprocessing statistics (medians, means, categories) are
   learned from training data only - the test set stays "unseen".
3. Evaluate on the test set: accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix, ROC / precision-recall curves, calibration.
4. Save the fitted pipeline + metadata to models/credit_risk_model.joblib and
   the metrics to outputs/model_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import (CATEGORICAL_FEATURES, FEATURE_COLS, ID_COL, NUMERIC_FEATURES, PROJECT_DIR,
                         SOURCE_COL, TARGET_COL, category_label, data_quality_report, export_quality_report,
                         load_dataset)

RANDOM_SEED = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50  # probability above which we call a prediction "default"

MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "credit_risk_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"


def build_pipeline(class_weight: str | None = None) -> Pipeline:
    """Preprocessing + logistic regression in one object.

    * Numeric columns: fill missing values with the training MEDIAN, then
      standardise (mean 0, std 1) so features in NT$ and in years are comparable.
    * Categorical columns: fill missing with the most frequent code, then
      one-hot encode (one 0/1 column per category). ``handle_unknown="ignore"``
      means a code never seen in training produces all zeros instead of an error.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                                    ("cat", categorical, CATEGORICAL_FEATURES)])
    model = LogisticRegression(max_iter=5000, class_weight=class_weight, random_state=RANDOM_SEED)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def transformed_feature_info(pipeline: Pipeline) -> tuple[list[str], list[str]]:
    """Readable names for each column the model actually sees, and the original feature it came from."""
    names, sources = list(NUMERIC_FEATURES), list(NUMERIC_FEATURES)
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_):
        for code in cats:
            names.append(f"{col} = {category_label(col, code)}")
            sources.append(col)
    return names, sources


def _curve_points(x, y, max_points=300):
    """Thin out long curves so the saved JSON / charts stay small."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), max_points)).astype(int))
    return [float(v) for v in np.asarray(x)[idx]], [float(v) for v in np.asarray(y)[idx]]


def evaluate(y_true, proba, threshold: float = DECISION_THRESHOLD) -> dict:
    """All headline metrics for a set of true labels and predicted probabilities."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baseline_accuracy_always_no_default": float(1 - np.mean(y_true)),
    }


def train_model(df: pd.DataFrame, class_weight: str | None = None, dataset_info: dict | None = None,
                seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> dict:
    """Train, evaluate and return a 'bundle' (model + everything needed to use it later)."""
    X, y, ids = df[FEATURE_COLS], df[TARGET_COL].astype(int), df[ID_COL]
    source = df[SOURCE_COL] if SOURCE_COL in df else pd.Series("dataset", index=df.index)
    # Stratify by outcome AND source file, so every file keeps its default share and
    # is represented in both the training and the test set.
    strata = source.astype(str) + "|" + y.astype(str)
    if strata.value_counts().min() < 2:  # a group too small to split -> stratify by outcome only
        strata = y
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=test_size, stratify=strata, random_state=seed)

    pipeline = build_pipeline(class_weight)
    pipeline.fit(X_train, y_train)  # <- only TRAINING data is used here

    proba_test = pipeline.predict_proba(X_test)[:, 1]  # column 1 = probability of class 1 (default)
    metrics = evaluate(y_test.values, proba_test)
    metrics["classification_report"] = classification_report(
        y_test, (proba_test >= DECISION_THRESHOLD).astype(int),
        target_names=["No default", "Default"], output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    frac_pos, mean_pred = calibration_curve(y_test, proba_test, n_bins=10, strategy="quantile")
    metrics["roc_curve"] = dict(zip(("fpr", "tpr"), _curve_points(fpr, tpr)))
    metrics["pr_curve"] = dict(zip(("recall", "precision"), _curve_points(rec[::-1], prec[::-1])))
    metrics["calibration"] = {"mean_predicted": mean_pred.tolist(), "observed_default_rate": frac_pos.tolist()}
    metrics["test_actual_default_rate"] = float(y_test.mean())
    metrics["test_mean_predicted_probability"] = float(proba_test.mean())

    # The same headline metrics separately for each source file's test customers.
    test_source = source.loc[X_test.index].values
    by_source = []
    for name in pd.unique(test_source):
        mask = test_source == name
        row = {"source": str(name), "n_test": int(mask.sum()), "actual_default_rate": float(y_test.values[mask].mean()),
               "mean_predicted_probability": float(proba_test[mask].mean())}
        if len(set(y_test.values[mask])) == 2:  # ROC-AUC needs both outcomes present
            m = evaluate(y_test.values[mask], proba_test[mask])
            row.update({k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
        by_source.append(row)
    metrics["by_source"] = by_source

    # --- Interpretation -----------------------------------------------------
    names, sources = transformed_feature_info(pipeline)
    coefs = pipeline.named_steps["model"].coef_[0]
    coefficients = pd.DataFrame({"Feature": names, "Source": sources, "Coefficient": coefs,
                                 "Odds_Ratio": np.exp(coefs)})

    # Permutation importance: shuffle one ORIGINAL column at a time in the TEST
    # set and measure how much ROC-AUC drops. Bigger drop = model relies more on it.
    perm = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=seed, n_jobs=1)
    importance = (pd.DataFrame({"Feature": FEATURE_COLS, "AUC_Drop_Mean": perm.importances_mean,
                                "AUC_Drop_Std": perm.importances_std})
                  .sort_values("AUC_Drop_Mean", ascending=False).reset_index(drop=True))

    # Average transformed training row - used to explain individual predictions
    # ("how does this customer differ from an average training customer?").
    train_mean = pipeline.named_steps["preprocess"].transform(X_train).mean(axis=0)

    return {
        "pipeline": pipeline,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "transformed_names": names,
        "transformed_sources": sources,
        "train_transformed_mean": train_mean,
        "class_weight": class_weight,
        "random_seed": seed,
        "test_size": test_size,
        "decision_threshold": DECISION_THRESHOLD,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_ids": id_test.tolist(),
        "test_results": pd.DataFrame({ID_COL: id_test.values, SOURCE_COL: test_source,
                                      "Actual_Default": y_test.values, "Predicted_Default_Probability": proba_test}),
        "dataset_files": [i["file_name"] for i in (dataset_info or {}).get("files", [])],
        "metrics": metrics,
        "coefficients": coefficients,
        "permutation_importance": importance,
        "dataset_file": (dataset_info or {}).get("file_name"),
        "dataset_fingerprint": (dataset_info or {}).get("fingerprint"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }


def save_bundle(bundle: dict, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    summary = {
        "model": "LogisticRegression",
        "class_weight": bundle["class_weight"],
        "trained_at": bundle["trained_at"],
        "dataset_file": bundle["dataset_file"],
        "n_train": bundle["n_train"],
        "n_test": bundle["n_test"],
        "random_seed": bundle["random_seed"],
        "metrics_on_test_set": bundle["metrics"],
        "permutation_importance_test_set": bundle["permutation_importance"].to_dict(orient="records"),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit-risk model.")
    parser.add_argument("--balanced", action="store_true", help='use class_weight="balanced"')
    args = parser.parse_args()

    from prediction import export_predictions  # local import avoids a circular import

    df, info = load_dataset()
    for f in info["files"]:
        print(f"Loaded {f['rows_used']:,} rows from {f['file_name']}")
    bundle = train_model(df, class_weight="balanced" if args.balanced else None, dataset_info=info)
    save_bundle(bundle)
    paths = export_predictions(df, bundle)
    paths.append(export_quality_report(data_quality_report(df, info)))
    m = bundle["metrics"]
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test ROC-AUC {m['roc_auc']:.3f} | accuracy {m['accuracy']:.3f} | "
          f"precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    for s in m["by_source"]:
        auc = f"ROC-AUC {s['roc_auc']:.3f}, recall {s['recall']:.3f}" if "roc_auc" in s else "too few cases for ROC-AUC"
        print(f"  {s['source']}: {s['n_test']:,} test customers, {auc}")
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
