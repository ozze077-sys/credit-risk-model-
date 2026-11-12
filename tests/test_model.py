"""
Basic tests. Run from the project folder with:   python -m pytest -v

They use the real dataset in the project folder. The model is trained once
for the whole test module (takes ~30 seconds).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import (FEATURE_COLS, ID_COL, SOURCE_COL, TARGET_COL, DatasetError,  # noqa: E402
                         data_quality_report, find_dataset_files, load_dataset)
from model_training import train_model  # noqa: E402
from prediction import (ModelError, assign_segments, build_prediction_table, explain_prediction,  # noqa: E402
                        export_predictions, load_model, predict_probability, validate_customer_input)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def bundle(dataset):
    df, info = dataset
    return train_model(df, dataset_info=info)


# --- Data loading ---------------------------------------------------------------
def test_all_dataset_files_combined(dataset):
    df, info = dataset
    files = find_dataset_files()
    assert len(files) >= 1 and all(p.suffix.lower() in (".xls", ".xlsx", ".csv") for p in files)
    assert len(info["files"]) == len(files)
    assert len(df) == sum(i["rows_used"] for i in info["files"])
    assert set(df[SOURCE_COL]) == {p.name for p in files}
    assert set(df[TARGET_COL].unique()) == {0, 1}
    assert df[ID_COL].is_unique


def test_uci_file_shape_and_header():
    uci = next(p for p in find_dataset_files() if p.name.lower().startswith("default of credit card"))
    df, info = load_dataset(uci)
    assert df.drop(columns=[SOURCE_COL]).shape == (30_000, 25)
    assert info["files"][0]["header_row_excel"] == 2  # the X1..Y row sits above the real names


def _write_csv(path, ids, target):
    cols = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
            "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] \
           + ["default.payment.next.month"]
    rows = [[i, 50000, 1, 2, 1, 30] + [0] * 6 + [1000] * 6 + [100] * 6 + [t] for i, t in zip(ids, target)]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_multiple_files_are_stacked(tmp_path):
    _write_csv(tmp_path / "a.csv", [1, 2], [0, 1])
    _write_csv(tmp_path / "b.csv", [3, 4, 5], [1, 0, 0])
    (tmp_path / "notes.csv").write_text("hello,world\n1,2\n")  # unrelated file is skipped
    df, info = load_dataset(folder=tmp_path)
    assert len(df) == 5 and df[SOURCE_COL].value_counts().to_dict() == {"b.csv": 3, "a.csv": 2}


def test_same_id_in_two_files_raises(tmp_path):
    _write_csv(tmp_path / "a.csv", [1, 2], [0, 1])
    _write_csv(tmp_path / "b.csv", [2, 3], [1, 0])
    with pytest.raises(DatasetError, match="same customer ID"):
        load_dataset(folder=tmp_path)


def test_quality_report(dataset):
    report = data_quality_report(*dataset)
    assert report["summary"]["Rows used (valid target)"] == len(dataset[0])
    assert report["target"]["Customers"].sum() == len(dataset[0])


def test_extra_header_row_and_alias_detected(tmp_path):
    cols = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4",
            "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] \
           + ["default.payment.next.month"]
    rows = [[f"X{i}" for i in range(len(cols))], cols, list(range(1, 26)), [2] + list(range(1, 24)) + [0]]
    rows[2][-1] = 1
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    df, info = load_dataset(path)
    assert info["files"][0]["header_row_excel"] == 2 and "PAY_0" in df and len(df) == 2


def test_missing_target_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ID": [1], "LIMIT_BAL": [1], "SEX": [1], "EDUCATION": [1], "MARRIAGE": [1], "AGE": [30]}) \
        .to_csv(path, index=False)
    with pytest.raises(DatasetError):
        load_dataset(path)


# --- Model ------------------------------------------------------------------------
def test_features_exclude_id_and_target():
    assert ID_COL not in FEATURE_COLS and TARGET_COL not in FEATURE_COLS


def test_train_test_split(bundle, dataset):
    df, _ = dataset
    assert bundle["n_train"] + bundle["n_test"] == len(df)
    assert abs(bundle["n_test"] / len(df) - 0.2) < 0.001
    tr = bundle["test_results"]
    assert abs(tr["Actual_Default"].mean() - df[TARGET_COL].mean()) < 0.01  # stratified


def test_probabilities_valid(bundle, dataset):
    p = predict_probability(bundle, dataset[0].head(500))
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))


def test_metrics_match_sklearn(bundle):
    tr, m = bundle["test_results"], bundle["metrics"]
    y, p = tr["Actual_Default"], tr["Predicted_Default_Probability"]
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["recall"] == pytest.approx(recall_score(y, p >= 0.5))
    assert m["precision"] == pytest.approx(precision_score(y, p >= 0.5))
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == len(tr)
    assert m["roc_auc"] > 0.65  # clearly better than random guessing (0.5)


def test_explanation_adds_up(bundle, dataset):
    row = dataset[0].iloc[[0]]
    p = predict_probability(bundle, row)[0]
    model = bundle["pipeline"].named_steps["model"]
    base = model.intercept_[0] + model.coef_[0] @ bundle["train_transformed_mean"]
    total = explain_prediction(bundle, row, top_n=100)["Contribution"].sum()
    assert np.log(p / (1 - p)) == pytest.approx(base + total, abs=1e-6)


# --- Segmentation, validation, export ----------------------------------------------
def test_segments_boundaries():
    seg = assign_segments([0.0, 0.2999, 0.30, 0.4999, 0.50, 1.0], 0.30, 0.50)
    assert list(seg) == ["Low risk", "Low risk", "Medium risk", "Medium risk", "High risk", "High risk"]
    with pytest.raises(ValueError):
        assign_segments([0.5], 0.6, 0.5)


def test_input_validation(dataset):
    good = dataset[0][FEATURE_COLS].iloc[0].to_dict()
    assert validate_customer_input(good) == []
    bad = {**good, "AGE": 5, "EDUCATION": 9, "PAY_AMT1": -10}
    assert len(validate_customer_input(bad)) == 3


def test_export(bundle, dataset, tmp_path):
    all_path, test_path = export_predictions(dataset[0], bundle, outputs_dir=tmp_path)
    out = pd.read_csv(all_path)
    for col in ["ID", "Actual_Default", "Predicted_Default_Probability", "Predicted_Default", "Risk_Segment"]:
        assert col in out
    assert len(out) == len(dataset[0])
    test = pd.read_csv(test_path)
    assert len(test) == bundle["n_test"] and set(test["Data_Split"]) == {"Test"}
    pred = build_prediction_table(dataset[0], bundle)
    assert (pred["Predicted_Default"] == (pred["Predicted_Default_Probability"] >= 0.5)).all()


def test_missing_and_corrupt_model(tmp_path):
    with pytest.raises(ModelError):
        load_model(tmp_path / "nope.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelError):
        load_model(bad)
