"""
Tests for scoring a file of new customers (predict_file.py).
Run with:   python -m pytest -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import FEATURE_COLS, find_dataset_files  # noqa: E402
from predict_file import score_file  # noqa: E402
from prediction import load_model, predict_probability  # noqa: E402

COLS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
        "PAY_6"] + [c for i in range(1, 7) for c in (f"BILL_AMT{i}", f"PAY_AMT{i}")]  # different column order


def make_file(path: Path, rows: list[list]) -> Path:
    pd.DataFrame(rows, columns=COLS).to_excel(path, index=False)
    return path


def test_file_without_target_is_never_training_data(tmp_path):
    make_file(tmp_path / "new_customers.xlsx", [[1, 50000, 1, 2, 1, 30] + [0] * 6 + [1000, 100] * 6])
    pd.DataFrame({"ID": [9], **{c: [0] for c in FEATURE_COLS}, "default.payment.next.month": [0]}) \
        .to_csv(tmp_path / "train.csv", index=False)
    assert [p.name for p in find_dataset_files(tmp_path)] == ["train.csv"]


def test_score_file_writes_matching_predictions(tmp_path):
    rows = [[1, 50000, 1, 2, 1, 30] + [0] * 6 + [1000, 100] * 6,
            [2, 20000, 2, 2, 2, 25] + [3, 2, 2, 2, 2, 2] + [19000, 0] * 6,
            [3, 20000, 2, 9, 2, 5] + [0] * 6 + [1000, 100] * 6]  # invalid EDUCATION and AGE -> warnings
    src = make_file(tmp_path / "customers.xlsx", rows)
    out, result = score_file(src, outputs_dir=tmp_path / "out")
    assert out.exists()
    sheet = pd.read_excel(out, sheet_name="Predictions")
    assert list(sheet.columns[:len(COLS)]) == COLS  # original columns kept, in the original order
    assert list(sheet["ID"]) == [1, 2, 3]
    expected = predict_probability(load_model(), pd.DataFrame(rows, columns=COLS))
    assert sheet["Predicted_Default_Probability"].tolist() == pytest.approx(expected.round(4).tolist())
    assert sheet.loc[1, "Predicted_Default_Probability"] > sheet.loc[0, "Predicted_Default_Probability"]
    assert pd.isna(sheet.loc[0, "Data_Warnings"]) and "EDUCATION" in sheet.loc[2, "Data_Warnings"]
    assert "Summary" in pd.read_excel(out, sheet_name=None)
