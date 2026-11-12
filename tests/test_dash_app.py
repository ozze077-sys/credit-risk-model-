"""
Smoke tests for the Dash dashboard: every page builds and the main callbacks run.
Run with:   python -m pytest -v
"""

import json
import sys
from pathlib import Path

import plotly
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THRESHOLDS = [30, 50]


@pytest.fixture(scope="module")
def app_module():
    import dash_app
    assert dash_app.LOAD_ERROR is None, dash_app.LOAD_ERROR
    return dash_app


def as_json(component) -> str:
    return json.dumps(component, cls=plotly.utils.PlotlyJSONEncoder)


@pytest.mark.parametrize("path", ["/", "/exploration", "/segmentation", "/individual", "/performance", "/quality"])
def test_every_page_renders(app_module, path):
    page, nav, subtitle, *_ = app_module.render_page(path, THRESHOLDS, 0)
    assert app_module.PAGES[path] in subtitle
    as_json(page)


def test_unknown_path_falls_back_to_overview(app_module):
    assert app_module.render_page("/does-not-exist", THRESHOLDS, 0)[2].startswith("Overview")


def test_segment_filters(app_module):
    d = app_module
    _, count = d.update_segment_table(d.SEGMENTS, d.AGE_LABELS, d.LIMIT_LABELS, ["Test"], 25, THRESHOLDS)
    assert count.startswith(f"{d.STATE['bundle']['n_test']:,} of {len(d.DF):,}")
    _, count = d.update_segment_table([], d.AGE_LABELS, d.LIMIT_LABELS, ["Test"], 25, THRESHOLDS)
    assert count.startswith("0 of")


def test_existing_customer_and_unknown_id(app_module):
    assert "Estimated probability" in as_json(app_module.show_existing(1, THRESHOLDS))
    assert "No customer with ID" in as_json(app_module.show_existing(999_999, THRESHOLDS))


def test_new_customer_validation(app_module):
    d = app_module
    values = d.DF[d.FEATURE_COLS].iloc[0].tolist()
    assert "Estimated probability" in as_json(d.estimate_new(1, *values, THRESHOLDS))
    values[d.FEATURE_COLS.index("AGE")] = 5
    assert "Please fix the following" in as_json(d.estimate_new(1, *values, THRESHOLDS))
