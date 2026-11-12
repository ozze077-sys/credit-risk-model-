"""
Tests for the plain-Python server (server.py): the JSON endpoints and the real HTTP layer.
Run with:   python -m pytest -v
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402

Q = {"medium": ["30"], "high": ["50"]}


@pytest.fixture(scope="module")
def base_url():
    """Load data/model once and run the real HTTP server on a free port in a background thread."""
    server.STATE = server.AppState()
    assert server.STATE.error is None, server.STATE.error
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_plotly_served(base_url):
    status, ctype, body = get(base_url, "/")
    assert status == 200 and "text/html" in ctype and b"/api/overview" in body
    status, _, body = get(base_url, "/static/plotly.min.js")
    assert status == 200 and len(body) > 1_000_000


@pytest.mark.parametrize("path", [
    "/api/status", "/api/overview?medium=30&high=50", "/api/exploration", "/api/exploration/bill-payment?stat=Mean",
    "/api/exploration/distribution?var=AGE", "/api/segmentation?medium=30&high=50",
    "/api/segmentation/customers?medium=30&high=50", "/api/customer?id=1", "/api/form-defaults",
    "/api/performance", "/api/performance/threshold?t=0.3", "/api/quality",
])
def test_json_endpoints(base_url, path):
    status, ctype, body = get(base_url, path)
    assert status == 200 and "application/json" in ctype
    assert "error" not in json.loads(body)


def test_values_match_model(base_url):
    data = json.loads(get(base_url, "/api/overview?medium=30&high=50")[2])
    labels = {k["label"]: k["value"] for k in data["kpis"]}
    df = server.STATE.df
    assert labels["Customers"] == f"{len(df):,}"
    assert labels["Missed a payment"] == f"{df['default.payment.next.month'].mean():.1%}"
    assert len(data["summary"]) >= 3  # plain-language "In short" sentences
    customer = json.loads(get(base_url, "/api/customer?id=1&medium=30&high=50")[2])
    assert customer["actual"] == "Missed payment" and 0 <= customer["probability"] <= 1
    seg = json.loads(get(base_url, "/api/segmentation/customers?split=Test&top=5")[2])
    assert seg["count"] == server.STATE.bundle["n_test"] and len(seg["table"]["rows"]) == 5


@pytest.mark.parametrize("path,status", [
    ("/api/overview?medium=60&high=50", 400), ("/api/customer?id=999999", 404), ("/api/customer?id=abc", 400),
    ("/api/exploration/distribution?var=ID", 400), ("/api/unknown", 404), ("/../server.py", 404),
])
def test_bad_requests_return_json_errors(base_url, path, status):
    code, ctype, body = get(base_url, path)
    assert code == status and "error" in json.loads(body)


def test_downloads(base_url):
    for path, marker in [("/api/download/metrics.json", b"metrics_on_test_set"),
                         ("/api/download/test_predictions.csv", b"Correct_Prediction"),
                         ("/api/download/data_quality.csv", b"Section"),
                         ("/api/segmentation/export.csv?segment=High%20risk", b"Risk_Segment")]:
        status, _, body = get(base_url, path)
        assert status == 200 and marker in body


def test_predict_and_validation(base_url):
    form = json.loads(get(base_url, "/api/form-defaults")[2])
    values = {f["name"]: f["value"] for f in form["profile"] + [f for row in form["months"] for f in row]}
    status, result = post(base_url, "/api/predict", {"values": values, "medium": 30, "high": 50})
    assert status == 200 and result["segment"] in ("Low risk", "Medium risk", "High risk")
    status, result = post(base_url, "/api/predict", {"values": {**values, "AGE": 5}, "medium": 30, "high": 50})
    assert status == 400 and "Age must be between" in result["error"]
