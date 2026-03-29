import pytest
import src.app as app_module
from fastapi.testclient import TestClient

client = TestClient(app_module.app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_payload(**overrides) -> dict:
    """Return a complete valid transaction payload."""
    payload = {f"V{i}": float(i % 5 - 2) for i in range(1, 29)}
    payload["Amount"] = 150.0
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------

def test_model_info_shape():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_name" in body
    assert "version" in body
    assert "threshold" in body
    assert isinstance(body["threshold"], float)


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_count():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_predictions" in body
    assert isinstance(body["total_predictions"], int)


def test_metrics_increments_after_predict():
    before = client.get("/metrics").json()["total_predictions"]
    client.post("/predict", json=_valid_payload())
    after = client.get("/metrics").json()["total_predictions"]
    assert after == before + 1


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_valid_returns_200():
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200


def test_predict_response_fields():
    resp = client.post("/predict", json=_valid_payload())
    body = resp.json()
    assert "fraud_probability" in body
    assert "prediction" in body
    assert isinstance(body["fraud_probability"], float)
    assert body["prediction"] in (0, 1)


def test_predict_with_time_field_ignored():
    """Time is optional and must be accepted without affecting the result."""
    payload = _valid_payload(Time=86400.0)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_probability_from_mock():
    """Mock returns prob_class1=0.05, which is below threshold 0.7 → prediction=0."""
    resp = client.post("/predict", json=_valid_payload())
    body = resp.json()
    assert body["fraud_probability"] == pytest.approx(0.05)
    assert body["prediction"] == 0


def test_predict_fraud_flag_above_threshold(monkeypatch):
    """When prob_class1 > threshold, prediction should be 1."""
    app_module.model._model_impl.sklearn_model.predict_proba.return_value = [[0.1, 0.95]]
    resp = client.post("/predict", json=_valid_payload())
    body = resp.json()
    assert body["prediction"] == 1
    # Restore
    app_module.model._model_impl.sklearn_model.predict_proba.return_value = [[0.2, 0.05]]


# ---------------------------------------------------------------------------
# /predict — validation failures (expect 422)
# ---------------------------------------------------------------------------

def test_predict_missing_required_field():
    payload = _valid_payload()
    del payload["V1"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_amount():
    payload = _valid_payload()
    del payload["Amount"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_amount_rejected():
    resp = client.post("/predict", json=_valid_payload(Amount=-10.0))
    assert resp.status_code == 422


def test_predict_v_feature_too_large():
    resp = client.post("/predict", json=_valid_payload(V1=999.0))
    assert resp.status_code == 422


def test_predict_v_feature_too_small():
    resp = client.post("/predict", json=_valid_payload(V5=-999.0))
    assert resp.status_code == 422


def test_predict_wrong_type_string_rejected():
    payload = _valid_payload()
    payload["Amount"] = "not_a_number"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_body_rejected():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
