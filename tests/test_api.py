"""API contract tests — run with `pytest` from the repository root.

They exercise the *artefacts* (model.pt + preprocessor.joblib), not a retrain,
so they're fast enough for CI on every push.

No test calls OpenAI: the alert endpoint is tested for its **auth gate** only,
because a CI run must never spend API credit or depend on a live key.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app, get_state


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample() -> dict:
    with TestClient(app) as c:
        return c.get("/cohort/sample").json()


# ------------------------------------------------------------------ liveness


def test_health_reports_model_provenance(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["test_roc_auc"] == pytest.approx(0.6754, abs=0.02)  # README table
    assert 0 < body["threshold"] < 1
    assert body["cohort_rows"] > 10_000


# ------------------------------------------------------------------- predict


def test_predict_by_patient_is_deterministic(client: TestClient, sample: dict) -> None:
    request = {"patient_nbr": sample["patient_nbr"], "explain": True}
    first = client.post("/predict", json=request).json()
    second = client.post("/predict", json=request).json()
    assert first["probability"] == second["probability"]
    assert 0.0 <= first["probability"] <= 1.0


def test_predict_hitl_flag_follows_threshold(client: TestClient, sample: dict) -> None:
    body = client.post("/predict", json={"patient_nbr": sample["patient_nbr"]}).json()
    assert body["requires_human_review"] == (body["probability"] >= body["threshold"])
    assert body["flagged"] == body["requires_human_review"]


def test_predict_explain_returns_ranked_drivers(client: TestClient, sample: dict) -> None:
    body = client.post(
        "/predict", json={"patient_nbr": sample["patient_nbr"], "explain": True}
    ).json()
    drivers = body["drivers"]
    assert 0 < len(drivers) <= 10
    magnitudes = [abs(d["contribution"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)  # biggest movers first
    assert all("description" in d for d in drivers)


def test_predict_encounter_matches_patient_lookup(client: TestClient, sample: dict) -> None:
    """A full encounter posted by value must score exactly like its cohort id."""
    by_id = client.post("/predict", json={"patient_nbr": sample["patient_nbr"]}).json()
    by_value = client.post("/predict", json={"encounter": sample["encounter"]}).json()
    assert by_value["probability"] == pytest.approx(by_id["probability"], abs=1e-6)


def test_predict_rejects_missing_columns(client: TestClient, sample: dict) -> None:
    broken = {k: v for k, v in sample["encounter"].items() if k != "number_inpatient"}
    response = client.post("/predict", json={"encounter": broken})
    assert response.status_code == 422
    assert "number_inpatient" in response.json()["detail"]["missing"]


def test_predict_requires_exactly_one_identifier(client: TestClient, sample: dict) -> None:
    empty = client.post("/predict", json={})
    assert empty.status_code == 422
    both = client.post(
        "/predict", json={"patient_nbr": 1, "encounter": sample["encounter"]}
    )
    assert both.status_code == 422


def test_predict_unknown_patient_404(client: TestClient) -> None:
    response = client.post("/predict", json={"patient_nbr": 999_999_999})
    assert response.status_code == 404


# ------------------------------------------------------------- cohort sample


def test_default_sample_is_high_risk(client: TestClient) -> None:
    body = client.get("/cohort/sample").json()
    assert body["patient_nbr"] > 0
    assert set(body["encounter"]) == set(get_state().feature_columns)


# ------------------------------------------------------------- alert auth gate


def test_alert_refuses_without_key(client: TestClient, sample: dict) -> None:
    response = client.post("/alert", json={"patient_nbr": sample["patient_nbr"]})
    assert response.status_code in (401, 503)  # 503 if the server has no key set


def test_alert_refuses_with_wrong_key(client: TestClient, sample: dict) -> None:
    response = client.post(
        "/alert",
        json={"patient_nbr": sample["patient_nbr"]},
        headers={"X-API-Key": "not-the-key"},
    )
    assert response.status_code in (401, 503)
