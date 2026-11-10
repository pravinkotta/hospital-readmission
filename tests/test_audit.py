"""Governance artefacts must stay well-formed: fairness + drift summaries.

These tests read the committed JSON (produced by ``python -m src.audit``);
CI regenerates them on every push, so a code change that silently corrupts the
audit output fails the build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def load(name: str) -> dict:
    path = ARTIFACTS / name
    assert path.is_file(), f"{name} missing — run: python -m src.audit"
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ fairness


@pytest.fixture(scope="module")
def fairness() -> dict:
    return load("fairness.json")


def test_all_sensitive_attributes_audited(fairness: dict) -> None:
    assert set(fairness["attributes"]) == {"race", "gender", "age"}
    assert fairness["notes"], "the interpretation caveat is required, not optional"


def test_difference_metrics_are_probabilities(fairness: dict) -> None:
    for attribute, block in fairness["attributes"].items():
        for key in ("demographic_parity_difference", "equalized_odds_difference"):
            value = block[key]
            assert 0.0 <= value <= 1.0, f"{attribute}.{key}={value} outside [0,1]"


def test_group_rates_sum_to_cohort(fairness: dict) -> None:
    total = sum(g["n"] for g in fairness["attributes"]["race"]["groups"].values())
    assert total == fairness["test_size"]
    assert 10_000 < total < 20_000  # the held-out split is ~14.8k encounters


# ---------------------------------------------------------------------- drift


@pytest.fixture(scope="module")
def drift() -> dict:
    return load("drift_summary.json")


def test_drift_summary_is_internally_consistent(drift: dict) -> None:
    assert drift["recommend_retrain"] == drift["dataset_drift"]
    assert set(drift["drifted_columns"]) == {
        name for name, info in drift["per_column"].items() if info["drifted"]
    }
    assert drift["number_of_drifted_columns"] == len(drift["drifted_columns"])


def test_drift_verdict_respects_its_own_rule(drift: dict) -> None:
    share = drift["share_of_drifted_columns"]
    assert share == pytest.approx(
        drift["number_of_drifted_columns"] / drift["columns_checked"], abs=1e-6
    )
    assert drift["dataset_drift"] == (share > drift["drift_share_threshold"])


def test_drift_interpretation_is_recorded(drift: dict) -> None:
    """An offline train/test comparison must not be sold as production drift."""
    assert "production" in drift["interpretation"]
    assert Path(ARTIFACTS / drift["report"]).name == drift["report"]
