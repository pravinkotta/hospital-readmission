"""FastAPI service around the trained ANN + SHAP + GenAI alert stage.

    uvicorn src.api:app --port 8000          # local
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000   # EC2 / Docker

Endpoints
---------
``GET  /health``        liveness + model provenance (no auth)
``GET  /cohort/sample`` a full cleaned encounter to copy into a request
``POST /predict``       risk probability, band, HITL flag, optional SHAP drivers
``POST /alert``         LLM discharge summary — **requires** ``X-API-Key``

Design notes an interviewer will probe:

- **Loaded once, at first use** — model, preprocessor, SHAP explainer and the
  held-out cohort are singletons; a request never re-reads training code.
- **The API serves the artefacts, not the training pipeline** (C7/C8):
  ``model.pt`` + ``preprocessor.joblib`` are the contract.
- **Auth economics:** ``/predict`` costs milliseconds of CPU, so it is open by
  default; ``/alert`` spends *your* OpenAI credit, so it is gated by
  ``ALERT_API_KEY`` (header ``X-API-Key``) and refuses to run unconfigured —
  an exposed endpoint that burns API money is a vulnerability, not a demo.
- **HITL gate (EU AI Act Art. 14):** every response carries
  ``requires_human_review`` — true whenever the prediction crosses the
  validation-tuned threshold.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from . import config, explain, rag
from .alert import (
    build_messages,
    call_openai,
    describe_driver,
    encounter_facts,
    env_file_value,
    find_api_key,
    held_out_frame,
    load_artifacts,
    load_id_labels,
    resolve_api_key,
    retrieval_query,
    risk_band,
)
from .data import load_clean

logger = logging.getLogger("api")

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="30-day readmission risk for diabetic inpatients — ANN + SHAP + GenAI alerts",
    version="1.0.0",
)


# ------------------------------------------------------------------- request


class ScoreRequest(BaseModel):
    patient_nbr: int | None = Field(
        default=None,
        description="Score a patient from the held-out cohort (demo shortcut)",
    )
    encounter: dict[str, Any] | None = Field(
        default=None,
        description="A full cleaned encounter; GET /cohort/sample for the exact shape",
    )
    explain: bool = Field(default=False, description="Include top SHAP drivers")


# ------------------------------------------------------------------ singleton


class State:
    def __init__(self) -> None:
        self.stack = load_artifacts()          # model + preprocessor + explainer
        self.cohort = held_out_frame(load_clean())
        self.features: list[str] = self.stack["features"]
        self.feature_columns = [
            c for c in self.cohort.columns if c not in (config.TARGET, config.ID_COL)
        ]
        self.id_labels = load_id_labels()
        matrix = np.asarray(
            self.stack["preprocessor"].transform(self.cohort[self.feature_columns]),
            dtype=np.float32,
        )
        probabilities = self.stack["model"].predict_proba(
            torch.from_numpy(matrix)
        ).numpy()
        self.cohort = self.cohort.assign(_probability=probabilities)
        logger.info("API state ready: %d cohort rows, %d inputs", len(self.cohort), len(self.features))


@lru_cache(maxsize=1)
def get_state() -> State:
    return State()


# ------------------------------------------------------------------ helpers


def alert_api_key() -> str:
    return os.environ.get("ALERT_API_KEY", "").strip() or env_file_value("ALERT_API_KEY")


def check_api_key(x_api_key: str | None) -> None:
    expected = alert_api_key()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="/alert is disabled: ALERT_API_KEY is not configured on this server",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def coerce_encounter(state: State, encounter: dict[str, Any]) -> pd.DataFrame:
    """Validate + type-align a request encounter to the training schema.

    Categoricals must be strings at predict time — the fitted one-hot encoder
    saw ``"1"`` and ``1`` is a *different* level, so a mismatch would silently
    encode as all-zeros. Numbers arrive as ``70-80``-style bands and 0/1 flags.
    """
    missing = [c for c in state.feature_columns if c not in encounter]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "encounter is missing model inputs",
                "missing": missing[:15],
                "hint": "GET /cohort/sample returns a complete encounter",
            },
        )

    row = {c: encounter[c] for c in state.feature_columns}
    frame = pd.DataFrame([row], columns=state.feature_columns)

    for column in state.feature_columns:
        if column in config.CATEGORICAL_COLS or column.startswith("diag_"):
            value = frame[column].iloc[0]
            if isinstance(value, bool):
                frame[column] = "1" if value else "0"
            elif isinstance(value, float) and float(value).is_integer():
                frame[column] = str(int(value))  # 0.0/1.0 -> "0"/"1"
            else:
                frame[column] = str(value)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def score(state: State, frame: pd.DataFrame) -> tuple[float, np.ndarray]:
    matrix = np.asarray(state.stack["preprocessor"].transform(frame), dtype=np.float32)
    probability = float(state.stack["model"].predict_proba(
        torch.from_numpy(matrix)
    ).numpy()[0])
    return probability, matrix


def base_payload(state: State, row: pd.Series, probability: float) -> dict:
    band, action = risk_band(probability)
    threshold = state.stack["threshold"]
    flagged = probability >= threshold
    return {
        "patient_nbr": int(row[config.ID_COL]),
        "probability": round(probability, 4),
        "risk_band": band,
        "threshold": threshold,
        "flagged": flagged,
        "requires_human_review": flagged,  # EU AI Act Art. 14 HITL gate
        "recommended_action": action,
    }


def drivers_for(state: State, row: pd.Series, matrix_row: np.ndarray) -> list[dict]:
    drivers = explain.contributions(state.stack["explainer"], matrix_row, state.features)
    return [
        {**d, "description": describe_driver(d, row, state.id_labels)}
        for d in drivers
    ]


# ------------------------------------------------------------------ endpoints


def openai_configured() -> bool:
    """True if a usable key exists (env, named .env entry, or bare sk- line)."""
    return bool(find_api_key())


@lru_cache(maxsize=1)
def rag_status() -> str:
    """One-time check of the guideline index, reported by /health."""
    try:
        rag.require_index()
    except SystemExit as error:
        return f"unavailable: {str(error).splitlines()[0]}"
    return f"enabled ({rag.COLLECTION}, top-{config.RAG_TOP_K})"


@app.get("/health")
def health() -> dict:
    state = get_state()
    return {
        "status": "ok",
        "model": "ReadmissionMLP 120->64->32->1",
        "threshold": state.stack["threshold"],
        "test_roc_auc": state.stack["model_roc_auc"],
        "cohort_rows": len(state.cohort),
        "alert_endpoint": "enabled" if alert_api_key() else "disabled (ALERT_API_KEY not set)",
        "openai_key_configured": openai_configured(),
        "rag": rag_status(),
    }


def python_scalar(value: Any) -> Any:
    """numpy scalars (int64, float64, str_) -> plain Python, NaN -> null.

    FastAPI/pydantic cannot serialise ``numpy.int64``; converting here keeps
    the endpoint's JSON identical whether the row came from pandas or a client.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if pd.isna(number) else number
    if isinstance(value, str):  # np.str_ is a str subclass
        return str(value)
    if pd.isna(value):
        return None
    return value


@app.get("/cohort/sample")
def cohort_sample(patient_nbr: int | None = Query(default=None)) -> dict:
    """A complete cleaned encounter — paste it straight into POST /predict."""
    state = get_state()
    if patient_nbr is not None:
        rows = state.cohort[state.cohort[config.ID_COL] == patient_nbr]
        if rows.empty:
            raise HTTPException(404, f"patient {patient_nbr} is not in the held-out test split")
        # A patient can have many encounters in the split — take the
        # highest-risk one, exactly as src/alert.py does (they must agree).
        rows = rows.sort_values("_probability", ascending=False)
    else:
        rows = state.cohort.nlargest(1, "_probability")  # highest-risk by default
    row = rows.iloc[0]
    return {
        "patient_nbr": int(row[config.ID_COL]),
        "label_known_for_testing": int(row[config.TARGET]),
        "encounter": {
            c: python_scalar(row[c]) for c in state.feature_columns
        },
    }


@app.post("/predict")
def predict(request: ScoreRequest) -> dict:
    """Risk score for one encounter; ``explain: true`` adds SHAP drivers."""
    state = get_state()
    row, matrix_row, probability = _resolve(state, request)
    payload = base_payload(state, row, probability)
    if request.explain:
        payload["drivers"] = drivers_for(state, row, matrix_row)
    return payload


def _resolve(state: State, request: ScoreRequest) -> tuple[pd.Series, np.ndarray, float]:
    if (request.patient_nbr is None) == (request.encounter is None):
        raise HTTPException(
            422, "provide exactly one of: patient_nbr, encounter"
        )
    if request.patient_nbr is not None:
        rows = state.cohort[state.cohort[config.ID_COL] == request.patient_nbr]
        if rows.empty:
            raise HTTPException(404, f"patient {request.patient_nbr} is not in the held-out test split")
        # highest-risk encounter of this patient — same rule as src/alert.py
        row = rows.sort_values("_probability", ascending=False).iloc[0]
        frame = pd.DataFrame([[row[c] for c in state.feature_columns]], columns=state.feature_columns)
    else:
        frame = coerce_encounter(state, request.encounter or {})
        # a hand-built encounter has no patient id in the cohort
        row = frame.iloc[0]
        row = row.copy()
        row[config.ID_COL] = -1
        row[config.TARGET] = -1
    probability, matrix_row = score(state, frame)
    return row, matrix_row, probability


@app.post("/alert")
def alert(request: ScoreRequest, x_api_key: str | None = Header(default=None)) -> dict:
    """LLM discharge summary — authenticated because it spends API credit."""
    check_api_key(x_api_key)
    try:
        resolve_api_key()  # env or .env — raises SystemExit if genuinely absent
    except SystemExit as error:
        raise HTTPException(503, "server has no OPENAI_API_KEY configured") from error

    state = get_state()
    row, matrix_row, probability = _resolve(state, request)
    band, action = risk_band(probability)
    threshold = state.stack["threshold"]
    drivers = drivers_for(state, row, matrix_row)

    case = {
        "patient_nbr": int(row[config.ID_COL]),
        "probability": round(probability, 4),
        "threshold": threshold,
        "flagged": probability >= threshold,
        "risk_band": band,
        "recommended_action": action,
        "facts": encounter_facts(row, state.id_labels),
        "drivers": drivers,
    }

    # Guideline retrieval: fail loud (503) instead of writing an alert that
    # claims to be grounded but had nothing to ground on.
    try:
        rag.require_index()
    except SystemExit as error:
        raise HTTPException(
            503, f"RAG index unavailable on this server: {str(error).splitlines()[0]}"
        ) from error
    query = retrieval_query(case)
    case["retrieval_query"] = query
    case["guideline_chunks"] = rag.retrieve(query, k=config.RAG_TOP_K)

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        text, usage = call_openai(build_messages(case), model_name)
    except Exception as error:  # noqa: BLE001 - surfaced as an API error, never faked
        raise HTTPException(502, f"OpenAI call failed ({type(error).__name__}): {error}") from error

    return {**{k: case[k] for k in ("patient_nbr", "probability", "risk_band", "threshold", "flagged")},
            "requires_human_review": case["flagged"],
            "source": f"openai:{model_name}",
            "retrieval": {
                "collection": rag.COLLECTION,
                "top_k": config.RAG_TOP_K,
                "query": query,
                "chunks": [
                    {
                        "label": f"[S{chunk['rank']}]",
                        "document": chunk["title"],
                        "url": chunk["url"],
                        "score": chunk["score"],
                    }
                    for chunk in case["guideline_chunks"]
                ],
            },
            "usage": usage,
            "alert_markdown": text}
