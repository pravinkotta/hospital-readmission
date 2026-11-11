"""GenAI discharge alert — the last stage of the pipeline.

    Data -> Feature Engineering -> ANN -> SHAP -> **RAG + GenAI Alert**

    python -m src.alert                  # 3 highest-risk held-out patients
    python -m src.alert --n 1            # just the top one
    python -m src.alert --patient 123456 # a specific patient from the test split
    python -m src.alert --no-rag         # skip guideline retrieval

Nothing here retrains anything: the trained artefacts are loaded from
``artifacts/``, patients are scored on the **same held-out test split** used for
the README metrics, and each patient's SHAP drivers are handed to an OpenAI
model together with the encounter facts **and the top-5 chunks retrieved from
discharge-guideline documents** (``src.rag``). The model writes the alert; it
never sees or invents the numbers — every fact in the prompt comes from the
dataset, and every suggested action must cite a retrieved chunk as [S1].

This stage **requires** ``OPENAI_API_KEY`` — there is no template fallback. A
missing key, a missing RAG index (unless ``--no-rag``), or a failed API call
stops the run with a non-zero exit code rather than writing an alert that no
LLM produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config, explain, preprocess, rag
from .data import load_clean
from .model import ReadmissionMLP
from .train import patient_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
)
logger = logging.getLogger("alert")

ALERT_DIR = config.ARTIFACT_DIR / "alerts"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------------------------------------------- wording

# Feature names arrive from the fitted ColumnTransformer ("number_inpatient",
# "age_70-80", "discharge_disposition_id_6"); these labels turn them into
# sentences a clinician can read.
FEATURE_LABELS = {
    "time_in_hospital": "length of stay (days)",
    "num_lab_procedures": "lab procedures this stay",
    "num_procedures": "non-lab procedures this stay",
    "num_medications": "medications this stay",
    "number_outpatient": "outpatient visits in the prior 6 months",
    "number_emergency": "emergency visits in the prior 6 months",
    "number_inpatient": "inpatient visits in the prior 6 months",
    "number_diagnoses": "diagnoses coded this stay",
    "age": "age band",
    "race": "race",
    "gender": "gender",
    "admission_type_id": "admission type",
    "discharge_disposition_id": "discharge destination",
    "admission_source_id": "admission source",
    "max_glu_serum": "glucose serum test result",
    "A1Cresult": "A1C result",
    "insulin": "insulin",
    "change": "medication changed this stay",
    "diabetesMed": "diabetes medication prescribed",
    "diag_1_group": "primary diagnosis (ICD-9 chapter)",
    "diag_2_group": "secondary diagnosis (ICD-9 chapter)",
    "diag_3_group": "tertiary diagnosis (ICD-9 chapter)",
    "diag_1": "primary diagnosis (ICD-9 code)",
    "diag_2": "secondary diagnosis (ICD-9 code)",
    "diag_3": "tertiary diagnosis (ICD-9 code)",
}

NUMERIC_FEATURES = set(config.NUMERIC_COLS)

SYSTEM_PROMPT = """\
You are a clinical decision-support writer producing a discharge alert for a
hospital transition-of-care team. You are given structured facts about one
encounter, the model's predicted 30-day readmission risk, the SHAP features
that drove that prediction (SHAP values are in log-odds), and retrieved chunks
from AHRQ discharge-guideline documents, labelled [S1], [S2], ...

Rules:
- Use ONLY the facts provided. Never invent symptoms, results, medications,
  vital signs, or history that is not in the input.
- SHAP contributions explain the model's score; describe them as reasons the
  *model* flagged the patient, not as proven causes of readmission.
- Every suggested discharge action must be grounded in a retrieved chunk and
  cite it inline as [S1] (or [S2], ...). If no retrieved chunk supports a
  suggestion, write "insufficient guidance" instead of suggesting one.
- Write for a discharge nurse: concrete, specific, non-alarmist.
- Output markdown with exactly these five sections, and nothing else:
  ## Risk summary
  ## Why the model flagged this patient
  ## Suggested discharge actions
  ## Sources
  ## Caveat
- 160-240 words total (the Sources list does not count toward the total).
  The Caveat section must state that this is a model-generated draft for
  clinician review, not medical advice. The Sources section must contain
  exactly the chunk labels cited in "Suggested discharge actions", one per
  line, formatted as: - [S1] document title — url
  Copy each chunk's document title and url verbatim from the input; never
  write a placeholder instead of the real URL.
"""


# ------------------------------------------------------------------ artefacts


def load_artifacts() -> dict:
    """Load the trained model, fitted preprocessor, SHAP explainer and split."""
    artifact = config.ARTIFACT_DIR
    meta = json.loads((artifact / "feature_meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))

    preprocessor = preprocess.load(artifact / "preprocessor.joblib")
    model = ReadmissionMLP(meta["n_inputs"])
    model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu"))
    model.eval()

    background = torch.load(artifact / "background.pt")
    explainer = explain.build_explainer(model, background)
    return {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "features": meta["features"],
        "threshold": float(metrics["threshold"]),
        "model_roc_auc": float(metrics["roc_auc"]),
    }


def held_out_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The test split — patients the model never saw in training or tuning.

    ``patient_split`` drops the patient id (the model never sees it), so the
    partition is recomputed here from the same seed and the id re-attached for
    traceability in the alert.
    """
    _, _, _, groups = patient_split(frame, return_groups=True)
    # reset_index so a row label is its position in the transformed matrix —
    # the original frame's index is non-contiguous after filtering.
    test = frame[frame[config.ID_COL].isin(groups["test"])].copy().reset_index(drop=True)
    logger.info("held-out test: %d encounters from %d patients", len(test), test[config.ID_COL].nunique())
    return test


# ------------------------------------------------------------------ encounter


def load_id_labels() -> dict[str, dict[str, str]]:
    """Read IDS_mapping.csv — several ``<id>,description`` sections in one file."""
    path = config.DATA_DIR / "IDS_mapping.csv"
    labels: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return labels
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = next(csv.reader([line]), [])
        if len(parts) < 2:
            continue
        key, description = parts[0].strip(), parts[1].strip().rstrip(" .")
        if key.endswith("_id"):  # a new "<id>,description" section header
            section = key
            labels.setdefault(section, {})
        elif section and key != "description":
            labels[section][key] = description
    return labels


def encounter_facts(row: pd.Series, id_labels: dict[str, dict[str, str]]) -> dict:
    """Human-readable facts about one encounter — the only things the LLM may cite."""
    def mapped(column: str) -> str:
        value = str(row[column])
        return id_labels.get(column, {}).get(value, value)

    def measured(column: str) -> str:
        """A1Cresult/max_glu_serum are NaN when the test was never run."""
        value = str(row.get(column, ""))
        return "not measured" if value in {"nan", "None", "NaN", ""} else value

    facts = {
        "age_band": str(row.get("age", "unknown")),
        "race": str(row.get("race", "Unknown")),
        "gender": str(row.get("gender", "Unknown")),
        "admission_type": mapped("admission_type_id"),
        "discharge_destination": mapped("discharge_disposition_id"),
        "length_of_stay_days": int(row.get("time_in_hospital", 0)),
        "prior_inpatient_visits": int(row.get("number_inpatient", 0)),
        "prior_emergency_visits": int(row.get("number_emergency", 0)),
        "prior_outpatient_visits": int(row.get("number_outpatient", 0)),
        "diagnoses_coded": int(row.get("number_diagnoses", 0)),
        "medications_this_stay": int(row.get("num_medications", 0)),
        "primary_diagnosis": str(row.get("diag_1_group", "unknown")),
        "secondary_diagnosis": str(row.get("diag_2_group", "unknown")),
        "a1c_result": measured("A1Cresult"),
        "glucose_serum_result": measured("max_glu_serum"),
        "insulin": str(row.get("insulin", "unknown")),
        "diabetes_medication_changed": bool(int(row.get("change", 0))),
        "on_diabetes_medication": bool(int(row.get("diabetesMed", 1))),
    }
    return facts


def describe_driver(driver: dict, row: pd.Series, id_labels: dict[str, dict[str, str]]) -> str:
    """Turn one SHAP row into a sentence grounded in this patient's raw values.

    Numerics are standardised inside the model matrix, so the *raw* value is
    read from the encounter — the transformed number would be meaningless to a
    clinician. For one-hot levels a 0 means the level is absent, which is also
    informative ("not discharged home") and is phrased as such.
    """
    feature = driver["feature"]
    contribution = driver["contribution"]

    if feature in row.index:
        label = FEATURE_LABELS.get(feature, feature)
        if feature in NUMERIC_FEATURES:
            value: object = row[feature]
        elif feature in id_labels:
            value = id_labels[feature].get(str(row[feature]), str(row[feature]))
        elif feature == "change":
            value = "yes" if int(row[feature]) else "no"
        else:
            value = row[feature]
            if str(value) in {"nan", "None", "NaN"}:  # unrun tests, missing categoricals
                value = "not measured"
        return f"{label}: {value} (SHAP {contribution:+.3f} log-odds)"

    # one-hot level, e.g. "age_70-80" or "discharge_disposition_id_6"
    for column in sorted((c for c in set(FEATURE_LABELS) | set(id_labels) if feature.startswith(f"{c}_")), key=len, reverse=True):
        level = feature[len(column) + 1 :]
        label = FEATURE_LABELS.get(column, column)
        # A description is used whether the level is present or absent —
        # "not discharged home" is as meaningful to a clinician as "home".
        level = id_labels.get(column, {}).get(level, level)
        present = float(driver["value"]) > 0.5
        phrase = f"{label}: {level}" if present else f"{label}: not {level}"
        return f"{phrase} (SHAP {contribution:+.3f} log-odds)"

    return f"{feature} (SHAP {contribution:+.3f} log-odds)"


# ------------------------------------------------------------------- selection


def risk_band(probability: float) -> tuple[str, str]:
    for ceiling, label, action in config.RISK_BANDS:
        if probability < ceiling:
            return label, action
    return config.RISK_BANDS[-1][1], config.RISK_BANDS[-1][2]


def select_cases(
    test: pd.DataFrame,
    stack: dict,
    n: int,
    patients: list[int] | None,
) -> list[dict]:
    """Score the held-out test split and return the ``n`` highest-risk cases.

    Highest-risk first, not random: the alert stage exists to reach the
    patients who need follow-up, so the demo shows exactly that population.
    """
    feature_columns = [c for c in test.columns if c not in (config.TARGET, config.ID_COL)]
    matrix = np.asarray(stack["preprocessor"].transform(test[feature_columns]), dtype=np.float32)
    probabilities = stack["model"].predict_proba(torch.from_numpy(matrix)).numpy()

    work = test.assign(_probability=probabilities)
    if patients:
        work = work[work[config.ID_COL].isin(patients)]
        if work.empty:
            raise SystemExit(f"patient(s) {patients} not in the held-out test split")
    # A patient can have several encounters in the split; always keep the
    # highest-risk one, whether or not the patient was named explicitly.
    chosen = work.sort_values("_probability", ascending=False).head(n)

    id_labels = load_id_labels()
    cases = []
    for position, (index, row) in enumerate(chosen.iterrows(), start=1):
        probability = float(row["_probability"])
        drivers = explain.contributions(stack["explainer"], matrix[index], stack["features"])
        band, action = risk_band(probability)
        logger.info(
            "case %d | patient %s | risk %.3f (%s)",
            position, int(row[config.ID_COL]), probability, band,
        )
        cases.append(
            {
                "rank": position,
                "patient_nbr": int(row[config.ID_COL]),
                "probability": round(probability, 4),
                "threshold": stack["threshold"],
                "flagged": bool(probability >= stack["threshold"]),
                "risk_band": band,
                "recommended_action": action,
                "facts": encounter_facts(row, id_labels),
                "drivers": [
                    {**driver, "description": describe_driver(driver, row, id_labels)}
                    for driver in drivers
                ],
            }
        )
    return cases


# -------------------------------------------------------------------- prompt


def retrieval_query(case: dict) -> str:
    """Query the guideline index for this case.

    The corpus is generic discharge guidance (not patient-specific), so the
    query carries the risk band, primary diagnosis and discharge context —
    enough to bias retrieval toward the right part of the toolkit.
    """
    facts = case["facts"]
    return (
        f"{case['risk_band']} 30-day readmission risk: discharge planning and "
        f"post-discharge follow-up for a patient with primary diagnosis "
        f"{facts['primary_diagnosis']}, {facts['prior_inpatient_visits']} prior "
        f"inpatient visits, discharged to {facts['discharge_destination']}"
    )


def build_messages(case: dict) -> list[dict]:
    payload = {
        "predicted_30_day_readmission_probability": case["probability"],
        "decision_threshold": case["threshold"],
        "flagged_for_follow_up": case["flagged"],
        "risk_band": case["risk_band"],
        "recommended_action_from_protocol": case["recommended_action"],
        "encounter_facts": case["facts"],
        "shap_top_drivers": [
            {"feature": d["feature"], "description": d["description"], "contribution_log_odds": round(d["contribution"], 4)}
            for d in case["drivers"]
        ],
        "retrieved_guideline_chunks": [
            {
                "label": f"[S{chunk['rank']}]",
                "document": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
            }
            for chunk in case.get("guideline_chunks", [])
        ],
    }
    user = (
        "Write the discharge alert for this encounter. The retrieved "
        "guideline chunks are the only permitted source for suggested "
        "actions; cite them as [S1], [S2], ... where used.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_openai(messages: list[dict], model_name: str) -> tuple[str, dict]:
    from openai import OpenAI

    resolve_api_key()  # copies .env key into os.environ — the client only reads env
    client = OpenAI()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    elapsed = time.perf_counter() - started
    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
        "seconds": round(elapsed, 2),
    }
    return response.choices[0].message.content or "", usage


# ----------------------------------------------------------------------- main


def save_case(case: dict, text: str, source: str, model_name: str, usage: dict) -> Path:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERT_DIR / f"alert_patient_{case['patient_nbr']}.md"
    header = (
        f"<!-- generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"by src.alert | source={source} -->\n\n"
    )
    path.write_text(header + text, encoding="utf-8")
    case.update({"source": source, "model": model_name, "usage": usage, "file": path.name})
    return path


def env_file_value(name: str) -> str:
    """Read ``NAME`` from the project's ``.env`` file ('' if absent).

    Supports ``NAME=value``, ``export NAME=value`` and quoted values — no dotenv
    dependency, deliberately.
    """
    env_file = config.ROOT / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if line.startswith(name):
            _, _, value = line.partition("=")
            found = value.strip().strip('"').strip("'")
            if found:
                return found
    return ""


def find_api_key() -> str:
    """Locate the OpenAI key: env first, then `.env` (named or bare line).

    A key must be a single whitespace-free token — that check is what stops a
    stale/corrupted value (e.g. an earlier parse that swallowed several lines)
    from being shipped as an HTTP header, where httpx rejects it as an
    "Illegal header" and openai reports it as an opaque connection error.
    """
    def usable(value: str) -> str:
        value = value.strip()
        return value if value and not any(c.isspace() for c in value) else ""

    key = usable(os.environ.get("OPENAI_API_KEY", ""))
    if key:
        return key

    key = usable(env_file_value("OPENAI_API_KEY"))
    if key:
        return key

    env_file = config.ROOT / ".env"
    if env_file.is_file():  # a bare `sk-...` line, not `OPENAI_API_KEY=...`
        for line in env_file.read_text(encoding="utf-8").splitlines():
            candidate = usable(line)
            if candidate.startswith("sk-"):
                return candidate
    return ""


def resolve_api_key() -> str:
    """Resolve the OpenAI key and publish it in ``os.environ`` for the client."""
    key = find_api_key()
    if not key:
        env_file = config.ROOT / ".env"
        raise SystemExit(
            "OPENAI_API_KEY is not set. This stage always calls the OpenAI API.\n"
            f"Put it in {env_file} as OPENAI_API_KEY=sk-... (gitignored), or in this session:\n"
            '  $env:OPENAI_API_KEY = "sk-..."'
        )
    os.environ["OPENAI_API_KEY"] = key
    return key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SHAP- and RAG-grounded discharge alerts via the OpenAI API"
    )
    parser.add_argument("--n", type=int, default=3, help="number of highest-risk patients (default 3)")
    parser.add_argument("--patient", type=int, action="append", help="specific patient_nbr (repeatable)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="skip guideline retrieval (alerts lose [S1] grounding and ## Sources)",
    )
    args = parser.parse_args()

    # Fail fast, before the data load, so a missing key costs no time.
    resolve_api_key()

    # Fail loud if the RAG index was never built — before any OpenAI call.
    if not args.no_rag:
        rag.require_index()

    stack = load_artifacts()
    test = held_out_frame(load_clean())
    cases = select_cases(test, stack, args.n, args.patient)
    if not cases:
        raise SystemExit("no cases selected")

    if not args.no_rag:
        for case in cases:
            query = retrieval_query(case)
            case["retrieval_query"] = query
            case["guideline_chunks"] = rag.retrieve(query, k=config.RAG_TOP_K)

    index = []
    for case in cases:
        try:
            text, usage = call_openai(build_messages(case), args.model)
        except Exception as error:  # noqa: BLE001 - reported cleanly, then stop
            logger.error("OpenAI call failed for patient %s", case["patient_nbr"])
            raise SystemExit(f"OpenAI call failed ({type(error).__name__}): {error}") from error
        source = f"openai:{args.model}"
        path = save_case(case, text, source, args.model, usage)
        logger.info(
            "wrote %s [%s, %s prompt / %s completion tokens]",
            path.name, source, usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        index.append(case)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": index[0]["source"],
        "threshold": index[0]["threshold"],
        "rag": (
            {
                "enabled": False,
                "note": "--no-rag: guidelines were not retrieved for this run",
            }
            if args.no_rag
            else {
                "enabled": True,
                "collection": rag.COLLECTION,
                "top_k": config.RAG_TOP_K,
                "index": str(config.CHROMA_DIR),
            }
        ),
        "cases": index,
    }
    target = config.ARTIFACT_DIR / "alerts.json"
    target.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s (%.1f KB)", target.name, target.stat().st_size / 1024)
    print(json.dumps({"alerts": len(index), "source": index[0]["source"], "dir": str(ALERT_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
