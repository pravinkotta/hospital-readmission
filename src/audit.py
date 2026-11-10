"""Fairness audit + drift monitoring — the governance loop of the pipeline.

    python -m src.audit

Produces two artefacts from the *same* trained model (nothing is retrained):

``artifacts/fairness.json``
    Per sensitive attribute (race, gender, age) on the held-out test set:
    group sizes, selection rate (share flagged at the tuned threshold),
    TPR/FPR, plus Fairlearn's demographic-parity and equalised-odds
    differences. Honest framing is part of the artefact: a disparity is an
    *observation about this model on this data*, not proof of discrimination.

``artifacts/drift_report.html`` + ``artifacts/drift_summary.json``
    Evidently data-drift report comparing the training distribution
    (reference) with the held-out test distribution (current), machine-readable
    summary for the retrain trigger.

The mentor's loop closes here: ``drift_detected`` in the summary is what a
scheduler would watch to recommend retraining — printed and recorded, never
executed automatically (C9: measured, not magic).
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from . import config
from .data import load_clean
from .model import ReadmissionMLP
from .preprocess import load as load_preprocessor
from .train import patient_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
)
logger = logging.getLogger("audit")

SENSITIVE_ATTRIBUTES = ("race", "gender", "age")
CATEGORIES = ("race", "gender", "age", "diag_1", "diag_2", "diag_3",
              "admission_type_id", "discharge_disposition_id", "admission_source_id")


# ---------------------------------------------------------------------- model


def load_stack() -> tuple[ReadmissionMLP, object, float]:
    meta = json.loads((config.ARTIFACT_DIR / "feature_meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((config.ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    preprocessor = load_preprocessor(config.ARTIFACT_DIR / "preprocessor.joblib")
    model = ReadmissionMLP(meta["n_inputs"])
    model.load_state_dict(torch.load(config.ARTIFACT_DIR / "model.pt", map_location="cpu"))
    model.eval()
    return model, preprocessor, float(metrics["threshold"])


def scores(model, preprocessor, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = np.asarray(preprocessor.transform(frame[columns]), dtype=np.float32)
    with torch.inference_mode():
        return model.predict_proba(torch.from_numpy(matrix)).numpy()


# -------------------------------------------------------------------- fairness


def group_rates(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> dict:
    """Count, selection rate, TPR and FPR per level of a sensitive attribute."""
    out = {}
    for level in sorted(groups.astype(str).unique()):
        mask = groups.astype(str).to_numpy() == level
        n = int(mask.sum())
        if n == 0:
            continue
        positives = y_true[mask] == 1
        negatives = ~positives
        predicted = y_pred[mask] == 1
        out[level] = {
            "n": n,
            "label_rate": round(float(positives.mean()), 4),
            "selection_rate": round(float(predicted.mean()), 4),
            "tpr": round(float(predicted[positives].mean()), 4) if positives.any() else None,
            "fpr": round(float(predicted[negatives].mean()), 4) if negatives.any() else None,
        }
    return out


def fairness_report(
    test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    report = {"threshold_used": None, "attributes": {}}
    for attribute in SENSITIVE_ATTRIBUTES:
        features = test[attribute].astype(str)
        report["attributes"][attribute] = {
            "groups": group_rates(y_true, y_pred, features),
            "demographic_parity_difference": round(
                float(demographic_parity_difference(
                    y_true, y_pred, sensitive_features=features
                )), 4,
            ),
            "equalized_odds_difference": round(
                float(equalized_odds_difference(
                    y_true, y_pred, sensitive_features=features
                )), 4,
            ),
        }
    report["notes"] = (
        "Disparities are observations about this model on the UCI test split, "
        "not proof of discrimination; small groups carry wide uncertainty. "
        "Any deployment would require a domain review before acting on these numbers."
    )
    return report


# ----------------------------------------------------------------------- drift


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Evidently 0.7: ``Report.run`` returns a ``Snapshot`` (not the report).

    The snapshot carries the HTML renderer and a JSON dump; the per-column
    verdicts live in ``widgets[*].params.data[*]`` rows shaped
    ``{column_name, data_drift, drift_score, stattest_name}``. The overall
    dataset-drift verdict is Evidently's own rule: share of drifted columns
    > its configured ``drift_share`` (0.5 by default).
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=train, current_data=test)

    html_path = config.ARTIFACT_DIR / "drift_report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    data = json.loads(snapshot.dumps())
    rows = []
    for widget in data.get("widgets", []):
        params = widget.get("params") or {}
        for row in (params.get("data") or []):
            if isinstance(row, dict) and "column_name" in row and "data_drift" in row:
                rows.append(row)

    drift_share_threshold = 0.5
    for entry in (data.get("metric_results") or {}).values():
        params = (entry.get("metric_value_location", {}).get("metric", {}) or {}).get("params") or {}
        if "drift_share" in params:
            drift_share_threshold = float(params["drift_share"])
            break

    per_column = {
        row["column_name"]: {
            "drifted": str(row["data_drift"]).lower().startswith("detected"),
            "p_value": round(float(row.get("drift_score", 0.0)), 5),
            "test": row.get("stattest_name"),
        }
        for row in rows
    }
    drifted = sorted(name for name, info in per_column.items() if info["drifted"])
    share = round(len(drifted) / len(per_column), 4) if per_column else 0.0
    dataset_drift = share > drift_share_threshold

    summary = {
        "reference": "training split (70%, patients the model learned from)",
        "current": "held-out test split (unseen patients)",
        "dataset_drift": dataset_drift,
        "drift_share_threshold": drift_share_threshold,
        "share_of_drifted_columns": share,
        "number_of_drifted_columns": len(drifted),
        "columns_checked": len(per_column),
        "drifted_columns": drifted,
        "per_column": per_column,
        "report": html_path.name,
        "interpretation": (
            "Reference vs held-out is the *offline* demonstration of the "
            "monitor. In production the current set would be the incoming "
            "week's discharges; dataset_drift=true is the retrain trigger."
        ),
    }
    summary["recommend_retrain"] = dataset_drift
    (config.ARTIFACT_DIR / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


# ----------------------------------------------------------------------- main


def main() -> int:
    frame = load_clean()
    train, _, test = patient_split(frame)  # validation split unused here
    feature_columns = [c for c in train.columns if c != config.TARGET]

    model, preprocessor, threshold = load_stack()
    y_test = test[config.TARGET].to_numpy()
    probabilities = scores(model, preprocessor, test, feature_columns)
    y_pred = (probabilities >= threshold).astype(int)

    fairness = fairness_report(test, y_test, y_pred)
    fairness["threshold_used"] = threshold
    fairness["test_size"] = int(len(test))
    target = config.ARTIFACT_DIR / "fairness.json"
    target.write_text(json.dumps(fairness, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", target.name)
    for attribute, block in fairness["attributes"].items():
        logger.info(
            "%-6s demographic parity %.4f | equalised odds %.4f",
            attribute, block["demographic_parity_difference"], block["equalized_odds_difference"],
        )

    # Evidently wants explicit column types: numbers numeric, the rest category
    reference = train[feature_columns].copy()
    current = test[feature_columns].copy()
    for column in reference.columns:
        kind = "num" if pd.api.types.is_numeric_dtype(reference[column]) else "cat"
        for table in (reference, current):
            table[column] = table[column].astype("category" if kind == "cat" else "float64")

    drift = drift_report(reference, current)
    logger.info(
        "drift: %s (%s/%s columns)",
        "DETECTED — retrain recommended" if drift["dataset_drift"] else "not detected",
        drift["number_of_drifted_columns"], len(feature_columns),
    )

    print(json.dumps({
        "fairness": str(target.name),
        "drift": drift["dataset_drift"],
        "recommend_retrain": drift["recommend_retrain"],
        "drift_report": drift["report"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
