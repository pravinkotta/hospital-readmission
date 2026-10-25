"""Build the interactive HTML report from the trained artefacts.

    python scripts/build_report.py

Reads `artifacts/` (never retraining), re-scores the held-out test set to
derive the per-threshold sweep, lift chart and patient explanations, then
renders a self-contained interactive page — no CDN, no server, works offline.

Output: Hospital_Readmission_ANN_SHAP.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python scripts/build_report.py` from anywhere

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config, explain
from src.data import load_clean
from src.model import ReadmissionMLP
from src.preprocess import load as load_preprocessor
from src.train import patient_split

TEMPLATE = Path(__file__).with_name("report_template.html")
OUTPUT = ROOT / "Hospital_Readmission_ANN_SHAP.html"


def read_json(name: str):
    return json.loads((config.ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def risk_band(probability: float) -> str:
    for limit, label, _action in config.RISK_BANDS:
        if probability < limit:
            return label
    return config.RISK_BANDS[-1][1]


def threshold_sweep(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Confusion counts + derived rates at every cut-off on the test set."""
    thresholds = np.arange(0.05, 0.955, 0.01)
    tp, fp, fn, tn = [], [], [], []
    for cut in thresholds:
        pred = probabilities >= cut
        tp.append(int((pred & (labels == 1)).sum()))
        fp.append(int((pred & (labels == 0)).sum()))
        fn.append(int((~pred & (labels == 1)).sum()))
        tn.append(int((~pred & (labels == 0)).sum()))
    tp, fp, fn, tn = map(np.asarray, (tp, fp, fn, tn))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    accuracy = (tp + tn) / len(labels)
    return {
        "thresholds": [round(float(t), 2) for t in thresholds],
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "tn": tn.tolist(),
        "precision": [round(float(v), 4) for v in precision],
        "recall": [round(float(v), 4) for v in recall],
        "f1": [round(float(v), 4) for v in f1],
        "accuracy": [round(float(v), 4) for v in accuracy],
    }


def lift_table(labels: np.ndarray, probabilities: np.ndarray, n_deciles: int = 10) -> list:
    """Observed readmission rate vs mean predicted risk, by predicted-risk decile."""
    order = np.argsort(probabilities)
    buckets = np.array_split(order, n_deciles)  # lowest risk first
    rows = []
    for position, idx in enumerate(buckets, start=1):
        rows.append(
            {
                "decile": position,  # 1 = lowest predicted risk
                "mean_predicted": round(float(probabilities[idx].mean()), 4),
                "observed": round(float(labels[idx].mean()), 4),
                "patients": int(len(idx)),
            }
        )
    return rows


def pick_patients(matrix, probabilities, values, features, raw_test, n: int = 9) -> list:
    """Deterministic sample: the most at-risk, the median and the least at-risk."""
    order = np.argsort(-probabilities)
    chosen: list[int] = []
    anchors = [0, 1, 2, len(order) // 2 - 1, len(order) // 2, len(order) // 2 + 1,
               len(order) - 3, len(order) - 2, len(order) - 1][:n]
    for anchor in anchors:
        index = int(order[anchor])
        if index not in chosen:
            chosen.append(index)

    labels_by_position = {}
    for rank, index in enumerate(order):
        labels_by_position[index] = rank + 1

    display_cols = [
        "age", "race", "gender", "time_in_hospital", "num_medications",
        "number_inpatient", "number_diagnoses", "insulin", "diag_1_group",
        "discharge_disposition_id",
    ]

    patients = []
    for slot, index in enumerate(chosen, start=1):
        rank = labels_by_position[index]
        total = len(order)
        if rank == 1:
            short = "Most at-risk"
        elif rank in (2, 3):
            short = f"{rank}nd most at-risk" if rank == 2 else "3rd most at-risk"
        elif rank == total:
            short = "Least at-risk"
        elif rank == total - 1:
            short = "2nd least at-risk"
        elif rank == total - 2:
            short = "3rd least at-risk"
        else:
            short = f"Rank {rank:,}"
        top = np.argsort(-np.abs(values[index]))[:10]
        row = raw_test.iloc[index]
        inputs = {col: (row[col].item() if hasattr(row[col], "item") else row[col]) for col in display_cols
                  if col in raw_test.columns}
        patients.append(
            {
                "label": f"Rank {rank} of {len(order)} by predicted risk",
                "short": short,
                "probability": round(float(probabilities[index]), 4),
                "band": risk_band(float(probabilities[index])),
                "inputs": {k: str(v) for k, v in inputs.items()},
                "contributions": [
                    {
                        "feature": features[i],
                        "contribution": round(float(values[index, i]), 4),
                    }
                    for i in top
                ],
            }
        )
        patients[-1]["slot"] = slot
    return patients


def build_payload() -> dict:
    metrics = read_json("metrics.json")
    profile = read_json("data_profile.json")
    global_shap = read_json("global_shap.json")
    meta = read_json("feature_meta.json")

    # --- rebuild the held-out test set exactly as training did --------------
    frame = load_clean()
    _train, _val, test, _groups = patient_split(frame, return_groups=True)
    feature_columns = [c for c in test.columns if c != config.TARGET]
    preprocessor = load_preprocessor(config.ARTIFACT_DIR / "preprocessor.joblib")
    matrix = np.asarray(preprocessor.transform(test[feature_columns]), dtype=np.float32)
    labels = test[config.TARGET].to_numpy()
    raw_test = test.reset_index(drop=True)

    model = ReadmissionMLP(meta["n_inputs"])
    model.load_state_dict(torch.load(config.ARTIFACT_DIR / "model.pt", weights_only=True))
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(matrix)
        probabilities = model.predict_proba(tensor).numpy()

    # --- curves, sweep, lift ------------------------------------------------
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
    # sklearn marks the first ROC threshold as +inf (a cut above the max score);
    # JSON has no infinity, and the number is only used for hover text anyway.
    roc_thresholds = np.where(np.isinf(roc_thresholds), 1.0, roc_thresholds)
    step = max(1, len(fpr) // 300)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probabilities)

    sweep = threshold_sweep(labels, probabilities)
    lift = lift_table(labels, probabilities)

    # --- per-patient SHAP explanations -------------------------------------
    background = torch.load(config.ARTIFACT_DIR / "background.pt", weights_only=True)
    explainer = explain.build_explainer(model, background)
    shap_values = np.asarray(explainer.shap_values(tensor), dtype=np.float64)
    shap_values = explain._squeeze_outputs(shap_values, meta["n_inputs"])
    patients = pick_patients(matrix, probabilities, shap_values, meta["features"], raw_test)

    return {
        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"],
            "test_size": metrics["test_size"],
            "positive_rate": metrics["positive_rate"],
            "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
            "epochs": len(metrics["history"]),
            "best_val_roc_auc": metrics["best_val_roc_auc"],
            "features": meta["n_inputs"],
            "parameters": 9857,
        },
        "history": metrics["history"],
        "roc": {
            "fpr": [round(float(v), 5) for v in fpr[::step]],
            "tpr": [round(float(v), 5) for v in tpr[::step]],
            "thr": [round(float(v), 3) for v in roc_thresholds[::step]],
        },
        "pr": {
            "precision": [round(float(v), 5) for v in precision_curve[::step]],
            "recall": [round(float(v), 5) for v in recall_curve[::step]],
        },
        "sweep": sweep,
        "lift": lift,
        "global_shap": global_shap,
        "profile": profile,
        "patients": patients,
    }


def main() -> int:
    payload = build_payload()
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__REPORT_DATA__", data.replace("</", "<\\/")
    )
    OUTPUT.write_text(html, encoding="utf-8")

    print(f"wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print(f"  sections: overview, cohort, training, performance, explainability, patient, method")
    print(f"  interactive: threshold slider over {len(payload['sweep']['thresholds'])} cut-offs, "
          f"{len(payload['patients'])} selectable patients, {len(payload['lift'])} lift deciles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
