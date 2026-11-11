"""XGBoost baseline — the gradient-boosting comparison for the ANN.

    python -m src.baseline

The mentor's Case A question: *does the ANN actually beat gradient boosting?*
This answers it honestly on equal footing: same cleaned data, same
patient-level split, same fitted preprocessor, same validation-set threshold
tuning and the same ``evaluate`` metrics as the ANN — only the estimator
changes. The result is written to ``artifacts/baseline.json`` and compared
against ``artifacts/metrics.json`` (the ANN) in the README.
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import xgboost
from xgboost import XGBClassifier

from . import config
from .data import load_clean
from .preprocess import load as load_preprocessor
from .train import best_threshold, evaluate, patient_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
)
logger = logging.getLogger("baseline")

PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}


def verdict(ann: dict, xgb: dict) -> str:
    """One honest sentence — no spinning a near-tie either way."""
    delta = ann["roc_auc"] - xgb["roc_auc"]
    if delta > 0.01:
        return f"ANN beats XGBoost by {delta:.4f} ROC-AUC on this dataset"
    if delta < -0.01:
        return f"XGBoost beats the ANN by {-delta:.4f} ROC-AUC on this dataset"
    return (
        f"statistical tie (delta {delta:+.4f} ROC-AUC, within 0.01): on a "
        "~100K-row tabular dataset the ANN does not outperform gradient "
        "boosting — the ANN is kept for its SHAP explainability path"
    )


def main() -> int:
    started = time.perf_counter()
    frame = load_clean()
    train, val, test = patient_split(frame)
    feature_columns = [c for c in train.columns if c != config.TARGET]

    # Same fitted preprocessor as the ANN — feature parity, no refit.
    preprocessor = load_preprocessor(config.ARTIFACT_DIR / "preprocessor.joblib")
    x_train = np.asarray(preprocessor.transform(train[feature_columns]), dtype=np.float32)
    x_val = np.asarray(preprocessor.transform(val[feature_columns]), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(test[feature_columns]), dtype=np.float32)
    y_train = train[config.TARGET].to_numpy()
    y_val = val[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()

    positives = max(int(y_train.sum()), 1)
    scale_pos_weight = float(len(y_train) - positives) / positives

    model = XGBClassifier(
        **PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=config.RANDOM_STATE,
        n_jobs=4,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold = best_threshold(y_val, val_prob)
    test_metrics = evaluate(y_test, model.predict_proba(x_test)[:, 1], threshold)

    ann = json.loads((config.ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    comparison = {
        key: {"ann": ann[key], "xgboost": test_metrics[key]}
        for key in ("roc_auc", "pr_auc", "recall", "precision", "f1")
    }
    payload = {
        "model": "xgboost",
        "library": f"xgboost {xgboost.__version__}",
        "purpose": "baseline: does the ANN beat gradient boosting?",
        "data_parity": {
            "split": "same patient-level split as the ANN",
            "preprocessor": "artifacts/preprocessor.joblib (fitted on train, shared)",
            "threshold_tuning": "same F1-maximising sweep on the validation split",
        },
        "params": {**PARAMS, "scale_pos_weight": round(scale_pos_weight, 3)},
        "best_iteration": int(model.best_iteration) + 1,
        "test": test_metrics,
        "comparison": comparison,
        "roc_auc_delta_ann_minus_xgb": round(ann["roc_auc"] - test_metrics["roc_auc"], 4),
        "verdict": verdict(ann, test_metrics),
    }

    path = config.ARTIFACT_DIR / "baseline.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    print(
        json.dumps(
            {
                "xgboost_roc_auc": test_metrics["roc_auc"],
                "ann_roc_auc": ann["roc_auc"],
                "delta_ann_minus_xgb": payload["roc_auc_delta_ann_minus_xgb"],
                "best_iteration": payload["best_iteration"],
                "seconds": round(time.perf_counter() - started, 1),
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
