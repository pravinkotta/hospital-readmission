"""Experiment tracking + model registry with MLflow (local SQLite backend).

    python -m src.train                 # logs a run as a side effect
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Why SQLite and not the default ``./mlruns`` file store: MLflow 3 put the
filesystem backend in maintenance mode, and a database store is what the
**model registry** needs (registered models, versions, aliases live in tables).
Artifacts still land on disk under ``mlruns/`` — only the metadata is a DB.

Every call here is best-effort: if MLflow is unavailable or the DB is locked,
training continues and ``artifacts/metrics.json`` remains the source of truth
(never the other way round).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)

EXPERIMENT = "hospital-readmission"
REGISTERED_MODEL = "readmission-ann"
CHAMPION_ALIAS = "champion"  # mlflow 3 replaced "stages" with aliases


def _uri() -> str:
    return f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is a nicety, not a requirement
        return None


def log_run(
    model,
    metrics: dict,
    training: dict,
    n_features: int,
) -> str | None:
    """Log params, metrics, history and the model itself; register a version.

    Returns the run id, or ``None`` if tracking failed — the caller logs a
    warning and carries on.
    """
    try:
        import mlflow
        import mlflow.pytorch

        mlflow.set_tracking_uri(_uri())
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name=datetime.now(timezone.utc).strftime("ann-%Y%m%d-%H%M%S")) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "architecture": f"{n_features}->{config.HIDDEN_1}->{config.HIDDEN_2}->1",
                    "dropout": config.DROPOUT,
                    "learning_rate": config.LEARNING_RATE,
                    "weight_decay": config.WEIGHT_DECAY,
                    "batch_size": config.BATCH_SIZE,
                    "max_epochs": config.MAX_EPOCHS,
                    "patience": config.PATIENCE,
                    "loss": "BCEWithLogitsLoss + pos_weight",
                    "n_features": n_features,
                    "split": "patient-level 70/15/15",
                    "data": config.DATA_SOURCE,
                }
            )
            # test-set metrics + the validation-tuned threshold: the numbers
            # someone else would have to re-derive without this run.
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_metrics(
                {"best_val_roc_auc": float(training["best_val_roc_auc"]),
                 "epochs_run": float(len(training["history"]))}
            )
            mlflow.log_dict(training["history"], "training_history.json")

            input_example = np.zeros((1, n_features), dtype=np.float32)
            mlflow.pytorch.log_model(model, name="model", input_example=input_example)

            commit = _git_commit()
            if commit:
                mlflow.set_tag("git_commit", commit)
            mlflow.set_tag("stage", "training")

            # ---- registry: every run becomes a version, this run is "champion"
            client = mlflow.MlflowClient()
            result = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL)
            client.set_registered_model_alias(
                REGISTERED_MODEL, CHAMPION_ALIAS, result.version
            )
            logger.info(
                "mlflow run %s | registered %s v%s aliased '%s'",
                run_id, REGISTERED_MODEL, result.version, CHAMPION_ALIAS,
            )
            return run_id

    except Exception:  # noqa: BLE001 - tracking must never break a training run
        logger.warning("MLflow logging skipped", exc_info=True)
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
