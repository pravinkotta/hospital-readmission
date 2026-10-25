"""End-to-end training run.

    python -m src.train

Produces everything the app needs in ``artifacts/``:
model weights, the fitted preprocessor, metrics, SHAP explanations and the
UI option lists — so the app never recomputes training logic.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from . import config, explain
from .data import load_clean
from .model import ReadmissionMLP
from .preprocess import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
)
logger = logging.getLogger("train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- helpers


def patient_split(
    frame: pd.DataFrame, return_groups: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple:
    """Split on patient, not encounter.

    The same person is often admitted several times; splitting by encounter
    would let one admission appear in train and a later one in test, flattering
    every metric.

    With ``return_groups=True`` the patient-id sets are returned as well, so a
    caller can assert the partitions are disjoint.
    """
    patients = frame[config.ID_COL].drop_duplicates().to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * config.TRAIN_FRACTION)
    n_val = int(n * config.VAL_FRACTION)

    groups = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train : n_train + n_val]),
        "test": set(patients[n_train + n_val :]),
    }
    splits = [
        frame[frame[config.ID_COL].isin(groups[name])].drop(columns=config.ID_COL)
        for name in ("train", "val", "test")
    ]
    logger.info(
        "Patient split: %d / %d / %d encounters (%d / %d / %d patients)",
        *(len(part) for part in splits),
        *(len(groups[name]) for name in ("train", "val", "test")),
    )
    if return_groups:
        return splits[0], splits[1], splits[2], groups
    return splits  # type: ignore[return-value]


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    stride = max(1, len(fpr) // 80)
    return {
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 4),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "threshold": round(float(threshold), 3),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "roc_curve": {
            "fpr": [round(float(v), 5) for v in fpr[::stride]],
            "tpr": [round(float(v), 5) for v in tpr[::stride]],
        },
    }


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Pick the operating point on validation data that maximises F1.

    The dataset is ~11% positive, so 0.5 is rarely the right cut-off.
    """
    candidates = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probabilities >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def predict(model: ReadmissionMLP, matrix: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model.predict_proba(torch.tensor(matrix, dtype=torch.float32)).numpy()


# ----------------------------------------------------------------------- train


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[ReadmissionMLP, dict]:
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    model = ReadmissionMLP(x_train.shape[1]).to(DEVICE)
    positives = max(int(y_train.sum()), 1)
    negatives = len(y_train) - positives
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32)

    history: list[dict] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_auc, stale = 0.0, 0
    started = time.perf_counter()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(features)

        val_prob = predict(model, x_val)
        val_auc = float(roc_auc_score(y_val, val_prob))
        train_loss = running / len(x_train)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_roc_auc": round(val_auc, 4)})
        logger.info("epoch %02d  loss %.4f  val ROC-AUC %.4f", epoch, train_loss, val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc, stale = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.PATIENCE:
                logger.info("early stopping at epoch %d (best %.4f)", epoch, best_auc)
                break

    model.load_state_dict(best_state)
    logger.info("trained %d epochs in %.1fs on %s", len(history), time.perf_counter() - started, DEVICE)
    return model, {"history": history, "best_val_roc_auc": round(best_auc, 4)}


# ------------------------------------------------------------------- artefacts


def dump(name: str, payload: object) -> None:
    path = config.ARTIFACT_DIR / name
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj),
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)


def build_ui_options(frame: pd.DataFrame) -> dict:
    categorical = [
        "age", "race", "gender", "admission_type_id",
        "insulin", "A1Cresult", "max_glu_serum", "diag_1_group",
    ]
    options = {}
    for column in categorical:
        if column in frame.columns:
            values = frame[column].astype(str).value_counts()
            options[column] = list(values.index)

    ranges = {}
    for column in config.NUMERIC_COLS:
        series = frame[column].dropna()
        ranges[column] = [int(series.min()), int(series.max())]

    defaults = {
        column: (frame[column].mode()[0] if not frame[column].mode().empty else 0)
        for column in list(config.NUMERIC_COLS) + list(config.CATEGORICAL_COLS)
        if column in frame.columns
    }
    return {"categorical": options, "ranges": ranges, "defaults": defaults}


def build_profile(frame: pd.DataFrame) -> dict:
    by_age = (
        frame.groupby("age")[config.TARGET]
        .agg(rate="mean", count="size")
        .reset_index()
        .to_dict("records")
    )
    by_prior = (
        frame.assign(bucket=frame["number_inpatient"].clip(upper=5))
        .groupby("bucket")[config.TARGET]
        .agg(rate="mean", count="size")
        .reset_index()
        .to_dict("records")
    )
    return {
        "rows": int(len(frame)),
        "positive_rate": round(float(frame[config.TARGET].mean()), 4),
        "by_age": by_age,
        "by_prior_inpatient": by_prior,
        "length_of_stay_mean": round(float(frame["time_in_hospital"].mean()), 2),
        "prior_visits_mean": round(float(frame["number_inpatient"].mean()), 2),
    }


# ----------------------------------------------------------------------- main


def main() -> int:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = load_clean()
    train, val, test = patient_split(frame)

    feature_columns = [c for c in train.columns if c != config.TARGET]

    preprocessor = build_preprocessor(train)
    x_train = preprocessor.fit_transform(train[feature_columns])
    x_val = preprocessor.transform(val[feature_columns])
    x_test = preprocessor.transform(test[feature_columns])

    # Column names come from the fitted pipeline, never hardcoded — they must
    # match the training matrix exactly for SHAP labels to be correct.
    feature_names = list(preprocessor.get_feature_names_out())
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = train[config.TARGET].to_numpy()
    y_val = val[config.TARGET].to_numpy()
    y_test = test[config.TARGET].to_numpy()
    logger.info("feature matrix: %d inputs", x_train.shape[1])

    model, training = fit_model(x_train, y_train, x_val, y_val)

    val_prob = predict(model, x_val)
    threshold = best_threshold(y_val, val_prob)
    metrics = evaluate(y_test, predict(model, x_test), threshold)
    # Context keys are added here — before the training block — so the JSON key
    # order matches what the notebook writes; otherwise every run flips the file
    # for no reason and pollutes version control.
    metrics["test_size"] = int(len(test))
    metrics["positive_rate"] = round(float(y_test.mean()), 4)
    metrics["majority_baseline_accuracy"] = round(float(max(y_test.mean(), 1 - y_test.mean())), 4)
    metrics.update(training)
    logger.info(
        "test ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f @ %.2f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["f1"], threshold,
    )

    # ---- persist the essentials first, so a SHAP failure never costs a run --
    from .preprocess import save as save_preprocessor

    torch.save(model.state_dict(), config.ARTIFACT_DIR / "model.pt")
    save_preprocessor(preprocessor, config.ARTIFACT_DIR / "preprocessor.joblib")
    dump("metrics.json", metrics)
    dump("feature_meta.json", {"features": feature_names, "n_inputs": len(feature_names)})
    dump("ui_options.json", build_ui_options(train))

    # ---- explainability ----------------------------------------------------
    background = explain.build_background(x_train)
    explainer = explain.build_explainer(model, background)
    global_shap = explain.global_importance(explainer, x_test, feature_names)

    # ---- persist -----------------------------------------------------------
    torch.save(background, config.ARTIFACT_DIR / "background.pt")
    dump("global_shap.json", global_shap)
    dump(
        "data_profile.json",
        {
            **build_profile(frame),
            "source": config.DATA_SOURCE,
            "rows_modelled": int(len(frame)),
            "patients": int(frame[config.ID_COL].nunique()),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        },
    )

    logger.info("done in %.1fs", time.perf_counter() - started)
    print(
        json.dumps(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "features": len(feature_names),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
