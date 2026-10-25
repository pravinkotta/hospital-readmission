"""SHAP explainability: one explainer used for both training-time and live use."""

from __future__ import annotations

import copy
import logging

import numpy as np
import shap
import torch

from .model import ReadmissionMLP, LogitHead

logger = logging.getLogger(__name__)

BACKGROUND_ROWS = 128


def build_background(matrix: np.ndarray, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sample = matrix[rng.choice(matrix.shape[0], size=min(BACKGROUND_ROWS, len(matrix)), replace=False)]
    return torch.tensor(sample, dtype=torch.float32)


def build_explainer(
    model: ReadmissionMLP,
    background: torch.Tensor,
) -> shap.Explainer:
    """DeepExplainer is exact and fast on this network; KernelExplainer is the
    model-agnostic fallback in case a future architecture breaks it.

    The explainer is built on a **deep copy** of the model: SHAP's PyTorch
    backend attaches ``nn.Parameter`` attributes (``x``, ``y``) to activation
    layers to compute gradients, and those would otherwise end up inside
    ``model.state_dict()`` — bloating the checkpoint and making a strict
    ``load_state_dict`` fail with "unexpected keys". The copy shares weights,
    so the attributions are identical.
    """
    head = LogitHead(copy.deepcopy(model))
    try:
        explainer = shap.DeepExplainer(head, background)
        logger.info("Using shap.DeepExplainer (log-odds output)")
        return explainer
    except Exception:  # noqa: BLE001 - fall back rather than fail the pipeline
        logger.exception("DeepExplainer failed, falling back to KernelExplainer")
        background_np = background.numpy()
        return shap.KernelExplainer(
            lambda x: head(torch.tensor(x, dtype=torch.float32)).numpy(),
            shap.kmeans(background_np, 20),
        )


def _squeeze_outputs(values: np.ndarray, n_features: int) -> np.ndarray:
    """Normalise shap's output to ``(rows, features)``.

    It arrives as ``(rows, features, 1)`` or ``(rows, 1, features)`` depending
    on the model's output rank. Both layouts have a singleton axis, and the
    other two axes can *both* equal ``n_features`` in odd cases — so identify
    the output axis by it being 1, never by matching the feature count alone.
    Slicing the wrong axis silently yields a single meaningless number.
    """
    if values.ndim != 3:
        return values
    rows, axis_one, axis_two = values.shape
    if axis_two == 1 and (axis_one == n_features or axis_one != 1):
        return values[:, :, 0]
    if axis_one == 1:
        return values[:, 0, :]
    return values.reshape(rows, -1)


def contributions(explainer, row: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Return [{feature, value, contribution}] for a single patient, biggest movers first."""
    array = np.asarray(row, dtype=np.float32).reshape(1, -1)
    values = explainer.shap_values(torch.from_numpy(array))

    flat = _squeeze_outputs(np.asarray(values, dtype=np.float64), len(feature_names)).reshape(-1)
    if flat.shape[0] != len(feature_names):  # defensive: ragged layouts
        flat = flat.ravel()[: len(feature_names)]

    order = np.argsort(-np.abs(flat))[:10]
    return [
        {
            "feature": feature_names[index],
            "value": float(array.reshape(-1)[index]),
            "contribution": float(flat[index]),
        }
        for index in order
    ]


def global_importance(
    explainer,
    matrix: np.ndarray,
    feature_names: list[str],
    max_rows: int = 500,
    top: int = 12,
) -> list[dict]:
    sample = torch.from_numpy(np.asarray(matrix[:max_rows], dtype=np.float32))
    values = np.asarray(explainer.shap_values(sample), dtype=np.float64)
    values = _squeeze_outputs(values, len(feature_names))
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    return [
        {"feature": feature_names[i], "mean_abs": float(mean_abs[i])}
        for i in order
    ]
