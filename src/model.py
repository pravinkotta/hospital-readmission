"""The ANN used for readmission risk."""

from __future__ import annotations

import torch
from torch import nn

from . import config


class ReadmissionMLP(nn.Module):
    """Two hidden layers, dropout, single logit output.

    Returns raw logits so it can be paired with ``BCEWithLogitsLoss`` (which is
    numerically stabler than sigmoid-then-BCE).
    """

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, config.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_1, config.HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(inputs))


class LogitHead(nn.Module):
    """Logits reshaped to ``(n, 1)`` — the layout shap's PyTorch backend needs.

    We explain **log-odds, not probability**. DeepSHAP is exact only for
    piecewise-linear graphs; bolting a sigmoid on top broke its additivity
    guarantee on this model by ~0.25, which would have shipped quietly wrong
    attributions. Log-odds keeps the explanations trustworthy.
    """

    def __init__(self, model: ReadmissionMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).unsqueeze(-1)
