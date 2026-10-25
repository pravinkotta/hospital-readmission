"""Feature pipeline.

Fitted once during training and saved as a single artefact so the app can never
drift from the training code — the same object transforms both.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class RareGrouper(BaseEstimator, TransformerMixin):
    """Collapse a column to its top-k values plus ``Other``.

    One-hot encoding 29 discharge destinations would spend 29 columns on values
    that appear only a handful of times each — so the frequent ones are kept
    and the long tail becomes a single level.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        # Sorted lists, not sets: set ordering depends on PYTHONHASHSEED, which
        # would make the pickled preprocessor differ byte-for-byte between runs.
        self.keep_: list[list[str]] = []

    @staticmethod
    def _as_strings(array) -> np.ndarray:
        frame = pd.DataFrame(array).astype(str)
        frame = frame.replace({"nan": "Missing", "None": "Missing", "?": "Missing", "": "Missing"})
        return frame.to_numpy(dtype=object)

    def fit(self, array, y=None):  # noqa: ARG002 - sklearn API
        values = self._as_strings(array)
        self.keep_ = []
        for column in range(values.shape[1]):
            levels, counts = np.unique(values[:, column], return_counts=True)
            order = np.argsort(-counts)
            self.keep_.append(sorted(levels[order][: self.top_k].tolist()))
        return self

    def transform(self, array):
        values = self._as_strings(array).copy()
        for column, keep in enumerate(self.keep_):
            current = values[:, column]
            values[:, column] = np.where(np.isin(current, keep), current, "Other")
        return values

    def get_feature_names_out(self, input_features=None):
        """The grouping is column-preserving, so names pass straight through."""
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.keep_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    rare_cols = [c for c in config.HIGH_CARDINALITY if c in frame.columns]
    cat_cols = [
        c for c in config.CATEGORICAL_COLS
        if c in frame.columns and c not in rare_cols
    ]
    num_cols = [c for c in config.NUMERIC_COLS if c in frame.columns]

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        ("cat", categorical, cat_cols),
    ]
    if rare_cols:
        transformers.append(
            (
                "rare",
                Pipeline(
                    [
                        ("group", RareGrouper(top_k=config.RARE_GROUP_TOP_K)),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="if_binary",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                rare_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save(preprocessor: ColumnTransformer, path) -> None:
    joblib.dump(preprocessor, path, compress=3)


def load(path) -> ColumnTransformer:
    return joblib.load(path)
