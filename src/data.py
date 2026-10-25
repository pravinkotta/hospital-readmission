"""Load the raw UCI extract and turn it into a clean modelling frame."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def icd9_group(code: object) -> str:
    """Map a raw ICD-9 code to a clinically meaningful chapter."""
    text = str(code).strip()
    if text in {"", "nan", "?", "None"}:
        return "unknown"
    if text.startswith("250"):  # diabetes mellitus
        return "diabetes"
    try:
        number = int(float(text))
    except ValueError:  # V and E codes
        return "other"
    if number in config.ICD_SINGLES:
        return config.ICD_SINGLES[number]
    if 780 <= number <= 799:  # symptoms/signs not elsewhere classified
        return "other"
    for low, high, name in config.ICD_BANDS:
        if low <= number <= high:
            return name
    return "other"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv = path or config.RAW_CSV
    if not csv.is_file():
        raise FileNotFoundError(
            f"Missing {csv}\n"
            "Download the UCI dataset first:\n"
            "  https://archive.ics.uci.edu/static/public/296/"
            "diabetes+130-us+hospitals+for+years+1999-2008.zip"
        )
    df = pd.read_csv(csv, na_values=["?"], low_memory=False)
    logger.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with a binary target and ready-to-encode features."""
    out = df.copy()

    # ---- target ------------------------------------------------------------
    out[config.TARGET] = (out[config.TARGET].astype(str) == config.POSITIVE_LABEL).astype(int)

    # ---- patient leakage ---------------------------------------------------
    # Drop encounters that end in death/hospice: they can never be readmitted,
    # which would otherwise teach the model "certain discharge = safe".
    before = len(out)
    out = out[~out["discharge_disposition_id"].isin(config.NON_READMITTABLE_DISCHARGE)]
    logger.info("Dropped %s non-readmittable discharges", f"{before - len(out):,}")

    # ---- identifiers / sparse columns --------------------------------------
    # patient_nbr is retained for one more step: splitting by patient so no
    # person's earlier and later encounters land on both sides of the split.
    dropped = [
        c for c in config.DROP_COLS
        if c in out.columns and c != config.ID_COL
    ]
    out = out.drop(columns=dropped, errors="ignore")

    # ---- missing values ----------------------------------------------------
    for col in ("race", "gender", "diag_1", "diag_2", "diag_3"):
        out[col] = out[col].fillna("Unknown")

    # ---- ICD grouping ------------------------------------------------------
    for col in ("diag_1", "diag_2", "diag_3"):
        out[f"{col}_group"] = out[col].map(icd9_group)

    # ---- medication encoding ----------------------------------------------
    for col in config.BINARY_DRUG_COLS:
        if col not in out.columns:
            continue
        out[col] = (out[col].fillna("No") != "No").astype(int)

    out["change"] = (out["change"] == "Ch").astype(int)
    out["diabetesMed"] = (out["diabetesMed"] == "Yes").astype(int)

    # Columns that are constant across the whole dataset add no signal.
    constant = [c for c in config.DRUG_COLS if out[c].nunique() <= 1]
    out = out.drop(columns=constant, errors="ignore")
    if constant:
        logger.info("Dropped %d constant medication columns", len(constant))

    # Every categorical becomes a string so the fitted one-hot encoder and the
    # app agree on levels — integer 1 and string "1" are different categories,
    # and a mismatch would silently encode as all-zeros.
    for col in config.CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str)

    # ---- keep only modelled columns ---------------------------------------
    keep = (
        list(config.NUMERIC_COLS)
        + list(config.CATEGORICAL_COLS)
        + ["age", "diag_1", "diag_2", "diag_3", "insulin"]
    )
    keep = [c for c in dict.fromkeys(keep) if c in out.columns]
    out = out[keep + [config.TARGET, config.ID_COL]]

    # Injected categoricals must exist even if the source dropped them.
    for col in config.CATEGORICAL_COLS:
        if col not in out.columns:
            out[col] = "Unknown"

    logger.info("Clean frame: %s rows x %s features", f"{len(out):,}", out.shape[1] - 2)
    return out


def load_clean(path: Path | None = None) -> pd.DataFrame:
    return clean(load_raw(path))
