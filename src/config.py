"""Everything dataset- and product-specific lives here.

Source: UCI ML Repository #296 — "Diabetes 130-US Hospitals for Years 1999-2008"
(101,766 inpatient encounters, 50 features, Strack et al. 2014).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RAW_CSV = DATA_DIR / "diabetic_data.csv"

# --------------------------------------------------------------------- target

TARGET = "readmitted"
POSITIVE_LABEL = "<30"  # readmitted within 30 days
NEGATIVE_LABELS = ("NO", ">30")

# ------------------------------------------------------------------- columns

# Identifiers leak the target (repeat encounters) or are >40% empty.
DROP_COLS = (
    "encounter_id",
    "patient_nbr",
    "weight",  # 96.9% missing
    "payer_code",  # ~40% missing
    "medical_specialty",  # ~49% missing
)
ID_COL = "patient_nbr"  # used only to keep one patient entirely in one split

# Encounters where the patient died or went to hospice cannot be readmitted;
# keeping them would let a "cannot be readmitted" flag leak into the negatives.
NON_READMITTABLE_DISCHARGE = (11, 13, 14, 19, 20, 21)

NUMERIC_COLS = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)

DRUG_COLS = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)
INSULIN = "insulin"
# Insulin has real clinical meaning in Up/Down/Steady, the rest are near-constant.
BINARY_DRUG_COLS = tuple(c for c in DRUG_COLS if c != INSULIN)

HIGH_CARDINALITY = ("discharge_disposition_id", "admission_source_id")
RARE_GROUP_TOP_K = 10  # keep the 10 most common values, collapse the rest

CATEGORICAL_COLS = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    INSULIN,
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
) + BINARY_DRUG_COLS

# ---------------------------------------------------------------- ICD-9 grouping
# Strack et al. (2014), Table 1 — raw ICD codes are too sparse to one-hot.

ICD_BANDS = (
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (800, 999, "injury"),
    (710, 739, "musculoskeletal"),
    (580, 629, "genitourinary"),
    (140, 239, "neoplasm"),
)
ICD_SINGLES = {
    785: "circulatory",
    786: "respiratory",
    787: "digestive",
    788: "genitourinary",
}
ICD_GROUPS = (
    "circulatory",
    "respiratory",
    "digestive",
    "diabetes",
    "injury",
    "musculoskeletal",
    "genitourinary",
    "neoplasm",
    "other",
    "unknown",
)

# ------------------------------------------------------------------- splits

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # remainder is the held-out test set
RANDOM_STATE = 42

# --------------------------------------------------------------------- model

HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.3
BATCH_SIZE = 1024
MAX_EPOCHS = 40
PATIENCE = 6  # early stopping on validation ROC-AUC
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Risk bands shown to clinicians
RISK_BANDS = (
    (0.30, "Low", "Routine discharge planning"),
    (0.60, "Moderate", "Consider medication reconciliation"),
    (1.01, "High", "Escalate: transition-of-care follow-up"),
)

# ------------------------------------------------------------------- app UI

APP_TITLE = "Hospital readmission risk"
APP_SUBTITLE = "30-day readmission for diabetic inpatients"

# Provenance recorded in every artefact, so a stray JSON can always be traced.
DATA_SOURCE = "UCI ML Repository #296 — Diabetes 130-US Hospitals for Years 1999-2008"
