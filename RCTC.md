# Case study spec — Hospital readmission prediction (ANN + SHAP)

This document frames the project as **Role → Context → Task → Constraints**.
Every file in this repository exists to satisfy a line below, and every design
decision in the code traces back to a constraint.

---

## R — Role

**Machine learning engineer on a healthcare analytics team**, producing an
end-to-end case study for mentor review.

The role owns, and must be able to defend:

- data ingestion, cleaning and cohort definition;
- experiment design (splitting, leakage control, class imbalance);
- model implementation, training and evaluation;
- model explainability (global + per-patient) and its limits;
- packaging the work so someone else can reproduce it from a clean clone.

The deliverable is judged not only on metrics, but on whether the method holds
up to questioning — every number shown must be measured, and every design
choice must have a stated reason.

---

## C — Context

**Problem.** 30-day hospital readmission is a proxy for care quality and is
financially penalised under US CMS policy. Predicting it at discharge lets a
care team target transition-of-care follow-up at the patients who need it.

**Data.** UCI ML Repository #296 — *Diabetes 130-US Hospitals for Years
1999–2008* (Strack et al., 2014): 101,766 inpatient encounters, 50 columns,
≈11.4% positive (`readmitted = <30`). Real, public, benchmark-grade —
chosen over synthetic data so the numbers mean something.

**Known data pathologies** (all handled in `src/data.py`):

| Pathology | Consequence if ignored |
|---|---|
| Repeat encounters per patient (`patient_nbr`) | the same person in train *and* test → inflated metrics |
| 2,423 death/hospice discharges | "cannot be readmitted" leaks into the negative class |
| 96.9% / ~40% / ~49% missing `weight`, `payer_code`, `medical_specialty` | unusable as features |
| Raw ICD-9 codes (thousands of levels) | one-hot explosion, near-zero support per code |
| High-cardinality IDs (29 discharge destinations) | hundreds of one-hot columns, most rare |
| ~11% positive rate | accuracy and 0.5 threshold are both meaningless |

**Available environment.** Python 3.10, Windows, CPU only. Installed stack:
`torch 2.14`, `scikit-learn 1.7`, `pandas 2.3`, `numpy 2.2`, `shap 0.49`,
`matplotlib`, `streamlit 1.65`, `joblib`, `altair`. **Not available:**
TensorFlow / Keras, XGBoost, plotly.

**Stakeholder and timing.** Mentor review **today**; repository is a portfolio
artifact on GitHub. Live deployment is deliberately a *later* phase.

---

## T — Task

Deliver a reproducible, honestly measured, GitHub-ready ML repository:

1. **Cohort** — load the raw extract; define a binary 30-day target; exclude
   non-readmittable discharges; group ICD-9 codes into clinical chapters.
2. **Split** — patient-level 70/15/15 train/val/test so no encounter from a
   person appears on both sides of any split.
3. **Preprocess** — impute + standardise numerics, one-hot categoricals with
   `handle_unknown="ignore"`, collapse rare ID levels to top-k + `Other`;
   fit on train only and persist as **one** joblib artefact so the app can
   never drift from training.
4. **Model** — PyTorch ANN `118→64→32→1`, ReLU, dropout 0.3, trained with
   `BCEWithLogitsLoss` + `pos_weight`, Adam, gradient clipping, early stopping
   on validation ROC-AUC.
5. **Operating point** — tune the decision threshold on *validation* to
   maximise F1 (never on the test set), then evaluate once on held-out test.
6. **Evaluation** — ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix,
   learning curve; report alongside the majority-class baseline so the numbers
   can be read honestly.
7. **Explainability** — SHAP DeepExplainer: global feature importance over the
   test set, and a per-patient waterfall for a single prediction; state
   explicitly what scale is being explained.
8. **Packaging** — one-command retraining (`python -m src.train`), artefacts
   under `artifacts/`, a notebook that runs top-to-bottom with markdown steps
   and captured outputs, an HTML export for readers without Jupyter, a README,
   a data downloader (the 19 MB CSV is not committed), and a `.gitignore`.

**Definition of done**

- [ ] `python -m src.train` regenerates every artefact from a clean clone
      (after `scripts/download_data.py`) and the metrics match the README table.
- [ ] Notebook executes end-to-end without manual intervention; all outputs
      visible; markdown explains *why* at each step, not just *what*.
- [ ] HTML export opens standalone in a browser and shows the same outputs.
- [ ] SHAP additivity check passes (no silent wrong attributions).
- [ ] Repository contains no raw dataset, no `__pycache__`, no checkpoints;
      README documents structure, results, and limitations.

---

## C — Constraints

| # | Constraint | How it is enforced |
|---|---|---|
| C1 | Real benchmark data only — no synthetic rows | dataset downloaded from UCI; download script ships in `scripts/` |
| C2 | Measured numbers only — nothing fabricated or hand-written | every figure in README/notebook regenerated from `artifacts/*.json` |
| C3 | Use only the installed stack (no TF/Keras/XGBoost/plotly) | `src/model.py` is pure `torch`; charts use matplotlib/altair |
| C4 | CPU only, full pipeline ≈1 minute | batch 1024, ≤40 epochs, early stopping patience 6 |
| C5 | No leakage of any kind | patient-level split; preprocessor fit on train only; death/hospice excluded; threshold tuned on val |
| C6 | Explanations must be verifiable | DeepExplainer run against **log-odds** output — the sigmoid form failed SHAP's additivity check (max diff 0.25 ≫ tolerance 0.01), so probability-space explanations were rejected rather than shipped |
| C7 | One-command reproducibility | `python -m src.train` writes all artefacts; no notebook required to retrain |
| C8 | **No deployment in this phase** | repo is training + documentation only; hosting is a later, separately agreed step |
| C9 | Same-day delivery | pipeline first, then docs; features beyond scope deferred |
| C10 | Clean GitHub repository | `.gitignore` excludes the raw 19 MB CSV and caches; `artifacts/` (≈120 KB total) is committed on purpose so results are inspectable without retraining; README states structure, measured results and limitations |

---

## Out of scope (agreed)

- Live hosting / URL for the demo (deferred — see C8).
- Hyperparameter search, alternative architectures, calibration.
- Prospective validation on any dataset other than the UCI benchmark.
