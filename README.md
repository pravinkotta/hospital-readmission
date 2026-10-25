# Hospital readmission prediction — ANN + SHAP

Predicting **30-day readmission** for diabetic inpatients with a PyTorch
artificial neural network, explained feature-by-feature with **SHAP** — built as
a case study against an explicit [Role / Context / Task / Constraints spec](RCTC.md).

> **Read the results in one page:** [Hospital_Readmission_ANN_SHAP.html](Hospital_Readmission_ANN_SHAP.html)
> (a rendered copy of the notebook — no Jupyter needed) ·
> source: [notebooks/hospital_readmission_ann_shap.ipynb](notebooks/hospital_readmission_ann_shap.ipynb)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.49-e67e22)
![Dataset](https://img.shields.io/badge/dataset-UCI%20%23296-green)

---

## The problem

Roughly 1 in 5 hospital readmissions within 30 days is considered avoidable, and
US hospitals are financially penalised for them. Knowing at **discharge** who is
likely to come back lets a care team spend its limited follow-up capacity where
it changes outcomes.

The question modelled here:

> Given everything known at discharge, what is the probability this patient is
> readmitted within 30 days — and which factors drove that number?

---

## Results (measured, not claimed)

Held-out **test set**: 14,841 encounters from 10,499 patients the model never
saw during training or threshold selection.

| Metric | Value | Read as |
|---|---:|---|
| ROC-AUC | **0.6754** | ranking quality across all thresholds |
| PR-AUC | **0.2319** | average precision (baseline = 0.1179) |
| F1 @ threshold 0.57 | **0.2862** | threshold tuned on *validation*, not test |
| Precision | 0.2127 | |
| Recall | 0.4377 | |
| Accuracy | 0.7426 | below the 0.8821 majority baseline — and that is fine |

Confusion matrix at the chosen operating point:

| | predicted not readmitted | predicted readmitted |
|---|---:|---:|
| **actual not readmitted** | 10,255 | 2,836 |
| **actual readmitted** | 984 | 766 |

**Honest framing.** Accuracy alone would be misleading on an 11.4% positive
class — predicting "nobody is readmitted" scores 88.2% and is clinically
useless. The model ranks patients meaningfully (AUC 0.675), but this is a
*triage aid*, not a decision-maker: it is nowhere near the accuracy a
clinical deployment would require, and the data is from 1999–2008.

Training: 22 epochs, early-stopped, best validation ROC-AUC **0.6751**, ~40 s on
CPU. 120 input features, 9,857 parameters.

---

## Data

**UCI ML Repository #296 — Diabetes 130-US Hospitals for Years 1999–2008**
(Strack et al., 2014): 101,766 inpatient encounters, 50 columns.

After cleaning: **99,343 encounters / 69,990 patients / 11.39% positive**.

| Decision | Why |
|---|---|
| Target collapsed to `1 = readmitted <30 days` | the clinical question is 30-day readmission |
| 2,423 death & hospice discharges dropped | they cannot be readmitted — keeping them leaks the answer |
| `weight` (96.9% missing), `payer_code`, `medical_specialty` dropped | too sparse to impute honestly |
| ICD-9 codes grouped into 10 clinical chapters | raw codes have thousands of near-empty levels |
| rare ID levels collapsed to top-10 + `Other` | avoids spending columns on 3-occurrence values |

The raw CSV is **not** committed — fetch it with one command (below).

---

## Method

1. **Patient-level split (70/15/15).** The same person is often admitted more
   than once; splitting by encounter would put one admission in training and a
   later one in testing. The notebook asserts the patient sets are disjoint —
   0 overlaps.
2. **One preprocessing artefact.** Impute + standardise numerics, one-hot
   categoricals with `handle_unknown="ignore"`, group rare ID levels — fitted on
   training data only and saved as a single `joblib` file, so training and any
   later consumer can never disagree.
3. **ANN** `120 → 64 → 32 → 1`, ReLU, dropout 0.3, `BCEWithLogitsLoss` with
   `pos_weight` for the class imbalance, Adam, gradient clipping, early stopping
   on validation ROC-AUC.
4. **Threshold tuned on validation** to maximise F1 (0.57 rather than the
   default 0.5), then the test set is evaluated exactly once.

## Explainability (SHAP)

| | |
|---|---|
| Global drivers | `number_inpatient` (prior admissions), discharge destination, `number_diagnoses`, length of stay, circulatory primary diagnosis |
| Per patient | waterfall from baseline logit → model logit, ~10 ms |

**A detail worth knowing before anyone asks:** DeepSHAP is exact for
piecewise-linear graphs. Wrapping the network's output in a sigmoid broke its
own additivity check (max diff **0.25** vs tolerance **0.01**), which would have
produced silently wrong attributions. The explainer therefore targets the
**log-odds output**, where the check passes at every background size — and the
notebook asserts additivity at runtime:

```
additivity check passed | sum(SHAP) + baseline = 3.2050 vs logit = 3.2050
```

So SHAP values in this project are in **log-odds, not probability**.

---

## Repository layout

```
├── RCTC.md                              Role / Context / Task / Constraints spec
├── README.md                            this file
├── Hospital_Readmission_ANN_SHAP.html   rendered notebook (open in any browser)
├── requirements.txt
├── notebooks/
│   └── hospital_readmission_ann_shap.ipynb   the full case study, executed
├── scripts/
│   └── download_data.py                 fetches the UCI dataset
├── src/
│   ├── config.py        columns, ICD bands, hyperparameters, risk bands
│   ├── data.py          load, clean, ICD-9 chapter grouping
│   ├── preprocess.py    ColumnTransformer + rare-level grouping
│   ├── model.py         ReadmissionMLP + the (n,1) logit head SHAP needs
│   ├── explain.py       DeepExplainer wrapper, global + local contributions
│   └── train.py         `python -m src.train` — the whole pipeline
├── artifacts/           trained model, fitted preprocessor, metrics, SHAP output
└── data/                raw dataset (downloaded, not committed)
```

## Reproduce it

```bash
git clone <your-repo-url>
cd hospital-readmission
pip install -r requirements.txt

python scripts/download_data.py   # ~3 MB download, extracts to data/
python -m src.train               # trains + evaluates + explains, ~1 min
```

That regenerates everything in `artifacts/`. The committed values are what a
fresh run produces — the notebook and the command-line pipeline are two views
of the same code and write the same artefacts.

To read the case study instead:

- open [`Hospital_Readmission_ANN_SHAP.html`](Hospital_Readmission_ANN_SHAP.html) in a browser, or
- run `jupyter notebook notebooks/hospital_readmission_ann_shap.ipynb` and execute it top-to-bottom (~1 min, CPU).

## Artefacts

| File | Contents |
|---|---|
| `model.pt` | trained weights (9,857 parameters) |
| `preprocessor.joblib` | fitted feature pipeline — the single source of preprocessing truth |
| `background.pt` | 128-row SHAP background sample |
| `metrics.json` | test metrics, confusion matrix, ROC curve, full training history |
| `global_shap.json` | top-12 features by mean \|SHAP\| |
| `feature_meta.json` | the 120 model input names |
| `ui_options.json` / `data_profile.json` | cohort defaults and dataset profile |

## Limitations

- **AUC 0.675 is moderate** — good enough to rank patients for follow-up, not
  to act on alone.
- **Old data.** 1999–2008 US hospitals; medication practice has moved on.
  External validation on a recent cohort is required before any real use.
- **Dropped features carry signal.** `medical_specialty` (49% missing) and
  `payer_code` (40% missing) were excluded rather than modelled.
- **SHAP explains the model, not causality.** A contribution is what moved
  *this model's* score, not what would change the patient's outcome.
- **Benchmark exercise, not a clinical instrument.** No prospective validation,
  no calibration analysis, no fairness audit across demographic groups.

## Next steps

Calibration curve · gradient-boosted baseline for comparison · hyperparameter
search · external validation · deployment (deliberately out of scope for this
phase — see [RCTC.md](RCTC.md), constraint C8).

## Citation

Dataset: Strack, D. et al. (2014). *Diabetes 130-US Hospitals for Years
1999-2008*. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
