# Hospital readmission prediction — ANN + SHAP

Predicting **30-day readmission** for diabetic inpatients with a PyTorch
artificial neural network, explained feature-by-feature with **SHAP**, then
written up as a clinician-facing discharge alert by an LLM — built as
a case study against an explicit [Role / Context / Task / Constraints spec](RCTC.md).

> **Read the results interactively:** [Hospital_Readmission_ANN_SHAP.html](Hospital_Readmission_ANN_SHAP.html)
> — a self-contained dashboard (drag the decision threshold, switch patients,
> hover any chart). No server, no CDN, works offline.
> Linear alternative: [notebook as HTML](notebooks/hospital_readmission_ann_shap_output.html) ·
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

## GenAI discharge alert

The last stage of the pipeline — `ANN → SHAP → LLM` — turns a prediction into
something a discharge team can actually use:

**This stage always calls the OpenAI API** — it requires an OpenAI key and has
no template fallback. The key is read from `OPENAI_API_KEY`, or from a `.env`
file in the project root (either `OPENAI_API_KEY=sk-...` or a bare `sk-...`;
`.env` is gitignored — never commit it). A missing key stops the run before any
data is loaded; a failed API call stops it with the API's own error. Nothing
that no LLM wrote is ever saved as an alert.

```bash
echo "OPENAI_API_KEY=sk-..." > .env   # once, gitignored
python -m src.alert             # 3 highest-risk held-out patients
python -m src.alert --n 1       # just the top one
python -m src.alert --patient 12345678 --patient 87654321
python -m src.alert --model gpt-4o   # different OpenAI model (env OPENAI_MODEL also works)
```

For each patient `src/alert.py` re-scores the **same held-out test split** used
for the metrics above, computes that patient's SHAP drivers, writes an
encounter fact sheet (demographics, prior visits, diagnoses, discharge
destination read from `IDS_mapping.csv`), and asks an OpenAI model
(`gpt-4o-mini` by default) to draft a four-section
alert: **risk summary · why the model flagged this patient · suggested
discharge actions · caveat**.

Design rules, so the stage survives questioning:

- **The LLM never does arithmetic or lookup.** Probability, threshold, band and
  every fact in the prompt come from the artefacts and the dataset — the model
  only words them. The system prompt forbids inventing anything not in the input.
- **SHAP contributions are labelled as log-odds** and framed as reasons the
  *model* flagged the patient, not causes of readmission.
- **Fail loudly, never silently.** Missing key → immediate non-zero exit before
  any data loads; API error → the API's own error, non-zero exit, and nothing
  saved. `alerts.json` is written only after every selected alert succeeded, so
  a failed run can never leave an index pointing at half a result. It records
  `source: openai:<model>` and the token usage for every alert, so each file
  traces back to the call that produced it.

Outputs: `artifacts/alerts/alert_patient_<id>.md` per patient plus
`artifacts/alerts.json` (risk, drivers, facts, model, token usage, source).

---

## Serving — FastAPI

```bash
uvicorn src.api:app --port 8000        # docs at http://localhost:8000/docs
```

| Endpoint | Auth | Returns |
|---|---|---|
| `GET /health` | — | status, threshold, test ROC-AUC, whether `/alert` is enabled |
| `GET /cohort/sample` | — | a complete cleaned encounter (highest-risk patient by default) |
| `POST /predict` | — | `probability`, `risk_band`, **`requires_human_review`**, optional `explain: true` → SHAP drivers |
| `POST /alert` | `X-API-Key` | LLM discharge summary + token usage |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"patient_nbr": 88479036, "explain": true}'
```

Two decisions worth defending:

- **`/predict` is open, `/alert` is not** — scoring costs ~1 ms of CPU, an
  alert spends *your OpenAI credit*. The gate uses constant-time comparison and
  the endpoint returns **503 rather than running unconfigured**.
- **`requires_human_review`** on every response is the EU AI Act Art. 14
  human-oversight gate: nothing in this repo acts on a patient, it scores and
  drafts — the decision stays with a human.

Tests: `python -m pytest tests -q` — **17 tests**, covering the API contract,
the README metrics, determinism, auth refusal and the governance artefacts.
CI (`.github/workflows/ci.yml`) re-trains from the raw CSV and runs them on
every push.

## Experiment tracking — MLflow

`python -m src.train` logs every run (params, test metrics, training history,
model artefact, git commit) to a local **SQLite** backend — MLflow 3 put the
old `./mlruns` file store in maintenance mode, and a database store is what the
**model registry** needs. Each run registers a version of `readmission-ann`
and tags the newest one with the `champion` alias.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Tracking is a *side channel*: `artifacts/metrics.json` is untouched by it, so
the notebook and the CLI still write byte-identical artefacts.

## Governance — fairness, drift, EU AI Act

```bash
python -m src.audit     # fairness.json + drift_report.html + drift_summary.json
```

Measured on the held-out test set at threshold 0.57:

| Attribute | Demographic parity Δ | Equalised odds Δ |
|---|---:|---:|
| race | 0.1272 | 0.3500 |
| gender | 0.0203 | 0.0203 |
| age | **0.2943** | **0.5556** |

**Age is the largest disparity** — a real finding, recorded with its caveat
(observations about this model, not proof of discrimination; small groups carry
wide uncertainty). The Evidently drift monitor writes a machine-readable
`recommend_retrain` flag — the `Drift Monitor → Retrain Trigger` edge — which
is recorded, never executed (automatic retraining is what Art. 14 exists to
prevent).

[**GOVERNANCE.md**](GOVERNANCE.md) maps EU AI Act Articles 10/11/13/14/15 to
the implementing files, and explains why this project uses **SHAP only**: LIME
is for image/text surrogates and Grad-CAM requires a CNN — neither has a
job on a tabular MLP, and saying so is part of the answer.

## Docker

```bash
docker build -t readmission-api .
docker run --rm -p 8000:8000 --env-file .env readmission-api
```

CPU-only torch wheel — measured image size **3.3 GB** (the full dependency
stack: torch, MLflow, Evidently, SHAP), `.env` injected at run time —
`.dockerignore` keeps the secret out of every image layer. Verified
end-to-end: `/health`, `/predict` and a real authenticated `/alert` call all
answer from inside the container.

## Deploy — EC2

Full walkthrough in [**DEPLOY.md**](DEPLOY.md) — executed against a live
`t3.micro` running **Amazon Linux 2023**: security group (22 + 8000) → Docker
via `dnf` → 2 GB swap → `docker run --restart unless-stopped`, plus the billing,
key-rotation and Windows `.pem` ACL notes. Stop the instance between sessions.

---

## Repository layout

```
├── RCTC.md                              Role / Context / Task / Constraints spec
├── README.md                            this file
├── GOVERNANCE.md                        EU AI Act mapping + fairness/XAI rationale
├── DEPLOY.md                            EC2 deployment walkthrough
├── Dockerfile / .dockerignore           service image (API + artefacts)
├── .github/workflows/ci.yml             CI: retrain → audit → tests → metric check
├── Hospital_Readmission_ANN_SHAP.html   interactive report (open in any browser)
├── requirements.txt
├── notebooks/
│   ├── hospital_readmission_ann_shap.ipynb   the full case study, executed
│   └── hospital_readmission_ann_shap_output.html  notebook rendered linearly
├── scripts/
│   ├── download_data.py                 fetches the UCI dataset
│   ├── build_report.py                  rebuilds the interactive report
│   └── report_template.html             its HTML/CSS/JS template (no CDN)
├── src/
│   ├── config.py        columns, ICD bands, hyperparameters, risk bands
│   ├── data.py          load, clean, ICD-9 chapter grouping
│   ├── preprocess.py    ColumnTransformer + rare-level grouping
│   ├── model.py         ReadmissionMLP + the (n,1) logit head SHAP needs
│   ├── explain.py       DeepExplainer wrapper, global + local contributions
│   ├── alert.py         `python -m src.alert` — SHAP + facts → LLM discharge alert
│   ├── api.py           FastAPI service (/health /cohort/sample /predict /alert)
│   ├── tracking.py      MLflow run + model registry (SQLite backend)
│   ├── audit.py         `python -m src.audit` — Fairlearn + Evidently
│   └── train.py         `python -m src.train` — the whole pipeline
├── tests/               17 tests: API contract, metrics, explainability, governance
├── artifacts/           trained model, fitted preprocessor, metrics, SHAP output,
│                        discharge alerts, fairness + drift summaries
└── data/                raw dataset (downloaded, not committed)
```

## Reproduce it

```bash
git clone <your-repo-url>
cd hospital-readmission
pip install -r requirements.txt

python scripts/download_data.py   # ~3 MB download, extracts to data/
python -m src.train               # trains + evaluates + explains + logs to MLflow, ~1.5 min
python -m src.audit               # fairness + drift artefacts
python -m src.alert               # SHAP-grounded discharge alerts (needs OPENAI_API_KEY)
python -m pytest tests -q         # 17 tests
python scripts/build_report.py    # rebuilds the interactive HTML report

uvicorn src.api:app --port 8000   # serve the API (docs at /docs)
```

That regenerates everything in `artifacts/`, and the report is derived from
those artefacts only — it never re-trains. The committed values are what a
fresh run produces: the notebook and the command-line pipeline are two views
of the same code and write the same artefacts.

To read the case study instead:

- open [`Hospital_Readmission_ANN_SHAP.html`](Hospital_Readmission_ANN_SHAP.html) in a browser —
  the interactive report: threshold explorer over 91 cut-offs, per-patient SHAP
  explainer, hoverable ROC/PR/lift charts;
- open [`notebooks/hospital_readmission_ann_shap_output.html`](notebooks/hospital_readmission_ann_shap_output.html) for the
  notebook rendered top-to-bottom, code and outputs, or
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
| `alerts.json` + `alerts/` | generated discharge alerts: risk, SHAP drivers, encounter facts, which model wrote each one |
| `fairness.json` | per-group rates + Fairlearn demographic-parity / equalised-odds differences |
| `drift_summary.json` | Evidently drift verdict per column + `recommend_retrain` flag |
| `drift_report.html` | the full Evidently report — regenerable (`python -m src.audit`), too big to commit |

## Limitations

- **AUC 0.675 is moderate** — good enough to rank patients for follow-up, not
  to act on alone.
- **Old data.** 1999–2008 US hospitals; medication practice has moved on.
  External validation on a recent cohort is required before any real use.
- **Dropped features carry signal.** `medical_specialty` (49% missing) and
  `payer_code` (40% missing) were excluded rather than modelled.
- **SHAP explains the model, not causality.** A contribution is what moved
  *this model's* score, not what would change the patient's outcome.
- **Benchmark exercise, not a clinical instrument.** No prospective validation
  and no calibration analysis. A fairness audit now exists (`fairness.json`),
  but it covers three attributes on one dataset — a real system needs
  intersectional analysis and a documented mitigation process.

## Next steps

Calibration curve · gradient-boosted baseline for comparison · TLS/ALB in
front of the EC2 demo · scheduled drift job (cron) feeding the retrain gate ·
external validation on a recent cohort · intersectional fairness analysis.
Deployment was originally out of scope (RCTC constraint C8) and was added in
phase 2 by the MLOps amendment — see [RCTC.md](RCTC.md).

## Citation

Dataset: Strack, D. et al. (2014). *Diabetes 130-US Hospitals for Years
1999-2008*. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
