# Hospital Readmission Risk — 30-Day Readmission Prediction with SHAP, Fairness and RAG-Grounded LLM Alerts

**AI Career Compass · Capstone (Healthcare track) · live at http://readmission.duckdns.org:8000 · github.com/pravinkotta/hospital-readmission**

---

## Problem

About 1 in 5 30-day readmissions is considered avoidable, and US hospitals are penalised for them (up to 3% of Medicare payments); one missed readmission costs an estimated **$14–25K**. Knowing **at discharge** who is likely to come back lets a care team spend limited follow-up capacity where it changes outcomes. Because a false alarm costs ~nothing while a miss costs five figures, **accuracy is the wrong headline metric** — the design decision is a recall-weighted operating threshold.

## Data & method

- **UCI #296**, Diabetes 130-US Hospitals 1999–2008: 101,766 encounters → **99,343 cleaned** (11.39% positive); death/hospice discharges dropped, sparse columns dropped, ICD-9 grouped into 10 chapters.
- **Patient-level 70/15/15 split** (never encounter-level) — no patient crosses splits; test set 14,841 encounters / 10,499 patients evaluated once.
- **ANN** `120→64→32→1`, dropout 0.3, class-weighted BCE, early stopping; **threshold tuned on validation** (0.57, not 0.5).
- One fitted preprocessor artefact shared by training, API and baseline — training/serving skew impossible by construction.

## Results (held-out test set)

| Metric | ANN | XGBoost baseline |
|---|---:|---:|
| ROC-AUC | 0.6754 | 0.6685 |
| PR-AUC (baseline 0.1179) | 0.2319 | 0.2326 |
| Recall @ tuned threshold | 0.4377 (was 0.35 @ 0.5) | 0.4377 |
| F1 | 0.2862 | 0.2861 |

**Honest verdict:** statistical tie — the ANN does not outperform gradient boosting on 100K tabular rows; it is kept for its **SHAP explainability path**. Accuracy 0.74 sits below the 0.88 majority baseline *by design* (an 11% positive class makes accuracy meaningless).

## GenAI layer — RAG, not vibes

`ANN → SHAP → RAG → LLM`: AHRQ Re-Engineered Discharge toolkit (4 documents) chunked **512 tokens / 50 overlap**, embedded with `text-embedding-3-small`, indexed in **Chroma** (38 chunks, cosine); the top-**5** chunks retrieved per patient are handed to `gpt-4o-mini` as `[S1]…[S5]`. The system prompt forces **five sections** — risk summary · model reasons · suggested actions (each citing a chunk) · **Sources with URLs** · caveat — and requires *"insufficient guidance"* when no chunk supports a suggestion. **Fail-loud:** missing API key or missing index stops the run with instructions, before any OpenAI call.

## Governance, serving, MLOps

- **Fairlearn** audit: largest disparity = age (Δ 0.294 DP / 0.556 EO) — recorded with caveats; **Evidently** PSI drift writes a `recommend_retrain` flag that is *recorded, never auto-executed* (EU AI Act Art. 14).
- **GOVERNANCE.md** maps EU AI Act Art. 10/11/13/14/15 to files; every `/predict` response carries `requires_human_review`.
- **FastAPI** (`/health` `/cohort/sample` `/predict` `/alert` with `X-API-Key`, constant-time compare) → **Docker** → **EC2 t3.micro** with `--restart unless-stopped` and a self-syncing **DuckDNS** name; **28 offline tests** in GitHub Actions CI re-train and verify metrics on every push.

**Limitations (stated, not hidden):** AUC 0.675 = triage aid, not a decision-maker; data is 1999–2008; no prospective validation; SHAP explains the model, not causality.
