# Hospital Readmission Risk

## 30-day readmission prediction — ANN + SHAP + fair + RAG-grounded LLM alerts

- **AI Career Compass · Capstone · Healthcare track**
- Dataset: UCI #296 — 101,766 hospital encounters
- **Live API:** http://readmission.duckdns.org:8000
- **Code:** github.com/pravinkotta/hospital-readmission

---

# The problem — and why accuracy is wrong

## Misses cost five figures, false alarms cost nothing

- ~**1 in 5** 30-day readmissions considered avoidable
- HRRP penalty: up to **3% of Medicare payments** withheld
- One missed readmission: **$14–25K** + patient harm
- Positive class = **11.4%** → "nobody is readmitted" scores **88.2% accuracy** and is clinically useless

**Design consequence:** choose the operating threshold for **recall-weighted F1 on validation**, not accuracy at 0.5

---

# Data & leakage controls

## 101,766 → 99,343 encounters, cleaned with reasons

| Decision | Why |
|---|---|
| Target = `readmitted <30` | the clinical question |
| −2,423 death/hospice discharges | they cannot be readmitted (leakage) |
| drop `weight` 96.9% missing etc. | too sparse to impute honestly |
| ICD-9 → 10 chapters | thousands of near-empty levels |

- **Patient-level 70/15/15 split** — same person never crosses splits
- One fitted `preprocessor.joblib` shared by train / API / baseline → **no training-serving skew by construction**

---

# Model & threshold

## ANN 120 → 64 → 32 → 1 (9,857 parameters)

- Class-weighted `BCEWithLogitsLoss`, dropout 0.3, early stopping (22 epochs, ~40 s CPU)
- **Threshold tuned on validation: 0.57** → recall **0.35 → 0.44**, test set touched once
- Test: **ROC-AUC 0.6754 · PR-AUC 0.2319** (vs 0.1179 baseline) · F1 0.2862

| | predicted not-readmitted | predicted readmitted |
|---|---:|---:|
| actual not-readmitted | 10,255 | 2,836 |
| actual readmitted | 984 | **766 caught** |

- **Read as:** a *triage aid* ranking the 14.8k test patients — not a clinical decision-maker

---

# Does the ANN beat gradient boosting?

## Measured on identical footing — `python -m src.baseline`

| Metric | ANN | XGBoost | Δ |
|---|---:|---:|---:|
| ROC-AUC | **0.6754** | 0.6685 | +0.0069 |
| PR-AUC | 0.2319 | **0.2326** | −0.0007 |
| Recall | 0.4377 | 0.4377 | 0.0000 |
| F1 | 0.2862 | 0.2861 | +0.0001 |

- Same split · same preprocessor · same validation-tuned threshold · same metrics code
- **Verdict: statistical tie** — the ANN does not win on ~100K tabular rows
- **Why keep the ANN?** the SHAP explainability path — and saying so honestly *is* the interview answer

---

# Explainability — SHAP that actually adds up

## Global drivers → per-patient waterfall → log-odds honesty

- **Global:** prior inpatient visits · discharge destination · diagnoses count · length of stay · circulatory diagnosis
- **Per patient:** waterfall baseline logit → model logit, ~10 ms, returned by the API
- **The gotcha:** wrapping output in a sigmoid broke DeepSHAP's additivity check (diff 0.25 vs 0.01) → silently wrong attributions. The explainer targets **log-odds**, asserted at runtime:

> additivity check passed | sum(SHAP) + baseline = 3.2050 vs logit = 3.2050

- SHAP = what moved *this model's* score, **not** causality — stated in README + prompt

---

# GenAI layer — RAG, not vibes

## ANN → SHAP → RAG → LLM, with citations enforced

```text
guidelines/ (AHRQ RED toolkit, 4 docs)
   │ 512 tokens / 50 overlap → text-embedding-3-small
   ▼
artifacts/chroma  (38 chunks, cosine)
   │ query = risk band + diagnosis + discharge context  → top-5
   ▼
gpt-4o-mini  →  5 sections: risk · model reasons · actions [S1]… · Sources · caveat
```

- Every suggested action must **cite a retrieved chunk**; unsupported → *"insufficient guidance"*
- `## Sources` lists exact documents + URLs (no placeholders)
- **Fail loud:** missing key *or* missing index exits with instructions **before** any OpenAI call · `--no-rag` opt-out recorded in `alerts.json`

---

# Governance — fairness, drift, human in the loop

## EU AI Act Art. 10/11/13/14/15 mapped to files (GOVERNANCE.md)

| Attribute | Demographic parity Δ | Equalised odds Δ |
|---|---:|---:|
| race | 0.1272 | 0.3500 |
| gender | 0.0203 | 0.0203 |
| **age** | **0.2943** | **0.5556** |

- **Age = largest disparity** — recorded with caveats (observations, not proof)
- Evidently PSI drift → `recommend_retrain` flag: **recorded, never auto-executed** (Art. 14)
- Every response carries **`requires_human_review`** — nothing here acts on a patient
- `/alert` spends API credit → `X-API-Key` + constant-time compare + 503 if unconfigured

---

# Ship it — MLOps & live demo

## FastAPI → Docker → EC2 → DuckDNS

- **Endpoints:** `/health` (reports RAG status) · `/cohort/sample` · `/predict` (SHAP on demand) · `/alert` (RAG-grounded, key-gated)
- **Docker** 3.3 GB CPU image ships artefacts **incl. the Chroma index**; `.env` injected at run time, never in a layer
- **EC2 t3.micro**, Amazon Linux 2023, 2 GB swap, `--restart unless-stopped`
- **Stable URL:** `readmission.duckdns.org` — systemd timer re-syncs every 5 min; stop/start never changes the link
- **CI:** GitHub Actions re-trains from raw CSV → audit → **28 offline tests** (RAG included: chunking, fake-embedder index round-trip, `[Sx]` contract, fail-loud)

**Live now:** http://readmission.duckdns.org:8000/docs

---

# Takeaways

## Honest results, every layer

- **AUC 0.675** = useful ranking, not a clinical instrument — thresholds, not accuracy, carry the decision
- **ANN ≈ XGBoost (tie)** — kept for SHAP, not metrics; comparison committed as `baseline.json`
- **RAG grounds the LLM:** real AHRQ documents, [S1] citations, fail-loud without an index
- **Fairness measured** (age gap found), **drift monitored**, **human always in the loop**

### Next

calibration · hybrid reranking (BM25 + vector) · more guideline sources · TLS/ALB · recent external validation

**Links:** live API `readmission.duckdns.org:8000` · `github.com/pravinkotta/hospital-readmission` · interactive report in repo
