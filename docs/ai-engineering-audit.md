# AI Engineering Audit — does this repo fulfil the course checklists?

Cross-check of `hospital-readmission` against the three AI Career Compass reference
files:

1. **AI Predictive Analytics: End-to-End Pipeline** (Session 6–7) — the 7-layer stack
2. **The Modern AI Stack — LLM, RAG, AI Agent & MCP** — the 4-layer model
3. **Week 7 — MLOps, Deployment, XAI & Governance** — MLOps 8 steps + LLMOps 7 steps

**Verdict:** the predictive-pipeline checklist and the Week 7 MLOps checklist are
covered — in several places *better* than the material's own example code. The
Modern AI Stack four-layer model is the gap: **LLM ✅ · RAG ✅ · AI Agent ❌ · MCP ❌**
— the latter two exist in none of the three planned capstones yet.

---

## 1. Modern AI Stack — 4 layers

| Layer | Required | Status | Evidence |
|---|---|---|---|
| 1 · LLM | core reasoning model | ✅ | `gpt-4o-mini` in `src/alert.py`; missing key fails loud before any call; no template fallback |
| 2 · RAG | retrieve-then-generate | ✅ | `src/rag.py`: 512-token / 50-overlap chunks → `text-embedding-3-small` → Chroma → top-5 as `[S1]…[S5]` with inline citations — matches the Week 7 LLMOps step 2 pattern verbatim |
| 3 · AI Agent | plan → tools → act → memory + human boundary | ❌ | The alert stage is a **single-shot LLM call**. The retrieval query is built deterministically from SHAP drivers; no tool loop, no memory, no escalate-to-human decision (the Klarna boundary pattern is absent) |
| 4 · MCP | standard tool-connection layer | ❌ | No MCP server/client anywhere; tools are called from Python directly |

Agent-failure coverage (the "9 problems" table): only **Problem 1 (hallucination →
RAG)** is addressed. Problems 2 (wrong tool), 4 (prompt injection), 6 (memory),
7 (loops), 9 (tracing) don't apply yet — because there is no agent.

---

## 2. Week 7 — MLOps 8 steps

| Step | Status | Notes |
|---|---|---|
| 1 · DVC data versioning | ❌ | Raw CSV re-downloaded by `scripts/download_data.py`; no hash tracking. *Interview answer:* the frozen UCI download + CI metric assertions cover the reproducibility goal at this scale; DVC at ~10M rows |
| 2 · MLflow experiment tracking | ✅ | SQLite backend (`mlflow.db`), params/metrics/artefact/git-commit per run |
| 3 · Model registry Staging→Production | ✅ | `readmission-ann`, `champion` alias. *Honest deviation:* the API loads `artifacts/model.pt` on purpose — MLflow is a side channel (constraint P3: tracking must not fork the source of truth) |
| 4 · Pipeline packaging (anti-skew) | ✅ | shared `preprocessor.joblib` fitted on train only, used identically by training and serving |
| 5 · FastAPI + Docker | ✅ | Pydantic `ScoreRequest` validation; image 1.95 GB CPU-only. *Minor deviation:* threshold comes from `metrics.json`/config, not an env var |
| 6 · XAI at serving time | ✅ | `POST /predict?explain=true` returns SHAP drivers — with an **additivity assertion** on log-odds, stronger than the lab's top-3 |
| 7 · Evidently drift + retrain trigger | ✅ | `src/audit.py` DataDriftPreset; trigger **recorded, not executed** (documented scope) |
| 8 · Automated retraining loop | ⚠️ | recorded-not-executed by design |

## Week 7 — LLMOps 7 steps

| Step | Status | Notes |
|---|---|---|
| 1 · Prompts versioned in Git | ✅ | `SYSTEM_PROMPT` lives in `src/alert.py` — PR-able (a `prompts/` dir would be more textbook) |
| 2 · RAG pipeline (chunk/embed/vector store) | ✅ | 38 chunks from 4 AHRQ docs, exactly the prescribed pattern |
| 3 · RAGAS evaluation in CI | ❌ | No golden Q&A set, no faithfulness gate — **cheapest remaining win** |
| 4 · Guardrails | ⚠️ | Prompt-level rules + citation enforcement + a test rejecting `<URL>` placeholders; no NeMo/LlamaIndex guardrails. Mitigation: inputs are structured facts, not free user text |
| 5 · Token cost per query | ✅ | `prompt_tokens`/`completion_tokens` logged to `alerts.json` and returned in the `/alert` response |
| 6 · Hallucination-rate monitoring (LLM judge) | ❌ | Not implemented |
| 7 · RAG index freshness (scheduled re-index) | ❌ | Manual `python -m src.rag --build` only — a systemd timer (like the DuckDNS one) would close it |

## Week 7 — XAI, governance, delivery

| Item | Status |
|---|---|
| SHAP global + local (DeepExplainer) | ✅ with runtime additivity assertion |
| LIME / Grad-CAM | ❌ — correctly N/A for tabular (the material itself prescribes "SHAP for tabular") |
| Fairlearn MetricFrame audit | ✅ `src/audit.py` → `fairness.json` |
| EU AI Act mapping (Arts 10/11/13/14) | ✅ `GOVERNANCE.md` |
| FastAPI + Pydantic · Docker · GitHub Actions CI | ✅ all three; CI also re-trains from raw CSV and asserts README metrics ±0.01 |
| Training-serving skew | ✅ shared preprocessor — the exact fix the material prescribes |

---

## 3. Predictive Analytics Pipeline — 7-layer stack

| Layer | Status | Notes |
|---|---|---|
| 1 · Data ingestion | ✅ | UCI CSV download script (no SQL/DICOM/PDF paths — not this dataset) |
| 2 · Data engineering (Bronze→Silver→Gold, PySpark, dbt, Great Expectations, Feast) | ❌ consciously | pandas on 100K rows, ~1-min pipeline. *Interview answer:* "no distributed compute is justified at 100K rows; medallion layers are for scale, mine is reproducible end-to-end instead" |
| 3 · Feature engineering + XAI | ✅ | ColumnTransformer pipeline; SHAP is the correct XAI for tabular |
| 4 · Model layer | ✅ | ANN **and** the XGBoost baseline — the matrix's "Hospital readmission · ANN + SHAP" capstone, both heads |
| 5 · GenAI augmentation | ⚠️ | RAG ✅ · LLM narrative ✅ · **agentic loops ❌** |
| 6 · Deployment (FastAPI · Docker · MLflow · CI/CD · drift) | ✅ | all five |
| 7 · Business layer (ROI, governance, HITL) | ✅ | human-oversight flag on every response, EU AI Act mapping, honest cost/limitation notes |

Cross-cutting "Right Metric First" principle: ✅ recall/PR-AUC framed against the
0.1179 base rate; accuracy explicitly called out as a vanity metric.

**The healthcare capstone named in this file — "Patient readmission prediction API
(ANN + SHAP + FastAPI)" — is this repository. That track is complete.**

---

## 4. Gap → which project carries it

| Gap | Best home | Why (the course's own wording) |
|---|---|---|
| **AI Agent (Layer 3)** | **Capstone 2 — MRI predictive maintenance** | The pipeline file's manufacturing capstone is literally *"MRI predictive maintenance with LSTM + LangGraph ticket agent"*. Boundary: never close a ticket or dispatch a technician without human confirmation (Klarna pattern). Gives all 9 failure modes somewhere to live |
| **MCP (Layer 4)** | **Capstone 3 — Insurance claims** (and MRI) | The Modern AI Stack file's worked example *is* the claims capstone: LLM summarizes → RAG retrieves policy → Agent flags & routes to adjuster → **MCP connects claims DB + document store + case-management through one interface** instead of three custom integrations |
| **RAGAS eval gate** | ⭐ hospital-readmission (small add) | 10–15 golden Q&A pairs over the AHRQ corpus, faithfulness/context-recall thresholds as a CI step — turns "RAG works" into a number |
| **Index-freshness timer + LLM-judge hallucination sampling** | hospital-readmission (small add) | Weekly re-index (systemd timer pattern already exists for DuckDNS) + sample 2% of alerts for faithfulness scoring |
| **Prompt-injection guardrail** | insurance claims (free with the agent) | Structured inputs keep us safe *today*; free-text claim notes are exactly where injection matters (Problem 4) |
| **DVC / PySpark / dbt / Great Expectations / Feast** | neither by default | Right-sized out. One-sentence answer ready for interviews (see Week 7 step 1) |

**Interview one-liner:** *"RAG and MLOps are done and deployed in repo 1; agents and
MCP are exactly what capstone 2 (LangGraph maintenance-ticket agent) and capstone 3
(claims assistant over MCP) are for — the course material prescribes them there
itself."*
