# Governance — XAI, fairness and EU AI Act mapping

This project deploys AI into a **high-risk** domain (healthcare). Under the EU
AI Act (Regulation 2024/1689) that means the requirements below are not
optional extras — this file maps each one to the file in this repository that
implements it, and to the artefact that proves it.

Scope claim, stated honestly: this is a **benchmark case study**, not a
conformity-assessed medical device. Art. 43 conformity assessment itself
requires a notified body and is out of scope — what is demonstrated here is
that the *engineering practices* the articles demand are implemented and
machine-checked.

---

## Compliance checklist

| Requirement | Article | Implemented in | Evidence |
|---|---|---|---|
| **Transparency** — explainability per prediction | Art. 13 | `src/explain.py` (DeepSHAP on log-odds, additivity asserted at runtime), `POST /predict?explain=true` returns the drivers | `artifacts/global_shap.json`, per-patient drivers in the response, `notebooks/…` waterfall |
| **Human oversight** — no automatic action on high-risk output | Art. 14 | every response carries `requires_human_review`; `/alert` is a *draft for clinician review* and says so in its own Caveat section | `src/api.py` (`base_payload`), `alerts.json` |
| **Accuracy & robustness** — monitoring, evaluation | Art. 15 | patient-level split, threshold tuned on validation only, `python -m src.audit` drift monitoring, CI re-runs the whole pipeline | `artifacts/metrics.json`, `drift_summary.json`, `.github/workflows/ci.yml` |
| **Technical documentation** — traceable experiments | Art. 11 | MLflow: params, metrics, model artefact, git commit tag per run | `mlflow.db` + `mlruns/`, registry `readmission-ann` (alias `champion`) |
| **Data governance** — provenance, integrity | Art. 10 | pinned UCI source + `scripts/download_data.py`, row-count profile of every stage, exclusions documented (death/hospice, sparse columns) | `artifacts/data_profile.json`, `src/data.py` docstrings |
| **Bias / conformity inputs** | Art. 10(2)(f), 43 prep | Fairlearn audit across race, gender and age on the held-out set | `artifacts/fairness.json` |

**Regulatory note:** the AI Act entered into force 1 Aug 2024 with obligations
phasing in (high-risk Annex III systems: Aug 2026/2027). The mapping above
follows Articles 10, 11, 13, 14, 15 and 43 of the final text.

---

## The three XAI techniques — and why this project uses one

| Technique | Modality | Why it fits / doesn't fit here |
|---|---|---|
| **SHAP** | tabular (any model) | ✅ **Used.** DeepSHAP is *exact* for piecewise-linear networks, so attributions are verifiable — and this project's additivity assertion (`max diff ≪ 0.01` on log-odds) is what makes that a fact rather than a claim. Waterfall per patient + summary plot globally. |
| **LIME** | image / text | ❌ Not used: it fits a local linear surrogate around perturbed inputs — designed when the model's interior is unknowable (CNNs over pixels). For a 9,857-parameter MLP on 120 engineered columns, a surrogate of the model is strictly worse than explaining the model itself. |
| **Grad-CAM** | CNN only | ❌ Not applicable: there is no convolutional layer and no spatial input. Grad-CAM needs gradients of a last conv feature map; an MLP has nothing to localise on. Applying it would be decoration, not evidence. |

Rule applied: **one right technique, defended, beats three techniques
checkboxed.** If this project gained an imaging task (e.g. chest X-ray
readmission risk), Grad-CAM/LIME would be the correct additions there.

---

## Fairness results (measured, held-out test set)

From `artifacts/fairness.json` at threshold 0.57:

| Attribute | Demographic parity difference | Equalised odds difference |
|---|---:|---:|
| race | 0.1272 | 0.3500 |
| gender | 0.0203 | 0.0203 |
| age | **0.2943** | **0.5556** |

Reading these honestly:

- **Age is the largest disparity** — the model flags older cohorts at
  materially different rates (equalised odds 0.56 means TPR/FPR gaps across
  age bands are wide). Prior visits (`number_inpatient`) correlate with age in
  this dataset, so the disparity is at least partly a *feature* effect, not a
  direct age effect — that is a hypothesis for a reviewer to test, not a
  conclusion.
- **Gender disparity is near zero**; race sits in between.
- **These are observations, not verdicts.** Small groups (Asian, Hispanic) have
  wide uncertainty; a demographic-parity gap is a prompt for domain review
  before deployment, never proof of discrimination.
- The JSON carries this caveat in its own `notes` field so the artefact cannot
  be quoted without it.

## Drift monitor and the retrain trigger

`python -m src.audit` builds an Evidently report (reference = training
distribution, current = held-out) and writes `drift_summary.json`, whose
`recommend_retrain` flag is the hook a scheduler would watch — the
mentor-pipeline edge `Drift Monitor → Retrain Trigger`. It is **recorded, not
executed**: automatic retraining without review is exactly what an Art. 14
human-oversight gate exists to prevent.

Current measured state: **0 of 47 columns drifted** (reference vs held-out from
the same source — as expected; the demonstration is that the monitor runs and
its rule is machine-checked in `tests/test_audit.py`). Production drift means
*incoming* weeks diverging from training, which this dataset cannot show.

## HITL gate

```
POST /predict → "requires_human_review": true|false   (= threshold crossing)
POST /alert   → requires X-API-Key + is worded as a draft for clinician review
```

No endpoint in this repository acts on a patient — it scores and drafts. The
decision stays with a human, which is the whole point of Art. 14.

## Known gaps (stated, not hidden)

- No Art. 43 conformity assessment (needs a notified body; out of scope).
- No prospective validation — data is 1999–2008.
- Calibration not measured; probabilities are used for *ranking + threshold*,
  not as literal risk percentages (README Limitations).
- Fairness audit covers three attributes on one dataset; a real system needs
  intersectional analysis and a documented mitigation process.
