# Interview guide — hospital-readmission capstone

Everything you need to present this project: the pitch, the process start-to-finish,
**how to show live output**, and the errors-and-fixes story. Deep detail lives in
[project-report.md](project-report.md); this is the version you study before an interview.

---

## 1. The 30-second pitch

> "I built an end-to-end hospital readmission risk system on real US data
> (UCI Diabetes 130-Hospitals, 100k encounters): a PyTorch ANN that predicts
> 30-day readmission, explained with **additivity-verified SHAP**, a
> **RAG-grounded LLM discharge alert** that cites real AHRQ guidance, a
> fairness and drift audit, MLflow tracking, and a Dockerized FastAPI service
> **live on EC2** at readmission.duckdns.org:8000. 28 tests and CI that
> retrains the model on every push. The honest headline: accuracy is a vanity
> metric here — I report PR-AUC and validation-tuned recall."

**One-line answer to "is the ANN even better than XGBoost?"** — "No, it's a
statistical tie (+0.0069 ROC-AUC, identical recall to four decimals). I
published the tie; I keep the ANN because DeepSHAP gives me verifiable
attributions."

---

## 2. Process start-to-finish (what you did, in order)

| Step | Command / artefact | What it produced |
|---|---|---|
| 1. Spec first | `RCTC.md` — constraints C1–C10 | every later decision traces to a written constraint |
| 2. Data | `python scripts/download_data.py` | UCI #296 CSV; cleaning: **101,766 → 99,343** encounters, 2,423 death/hospice discharges dropped (they *cannot* be readmitted — leakage, not cleaning), ICD-9 grouped into 10 chapters |
| 3. Leakage-safe split | `patient_split` on `patient_nbr` | **patient-level** 70/15/15 — no patient on both sides; test = 14,841 encounters, 11.79% positive |
| 4. Model | `python -m src.train` | ReadmissionMLP 120→64→32→1, pure torch, `BCEWithLogitsLoss` + `pos_weight`, ~40 s CPU, best val ROC-AUC 0.6751 (22 epochs, early stop) |
| 5. Threshold | F1-max sweep **on validation only** | threshold **0.57** — test touched exactly once |
| 6. Explainability | `src/explain.py` | DeepSHAP on the **raw log-odds** head; additivity asserted at runtime (`Σ SHAP + baseline = 3.2050 = logit` for the demo patient); global `global_shap.json` + per-patient waterfalls |
| 7. Baseline | `python -m src.baseline` | XGBoost on identical footing → the published tie |
| 8. RAG | `python scripts/fetch_guidelines.py` + `python -m src.rag --build` | 4 AHRQ RED toolkit documents → **38 chunks** (512 tokens / 50 overlap) → Chroma, top-5 cited `[S1]…[S5]` |
| 9. LLM alert | `python -m src.alert` | gpt-4o-mini writes a 160–240-word discharge alert from **structured facts only** (risk, band, drivers, retrieved chunks); 5 fixed sections incl. `## Sources` with verbatim URLs; **no template fallback** |
| 10. Governance | Fairlearn audit + Evidently drift + `GOVERNANCE.md` | race/gender/age fairness audit, drift snapshot with recorded-not-executed retrain trigger, mapping to EU AI Act Articles 11/13/14 |
| 11. Tracking | MLflow + SQLite backend | run + model registry (`champion` alias); `metrics.json` stayed byte-identical before/after (side channel, never forks the source of truth) |
| 12. Service | `python -m uvicorn src.api:app --port 8000` | `/health`, `/cohort/sample`, `/predict` (SHAP drivers + human-oversight flag), `/alert` (X-API-Key, constant-time compare) |
| 13. Tests + CI | `python -m pytest tests -q` → **28 tests**; `.github/workflows/ci.yml` | CI retrains from the raw CSV and asserts README metrics **±0.01**; secret scan |
| 14. Deploy | `DEPLOY.md` — Docker CPU image, EC2 t3.micro, DuckDNS | **live at http://readmission.duckdns.org:8000**, restart-policy recovery verified |
| 15. Deliverables | `python docs/build_deliverables.py` | 1-page write-up + 10-slide deck + full report (PDF page counts verified) |

---

## 3. How to verify the output LIVE (the demo script)

**Before the demo:** start instance `i-0bdba43e891c18402` from the **AWS
console** (your IAM lacks `ec2:StartInstances`). Both containers come back
automatically (`--restart unless-stopped`).

### In the browser (no commands)
1. `http://readmission.duckdns.org:8000/health` → `"status":"ok"`, model
   provenance, RAG enabled.
2. `http://readmission.duckdns.org:8000/docs` → the interactive Swagger page.
3. **GET /cohort/sample** → a real test patient.
4. **POST /predict** → Try it out → body:
   ```json
   {"patient_nbr": 88479036, "explain": true}
   ```
   → expected: **risk 0.961, band High, 10 SHAP drivers**. Compare against the
   committed artefacts — same score and drivers = the deployed model is the
   measured one.
5. Fresh browser tab: `http://readmission.duckdns.org:8000` works because
   DuckDNS tracks the box's IP.

### In the terminal (PowerShell)
```powershell
Invoke-RestMethod http://readmission.duckdns.org:8000/health

Invoke-RestMethod -Method POST -Uri http://readmission.duckdns.org:8000/predict `
  -ContentType "application/json" -Body '{"patient_nbr": 88479036, "explain": true}'
```

### The spendy endpoint (real LLM) — terminal, needs the key
```powershell
cd "C:\Users\pravi\Documents\Default Project\hospital-readmission"
$key = (Get-Content .env | Select-String "^ALERT_API_KEY=(.+)$").Matches[0].Groups[1].Value
Invoke-RestMethod -Method POST -Uri http://readmission.duckdns.org:8000/alert `
  -ContentType "application/json" -Headers @{ "X-API-Key" = $key } `
  -Body '{"patient_nbr": 88479036}'
```
Expected: 5 sections, **5 retrieved chunks cited `[S1]…[S5]`**, grounding
scores ~0.56–0.59, **real ahrq.gov URLs** in `## Sources`, clinician-review
caveat. Server log shows `retrieved 5 chunks`.

### "Is the build healthy?" — CI
```powershell
gh run list --limit 3    # green ✓ = fresh retrain reproduced the README numbers
```

**After the demo:** stop the instance (it bills only while running).

---

## 4. Errors I faced and solved (the tellable ones)

Full catalogue: `project-report.md` §5 (22 errors). The five that always land:

1. **SHAP additivity failed on sigmoid output** — `max diff 0.25` vs tolerance
   0.01. Root cause: the efficiency property holds on the **raw log-odds**, the
   sigmoid breaks it. Fix: explain log-odds, assert `Σ SHAP + baseline = logit`
   at runtime. *Lesson: a wrong explanation looks exactly like a right one —
   only the library's own check caught it.*
2. **Patient-level leakage** — a random row split puts the same patient in
   train and test; metrics looked suspiciously strong. Fix: split on the
   entity (`patient_nbr`), not the row.
3. **Docker build ENOSPC on the 8 GB box** — one requirements file shipped the
   whole dev stack including a 340 MB CUDA wheel. Fix: serving-only
   `requirements-service.txt`; image **2.27 GB → 1.95 GB**. *Lesson: ship the
   service's dependencies, not the repository's.*
4. **DuckDNS updated with my laptop's IP** — the update call with empty `ip=`
   stores the *caller's* IP. Fix: `deploy/duckdns.sh` runs **on the instance**
   as a systemd timer reading IMDSv2 every 5 minutes. *Lesson: "update the
   DNS" must run where the IP is.*
5. **README numbers drafted from memory were wrong** — caught against
   `artifacts/*.json`. Fix: every figure regenerated from the artefacts; CI
   now retrains and asserts ±0.01, so a wrong number fails the build.

Also worth having ready: threshold initially tuned on the test set (caught in
mentor review — tune on validation, report test once); MLflow 3 registry needs
a database backend (SQLite); Windows `.pem` ACL rejection; OOM pip install on
the micro (2 GB swap); `<URL>` placeholder in LLM output (prompt now says
*copy verbatim*, test rejects placeholders).

---

## 5. Numbers cheat sheet (all measured, from `artifacts/`)

| Fact | Value |
|---|---|
| Data | UCI #296: 101,766 → 99,343 encounters (patient-level 70/15/15) |
| Test set | n = 14,841, positive rate 0.1179 |
| ANN test | ROC-AUC **0.6754**, PR-AUC **0.2319**, recall **0.4377** @ threshold **0.57**, F1 0.2862 |
| XGBoost test | ROC-AUC 0.6685, PR-AUC 0.2326, recall 0.4377 @ 0.54 — **statistical tie** |
| Majority baseline | "always no" = 0.8821 accuracy — why accuracy is not reported |
| SHAP | log-odds, additivity max diff 0.000000 (asserted) |
| RAG | 4 AHRQ docs → 38 chunks, top-5 `[S1]…[S5]` |
| Tests / CI | 28 tests; CI retrain + assert ±0.01 |
| Live | http://readmission.duckdns.org:8000 (EC2 t3.micro, port 8000) |
| Demo patient | 88479036 → risk 0.961 High, 10 drivers |

---

## 6. Likely questions — one-line answers

- **"Why is your accuracy lower than the baseline?"** — "Accuracy on 11.8%
  positives is meaningless; 'always no' scores 0.8821. I report PR-AUC and
  validation-tuned recall (0.35 → 0.44 without touching test)."
- **"How do you know the deployed thing works?"** — "`/health` proves
  provenance; CI retrains from the raw CSV and asserts ±0.01; 28 keyless
  tests; the DNS name self-heals because my first update recorded my
  *laptop's* IP."
- **"How do you prevent hallucination in the alert?"** — "The LLM never
  computes — risk/band/drivers come from my artefacts; RAG supplies cited
  AHRQ chunks; every action cites or says 'insufficient guidance'; sources are
  copied verbatim (test-enforced); no template fallback."
- **"Why EC2 instead of a managed platform?"** — "Because the errors are the
  job: swap, disk, image size, ACLs, IAM, restart policies. Managed platforms
  hide all of it."
