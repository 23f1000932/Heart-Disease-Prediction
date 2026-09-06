# Heart Disease Prediction — Production Deployment on GCP

**OPPE-2 | IITM BS MLOps**
**Repository:** `23f1000932_IITMBS_MLOPS_OPPE2_MAY_2026`
**Roll Number:** 23f1000932

A production-ready, explainable, observable, scalable, and maintainable deployment
of a heart disease classification model on Google Cloud Platform — dockerized,
served via FastAPI, deployed to GKE with autoscaling, and wired to a full
GitHub Actions CI/CD pipeline. Includes SHAP explainability, Fairlearn bias
auditing, per-sample prediction logging with Cloud Logging observability,
`wrk` load testing, and KS-test input drift detection.

---

## 1. Overview

| | |
|---|---|
| **Task** | Binary classification — heart disease present (`yes`) or absent (`no`) |
| **Input** | 13 clinical attributes per patient |
| **Model** | Logistic Regression (`RandomizedSearchCV`, 5-fold CV, 20 iterations) |
| **Test accuracy** | ~98% |
| **Serving** | FastAPI, containerized, deployed on GKE Autopilot |
| **Scaling** | Kubernetes HPA, CPU-based, min 1 / max 3 pods |
| **CI/CD** | GitHub Actions — build → push to Artifact Registry → rolling deploy to GKE |
| **GCP Project ID** | `project-bcb80534-6ef1-4dcc-951` |
| **Region** | `us-central1` |

### Architecture

```
                 ┌────────────────────┐
  git push       │  GitHub Actions     │
 ───────────────►│  build → push → deploy│
                 └─────────┬──────────┘
                           │ image: heart-disease-api:<sha>
                           ▼
              ┌─────────────────────────┐
              │ Artifact Registry        │
              │ heart-disease-repo       │
              └─────────┬───────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │  GKE Autopilot — heart-disease-cluster   │
   │  ┌───────────────────────────────────┐  │
   │  │ Deployment: heart-disease-api      │  │
   │  │  ├─ Pod (FastAPI + model.joblib)   │  │
   │  │  ├─ Pod (autoscaled)               │  │
   │  │  └─ Pod (autoscaled, max 3)        │  │
   │  └───────────────────────────────────┘  │
   │  HPA: cpu target 60%, min 1 / max 3      │
   │  Service: LoadBalancer (public IP)       │
   └─────────────────┬─────────────────────────┘
                     │  stdout/stderr
                     ▼
            GCP Cloud Logging (per-prediction logs)
```

Client requests → LoadBalancer Service → pods running the FastAPI container →
model inference → response + structured log line → automatically captured by
Cloud Logging (no extra agent needed on GKE).

---

## 2. Repository Structure

```
.
├── data/data.csv                          # Source dataset (14 cols incl. index + target)
├── HeartDiseaseTrainingAndPrediction.ipynb# Reference notebook (source of truth for train.py)
├── train.py                                # Standalone training script → model artifact
├── model/
│   ├── model.joblib                        # Trained model + feature_columns + gender mapping
│   ├── metrics.json                        # Training metrics (best params, test accuracy)
│   └── train_reference.csv                 # Cleaned training split (used as drift baseline)
│
├── app.py                                  # FastAPI service (/health, /predict)
├── Dockerfile
├── requirements.txt
│
├── k8s-deployment.yaml                     # Deployment (resources, probes)
├── k8s-service.yaml                        # LoadBalancer Service
├── k8s-hpa.yaml                            # HPA, min 1 / max 3, cpu 60%
├── .github/workflows/deploy.yml            # CI/CD: build → push → deploy
│
├── explain.py            + explainability/ # Deliverable 2 — SHAP
├── fairness.py           + fairness/       # Deliverable 3 — Fairlearn
├── generate_and_predict.py + predictions/  # Deliverable 5 — logging & observability
├── post.lua, wrk_results.txt + wrk_results_README.md   # Deliverable 6 — stress test
├── drift_detection.py    + drift/          # Deliverable 7 — input drift
│
└── .gitignore
```

---

## 3. Dataset & Model

Source data: [`IITMBSMLOps/MLOPS_MAY_2026_OPPE2`](https://github.com/IITMBSMLOps/MLOPS_MAY_2026_OPPE2)
(`data/data.csv`), 14 columns: `sno, age, gender, cp, trestbps, chol, fbs,
restecg, thalach, exang, oldpeak, slope, ca, thal, target`.

**Preprocessing (`train.py`):**
- `gender` (`male`/`female`) factorized to `0`/`1`
- Rows with missing values dropped (`dropna()`) — ~300 → 293 rows
- `sno` (observation index) **excluded from features** — it has no clinical
  meaning and was mistakenly left in the reference notebook's feature set
- `target` kept as raw string labels (`yes`/`no`)
- 80/20 train/test split with `random_state=42` (fixed for reproducibility —
  the reference notebook set `np.random.seed(42)` globally, which does not
  reliably seed `train_test_split` on its own)

**Model:** `LogisticRegression` tuned via `RandomizedSearchCV` (`C`,
`solver='liblinear'`, 5-fold CV, 20 iterations). Test accuracy ≈ 98%.

**Fixed from the reference notebook:** the notebook's final inference cell
(`model.predict(x_test.iloc[0])`) passes a 1D Series where a 2D array is
required and would error as written — `app.py` instead wraps a single row in
a one-row `DataFrame` before calling `.predict()`.

Reproduce training:
```bash
python train.py --data data/data.csv --out model/model.joblib --metrics model/metrics.json
```

---

## 4. Deliverable 2 — Explainability (SHAP)

`explain.py` fits a `shap.LinearExplainer` on the trained logistic regression
model and ranks features by mean absolute SHAP value on the held-out test set.

**Ranking (highest → lowest impact):**
`cp > ca > oldpeak > thalach > thal > exang > trestbps > gender > slope > restecg > chol > age > fbs`

**Least impactful features: cholesterol (`chol`), age, and fasting blood sugar (`fbs`)** —
`fbs` is essentially zero impact (mean |SHAP| = 0.0006 vs. 0.55 for the top feature).

> Once the model already knows a patient's chest pain type, number of major
> vessels colored by fluoroscopy, ST depression (`oldpeak`), and max heart
> rate achieved, additional knowledge of cholesterol, age, or fasting blood
> sugar barely changes its prediction. This doesn't mean these are clinically
> irrelevant to heart disease in general — their signal in this dataset is
> largely redundant with, or dominated by, the stronger predictors above.
> `fbs` is a simple binary threshold (>120 mg/dl) with little separating
> power on its own; cholesterol and age vary broadly across both
> disease-positive and disease-negative patients in this sample.

Run: `python explain.py --data data/data.csv --model model/model.joblib --out explainability/`
Outputs: `feature_importance.{csv,json}`, `shap_summary_bar.png`, `shap_summary_beeswarm.png`.

---

## 5. Deliverable 3 — Fairness (Fairlearn)

`fairness.py` uses Fairlearn's `MetricFrame` to compute accuracy, selection
rate, precision, and recall per group, plus demographic parity
difference/ratio and equalized odds difference. **Primary sensitive
attribute: age** (bucketed `<40`, `40-49`, `50-59`, `60+`), per the
deliverable spec; **gender** computed as a secondary check.

**Age**

| Group | Accuracy | Selection rate | Precision | Recall |
|-------|----------|-----------------|-----------|--------|
| <40   | 0.800    | 1.000           | 0.800     | 1.000  |
| 40-49 | 0.882    | 0.765           | 0.846     | 1.000  |
| 50-59 | 0.739    | 0.826           | 0.684     | 1.000  |
| 60+   | 0.786    | 0.643           | 0.667     | 1.000  |

Demographic parity difference: **0.357** · ratio: **0.643** · Equalized odds difference: **0.667**

**Gender**

| Group  | Accuracy | Selection rate | Precision | Recall |
|--------|----------|-----------------|-----------|--------|
| female | 0.941    | 0.882           | 0.933     | 1.000  |
| male   | 0.738    | 0.738           | 0.645     | 1.000  |

Demographic parity difference: **0.144** · ratio: **0.837** · Equalized odds difference: **0.167**

**Conclusion:** Recall is 1.0 for every subgroup in both attributes — the
model never misses a true positive in this test split. However, it is **not
equally fair across age groups**: patients under 40 are predicted positive
100% of the time vs. 64.3% for patients 60+, a 35.7-point demographic parity
gap (the largest disparity found). Gender shows a smaller (14.4-point) gap
favoring female patients. Caveat: the test set is small (~59 rows across 4
age groups), so some variation may be sample-size noise rather than a
systematic bias pattern.

Run: `python fairness.py --data data/data.csv --model model/model.joblib --out fairness/`

---

## 6. Deliverable 4 — Dockerized API on GKE with CI/CD

### 6.1 API (`app.py`)

FastAPI service with two endpoints:
- `GET /health` — liveness/readiness, returns model status, feature list, gender mapping
- `POST /predict` — accepts the 13 clinical features, returns `{prediction, probability_yes, timestamp}`; every request is logged via Python `logging`

### 6.2 Container

`Dockerfile` — `python:3.11-slim` base, installs `requirements.txt`, copies
`app.py` and `model/`, exposes port `8000`, runs via `uvicorn`.

```bash
docker build -t heart-disease-api:v1 .
docker run -d -p 8081:8000 --name heart-api-test heart-disease-api:v1
curl http://localhost:8081/health
```

### 6.3 Google Artifact Registry

```bash
gcloud artifacts repositories create heart-disease-repo \
  --repository-format=docker --location=us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag heart-disease-api:v1 us-central1-docker.pkg.dev/project-bcb80534-6ef1-4dcc-951/heart-disease-repo/heart-disease-api:v1
docker push us-central1-docker.pkg.dev/project-bcb80534-6ef1-4dcc-951/heart-disease-repo/heart-disease-api:v1
```

### 6.4 GKE Autopilot cluster

```bash
gcloud container clusters create-auto heart-disease-cluster --region=us-central1
gcloud container clusters get-credentials heart-disease-cluster --region=us-central1
```

### 6.5 Kubernetes manifests

- **`k8s-deployment.yaml`** — 1 replica base, `250m`/`256Mi` requests,
  `500m`/`512Mi` limits, readiness + liveness probes on `/health`
- **`k8s-service.yaml`** — `LoadBalancer`, port `80` → container port `8000`
- **`k8s-hpa.yaml`** — `autoscaling/v2` HPA, **min 1 / max 3 pods**, CPU
  target utilization **60%**

```bash
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-hpa.yaml
kubectl get service heart-disease-api-service   # external IP
```

Live API confirmed reachable at the LoadBalancer's external IP with both
`/health` and `/predict` returning correct results, and `kubectl get hpa`
showing `min 1 / max 3` registered.

### 6.6 CI/CD — GitHub Actions (`.github/workflows/deploy.yml`)

Triggers on every push to `main`:

1. Checkout code
2. Authenticate to GCP (`GCP_SA_KEY` secret, service account `github-ci-cd`)
3. Configure Docker for Artifact Registry
4. Build image, tag with `${{ github.sha }}` and `latest`
5. Push both tags to Artifact Registry
6. Install `gke-gcloud-auth-plugin`, fetch GKE credentials
7. `kubectl set image` on the deployment + `kubectl rollout status`

Required GitHub repo secrets: `GCP_SA_KEY` (service account JSON key),
`GCP_PROJECT_ID`. Service account `github-ci-cd` needs
`roles/artifactregistry.writer` and `roles/container.developer`.

Verified end-to-end: pushing a commit produced a running pod whose image tag
matched the commit SHA, confirming the automated build → push → deploy loop.

---

## 7. Deliverable 5 — Per-Sample Prediction, Logging & Observability

`generate_and_predict.py` generates **100 random patient rows** (seeded
independently from training data, `RANDOM_STATE=123`) covering realistic
per-feature ranges, and sends each as an **individual** HTTP `POST` to the
live `/predict` endpoint — one request per sample, not a batch call.

Every request is logged inside `app.py` (input features, prediction,
probability, timestamp). Since the API runs in a GKE pod, stdout/stderr is
automatically captured by **GCP Cloud Logging** with zero extra
configuration.

```bash
python generate_and_predict.py --api-url http://<EXTERNAL-IP> --n 100 --out predictions/
```

Observability verified via:
```bash
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.cluster_name="heart-disease-cluster" AND textPayload:"prediction request"' \
  --limit=10 --format="table(timestamp, textPayload)" --freshness=1h
```

**Result: 100/100 requests succeeded**, each with its own log entry showing
input features, prediction, probability, and timestamp. Outputs saved to
`predictions/generated_100_input.csv` and `generated_100_with_predictions.csv`.

---

## 8. Deliverable 6 — Stress Testing (`wrk`)

`post.lua` configures `wrk` to send `POST` requests with a JSON body against
`/predict`.

```bash
wrk -t8 -c2000 -d30s --timeout 10s --latency -s post.lua http://<EXTERNAL-IP>/predict
```

**Results (2000 concurrent connections, 8 threads, 30s):**

| Metric | Value |
|---|---|
| Requests/sec | 40.87 |
| Completed requests | 1,230 in 30.1s |
| Socket timeouts | 835 |
| Avg latency | 6.95s |
| p50 / p75 / p90 / p99 latency | 6.85s / 8.41s / 9.59s / 9.89s |

**HPA behavior:** CPU hit 130% against the 60% target; the HPA correctly
scaled the deployment from 1 pod to the configured maximum of **3 pods**
within the test window.

**Analysis:** even at the maximum of 3 pods, the API could not absorb 2,000
concurrent connections — ~40% of requests timed out and latency ballooned
into multiple seconds. The autoscaler behaved correctly; the bottleneck is
per-pod resource allocation: `250m` CPU request / `500m` CPU limit per pod
means 3 pods provide at most ~1.5 CPU cores total, which is insufficient
for this load. Increasing per-pod CPU limits, using async request handling,
batching, or a lighter serving layer would be the next steps to absorb
higher concurrency within the fixed 3-pod cap.

---

## 9. Deliverable 7 — Input Drift Detection

`drift_detection.py` runs a **Kolmogorov-Smirnov two-sample test** per
feature, comparing:
- **Reference:** `model/train_reference.csv` (real training distribution)
- **New:** `predictions/generated_100_input.csv` (100 synthetic rows from Deliverable 5)

A feature is flagged as drifted at `p < 0.05`.

**Result: 13/13 features showed statistically significant drift.** Largest
shifts (by KS statistic): `chol` (0.54), `oldpeak` (0.52), `thal` (0.45).

**Interpretation:** this is expected, by design, not a data quality issue.
The 100-row dataset was generated by sampling **uniformly at random** across
each feature's full observed min–max range (e.g., cholesterol uniform
between 126–564), whereas real training data follows a natural, non-uniform
clinical distribution (cholesterol clustered around a mean of ~248 with a
long right tail). Uniform random sampling systematically produces different
means and higher variance than the real-world distribution it was drawn
from — exactly what a KS test is designed to catch.

```bash
python drift_detection.py --train model/train_reference.csv --new predictions/generated_100_input.csv --out drift/
```

---

## 10. Notable Issues Encountered & Fixes

| Issue | Fix |
|---|---|
| `numpy`/`scikit-learn` version mismatch broke `shap` import | Upgraded `scikit-learn` to a NumPy-2-compatible version, retrained the model to avoid a cross-version pickle mismatch |
| `train.py` and `model/` were never committed to git | CI build failed with a missing `model/` directory; added with `git add -f model/` and committed |
| GKE Autopilot cluster creation failed: `SSD_TOTAL_GB` quota exceeded | Found a stuck failed cluster and an unrelated leftover `iris-cluster` both holding SSD quota; deleted both, freed headroom, retried successfully |
| `kubectl` couldn't authenticate to the new cluster | Installed the missing `google-cloud-cli-gke-gcloud-auth-plugin` and set `USE_GKE_GCLOUD_AUTH_PLUGIN=True` |
| GitHub Actions failed pushing to Artifact Registry (`permission denied`) | The Workbench instance's Compute Engine default service account lacked IAM-Admin rights to grant roles; roles were instead granted to the `github-ci-cd` service account via the GCP Console (Owner-level account) |

---

## 11. Reproducing This Project

```bash
git clone https://github.com/23f1000932/23f1000932_IITMBS_MLOPS_OPPE2_MAY_2026.git
cd 23f1000932_IITMBS_MLOPS_OPPE2_MAY_2026

# 1. Train
python train.py --data data/data.csv --out model/model.joblib --metrics model/metrics.json

# 2. Explainability & fairness
python explain.py --data data/data.csv --model model/model.joblib --out explainability/
python fairness.py --data data/data.csv --model model/model.joblib --out fairness/

# 3. Run locally
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Or via Docker
docker build -t heart-disease-api:v1 .
docker run -d -p 8081:8000 heart-disease-api:v1

# 5. Deploy to GKE (cluster + Artifact Registry already provisioned)
kubectl apply -f k8s-deployment.yaml -f k8s-service.yaml -f k8s-hpa.yaml

# 6. Generate observability & drift data
python generate_and_predict.py --api-url http://<EXTERNAL-IP> --n 100 --out predictions/
python drift_detection.py --train model/train_reference.csv --new predictions/generated_100_input.csv --out drift/

# 7. Stress test
wrk -t8 -c2000 -d30s --timeout 10s --latency -s post.lua http://<EXTERNAL-IP>/predict
```

CI/CD handles steps 4–5 automatically on every push to `main` once
`GCP_SA_KEY` and `GCP_PROJECT_ID` are set as repository secrets.

---

## 12. Deliverables Summary

| Deliverable | Marks | Status | Key files |
|---|---|---|---|
| 1. Private repo + collaborator | Mandatory | ✅ | Repo settings |
| 2. Model explainability | 10 | ✅ | `explain.py`, `explainability/` |
| 3. Fairness testing | 10 | ✅ | `fairness.py`, `fairness/` |
| 4. Dockerized API on GKE + CI/CD | 40 | ✅ | `app.py`, `Dockerfile`, `k8s-*.yaml`, `.github/workflows/deploy.yml` |
| 5. Per-sample logging & observability | 20 | ✅ | `generate_and_predict.py`, `predictions/` |
| 6. Stress testing (`wrk`) | 10 | ✅ | `post.lua`, `wrk_results.txt`, `wrk_results_README.md` |
| 7. Input drift detection | 10 | ✅ | `drift_detection.py`, `drift/` |
| **Total** | **100** | | |
