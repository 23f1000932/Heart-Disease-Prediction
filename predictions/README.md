# Deliverable 5: Per-Sample Prediction with Logging & Observability

## Method
100 random patient rows were generated (`generate_and_predict.py`, seeded
independently from training data) covering realistic value ranges for all
13 clinical features. Each row was sent as an individual HTTP POST request
to the live `/predict` endpoint on the deployed GKE service
(http://35.225.151.184/predict) — one request per sample, not a batch call.

## Logging
Every prediction request is logged inside `app.py` via Python's `logging`
module, capturing: the full input feature dict, the prediction, the
probability, and a timestamp. Since the API runs inside a GKE pod, all
stdout/stderr is automatically collected by GCP Cloud Logging with zero
extra configuration.

## Observability evidence
Verified via:
gcloud logging read
'resource.type="k8s_container" AND resource.labels.cluster_name="heart-disease-cluster" AND textPayload:"prediction request"'
--limit=10 --format="table(timestamp, textPayload)" --freshness=1h

This returns individual log entries per prediction request, each with input
features, prediction, probability, and timestamp — confirming per-sample
observability via GCP Cloud Logging.

## Results
- 100/100 requests succeeded (0 failures)
- Predictions and inputs saved to `generated_100_with_predictions.csv`
- Class split: see CSV for full distribution of yes/no predictions
