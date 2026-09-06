"""
Deliverable 5: generate a 100-row random dataset and run per-sample
predictions through the deployed API. Each request is logged by the
API itself (via Python logging -> stdout -> GCP Cloud Logging on GKE).

This script also saves the generated 100-row dataset (with predictions)
locally, since Deliverable 7 (drift detection) reuses this same data.

Usage:
    python generate_and_predict.py --api-url http://35.225.151.184 --n 100 --out predictions/
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

RANDOM_STATE = 123

FEATURE_RANGES = {
    "age": (29, 77),
    "gender": (0, 1),
    "cp": (0, 3),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (71, 202),
    "exang": (0, 1),
    "oldpeak": (0.0, 6.2),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (0, 3),
}


def generate_random_patients(n: int) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    data = {}
    for feature, (low, high) in FEATURE_RANGES.items():
        if feature == "oldpeak":
            data[feature] = np.round(rng.uniform(low, high, size=n), 1)
        else:
            data[feature] = rng.randint(low, high + 1, size=n)
    df = pd.DataFrame(data)
    return df


def run_predictions(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    results = []
    for i, row in df.iterrows():
        payload = row.to_dict()
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            results.append({
                **payload,
                "prediction": body["prediction"],
                "probability_yes": body["probability_yes"],
                "api_timestamp": body["timestamp"],
            })
            print(f"[{i+1}/{len(df)}] prediction={body['prediction']} prob_yes={body['probability_yes']:.3f}")
        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{len(df)}] ERROR: {e}")
            results.append({**payload, "prediction": None, "probability_yes": None, "api_timestamp": None})
        time.sleep(0.05)
    return pd.DataFrame(results)


def main(api_url: str, n: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = generate_random_patients(n)
    input_path = os.path.join(out_dir, "generated_100_input.csv")
    df.to_csv(input_path, index=False)
    print(f"Generated {n} random patient rows -> {input_path}")

    results_df = run_predictions(df, api_url)
    results_path = os.path.join(out_dir, "generated_100_with_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nAll predictions complete. Results saved to {results_path}")

    n_failed = results_df["prediction"].isna().sum()
    print(f"Successful: {len(results_df) - n_failed}/{len(results_df)}, Failed: {n_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="predictions")
    args = parser.parse_args()

    main(args.api_url.rstrip("/"), args.n, args.out)
