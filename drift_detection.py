"""
Deliverable 7: Input drift detection.

Compares the training data distribution (model/train_reference.csv) against
the 100-row randomly generated dataset used for prediction in Deliverable 5
(predictions/generated_100_input.csv), using the Kolmogorov-Smirnov test
per feature to detect distributional shift.

Usage:
    python drift_detection.py --train model/train_reference.csv --new predictions/generated_100_input.csv --out drift/
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import ks_2samp

FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

ALPHA = 0.05


def main(train_path: str, new_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_path)

    results = []
    for feature in FEATURE_COLUMNS:
        stat, p_value = ks_2samp(train_df[feature], new_df[feature])
        drifted = p_value < ALPHA
        results.append({
            "feature": feature,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(drifted),
            "train_mean": round(float(train_df[feature].mean()), 2),
            "new_mean": round(float(new_df[feature].mean()), 2),
            "train_std": round(float(train_df[feature].std()), 2),
            "new_std": round(float(new_df[feature].std()), 2),
        })

    result_df = pd.DataFrame(results).sort_values("p_value")
    result_df.to_csv(os.path.join(out_dir, "drift_report.csv"), index=False)
    with open(os.path.join(out_dir, "drift_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    n_drifted = result_df["drift_detected"].sum()
    print(result_df.to_string(index=False))
    print(f"\n{n_drifted}/{len(FEATURE_COLUMNS)} features show statistically significant drift (p < {ALPHA})")
    print(f"Report saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="model/train_reference.csv")
    parser.add_argument("--new", default="predictions/generated_100_input.csv")
    parser.add_argument("--out", default="drift")
    args = parser.parse_args()

    main(args.train, args.new, args.out)
