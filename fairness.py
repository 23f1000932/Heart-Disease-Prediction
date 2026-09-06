"""
Fairness testing using Fairlearn.

Deliverable 3 specifies "age" as the sensitive attribute. Since age is
continuous, we bucket it into clinically meaningful groups to compute
group fairness metrics. We also compute the same metrics for gender
as a secondary check, since the Pipeline Overview mentions gender.

Usage:
    python fairness.py --data data/data.csv --model model/model.joblib --out fairness/
"""

import argparse
import json
import os

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, recall_score, precision_score

RANDOM_STATE = 42
TARGET_COLUMN = "target"

AGE_BINS = [0, 40, 50, 60, 100]
AGE_LABELS = ["<40", "40-49", "50-59", "60+"]


def load_and_clean(data_path: str):
    df = pd.read_csv(data_path)
    df["gender"], gender_categories = pd.factorize(df["gender"])
    df = df.dropna().reset_index(drop=True)
    return df, list(gender_categories)


def run_fairness(y_true, y_pred, sensitive_feature, label: str, out_dir: str):
    y_true_bin = (y_true == "yes").astype(int)
    y_pred_bin = (pd.Series(y_pred) == "yes").astype(int)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "precision": precision_score,
            "recall": recall_score,
        },
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        sensitive_features=sensitive_feature,
    )

    dp_diff = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_feature)
    dp_ratio = demographic_parity_ratio(y_true_bin, y_pred_bin, sensitive_features=sensitive_feature)
    eo_diff = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_feature)

    result = {
        "sensitive_attribute": label,
        "by_group": mf.by_group.to_dict(),
        "overall": mf.overall.to_dict(),
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
    }

    out_path = os.path.join(out_dir, f"fairness_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n=== Fairness report: sensitive attribute = {label} ===")
    print(mf.by_group)
    print(f"Demographic parity difference: {dp_diff:.4f}")
    print(f"Demographic parity ratio:      {dp_ratio:.4f}")
    print(f"Equalized odds difference:     {eo_diff:.4f}")
    print(f"Saved to {out_path}")

    return result


def main(data_path: str, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df, gender_categories = load_and_clean(data_path)
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    y_pred = model.predict(x_test)

    age_groups = pd.cut(x_test["age"], bins=AGE_BINS, labels=AGE_LABELS)
    run_fairness(y_test.reset_index(drop=True), y_pred, age_groups.reset_index(drop=True), "age", out_dir)

    gender_labels = x_test["gender"].map(dict(enumerate(gender_categories)))
    run_fairness(y_test.reset_index(drop=True), y_pred, gender_labels.reset_index(drop=True), "gender", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--model", default="model/model.joblib")
    parser.add_argument("--out", default="fairness")
    args = parser.parse_args()

    main(args.data, args.model, args.out)
