"""
Training script for the heart disease classifier.

Mirrors the logic in HeartDiseaseTrainingAndPrediction.ipynb, with two
deliberate fixes over the notebook:
  1. 'sno' (observation index) is dropped before training — it is not one
     of the 14 clinical features and has no predictive meaning.
  2. train_test_split is given an explicit random_state for reproducibility.

Usage:
    python train.py --data data/data.csv --out model/model.joblib
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"
RANDOM_STATE = 42


def load_and_clean(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Encode gender: male/female -> 0/1 (factorize, same as notebook)
    df["gender"], gender_categories = pd.factorize(df["gender"])

    # Drop rows with any missing values (same as notebook's dropna())
    df = df.dropna().reset_index(drop=True)

    return df, list(gender_categories)


def train(data_path: str, out_path: str, metrics_path: str):
    df, gender_categories = load_and_clean(data_path)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    log_reg_grid = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["liblinear"],
    }

    rs_log_reg = RandomizedSearchCV(
        LogisticRegression(max_iter=1000),
        param_distributions=log_reg_grid,
        cv=5,
        n_iter=20,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rs_log_reg.fit(x_train, y_train)

    best_model = rs_log_reg.best_estimator_
    test_accuracy = best_model.score(x_test, y_test)
    preds = best_model.predict(x_test)

    report = metrics.classification_report(y_test, preds, output_dict=True)

    print(f"Best params: {rs_log_reg.best_params_}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "gender_categories": gender_categories,  # e.g. ['male', 'female'] -> index is the encoding
            "best_params": rs_log_reg.best_params_,
        },
        out_path,
    )
    print(f"Model artifact saved to {out_path}")

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "test_accuracy": test_accuracy,
                    "best_params": rs_log_reg.best_params_,
                    "classification_report": report,
                    "n_train": len(x_train),
                    "n_test": len(x_test),
                },
                f,
                indent=2,
            )
        print(f"Metrics saved to {metrics_path}")

    # Also persist the cleaned training data (features + target) — this is
    # the reference distribution we'll compare against for drift detection later.
    train_ref_path = os.path.join(os.path.dirname(out_path) or ".", "train_reference.csv")
    x_train.assign(**{TARGET_COLUMN: y_train}).to_csv(train_ref_path, index=False)
    print(f"Training reference distribution saved to {train_ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--out", default="model/model.joblib")
    parser.add_argument("--metrics", default="model/metrics.json")
    args = parser.parse_args()

    train(args.data, args.out, args.metrics)
