"""
Model explainability using SHAP.

Loads the trained model artifact, computes SHAP values on the test split,
ranks features by mean absolute SHAP value (global importance), and writes
out both a summary plot and a JSON ranking so we can identify and describe
the least-impactful features in plain English.

Usage:
    python explain.py --data data/data.csv --model model/model.joblib --out explainability/
"""

import argparse
import json
import os

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COLUMN = "target"


def load_and_clean(data_path: str):
    df = pd.read_csv(data_path)
    df["gender"], _ = pd.factorize(df["gender"])
    df = df.dropna().reset_index(drop=True)
    return df


def main(data_path: str, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = load_and_clean(data_path)
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    explainer = shap.LinearExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
        json.dump(importance.to_dict(orient="records"), f, indent=2)

    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_bar.png"), dpi=150)
    plt.close()

    shap.summary_plot(shap_values, x_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_beeswarm.png"), dpi=150)
    plt.close()

    print("\nFeature importance ranking (highest to lowest impact):")
    print(importance.to_string(index=False))

    least_impactful = importance.tail(3)["feature"].tolist()
    print(f"\nLeast impactful features: {least_impactful}")
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--model", default="model/model.joblib")
    parser.add_argument("--out", default="explainability")
    args = parser.parse_args()

    main(args.data, args.model, args.out)
