# Deliverable 2: Model Explainability (SHAP)

## Method
SHAP LinearExplainer was used on the trained logistic regression model,
computing SHAP values on the held-out test set. Global feature importance
was ranked by mean absolute SHAP value across all test samples.

## Ranking (highest to lowest impact)
cp > ca > oldpeak > thalach > thal > exang > trestbps > gender > slope >
restecg > chol > age > fbs

## Least impactful features
**cholesterol (chol), age, and fasting blood sugar (fbs)** have the
least impact on the model's predictions. Once the model accounts for
chest pain type, number of major vessels, ST depression, and max heart
rate, these three features add almost no additional predictive signal —
fbs in particular (mean |SHAP| = 0.0006) is a simple binary threshold
with very little discriminative power in this dataset, while cholesterol
and age vary broadly across both disease-positive and disease-negative
patients, so their individual contribution is largely redundant with the
stronger predictors.

See `shap_summary_bar.png` and `shap_summary_beeswarm.png` for supporting plots.
