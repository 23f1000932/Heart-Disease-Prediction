# Deliverable 3: Fairness Testing (Fairlearn)

## Method
Fairlearn's MetricFrame was used to compute accuracy, selection rate,
precision, and recall grouped by sensitive attribute, on the held-out
test set. Demographic parity difference/ratio and equalized odds
difference were computed as summary fairness metrics.

Primary sensitive attribute (per Deliverable 3): **age**, bucketed into
<40, 40-49, 50-59, 60+.
Secondary check (per Pipeline Overview): **gender**.

## Results — Age
| Group | Accuracy | Selection rate | Precision | Recall |
|-------|----------|-----------------|-----------|--------|
| <40   | 0.800    | 1.000           | 0.800     | 1.000  |
| 40-49 | 0.882    | 0.765           | 0.846     | 1.000  |
| 50-59 | 0.739    | 0.826           | 0.684     | 1.000  |
| 60+   | 0.786    | 0.643           | 0.667     | 1.000  |

Demographic parity difference: 0.357
Demographic parity ratio: 0.643
Equalized odds difference: 0.667

## Results — Gender
| Group  | Accuracy | Selection rate | Precision | Recall |
|--------|----------|-----------------|-----------|--------|
| female | 0.941    | 0.882           | 0.933     | 1.000  |
| male   | 0.738    | 0.738           | 0.645     | 1.000  |

Demographic parity difference: 0.144
Demographic parity ratio: 0.837
Equalized odds difference: 0.167

## Conclusion
Recall is 1.0 for every subgroup in both attributes — the model does not
miss any true heart-disease-positive patient in any age or gender group
in this test split. However, the model is **not equally fair across
age groups**: younger patients (<40) are predicted positive 100% of the
time, while patients 60+ are predicted positive only 64.3% of the time
— a 35.7-point demographic parity gap, and the largest disparity metric
in this analysis. Gender shows a smaller but still present gap (14.4
points), with the model more favorable (higher accuracy and precision)
toward female patients than male patients in this test set.

Caveat: the test set is small (~59 rows split across 4 age groups),
so some of this variation may reflect sample-size noise rather than a
robust bias pattern. A larger holdout or cross-validated fairness
evaluation would be needed to confirm these disparities are systematic
rather than incidental to this particular split.
