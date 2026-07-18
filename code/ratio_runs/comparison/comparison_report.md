# Primary ratio comparison: 1:1 versus 1:2

## Design

Both branches use the same raw database snapshots, positive-pathway split, model settings, random seed, and four negative types. Each branch independently generates controls and selects 60 GO terms from its own training records. This compares complete 1:1 and 1:2 training pipelines rather than reweighting one fitted model.

## Primary XGBoost results

| Metric | 1:1 | 1:2 | Delta (1:1 - 1:2) |
|---|---:|---:|---:|
| CV AUROC | 0.863 | 0.871 | -0.009 |
| Test AUROC | 0.842 | 0.854 | -0.012 |
| Test AUPRC | 0.807 | 0.682 | +0.125 |
| F1 | 0.802 | 0.712 | +0.090 |
| Precision | 0.750 | 0.656 | +0.094 |
| Recall | 0.861 | 0.778 | +0.083 |
| Brier score | 0.156 | 0.156 | -0.000 |

## Interpretation

- The random AUPRC baseline is 0.500 for 1:1 and 0.333 for 1:2. Raw AUPRC values should not be compared without this prevalence difference.
- Prevalence-adjusted AUPRC is 0.614 for 1:1 and 0.524 for 1:2.
- The GO lists share 21 of 60 terms (Jaccard 0.212).
- AUROC is the cleaner direct discrimination comparison because it is less tied to class prevalence. F1, precision, recall, Brier score, and raw AUPRC also reflect test prevalence and the fixed 0.5 threshold.
- One seed-42 split cannot establish that one ratio is universally better. The choice should also reflect the objective: 1:1 balances classes, while 1:2 exposes the model to more constructed controls.
- The top held-out AUROC model is Random Forest for 1:1 and CatBoost for 1:2.
- Mean AUROC across the 13 models is 0.804 for 1:1 and 0.818 for 1:2.
- Mean LOFO AUROC is 0.857 for 1:1 and 0.873 for 1:2.
- The size-only AUROC remains similar (0.768 for 1:1; 0.773 for 1:2), so changing the class ratio does not remove the size signal already documented by the ablation analysis.

## Threshold 0.5 diagnostics

| Metric | 1:1 | 1:2 |
|---|---:|---:|
| Accuracy | 0.787 | 0.790 |
| Sensitivity | 0.861 | 0.778 |
| Specificity | 0.713 | 0.796 |
| Balanced accuracy | 0.787 | 0.787 |
| Matthews correlation coefficient | 0.580 | 0.554 |

## Practical decision

For this run, the 1:2 branch remains the stronger primary benchmark. It has the higher XGBoost held-out and CV AUROC, the higher mean LOFO AUROC, and more constructed controls. The 1:1 branch is useful as a sensitivity analysis. Its higher raw AUPRC and F1 should not be read as an unqualified improvement because their baseline changes with class prevalence.

## Reproducibility checks

- 1:1 main row equals its sensitivity 1:1 row: True
- 1:2 main row equals its sensitivity 1:2 row: True
- Full comparison tables are stored beside this report.
