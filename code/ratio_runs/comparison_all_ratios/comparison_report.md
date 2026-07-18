# End-to-end primary-ratio comparison

Each ratio was independently trained. Controls and the 60 GO terms were selected from that ratio's own training records.

| Ratio | AUROC | Raw AUPRC | Random baseline | Normalized AUPRC | F1 |
|---|---:|---:|---:|---:|---:|
| 1:1 | 0.842 | 0.807 | 0.500 | 0.614 | 0.802 |
| 1:2 | 0.854 | 0.682 | 0.333 | 0.524 | 0.712 |
| 1:3 | 0.861 | 0.647 | 0.250 | 0.530 | 0.623 |
| 1:4 | 0.866 | 0.596 | 0.200 | 0.495 | 0.580 |
| 1:5 | 0.874 | 0.537 | 0.167 | 0.444 | 0.554 |

## Checks

- Positive train/test pathway IDs are identical across all five branches.
- Every branch's main XGBoost metrics equal its corresponding sensitivity row.
- Highest AUROC: 1:5 (0.874).
- Highest normalized AUPRC: 1:1 (0.614).
- Pairwise GO overlap ranges from 17 to 30 of 60 terms.

This comparison describes complete pipelines. Because each ratio also has a different held-out prevalence and negative set, it should be interpreted with the normalized AUPRC and complemented by a future common-test-set comparison.
