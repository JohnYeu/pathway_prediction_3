# Primary class-ratio analyses

This directory keeps independently trained primary-ratio branches separate
from the paper's formal 1:1 output under `code/generated/`.

## Layout

- ratio_1_1 through ratio_1_5: complete primary-ratio reruns. Each branch has
  independently generated controls and independently selected training-only
  GO terms.
- ratio_1_1: reproducibility snapshot of the current formal 1:1 outputs.
- ratio_1_2: preserved former 1:2 formal output and sensitivity branch.
- comparison: archived two-ratio diagnostics retained for traceability.
- comparison_all_ratios: primary-ratio tables, GO-overlap audit, figure, and
  report comparing all five branches.

All branches use the same raw inputs, outer positive-pathway split, four
negative-set construction methods, model settings, and seed. The primary
positive-to-negative ratio is the intended experimental difference.

## Commands

Run any complete branch, for example 1:3:

    python3 reproducible_pipeline.py --full --primary-ratio 3 --sections all

Regenerate its held-out diagnostic figures:

    python3 make_metric_figures.py --primary-ratio 3

Compare the completed branches:

    python3 compare_primary_ratios.py --publish-dir generated

The comparison scripts do not retrain models or edit any source branch.
