# Reproducible Code Notes

This folder was originally a manuscript package: it contained LaTeX, cached
JSON data, and finished figures, but not a complete raw-data-to-paper pipeline.
`reproducible_pipeline.py` now rebuilds the main analysis from documented raw
source files under `raw_sources/`.

## What This Code Rebuilds

The pipeline regenerates the main intermediate products under `generated/`:

- source records and exact gene-set groups: `data/pathway_source_records.json`
  and `data/pathway_gene_set_groups.json`
- grouping audit: `tables/pathway_grouping_summary.csv` and
  `tables/pathway_duplicate_groups.csv`
- main held-out benchmark: `tables/main_benchmark.csv`
- held-out predictions, confusion matrix, and negative-type diagnostics:
  `tables/heldout_predictions.csv`, `tables/heldout_confusion_matrix.csv`,
  `tables/heldout_score_by_type.csv`, and
  `tables/heldout_negative_type_performance.csv`
- training-only nested ratio comparison: `tables/ratio_cv_per_fold.csv`,
  `tables/ratio_cv_summary.csv`, and `tables/ratio_cv_consensus_summary.csv`
- training-only 12-method comparison: `tables/model_comparison.csv` and
  `tables/model_comparison_per_fold.csv`
- six-tree soft-voting evidence: `tables/tree_ensemble_cv.csv`,
  `tables/tree_ensemble_cv_per_fold.csv`, and
  `data/primary_ensemble_manifest.json`
- model-specific tree-parameter selection:
  `tables/tree_hyperparameter_tuning_summary.csv`,
  `tables/tree_hyperparameter_selected.csv`, and
  `data/selected_tree_hyperparameters.json`
- ablation study: `tables/ablation.csv`
- ensemble SHAP feature importance: `tables/feature_importance.csv`,
  `tables/shap_component_additivity.csv`, and
  `data/shap_ensemble_manifest.json`
- split, negative-source, GO-selection, and CV audits under `tables/` and `data/`

It also creates the analysis figures that can be synchronized with the manuscript
after a completed full run has been reviewed:

- `generated/figures/Fig3_shap.png`
- `generated/figures/Fig5_datastats.png`
- `generated/figures/Fig7_robustness.png`
- `generated/figures/FigC_roc_pr.png`
- `generated/figures/FigF_confusion.png`
- `generated/figures/FigG_score_by_type.png`
- `generated/figures/Fig8_methods.png`
- `generated/figures/Fig_ablation.png`
- `generated/figures/tree_hyperparameter_tuning.png`

## Data Source

Use the raw-source pipeline for the current project:

```bash
cd code
python3 reproducible_pipeline.py --sections all --quick
```

The default raw source directory is:

```text
raw_sources/
```

It contains:

```text
ATH_GO_GOSLIM.txt
kegg_pathway_names.txt
kegg_pathway_genes.txt
aracyc_pathways.20251021
source_manifest.csv
sha256sums.txt
```

The raw-source pipeline parses TAIR `ATH_GO_GOSLIM.txt`, KEGG REST pathway
tables, and the PMN/AraCyc pathway dump.  Duplicate TAIR gene-GO pairs are
collapsed before feature construction because ATH_GO_GOSLIM expands GO
annotations across GO slim categories.

For `--sections all`, the formal run has two stages. It first runs the
training-only ratio, model, GO feature-count, and model-specific tree-parameter
comparisons. Their choices and file hashes are saved in
`generated/data/frozen_selection_config.json`. The main held-out benchmark,
ablation, and SHAP analyses run only after that configuration has been validated.

## Evaluation Protocol

The 538 KEGG/AraCyc source records are retained as provenance. Records with
exactly the same normalized gene membership are represented by one of 512
gene-set modelling instances. Near-duplicate sets are not combined.

The primary benchmark splits those modelling instances before controls are
generated. KEGG/AraCyc proportions are preserved with a fixed seed-42 split.
Training controls are derived only from training pathways, and test controls
are derived only from test pathways.

The 60 GO-frequency terms are selected once using the primary training records:

```text
label-free background frequency filter
-> variance filter on primary training records
-> mutual-information ranking on primary training labels
-> top 60 GO terms
```

The primary model is an equal-weight soft-voting ensemble of Random Forest,
Extra Trees, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Each component
uses the same 60-term representation and the final score is the arithmetic mean
of the six positive-class probabilities. The comparison contains 12 base
classifiers, while the primary ensemble contains the six tree components.

Model-family comparison and ratio comparison are performed only inside outer
training. The ratio comparison uses Logistic Regression, Random Forest, and
XGBoost on the same folds, negative samples, and fold-selected GO terms. Each
fold independently generates controls and repeats variance and
mutual-information selection on fold training. The 12 base classifiers use the
same fold-specific representation in the model comparison, and the six-tree
ensemble score is derived from the six component predictions. The outer test
set is not used to choose a model or class ratio.

After the ratio, model family, and 60-term representation are fixed, each of the
six tree components is evaluated over a 12-configuration model-specific grid.
All candidates reuse the same 15 training-only folds, negative samples, and
fold-specific GO representations. One parameter set per component is selected
by mean AUROC, followed by AUPRC, F1, AUROC stability, and a fixed configuration
identifier for exact ties. Runtime is recorded but is not a selection criterion.
The selected parameter sets are frozen before the outer test is evaluated.

For interpretation, Tree SHAP is computed for each of the six components on a
common outer-training background with probability output. Signed local
attributions and base values are averaged across components before global mean
absolute importance is calculated. Additivity is checked for every component
and for the averaged ensemble explanation.

An optional cached-data mode is available for comparison with archived inputs:

```bash
python3 reproducible_pipeline.py --dataset-source cached --sections all --quick
```

The paper results use raw-source mode rather than cached-data mode.

## Quick vs Full

`--quick` is for checking that the code and data flow work. It uses smaller
models and three folds, and writes to `/tmp/pathwayml_grouped_cv_quick`.

For heavier paper-style reruns, omit `--quick`:

```bash
python3 reproducible_pipeline.py --sections all
```

Full mode writes to `generated/`. Quick runs remain isolated under `/tmp` and
cannot overwrite the paper results.

## Ratio Comparison

The `ratio` section compares Logistic Regression, Random Forest, and XGBoost at
1:1 through 1:5 only within outer training by three-repeat, five-fold
cross-validation in full mode. Each ratio's folds, negative samples, and
fold-selected 60 GO terms are shared across the three classifiers. The detailed
table reports raw AUPRC, its random prevalence baseline, normalized AUPRC, F1,
and the other diagnostic metrics for each model. A second table gives descriptive
means across the three models; it is not a fitted ensemble. The normalization is
useful for context but does not make PR performance completely independent of
prevalence. The code records 1:1 as the preselected candidate and does not select
a winner automatically.

## Reproducibility Inputs

The input inventory is documented by `raw_sources/source_manifest.csv` and
`raw_sources/sha256sums.txt`. Generated JSON files under `generated/data/` are
reproducibility artifacts rather than primary input data.
