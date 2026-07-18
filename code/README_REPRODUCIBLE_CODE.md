# Reproducible Code Notes

This folder was originally a manuscript package: it contained LaTeX, cached
JSON data, and finished figures, but not a complete raw-data-to-paper pipeline.
`reproducible_pipeline.py` now rebuilds the main analysis from documented raw
source files under `raw_sources/`.

## What This Code Rebuilds

The current candidate pipeline regenerates the main intermediate products under
`generated_grouped_cv_candidate/`:

- source records and exact gene-set groups: `data/pathway_source_records.json`
  and `data/pathway_gene_set_groups.json`
- grouping audit: `tables/pathway_grouping_summary.csv` and
  `tables/pathway_duplicate_groups.csv`
- main held-out benchmark: `tables/main_benchmark.csv`
- training-only nested ratio comparison: `tables/ratio_cv_per_fold.csv` and
  `tables/ratio_cv_summary.csv`
- 13-method comparison: `tables/model_comparison.csv`
- leave-one-family-out validation: `tables/lofo.csv`
- ablation study: `tables/ablation.csv`
- feature importance: `tables/feature_importance.csv`
- split, negative-source, GO-selection, and CV audits under `tables/` and `data/`

It also creates candidate figures without changing the current manuscript:

- `generated/figures/Fig3_shap.png`
- `generated/figures/Fig5_datastats.png`
- `generated/figures/Fig6_lofo.png`
- `generated_grouped_cv_candidate/figures/Fig7_robustness.png`
- `generated/figures/Fig8_methods.png`
- `generated/figures/Fig_ablation.png`

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

Held-out reference models and supplementary classifiers use the same 60 terms
selected from the complete outer-training side. Ratio comparison is performed
only inside outer training. Each ratio/fold independently generates controls
and repeats variance and mutual-information selection on fold training. The
outer test set is evaluated only for the preselected 1:1 candidate. LOFO uses
the fixed outer-training representation and is not nested.

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

Full mode writes to `generated_grouped_cv_candidate/`. The preserved
pre-grouping result directories and LaTeX files are not overwritten.

## Ratio Comparison

The `ratio` section compares 1:1 through 1:5 only within outer training by
three-repeat, five-fold cross-validation in full mode. Each fold selects its own
60 GO terms. The table reports raw AUPRC, its random prevalence baseline, and
normalized AUPRC. The normalization is useful for context but does not make PR
performance completely independent of prevalence. The code records 1:1 as the
preselected candidate and does not select a winner automatically.

## Reproducibility Inputs

The input inventory is documented by `raw_sources/source_manifest.csv` and
`raw_sources/sha256sums.txt`. Generated JSON files under `generated/data/` are
reproducibility artifacts rather than primary input data.
