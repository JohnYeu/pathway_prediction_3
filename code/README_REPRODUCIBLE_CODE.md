# Reproducible Code Notes

This folder was originally a manuscript package: it contained LaTeX, cached
JSON data, and finished figures, but not a complete raw-data-to-paper pipeline.
`reproducible_pipeline.py` now rebuilds the main analysis from documented raw
source files under `raw_sources/`.

## What This Code Rebuilds

The pipeline regenerates the main intermediate products needed by the paper:

- dataset inventory: `generated/tables/pathway_inventory.csv`
- dataset summary: `generated/tables/dataset_summary.csv`
- main held-out benchmark: `generated/tables/table_main_benchmark.csv`
- 13-method comparison: `generated/tables/table_method_comparison.csv`
- negative-ratio sensitivity: `generated/tables/table_robustness.csv`
- leave-one-family-out validation: `generated/tables/table_lofo.csv`
- ablation study: `generated/tables/table_ablation.csv`
- feature importance / SHAP fallback: `generated/tables/table_top_features.csv`
- primary split audit: `generated/tables/main_split_audit.csv`
- negative-source metadata: `generated/tables/negative_metadata.csv`
- GO-selection audit: `generated/tables/go_selection_audit.csv`
- CV split audit: `generated/tables/cv_split_audit.csv`
- LaTeX table fragments: `generated/tables/latex/*.tex`

It also regenerates paper-facing replacement figures:

- `generated/figures/Fig3_shap.png`
- `generated/figures/Fig5_datastats.png`
- `generated/figures/Fig6_lofo.png`
- `generated/figures/Fig7_robustness.png`
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

The primary benchmark splits curated positive pathways before controls are
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

All reference models and supplementary classifiers use this same fixed feature
representation. Repeated CV regenerates controls independently on each fold
side, but does not repeat GO selection inside each fold. LOFO likewise uses the
fixed primary training-selected representation and is therefore not a nested
feature-selection analysis.

An optional cached-data mode is available for comparison with archived inputs:

```bash
python3 reproducible_pipeline.py --dataset-source cached --sections all --quick
```

The paper results use raw-source mode rather than cached-data mode.

## Quick vs Full

`--quick` is for checking that the code and data flow work. It uses smaller
models and lighter cross-validation, and writes to `generated_quick/` so it
cannot overwrite the formal outputs in `generated/`.

For heavier paper-style reruns, omit `--quick`:

```bash
python3 reproducible_pipeline.py --sections all
```

Full mode takes substantially longer because it reruns boosted trees,
cross-validation, model comparison, LOFO, and feature importance.

## Reproducibility Inputs

The input inventory is documented by `raw_sources/source_manifest.csv` and
`raw_sources/sha256sums.txt`. Generated JSON files under `generated/data/` are
reproducibility artifacts rather than primary input data.
