#!/usr/bin/env python3
"""Evaluate each negative-sample type separately.

This script turns the one-off "which negative type is hard for ML?" check into
an explicit, reproducible analysis step for the thesis project.

Input:
    outputs_ath_goslim/feature_matrix.csv
    outputs_ath_goslim/sample_metadata.csv

These files are produced by `run_new_dataset_13_models_ath_goslim.py`.  They
already contain the fixed 69-dimensional feature matrix:

    60 GO-frequency features + 9 engineered features

For each negative strategy, this script builds a binary benchmark containing:

    all positive pathways + only one negative-sample type

Then it trains the 13 notebook-style ML models and reports held-out test
performance.  ORA is included as a conventional enrichment baseline, but it is
not an ML model.

The purpose is not to replace the mixed-negative benchmark.  The purpose is to
measure difficulty: random and shuffled negatives should be easy, whereas
partial pathways and cross-pathway mixtures are expected to be harder because
they retain real pathway-like signal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# Reuse the professor-notebook-like feature/model definitions so this analysis
# stays aligned with the main ATH_GO_GOSLIM benchmark.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_new_dataset_13_models_ath_goslim as bench  # noqa: E402


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / "outputs_ath_goslim"
DEFAULT_FEATURE_MATRIX = DEFAULT_OUTPUT_DIR / "feature_matrix.csv"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "sample_metadata.csv"

NEGATIVE_TYPES = [
    "random_gene_set",
    "shuffled_pathway",
    "partial_pathway",
    "cross_pathway_mixture",
]


def load_inputs(feature_matrix_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed feature matrix and sample metadata.

    The feature matrix and metadata must be row-aligned.  The metadata file
    stores `genes_list` as a string representation of a Python list, so it is
    parsed back into a sorted gene list for ORA scoring.
    """
    if not feature_matrix_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feature_matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    feature_matrix = pd.read_csv(feature_matrix_path)
    metadata = pd.read_csv(metadata_path)

    if len(feature_matrix) != len(metadata):
        raise ValueError(
            "feature_matrix.csv and sample_metadata.csv must have the same number of rows "
            f"({len(feature_matrix)} vs {len(metadata)})"
        )

    required = {"sample_id", "label", "strategy", "genes_list"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"sample_metadata.csv is missing required columns: {missing}")

    metadata = metadata.copy()
    metadata["genes_list"] = metadata["genes_list"].map(bench.parse_genes)
    metadata["label"] = metadata["label"].astype(int)
    return feature_matrix, metadata


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit one ML model and return metric summary plus per-sample predictions."""
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    scores = bench.predict_positive_scores(fitted, X_test)
    predictions = (scores >= 0.5).astype(int)
    metrics = bench.metric_summary(y_test, scores, predictions)
    return metrics, scores, predictions


def run_one_negative_type(
    negative_type: str,
    feature_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    models: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train/evaluate all models for one negative-sample type.

    Every run uses:

        positives: all label=1 rows
        negatives: only rows whose `strategy` equals `negative_type`

    This design directly answers: "If this were the only negative class, how
    hard would it be for ML/ORA to separate it from curated positives?"
    """
    mask = (metadata["label"] == 1) | ((metadata["label"] == 0) & (metadata["strategy"] == negative_type))
    indices = np.flatnonzero(mask.to_numpy())
    if len(indices) == 0:
        raise ValueError(f"No samples found for negative type: {negative_type}")

    X_sub = feature_matrix.iloc[indices].reset_index(drop=True)
    meta_sub = metadata.iloc[indices].reset_index(drop=True)
    y = meta_sub["label"].to_numpy(dtype=int)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both positives and negatives for {negative_type}; got n_pos={n_pos}, n_neg={n_neg}")

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    X_train = X_sub.iloc[train_idx]
    X_test = X_sub.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    meta_test = meta_sub.iloc[test_idx].copy()

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    # ORA baseline: score each held-out gene set by its best overlap enrichment
    # against the complete positive-pathway catalogue.
    positive_rows = metadata[metadata["label"] == 1].copy()
    background_size = len({gene for genes in metadata["genes_list"] for gene in genes})
    start = time.time()
    ora_metrics = bench.evaluate_ora(meta_test, positive_rows, background_size)
    rows.append(
        {
            "negative_type": negative_type,
            "method": "ORA",
            **ora_metrics,
            "n_total": int(len(y)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "elapsed_s": time.time() - start,
            "status": "ok",
        }
    )
    print(
        f"[{negative_type}] ORA AUROC={ora_metrics['AUROC']:.3f} "
        f"AUPRC={ora_metrics['AUPRC']:.3f}",
        flush=True,
    )

    for model_name, model in models.items():
        # Reset global RNGs before every model.  The estimators also carry their
        # own random_state/random_seed where supported.
        random.seed(seed)
        np.random.seed(seed)

        start = time.time()
        try:
            metrics, scores, predictions = evaluate_model(model, X_train, y_train, X_test, y_test)
            status = "ok"
        except Exception as exc:  # Keep the run auditable even if one model fails.
            metrics = {"AUROC": np.nan, "AUPRC": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan}
            scores = np.full(len(y_test), np.nan)
            predictions = np.full(len(y_test), -1)
            status = f"failed: {type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        rows.append(
            {
                "negative_type": negative_type,
                "method": model_name,
                **metrics,
                "n_total": int(len(y)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "elapsed_s": elapsed,
                "status": status,
            }
        )

        for sample_id, label, strategy, score, pred in zip(
            meta_test["sample_id"],
            y_test,
            meta_test["strategy"],
            scores,
            predictions,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "negative_type": negative_type,
                    "method": model_name,
                    "sample_id": sample_id,
                    "label": int(label),
                    "strategy": strategy,
                    "score": float(score) if not np.isnan(score) else np.nan,
                    "pred_label_0p5": int(pred),
                }
            )

        print(
            f"[{negative_type}] {model_name} AUROC={metrics['AUROC']:.3f} "
            f"AUPRC={metrics['AUPRC']:.3f} F1={metrics['F1']:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    return rows, prediction_rows


def make_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create a compact thesis-friendly table with XGBoost and best-ML rows."""
    ml_results = results[results["method"] != "ORA"].copy()
    best_rows = (
        ml_results.sort_values(["negative_type", "AUROC", "AUPRC"], ascending=[True, False, False])
        .groupby("negative_type", as_index=False)
        .head(1)
    )

    xgb = results[results["method"] == "XGBoost"].set_index("negative_type")
    best = best_rows.set_index("negative_type")
    ora = results[results["method"] == "ORA"].set_index("negative_type")

    rows = []
    for negative_type in NEGATIVE_TYPES:
        rows.append(
            {
                "negative_type": negative_type,
                "xgboost_auroc": round(float(xgb.loc[negative_type, "AUROC"]), 3),
                "xgboost_auprc": round(float(xgb.loc[negative_type, "AUPRC"]), 3),
                "ora_auroc": round(float(ora.loc[negative_type, "AUROC"]), 3),
                "ora_auprc": round(float(ora.loc[negative_type, "AUPRC"]), 3),
                "best_ml_model": str(best.loc[negative_type, "method"]),
                "best_ml_auroc": round(float(best.loc[negative_type, "AUROC"]), 3),
                "best_ml_auprc": round(float(best.loc[negative_type, "AUPRC"]), 3),
                "best_ml_f1": round(float(best.loc[negative_type, "F1"]), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=bench.SEED)
    parser.add_argument("--skip-stacking", action="store_true", help="Skip stacking for a faster diagnostic run.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix, metadata = load_inputs(args.feature_matrix, args.metadata)
    models = bench.get_models(include_stacking=not args.skip_stacking)

    all_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    for negative_type in NEGATIVE_TYPES:
        rows, prediction_rows = run_one_negative_type(
            negative_type=negative_type,
            feature_matrix=feature_matrix,
            metadata=metadata,
            models=models,
            seed=args.seed,
        )
        all_rows.extend(rows)
        all_prediction_rows.extend(prediction_rows)

    columns = [
        "negative_type",
        "method",
        "AUROC",
        "AUPRC",
        "F1",
        "Precision",
        "Recall",
        "n_total",
        "n_pos",
        "n_neg",
        "n_train",
        "n_test",
        "elapsed_s",
        "status",
    ]
    results = pd.DataFrame(all_rows)[columns]
    predictions = pd.DataFrame(all_prediction_rows)
    summary = make_summary_table(results)

    results_path = args.output_dir / "single_negative_type_model_results.csv"
    predictions_path = args.output_dir / "single_negative_type_test_predictions.csv"
    summary_path = args.output_dir / "single_negative_type_summary_for_paper.csv"
    run_summary_path = args.output_dir / "single_negative_type_run_summary.json"

    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    run_summary_path.write_text(
        json.dumps(
            {
                "analysis": "Each benchmark uses all positives plus one negative strategy; models are retrained per strategy.",
                "feature_matrix": str(args.feature_matrix),
                "metadata": str(args.metadata),
                "seed": args.seed,
                "test_size": 0.2,
                "negative_types": NEGATIVE_TYPES,
                "include_stacking": not args.skip_stacking,
                "outputs": {
                    "full_results": str(results_path),
                    "test_predictions": str(predictions_path),
                    "paper_summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  {results_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {run_summary_path}")
    print("\nPaper summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
