#!/usr/bin/env python3
"""Run the 13-model benchmark on the new curated CSV.

This script is intentionally isolated from the main PathwayML-Ath pipeline.
It does not regenerate positive/negative samples and does not overwrite any
existing paper tables.  The input CSV is treated as a fixed benchmark dataset:

    sample rows -> 69-dimensional PathwayML-Ath feature matrix -> 13 classifiers

Saved intermediates include the copied input, parsed sample metadata, feature
matrix, labels, train/test split, per-model predictions, metrics, model files,
and a manifest.  This makes the new-data benchmark rerunnable without
touching the original project outputs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BENCHMARK_ROOT / "input" / "training_dataset_ratio_1_2_curated.csv"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "outputs"

# Import the feature construction and model definitions used by the project
# reproducible paper code.  Keeping this as an import avoids silently drifting
# from the project feature logic.
sys.path.insert(0, str(PROJECT_ROOT))
from reproducible_pipeline import (  # noqa: E402
    build_feature_vector,
    load_cached_bundle,
    model_catalog,
    predict_scores,
)


SEED = 42


@dataclass
class DatasetSummary:
    input_file: str
    input_sha256: str
    n_samples: int
    n_positive: int
    n_negative: int
    positive_sources: dict[str, int]
    strategies: dict[str, int]
    feature_source: str
    n_features: int
    n_selected_go: int
    n_train: int
    n_test: int
    random_seed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gene_list(value: Any) -> list[str]:
    """Parse the CSV genes column into a stable, unique list of gene IDs."""
    if isinstance(value, list):
        genes = value
    elif pd.isna(value):
        genes = []
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                genes = list(parsed)
            else:
                genes = [text]
        except (SyntaxError, ValueError):
            # Fallback for comma/semicolon separated gene lists.
            for sep in [";", ",", " "]:
                if sep in text:
                    genes = [g for g in text.replace(";", ",").replace(" ", ",").split(",") if g]
                    break
            else:
                genes = [text] if text else []
    return sorted({str(g).strip().upper() for g in genes if str(g).strip()})


def positive_class_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute probability and threshold metrics for the held-out test set."""
    pred = (scores >= 0.5).astype(int)
    return {
        "test_auroc": float(roc_auc_score(y_true, scores)),
        "test_auprc": float(average_precision_score(y_true, scores)),
        "test_accuracy": float(accuracy_score(y_true, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "test_f1": float(f1_score(y_true, pred, zero_division=0)),
        "test_precision": float(precision_score(y_true, pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, pred, zero_division=0)),
        "test_brier": float(brier_score_loss(y_true, scores)),
    }


def cv_metrics(model: Any, X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> dict[str, float]:
    """Run stratified CV on the training fold only and summarize AUROC/AUPRC."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y):
        estimator = clone(model)
        estimator.fit(X[train_idx], y[train_idx])
        scores = predict_scores(estimator, X[valid_idx])
        aurocs.append(float(roc_auc_score(y[valid_idx], scores)))
        auprcs.append(float(average_precision_score(y[valid_idx], scores)))
    return {
        "cv_auroc_mean": float(np.mean(aurocs)),
        "cv_auroc_sd": float(np.std(aurocs, ddof=1)),
        "cv_auprc_mean": float(np.mean(auprcs)),
        "cv_auprc_sd": float(np.std(auprcs, ddof=1)),
        "cv_folds": folds,
    }


def build_sample_table(input_csv: Path, gene_go: dict[str, set[str]]) -> pd.DataFrame:
    """Read the curated CSV and add parsed gene lists and GO coverage statistics."""
    df = pd.read_csv(input_csv)
    required = {"sample_id", "label", "genes"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    parsed = df["genes"].apply(parse_gene_list)
    df = df.copy()
    df["parsed_genes"] = parsed.apply(lambda genes: ";".join(genes))
    df["parsed_gene_count"] = parsed.apply(len)
    df["go_annotated_gene_count"] = parsed.apply(lambda genes: sum(1 for g in genes if g in gene_go))
    df["go_coverage_fraction"] = df.apply(
        lambda row: row["go_annotated_gene_count"] / row["parsed_gene_count"] if row["parsed_gene_count"] else 0.0,
        axis=1,
    )
    return df


def save_feature_tables(
    X: np.ndarray,
    y: np.ndarray,
    sample_df: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Persist feature matrix in both NumPy and CSV form for reproducibility."""
    np.save(output_dir / "feature_matrix.npy", X)
    np.save(output_dir / "labels.npy", y)
    pd.DataFrame({"feature_index": range(len(feature_names)), "feature_name": feature_names}).to_csv(
        output_dir / "feature_names.csv", index=False
    )

    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df.insert(0, "label", y)
    feature_df.insert(0, "sample_id", sample_df["sample_id"].astype(str).values)
    feature_df.to_csv(output_dir / "feature_matrix.csv", index=False)


def simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without pandas' optional tabulate dependency."""
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def run_models(
    X: np.ndarray,
    y: np.ndarray,
    sample_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
    skip_cv_for: Iterable[str] = ("MLP", "LightGBM", "CatBoost"),
) -> pd.DataFrame:
    """Train all 13 models, save metrics/predictions/joblib snapshots.

    CV is skipped for MLP/LightGBM/CatBoost because their repeated CV is
    disproportionately slow.  Partial results are checkpointed after each
    model so an interrupted run still produces useful diagnostics.
    """
    split_train, split_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
    )
    split_df = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "sample_id": sample_df["sample_id"].astype(str).values,
            "label": y,
            "split": np.where(np.isin(np.arange(len(y)), split_test), "test", "train"),
        }
    )
    split_df.to_csv(output_dir / "train_test_split.csv", index=False)
    np.save(output_dir / "train_indices.npy", split_train)
    np.save(output_dir / "test_indices.npy", split_test)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in model_catalog(fast=False).items():
        print(f"[model] {model_name}", flush=True)
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            .replace("-", "_")
        )
        t0 = time.time()
        row: dict[str, Any] = {"model": model_name}
        if model_name in set(skip_cv_for):
            row.update({"cv_auroc_mean": np.nan, "cv_auroc_sd": np.nan, "cv_auprc_mean": np.nan, "cv_auprc_sd": np.nan, "cv_folds": 0})
        else:
            row.update(cv_metrics(model, X[split_train], y[split_train], seed=seed, folds=5))

        model.fit(X[split_train], y[split_train])
        test_scores = predict_scores(model, X[split_test])
        row.update(positive_class_metrics(y[split_test], test_scores))
        row["elapsed_s"] = float(time.time() - t0)
        metrics_rows.append(row)

        joblib.dump(model, models_dir / f"{safe_name}.joblib")
        pred_df = pd.DataFrame(
            {
                "model": model_name,
                "row_index": split_test,
                "sample_id": sample_df.iloc[split_test]["sample_id"].astype(str).values,
                "true_label": y[split_test],
                "score_positive": test_scores,
                "pred_label_0p5": (test_scores >= 0.5).astype(int),
            }
        )
        prediction_frames.append(pred_df)
        # Persist incrementally so an interrupted long run still leaves useful
        # partial diagnostics.  The final write below sorts the metrics table.
        pd.DataFrame(metrics_rows).to_csv(output_dir / "model_metrics_partial.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "model_test_predictions_partial.csv", index=False
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(["test_auroc", "test_auprc"], ascending=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "model_test_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="New curated CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for benchmark outputs.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    input_csv = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_input = output_dir / "input_training_dataset_ratio_1_2_curated.csv"
    shutil.copy2(input_csv, copied_input)

    bundle = load_cached_bundle()
    sample_df = build_sample_table(input_csv, bundle.gene_go)
    y = sample_df["label"].astype(int).to_numpy()
    genes_per_sample = [row.split(";") if row else [] for row in sample_df["parsed_genes"].tolist()]
    X = np.vstack(
        [build_feature_vector(genes, bundle.selected_go, bundle.gene_go) for genes in genes_per_sample]
    ).astype(np.float32)

    sample_df.to_csv(output_dir / "sample_metadata_with_go_coverage.csv", index=False)
    save_feature_tables(X, y, sample_df, bundle.feature_names, output_dir)

    metrics = run_models(X, y, sample_df, output_dir, seed=args.seed)

    summary = DatasetSummary(
        input_file=str(copied_input),
        input_sha256=sha256_file(copied_input),
        n_samples=int(len(y)),
        n_positive=int(y.sum()),
        n_negative=int((y == 0).sum()),
        positive_sources={k: int(v) for k, v in sample_df.loc[sample_df["label"] == 1, "source"].value_counts(dropna=False).items()},
        strategies={k: int(v) for k, v in sample_df["strategy"].value_counts(dropna=False).items()} if "strategy" in sample_df else {},
        feature_source="code/data_robustness cached GO feature set",
        n_features=int(X.shape[1]),
        n_selected_go=int(len(bundle.selected_go)),
        n_train=int((pd.read_csv(output_dir / "train_test_split.csv")["split"] == "train").sum()),
        n_test=int((pd.read_csv(output_dir / "train_test_split.csv")["split"] == "test").sum()),
        random_seed=int(args.seed),
    )
    (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    manifest = {
        "purpose": "13-model benchmark on new fixed curated 1:2 dataset",
        "project_root": str(PROJECT_ROOT),
        "benchmark_root": str(BENCHMARK_ROOT),
        "created_by_script": str(Path(__file__).resolve()),
        "seed": args.seed,
        "input": str(input_csv),
        "copied_input": str(copied_input),
        "outputs": {
            "sample_metadata": "sample_metadata_with_go_coverage.csv",
            "feature_matrix_npy": "feature_matrix.npy",
            "feature_matrix_csv": "feature_matrix.csv",
            "labels": "labels.npy",
            "feature_names": "feature_names.csv",
            "train_test_split": "train_test_split.csv",
            "metrics": "model_metrics.csv",
            "predictions": "model_test_predictions.csv",
            "models": "models/*.joblib",
        },
        "top_models_by_test_auroc": metrics[["model", "test_auroc", "test_auprc", "test_f1"]].head(5).to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = [
        "# New Dataset 13-Model Benchmark",
        "",
        "This folder contains an isolated benchmark run on the new",
        "`training_dataset_ratio_1_2_curated.csv`. It does not replace or overwrite",
        "the main PathwayML-Ath data or generated paper tables.",
        "",
        "Key outputs:",
        "- `model_metrics.csv`: 13-model performance table.",
        "- `model_test_predictions.csv`: held-out test scores for every model.",
        "- `feature_matrix.csv` / `feature_matrix.npy`: 69-dimensional feature matrix.",
        "- `sample_metadata_with_go_coverage.csv`: parsed gene sets and GO coverage.",
        "- `train_test_split.csv`: deterministic seed-42 split.",
        "- `models/*.joblib`: trained estimator snapshots.",
        "",
        "Top models by held-out AUROC:",
        "",
        simple_markdown_table(
            metrics[["model", "test_auroc", "test_auprc", "test_f1", "test_precision", "test_recall"]].head(13)
        ),
        "",
    ]
    (output_dir / "README_results.md").write_text("\n".join(readme), encoding="utf-8")

    print("\nDone. Results written to:", output_dir)
    print(metrics[["model", "test_auroc", "test_auprc", "test_f1", "test_precision", "test_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
