#!/usr/bin/env python3
"""Run a notebook-like 13-model benchmark using ATH_GO_GOSLIM.txt.

This is a comparison run for the new curated CSV.  It follows the professor
notebook more closely than `run_new_dataset_13_models.py`:

1. Parse `ATH_GO_GOSLIM.txt` directly.
2. Deduplicate gene-GO pairs.
3. Re-select 60 GO terms from this GO source.
4. Build 60 GO-frequency + 9 engineered features.
5. Evaluate the 13 notebook-style ML models on a seed-42 stratified split.

The default run saves held-out test metrics and all important intermediate
files.  Full repeated 5x3 CV can be enabled with `--with-repeated-cv`, but it is
slow for the heavier boosting and stacking models.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import fisher_exact
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from sklearn.ensemble import StackingClassifier
except ImportError:  # pragma: no cover - compatibility fallback
    StackingClassifier = None


SEED = 42
K_GO = 60
MIN_GO_GENES = 20
MAX_GO_PERCENT = 0.30
PAIRWISE_CAP = 15

AGI_RE = re.compile(r"^AT[1-5CM]G\d{5}$")
GO_RE = re.compile(r"^GO:\d+$")

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BENCHMARK_ROOT / "input" / "training_dataset_ratio_1_2_curated.csv"
DEFAULT_GO = BENCHMARK_ROOT / "input" / "ATH_GO_GOSLIM.txt"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "outputs_ath_goslim"

# Shared boosting hyperparameters for XGBoost/LightGBM/CatBoost
BOOSTING_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


class LegacyStackingClassifier:
    """Fallback stacking classifier for sklearn versions without StackingClassifier.

    Implements out-of-fold stacking: each base estimator produces OOF predictions
    via CV, which become meta-features for the final estimator.
    """

    def __init__(self, estimators: list[tuple[str, Any]], final_estimator: Any, cv: int = 5, random_state: int = SEED):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "estimators": self.estimators,
            "final_estimator": self.final_estimator,
            "cv": self.cv,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "LegacyStackingClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LegacyStackingClassifier":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y)
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=1, random_state=self.random_state)
        self.fitted_estimators_ = []
        meta_features = np.zeros((len(X_df), len(self.estimators)), dtype=float)

        for est_idx, (_, estimator) in enumerate(self.estimators):
            oof = np.zeros(len(X_df), dtype=float)
            for train_idx, test_idx in cv.split(X_df, y_arr):
                fitted = clone(estimator)
                fitted.fit(X_df.iloc[train_idx], y_arr[train_idx])
                oof[test_idx] = predict_positive_scores(fitted, X_df.iloc[test_idx])
            meta_features[:, est_idx] = oof
            fitted_full = clone(estimator)
            fitted_full.fit(X_df, y_arr)
            self.fitted_estimators_.append(fitted_full)

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(meta_features, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        meta_features = np.zeros((len(X_df), len(self.fitted_estimators_)), dtype=float)
        for est_idx, fitted in enumerate(self.fitted_estimators_):
            meta_features[:, est_idx] = predict_positive_scores(fitted, X_df)
        return self.final_estimator_.predict_proba(meta_features)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_genes(raw: Any) -> list[str]:
    if isinstance(raw, list):
        genes = raw
    else:
        genes = ast.literal_eval(str(raw))
    return sorted({str(g).strip().upper() for g in genes if str(g).strip()})


def parse_go_annotations(go_file: Path) -> tuple[int, pd.DataFrame, dict[str, set[str]]]:
    """Parse ATH_GO_GOSLIM.txt into deduplicated gene-GO pairs.

    Returns (raw_record_count, deduplicated_pairs_df, gene_to_go_dict).
    Filters to valid AGI locus IDs (AT[1-5CM]G) and GO:NNNNNNN patterns.
    """
    records: list[tuple[str, str]] = []
    raw_record_count = 0
    with go_file.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("!"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            gene = cols[0].strip().upper()
            go_id = cols[5].strip()
            if AGI_RE.match(gene) and GO_RE.match(go_id):
                records.append((gene, go_id))
                raw_record_count += 1

    pairs = pd.DataFrame(records, columns=["Gene", "GO_ID"]).drop_duplicates()
    gene_to_go: dict[str, set[str]] = defaultdict(set)
    for gene, go_id in pairs.itertuples(index=False):
        gene_to_go[gene].add(go_id)
    return raw_record_count, pairs, dict(gene_to_go)


def build_go_binary_matrix(gene_to_go: dict[str, set[str]]) -> pd.DataFrame:
    """Build a binary gene x GO term matrix for feature selection."""
    genes = sorted(gene_to_go.keys())
    all_go = sorted({go for gos in gene_to_go.values() for go in gos})
    matrix = pd.DataFrame(0, index=genes, columns=all_go, dtype=np.uint8)
    for gene, gos in gene_to_go.items():
        matrix.loc[gene, list(gos)] = 1
    return matrix


def select_go_terms(X_go: pd.DataFrame, positive_gene_universe: set[str]) -> dict[str, Any]:
    """Three-stage GO feature selection: frequency -> variance -> MI.

    Uses gene-level MI (positive gene universe as labels) rather than
    pathway-level MI, following the professor notebook convention.
    """
    n_genes = X_go.shape[0]
    max_genes = int(n_genes * MAX_GO_PERCENT)

    go_counts = X_go.sum(axis=0)
    X_freq = X_go.loc[:, (go_counts >= MIN_GO_GENES) & (go_counts <= max_genes)]

    variances = X_freq.var(axis=0)
    variance_cutoff = float(variances.quantile(0.15))
    X_var = X_freq.loc[:, variances > variance_cutoff]

    y_gene = X_var.index.to_series().isin(positive_gene_universe).astype(np.uint8).values
    mi_scores = mutual_info_classif(X_var.values, y_gene, discrete_features=True, random_state=SEED)
    mi_series = pd.Series(mi_scores, index=X_var.columns).sort_values(ascending=False)
    selected_go = list(mi_series.head(K_GO).index)

    return {
        "selected_go": selected_go,
        "mi_series": mi_series,
        "n_all": int(X_go.shape[1]),
        "n_freq": int(X_freq.shape[1]),
        "n_var": int(X_var.shape[1]),
    }


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b)) / float(len(union)) if union else 0.0


def compute_entropy(values: list[float]) -> float:
    """Shannon entropy (base-2) of a non-negative value list; zeros are excluded."""
    arr = np.array([x for x in values if x > 0], dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(-np.sum(arr * np.log2(arr)))


def feature_names(selected_go: list[str]) -> list[str]:
    return ["gofreq::" + go for go in selected_go] + [
        "jaccard_mean",
        "jaccard_min",
        "jaccard_max",
        "jaccard_std",
        "gene_count",
        "log_gene_count",
        "go_entropy",
        "per_gene_go_count_std",
        "per_gene_go_count_mean",
    ]


def featurize_gene_set(gene_set: list[str], selected_go: list[str], gene_to_go: dict[str, set[str]]) -> list[float]:
    """Build a 69-dimensional feature vector: 60 GO freqs + 4 Jaccard + 5 size/entropy."""
    genes = sorted(set(gene_set))
    n = len(genes)
    per_gene_go_counts = [len(gene_to_go.get(gene, set())) for gene in genes]

    go_freqs = []
    for go in selected_go:
        count = sum(1 for gene in genes if go in gene_to_go.get(gene, set()))
        go_freqs.append(float(count) / float(n) if n > 0 else 0.0)

    # Match the notebook behavior: for large gene sets, sample 15 genes from
    # the set using Python's global random state.
    sampled = random.sample(genes, PAIRWISE_CAP) if len(genes) > PAIRWISE_CAP else genes
    sims = [jaccard(gene_to_go.get(g1, set()), gene_to_go.get(g2, set())) for g1, g2 in combinations(sampled, 2)]

    if sims:
        j_mean = float(np.mean(sims))
        j_min = float(np.min(sims))
        j_max = float(np.max(sims))
        j_std = float(np.std(sims))
    else:
        j_mean = j_min = j_max = j_std = 0.0

    return go_freqs + [
        j_mean,
        j_min,
        j_max,
        j_std,
        float(n),
        float(np.log1p(n)),
        compute_entropy(go_freqs),
        float(np.std(per_gene_go_counts)) if per_gene_go_counts else 0.0,
        float(np.mean(per_gene_go_counts)) if per_gene_go_counts else 0.0,
    ]


def build_feature_dataset(training_df: pd.DataFrame, selected_go: list[str], gene_to_go: dict[str, set[str]]) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_names(selected_go)
    rows = []
    for row in training_df.itertuples(index=False):
        features = featurize_gene_set(row.genes_list, selected_go, gene_to_go)
        record = {
            "sample_id": row.sample_id,
            "label": row.label,
            "strategy": row.strategy,
            "source": row.source,
            "genes": row.genes,
            "genes_list": row.genes_list,
        }
        record.update(dict(zip(columns, features)))
        rows.append(record)
    return pd.DataFrame(rows), columns


def make_xgb() -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=-1, **BOOSTING_PARAMS)


def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(objective="binary", random_state=SEED, n_jobs=-1, verbosity=-1, **BOOSTING_PARAMS)


def make_catboost() -> CatBoostClassifier:
    params = dict(BOOSTING_PARAMS)
    # CatBoost does not accept all XGBoost/LightGBM-style parameters used in
    # BOOSTING_PARAMS.  Keep the shared tree count/depth/learning-rate/subsample
    # settings and drop incompatible regularisation aliases.
    for key in ["colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]:
        params.pop(key, None)
    return CatBoostClassifier(loss_function="Logloss", random_seed=SEED, verbose=False, **params)


def get_models(include_stacking: bool = True) -> dict[str, Any]:
    """Build the full 13-model catalogue (12 ML + optional stacking).

    All models use Pipeline with SimpleImputer for NaN robustness.
    Models requiring normalized input (LR, SVM, KNN, MLP) include StandardScaler.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]),
        "LinearSVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED)),
        ]),
        "RBFSVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "GaussianNB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GaussianNB()),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", make_xgb()),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", make_lgbm()),
        ]),
        "CatBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", make_catboost()),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam", max_iter=500, random_state=SEED)),
        ]),
    }
    if include_stacking:
        estimators = [
            ("lr", models["LogisticRegression"]),
            ("rf", models["RandomForest"]),
            ("lgbm", models["LightGBM"]),
        ]
        if StackingClassifier is not None:
            models["Stacking_LR_RF_LightGBM_to_XGBoost"] = StackingClassifier(
                estimators=estimators,
                final_estimator=make_xgb(),
                stack_method="predict_proba",
                passthrough=False,
                n_jobs=-1,
            )
        else:
            models["Stacking_LR_RF_LightGBM_to_XGBoost"] = LegacyStackingClassifier(
                estimators=estimators,
                final_estimator=make_xgb(),
                cv=5,
                random_state=SEED,
            )
    return models


def predict_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def fit_predict_scores(model: Any, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    scores = predict_positive_scores(fitted, X_test)
    preds = (scores >= 0.5).astype(int)
    return scores, preds


def metric_summary(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "Precision": float(precision_score(y_true, preds, zero_division=0)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def repeated_cv(model: Any, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        scores, preds = fit_predict_scores(model, X.iloc[train_idx], y[train_idx], X.iloc[test_idx])
        metrics = metric_summary(y[test_idx], scores, preds)
        metrics["fold"] = fold_idx
        rows.append(metrics)
    return pd.DataFrame(rows)


def ora_best_score(gene_set: list[str], positive_rows: pd.DataFrame, background_size: int) -> float:
    """Compute the best -log10(p) Fisher exact enrichment against all known pathways (ORA baseline)."""
    gene_set_set = set(gene_set)
    best = 0.0
    for row in positive_rows.itertuples(index=False):
        pathway = set(row.genes_list)
        a = len(gene_set_set & pathway)
        b = len(gene_set_set - pathway)
        c = len(pathway - gene_set_set)
        d = background_size - a - b - c
        if d < 0:
            continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        best = max(best, -math.log10(max(pval, 1e-300)))
    return best


def evaluate_ora(meta_test: pd.DataFrame, positive_rows: pd.DataFrame, background_size: int) -> dict[str, float]:
    scores = np.array([ora_best_score(genes, positive_rows, background_size) for genes in meta_test["genes_list"]])
    preds = (scores >= np.median(scores)).astype(int)
    return metric_summary(meta_test["label"].values, scores, preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--go-file", type=Path, default=DEFAULT_GO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-repeated-cv", action="store_true", help="Run full 5x3 repeated CV for every model.")
    parser.add_argument("--skip-stacking", action="store_true", help="Skip stacking if only the non-stacking models are needed.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_df = pd.read_csv(args.input)
    training_df["genes_list"] = training_df["genes"].map(parse_genes)
    positive_df = training_df[training_df["label"] == 1].copy().reset_index(drop=True)
    positive_gene_universe = set(gene for genes in positive_df["genes_list"] for gene in genes)

    raw_go_count, pairs, gene_to_go = parse_go_annotations(args.go_file)
    pairs.to_csv(args.output_dir / "deduplicated_gene_go_pairs.csv", index=False)

    X_go = build_go_binary_matrix(gene_to_go)
    go_info = select_go_terms(X_go, positive_gene_universe)
    selected_go = go_info["selected_go"]

    feature_df, feat_cols = build_feature_dataset(training_df, selected_go, gene_to_go)
    feature_df.to_csv(args.output_dir / "training_dataset_ratio_1_2_curated_with_features.csv", index=False)
    feature_df[["sample_id", "label", "strategy", "source", "genes_list"]].to_csv(
        args.output_dir / "sample_metadata.csv", index=False
    )
    pd.DataFrame({"feature_name": feat_cols}).to_csv(args.output_dir / "feature_names.csv", index=False)
    pd.DataFrame({"Selected_GO": selected_go}).to_csv(args.output_dir / "Selected_GO_terms.csv", index=False)
    pd.DataFrame({"GO_Term": go_info["mi_series"].index, "MI_Score": go_info["mi_series"].values}).to_csv(
        args.output_dir / "GO_MI_scores.csv", index=False
    )

    X = feature_df[feat_cols]
    y = feature_df["label"].values.astype(int)
    X.to_csv(args.output_dir / "feature_matrix.csv", index=False)
    np.save(args.output_dir / "feature_matrix.npy", X.to_numpy(dtype=np.float32))
    np.save(args.output_dir / "labels.npy", y)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        feature_df[["sample_id", "label", "strategy", "source", "genes_list"]],
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    pd.DataFrame(
        {
            "sample_id": pd.concat([meta_train["sample_id"], meta_test["sample_id"]]).values,
            "label": np.concatenate([y_train, y_test]),
            "split": ["train"] * len(y_train) + ["test"] * len(y_test),
        }
    ).to_csv(args.output_dir / "train_test_split.csv", index=False)

    results = []
    prediction_rows = []
    for model_name, model in get_models(include_stacking=not args.skip_stacking).items():
        print(f"[model] {model_name}", flush=True)
        t0 = time.time()
        if args.with_repeated_cv:
            cv_df = repeated_cv(model, X_train, y_train)
            cv_df.insert(0, "method", model_name)
            cv_df.to_csv(args.output_dir / f"cv_folds_{model_name}.csv".replace("/", "_"), index=False)
            cv_auroc_mean = float(cv_df["AUROC"].mean())
            cv_auroc_sd = float(cv_df["AUROC"].std())
        else:
            cv_auroc_mean = float("nan")
            cv_auroc_sd = float("nan")

        scores, preds = fit_predict_scores(model, X_train, y_train, X_test)
        metrics = metric_summary(y_test, scores, preds)
        results.append(
            {
                "method": model_name,
                "cv_auroc_mean": cv_auroc_mean,
                "cv_auroc_sd": cv_auroc_sd,
                "test_auroc": metrics["AUROC"],
                "test_auprc": metrics["AUPRC"],
                "test_f1": metrics["F1"],
                "test_precision": metrics["Precision"],
                "test_recall": metrics["Recall"],
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "elapsed_s": float(time.time() - t0),
            }
        )
        prediction_rows.extend(
            {
                "method": model_name,
                "sample_id": sample_id,
                "true_label": int(label),
                "score_positive": float(score),
                "pred_label_0p5": int(pred),
            }
            for sample_id, label, score, pred in zip(meta_test["sample_id"], y_test, scores, preds)
        )
        pd.DataFrame(results).to_csv(args.output_dir / "method_comparison_results_partial.csv", index=False)

    ora_metrics = evaluate_ora(meta_test, positive_df, background_size=len(positive_gene_universe))
    results.append(
        {
            "method": "ORA",
            "cv_auroc_mean": float("nan"),
            "cv_auroc_sd": float("nan"),
            "test_auroc": ora_metrics["AUROC"],
            "test_auprc": ora_metrics["AUPRC"],
            "test_f1": ora_metrics["F1"],
            "test_precision": ora_metrics["Precision"],
            "test_recall": ora_metrics["Recall"],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "elapsed_s": float("nan"),
        }
    )

    results_df = pd.DataFrame(results).sort_values("test_auroc", ascending=False)
    results_df.to_csv(args.output_dir / "method_comparison_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "model_test_predictions.csv", index=False)

    summary = {
        "seed": SEED,
        "training_file": str(args.input),
        "training_file_sha256": sha256_file(args.input),
        "go_file": str(args.go_file),
        "go_file_sha256": sha256_file(args.go_file),
        "n_total": int(len(feature_df)),
        "n_positive": int((feature_df["label"] == 1).sum()),
        "n_negative": int((feature_df["label"] == 0).sum()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "go_raw_records": int(raw_go_count),
        "go_unique_pairs": int(pairs.shape[0]),
        "go_pipeline": {
            "all": int(go_info["n_all"]),
            "freq_filtered": int(go_info["n_freq"]),
            "variance_filtered": int(go_info["n_var"]),
            "selected": int(len(selected_go)),
        },
        "feature_dim": int(len(feat_cols)),
        "cv_scheme": "RepeatedStratifiedKFold(5 folds x 3 repeats)" if args.with_repeated_cv else "not run",
        "include_stacking": bool(not args.skip_stacking),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(results_df.to_string(index=False))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
