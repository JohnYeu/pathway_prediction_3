#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the three held-out metric diagnostic figures for the reference model.

This script produces held-out diagnostic figures for the reference model:

    FigC_roc_pr.png        ROC curve and precision--recall curve
    FigF_confusion.png     confusion matrix at the 0.5 threshold
    FigG_score_by_type.png held-out score distribution by gene-set type

To guarantee the figures match the numbers reported in the paper, the script does
not retrain with fresh settings. It reuses the functions in ``reproducible_pipeline``
to rebuild the *exact* seed-42, 1:2 held-out split used for the main benchmark, then
cross-checks the recomputed AUROC/AUPRC against
``generated/tables/table_main_benchmark.csv`` and aborts if they have drifted.

Usage:
    python make_metric_figures.py
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: no display needed when only saving PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix, f1_score)

import reproducible_pipeline as P

# Colour palette shared with the other manuscript figures (Vega-10 subset).
BLUE, ORANGE, TEAL = "#4c78a8", "#f58518", "#72b7b2"

# Scientific outputs stay under code/generated/. Manuscript asset management is
# handled separately from the analysis repository.
ROOT = Path(__file__).resolve().parent
GENERATED_FIGURES = ROOT / "generated" / "figures"
GENERATED_TABLES = ROOT / "generated" / "tables"
METRIC_FIGURES = ["FigC_roc_pr.png", "FigF_confusion.png", "FigG_score_by_type.png"]


def output_dirs() -> list[Path]:
    """Return the scientific output directory for generated figures."""
    GENERATED_FIGURES.mkdir(parents=True, exist_ok=True)
    return [GENERATED_FIGURES]


def save_figure(fig: plt.Figure, name: str, dirs: list[Path]) -> None:
    """Save one figure under ``name`` to every requested output directory."""
    for directory in dirs:
        fig.savefig(directory / name, dpi=200)


def main() -> None:
    """Rebuild the seed-42 held-out predictions and render the three figures."""
    dirs = output_dirs()

    # --- Reproduce the main held-out benchmark exactly (XGBoost, seed 42, 1:2 ratio) ---
    # Rebuild the same pathway-level split and training-selected 60-term
    # representation used by run_main_benchmark().  No figure-specific split or
    # feature selection is allowed here.
    bundle = P.load_raw_bundle(P.DEFAULT_RAW_DIR)
    context = P.prepare_primary_context(bundle, ratio=2)
    spw = P.scale_pos_weight_from_labels(context.y_train, "metric_figures", {"ratio": "1:2"})
    model = P.xgb_model(scale_pos_weight=spw, fast=False)
    model.fit(context.X_train, context.y_train)
    scores = P.predict_scores(model, context.X_test)
    yt = context.y_test

    # Threshold-free scores drive AUROC/AUPRC; the 0.5-threshold labels drive F1
    # and the confusion matrix.
    auroc = roc_auc_score(yt, scores)
    auprc = average_precision_score(yt, scores)
    pred = (scores >= 0.5).astype(int)
    f1 = f1_score(yt, pred)

    # Guard against stale figures: the recomputed metrics must match the values
    # already recorded in the main benchmark table. If they differ, something
    # upstream (data, seed, model settings) has changed and the figures should not
    # be published, so fail loudly instead.
    benchmark_path = GENERATED_TABLES / "table_main_benchmark.csv"
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        benchmark = {row["model"]: row for row in csv.DictReader(handle)}
    exp_auroc = float(benchmark["XGBoost"]["test_auroc"])
    exp_auprc = float(benchmark["XGBoost"]["test_auprc"])
    print(f"REPRODUCED: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1@0.5={f1:.4f}  n_test={len(yt)}")
    print(f"TABLE     : AUROC={exp_auroc:.4f}  AUPRC={exp_auprc:.4f}  (table_main_benchmark.csv)")
    if abs(auroc - exp_auroc) >= 1e-3 or abs(auprc - exp_auprc) >= 1e-3:
        raise RuntimeError("Recomputed metrics differ from table_main_benchmark.csv; diagnostic figures would be stale.")

    # --- Figure C: ROC curve (left) and precision--recall curve (right) ---
    # The PR baseline is the positive fraction (yt.mean()), not 0.5, because the
    # held-out split is imbalanced (roughly one positive per two negatives).
    fpr, tpr, _ = roc_curve(yt, scores)
    prec, rec, _ = precision_recall_curve(yt, scores)
    baseline = yt.mean()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(fpr, tpr, color=BLUE, lw=2.2, label=f"XGBoost (AUROC = {auroc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Random (0.500)")
    ax[0].set_xlabel("False positive rate (FP / (FP+TN))")
    ax[0].set_ylabel("True positive rate / recall (TP / (TP+FN))")
    ax[0].set_title("ROC curve")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_xlim(-0.02, 1.02)
    ax[0].set_ylim(-0.02, 1.02)

    ax[1].plot(rec, prec, color=ORANGE, lw=2.2, label=f"XGBoost (AUPRC = {auprc:.3f})")
    ax[1].axhline(baseline, ls="--", color="#999999", lw=1, label=f"Random baseline ({baseline:.3f})")
    ax[1].set_xlabel("Recall (TP / (TP+FN))")
    ax[1].set_ylabel("Precision (TP / (TP+FP))")
    ax[1].set_title("Precision--recall curve")
    ax[1].legend(loc="lower left", fontsize=9)
    ax[1].set_xlim(-0.02, 1.02)
    ax[1].set_ylim(-0.02, 1.02)
    fig.suptitle("Held-out discrimination of the reference XGBoost model (seed-42 split)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "FigC_roc_pr.png", dirs)
    plt.close(fig)

    # --- Figure F: confusion matrix at the default 0.5 threshold ---
    cm = confusion_matrix(yt, pred)  # rows = actual class [0, 1], cols = predicted [0, 1]
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred. negative", "Pred. positive"])
    ax.set_yticks([0, 1], labels=["Actual control\n(y = 0)", "Actual pathway\n(y = 1)"])
    # Cell grid matches the row/col order above: [[TN, FP], [FN, TP]].
    cells = [[f"TN = {tn}", f"FP = {fp}"], [f"FN = {fn}", f"TP = {tp}"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cells[i][j],
                ha="center",
                va="center",
                # White text on the dark (high-count) cells, black on the light ones.
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_title("Confusion matrix (threshold = 0.5)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "FigF_confusion.png", dirs)
    plt.close(fig)

    print("saved: FigC_roc_pr.png, FigF_confusion.png  (TN=%d FP=%d FN=%d TP=%d)" % (tn, fp, fn, tp))

    # --- Figure G: held-out score distribution by gene-set type ---
    # Split the held-out scores into the curated positives and the four constructed
    # negative types, so the box plot shows which negatives are hardest to reject.
    te_records = context.test_records
    type_map = {
        "random_5_30": "Random",
        "shuffled": "Shuffled",
        "partial_50_80": "Partial\npathway",
        "cross_pathway": "Cross-pathway\nmixture",
    }
    order = ["Curated\npathway", "Random", "Shuffled", "Partial\npathway", "Cross-pathway\nmixture"]
    groups = {key: [] for key in order}
    for record, score in zip(te_records, scores, strict=True):
        label = "Curated\npathway" if record["label"] == 1 else type_map[record["negative_type"]]
        groups[label].append(float(score))
    data = [groups[key] for key in order]
    rng = np.random.default_rng(0)  # fixed seed so the jittered points are reproducible
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color="black", lw=1.5))
    # Positives teal, easy negatives grey, the two hard negative types orange.
    box_colors = [TEAL, "#c7c7c7", "#c7c7c7", ORANGE, ORANGE]
    for patch, color in zip(bp["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for i, values in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.05, size=len(values)), values, s=6, color="#333333", alpha=0.25, zorder=3)
    ax.axhline(0.5, ls="--", color="#999999", lw=1, label="Decision threshold (0.5)")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Model score $P(\\mathrm{pathway\\text{-}like})$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Held-out score distribution by gene-set type", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "FigG_score_by_type.png", dirs)
    plt.close(fig)

    medians = {key: (float(np.median(groups[key])) if groups[key] else float("nan")) for key in order}
    print("saved: FigG_score_by_type.png  medians:", {key.replace(chr(10), " "): round(value, 3) for key, value in medians.items()})

    # --- Provenance record: what was produced, from which split, and the key numbers ---
    # Lets the figures be traced back to this script and this exact run even when
    # they are viewed outside the repository.
    manifest = {
        "source_script": "make_metric_figures.py",
        "model": "XGBoost (reference)",
        "seed": int(P.SEED),
        "ratio": "1:2",
        "feature_selection_scope": "outer_train_only",
        "negative_source_isolation": True,
        "selected_go_sha256": P.stable_json_sha256(context.selected_go),
        "n_test": int(len(yt)),
        "reproduced": {
            "test_auroc": round(float(auroc), 6),
            "test_auprc": round(float(auprc), 6),
            "f1_at_0.5": round(float(f1), 6),
        },
        "cross_checked_against": "generated/tables/table_main_benchmark.csv",
        "confusion_at_0.5": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "score_medians_by_type": {
            # value == value is False only for NaN (empty group); store null in that case.
            key.replace(chr(10), " "): (round(float(value), 4) if value == value else None)
            for key, value in medians.items()
        },
        "figures": METRIC_FIGURES,
        "output_scope": "scientific_generated_figures_only",
    }
    (GENERATED_FIGURES.parent / "metric_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote provenance: generated/metric_figures_manifest.json")


if __name__ == "__main__":
    main()
