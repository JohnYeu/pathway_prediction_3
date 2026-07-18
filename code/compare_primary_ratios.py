#!/usr/bin/env python3
"""Compare independently trained 1:1 through 1:5 primary-ratio pipelines.

Each point in the comparison comes from a complete formal branch.  The branch
generates its own controls and selects its own 60 GO terms from its training
records before fitting XGBoost.  This is therefore a comparison of complete
primary configurations, not a fixed-feature resampling experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RATIOS = (1, 2, 3, 4, 5)
DEFAULT_RUN_ROOT = ROOT / "ratio_runs"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "comparison_all_ratios"
TABLE_NAME = "primary_ratio_performance_comparison.csv"
FIGURE_NAME = "primary_ratio_performance_comparison.png"
REPORT_NAME = "primary_ratio_comparison_report.md"


def parse_args() -> argparse.Namespace:
    """Parse branch, comparison, and optional manuscript publication paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help=(
            "Optional formal result directory. When supplied, the comparison "
            "table and Figure 7 alias are also written there."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    """Read a required JSON artifact with a useful missing-file error."""

    if not path.exists():
        raise FileNotFoundError(f"Required result artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact with a useful missing-file error."""

    if not path.exists():
        raise FileNotFoundError(f"Required result table is missing: {path}")
    return pd.read_csv(path)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one comparison input or output."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    """Prefer a path relative to the code directory in provenance metadata."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def xgboost_row(table: pd.DataFrame) -> pd.Series:
    """Return the unique XGBoost row from a generated benchmark table."""

    rows = table.loc[table["model"] == "XGBoost"]
    if len(rows) != 1:
        raise ValueError(f"Expected one XGBoost row, found {len(rows)}")
    return rows.iloc[0]


def positive_ids(split: dict, side: str) -> tuple[str, ...]:
    """Return sorted positive pathway IDs for one side of the outer split."""

    records = split[f"{side}_positive_records"]
    return tuple(sorted(str(record["id"]) for record in records))


def id_hash(ids: tuple[str, ...]) -> str:
    """Hash a sorted pathway-ID list for compact split comparison."""

    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def selected_go_terms(branch: Path) -> tuple[str, ...]:
    """Read and validate the 60 training-selected GO terms for one branch."""

    terms = read_json(branch / "data" / "selected_go_terms.json")
    if len(terms) != 60:
        raise ValueError(f"Expected 60 selected GO terms in {branch}, found {len(terms)}")
    return tuple(sorted(str(term) for term in terms))


def collect_branch(
    branch: Path,
    ratio: int,
) -> tuple[dict, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Read and cross-check one independently trained formal ratio branch."""

    split = read_json(branch / "data" / "main_split.json")
    train_ids = positive_ids(split, "train")
    test_ids = positive_ids(split, "test")
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Positive train/test overlap detected in {branch}")

    main = xgboost_row(read_table(branch / "tables" / "main_benchmark.csv"))
    sensitivity = read_table(branch / "tables" / "ratio_sensitivity.csv")
    own = sensitivity.loc[sensitivity["ratio"] == f"1:{ratio}"]
    if len(own) != 1:
        raise ValueError(f"Missing unique 1:{ratio} sensitivity row in {branch}")
    own = own.iloc[0]

    # These rows describe the same fitted primary configuration.  Equality is a
    # useful guard against comparing partially overwritten branch directories.
    metric_names = (
        "cv_auroc_mean",
        "test_auroc",
        "test_auprc",
        "f1",
        "precision",
        "recall",
        "brier",
    )
    for metric in metric_names:
        if abs(float(main[metric]) - float(own[metric])) > 1e-12:
            raise ValueError(f"Main/sensitivity mismatch for 1:{ratio} {metric}")

    train_records = split["train_records"]
    test_records = split["test_records"]
    n_train_pos = sum(int(record["label"]) == 1 for record in train_records)
    n_train_neg = len(train_records) - n_train_pos
    n_test_pos = sum(int(record["label"]) == 1 for record in test_records)
    n_test_neg = len(test_records) - n_test_pos
    prevalence = n_test_pos / (n_test_pos + n_test_neg)
    auprc = float(main["test_auprc"])

    row = {
        "ratio": f"1:{ratio}",
        "primary_ratio": ratio,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_test_pos": n_test_pos,
        "n_test_neg": n_test_neg,
        "cv_auroc_mean": float(main["cv_auroc_mean"]),
        "cv_auroc_sd": float(main["cv_auroc_sd"]),
        "test_auroc": float(main["test_auroc"]),
        "raw_auprc": auprc,
        "random_auprc_baseline": prevalence,
        "auprc_minus_baseline": auprc - prevalence,
        "normalized_auprc": (auprc - prevalence) / (1.0 - prevalence),
        "f1": float(main["f1"]),
        "precision": float(main["precision"]),
        "recall": float(main["recall"]),
        "brier": float(main["brier"]),
        "train_positive_ids_sha256": id_hash(train_ids),
        "test_positive_ids_sha256": id_hash(test_ids),
        "main_matches_own_sensitivity_row": True,
    }
    return row, selected_go_terms(branch), train_ids, test_ids


def go_overlap_table(go_by_ratio: dict[int, tuple[str, ...]]) -> pd.DataFrame:
    """Summarise pairwise overlap among the five selected-GO representations."""

    rows = []
    for left, right in combinations(RATIOS, 2):
        left_terms = set(go_by_ratio[left])
        right_terms = set(go_by_ratio[right])
        intersection = len(left_terms & right_terms)
        union = len(left_terms | right_terms)
        rows.append(
            {
                "ratio_a": f"1:{left}",
                "ratio_b": f"1:{right}",
                "n_selected_a": len(left_terms),
                "n_selected_b": len(right_terms),
                "n_shared": intersection,
                "jaccard": intersection / union if union else 1.0,
            }
        )
    return pd.DataFrame(rows)


def plot_primary_ratio_performance(results: pd.DataFrame, output: Path) -> None:
    """Plot AUROC and prevalence-normalized AUPRC for complete ratio runs."""

    x = list(range(len(results)))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        x,
        results["test_auroc"],
        color="#1f77b4",
        marker="o",
        linewidth=2.4,
        label="AUROC",
    )
    ax.plot(
        x,
        results["normalized_auprc"],
        color="#d62728",
        marker="o",
        linewidth=2.4,
        label="Normalized AUPRC",
    )
    ax.set_xticks(x, results["ratio"])
    ax.set_xlabel("Positive:negative ratio")
    ax.set_ylabel("Held-out metric")
    ax.set_ylim(0.40, 0.90)
    ax.set_title("Positive-to-negative ratio comparison")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.8, alpha=0.7)
    ax.legend(loc="lower left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(results: pd.DataFrame, overlap: pd.DataFrame, output: Path) -> None:
    """Write a compact, prevalence-aware interpretation of the comparison."""

    best_auroc = results.loc[results["test_auroc"].idxmax()]
    best_normalized = results.loc[results["normalized_auprc"].idxmax()]
    lines = [
        "# Primary positive-to-negative ratio comparison",
        "",
        "Each ratio was independently trained. Controls and the 60 GO terms were selected from that ratio's own training records.",
        "",
        "| Ratio | AUROC | Raw AUPRC | Random baseline | Normalized AUPRC | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.ratio} | {row.test_auroc:.3f} | {row.raw_auprc:.3f} | "
            f"{row.random_auprc_baseline:.3f} | {row.normalized_auprc:.3f} | {row.f1:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- Positive train/test pathway IDs are identical across all five branches.",
            "- Every branch's main XGBoost metrics equal its corresponding sensitivity row.",
            f"- Highest AUROC: {best_auroc['ratio']} ({best_auroc['test_auroc']:.3f}).",
            f"- Highest normalized AUPRC: {best_normalized['ratio']} ({best_normalized['normalized_auprc']:.3f}).",
            f"- Pairwise GO overlap ranges from {int(overlap['n_shared'].min())} to {int(overlap['n_shared'].max())} of 60 terms.",
            "",
            "The comparison describes complete pipelines. Normalized AUPRC adjusts for the different positive prevalence, but the held-out negative sets also differ in size. The results therefore support a practical ratio choice for this benchmark rather than a universal optimum.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    results: pd.DataFrame,
    overlap: pd.DataFrame,
    output: Path,
    figure_name: str = FIGURE_NAME,
) -> None:
    """Write one complete set of table, figure, report, and audit artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    table_path = output / TABLE_NAME
    overlap_path = output / "selected_go_pairwise_overlap.csv"
    figure_path = output / figure_name
    report_path = output / REPORT_NAME
    results.to_csv(table_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    plot_primary_ratio_performance(results, figure_path)
    write_report(results, overlap, report_path)


def main() -> None:
    """Validate all branches and generate the formal ratio comparison."""

    args = parse_args()
    run_root = args.run_root.resolve()
    output = args.out_dir.resolve()

    rows = []
    go_by_ratio: dict[int, tuple[str, ...]] = {}
    branch_inputs = []
    reference_train_ids: tuple[str, ...] | None = None
    reference_test_ids: tuple[str, ...] | None = None
    for ratio in RATIOS:
        branch = run_root / f"ratio_1_{ratio}"
        row, selected_go, train_ids, test_ids = collect_branch(branch, ratio)
        if reference_train_ids is None:
            reference_train_ids = train_ids
            reference_test_ids = test_ids
        elif train_ids != reference_train_ids or test_ids != reference_test_ids:
            raise ValueError(f"Positive split differs in branch 1:{ratio}")
        rows.append(row)
        go_by_ratio[ratio] = selected_go
        branch_inputs.append(
            {
                "ratio": f"1:{ratio}",
                "branch": project_path(branch),
                "main_benchmark_sha256": sha256_file(branch / "tables" / "main_benchmark.csv"),
                "selected_go_sha256": sha256_file(branch / "data" / "selected_go_terms.json"),
            }
        )

    results = pd.DataFrame(rows)
    overlap = go_overlap_table(go_by_ratio)
    write_outputs(results, overlap, output)

    published = None
    if args.publish_dir is not None:
        publish_dir = args.publish_dir.resolve()
        table_dir = publish_dir / "tables"
        figure_dir = publish_dir / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(table_dir / TABLE_NAME, index=False)
        overlap.to_csv(table_dir / "primary_ratio_selected_go_overlap.csv", index=False)
        plot_primary_ratio_performance(results, figure_dir / "Fig7_robustness.png")
        published = {
            "result_directory": project_path(publish_dir),
            "table": f"tables/{TABLE_NAME}",
            "figure": "figures/Fig7_robustness.png",
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "independently_trained_primary_ratio_comparison",
        "ratios": [f"1:{ratio}" for ratio in RATIOS],
        "feature_selection": "60 GO terms selected independently from each branch's training records",
        "normalized_auprc_formula": "(AUPRC - positive_prevalence) / (1 - positive_prevalence)",
        "branch_inputs": branch_inputs,
        "output_table": TABLE_NAME,
        "output_figure": FIGURE_NAME,
        "published": published,
    }
    (output / "primary_ratio_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Wrote comparison artifacts to {output}")
    if published:
        print(f"Published manuscript artifacts to {args.publish_dir.resolve()}")


if __name__ == "__main__":
    main()
