#!/usr/bin/env python3
"""Rebuild the PathwayML-Ath manuscript analyses from source data files.

This script rebuilds the analysis from raw source files rather than treating
cached JSON objects as primary data.  The raw source directory is expected to
contain:

* ATH_GO_GOSLIM.txt
* kegg_pathway_names.txt
* kegg_pathway_genes.txt
* aracyc_pathways.20251021
* source_manifest.csv
* sha256sums.txt

The pipeline:

1. rebuilds pathways and GO annotations from raw KEGG, AraCyc, and TAIR files;
2. splits positive pathways before generating split-specific controls;
3. selects 60 GO terms from the primary training records and constructs the
   69-dimensional representation described in the paper;
4. regenerates the four control classes used by this analysis
   (random[5,30], shuffled, partial, cross-pathway);
5. runs the main benchmark, ratio sensitivity, model comparison, ablation,
   and SHAP/importance summary;
6. saves machine-readable tables, figures, and provenance under an isolated
   result directory.

Raw mode is the default.  A cached mode is retained only for comparison with an
archived JSON data snapshot; it should not be used as the paper data source.

The code is designed for transparency, not maximum speed.  Use --quick for fast
validation runs and --full for manuscript analyses.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib

# Headless rendering avoids macOS GUI-backend failures in child processes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import catboost as cb
except Exception:  # pragma: no cover - optional dependency
    cb = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


# ---------------------------------------------------------------------------
# Project paths and global constants
# ---------------------------------------------------------------------------
#
# Generated outputs default to generated/.  Use --out-dir to write a fresh run
# into a separate subfolder for side-by-side comparisons.
#
ROOT = Path(__file__).resolve().parent
CACHED_DATA_DIR = ROOT / "data_robustness"
DEFAULT_RAW_DIR = ROOT / "raw_sources"
OUT_DIR = ROOT / "generated"
TABLE_DIR = OUT_DIR / "tables"
LATEX_TABLE_DIR = TABLE_DIR / "latex"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"

SEED = 42
# The formal manuscript configuration uses one constructed control per curated
# pathway.  Other ratios remain available as isolated sensitivity branches.
DEFAULT_PRIMARY_RATIO = 1
SUPPORTED_PRIMARY_RATIOS = (1, 2, 3, 4, 5)
N_GO_TERMS = 60
GO_FEATURE_COUNT_CANDIDATES = (20, 40, 60, 80, 100)
MIN_PATHWAY_GENES = 5
PRIMARY_TEST_SIZE = 0.20
PRIMARY_TRAIN_NEGATIVE_SEED = SEED + 100
PRIMARY_TEST_NEGATIVE_SEED = SEED + 200
CV_REPEAT_SEEDS = (42, 7, 13)
NEGATIVE_SCHEME = "four_type_random_shuffled_partial_cross"
GO_SELECTION_MODE = "outer_train_fixed60_with_nested_ratio_cv"
MODEL_COMPARISON_PROTOCOL = "heldout_only"
SCALE_POS_WEIGHT_RULE = "dynamic_n_negative_train_over_n_positive_train"
SCALE_POS_WEIGHT_LOG: List[Dict[str, Any]] = []

# Pairwise Jaccard uses every annotated gene for small and medium gene sets.
# Larger sets use a deterministic local sample so feature construction remains
# tractable without favouring low-sorting Arabidopsis locus identifiers.
JACCARD_EXACT_MAX_GENES = 30
JACCARD_SAMPLE_SIZE = 30
JACCARD_SAMPLING_SALT = "pathwayml-ath-jaccard-cap30-v1"


def configure_output_dir(out_dir: str | Path) -> None:
    """Redirect all generated output paths to a custom directory.

    Allows side-by-side comparison of runs (e.g. cached vs raw) by
    writing each run into an isolated folder.
    """
    global OUT_DIR, TABLE_DIR, LATEX_TABLE_DIR, FIG_DIR, DATA_DIR
    OUT_DIR = Path(out_dir).expanduser()
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
    TABLE_DIR = OUT_DIR / "tables"
    LATEX_TABLE_DIR = TABLE_DIR / "latex"
    FIG_DIR = OUT_DIR / "figures"
    DATA_DIR = OUT_DIR / "data"


@dataclass
class DatasetBundle:
    """Central data container shared across all analysis stages.

    Fields:
        pathways:       pid -> {name, genes, source}
        gene_go:        gene -> set of GO terms
        go_genes:       GO term -> set of genes (reverse index)
        go_term_names:  GO ID -> human-readable name
        selected_go:    60 GO terms surviving 4-stage feature selection
        feature_names:  ordered list of all 69 feature column names
        feature_stages: stage name -> count at each selection step
        source:         "raw" or "cached" indicating data provenance
    """

    pathways: Dict[str, Dict[str, Any]]
    source_pathways: Dict[str, Dict[str, Any]]
    gene_go: Dict[str, set]
    go_genes: Dict[str, set]
    go_term_names: Dict[str, str]
    selected_go: List[str]
    feature_names: List[str]
    feature_stages: Dict[str, int]
    source: str
    grouping_summary: Dict[str, Any] = field(default_factory=dict)
    duplicate_groups: List[Dict[str, Any]] = field(default_factory=list)
    mixed_source_groups: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NegativeSample:
    """One generated control set plus enough metadata to reproduce its origin.

    Source metadata is derived from pathway choices the generator already made;
    recording it must not consume additional random numbers.  Random controls
    have no source pathway, while shuffled/partial/cross controls have one or two.
    """

    genes: List[str]
    negative_type: str
    source_pathway_ids: List[str]
    source_overlap_count: int
    source_overlap_fraction: float
    generation_seed: int
    raw_gene_count: int
    annotated_gene_count: int
    source_raw_gene_count: int
    source_annotated_gene_count: int
    retention_fraction: float | None
    generation_attempt: int


@dataclass
class PrimaryContext:
    """Canonical outer split and training-selected feature representation.

    The positive pathways are split before any source-derived controls are made.
    GO variance/MI selection sees only ``train_records``.  The resulting 60 GO
    terms are then fixed for every paper analysis, including held-out comparison,
    ratio sensitivity, ablation, and SHAP.
    """

    selected_go: List[str]
    feature_names: List[str]
    feature_selection_stages: Dict[str, int]
    go_selection_audit: List[Dict[str, Any]]
    train_positive_records: List[Dict[str, Any]]
    test_positive_records: List[Dict[str, Any]]
    train_records: List[Dict[str, Any]]
    test_records: List[Dict[str, Any]]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ratio: int
    split_seed: int = SEED
    train_negative_seed: int = PRIMARY_TRAIN_NEGATIVE_SEED
    test_negative_seed: int = PRIMARY_TEST_NEGATIVE_SEED
    cv_folds_by_ratio: Dict[int, List["CVFoldData"]] = field(default_factory=dict)


@dataclass
class CVFoldData:
    """A source-isolated CV fold with feature selection fit on fold training."""

    repeat_index: int
    fold_index: int
    repeat_seed: int
    ratio: int
    train_negative_seed: int
    validation_negative_seed: int
    feature_selection_seed: int
    model_seed: int
    train_positive_ids: List[str]
    validation_positive_ids: List[str]
    selected_go: List[str]
    selected_go_sha256: str
    feature_selection_stages: Dict[str, int]
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    train_records: List[Dict[str, Any]]
    validation_records: List[Dict[str, Any]]
    scale_pos_weight: float


def ensure_dirs() -> None:
    for d in (OUT_DIR, TABLE_DIR, LATEX_TABLE_DIR, FIG_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def remove_obsolete_analysis_outputs() -> None:
    """Remove files produced only by the retired validation analysis.

    A rerun may target an existing result directory. Removing this short,
    explicit list prevents retired tables or metadata from being mistaken for
    outputs of the current source-stratified pipeline.
    """
    obsolete_paths = [
        OUT_DIR / "metric_figures_manifest.json",
        TABLE_DIR / "lofo.csv",
        TABLE_DIR / "table_lofo.csv",
        TABLE_DIR / "pathway_mixed_metadata_groups.csv",
        LATEX_TABLE_DIR / "table_lofo.tex",
        FIG_DIR / "lofo.png",
        FIG_DIR / "Fig6_lofo.png",
    ]
    for path in obsolete_paths:
        if path.exists():
            path.unlink()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of row dicts as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_table_aliases(rows: Sequence[Mapping[str, Any]], *names: str) -> None:
    """Save one logical result under stable and paper-friendly filenames.

    The LaTeX manuscript uses human labels such as "method comparison" or
    "robustness".  The reproducible pipeline also needs machine-friendly
    canonical names.  Saving aliases avoids ambiguity without duplicating logic.
    """
    for name in names:
        save_table(TABLE_DIR / name, rows)


def write_latex_tabular(path: Path, df: pd.DataFrame, caption: str, label: str) -> None:
    """Write a simple LaTeX table fragment for manual paper rebuilding.

    These fragments are intentionally plain tabular environments rather than a
    full manuscript.  They let the user regenerate the numbers and then paste or
    inspect the table independently from the manuscript prose.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
    path.write_text(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def normalize_gene(gene: str) -> str:
    """Normalize gene identifiers to uppercase for consistent matching."""
    return gene.strip().upper()


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity coefficient; two empty sets are treated as identical (0.0)."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def select_genes_for_jaccard(genes: Sequence[str]) -> List[str]:
    """Return the deterministic gene subset used for pairwise Jaccard.

    Gene sets with at most ``JACCARD_EXACT_MAX_GENES`` annotated members use
    every member.  For larger sets, a seed derived from the complete sorted
    gene membership and a versioned algorithm salt drives a local RNG.  The
    local generator does not change Python's global random state, and the
    selected subset is therefore independent of record order and call order.
    """
    ordered_genes = sorted(set(genes))
    if len(ordered_genes) <= JACCARD_EXACT_MAX_GENES:
        return ordered_genes

    payload = "\n".join([JACCARD_SAMPLING_SALT, *ordered_genes]).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    selected = random.Random(seed).sample(ordered_genes, JACCARD_SAMPLE_SIZE)
    return sorted(selected)


def jaccard_feature_protocol() -> Dict[str, Any]:
    """Return the pairwise-GO sampling settings recorded with each run."""
    return {
        "small_gene_sets": f"all annotated genes when |G*| <= {JACCARD_EXACT_MAX_GENES}",
        "large_gene_sets": f"sample {JACCARD_SAMPLE_SIZE} annotated genes without replacement",
        "sampling_seed": "SHA256 of sorted complete G* membership and algorithm salt",
        "sampling_salt": JACCARD_SAMPLING_SALT,
        "input_order_invariant": True,
        "global_rng_state_unchanged": True,
    }


def scale_pos_weight_from_labels(
    y_train: Sequence[int],
    analysis: str,
    context: Mapping[str, Any] | None = None,
) -> float:
    """Compute and audit the XGBoost/boosting positive-class weight.

    XGBoost defines ``scale_pos_weight`` as negative examples divided by
    positive examples. Computing the value from the actual training labels
    keeps the 1:1 and 1:2 branches and ratio-sensitivity runs
    consistent with the data they actually use.
    """
    y_arr = np.asarray(y_train)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    value = float(n_neg / n_pos) if n_pos else 1.0
    row = {
        "analysis": analysis,
        "n_positive_train": n_pos,
        "n_negative_train": n_neg,
        "scale_pos_weight": value,
        "rule": SCALE_POS_WEIGHT_RULE,
    }
    if context:
        row.update(dict(context))
    SCALE_POS_WEIGHT_LOG.append(row)
    return value


def run_mode_from_args(args: argparse.Namespace) -> str:
    """Return the reproducibility run mode used in audit files."""
    return "quick" if args.quick else "full"


def dataset_mode_from_args(args: argparse.Namespace) -> str:
    """Map the command-line data source to a paper-facing provenance label."""
    return "cached_legacy" if args.dataset_source == "cached" else "raw_full"


def sha256_file(path: Path) -> str:
    """Compute a SHA256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> Dict[str, str]:
    """Read a sha256sums.txt file in the common '<hash>  <filename>' format."""
    expected: Dict[str, str] = {}
    if not path.exists():
        return expected
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            expected[Path(parts[-1]).name] = parts[0].lower()
    return expected


def raw_source_checksums(raw_dir: Path) -> List[Dict[str, Any]]:
    """Audit raw-source checksums against raw_sources/sha256sums.txt when present."""
    raw_dir = Path(raw_dir)
    expected = parse_sha256sums(raw_dir / "sha256sums.txt")
    rows: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not path.is_file():
            continue
        actual = sha256_file(path)
        exp = expected.get(path.name)
        rows.append(
            {
                "file": path.name,
                "path": project_relative_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": exp if exp is not None else "",
                "matches_sha256sums": bool(exp == actual) if exp is not None else "",
            }
        )
    return rows


def git_commit() -> str | None:
    """Return the current git commit hash if the project is inside a git checkout."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def git_working_tree_dirty() -> bool | None:
    """Report source changes relative to HEAD, excluding generated artifacts.

    The manifest is written after a run has created its output directory, so a
    raw ``git status`` would always report that new directory as a modification.
    Excluding ``code/generated*`` keeps this field focused on code and input
    changes that could alter the analysis.
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        changed_paths = [line[3:].strip() for line in output.splitlines() if len(line) >= 4]
        output_prefix = project_relative_path(OUT_DIR).rstrip("/") + "/"
        source_changes = [
            path
            for path in changed_paths
            if not path.startswith("code/generated") and not path.startswith(output_prefix)
        ]
        return bool(source_changes)
    except Exception:
        return None


def project_relative_path(path: Path) -> str:
    """Use a project-relative path in portable provenance where possible."""
    resolved = path.expanduser().resolve()
    project_root = ROOT.parent.resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def software_versions() -> Dict[str, str]:
    """Collect package versions used by the reproducible pipeline."""
    versions: Dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        pass
    if lgb is not None:
        versions["lightgbm"] = lgb.__version__
    if cb is not None:
        versions["catboost"] = cb.__version__
    if shap is not None:
        versions["shap"] = shap.__version__
    return versions


def hardware_profile() -> Dict[str, Any]:
    """Collect a privacy-safe description of the computer used for a run.

    Runtime values are only interpretable together with the machine on which
    they were measured. On macOS, ``system_profiler`` provides the model, chip,
    core layout, and memory. Device identifiers such as the serial number,
    hardware UUID, and user name are deliberately not collected.
    """
    profile: Dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    if platform.system() == "Darwin":
        try:
            raw = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType", "-json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            overview = json.loads(raw).get("SPHardwareDataType", [{}])[0]
            core_layout = overview.get("number_processors")
            profile.update(
                {
                    "model_name": overview.get("machine_name"),
                    "model_identifier": overview.get("machine_model"),
                    "chip": overview.get("chip_type"),
                    "core_layout": core_layout,
                    "memory": overview.get("physical_memory"),
                }
            )
            # Apple Silicon reports strings such as ``proc 12:8:4:0``.  Keep
            # the original value and expose the three useful counts so the
            # runtime table does not depend on undocumented display syntax.
            match = re.fullmatch(r"proc\s+(\d+):(\d+):(\d+):\d+", str(core_layout))
            if match:
                profile.update(
                    {
                        "cpu_cores_total": int(match.group(1)),
                        "cpu_performance_cores": int(match.group(2)),
                        "cpu_efficiency_cores": int(match.group(3)),
                    }
                )
        except Exception:
            pass
        try:
            profile["os_version"] = subprocess.check_output(
                ["sw_vers", "-productVersion"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            profile["os_build"] = subprocess.check_output(
                ["sw_vers", "-buildVersion"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            pass
    return {key: value for key, value in profile.items() if value not in (None, "")}


def save_compute_environment(profile: Mapping[str, Any]) -> None:
    """Save hardware metadata in JSON and a human-readable two-column table."""
    save_json(OUT_DIR / "hardware_profile.json", dict(profile))
    save_table(
        TABLE_DIR / "compute_environment.csv",
        [{"field": key, "value": value} for key, value in profile.items()],
    )


def scale_pos_weight_audit_rows() -> List[Dict[str, Any]]:
    """Return all scale_pos_weight rows from previous chunks plus this process."""
    existing: List[Dict[str, Any]] = []
    path = TABLE_DIR / "scale_pos_weight_audit.csv"
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")
        existing = [row for row in existing if str(row.get("analysis", "")) != "lofo"]
    if SCALE_POS_WEIGHT_LOG:
        df = pd.concat([pd.DataFrame(existing), pd.DataFrame(SCALE_POS_WEIGHT_LOG)], ignore_index=True)
        df = df.drop_duplicates()
        rows = df.to_dict("records")
    else:
        rows = existing
    if rows:
        save_table(path, rows)
    return rows


def scale_pos_weight_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarise dynamic scale_pos_weight values for manifest/result_summary."""
    values = [float(r["scale_pos_weight"]) for r in rows if "scale_pos_weight" in r and pd.notna(r["scale_pos_weight"])]
    if not values:
        return {"rule": SCALE_POS_WEIGHT_RULE, "n_records": 0}
    return {
        "rule": SCALE_POS_WEIGHT_RULE,
        "n_records": len(values),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# GO annotation parsers
# ---------------------------------------------------------------------------
#
# Two formats are supported:
#   1. GAF 2.2 (Gene Association File) from the Gene Ontology Consortium
#   2. ATH_GO_GOSLIM.txt from TAIR (tab-delimited, 15-column format)
#
# The pipeline defaults to ATH_GO_GOSLIM.txt for raw mode.  The GAF parser
# is retained as a fallback for users who only have the GOC file.
#

def parse_gaf(path: Path) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Parse a GOC GAF 2.2 file into gene->GO and GO->gene maps.

    GAF column layout: col[1] = gene symbol, col[4] = GO term.
    Only keeps Arabidopsis AGI locus identifiers (ATxGxxxxx pattern).
    """
    gene_go: Dict[str, set] = defaultdict(set)
    go_genes: Dict[str, set] = defaultdict(set)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            gene = normalize_gene(parts[1])
            go_term = parts[4]
            if re.match(r"AT[0-9]G[0-9]{5}$", gene):
                gene_go[gene].add(go_term)
                go_genes[go_term].add(gene)
    return dict(gene_go), dict(go_genes)


def parse_ath_go_goslim(path: Path) -> Tuple[Dict[str, set], Dict[str, set], Dict[str, str], Dict[str, int]]:
    """Parse TAIR ATH_GO_GOSLIM.txt into deduplicated gene-GO annotations.

    ATH_GO_GOSLIM is a TAIR tab-delimited GO/GO Slim export.  It can contain
    multiple rows for the same gene-GO pair because the same GO annotation is
    mapped to different GO slim categories.  PathwayML-Ath uses only the
    gene-to-GO-term relation, so this parser collapses duplicate gene-GO pairs
    before feature construction and ignores the GO slim category as a feature.

    Relevant columns in the current TAIR file are:

    * column 1: AGI locus, for example AT1G01010
    * column 5: GO term name
    * column 6: GO ID, for example GO:0006355
    """
    gene_go: Dict[str, set] = defaultdict(set)
    go_genes: Dict[str, set] = defaultdict(set)
    go_term_names: Dict[str, str] = {}
    stats = Counter()

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                stats["header_rows"] += 1
                continue
            stats["raw_rows"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                stats["short_rows"] += 1
                continue
            gene = normalize_gene(parts[0])
            go_name = parts[4].strip()
            go_term = parts[5].strip()
            if not re.match(r"AT[1-5CM]G[0-9]{5}$", gene):
                stats["non_agi_rows"] += 1
                continue
            if not re.match(r"GO:[0-9]{7}$", go_term):
                stats["invalid_go_rows"] += 1
                continue

            before = len(gene_go[gene])
            gene_go[gene].add(go_term)
            go_genes[go_term].add(gene)
            if len(gene_go[gene]) == before:
                stats["duplicate_gene_go_rows"] += 1
            if go_name:
                go_term_names.setdefault(go_term, go_name)

    stats["unique_gene_go_pairs"] = sum(len(terms) for terms in gene_go.values())
    stats["go_annotated_genes"] = len(gene_go)
    stats["unique_go_terms"] = len(go_genes)
    return dict(gene_go), dict(go_genes), dict(go_term_names), {k: int(v) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Raw-data parsers
# ---------------------------------------------------------------------------
#
# These functions are used by the default raw-source pipeline.  The goal is to
# make all primary data provenance explicit through raw_sources/source_manifest.csv
# and raw_sources/sha256sums.txt instead of relying on cached JSON objects.
#
def parse_kegg(raw_dir: Path) -> Tuple[Dict[str, set], Dict[str, str]]:
    """Parse KEGG pathway-gene and pathway-name files."""
    genes_path = raw_dir / "kegg_pathway_genes.txt"
    names_path = raw_dir / "kegg_pathway_names.txt"
    kegg: Dict[str, set] = defaultdict(set)
    names: Dict[str, str] = {}

    with genes_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            pid = parts[0].replace("path:", "")
            gene = normalize_gene(parts[1].replace("ath:", ""))
            if re.match(r"AT[0-9]G[0-9]{5}$", gene) or gene.startswith("ARTH"):
                kegg[pid].add(gene)

    if names_path.exists():
        with names_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    names[parts[0].replace("path:", "")] = parts[1].split(" - ")[0].strip()
    return dict(kegg), names


def parse_aracyc(raw_dir: Path) -> Tuple[Dict[str, set], Dict[str, str], Dict[str, Any]]:
    """Parse the PMN/AraCyc tab-delimited pathway dump."""
    path = raw_dir / "aracyc_pathways.20251021"
    aracyc: Dict[str, set] = defaultdict(set)
    names: Dict[str, str] = {}
    stats = Counter()
    raw_pathway_ids = set()

    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        required = ["Pathway-id", "Pathway-name", "Gene-id"]
        missing = [name for name in required if name not in idx]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for line in f:
            stats["rows"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx.values()):
                stats["short_rows"] += 1
                continue
            pid = parts[idx["Pathway-id"]]
            name = parts[idx["Pathway-name"]]
            gene = normalize_gene(parts[idx["Gene-id"]])
            raw_pathway_ids.add(pid)
            if not gene or gene == "NIL":
                stats["nil_gene_rows"] += 1
                names.setdefault(pid, name)
                continue
            if not gene.startswith("AT"):
                stats["non_at_gene_rows"] += 1
                names.setdefault(pid, name)
                continue
            aracyc[pid].add(gene)
            names[pid] = name

    stats_out = {
        "rows": int(stats["rows"]),
        "nil_gene_rows": int(stats["nil_gene_rows"]),
        "non_at_gene_rows": int(stats["non_at_gene_rows"]),
        "raw_distinct_pathway_ids": len(raw_pathway_ids),
        "pathways_with_valid_at_gene": len(aracyc),
    }
    return dict(aracyc), names, stats_out


def build_raw_pathways(raw_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Merge KEGG and AraCyc source records into a unified dictionary.

    Applies the MIN_PATHWAY_GENES filter (>=5 genes). AraCyc pathway IDs are
    prefixed with ``AC_`` to avoid collisions with KEGG IDs. Database source is
    retained because it is an explicit field used for train/test stratification.
    """
    kegg, kegg_names = parse_kegg(raw_dir)
    aracyc, aracyc_names, aracyc_stats = parse_aracyc(raw_dir)

    pathways: Dict[str, Dict[str, Any]] = {}
    for pid in sorted(kegg):
        genes = sorted(kegg[pid])
        if len(genes) >= MIN_PATHWAY_GENES:
            name = kegg_names.get(pid, pid)
            pathways[pid] = {
                "name": name,
                "genes": genes,
                "source": "KEGG",
            }
    n_kegg = sum(1 for info in pathways.values() if info["source"] == "KEGG")

    for pid in sorted(aracyc):
        genes = sorted(aracyc[pid])
        if len(genes) >= MIN_PATHWAY_GENES:
            name = aracyc_names.get(pid, pid)
            pathways[f"AC_{pid}"] = {
                "name": name,
                "genes": genes,
                "source": "AraCyc",
            }

    stats = {
        "n_pathways": len(pathways),
        "n_kegg": n_kegg,
        "n_aracyc": len(pathways) - n_kegg,
        "raw_kegg_pathways": len(kegg),
        "raw_aracyc_stats": aracyc_stats,
    }
    return pathways, stats


def gene_set_sha256(genes: Sequence[str]) -> str:
    """Return a stable hash for a normalized, internally unique gene set."""
    normalized = sorted({normalize_gene(gene) for gene in genes if normalize_gene(gene)})
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_identical_gene_sets(
    source_pathways: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collapse exact gene-set duplicates into one modelling instance.

    The source database records are not deleted.  Each modelling instance keeps
    every source ID and name as provenance.  Grouping only prevents an identical
    input vector from being weighted more than once or split across train/test.
    It does not assert that aliased records have the same biological meaning.
    """
    buckets: Dict[Tuple[str, ...], List[Tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for pathway_id, info in sorted(source_pathways.items()):
        genes = tuple(sorted({normalize_gene(gene) for gene in info["genes"] if normalize_gene(gene)}))
        if len(genes) >= MIN_PATHWAY_GENES:
            buckets[genes].append((str(pathway_id), info))

    groups: Dict[str, Dict[str, Any]] = {}
    duplicate_rows: List[Dict[str, Any]] = []
    mixed_rows: List[Dict[str, Any]] = []
    for genes, members in sorted(buckets.items(), key=lambda item: item[0]):
        kegg_ids = sorted(pathway_id for pathway_id, info in members if info.get("source") == "KEGG")
        representative_id = kegg_ids[0] if kegg_ids else min(pathway_id for pathway_id, _ in members)
        representative_info = next(info for pathway_id, info in members if pathway_id == representative_id)
        digest = gene_set_sha256(genes)
        group_id = f"GSG_{digest[:16]}"
        pathway_ids = [pathway_id for pathway_id, _ in sorted(members)]
        names = [str(info.get("name", pathway_id)) for pathway_id, info in sorted(members)]
        sources = [str(info.get("source", "")) for _, info in sorted(members)]
        unique_sources = sorted(set(sources))
        group = {
            "name": representative_info.get("name", representative_id),
            "genes": list(genes),
            "source": representative_info.get("source", ""),
            "representative_pathway_id": representative_id,
            "gene_set_group_id": group_id,
            "gene_set_sha256": digest,
            "source_pathway_ids": pathway_ids,
            "source_pathway_names": names,
            "source_databases": sources,
            "source_record_count": len(members),
            "database_count": len(unique_sources),
            "mixed_source": len(unique_sources) > 1,
            "stratification_source": representative_info.get("source", ""),
        }
        groups[representative_id] = group

        if len(members) > 1:
            duplicate_rows.append(
                {
                    "gene_set_group_id": group_id,
                    "gene_set_sha256": digest,
                    "representative_pathway_id": representative_id,
                    "source_record_count": len(members),
                    "database_count": len(unique_sources),
                    "source_pathway_ids": ";".join(pathway_ids),
                    "source_pathway_names": ";".join(names),
                    "source_databases": ";".join(sources),
                    "raw_gene_count": len(genes),
                }
            )
        if group["mixed_source"]:
            mixed_rows.append(
                {
                    "gene_set_group_id": group_id,
                    "representative_pathway_id": representative_id,
                    "mixed_source": group["mixed_source"],
                    "source_pathway_ids": ";".join(pathway_ids),
                    "source_databases": ";".join(sources),
                }
            )

    source_counts = Counter(str(info.get("source", "")) for info in source_pathways.values())
    group_strata = Counter(str(info.get("stratification_source", "")) for info in groups.values())
    summary = {
        "source_record_count": len(source_pathways),
        "source_record_kegg": source_counts.get("KEGG", 0),
        "source_record_aracyc": source_counts.get("AraCyc", 0),
        "modelling_group_count": len(groups),
        "modelling_group_kegg_stratum": group_strata.get("KEGG", 0),
        "modelling_group_aracyc_stratum": group_strata.get("AraCyc", 0),
        "exact_duplicate_group_count": len(duplicate_rows),
        "collapsed_extra_record_count": len(source_pathways) - len(groups),
        "mixed_source_group_count": sum(bool(info["mixed_source"]) for info in groups.values()),
        "grouping_rule": "exact equality of sorted normalized gene membership",
    }
    return groups, summary, duplicate_rows, mixed_rows


def save_kegg_filter_audit(raw_dir: Path) -> None:
    """Record how the raw KEGG snapshot is reduced to modelling pathways.

    KEGG pathway-gene links can include very small entries.  The project keeps
    a pathway only when at least five distinct normalized genes remain after
    parsing.  This table makes the 161-to-155 transition inspectable instead of
    leaving it implicit in a single conditional.
    """
    kegg, names = parse_kegg(raw_dir)
    rows = []
    for pathway_id in sorted(kegg):
        n_unique_genes = len(kegg[pathway_id])
        kept = n_unique_genes >= MIN_PATHWAY_GENES
        rows.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": names.get(pathway_id, pathway_id),
                "n_unique_linked_genes": n_unique_genes,
                "minimum_required_genes": MIN_PATHWAY_GENES,
                "kept_for_modelling": kept,
                "exclusion_reason": "" if kept else "fewer_than_5_unique_linked_genes",
            }
        )
    save_table(TABLE_DIR / "kegg_pathway_filter_audit.csv", rows)
    save_json(
        TABLE_DIR / "kegg_pathway_filter_summary.json",
        {
            "raw_distinct_pathways_with_gene_links": len(rows),
            "kept_pathways": sum(bool(row["kept_for_modelling"]) for row in rows),
            "excluded_pathways": sum(not bool(row["kept_for_modelling"]) for row in rows),
            "rule": "keep pathways with at least 5 distinct normalized linked genes",
        },
    )


def load_cached_bundle() -> DatasetBundle:
    """Load the archived JSON data snapshot for comparison runs.

    This mode uses data_robustness/all_pathways.json (543 pathways = 160 KEGG
    + 383 AraCyc) and should only be used for comparison with earlier manuscript
    numbers. Raw mode is preferred for auditability.
    """
    with (CACHED_DATA_DIR / "all_pathways.json").open(encoding="utf-8") as f:
        pathways = json.load(f)
    with (CACHED_DATA_DIR / "gene_go.json").open(encoding="utf-8") as f:
        gene_go_raw = json.load(f)
    with (CACHED_DATA_DIR / "feature_info.json").open(encoding="utf-8") as f:
        feature_info = json.load(f)

    gene_go = {gene: set(terms) for gene, terms in gene_go_raw.items()}
    go_genes: Dict[str, set] = defaultdict(set)
    for gene, terms in gene_go.items():
        for term in terms:
            go_genes[term].add(gene)

    # The cached all_pathways.json only has name/genes. Add the explicit source
    # field so comparison runs use the same source-based stratification.
    for pid, info in pathways.items():
        if pid.startswith("ath"):
            info.setdefault("source", "KEGG")
        else:
            info.setdefault("source", "AraCyc")
        info["genes"] = sorted(set(info["genes"]))

    source_pathways = {pid: dict(info) for pid, info in pathways.items()}
    groups, grouping_summary, duplicate_groups, mixed_source_groups = group_identical_gene_sets(source_pathways)
    return DatasetBundle(
        pathways=groups,
        source_pathways=source_pathways,
        gene_go=gene_go,
        go_genes=dict(go_genes),
        go_term_names=json.load((CACHED_DATA_DIR / "go_term_names.json").open(encoding="utf-8"))
        if (CACHED_DATA_DIR / "go_term_names.json").exists()
        else {},
        selected_go=list(feature_info["selected_go"]),
        feature_names=list(feature_info["feature_names"]),
        feature_stages=dict(feature_info.get("feat_sel_stages", {})),
        source="cached",
        grouping_summary=grouping_summary,
        duplicate_groups=duplicate_groups,
        mixed_source_groups=mixed_source_groups,
    )


# ---------------------------------------------------------------------------
# Feature construction and negative sampling
# ---------------------------------------------------------------------------
#
# The manuscript package uses the current thesis negative-sampling logic:
#   positives: KEGG + AraCyc curated pathways
#   negatives: random[5,30], full replacement shuffled, partial 50-80%,
#              and cross-pathway mixtures
#   features:  60 GO frequency terms + 9 hand-engineered pathway descriptors
#
# This section keeps that logic explicit so the generated outputs stay aligned
# with the current manuscript.
#
def generate_negative_samples(
    pathways: Mapping[str, Mapping[str, Any]],
    gene_pool: Sequence[str],
    n_negatives: int,
    seed: int,
) -> List[NegativeSample]:
    """Generate the four control classes and record their pathway sources.

    Four negative types are generated in round-robin order (i % 4):
      0. random_5_30:      random genes, size uniformly drawn from [5, 30]
      1. shuffled:         random genes matching the size of a real pathway
      2. partial_50_80:    50-80% subsample of a real pathway's genes
      3. cross_pathway:    half from one pathway + half from another

    Every emitted set contains at least five GO-annotated genes. Cross-pathway
    controls use two distinct modelling instances. No inferred functional
    category is used when choosing either source.
    """
    rng = random.Random(seed)
    annotated_pool = set(gene_pool)
    sorted_gene_pool = sorted(annotated_pool)
    pathway_items = [
        (pid, info, list(info["genes"]))
        for pid, info in pathways.items()
        if len(info["genes"]) >= MIN_PATHWAY_GENES
        and len(set(info["genes"]) & annotated_pool) >= MIN_PATHWAY_GENES
    ]
    if len(pathway_items) < 2:
        raise ValueError("At least two pathways with five GO-annotated genes are required.")
    partial_items = [
        item
        for item in pathway_items
        if math.floor(0.8 * len(item[2])) >= MIN_PATHWAY_GENES
    ]
    if not partial_items:
        raise ValueError("No pathway can produce a valid partial_50_80 control.")

    out: List[NegativeSample] = []
    for i in range(n_negatives):
        kind = i % 4
        for attempt in range(1, 1001):
            retention_fraction: float | None = None
            if kind == 0:
                size = rng.randint(5, 30)
                genes = rng.sample(sorted_gene_pool, size)
                negative_type = "random_5_30"
                source_items: List[Tuple[str, Mapping[str, Any], List[str]]] = []
            elif kind == 1:
                source_items = [rng.choice(pathway_items)]
                src = source_items[0][2]
                genes = rng.sample(sorted_gene_pool, len(src))
                negative_type = "shuffled"
            elif kind == 2:
                source_items = [rng.choice(partial_items)]
                src = source_items[0][2]
                low = math.ceil(0.5 * len(src))
                high = math.floor(0.8 * len(src))
                keep = rng.randint(low, high)
                genes = rng.sample(src, keep)
                retention_fraction = keep / len(src)
                negative_type = "partial_50_80"
            else:
                first = rng.choice(pathway_items)
                second_candidates = [item for item in pathway_items if item[0] != first[0]]
                second = rng.choice(second_candidates)
                source_items = [first, second]
                s1, s2 = first[2], second[2]
                genes = rng.sample(s1, min(max(2, len(s1) // 2), len(s1)))
                genes += rng.sample(s2, min(max(2, len(s2) // 2), len(s2)))
                genes = sorted(set(genes))
                negative_type = "cross_pathway"

            genes = sorted(set(genes))
            annotated_count = len(set(genes) & annotated_pool)
            if annotated_count >= MIN_PATHWAY_GENES:
                break
        else:  # pragma: no cover - defensive failure for malformed source data
            raise RuntimeError(
                f"Could not generate a valid {negative_type} control after 1000 attempts "
                f"(seed={seed}, sample_index={i})."
            )

        source_ids = [item[0] for item in source_items]
        if negative_type == "cross_pathway" and len(set(source_ids)) != 2:
            raise AssertionError("cross_pathway controls require two distinct source IDs.")
        source_gene_union = set().union(*(set(item[2]) for item in source_items)) if source_items else set()
        overlap_count = len(set(genes) & source_gene_union)
        out.append(
            NegativeSample(
                genes=list(genes),
                negative_type=negative_type,
                source_pathway_ids=source_ids,
                source_overlap_count=overlap_count,
                source_overlap_fraction=float(overlap_count / len(set(genes))) if genes else 0.0,
                generation_seed=seed,
                raw_gene_count=len(genes),
                annotated_gene_count=annotated_count,
                source_raw_gene_count=len(source_gene_union),
                source_annotated_gene_count=len(source_gene_union & annotated_pool),
                retention_fraction=retention_fraction,
                generation_attempt=attempt,
            )
        )
    return out


def generate_negative_gene_sets(
    pathways: Mapping[str, Mapping[str, Any]],
    gene_pool: Sequence[str],
    n_negatives: int,
    seed: int,
) -> List[List[str]]:
    """Compatibility wrapper returning only gene lists.

    All new analysis code consumes :func:`generate_negative_samples` so source
    provenance is never lost.  This wrapper is retained for reproducibility
    checks and external callers that only need the historical gene-list shape.
    """
    return [sample.genes for sample in generate_negative_samples(pathways, gene_pool, n_negatives, seed)]


def feature_names_for(selected_go: Sequence[str]) -> List[str]:
    """Return the ordered 60 GO + 9 engineered feature names."""
    return list(selected_go) + [
        "jaccard_mean",
        "jaccard_min",
        "jaccard_max",
        "jaccard_std",
        "annotated_gene_count",
        "log_annotated_gene_count",
        "go_entropy",
        "go_size_std",
        "mean_go_per_gene",
    ]


def select_go_terms_from_records(
    records: Sequence[Mapping[str, Any]],
    gene_go: Mapping[str, set],
    go_genes: Mapping[str, set],
    seed: int,
    k: int = N_GO_TERMS,
) -> Tuple[List[str], Dict[str, int], List[Dict[str, Any]]]:
    """Select GO terms using only records from the primary training side.

    The background frequency filter is label-free and therefore uses the full
    GO-annotated gene universe.  Sample variance and mutual information are fit
    only on ``records``; outer-test labels never enter this function.
    """
    annotated_n = len(gene_go)
    upper = int(0.30 * annotated_n)
    freq_terms = sorted([term for term, genes in go_genes.items() if 20 <= len(genes) <= upper])
    labels = np.array([int(record["label"]) for record in records], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("GO feature selection requires both positive and negative training records.")

    term_index = {term: index for index, term in enumerate(freq_terms)}
    X = np.zeros((len(records), len(freq_terms)), dtype=np.float32)
    for i, record in enumerate(records):
        genes = record["genes"]
        valid = [g for g in genes if g in gene_go]
        denom = max(len(valid), 1)
        counts: Counter[str] = Counter()
        for gene in valid:
            counts.update(term for term in gene_go[gene] if term in term_index)
        for term, count in counts.items():
            X[i, term_index[term]] = count / denom

    variances = X.var(axis=0)
    cutoff = np.percentile(variances, 15)
    keep_mask = variances > cutoff
    variance_terms = [term for term, keep in zip(freq_terms, keep_mask) if keep]
    X_var = X[:, keep_mask]

    mi_scores = np.zeros(len(variance_terms), dtype=float)
    if X_var.shape[1] <= k:
        selected = variance_terms
        ranked_indices = np.arange(len(variance_terms))
    else:
        mi_scores = mutual_info_classif(X_var, labels, random_state=seed)
        ranked_indices = np.asarray(
            sorted(range(len(variance_terms)), key=lambda idx: (-float(mi_scores[idx]), variance_terms[idx])),
            dtype=int,
        )
        selected = [variance_terms[i] for i in ranked_indices[:k]]

    rank_by_term = {variance_terms[idx]: rank for rank, idx in enumerate(ranked_indices, 1)}
    mi_by_term = {term: float(score) for term, score in zip(variance_terms, mi_scores)}
    variance_by_term = {term: float(value) for term, value in zip(freq_terms, variances)}
    selected_set = set(selected)
    audit_rows = [
        {
            "go_term": term,
            "background_gene_count": int(len(go_genes[term])),
            "train_variance": variance_by_term[term],
            "passed_variance_filter": bool(term in mi_by_term),
            "mutual_information": mi_by_term.get(term, float("nan")),
            "mi_rank": rank_by_term.get(term, ""),
            "selected": bool(term in selected_set),
        }
        for term in freq_terms
    ]

    stages = {
        "raw": len(go_genes),
        "freq": len(freq_terms),
        "ic": len(freq_terms),
        "variance": len(variance_terms),
        "mi_select": len(selected),
    }
    return selected, stages, audit_rows


def load_raw_bundle(raw_dir: Path) -> DatasetBundle:
    """Parse raw source files without looking at benchmark labels.

    Expects raw_dir to contain ATH_GO_GOSLIM.txt, kegg_pathway_genes.txt,
    kegg_pathway_names.txt, and aracyc_pathways.20251021.  This is the
    preferred auditable data mode for the paper.  Supervised GO selection is
    deliberately deferred until after the primary pathway-level split.
    """
    raw_dir = Path(raw_dir)
    gene_go, go_genes, go_term_names, go_stats = parse_ath_go_goslim(raw_dir / "ATH_GO_GOSLIM.txt")
    source_pathways, raw_stats = build_raw_pathways(raw_dir)
    pathways, grouping_summary, duplicate_groups, mixed_source_groups = group_identical_gene_sets(source_pathways)
    stages = {"raw": len(go_genes)}
    stages.update({f"raw_pathway_stat_{k}": v for k, v in raw_stats.items() if isinstance(v, int)})
    stages.update({f"raw_go_stat_{k}": v for k, v in go_stats.items() if isinstance(v, int)})
    return DatasetBundle(
        pathways=pathways,
        source_pathways=source_pathways,
        gene_go=gene_go,
        go_genes=go_genes,
        go_term_names=go_term_names,
        selected_go=[],
        feature_names=[],
        feature_stages=stages,
        source="raw",
        grouping_summary=grouping_summary,
        duplicate_groups=duplicate_groups,
        mixed_source_groups=mixed_source_groups,
    )


def build_feature_vector(genes: Sequence[str], selected_go: Sequence[str], gene_go: Mapping[str, set]) -> np.ndarray:
    """Build the 69-dimensional feature vector for one gene set.

    Feature layout (total D = len(selected_go) + 9 = 69):
      [0:60]   GO-frequency features: fraction of genes annotated with each GO term
      [60]     jaccard_mean:     mean pairwise GO-set Jaccard among genes
      [61]     jaccard_min:      minimum pairwise Jaccard
      [62]     jaccard_max:      maximum pairwise Jaccard
      [63]     jaccard_std:      standard deviation of pairwise Jaccard
      [64]     annotated_gene_count:     number of genes in G* with GO annotation
      [65]     log_annotated_gene_count: log(1 + annotated_gene_count)
      [66]     go_entropy:       Shannon entropy of GO frequency distribution
      [67]     go_size_std:      std of per-gene GO annotation counts
      [68]     mean_go_per_gene: mean number of GO terms per gene

    Pairwise Jaccard uses all annotated genes when ``|G*| <= 30``.  Larger
    sets use 30 genes selected by :func:`select_genes_for_jaccard`, whose
    gene-set-derived seed removes dependence on stored gene order while keeping
    the calculation tractable.
    """
    valid = [g for g in genes if g in gene_go]
    if not valid:
        return np.zeros(len(selected_go) + 9, dtype=np.float32)
    go_sets = [set(gene_go.get(g, set())) for g in valid]
    n = len(valid)
    freq = [sum(1 for s in go_sets if term in s) / n for term in selected_go]

    jaccard_genes = select_genes_for_jaccard(valid)
    jaccard_go_sets = [set(gene_go.get(g, set())) for g in jaccard_genes]
    pairs = [
        jaccard(jaccard_go_sets[i], jaccard_go_sets[j])
        for i, j in combinations(range(len(jaccard_go_sets)), 2)
    ]
    sims = pairs if pairs else [0.0]
    freq_arr = np.array(freq, dtype=np.float64)
    if float(freq_arr.sum()) <= 0.0:
        # No selected GO signal should mean no entropy evidence, not a uniform
        # 60-bin distribution with the artificial maximum entropy ln(60).
        entropy = 0.0
    else:
        # Preserve the existing smoothing for non-zero profiles so this boundary
        # fix changes only samples with no annotations among the selected terms.
        freq_arr = freq_arr + 1e-9
        freq_arr /= freq_arr.sum()
        entropy = float(-np.sum(freq_arr * np.log(freq_arr + 1e-12)))
    go_sizes = [len(s) for s in go_sets]

    return np.array(
        freq
        + [
            float(np.mean(sims)),
            float(np.min(sims)),
            float(np.max(sims)),
            float(np.std(sims)),
            float(n),
            float(np.log1p(n)),
            entropy,
            float(np.std(go_sizes)),
            float(np.mean(go_sizes)),
        ],
        dtype=np.float32,
    )


def pathway_records(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    """Convert unique gene-set modelling instances to flat positive records."""
    records = []
    for pid, info in sorted(bundle.pathways.items()):
        genes = sorted(set(info["genes"]))
        if len(genes) < MIN_PATHWAY_GENES:
            continue
        records.append(
            {
                "id": pid,
                "name": info.get("name", pid),
                "genes": genes,
                "label": 1,
                "source": info.get("source", "KEGG" if pid.startswith("ath") else "AraCyc"),
                "stratification_source": info.get("stratification_source", info.get("source", "")),
                "gene_set_group_id": info.get("gene_set_group_id", f"GSG_{gene_set_sha256(genes)[:16]}"),
                "gene_set_sha256": info.get("gene_set_sha256", gene_set_sha256(genes)),
                "representative_pathway_id": info.get("representative_pathway_id", pid),
                "source_pathway_ids_all": list(info.get("source_pathway_ids", [pid])),
                "source_pathway_names_all": list(info.get("source_pathway_names", [info.get("name", pid)])),
                "source_databases_all": list(info.get("source_databases", [info.get("source", "")])),
                "source_record_count": int(info.get("source_record_count", 1)),
                "database_count": int(info.get("database_count", 1)),
                "mixed_source": bool(info.get("mixed_source", False)),
                "raw_gene_count": len(genes),
                "annotated_gene_count": sum(gene in bundle.gene_go for gene in genes),
                "negative_type": "NA",
                "source_pathway_ids": [],
                "source_overlap_count": 0,
                "source_overlap_fraction": 0.0,
                "generation_seed": None,
            }
        )
    return records


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON-serialisable value using stable separators and key order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_deterministic_seed(
    master_seed: int,
    ratio: int,
    repeat_seed: int,
    fold_index: int,
    split_side: str,
    purpose: str,
) -> int:
    """Derive a stable 32-bit seed without depending on loop traversal order."""
    payload = {
        "master_seed": int(master_seed),
        "ratio": int(ratio),
        "repeat_seed": int(repeat_seed),
        "fold_index": int(fold_index),
        "split_side": str(split_side),
        "purpose": str(purpose),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def primary_split_sha256(context: PrimaryContext) -> str:
    """Fingerprint the complete canonical split with one shared definition."""
    return stable_json_sha256(
        {
            "train_positive_ids": [record["id"] for record in context.train_positive_records],
            "test_positive_ids": [record["id"] for record in context.test_positive_records],
            "train_gene_set_group_ids": [record["gene_set_group_id"] for record in context.train_positive_records],
            "test_gene_set_group_ids": [record["gene_set_group_id"] for record in context.test_positive_records],
            "train_gene_set_hashes": [record["gene_set_sha256"] for record in context.train_positive_records],
            "test_gene_set_hashes": [record["gene_set_sha256"] for record in context.test_positive_records],
            "train_records": [(record["id"], record["genes"]) for record in context.train_records],
            "test_records": [(record["id"], record["genes"]) for record in context.test_records],
        }
    )


def pathway_mapping_from_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert positive records back to the generator's pathway mapping shape."""
    return {
        str(record["id"]): {
            "name": record.get("name", record["id"]),
            "genes": list(record["genes"]),
            "source": record.get("source", ""),
            "gene_set_group_id": record.get("gene_set_group_id", ""),
            "source_pathway_ids": list(record.get("source_pathway_ids_all", [record["id"]])),
        }
        for record in records
    }


def negative_sample_record(sample: NegativeSample, sample_id: str) -> Dict[str, Any]:
    """Convert a generated sample to the common record schema."""
    return {
        "id": sample_id,
        "name": f"{sample.negative_type}_{sample_id}",
        "genes": sorted(set(sample.genes)),
        "label": 0,
        "source": "synthetic",
        "negative_type": sample.negative_type,
        "source_pathway_ids": list(sample.source_pathway_ids),
        "source_overlap_count": int(sample.source_overlap_count),
        "source_overlap_fraction": float(sample.source_overlap_fraction),
        "generation_seed": int(sample.generation_seed),
        "raw_gene_count": int(sample.raw_gene_count),
        "annotated_gene_count": int(sample.annotated_gene_count),
        "source_raw_gene_count": int(sample.source_raw_gene_count),
        "source_annotated_gene_count": int(sample.source_annotated_gene_count),
        "retention_fraction": sample.retention_fraction,
        "generation_attempt": int(sample.generation_attempt),
        "gene_set_sha256": gene_set_sha256(sample.genes),
    }


def build_side_records(
    positive_records: Sequence[Mapping[str, Any]],
    bundle: DatasetBundle,
    ratio: int,
    negative_seed: int,
    prefix: str,
) -> List[Dict[str, Any]]:
    """Build one split side from its positives and source-derived controls.

    Partial/cross/shuffled sources are restricted to ``positive_records``.  The
    GO-annotated background remains global by design, so gene IDs may appear on
    both sides; the prohibited dependency is a source pathway crossing the split.
    """
    positives = [dict(record) for record in positive_records]
    source_pathways = pathway_mapping_from_records(positives)
    negative_samples = generate_negative_samples(
        source_pathways,
        sorted(bundle.gene_go),
        len(positives) * ratio,
        seed=negative_seed,
    )
    negatives = [
        negative_sample_record(sample, f"{prefix}_NEG_{idx:05d}")
        for idx, sample in enumerate(negative_samples)
    ]
    return positives + negatives


def assert_source_isolation(
    train_positive_records: Sequence[Mapping[str, Any]],
    test_positive_records: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> None:
    """Fail loudly if any source-derived control crosses a pathway split."""
    train_ids = {str(record["id"]) for record in train_positive_records}
    test_ids = {str(record["id"]) for record in test_positive_records}
    if train_ids & test_ids:
        raise AssertionError("Positive pathway IDs overlap between train and test.")
    train_group_ids = {str(record.get("gene_set_group_id", "")) for record in train_positive_records}
    test_group_ids = {str(record.get("gene_set_group_id", "")) for record in test_positive_records}
    train_hashes = {str(record.get("gene_set_sha256", gene_set_sha256(record["genes"]))) for record in train_positive_records}
    test_hashes = {str(record.get("gene_set_sha256", gene_set_sha256(record["genes"]))) for record in test_positive_records}
    if train_group_ids & test_group_ids:
        raise AssertionError("Gene-set group IDs overlap between train and test.")
    if train_hashes & test_hashes:
        raise AssertionError("Exact gene-set hashes overlap between train and test.")

    for split_name, records, allowed, forbidden in [
        ("train", train_records, train_ids, test_ids),
        ("test", test_records, test_ids, train_ids),
    ]:
        for record in records:
            if int(record["label"]) != 0:
                continue
            source_ids = set(record.get("source_pathway_ids", []))
            if not source_ids.issubset(allowed):
                raise AssertionError(f"{split_name} negative has a source outside its positive side: {record['id']}")
            if source_ids & forbidden:
                raise AssertionError(f"{split_name} negative leaks a source pathway across the split: {record['id']}")


def split_positive_records(
    positive_records: Sequence[Mapping[str, Any]],
    test_size: float = PRIMARY_TEST_SIZE,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split positive pathways first, preserving KEGG/AraCyc proportions.

    Database source is explicit in the input records and is the only grouping
    variable used for stratification. A source with fewer than two pathways is
    an explicit configuration error; there is no silent fallback to an
    unstratified split.
    """
    records = [dict(record) for record in positive_records]
    sources = [str(record.get("stratification_source", record["source"])) for record in records]
    source_counts = Counter(sources)
    too_small = {source: count for source, count in source_counts.items() if count < 2}
    if too_small:
        raise ValueError(f"Cannot stratify primary split by database source: {too_small}")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=np.asarray(sources),
    )
    train_records = sorted((records[int(i)] for i in train_idx), key=lambda record: str(record["id"]))
    test_records = sorted((records[int(i)] for i in test_idx), key=lambda record: str(record["id"]))
    return train_records, test_records


def records_to_matrix(
    records: Sequence[Mapping[str, Any]],
    selected_go: Sequence[str],
    gene_go: Mapping[str, set],
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode records under one fixed GO representation."""
    X = np.vstack([build_feature_vector(record["genes"], selected_go, gene_go) for record in records])
    y = np.asarray([int(record["label"]) for record in records], dtype=int)
    return X, y


def prepare_primary_context(
    bundle: DatasetBundle,
    ratio: int = DEFAULT_PRIMARY_RATIO,
) -> PrimaryContext:
    """Create the canonical outer split and select GO terms on training only."""
    if ratio < 1:
        raise ValueError(f"Primary negative ratio must be at least 1, got {ratio}.")
    positives = pathway_records(bundle)
    train_positives, test_positives = split_positive_records(positives, seed=SEED)
    train_records = build_side_records(
        train_positives,
        bundle,
        ratio=ratio,
        negative_seed=PRIMARY_TRAIN_NEGATIVE_SEED,
        prefix="TRAIN",
    )
    test_records = build_side_records(
        test_positives,
        bundle,
        ratio=ratio,
        negative_seed=PRIMARY_TEST_NEGATIVE_SEED,
        prefix="TEST",
    )
    assert_source_isolation(train_positives, test_positives, train_records, test_records)

    selected_go, selection_stages, audit_rows = select_go_terms_from_records(
        train_records,
        bundle.gene_go,
        bundle.go_genes,
        seed=SEED,
        k=N_GO_TERMS,
    )
    if len(selected_go) != N_GO_TERMS:
        raise AssertionError(f"Expected {N_GO_TERMS} selected GO terms, got {len(selected_go)}")

    # Downstream functions still accept DatasetBundle.  Updating these three
    # fields once makes every analysis consume the exact same canonical feature
    # set without re-running supervised selection independently.
    bundle.selected_go = list(selected_go)
    bundle.feature_names = feature_names_for(selected_go)
    bundle.feature_stages = {**bundle.feature_stages, **selection_stages}
    X_train, y_train = records_to_matrix(train_records, selected_go, bundle.gene_go)
    X_test, y_test = records_to_matrix(test_records, selected_go, bundle.gene_go)

    return PrimaryContext(
        selected_go=list(selected_go),
        feature_names=list(bundle.feature_names),
        feature_selection_stages=dict(bundle.feature_stages),
        go_selection_audit=audit_rows,
        train_positive_records=train_positives,
        test_positive_records=test_positives,
        train_records=train_records,
        test_records=test_records,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ratio=ratio,
    )


def xgb_model(
    scale_pos_weight: float = 2.0,
    fast: bool = False,
    random_state: int = SEED,
    n_jobs: int = -1,
) -> XGBClassifier:
    """Create an XGBoost classifier with paper-aligned hyperparameters.

    Full mode is the paper configuration and matches the LaTeX/ML notebook
    setting of 500 trees.  --quick mode uses fewer trees only for fast checks.
    The caller supplies the analysis-specific ``scale_pos_weight`` computed
    from the current training split.
    """
    return XGBClassifier(
        n_estimators=40 if fast else 500,
        max_depth=5,
        learning_rate=0.08 if fast else 0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Extract positive-class probability from any sklearn-compatible model.

    Falls back to sigmoid(decision_function) for models without predict_proba.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1 / (1 + np.exp(-decision))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics at threshold 0.5."""
    pred = (scores >= 0.5).astype(int)
    return {
        "test_auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "test_auprc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def heldout_diagnostic_tables(
    records: Sequence[Mapping[str, Any]],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build traceable held-out prediction and negative-type summaries.

    These tables all come from the same score vector used for the main XGBoost
    row. Keeping the calculation here prevents the diagnostic figures from
    silently fitting a second model with different settings.
    """
    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    if len(records) != len(y_arr) or len(y_arr) != len(score_arr):
        raise ValueError("Held-out records, labels, and scores must have equal length.")

    predicted = (score_arr >= threshold).astype(int)
    prediction_rows: List[Dict[str, Any]] = []
    for record, label, score, prediction in zip(
        records, y_arr, score_arr, predicted, strict=True
    ):
        gene_set_type = (
            "curated_pathway"
            if int(label) == 1
            else str(record.get("negative_type", "unknown_control"))
        )
        prediction_rows.append(
            {
                "sample_id": str(record["id"]),
                "label": int(label),
                "gene_set_type": gene_set_type,
                "score": float(score),
                "predicted_label_at_0_5": int(prediction),
                "raw_gene_count": int(record.get("raw_gene_count", len(record.get("genes", [])))),
                "annotated_gene_count": int(
                    record.get("annotated_gene_count", len(record.get("genes", [])))
                ),
                "source_pathway_ids": ";".join(record.get("source_pathway_ids", [])),
            }
        )

    matrix = confusion_matrix(y_arr, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    overall_metrics = metrics_from_scores(y_arr, score_arr)
    confusion_rows = [
        {
            "threshold": threshold,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "n_test": len(y_arr),
            "precision": overall_metrics["precision"],
            "recall": overall_metrics["recall"],
            "f1": overall_metrics["f1"],
        }
    ]

    prediction_frame = pd.DataFrame(prediction_rows)
    type_order = [
        "curated_pathway",
        "random_5_30",
        "shuffled",
        "partial_50_80",
        "cross_pathway",
    ]
    type_rows: List[Dict[str, Any]] = []
    for gene_set_type in type_order:
        subset = prediction_frame[prediction_frame["gene_set_type"] == gene_set_type]
        if subset.empty:
            continue
        values = subset["score"].to_numpy(dtype=float)
        is_negative = gene_set_type != "curated_pathway"
        positive_predictions = int((subset["predicted_label_at_0_5"] == 1).sum())
        type_rows.append(
            {
                "gene_set_type": gene_set_type,
                "n": len(subset),
                "mean_score": float(np.mean(values)),
                "score_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q1_score": float(np.quantile(values, 0.25)),
                "median_score": float(np.median(values)),
                "q3_score": float(np.quantile(values, 0.75)),
                "positive_prediction_count_at_0_5": positive_predictions,
                "false_positive_count_at_0_5": positive_predictions if is_negative else 0,
                "false_positive_rate_at_0_5": (
                    float(positive_predictions / len(subset)) if is_negative else float("nan")
                ),
            }
        )

    positive_mask = prediction_frame["label"] == 1
    negative_type_rows: List[Dict[str, Any]] = []
    for negative_type in type_order[1:]:
        negative_mask = prediction_frame["gene_set_type"] == negative_type
        subset = prediction_frame[positive_mask | negative_mask]
        if not negative_mask.any():
            continue
        metrics = metrics_from_scores(
            subset["label"].to_numpy(dtype=int),
            subset["score"].to_numpy(dtype=float),
        )
        negative_type_rows.append(
            {
                "negative_type": negative_type,
                "n_positive": int(positive_mask.sum()),
                "n_negative": int(negative_mask.sum()),
                **metrics,
            }
        )

    return {
        "predictions": prediction_rows,
        "confusion": confusion_rows,
        "score_by_type": type_rows,
        "negative_type_performance": negative_type_rows,
    }


def save_heldout_diagnostics(
    context: PrimaryContext,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Save held-out tables and figures from the fitted reference XGBoost model."""
    diagnostics = heldout_diagnostic_tables(
        context.test_records,
        context.y_test,
        scores,
        threshold=threshold,
    )
    save_table(TABLE_DIR / "heldout_predictions.csv", diagnostics["predictions"])
    save_table(TABLE_DIR / "heldout_confusion_matrix.csv", diagnostics["confusion"])
    save_table(TABLE_DIR / "heldout_score_by_type.csv", diagnostics["score_by_type"])
    save_table(
        TABLE_DIR / "heldout_negative_type_performance.csv",
        diagnostics["negative_type_performance"],
    )

    y_true = np.asarray(context.y_test, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, score_arr)
    precision_values, recall_values, _ = precision_recall_curve(y_true, score_arr)
    auroc = float(roc_auc_score(y_true, score_arr))
    auprc = float(average_precision_score(y_true, score_arr))
    baseline = float(np.mean(y_true == 1))

    blue, orange, teal = "#4c78a8", "#f58518", "#72b7b2"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, color=blue, lw=2.2, label=f"XGBoost (AUROC = {auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Random (0.500)")
    axes[0].set_xlabel("False positive rate (FP / (FP+TN))")
    axes[0].set_ylabel("True positive rate / recall (TP / (TP+FN))")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(
        recall_values,
        precision_values,
        color=orange,
        lw=2.2,
        label=f"XGBoost (AUPRC = {auprc:.3f})",
    )
    axes[1].axhline(
        baseline,
        ls="--",
        color="#999999",
        lw=1,
        label=f"Random baseline ({baseline:.3f})",
    )
    axes[1].set_xlabel("Recall (TP / (TP+FN))")
    axes[1].set_ylabel("Precision (TP / (TP+FP))")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle("Held-out discrimination of the reference XGBoost model", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "FigC_roc_pr.png", dpi=220)
    plt.close(fig)

    confusion_row = diagnostics["confusion"][0]
    matrix = np.array(
        [
            [confusion_row["true_negative"], confusion_row["false_positive"]],
            [confusion_row["false_negative"], confusion_row["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred. control", "Pred. pathway"])
    ax.set_yticks([0, 1], labels=["Actual control", "Actual pathway"])
    cell_labels = [
        [f"TN = {matrix[0, 0]}", f"FP = {matrix[0, 1]}"],
        [f"FN = {matrix[1, 0]}", f"TP = {matrix[1, 1]}"],
    ]
    for row_index in range(2):
        for column_index in range(2):
            ax.text(
                column_index,
                row_index,
                cell_labels[row_index][column_index],
                ha="center",
                va="center",
                color=("white" if matrix[row_index, column_index] > matrix.max() / 2 else "black"),
                fontsize=12,
                fontweight="bold",
            )
    ax.set_title(f"Confusion matrix (threshold = {threshold:.1f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigF_confusion.png", dpi=220)
    plt.close(fig)

    score_frame = pd.DataFrame(diagnostics["predictions"])
    display_names = {
        "curated_pathway": "Curated\npathway",
        "random_5_30": "Random",
        "shuffled": "Shuffled",
        "partial_50_80": "Partial\npathway",
        "cross_pathway": "Cross-pathway\nmixture",
    }
    type_order = list(display_names)
    groups = [
        score_frame.loc[score_frame["gene_set_type"] == gene_set_type, "score"].to_numpy(dtype=float)
        for gene_set_type in type_order
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    boxplot = ax.boxplot(
        groups,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "lw": 1.5},
    )
    for patch, color in zip(
        boxplot["boxes"],
        [teal, "#c7c7c7", "#c7c7c7", orange, orange],
        strict=True,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    jitter_rng = np.random.default_rng(0)
    for position, values in enumerate(groups, start=1):
        ax.scatter(
            jitter_rng.normal(position, 0.05, size=len(values)),
            values,
            s=6,
            color="#333333",
            alpha=0.25,
            zorder=3,
        )
    ax.axhline(threshold, ls="--", color="#999999", lw=1, label="Decision threshold (0.5)")
    ax.set_xticks(range(1, len(type_order) + 1))
    ax.set_xticklabels([display_names[key] for key in type_order], fontsize=9)
    ax.set_ylabel("Model score")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Held-out score distribution by gene-set type", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigG_score_by_type.png", dpi=220)
    plt.close(fig)

    manifest = {
        "source": "reference XGBoost scores from run_main_benchmark",
        "threshold": threshold,
        "primary_ratio": f"1:{context.ratio}",
        "n_test": len(y_true),
        "selected_go_sha256": stable_json_sha256(context.selected_go),
        "test_auroc": auroc,
        "test_auprc": auprc,
        "confusion_at_threshold": confusion_row,
        "figures": ["FigC_roc_pr.png", "FigF_confusion.png", "FigG_score_by_type.png"],
        "tables": [
            "heldout_predictions.csv",
            "heldout_confusion_matrix.csv",
            "heldout_score_by_type.csv",
            "heldout_negative_type_performance.csv",
        ],
    }
    save_json(OUT_DIR / "heldout_diagnostics_manifest.json", manifest)
    return manifest


def build_cv_fold_datasets(
    bundle: DatasetBundle,
    context: PrimaryContext,
    ratio: int,
    fast: bool,
    analysis_name: str,
) -> List[CVFoldData]:
    """Build source-isolated folds with fold-training-only GO selection."""
    if ratio in context.cv_folds_by_ratio:
        return context.cv_folds_by_ratio[ratio]
    positives = context.train_positive_records
    sources = np.asarray([str(record.get("stratification_source", record["source"])) for record in positives])
    n_splits = 3 if fast else 5
    repeats = (SEED,) if fast else CV_REPEAT_SEEDS
    source_counts = Counter(sources.tolist())
    if min(source_counts.values()) < n_splits:
        raise ValueError(f"Cannot run {n_splits}-fold source-stratified CV: {source_counts}")

    outer_test_ids = {str(record["id"]) for record in context.test_positive_records}
    folds: List[CVFoldData] = []
    indices = np.arange(len(positives))
    for repeat_index, repeat_seed in enumerate(repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
        for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(indices, sources)):
            fold_train_pos = sorted(
                (dict(positives[int(i)]) for i in train_idx), key=lambda record: str(record["id"])
            )
            fold_validation_pos = sorted(
                (dict(positives[int(i)]) for i in validation_idx), key=lambda record: str(record["id"])
            )
            train_seed = derive_deterministic_seed(
                SEED, ratio, repeat_seed, fold_index, "train", "negative_generation"
            )
            validation_seed = derive_deterministic_seed(
                SEED, ratio, repeat_seed, fold_index, "validation", "negative_generation"
            )
            feature_seed = derive_deterministic_seed(
                SEED, ratio, repeat_seed, fold_index, "train", "go_selection"
            )
            model_seed = derive_deterministic_seed(
                SEED, ratio, repeat_seed, fold_index, "train", "model_fit"
            )
            fold_train_records = build_side_records(
                fold_train_pos,
                bundle,
                ratio=ratio,
                negative_seed=train_seed,
                prefix=f"CV_R{repeat_index}_F{fold_index}_TRAIN",
            )
            fold_validation_records = build_side_records(
                fold_validation_pos,
                bundle,
                ratio=ratio,
                negative_seed=validation_seed,
                prefix=f"CV_R{repeat_index}_F{fold_index}_VALIDATION",
            )
            assert_source_isolation(
                fold_train_pos,
                fold_validation_pos,
                fold_train_records,
                fold_validation_records,
            )
            fold_source_ids = {
                str(source_id)
                for record in fold_train_records + fold_validation_records
                for source_id in record.get("source_pathway_ids", [])
            }
            if fold_source_ids & outer_test_ids:
                raise AssertionError("An outer-test pathway was used as a CV negative source.")

            selected_go, selection_stages, _ = select_go_terms_from_records(
                fold_train_records,
                bundle.gene_go,
                bundle.go_genes,
                seed=feature_seed,
                k=N_GO_TERMS,
            )
            if len(selected_go) != N_GO_TERMS:
                raise AssertionError(
                    f"Expected {N_GO_TERMS} fold-selected GO terms, got {len(selected_go)}"
                )
            X_train, y_train = records_to_matrix(fold_train_records, selected_go, bundle.gene_go)
            X_validation, y_validation = records_to_matrix(
                fold_validation_records, selected_go, bundle.gene_go
            )
            spw = scale_pos_weight_from_labels(
                y_train,
                analysis_name,
                {
                    "ratio": f"1:{ratio}",
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                },
            )
            folds.append(
                CVFoldData(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    repeat_seed=int(repeat_seed),
                    ratio=ratio,
                    train_negative_seed=train_seed,
                    validation_negative_seed=validation_seed,
                    feature_selection_seed=feature_seed,
                    model_seed=model_seed,
                    train_positive_ids=[str(record["id"]) for record in fold_train_pos],
                    validation_positive_ids=[str(record["id"]) for record in fold_validation_pos],
                    selected_go=list(selected_go),
                    selected_go_sha256=stable_json_sha256(selected_go),
                    feature_selection_stages=dict(selection_stages),
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    train_records=fold_train_records,
                    validation_records=fold_validation_records,
                    scale_pos_weight=spw,
                )
            )
    context.cv_folds_by_ratio[ratio] = folds
    return folds


def cv_auroc_from_folds(
    model_factory: Callable[[float, int], Any],
    folds: Sequence[CVFoldData],
) -> Tuple[float, float]:
    """Evaluate one model on prebuilt folds shared by all reference models."""
    scores: List[float] = []
    for fold in folds:
        model = model_factory(fold.scale_pos_weight, fold.model_seed)
        model.fit(fold.X_train, fold.y_train)
        validation_scores = predict_scores(model, fold.X_validation)
        scores.append(roc_auc_score(fold.y_validation, validation_scores))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


# ---------------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------------
#
# Every run_* function below does three things:
#   1. compute one analysis used by the LaTeX manuscript,
#   2. save a machine-readable CSV under generated/tables/,
#   3. when useful, save a paper-facing figure/table fragment.
#
def reference_model_factories(fast: bool) -> Dict[str, Callable[[float, int], Any]]:
    """Return fresh reference-model builders for fold and held-out fitting."""
    return {
        "XGBoost": lambda spw, model_seed: xgb_model(
            scale_pos_weight=spw, fast=fast, random_state=model_seed
        ),
        "Random Forest": lambda _spw, model_seed: RandomForestClassifier(
            n_estimators=120 if fast else 500,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=model_seed,
            n_jobs=-1,
        ),
        "Logistic Regression": lambda _spw, model_seed: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=model_seed,
                    ),
                ),
            ]
        ),
    }


def save_cv_fold_outputs(folds: Sequence[CVFoldData], context: PrimaryContext) -> None:
    """Persist nested primary-CV metadata without serialising feature matrices."""
    outer_test_ids = {str(record["id"]) for record in context.test_positive_records}
    audit_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for fold in folds:
        train_sources = {
            str(source_id)
            for record in fold.train_records
            for source_id in record.get("source_pathway_ids", [])
        }
        validation_sources = {
            str(source_id)
            for record in fold.validation_records
            for source_id in record.get("source_pathway_ids", [])
        }
        leakage_count = len((train_sources & set(fold.validation_positive_ids)) | (validation_sources & set(fold.train_positive_ids)))
        outer_test_source_count = len((train_sources | validation_sources) & outer_test_ids)
        audit_rows.append(
            {
                "repeat_index": fold.repeat_index,
                "fold_index": fold.fold_index,
                "repeat_seed": fold.repeat_seed,
                "ratio": f"1:{fold.ratio}",
                "n_train_positive": len(fold.train_positive_ids),
                "n_validation_positive": len(fold.validation_positive_ids),
                "n_train_total": len(fold.y_train),
                "n_validation_total": len(fold.y_validation),
                "train_negative_seed": fold.train_negative_seed,
                "validation_negative_seed": fold.validation_negative_seed,
                "feature_selection_seed": fold.feature_selection_seed,
                "model_seed": fold.model_seed,
                "scale_pos_weight": fold.scale_pos_weight,
                "selected_go_sha256": fold.selected_go_sha256,
                "selected_go_count": len(fold.selected_go),
                "cross_split_source_leakage_count": leakage_count,
                "outer_test_source_count": outer_test_source_count,
                "train_records_sha256": stable_json_sha256(
                    [(record["id"], record["genes"]) for record in fold.train_records]
                ),
                "validation_records_sha256": stable_json_sha256(
                    [(record["id"], record["genes"]) for record in fold.validation_records]
                ),
            }
        )
        manifest_rows.append(
            {
                "repeat_index": fold.repeat_index,
                "fold_index": fold.fold_index,
                "repeat_seed": fold.repeat_seed,
                "ratio": fold.ratio,
                "train_negative_seed": fold.train_negative_seed,
                "validation_negative_seed": fold.validation_negative_seed,
                "feature_selection_seed": fold.feature_selection_seed,
                "model_seed": fold.model_seed,
                "train_positive_ids": fold.train_positive_ids,
                "validation_positive_ids": fold.validation_positive_ids,
                "selected_go": fold.selected_go,
                "selected_go_sha256": fold.selected_go_sha256,
                "feature_selection_stages": fold.feature_selection_stages,
            }
        )
        if leakage_count or outer_test_source_count:
            raise AssertionError("CV source-isolation audit failed.")
    save_table(TABLE_DIR / "cv_split_audit.csv", audit_rows)
    save_json(
        DATA_DIR / "cv_fold_manifest.json",
        {
            "feature_selection_protocol": "fold_training_selected60",
            "nested_feature_selection": True,
            "outer_training_selected_go_sha256": stable_json_sha256(context.selected_go),
            "folds": manifest_rows,
        },
    )


def run_main_benchmark(bundle: DatasetBundle, context: PrimaryContext, fast: bool) -> Dict[str, Any]:
    """Main held-out benchmark: XGBoost, RF, and Logistic Regression on an 80:20 split.

    The outer test set did not participate in GO selection or negative-source
    construction.  Repeated CV uses source-isolated folds and repeats GO
    variance/MI selection within each fold's training side.
    """
    ratio_label = f"1:{context.ratio}"
    folds = build_cv_fold_datasets(
        bundle,
        context,
        ratio=context.ratio,
        fast=fast,
        analysis_name="main_benchmark_cv",
    )
    save_cv_fold_outputs(folds, context)
    spw = scale_pos_weight_from_labels(context.y_train, "main_benchmark", {"ratio": ratio_label})
    factories = reference_model_factories(fast)
    rows = []
    fitted: Dict[str, Any] = {}
    heldout_scores: Dict[str, np.ndarray] = {}
    for name, factory in factories.items():
        t0 = time.time()
        cv_mean, cv_sd = cv_auroc_from_folds(factory, folds)
        model = factory(spw, SEED)
        model.fit(context.X_train, context.y_train)
        scores = predict_scores(model, context.X_test)
        row = {
            "model": name,
            "primary_ratio": ratio_label,
            "cv_auroc_mean": cv_mean,
            "cv_auroc_sd": cv_sd,
            "cv_feature_selection_protocol": "nested_fold_training_selected60",
            "heldout_feature_selection_protocol": "outer_training_selected60",
            "scale_pos_weight": spw if name == "XGBoost" else float("nan"),
            "elapsed_s": time.time() - t0,
        }
        row.update(metrics_from_scores(context.y_test, scores))
        rows.append(row)
        fitted[name] = model
        heldout_scores[name] = np.asarray(scores, dtype=float)

    save_table_aliases(rows, "main_benchmark.csv", "table_main_benchmark.csv")
    latex_df = pd.DataFrame(rows)[
        ["model", "cv_auroc_mean", "cv_auroc_sd", "test_auroc", "test_auprc", "f1", "precision", "recall"]
    ].copy()
    latex_df.columns = ["Model", "CV AUROC", "CV SD", "Test AUROC", "AUPRC", "F1", "Precision", "Recall"]
    write_latex_tabular(
        LATEX_TABLE_DIR / "table_main_benchmark.tex",
        latex_df,
        "Main held-out benchmark.",
        "tab:main_benchmark_recomputed",
    )
    save_heldout_diagnostics(context, heldout_scores["XGBoost"])
    return {
        "rows": rows,
        "X_train": context.X_train,
        "y_train": context.y_train,
        "X_test": context.X_test,
        "y_test": context.y_test,
        "train_records": context.train_records,
        "test_records": context.test_records,
        "models": fitted,
        "heldout_scores": heldout_scores,
    }


def normalized_auprc(raw_auprc: float, positive_prevalence: float) -> float:
    """Scale AUPRC above its random baseline while retaining prevalence caveats."""
    if positive_prevalence >= 1.0:
        return float("nan")
    return float((raw_auprc - positive_prevalence) / (1.0 - positive_prevalence))


def evaluate_ratio_folds(folds: Sequence[CVFoldData], fast: bool) -> List[Dict[str, Any]]:
    """Fit one XGBoost model per nested fold and return fold-level metrics."""
    rows: List[Dict[str, Any]] = []
    for fold in folds:
        model = xgb_model(
            scale_pos_weight=fold.scale_pos_weight,
            fast=fast,
            random_state=fold.model_seed,
        )
        started = time.time()
        model.fit(fold.X_train, fold.y_train)
        scores = predict_scores(model, fold.X_validation)
        metrics = metrics_from_scores(fold.y_validation, scores)
        prevalence = float(np.mean(fold.y_validation == 1))
        rows.append(
            {
                "ratio": f"1:{fold.ratio}",
                "ratio_value": fold.ratio,
                "repeat_index": fold.repeat_index,
                "repeat_seed": fold.repeat_seed,
                "fold_index": fold.fold_index,
                "n_train_positive": int((fold.y_train == 1).sum()),
                "n_train_negative": int((fold.y_train == 0).sum()),
                "n_validation_positive": int((fold.y_validation == 1).sum()),
                "n_validation_negative": int((fold.y_validation == 0).sum()),
                "positive_prevalence": prevalence,
                "random_auprc_baseline": prevalence,
                "auroc": metrics["test_auroc"],
                "raw_auprc": metrics["test_auprc"],
                "normalized_auprc": normalized_auprc(metrics["test_auprc"], prevalence),
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "brier": metrics["brier"],
                "scale_pos_weight": fold.scale_pos_weight,
                "train_negative_seed": fold.train_negative_seed,
                "validation_negative_seed": fold.validation_negative_seed,
                "feature_selection_seed": fold.feature_selection_seed,
                "model_seed": fold.model_seed,
                "selected_go_sha256": fold.selected_go_sha256,
                "train_records_sha256": stable_json_sha256(
                    [(record["id"], record["genes"]) for record in fold.train_records]
                ),
                "validation_records_sha256": stable_json_sha256(
                    [(record["id"], record["genes"]) for record in fold.validation_records]
                ),
                "elapsed_s": time.time() - started,
            }
        )
    return rows


def selected_go_stability_rows(folds: Sequence[CVFoldData]) -> List[Dict[str, Any]]:
    """Report fold selection frequency and pairwise overlap for each ratio."""
    selected_sets = [set(fold.selected_go) for fold in folds]
    pairwise = [
        len(left & right) / len(left | right)
        for left, right in combinations(selected_sets, 2)
        if left or right
    ]
    frequencies = Counter(term for fold in folds for term in fold.selected_go)
    return [
        {
            "ratio": f"1:{folds[0].ratio}",
            "go_term": term,
            "selected_fold_count": count,
            "selected_fold_fraction": count / len(folds),
            "n_folds": len(folds),
            "pairwise_jaccard_mean": float(np.mean(pairwise)) if pairwise else 1.0,
            "pairwise_jaccard_sd": float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else 0.0,
            "pairwise_jaccard_min": float(np.min(pairwise)) if pairwise else 1.0,
            "pairwise_jaccard_max": float(np.max(pairwise)) if pairwise else 1.0,
        }
        for term, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    ]


def run_ratio_sensitivity(bundle: DatasetBundle, context: PrimaryContext, fast: bool) -> List[Dict[str, Any]]:
    """Compare class ratios only within outer training by nested repeated CV."""
    per_fold_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    manifest_folds: List[Dict[str, Any]] = []
    outer_test_ids = {str(record["id"]) for record in context.test_positive_records}

    for ratio in SUPPORTED_PRIMARY_RATIOS:
        folds = build_cv_fold_datasets(
            bundle,
            context,
            ratio=ratio,
            fast=fast,
            analysis_name="training_only_ratio_cv",
        )
        per_fold_rows.extend(evaluate_ratio_folds(folds, fast=fast))
        stability_rows.extend(selected_go_stability_rows(folds))
        for fold in folds:
            fold_positive_ids = set(fold.train_positive_ids) | set(fold.validation_positive_ids)
            if fold_positive_ids & outer_test_ids:
                raise AssertionError("Outer-test pathways entered ratio cross-validation.")
            manifest_folds.append(
                {
                    "ratio": fold.ratio,
                    "repeat_index": fold.repeat_index,
                    "repeat_seed": fold.repeat_seed,
                    "fold_index": fold.fold_index,
                    "train_negative_seed": fold.train_negative_seed,
                    "validation_negative_seed": fold.validation_negative_seed,
                    "feature_selection_seed": fold.feature_selection_seed,
                    "model_seed": fold.model_seed,
                    "selected_go": fold.selected_go,
                    "selected_go_sha256": fold.selected_go_sha256,
                    "train_positive_ids": fold.train_positive_ids,
                    "validation_positive_ids": fold.validation_positive_ids,
                    "train_records_sha256": stable_json_sha256(
                        [(record["id"], record["genes"]) for record in fold.train_records]
                    ),
                    "validation_records_sha256": stable_json_sha256(
                        [(record["id"], record["genes"]) for record in fold.validation_records]
                    ),
                }
            )

    per_fold_df = pd.DataFrame(per_fold_rows)
    metric_columns = ["auroc", "raw_auprc", "normalized_auprc", "f1", "precision", "recall", "brier"]
    summary_rows: List[Dict[str, Any]] = []
    for ratio in SUPPORTED_PRIMARY_RATIOS:
        subset = per_fold_df[per_fold_df["ratio_value"] == ratio]
        row: Dict[str, Any] = {
            "ratio": f"1:{ratio}",
            "ratio_value": ratio,
            "n_folds": len(subset),
            "preselected_primary_ratio": ratio == DEFAULT_PRIMARY_RATIO,
            "mean_positive_prevalence": float(subset["positive_prevalence"].mean()),
            "feature_selection_protocol": "nested_fold_training_selected60",
        }
        for metric in metric_columns:
            row[f"mean_{metric}"] = float(subset[metric].mean())
            row[f"sd_{metric}"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
        summary_rows.append(row)

    reference = summary_rows[0]
    for row in summary_rows:
        delta = row["mean_auroc"] - reference["mean_auroc"]
        pooled_sd = math.sqrt((row["sd_auroc"] ** 2 + reference["sd_auroc"] ** 2) / 2.0)
        row["delta_auroc_vs_1_1"] = delta
        row["delta_auroc_over_pooled_sd"] = delta / pooled_sd if pooled_sd > 0 else float("nan")
        row["interpretation_note"] = (
            "Normalized AUPRC adjusts the random baseline but is not fully prevalence invariant."
        )

    save_table(TABLE_DIR / "ratio_cv_per_fold.csv", per_fold_rows)
    save_table(TABLE_DIR / "ratio_cv_summary.csv", summary_rows)
    save_table(TABLE_DIR / "ratio_cv_selected_go_stability.csv", stability_rows)
    save_table_aliases(
        summary_rows,
        "ratio_sensitivity.csv",
        "table_robustness.csv",
        "table_negative_ratio_sensitivity.csv",
    )
    save_json(
        DATA_DIR / "ratio_cv_manifest.json",
        {
            "protocol": "outer-training-only repeated stratified CV",
            "outer_test_used": False,
            "preselected_primary_ratio": "1:1",
            "frequency_filter_scope": "label-free full GO-annotated background",
            "variance_mi_scope": "fold training only",
            "normalized_auprc_note": (
                "The normalization subtracts the random prevalence baseline and rescales the remainder; "
                "it does not make PR performance completely invariant to prevalence."
            ),
            "folds": manifest_folds,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [row["ratio"] for row in summary_rows]
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        [row["mean_auroc"] for row in summary_rows],
        yerr=[row["sd_auroc"] for row in summary_rows],
        marker="o",
        capsize=4,
        label="AUROC",
    )
    ax.errorbar(
        x,
        [row["mean_normalized_auprc"] for row in summary_rows],
        yerr=[row["sd_normalized_auprc"] for row in summary_rows],
        marker="o",
        capsize=4,
        color="#d62728",
        label="Normalized AUPRC",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Positive:negative ratio")
    ax.set_ylabel("Cross-validated metric (mean +/- SD)")
    ax.set_title("Cross-validated performance across class ratios")
    ax.legend()
    fig.tight_layout()
    for filename in ["ratio_cv_performance.png", "ratio_sensitivity.png", "Fig7_robustness.png"]:
        fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)

    latex_df = pd.DataFrame(summary_rows)[
        ["ratio", "n_folds", "mean_auroc", "sd_auroc", "mean_normalized_auprc", "sd_normalized_auprc", "mean_f1"]
    ].copy()
    latex_df.columns = ["Ratio", "Folds", "AUROC", "AUROC SD", "Normalized AUPRC", "AUPRC SD", "F1"]
    write_latex_tabular(
        LATEX_TABLE_DIR / "table_robustness.tex",
        latex_df,
        "Training-only nested cross-validation across positive-to-negative ratios.",
        "tab:robustness_recomputed",
    )
    return summary_rows


def ranked_go_terms_for_fold(
    bundle: DatasetBundle,
    fold: CVFoldData,
    max_count: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Return one fold-training MI ranking up to ``max_count`` terms.

    Feature-count candidates are prefixes of the same fold-specific ranking.
    This keeps the comparison paired: only the number of retained GO terms
    changes, while samples, controls, seeds, and the ranking procedure do not.
    """
    ranked_go, stages, _ = select_go_terms_from_records(
        fold.train_records,
        bundle.gene_go,
        bundle.go_genes,
        seed=fold.feature_selection_seed,
        k=max_count,
    )
    if len(ranked_go) != max_count:
        raise AssertionError(f"Expected {max_count} ranked GO terms, got {len(ranked_go)}.")
    if N_GO_TERMS <= max_count and ranked_go[:N_GO_TERMS] != fold.selected_go:
        raise AssertionError("The top-60 prefix changed during feature-count comparison.")
    return list(ranked_go), dict(stages)


def evaluate_go_feature_count_folds(
    bundle: DatasetBundle,
    folds: Sequence[CVFoldData],
    feature_counts: Sequence[int],
    fast: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate top-k GO representations on the same nested-CV folds."""
    counts = tuple(sorted(set(int(value) for value in feature_counts)))
    if not counts or counts[0] < 1:
        raise ValueError("GO feature counts must contain positive integers.")
    if N_GO_TERMS not in counts:
        raise ValueError(f"The comparison must include the current {N_GO_TERMS}-term representation.")

    per_fold_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    max_count = max(counts)
    for fold in folds:
        ranked_go, stages = ranked_go_terms_for_fold(bundle, fold, max_count)
        train_records_sha256 = stable_json_sha256(
            [(record["id"], record["genes"]) for record in fold.train_records]
        )
        validation_records_sha256 = stable_json_sha256(
            [(record["id"], record["genes"]) for record in fold.validation_records]
        )

        # Rebuild each matrix from the same records and a prefix of the same GO
        # ranking. The outer test set is never encoded or scored in this block.
        for count in counts:
            selected_go = ranked_go[:count]
            X_train, y_train = records_to_matrix(fold.train_records, selected_go, bundle.gene_go)
            X_validation, y_validation = records_to_matrix(
                fold.validation_records,
                selected_go,
                bundle.gene_go,
            )
            model = xgb_model(
                scale_pos_weight=fold.scale_pos_weight,
                fast=fast,
                random_state=fold.model_seed,
            )
            started = time.time()
            model.fit(X_train, y_train)
            scores = predict_scores(model, X_validation)
            metrics = metrics_from_scores(y_validation, scores)
            prevalence = float(np.mean(y_validation == 1))
            selected_go_sha256 = stable_json_sha256(selected_go)
            per_fold_rows.append(
                {
                    "n_go_terms": count,
                    "n_total_features": count + 9,
                    "repeat_index": fold.repeat_index,
                    "repeat_seed": fold.repeat_seed,
                    "fold_index": fold.fold_index,
                    "ratio": f"1:{fold.ratio}",
                    "n_train_positive": int((y_train == 1).sum()),
                    "n_train_negative": int((y_train == 0).sum()),
                    "n_validation_positive": int((y_validation == 1).sum()),
                    "n_validation_negative": int((y_validation == 0).sum()),
                    "positive_prevalence": prevalence,
                    "auroc": metrics["test_auroc"],
                    "auprc": metrics["test_auprc"],
                    "normalized_auprc": normalized_auprc(metrics["test_auprc"], prevalence),
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "brier": metrics["brier"],
                    "scale_pos_weight": fold.scale_pos_weight,
                    "train_negative_seed": fold.train_negative_seed,
                    "validation_negative_seed": fold.validation_negative_seed,
                    "feature_selection_seed": fold.feature_selection_seed,
                    "model_seed": fold.model_seed,
                    "selected_go_sha256": selected_go_sha256,
                    "train_records_sha256": train_records_sha256,
                    "validation_records_sha256": validation_records_sha256,
                    "elapsed_s": time.time() - started,
                }
            )
            manifest_rows.append(
                {
                    "n_go_terms": count,
                    "n_total_features": count + 9,
                    "repeat_index": fold.repeat_index,
                    "repeat_seed": fold.repeat_seed,
                    "fold_index": fold.fold_index,
                    "selected_go": selected_go,
                    "selected_go_sha256": selected_go_sha256,
                    "feature_selection_stages": {**stages, "mi_select": count},
                    "train_records_sha256": train_records_sha256,
                    "validation_records_sha256": validation_records_sha256,
                    "train_negative_seed": fold.train_negative_seed,
                    "validation_negative_seed": fold.validation_negative_seed,
                    "feature_selection_seed": fold.feature_selection_seed,
                    "model_seed": fold.model_seed,
                }
            )
    return per_fold_rows, manifest_rows


def run_go_feature_count_sensitivity(
    bundle: DatasetBundle,
    context: PrimaryContext,
    fast: bool,
) -> List[Dict[str, Any]]:
    """Compare 20-100 selected GO terms using outer-training nested CV only."""
    if context.ratio != DEFAULT_PRIMARY_RATIO:
        raise ValueError("GO feature-count sensitivity is defined for the primary 1:1 ratio.")

    folds = build_cv_fold_datasets(
        bundle,
        context,
        ratio=DEFAULT_PRIMARY_RATIO,
        fast=fast,
        analysis_name="go_feature_count_cv",
    )
    outer_test_ids = {str(record["id"]) for record in context.test_positive_records}
    for fold in folds:
        if (set(fold.train_positive_ids) | set(fold.validation_positive_ids)) & outer_test_ids:
            raise AssertionError("Outer-test pathways entered GO feature-count cross-validation.")

    per_fold_rows, manifest_rows = evaluate_go_feature_count_folds(
        bundle,
        folds,
        GO_FEATURE_COUNT_CANDIDATES,
        fast=fast,
    )
    per_fold_df = pd.DataFrame(per_fold_rows)
    metric_columns = ["auroc", "auprc", "normalized_auprc", "f1", "precision", "recall", "brier"]
    summary_rows: List[Dict[str, Any]] = []
    for count in GO_FEATURE_COUNT_CANDIDATES:
        subset = per_fold_df[per_fold_df["n_go_terms"] == count]
        row: Dict[str, Any] = {
            "n_go_terms": count,
            "n_total_features": count + 9,
            "n_folds": len(subset),
            "preselected_feature_count": count == N_GO_TERMS,
            "feature_selection_protocol": "fold_training_variance_mi_top_k",
            "outer_test_used": False,
        }
        for metric in metric_columns:
            row[f"mean_{metric}"] = float(subset[metric].mean())
            row[f"sd_{metric}"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
        summary_rows.append(row)

    reference = next(row for row in summary_rows if row["n_go_terms"] == N_GO_TERMS)
    for row in summary_rows:
        for metric in ("auroc", "auprc", "f1"):
            delta = row[f"mean_{metric}"] - reference[f"mean_{metric}"]
            pooled_sd = math.sqrt(
                (row[f"sd_{metric}"] ** 2 + reference[f"sd_{metric}"] ** 2) / 2.0
            )
            row[f"delta_{metric}_vs_60"] = delta
            row[f"delta_{metric}_over_pooled_sd_vs_60"] = (
                delta / pooled_sd if pooled_sd > 0 else float("nan")
            )

    best_auroc = max(summary_rows, key=lambda row: row["mean_auroc"])
    best_auprc = max(summary_rows, key=lambda row: row["mean_auprc"])
    best_f1 = max(summary_rows, key=lambda row: row["mean_f1"])
    larger_rows = [row for row in summary_rows if row["n_go_terms"] > N_GO_TERMS]
    larger_improvements_within_one_pooled_sd = all(
        row["delta_auroc_over_pooled_sd_vs_60"] <= 1.0
        and row["delta_auprc_over_pooled_sd_vs_60"] <= 1.0
        for row in larger_rows
    )

    save_table(TABLE_DIR / "go_feature_count_cv_per_fold.csv", per_fold_rows)
    save_table(TABLE_DIR / "go_feature_count_cv_summary.csv", summary_rows)
    save_json(
        DATA_DIR / "go_feature_count_cv_manifest.json",
        {
            "protocol": "outer-training-only repeated stratified nested cross-validation",
            "primary_ratio": "1:1",
            "candidate_go_counts": list(GO_FEATURE_COUNT_CANDIDATES),
            "preselected_go_count": N_GO_TERMS,
            "outer_test_used": False,
            "frequency_filter_scope": "label-free full GO-annotated background",
            "variance_mi_scope": "fold training only",
            "automatic_selection_performed": False,
            "best_mean_auroc_go_count": int(best_auroc["n_go_terms"]),
            "best_mean_auprc_go_count": int(best_auprc["n_go_terms"]),
            "best_mean_f1_go_count": int(best_f1["n_go_terms"]),
            "larger_count_improvements_within_one_pooled_sd_of_60": (
                larger_improvements_within_one_pooled_sd
            ),
            "interpretation_note": (
                "This is a sensitivity analysis of the preselected 60-term representation. "
                "It reports paired fold differences and does not use the outer test set or "
                "automatically select a winner."
            ),
            "folds": manifest_rows,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.asarray([row["n_go_terms"] for row in summary_rows], dtype=int)
    ax.errorbar(
        x,
        [row["mean_auroc"] for row in summary_rows],
        yerr=[row["sd_auroc"] for row in summary_rows],
        marker="o",
        capsize=4,
        label="AUROC",
    )
    ax.errorbar(
        x,
        [row["mean_auprc"] for row in summary_rows],
        yerr=[row["sd_auprc"] for row in summary_rows],
        marker="o",
        capsize=4,
        color="#d62728",
        label="AUPRC",
    )
    ax.axvline(N_GO_TERMS, color="#666666", linestyle="--", linewidth=1.2, label="Current setting (60)")
    ax.set_xticks(x)
    ax.set_xlabel("Number of selected GO terms")
    ax.set_ylabel("Cross-validated metric (mean +/- SD)")
    ax.set_title("Cross-validated performance across GO feature counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "go_feature_count_cv_performance.png", dpi=220)
    plt.close(fig)
    return summary_rows


def model_catalog(fast: bool, scale_pos_weight: float) -> Dict[str, Any]:
    """Return the 13-method model catalogue for the systematic comparison.

    Includes: Logistic Regression, Linear SVM, RBF SVM, k-NN, Gaussian NB,
    Random Forest, Extra Trees, Gradient Boosting, XGBoost, MLP, AdaBoost,
    and optionally LightGBM and CatBoost (if installed).
    """
    models: Dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=SEED))]
        ),
        "RBF SVM": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=SEED))]
        ),
        "k-NN (k=7)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=120 if fast else 500,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120 if fast else 500,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100 if fast else 300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED
        ),
        "XGBoost": xgb_model(scale_pos_weight=scale_pos_weight, fast=fast),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48) if fast else (32,),
                        max_iter=160 if fast else 80,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=80 if fast else 200, random_state=SEED),
    }
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=160 if fast else 500,
            max_depth=5,
            learning_rate=0.05 if fast else 0.03,
            num_leaves=31 if fast else 63,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if cb is not None:
        models["CatBoost"] = cb.CatBoostClassifier(
            iterations=160 if fast else 500,
            depth=5,
            learning_rate=0.05 if fast else 0.03,
            scale_pos_weight=scale_pos_weight,
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
    return models


def run_model_comparison(
    bundle: DatasetBundle,
    context: PrimaryContext,
    fast: bool,
) -> List[Dict[str, Any]]:
    """Systematic comparison of all 13 ML methods on the same dataset.

    The 13-model comparison is a supplementary model-selection check, not the
    main validation protocol.  It therefore uses the same seed-42 held-out split
    and reports held-out metrics only.  Cross-validation is reserved for the
    main XGBoost/RF/LR benchmark and ratio-sensitivity analyses.
    """
    # All models see exactly the same samples and the same 69 columns.  This
    # makes the table a model comparison rather than a comparison of different
    # random splits or feature-selection outcomes.
    X_train, y_train = context.X_train, context.y_train
    X_test, y_test = context.X_test, context.y_test
    ratio_label = f"1:{context.ratio}"
    spw = scale_pos_weight_from_labels(
        y_train,
        "supplementary_model_comparison",
        {"ratio": ratio_label},
    )
    selected_go_sha256 = stable_json_sha256(context.selected_go)
    split_sha256 = primary_split_sha256(context)
    rows = []
    models = model_catalog(fast, spw)
    missing_optional = []
    if lgb is None:
        missing_optional.append("LightGBM")
    if cb is None:
        missing_optional.append("CatBoost")

    for name, model in models.items():
        print(f"  - {name}", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        scores = predict_scores(model, X_test)
        row = {
            "model": name,
            "primary_ratio": ratio_label,
            "model_comparison_protocol": MODEL_COMPARISON_PROTOCOL,
            "feature_selection_protocol": "primary_train_fixed60",
            "selected_go_sha256": selected_go_sha256,
            "primary_split_sha256": split_sha256,
            "status": "ok",
            "cv_auroc_mean": float("nan"),
            "cv_auroc_sd": float("nan"),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "scale_pos_weight": spw,
            "elapsed_s": time.time() - t0,
        }
        row.update(metrics_from_scores(y_test, scores))
        rows.append(row)
        # Checkpoint after every method.  Full model comparison can be slow, so
        # this prevents losing completed methods if a later estimator hangs.
        partial_rows = sorted(rows, key=lambda r: r["test_auroc"], reverse=True)
        for i, partial_row in enumerate(partial_rows, 1):
            partial_row["rank_by_auroc"] = i
        save_table(TABLE_DIR / "model_comparison_partial.csv", partial_rows)

    for name in missing_optional:
        rows.append(
            {
                "model": name,
                "primary_ratio": ratio_label,
                "model_comparison_protocol": MODEL_COMPARISON_PROTOCOL,
                "feature_selection_protocol": "primary_train_fixed60",
                "selected_go_sha256": selected_go_sha256,
                "primary_split_sha256": split_sha256,
                "status": "skipped_missing_dependency",
                "cv_auroc_mean": float("nan"),
                "cv_auroc_sd": float("nan"),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "scale_pos_weight": spw,
                "elapsed_s": float("nan"),
                "test_auroc": float("nan"),
                "test_auprc": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "brier": float("nan"),
            }
        )

    rows = sorted(rows, key=lambda r: (-1 if pd.isna(r["test_auroc"]) else r["test_auroc"]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank_by_auroc"] = i
    save_table_aliases(rows, "model_comparison.csv", "table_method_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ordered = rows[::-1]
    axes[0].barh([r["model"] for r in ordered], [r["test_auroc"] for r in ordered])
    axes[0].set_xlabel("Test AUROC")
    axes[0].set_title("Method ranking")
    axes[1].barh([r["model"] for r in ordered], [r["test_auprc"] for r in ordered], color="#4c78a8")
    axes[1].set_xlabel("Test AUPRC")
    axes[1].set_title("Precision-recall performance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_auroc.png", dpi=220)
    fig.savefig(FIG_DIR / "Fig8_methods.png", dpi=220)
    plt.close(fig)

    latex_df = pd.DataFrame(rows)[
        ["rank_by_auroc", "model", "model_comparison_protocol", "test_auroc", "test_auprc", "f1", "brier", "elapsed_s"]
    ].copy()
    latex_df.columns = ["Rank", "Method", "Protocol", "Test AUROC", "AUPRC", "F1", "Brier", "Time (s)"]
    write_latex_tabular(
        LATEX_TABLE_DIR / "table_method_comparison.tex",
        latex_df,
        (
            "Systematic comparison of machine-learning methods. Wall-clock "
            "times use the environment recorded in compute_environment.csv."
        ),
        "tab:method_comparison_recomputed",
    )
    return rows


def feature_group_indices(bundle: DatasetBundle) -> Dict[str, List[int]]:
    """Define feature-group ablation configurations.

    Returns a dict mapping configuration name -> list of column indices.
    Configurations include the full model, each feature group alone, and each
    one-group removal. Two-group "keep" configurations are omitted because they
    duplicate the corresponding one-group removal.
    """
    n_go = len(bundle.selected_go)
    return {
        "full": list(range(n_go + 9)),
        "go_only": list(range(n_go)),
        "jaccard_only": list(range(n_go, n_go + 4)),
        "size_only": list(range(n_go + 4, n_go + 9)),
        "minus_go": list(range(n_go, n_go + 9)),
        "minus_jaccard": list(range(n_go)) + list(range(n_go + 4, n_go + 9)),
        "minus_size": list(range(n_go + 4)),
    }
    # Note: "keep two groups" configs (e.g. GO+Jaccard) are omitted because they
    # are identical to the corresponding leave-one-out configs (e.g. minus_size),
    # which would produce duplicate rows.


def run_ablation(bundle: DatasetBundle, context: PrimaryContext, fast: bool) -> List[Dict[str, Any]]:
    """Feature-group ablation: measure AUROC drop when removing each feature group.

    Tests 7 configurations (full, each group alone, and each group removed).
    Produces table_ablation.csv and Fig_ablation.png.
    """
    X_train, y_train = context.X_train, context.y_train
    X_test, y_test = context.X_test, context.y_test
    ratio_label = f"1:{context.ratio}"
    spw = scale_pos_weight_from_labels(y_train, "feature_ablation", {"ratio": ratio_label})
    rows = []
    for name, cols in feature_group_indices(bundle).items():
        model = xgb_model(scale_pos_weight=spw, fast=fast)
        model.fit(X_train[:, cols], y_train)
        scores = predict_scores(model, X_test[:, cols])
        row = {
            "configuration": name,
            "primary_ratio": ratio_label,
            "d": len(cols),
            "feature_selection_protocol": "primary_train_fixed60",
            "scale_pos_weight": spw,
        }
        row.update(metrics_from_scores(y_test, scores))
        rows.append(row)
    full = next(r["test_auroc"] for r in rows if r["configuration"] == "full")
    for row in rows:
        row["delta_auroc_vs_full"] = row["test_auroc"] - full
    save_table_aliases(rows, "ablation.csv", "table_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = sorted(rows, key=lambda r: r["test_auroc"])
    ax.barh([r["configuration"] for r in ordered], [r["test_auroc"] for r in ordered], color="#59a14f")
    ax.set_xlabel("Test AUROC")
    ax.set_title("Feature-group ablation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation.png", dpi=220)
    fig.savefig(FIG_DIR / "Fig_ablation.png", dpi=220)
    plt.close(fig)

    latex_df = pd.DataFrame(rows)[["configuration", "d", "test_auroc", "delta_auroc_vs_full"]].copy()
    latex_df.columns = ["Configuration", "$d$", "Test AUROC", "$\\Delta$ AUROC"]
    write_latex_tabular(
        LATEX_TABLE_DIR / "table_ablation.tex",
        latex_df,
        "Feature-group ablation.",
        "tab:ablation_recomputed",
    )
    return rows


def run_importance(
    bundle: DatasetBundle,
    context: PrimaryContext,
    model: Any,
    X_test: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute feature importance via SHAP (preferred) or model importances (fallback).

    Uses TreeExplainer on up to 300 test samples; if SHAP is unavailable,
    falls back to XGBoost's built-in feature_importances_.
    Produces table_top_features.csv and Fig3_shap.png.
    """
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_test[: min(300, len(X_test))])
            if isinstance(vals, list):
                vals = vals[-1]
            importance = np.mean(np.abs(vals), axis=0)
            method = "shap_mean_abs"
        except Exception:
            importance = getattr(model, "feature_importances_", np.zeros(len(bundle.feature_names)))
            method = "model_feature_importance_fallback"
    else:
        importance = getattr(model, "feature_importances_", np.zeros(len(bundle.feature_names)))
        method = "model_feature_importance_fallback"
    rows = [
        {
            "rank": i + 1,
            "feature": bundle.feature_names[idx],
            "importance": float(importance[idx]),
            "method": method,
            "primary_ratio": f"1:{context.ratio}",
        }
        for i, idx in enumerate(np.argsort(importance)[::-1][:30])
    ]
    save_table_aliases(rows, "feature_importance.csv", "table_top_features.csv")

    top = rows[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([r["feature"] for r in top], [r["importance"] for r in top], color="#76b7b2")
    ax.set_xlabel("Importance")
    ax.set_title("Top feature attributions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=220)
    fig.savefig(FIG_DIR / "Fig3_shap.png", dpi=220)
    plt.close(fig)

    latex_df = pd.DataFrame(rows[:10])[["rank", "feature", "importance", "method"]].copy()
    latex_df.columns = ["Rank", "Feature", "Importance", "Method"]
    write_latex_tabular(
        LATEX_TABLE_DIR / "table_top_features.tex",
        latex_df,
        "Top features by attribution.",
        "tab:top_features_recomputed",
    )
    return rows


def save_primary_context_outputs(
    context: PrimaryContext,
    gene_go: Mapping[str, set],
) -> None:
    """Persist the primary split, negative provenance, and GO-selection audit.

    JSON keeps the complete sample records (including gene IDs), while the CSV
    files provide compact summaries suitable for manual inspection and thesis
    appendices.  Gene lists are hashed rather than embedded in CSV cells.
    """
    selected_go_sha256 = stable_json_sha256(context.selected_go)
    split_sha256 = primary_split_sha256(context)
    save_json(DATA_DIR / "selected_go_terms.json", context.selected_go)
    save_table(TABLE_DIR / "go_selection_audit.csv", context.go_selection_audit)

    split_payload = {
        "protocol": {
            "primary_ratio": f"1:{context.ratio}",
            "split_unit": "unique_exact_gene_set_group",
            "stratification": "database_source",
            "test_size": PRIMARY_TEST_SIZE,
            "split_seed": context.split_seed,
            "train_negative_seed": context.train_negative_seed,
            "test_negative_seed": context.test_negative_seed,
            "feature_selection_scope": "outer_train_only",
            "jaccard_feature_protocol": jaccard_feature_protocol(),
            "selected_go_sha256": selected_go_sha256,
            "primary_split_sha256": split_sha256,
        },
        "train_positive_records": context.train_positive_records,
        "test_positive_records": context.test_positive_records,
        "train_records": context.train_records,
        "test_records": context.test_records,
    }
    save_json(DATA_DIR / "main_split.json", split_payload)

    train_positive_ids = {str(record["id"]) for record in context.train_positive_records}
    test_positive_ids = {str(record["id"]) for record in context.test_positive_records}
    negative_rows: List[Dict[str, Any]] = []
    split_audit_rows: List[Dict[str, Any]] = []
    jaccard_audit_rows: List[Dict[str, Any]] = []
    for split_name, records, own_positive_ids, other_positive_ids in [
        ("train", context.train_records, train_positive_ids, test_positive_ids),
        ("test", context.test_records, test_positive_ids, train_positive_ids),
    ]:
        positives = [record for record in records if int(record["label"]) == 1]
        negatives = [record for record in records if int(record["label"]) == 0]
        source_leakage_count = 0
        for record in negatives:
            source_ids = {str(source_id) for source_id in record.get("source_pathway_ids", [])}
            source_leakage_count += int(bool(source_ids & other_positive_ids) or not source_ids.issubset(own_positive_ids))
            negative_rows.append(
                {
                    "sample_id": record["id"],
                    "split": split_name,
                    "negative_type": record["negative_type"],
                    "raw_gene_count": record.get("raw_gene_count", len(record["genes"])),
                    "annotated_gene_count": record.get(
                        "annotated_gene_count", len(record["genes"])
                    ),
                    "source_pathway_ids": ";".join(record.get("source_pathway_ids", [])),
                    "source_overlap_count": record.get("source_overlap_count", 0),
                    "source_overlap_fraction": record.get("source_overlap_fraction", 0.0),
                    "generation_seed": record.get("generation_seed"),
                    "generation_attempt": record.get("generation_attempt"),
                    "source_raw_gene_count": record.get("source_raw_gene_count", 0),
                    "source_annotated_gene_count": record.get("source_annotated_gene_count", 0),
                    "retention_fraction": record.get("retention_fraction"),
                    "gene_set_sha256": stable_json_sha256(record["genes"]),
                }
            )

        # Record which primary samples use exact pairwise comparisons and
        # which use the deterministic cap-30 approximation.  The sampled gene
        # IDs remain available in main_split.json; hashes keep this CSV compact.
        for record in records:
            annotated_genes = sorted(
                {gene for gene in record["genes"] if gene in gene_go}
            )
            sampled_genes = select_genes_for_jaccard(annotated_genes)
            n_all = len(annotated_genes)
            n_used = len(sampled_genes)
            jaccard_audit_rows.append(
                {
                    "sample_id": record["id"],
                    "split": split_name,
                    "label": int(record["label"]),
                    "gene_set_type": record.get("negative_type", "curated_pathway"),
                    "annotated_gene_count": n_all,
                    "jaccard_gene_count": n_used,
                    "jaccard_mode": "exact" if n_all <= JACCARD_EXACT_MAX_GENES else "hash_sampled",
                    "all_pair_count": n_all * (n_all - 1) // 2,
                    "evaluated_pair_count": n_used * (n_used - 1) // 2,
                    "complete_gene_set_sha256": stable_json_sha256(annotated_genes),
                    "jaccard_subset_sha256": stable_json_sha256(sampled_genes),
                }
            )

        type_counts = Counter(record["negative_type"] for record in negatives)
        split_audit_rows.append(
            {
                "split": split_name,
                "primary_ratio": f"1:{context.ratio}",
                "n_positive": len(positives),
                "n_negative": len(negatives),
                "n_total": len(records),
                "n_kegg_positive": sum(record.get("source") == "KEGG" for record in positives),
                "n_aracyc_positive": sum(record.get("source") == "AraCyc" for record in positives),
                "n_random_5_30": type_counts.get("random_5_30", 0),
                "n_shuffled": type_counts.get("shuffled", 0),
                "n_partial_50_80": type_counts.get("partial_50_80", 0),
                "n_cross_pathway": type_counts.get("cross_pathway", 0),
                "cross_split_source_leakage_count": source_leakage_count,
                "records_sha256": stable_json_sha256([(record["id"], record["genes"]) for record in records]),
                "selected_go_sha256": selected_go_sha256,
                "primary_split_sha256": split_sha256,
            }
        )
        if source_leakage_count:
            raise AssertionError(f"Primary {split_name} source-isolation audit failed.")

    save_table(TABLE_DIR / "negative_metadata.csv", negative_rows)
    save_table(TABLE_DIR / "main_split_audit.csv", split_audit_rows)
    save_table(TABLE_DIR / "jaccard_sampling_audit.csv", jaccard_audit_rows)


def save_processed_data(bundle: DatasetBundle, context: PrimaryContext) -> None:
    """Persist source records, modelling groups, features, and dataset audits."""
    save_json(DATA_DIR / "all_pathways.json", bundle.pathways)
    save_json(DATA_DIR / "pathway_source_records.json", bundle.source_pathways)
    save_json(DATA_DIR / "pathway_gene_set_groups.json", bundle.pathways)
    save_table(TABLE_DIR / "pathway_duplicate_groups.csv", bundle.duplicate_groups)
    save_table(TABLE_DIR / "pathway_mixed_source_groups.csv", bundle.mixed_source_groups)
    save_table(TABLE_DIR / "pathway_grouping_summary.csv", [bundle.grouping_summary])
    save_json(DATA_DIR / "gene_go.json", {k: sorted(v) for k, v in bundle.gene_go.items()})
    save_json(DATA_DIR / "go_term_names.json", bundle.go_term_names)
    save_json(DATA_DIR / "selected_go_terms.json", bundle.selected_go)
    save_json(
        DATA_DIR / "feature_info.json",
        {
            "selected_go": bundle.selected_go,
            "feature_names": bundle.feature_names,
            "feat_sel_stages": bundle.feature_stages,
            "data_source": bundle.source,
            "primary_ratio": f"1:{context.ratio}",
            "go_selection_mode": GO_SELECTION_MODE,
            "feature_selection_scope": "outer_train_only",
            "modelling_unit": "unique_exact_gene_set_group",
            "jaccard_feature_protocol": jaccard_feature_protocol(),
        },
    )
    rows = [
        {
            "id": r["id"],
            "name": r["name"],
            "source": r["source"],
            "raw_gene_count": r["raw_gene_count"],
            "annotated_gene_count": r["annotated_gene_count"],
            "gene_set_group_id": r["gene_set_group_id"],
            "gene_set_sha256": r["gene_set_sha256"],
            "source_record_count": r["source_record_count"],
            "mixed_source": r["mixed_source"],
        }
        for r in pathway_records(bundle)
    ]
    save_table(TABLE_DIR / "pathway_inventory.csv", rows)
    save_table(
        TABLE_DIR / "pathway_source_inventory.csv",
        [
            {
                "pathway_id": pathway_id,
                "pathway_name": info.get("name", pathway_id),
                "source": info.get("source", ""),
                "raw_gene_count": len(info["genes"]),
                "gene_set_sha256": gene_set_sha256(info["genes"]),
            }
            for pathway_id, info in sorted(bundle.source_pathways.items())
        ],
    )
    summary = {
        "dataset_source": bundle.source,
        "primary_ratio": f"1:{context.ratio}",
        "positive_fraction_train": float(np.mean(context.y_train == 1)),
        "positive_fraction_test": float(np.mean(context.y_test == 1)),
        "n_pathway_source_records": len(bundle.source_pathways),
        "n_source_kegg": sum(1 for r in bundle.source_pathways.values() if r["source"] == "KEGG"),
        "n_source_aracyc": sum(1 for r in bundle.source_pathways.values() if r["source"] == "AraCyc"),
        "n_modelling_groups": len(rows),
        "n_group_kegg_stratum": sum(1 for r in rows if r["source"] == "KEGG"),
        "n_group_aracyc_stratum": sum(1 for r in rows if r["source"] == "AraCyc"),
        "n_pathways": len(rows),
        "n_kegg": sum(1 for r in rows if r["source"] == "KEGG"),
        "n_aracyc": sum(1 for r in rows if r["source"] == "AraCyc"),
        "n_exact_duplicate_groups": bundle.grouping_summary.get("exact_duplicate_group_count", 0),
        "n_collapsed_extra_records": bundle.grouping_summary.get("collapsed_extra_record_count", 0),
        "n_mixed_source_groups": bundle.grouping_summary.get("mixed_source_group_count", 0),
        "n_gene_go": len(bundle.gene_go),
        "n_raw_go_terms": len(bundle.go_genes),
        "n_selected_go": len(bundle.selected_go),
        "n_features": len(bundle.feature_names),
        "jaccard_feature_protocol": jaccard_feature_protocol(),
    }
    pathway_member_genes = {
        gene
        for pathway in bundle.source_pathways.values()
        for gene in pathway["genes"]
    }
    annotated_pathway_genes = pathway_member_genes & set(bundle.gene_go)
    coverage_fraction = (
        len(annotated_pathway_genes) / len(pathway_member_genes)
        if pathway_member_genes
        else 0.0
    )
    summary.update(
        {
            "n_pathway_member_genes": len(pathway_member_genes),
            "n_go_annotated_pathway_genes": len(annotated_pathway_genes),
            "pathway_gene_go_coverage": coverage_fraction,
        }
    )
    save_json(
        DATA_DIR / "pathway_gene_coverage.json",
        {
            "pathway_member_genes": len(pathway_member_genes),
            "go_annotated_pathway_genes": len(annotated_pathway_genes),
            "coverage_fraction": round(coverage_fraction, 6),
            "coverage_percent": round(100 * coverage_fraction, 2),
            "definition": "fraction of curated-pathway genes with at least one GO annotation",
        },
    )
    save_json(TABLE_DIR / "dataset_summary.json", summary)
    save_table(TABLE_DIR / "dataset_summary.csv", [summary])

    # Figure used by the manuscript's data-statistics panel.  The original PDF
    # used a richer composed figure; this reproducible replacement focuses on
    # the auditable quantities that come directly from the data snapshot.
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(
        ["KEGG", "AraCyc"],
        [summary["n_group_kegg_stratum"], summary["n_group_aracyc_stratum"]],
        color=["#4e79a7", "#59a14f"],
    )
    axes[0].set_title("Modelling groups by source stratum")
    axes[0].set_ylabel("Unique gene-set groups")
    axes[1].hist(df["raw_gene_count"], bins=30, color="#f28e2b", edgecolor="white")
    axes[1].set_title("Gene-set size distribution")
    axes[1].set_xlabel("Genes per modelling group")
    grouping_labels = ["Source records", "Modelling groups", "Duplicate groups"]
    grouping_values = [
        summary["n_pathway_source_records"],
        summary["n_modelling_groups"],
        summary["n_exact_duplicate_groups"],
    ]
    axes[2].bar(grouping_labels, grouping_values, color=["#bab0ab", "#76b7b2", "#e15759"])
    axes[2].set_title("Exact gene-set grouping")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_statistics.png", dpi=220)
    fig.savefig(FIG_DIR / "Fig5_datastats.png", dpi=220)
    plt.close(fig)


def xgboost_hyperparameters(fast: bool) -> Dict[str, Any]:
    """Canonical XGBoost hyperparameters for audit output."""
    return {
        "n_estimators": 40 if fast else 500,
        "max_depth": 5,
        "learning_rate": 0.08 if fast else 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "scale_pos_weight_rule": SCALE_POS_WEIGHT_RULE,
        "random_state": SEED,
        "n_jobs": -1,
    }


def random_forest_hyperparameters(fast: bool) -> Dict[str, Any]:
    """Canonical Random Forest hyperparameters for audit output."""
    return {
        "n_estimators": 120 if fast else 500,
        "max_depth": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }


def logistic_regression_hyperparameters() -> Dict[str, Any]:
    """Canonical Logistic Regression hyperparameters for audit output."""
    return {
        "solver": "lbfgs",
        "max_iter": 5000,
        "class_weight": "balanced",
        "random_state": SEED,
    }


def lightgbm_hyperparameters(fast: bool) -> Dict[str, Any]:
    """LightGBM supplementary-comparison parameters."""
    return {
        "n_estimators": 160 if fast else 500,
        "max_depth": 5,
        "learning_rate": 0.05 if fast else 0.03,
        "num_leaves": 31 if fast else 63,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight_rule": SCALE_POS_WEIGHT_RULE,
        "random_state": SEED,
        "n_jobs": -1,
    }


def catboost_hyperparameters(fast: bool) -> Dict[str, Any]:
    """CatBoost supplementary-comparison parameters."""
    return {
        "iterations": 160 if fast else 500,
        "depth": 5,
        "learning_rate": 0.05 if fast else 0.03,
        "scale_pos_weight_rule": SCALE_POS_WEIGHT_RULE,
        "random_seed": SEED,
        "allow_writing_files": False,
        "thread_count": -1,
    }


def hyperparameter_audit_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Create a compact model-parameter audit table."""
    paper_xgb = xgboost_hyperparameters(fast=False)
    quick_xgb = xgboost_hyperparameters(fast=True)
    paper_rf = random_forest_hyperparameters(fast=False)
    quick_rf = random_forest_hyperparameters(fast=True)
    lr = logistic_regression_hyperparameters()
    paper_lgb = lightgbm_hyperparameters(fast=False)
    quick_lgb = lightgbm_hyperparameters(fast=True)
    paper_cb = catboost_hyperparameters(fast=False)
    quick_cb = catboost_hyperparameters(fast=True)
    rows: List[Dict[str, Any]] = []
    for model, paper, quick, actual in [
        ("XGBoost", paper_xgb, quick_xgb, xgboost_hyperparameters(args.quick)),
        ("Random Forest", paper_rf, quick_rf, random_forest_hyperparameters(args.quick)),
        ("Logistic Regression", lr, lr, lr),
        ("LightGBM", paper_lgb, quick_lgb, lightgbm_hyperparameters(args.quick)),
        ("CatBoost", paper_cb, quick_cb, catboost_hyperparameters(args.quick)),
    ]:
        for parameter in sorted(set(paper) | set(quick) | set(actual)):
            status = "dynamic" if parameter == "scale_pos_weight_rule" else "match"
            if model == "LightGBM" and lgb is None:
                status = "not_applicable"
            if model == "CatBoost" and cb is None:
                status = "not_applicable"
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "paper_full_value": paper.get(parameter),
                    "quick_debug_value": quick.get(parameter),
                    "actual_used_value": actual.get(parameter),
                    "status": status,
                    "notes": "quick differs intentionally for fast validation runs" if paper.get(parameter) != quick.get(parameter) else "",
                }
            )
    for model, installed in [("LightGBM", lgb is not None), ("CatBoost", cb is not None)]:
        rows.append(
            {
                "model": model,
                "parameter": "availability",
                "paper_full_value": "optional_dependency",
                "quick_debug_value": "optional_dependency",
                "actual_used_value": "installed" if installed else "skipped_missing_dependency",
                "status": "match" if installed else "not_applicable",
                "notes": "supplementary model comparison only",
            }
        )
    return rows


def cv_scheme_audit_rows(args: argparse.Namespace, sections: Sequence[str]) -> List[Dict[str, Any]]:
    """Record the actual validation protocol used by each analysis section."""
    full_sections = ["main", "ratio", "models", "ablation", "importance"]
    section_set = set(full_sections if "all" in sections else sections)
    cv_splits = 3 if args.quick else 5
    cv_repeats = 1 if args.quick else 3
    rows: List[Dict[str, Any]] = []
    if "main" in section_set:
        rows.append(
            {
                "analysis_name": "seed42_reference_run",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": "",
                "n_splits": cv_splits,
                "n_repeats": cv_repeats,
                "total_folds": cv_splits * cv_repeats,
                "random_state": "42" if args.quick else "42;7;13",
                "metric_reported": "CV AUROC plus held-out AUROC/AUPRC/F1/precision/recall",
                "error_term_type": "SD",
                "output_table": "tables/main_benchmark.csv",
            }
        )
    if "ratio" in section_set:
        rows.append(
            {
                "analysis_name": "negative_ratio_sensitivity",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": "",
                "n_splits": cv_splits,
                "n_repeats": cv_repeats,
                "total_folds": cv_splits * cv_repeats,
                "random_state": "42" if args.quick else "42;7;13",
                "metric_reported": "outer-training-only CV AUROC, normalized AUPRC, F1, precision, recall, and Brier",
                "error_term_type": "SD",
                "output_table": "tables/ratio_sensitivity.csv",
            }
        )
    if "go-count" in section_set:
        rows.append(
            {
                "analysis_name": "go_feature_count_sensitivity",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": "",
                "n_splits": cv_splits,
                "n_repeats": cv_repeats,
                "total_folds": cv_splits * cv_repeats,
                "random_state": "42" if args.quick else "42;7;13",
                "metric_reported": "outer-training-only CV AUROC/AUPRC/F1/precision/recall/Brier",
                "error_term_type": "SD",
                "output_table": "tables/go_feature_count_cv_summary.csv",
            }
        )
    if "models" in section_set:
        rows.append(
            {
                "analysis_name": "supplementary_model_comparison",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": MODEL_COMPARISON_PROTOCOL,
                "n_splits": "",
                "n_repeats": "",
                "total_folds": "",
                "random_state": SEED,
                "metric_reported": "held-out AUROC/AUPRC/F1/precision/recall only",
                "error_term_type": "NA",
                "output_table": "tables/model_comparison.csv",
            }
        )
    if "ablation" in section_set:
        rows.append(
            {
                "analysis_name": "feature_ablation",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": "",
                "n_splits": "",
                "n_repeats": "",
                "total_folds": "",
                "random_state": SEED,
                "metric_reported": "held-out AUROC/AUPRC/F1/precision/recall",
                "error_term_type": "NA",
                "output_table": "tables/ablation.csv",
            }
        )
    if "importance" in section_set:
        rows.append(
            {
                "analysis_name": "feature_importance",
                "run_mode": run_mode_from_args(args),
                "model_comparison_protocol": "",
                "n_splits": "",
                "n_repeats": "",
                "total_folds": "",
                "random_state": SEED,
                "metric_reported": "SHAP/model feature attribution on held-out split",
                "error_term_type": "NA",
                "output_table": "tables/feature_importance.csv",
            }
        )
    for row in rows:
        row["primary_ratio"] = f"1:{args.primary_ratio}"
        if row["analysis_name"] in {
            "seed42_reference_run",
            "negative_ratio_sensitivity",
            "go_feature_count_sensitivity",
        }:
            row["feature_selection_protocol"] = "nested_fold_training_selected60_for_cv"
            row["nested_feature_selection"] = True
        else:
            row["feature_selection_protocol"] = "fixed_outer_training_selected60"
            row["nested_feature_selection"] = False
        row["negative_source_isolation"] = True
    return rows


def save_result_summary(
    bundle: DatasetBundle,
    context: PrimaryContext,
    sections: Sequence[str],
    spw_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Write a small JSON summary of dataset and primary outputs."""
    table_counts: Dict[str, int] = {}
    for name in [
        "main_benchmark.csv",
        "heldout_predictions.csv",
        "heldout_confusion_matrix.csv",
        "heldout_score_by_type.csv",
        "heldout_negative_type_performance.csv",
        "model_comparison.csv",
        "ratio_sensitivity.csv",
        "go_feature_count_cv_per_fold.csv",
        "go_feature_count_cv_summary.csv",
        "ablation.csv",
        "feature_importance.csv",
    ]:
        path = TABLE_DIR / name
        if path.exists():
            table_counts[name] = int(len(pd.read_csv(path)))
    pathway_member_genes = {
        gene
        for pathway in bundle.pathways.values()
        for gene in pathway["genes"]
    }
    annotated_pathway_genes = pathway_member_genes & set(bundle.gene_go)
    summary = {
        "dataset_source": bundle.source,
        "primary_ratio": f"1:{context.ratio}",
        "positive_fraction_train": float(np.mean(context.y_train == 1)),
        "positive_fraction_test": float(np.mean(context.y_test == 1)),
        "n_pathway_source_records": len(bundle.source_pathways),
        "n_source_kegg": sum(1 for r in bundle.source_pathways.values() if r["source"] == "KEGG"),
        "n_source_aracyc": sum(1 for r in bundle.source_pathways.values() if r["source"] == "AraCyc"),
        "n_modelling_groups": len(bundle.pathways),
        "n_group_kegg_stratum": sum(1 for r in bundle.pathways.values() if r["source"] == "KEGG"),
        "n_group_aracyc_stratum": sum(1 for r in bundle.pathways.values() if r["source"] == "AraCyc"),
        "n_pathways": len(bundle.pathways),
        "n_kegg": sum(1 for r in bundle.pathways.values() if r["source"] == "KEGG"),
        "n_aracyc": sum(1 for r in bundle.pathways.values() if r["source"] == "AraCyc"),
        "pathway_grouping_summary": bundle.grouping_summary,
        "n_go_annotated_genes": len(bundle.gene_go),
        "n_selected_go": len(bundle.selected_go),
        "n_features": len(bundle.feature_names),
        "n_pathway_member_genes": len(pathway_member_genes),
        "n_go_annotated_pathway_genes": len(annotated_pathway_genes),
        "pathway_gene_go_coverage": (
            len(annotated_pathway_genes) / len(pathway_member_genes)
            if pathway_member_genes
            else 0.0
        ),
        "n_train_positive": len(context.train_positive_records),
        "n_test_positive": len(context.test_positive_records),
        "n_train_total": len(context.train_records),
        "n_test_total": len(context.test_records),
        "jaccard_feature_protocol": jaccard_feature_protocol(),
        "selected_go_sha256": stable_json_sha256(context.selected_go),
        "primary_split_sha256": primary_split_sha256(context),
        "sections_run": list(sections),
        "table_row_counts": table_counts,
        "scale_pos_weight_summary": scale_pos_weight_summary(spw_rows),
    }
    save_json(OUT_DIR / "result_summary.json", summary)
    return summary


def save_manifest(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    context: PrimaryContext,
    sections: Sequence[str],
) -> None:
    """Write manifest/provenance files and audit CSVs for the current run."""
    run_mode = run_mode_from_args(args)
    raw_checksums = raw_source_checksums(Path(args.raw_dir)) if args.dataset_source == "raw" else []
    spw_rows = scale_pos_weight_audit_rows()
    hyper_rows = hyperparameter_audit_rows(args)
    cv_rows = cv_scheme_audit_rows(args, sections)
    save_table(TABLE_DIR / "hyperparameter_audit.csv", hyper_rows)
    save_table(TABLE_DIR / "cv_scheme_audit.csv", cv_rows)

    versions = software_versions()
    hardware = hardware_profile()
    save_json(OUT_DIR / "software_versions.json", versions)
    save_compute_environment(hardware)
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "working_tree_dirty": git_working_tree_dirty(),
        "working_tree_dirty_scope": (
            "source tree excluding code/generated* and the active output directory"
        ),
        "command": " ".join(["python3", project_relative_path(Path(__file__)), *sys.argv[1:]]),
        "project_root": ".",
        "output_dir": project_relative_path(OUT_DIR),
        "raw_dir": project_relative_path(Path(args.raw_dir)),
        "raw_source_checksums": raw_checksums,
        "hardware_profile": hardware,
    }
    save_json(OUT_DIR / "provenance.json", provenance)
    summary = save_result_summary(bundle, context, sections, spw_rows)

    selected_go_sha256 = stable_json_sha256(context.selected_go)
    split_sha256 = primary_split_sha256(context)
    manifest_args = dict(vars(args))
    manifest_args["raw_dir"] = project_relative_path(Path(args.raw_dir))
    manifest_args["out_dir"] = project_relative_path(Path(args.out_dir))

    is_paper_result = (
        not args.quick
        and context.ratio == DEFAULT_PRIMARY_RATIO
        and OUT_DIR.resolve() == (ROOT / "generated").resolve()
    )
    manifest = {
        "args": manifest_args,
        "run_mode": run_mode,
        "quick": bool(args.quick),
        "paper_result": is_paper_result,
        "candidate_result": (
            not bool(args.quick)
            and context.ratio == DEFAULT_PRIMARY_RATIO
            and not is_paper_result
        ),
        "formal_full_run": not bool(args.quick),
        "comparison_run": context.ratio != DEFAULT_PRIMARY_RATIO,
        "primary_ratio": f"1:{context.ratio}",
        "dataset_source": bundle.source,
        "dataset_mode": dataset_mode_from_args(args),
        "negative_scheme": NEGATIVE_SCHEME,
        "go_selection_mode": GO_SELECTION_MODE,
        "feature_selection_scope": {
            "heldout_models": "outer_train_only",
            "ratio_cv": "fold_train_only_for_variance_and_mutual_information",
            "go_feature_count_cv": "fold_train_only_for_variance_and_mutual_information",
            "frequency_prefilter": "label_free_full_go_background",
        },
        "jaccard_feature_protocol": jaccard_feature_protocol(),
        "primary_split_unit": "unique_exact_gene_set_group",
        "primary_split_stratification": "database_source",
        "negative_source_isolation": True,
        "ratio_cv_nested_feature_selection": True,
        "preselected_primary_ratio": "1:1",
        "pathway_grouping": bundle.grouping_summary,
        "selected_go_sha256": selected_go_sha256,
        "primary_split_sha256": split_sha256,
        "split_seeds": {
            "primary": context.split_seed,
            "train_negatives": context.train_negative_seed,
            "test_negatives": context.test_negative_seed,
            "cv_repeat_seeds": list(CV_REPEAT_SEEDS if not args.quick else (SEED,)),
        },
        "model_comparison_protocol": MODEL_COMPARISON_PROTOCOL,
        "scale_pos_weight_rule": SCALE_POS_WEIGHT_RULE,
        "scale_pos_weight_summary": scale_pos_weight_summary(spw_rows),
        "sections_run": list(sections),
        "software_versions": versions,
        "hardware_profile": hardware,
        "git_commit": provenance["git_commit"],
        "working_tree_dirty": provenance["working_tree_dirty"],
        "working_tree_dirty_scope": provenance["working_tree_dirty_scope"],
        "timestamp_utc": provenance["timestamp_utc"],
        "output_dir": project_relative_path(OUT_DIR),
        "raw_source_checksums": raw_checksums,
        "model_hyperparameters": {
            "XGBoost": xgboost_hyperparameters(args.quick),
            "Random Forest": random_forest_hyperparameters(args.quick),
            "Logistic Regression": logistic_regression_hyperparameters(),
            "LightGBM": {
                "available": lgb is not None,
                "parameters": lightgbm_hyperparameters(args.quick),
            },
            "CatBoost": {
                "available": cb is not None,
                "parameters": catboost_hyperparameters(args.quick),
            },
        },
        "cv_scheme_audit_table": "tables/cv_scheme_audit.csv",
        "hyperparameter_audit_table": "tables/hyperparameter_audit.csv",
        "scale_pos_weight_audit_table": "tables/scale_pos_weight_audit.csv",
        "result_summary": summary,
        "notes": [
            "Raw mode rebuilds the dataset from the documented TAIR, KEGG, and PMN/AraCyc inputs.",
            "The 60 GO terms are selected only from the primary outer-training records.",
            "Ratio comparison repeats variance and mutual-information selection inside each fold training side.",
            "Source-derived negative controls are generated independently on each split side.",
            "The supplementary 13-model comparison uses the common held-out split.",
        ],
        "paper_output_map": {
            "heldout_reference_diagnostics": (
                "tables/heldout_predictions.csv, tables/heldout_confusion_matrix.csv, "
                "tables/heldout_score_by_type.csv, tables/heldout_negative_type_performance.csv, "
                "figures/FigC_roc_pr.png, figures/FigF_confusion.png, and "
                "figures/FigG_score_by_type.png"
            ),
            "dataset_statistics": "figures/Fig5_datastats.png and tables/dataset_summary.csv",
            "kegg_pathway_filter": (
                "tables/kegg_pathway_filter_audit.csv and "
                "tables/kegg_pathway_filter_summary.json"
            ),
            "method_comparison": "tables/table_method_comparison.csv and figures/Fig8_methods.png",
            "training_only_ratio_cv": (
                "tables/ratio_cv_per_fold.csv, tables/ratio_cv_summary.csv, "
                "data/ratio_cv_manifest.json, and figures/Fig7_robustness.png"
            ),
            "go_feature_count_sensitivity": (
                "tables/go_feature_count_cv_per_fold.csv, "
                "tables/go_feature_count_cv_summary.csv, "
                "data/go_feature_count_cv_manifest.json, and "
                "figures/go_feature_count_cv_performance.png"
            ),
            "ablation": "tables/table_ablation.csv and figures/Fig_ablation.png",
            "feature_importance": "tables/table_top_features.csv and figures/Fig3_shap.png",
            "primary_split_audit": "tables/main_split_audit.csv and data/main_split.json",
            "negative_metadata": "tables/negative_metadata.csv",
            "go_selection_audit": "tables/go_selection_audit.csv",
            "cv_split_audit": "tables/cv_split_audit.csv and data/cv_fold_manifest.json",
            "pathway_grouping": (
                "data/pathway_source_records.json, data/pathway_gene_set_groups.json, "
                "and tables/pathway_grouping_summary.csv"
            ),
            "latex_fragments": "tables/latex/*.tex",
        },
    }
    save_json(OUT_DIR / "manifest.json", manifest)


def save_old_vs_candidate_comparison(bundle: DatasetBundle, context: PrimaryContext) -> None:
    """Compare the candidate run with the preserved pre-grouping result set."""
    old_dir = ROOT / "generated"
    if OUT_DIR.resolve() == old_dir.resolve() or not (old_dir / "tables" / "main_benchmark.csv").exists():
        return

    rows: List[Dict[str, Any]] = []

    def add(category: str, item: str, old: Any, candidate: Any) -> None:
        delta: Any = ""
        if isinstance(old, (int, float, np.number)) and isinstance(candidate, (int, float, np.number)):
            delta = float(candidate) - float(old)
        rows.append(
            {
                "category": category,
                "item": item,
                "previous_value": old,
                "candidate_value": candidate,
                "candidate_minus_previous": delta,
            }
        )

    add("dataset", "source database records", 538, len(bundle.source_pathways))
    add("dataset", "positive modelling instances", 538, len(bundle.pathways))
    add("dataset", "KEGG source records", 155, bundle.grouping_summary.get("source_record_kegg"))
    add("dataset", "AraCyc source records", 383, bundle.grouping_summary.get("source_record_aracyc"))
    add("dataset", "exact duplicate groups", 0, bundle.grouping_summary.get("exact_duplicate_group_count"))
    add("dataset", "collapsed extra records", 0, bundle.grouping_summary.get("collapsed_extra_record_count"))
    add("split", "outer training positives", 430, len(context.train_positive_records))
    add("split", "outer test positives", 108, len(context.test_positive_records))

    old_go_path = old_dir / "data" / "selected_go_terms.json"
    if old_go_path.exists():
        old_go = set(json.loads(old_go_path.read_text(encoding="utf-8")))
        candidate_go = set(context.selected_go)
        add("features", "selected GO overlap count", len(old_go), len(old_go & candidate_go))
        add(
            "features",
            "selected GO Jaccard",
            1.0,
            len(old_go & candidate_go) / len(old_go | candidate_go),
        )

    for table_name, key, metric_columns in [
        ("main_benchmark.csv", "model", ["cv_auroc_mean", "cv_auroc_sd", "test_auroc", "test_auprc", "f1"]),
        ("ablation.csv", "configuration", ["test_auroc", "test_auprc", "f1"]),
    ]:
        old_path = old_dir / "tables" / table_name
        new_path = TABLE_DIR / table_name
        if not old_path.exists() or not new_path.exists():
            continue
        old_df, new_df = pd.read_csv(old_path), pd.read_csv(new_path)
        for identifier in sorted(set(old_df[key]) & set(new_df[key])):
            old_row = old_df[old_df[key] == identifier].iloc[0]
            new_row = new_df[new_df[key] == identifier].iloc[0]
            for metric in metric_columns:
                if metric in old_row and metric in new_row:
                    add(table_name.removesuffix(".csv"), f"{identifier} {metric}", old_row[metric], new_row[metric])

    old_importance = old_dir / "tables" / "feature_importance.csv"
    new_importance = TABLE_DIR / "feature_importance.csv"
    if old_importance.exists() and new_importance.exists():
        old_top = set(pd.read_csv(old_importance).head(10)["feature"])
        new_top = set(pd.read_csv(new_importance).head(10)["feature"])
        renamed_old = {
            "annotated_gene_count" if feature == "pathway_size" else
            "log_annotated_gene_count" if feature == "log_size" else feature
            for feature in old_top
        }
        add("importance", "top-10 feature overlap", 10, len(renamed_old & new_top))

    old_negative = old_dir / "tables" / "negative_metadata.csv"
    new_negative = TABLE_DIR / "negative_metadata.csv"
    negative_comparison_rows: List[Dict[str, Any]] = []
    if old_negative.exists() and new_negative.exists():
        old_df, new_df = pd.read_csv(old_negative), pd.read_csv(new_negative)
        old_size_column = "n_genes" if "n_genes" in old_df else "raw_gene_count"
        for negative_type in sorted(set(old_df["negative_type"]) | set(new_df["negative_type"])):
            old_subset = old_df[old_df["negative_type"] == negative_type]
            new_subset = new_df[new_df["negative_type"] == negative_type]
            negative_comparison_rows.append(
                {
                    "negative_type": negative_type,
                    "previous_count": len(old_subset),
                    "candidate_count": len(new_subset),
                    "previous_mean_raw_gene_count": float(old_subset[old_size_column].mean()),
                    "candidate_mean_raw_gene_count": float(new_subset["raw_gene_count"].mean()),
                    "candidate_mean_annotated_gene_count": float(new_subset["annotated_gene_count"].mean()),
                    "previous_mean_source_overlap_fraction": float(old_subset["source_overlap_fraction"].mean()),
                    "candidate_mean_source_overlap_fraction": float(new_subset["source_overlap_fraction"].mean()),
                }
            )
    if negative_comparison_rows:
        save_table(TABLE_DIR / "negative_distribution_comparison.csv", negative_comparison_rows)

    add("protocol", "ratio comparison data", "outer held-out test", "outer-training nested CV")
    add("protocol", "ratio feature selection", "fixed primary feature set", "fold-training selection")
    add("protocol", "outer test ratios evaluated", "1:1 through 1:5", "preselected 1:1 only")
    save_table(TABLE_DIR / "old_vs_candidate_comparison.csv", rows)

    lines = [
        "# Previous and candidate analysis comparison",
        "",
        "The previous result directory remains unchanged. The candidate uses exact gene-set groups and training-only nested ratio cross-validation.",
        "",
        f"- Source records retained: {len(bundle.source_pathways)}",
        f"- Unique gene-set modelling instances: {len(bundle.pathways)}",
        f"- Exact duplicate groups: {bundle.grouping_summary.get('exact_duplicate_group_count', 0)}",
        f"- Collapsed extra records: {bundle.grouping_summary.get('collapsed_extra_record_count', 0)}",
        f"- Mixed-source groups: {bundle.grouping_summary.get('mixed_source_group_count', 0)}",
        "",
        "Numeric differences are listed in tables/old_vs_candidate_comparison.csv. Negative-set differences are listed in tables/negative_distribution_comparison.csv.",
        "Normalized AUPRC corrects its random baseline but is not completely invariant to class prevalence.",
        "No statistical-significance claim is made from the ratio-fold summaries.",
    ]
    (OUT_DIR / "old_vs_candidate_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checklist = [
        "Paper items to synchronize only after candidate approval:",
        "- Dataset description: 538 source records and 512 unique gene-set modelling instances.",
        "- Outer train/test sample counts and class prevalence.",
        "- Exact-duplicate grouping rule and alias provenance.",
        "- Engineered feature names: annotated_gene_count and log_annotated_gene_count.",
        "- Main benchmark, confusion matrix, score-by-negative-type, and model-comparison metrics.",
        "- Ratio protocol: outer-training nested CV; outer test evaluated only for preselected 1:1.",
        "- Ratio table and figure, including normalized-AUPRC caveat.",
        "- Ablation and feature-importance results.",
        "- Remove heuristic pathway-category labels and their held-out analysis from the manuscript.",
    ]
    (OUT_DIR / "paper_sync_checklist.txt").write_text("\n".join(checklist) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Main entry point: load data, run requested analyses, save outputs."""
    configure_output_dir(args.out_dir)

    # Running every analysis in one Python process can stress optional native
    # libraries (LightGBM/CatBoost/XGBoost) on some local installs.  For the
    # paper-wide "all" command, dispatch independent chunks to fresh processes.
    # This is slower than a single process but much more robust and clearer for
    # manual reproducibility checks.
    if "all" in args.sections and os.environ.get("PATHWAYML_CHILD") != "1":
        run_all_in_child_processes(args)
        return
    if (
        "models" in args.sections
        and len(args.sections) > 1
        and "all" not in args.sections
        and os.environ.get("PATHWAYML_CHILD") != "1"
    ):
        run_requested_sections_in_child_processes(args)
        return

    ensure_dirs()
    remove_obsolete_analysis_outputs()
    if args.dataset_source == "cached":
        print("Loading cached manuscript dataset from data_robustness/ ...", flush=True)
        bundle = load_cached_bundle()
    else:
        print(f"Rebuilding dataset from raw source files in {args.raw_dir} ...", flush=True)
        bundle = load_raw_bundle(Path(args.raw_dir))
        save_kegg_filter_audit(Path(args.raw_dir))
    print("Preparing source-isolated primary split and selecting GO terms on training records only ...", flush=True)
    context = prepare_primary_context(bundle, ratio=args.primary_ratio)
    print(
        f"Dataset: {len(bundle.pathways)} pathways, {len(bundle.gene_go)} GO-annotated genes, "
        f"{len(bundle.selected_go)} selected GO terms, {len(bundle.feature_names)} features.",
        flush=True,
    )
    save_primary_context_outputs(context, bundle.gene_go)
    save_processed_data(bundle, context)

    sections = args.sections
    if "all" in sections:
        sections = ["main", "ratio", "models", "ablation", "importance"]

    main_result = None
    if any(s in sections for s in ("main", "importance")):
        print("Running main benchmark ...", flush=True)
        main_result = run_main_benchmark(bundle, context, fast=args.quick)
        gc.collect()

    if "ratio" in sections:
        print("Running negative-ratio sensitivity ...", flush=True)
        run_ratio_sensitivity(bundle, context, fast=args.quick)
        gc.collect()
    if "go-count" in sections:
        print("Running GO feature-count sensitivity ...", flush=True)
        run_go_feature_count_sensitivity(bundle, context, fast=args.quick)
        gc.collect()
    if "models" in sections:
        print("Running model comparison ...", flush=True)
        run_model_comparison(bundle, context, fast=args.quick)
        gc.collect()
    if "ablation" in sections:
        print("Running feature ablation ...", flush=True)
        run_ablation(bundle, context, fast=args.quick)
        gc.collect()
    if "importance" in sections and main_result is not None:
        print("Running SHAP/model-importance summary ...", flush=True)
        xgb = main_result["models"]["XGBoost"]
        run_importance(bundle, context, xgb, main_result["X_test"])
        gc.collect()

    save_manifest(args, bundle, context, sections)
    print(f"Done. Outputs written to {OUT_DIR}")


def run_all_in_child_processes(args: argparse.Namespace) -> None:
    """Run all analyses in isolated child processes for stability.

    Splits the work into 3 chunks to avoid native library conflicts
    (LightGBM/CatBoost/XGBoost) when all run in one process.
    """
    configure_output_dir(args.out_dir)
    ensure_dirs()
    remove_obsolete_analysis_outputs()
    chunks = [
        ["main", "ratio", "importance"],
        ["models"],
        ["ablation"],
    ]
    env = os.environ.copy()
    env["PATHWAYML_CHILD"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    for chunk in chunks:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-source",
            args.dataset_source,
            "--raw-dir",
            args.raw_dir,
            "--out-dir",
            str(OUT_DIR),
            "--primary-ratio",
            str(args.primary_ratio),
            "--sections",
            *chunk,
        ]
        if args.quick:
            cmd.append("--quick")
        print(f"\n=== Running chunk: {' '.join(chunk)} ===", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    # The child processes each write their own manifest.  Replace that with a
    # single top-level manifest that documents the complete paper run.
    bundle = load_cached_bundle() if args.dataset_source == "cached" else load_raw_bundle(Path(args.raw_dir))
    if args.dataset_source == "raw":
        save_kegg_filter_audit(Path(args.raw_dir))
    context = prepare_primary_context(bundle, ratio=args.primary_ratio)
    save_primary_context_outputs(context, bundle.gene_go)
    save_processed_data(bundle, context)
    save_manifest(
        args,
        bundle,
        context,
        ["main", "ratio", "models", "ablation", "importance"],
    )
    save_old_vs_candidate_comparison(bundle, context)
    print(f"\nDone. Outputs written to {OUT_DIR}")


def run_requested_sections_in_child_processes(args: argparse.Namespace) -> None:
    """Run requested multi-section jobs in isolated processes when models are included.

    The supplementary 13-model block imports and exercises several optional
    native libraries.  Running it after XGBoost/RF/CV work in the same process
    can be unstable on some local Python builds, so mixed-section commands such
    as ``--sections main models`` use the same child-process isolation as the
    full paper run.
    """
    configure_output_dir(args.out_dir)
    ensure_dirs()
    remove_obsolete_analysis_outputs()
    sections = list(args.sections)
    chunks = []
    non_model = [s for s in sections if s != "models"]
    if non_model:
        chunks.append(non_model)
    chunks.append(["models"])

    env = os.environ.copy()
    env["PATHWAYML_CHILD"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    for chunk in chunks:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-source",
            args.dataset_source,
            "--raw-dir",
            args.raw_dir,
            "--out-dir",
            str(OUT_DIR),
            "--primary-ratio",
            str(args.primary_ratio),
            "--sections",
            *chunk,
        ]
        if args.quick:
            cmd.append("--quick")
        print(f"\n=== Running chunk: {' '.join(chunk)} ===", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    bundle = load_cached_bundle() if args.dataset_source == "cached" else load_raw_bundle(Path(args.raw_dir))
    if args.dataset_source == "raw":
        save_kegg_filter_audit(Path(args.raw_dir))
    context = prepare_primary_context(bundle, ratio=args.primary_ratio)
    save_primary_context_outputs(context, bundle.gene_go)
    save_processed_data(bundle, context)
    save_manifest(args, bundle, context, sections)
    print(f"\nDone. Outputs written to {OUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        choices=["cached", "raw"],
        default="raw",
        help="raw rebuilds from raw_sources/; cached uses archived JSON data for comparison only.",
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw KEGG/AraCyc/TAIR files.")
    parser.add_argument(
        "--primary-ratio",
        type=int,
        choices=SUPPORTED_PRIMARY_RATIOS,
        default=DEFAULT_PRIMARY_RATIO,
        help=(
            "Primary positive:negative ratio. The current paper branch uses 1:1; "
            "all other ratios are written to isolated comparison directories by default."
        ),
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        default=None,
        help=(
            "Directory for generated tables, figures, data, and manifest. "
            "Defaults to generated/ for full runs and generated_quick/ for --quick."
        ),
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["all"],
        choices=["all", "main", "ratio", "go-count", "models", "ablation", "importance"],
        help="Analyses to run.",
    )
    parser.add_argument("--quick", action="store_true", help="Use faster models/CV for fast validation runs.")
    parser.add_argument("--full", action="store_true", help="Alias for not --quick; kept for readability.")
    args = parser.parse_args()
    if args.full:
        args.quick = False
    if "go-count" in args.sections and ("all" in args.sections or len(args.sections) != 1):
        parser.error("Run --sections go-count separately so its outputs remain supplementary.")
    if "go-count" in args.sections:
        suffix = "_quick" if args.quick else ""
        default_out = ROOT / f"generated_go_feature_count_sensitivity{suffix}"
    elif args.primary_ratio == DEFAULT_PRIMARY_RATIO:
        default_out = Path("/tmp/pathwayml_grouped_cv_quick") if args.quick else ROOT / "generated"
    else:
        name = f"generated_grouped_cv_ratio_1_{args.primary_ratio}" + ("_quick" if args.quick else "")
        default_out = ROOT / name
    if args.out_dir is None:
        args.out_dir = str(default_out)
    else:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        if out_dir.resolve() == (ROOT / "generated").resolve() and (
            args.quick
            or args.primary_ratio != DEFAULT_PRIMARY_RATIO
            or "go-count" in args.sections
        ):
            parser.error(
                "Only a full 1:1 paper run may write to generated/. Omit --out-dir "
                "to use the isolated quick, ratio, or supplementary directory."
            )
        args.out_dir = str(out_dir)
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
