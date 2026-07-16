#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute the pathway-gene GO-annotation coverage for the current data build.

Coverage = of the unique genes that appear in the 538 curated pathways
(KEGG + AraCyc), what fraction carry at least one GO annotation in the TAIR
ATH_GO_GOSLIM file (i.e. can be turned into GO-frequency features).

This is the "97.97%" figure quoted in the thesis data-description section.
It reuses the main pipeline's parsers so the number is fully reproducible;
the result is also written to generated/data/pathway_gene_coverage.json.

Run:  python3 code/compute_pathway_gene_coverage.py
"""
import json
from pathlib import Path

import reproducible_pipeline as P

OUT = Path(__file__).resolve().parent / "generated" / "data" / "pathway_gene_coverage.json"


def main() -> None:
    bundle = P.load_raw_bundle(P.DEFAULT_RAW_DIR)

    # Unique genes across all curated pathways (KEGG + AraCyc).
    pathway_genes = set()
    for info in bundle.pathways.values():
        pathway_genes |= set(info["genes"])

    annotated = set(bundle.gene_go)          # genes with >=1 GO term in ATH_GO_GOSLIM
    covered = pathway_genes & annotated

    n_total = len(pathway_genes)
    n_covered = len(covered)
    coverage = n_covered / n_total if n_total else 0.0

    result = {
        "pathway_member_genes": n_total,
        "go_annotated_pathway_genes": n_covered,
        "coverage_fraction": round(coverage, 6),
        "coverage_percent": round(100 * coverage, 2),
        "definition": (
            "fraction of unique genes in the 538 curated pathways that carry at "
            "least one GO annotation in ATH_GO_GOSLIM"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"pathway-gene coverage = {n_covered}/{n_total} = {100 * coverage:.2f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
