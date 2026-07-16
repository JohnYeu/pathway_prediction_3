#!/usr/bin/env python3
"""Regression tests for split isolation and deterministic feature construction."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import reproducible_pipeline as pipeline  # noqa: E402


class ReproduciblePipelineTests(unittest.TestCase):
    """Load the raw inputs once and reuse the deterministic primary context."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = pipeline.load_raw_bundle(pipeline.DEFAULT_RAW_DIR)
        cls.context = pipeline.prepare_primary_context(cls.bundle, ratio=2)

    def test_primary_split_and_source_isolation(self) -> None:
        self.assertEqual(len(self.context.train_positive_records), 430)
        self.assertEqual(len(self.context.test_positive_records), 108)
        self.assertEqual(len(self.context.train_records), 1290)
        self.assertEqual(len(self.context.test_records), 324)
        pipeline.assert_source_isolation(
            self.context.train_positive_records,
            self.context.test_positive_records,
            self.context.train_records,
            self.context.test_records,
        )

    def test_training_selection_has_fixed_dimension(self) -> None:
        self.assertEqual(len(self.context.selected_go), pipeline.N_GO_TERMS)
        self.assertEqual(len(self.context.feature_names), pipeline.N_GO_TERMS + 9)
        self.assertEqual(self.context.feature_selection_stages["mi_select"], pipeline.N_GO_TERMS)

    def test_entropy_zero_boundary_and_nonzero_compatibility(self) -> None:
        # An annotated gene with none of the selected terms exercises the new
        # all-zero profile convention without taking the unannotated fast path.
        gene = next(
            gene
            for gene, terms in self.bundle.gene_go.items()
            if not (set(terms) & set(self.context.selected_go))
        )
        zero_profile = pipeline.build_feature_vector(
            [gene], self.context.selected_go, self.bundle.gene_go
        )
        self.assertEqual(float(zero_profile[-3]), 0.0)

        record = next(record for record in self.context.train_positive_records if record["genes"])
        valid = [gene for gene in record["genes"] if gene in self.bundle.gene_go]
        frequency = np.array(
            [
                sum(term in self.bundle.gene_go[gene] for gene in valid) / len(valid)
                for term in self.context.selected_go
            ],
            dtype=float,
        )
        smoothed = frequency + 1e-9
        smoothed /= smoothed.sum()
        previous_formula = np.float32(-np.sum(smoothed * np.log(smoothed + 1e-12)))
        current = pipeline.build_feature_vector(
            record["genes"], self.context.selected_go, self.bundle.gene_go
        )
        self.assertEqual(float(current[-3]), float(previous_formula))

    def test_negative_generator_rng_regression(self) -> None:
        # Adding source metadata must not consume random numbers or alter the
        # established fixed-seed control gene sets.
        expected = {
            42: "dbc1e16dde594002ed82e18f22e75e1c2396735d29520975938a93c94be57d54",
            142: "89ebac91371622e2b47040eafcc8e977f1ef07d095b8c6d6df1532e1aa334c99",
            242: "621733c38d87529709f503955b4743b00875d6a6774050743d04aed24d5dfb37",
        }
        for seed, expected_hash in expected.items():
            gene_sets = pipeline.generate_negative_gene_sets(
                self.bundle.pathways,
                sorted(self.bundle.gene_go),
                len(self.bundle.pathways) * 2,
                seed,
            )
            payload = json.dumps(gene_sets, separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main()
