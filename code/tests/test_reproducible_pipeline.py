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
        cls.context = pipeline.prepare_primary_context(
            cls.bundle,
            ratio=pipeline.DEFAULT_PRIMARY_RATIO,
        )

    def test_primary_split_and_source_isolation(self) -> None:
        self.assertEqual(len(self.context.train_positive_records), 409)
        self.assertEqual(len(self.context.test_positive_records), 103)
        self.assertEqual(self.context.ratio, 1)
        self.assertEqual(len(self.context.train_records), 818)
        self.assertEqual(len(self.context.test_records), 206)
        pipeline.assert_source_isolation(
            self.context.train_positive_records,
            self.context.test_positive_records,
            self.context.train_records,
            self.context.test_records,
        )
        train_hashes = {record["gene_set_sha256"] for record in self.context.train_positive_records}
        test_hashes = {record["gene_set_sha256"] for record in self.context.test_positive_records}
        self.assertFalse(train_hashes & test_hashes)

    def test_exact_gene_set_grouping(self) -> None:
        summary = self.bundle.grouping_summary
        self.assertEqual(summary["source_record_count"], 538)
        self.assertEqual(summary["source_record_kegg"], 155)
        self.assertEqual(summary["source_record_aracyc"], 383)
        self.assertEqual(summary["modelling_group_count"], 512)
        self.assertEqual(summary["modelling_group_kegg_stratum"], 155)
        self.assertEqual(summary["modelling_group_aracyc_stratum"], 357)
        self.assertEqual(summary["exact_duplicate_group_count"], 20)
        self.assertEqual(summary["collapsed_extra_record_count"], 26)
        self.assertEqual(summary["mixed_source_group_count"], 1)
        self.assertEqual(summary["mixed_family_group_count"], 4)
        source_union = {gene for info in self.bundle.source_pathways.values() for gene in info["genes"]}
        group_union = {gene for info in self.bundle.pathways.values() for gene in info["genes"]}
        self.assertEqual(source_union, group_union)

    def test_training_selection_has_fixed_dimension(self) -> None:
        self.assertEqual(len(self.context.selected_go), pipeline.N_GO_TERMS)
        self.assertEqual(len(self.context.feature_names), pipeline.N_GO_TERMS + 9)
        self.assertEqual(self.context.feature_selection_stages["mi_select"], pipeline.N_GO_TERMS)

    def test_negative_generator_boundaries(self) -> None:
        samples = pipeline.generate_negative_samples(
            self.bundle.pathways,
            sorted(self.bundle.gene_go),
            160,
            seed=142,
        )
        self.assertTrue(all(sample.annotated_gene_count >= pipeline.MIN_PATHWAY_GENES for sample in samples))
        for sample in samples:
            if sample.negative_type == "cross_pathway":
                self.assertEqual(len(sample.source_pathway_ids), 2)
                self.assertEqual(len(set(sample.source_pathway_ids)), 2)
            if sample.negative_type == "partial_50_80":
                self.assertIsNotNone(sample.retention_fraction)
                self.assertGreaterEqual(sample.retention_fraction, 0.5)
                self.assertLessEqual(sample.retention_fraction, 0.8)

    def test_nested_fold_selection_uses_fold_training_records(self) -> None:
        folds = pipeline.build_cv_fold_datasets(
            self.bundle,
            self.context,
            ratio=1,
            fast=True,
            analysis_name="test_nested_cv",
        )
        self.assertEqual(len(folds), 3)
        outer_test_ids = {record["id"] for record in self.context.test_positive_records}
        for fold in folds:
            self.assertFalse((set(fold.train_positive_ids) | set(fold.validation_positive_ids)) & outer_test_ids)
            selected, _, _ = pipeline.select_go_terms_from_records(
                fold.train_records,
                self.bundle.gene_go,
                self.bundle.go_genes,
                seed=fold.feature_selection_seed,
            )
            self.assertEqual(selected, fold.selected_go)
            self.assertEqual(pipeline.stable_json_sha256(selected), fold.selected_go_sha256)

    def test_deterministic_seed_derivation(self) -> None:
        cases = {
            (42, 1, 42, 0, "train", "negative_generation"): 1973658644,
            (42, 1, 42, 0, "validation", "negative_generation"): 3715151037,
            (42, 5, 13, 4, "train", "go_selection"): 525790726,
        }
        for values, expected in cases.items():
            self.assertEqual(pipeline.derive_deterministic_seed(*values), expected)
        forward = [pipeline.derive_deterministic_seed(*values) for values in cases]
        reverse = [pipeline.derive_deterministic_seed(*values) for values in reversed(cases)]
        self.assertEqual(forward, list(reversed(reverse)))

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
        # The fixture locks the corrected generator, including exact grouping,
        # integer partial retention, source-distinct cross controls, and retries.
        expected = {
            42: "21360319f1389052daa65a7860f490007ad8df8165622a27ad83ccd030cfa45c",
            142: "16445546c1fb5bc7fff57c1d4ed0495e7a53b8b4f6a9eed58a05bc7a01c372ca",
            242: "3bb07189b7ae09b08d9e42d8f83515f8cd9fe6030e5ecb0e7d3f200b4c6ce09d",
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

    def test_feature_names_use_annotated_gene_count(self) -> None:
        self.assertIn("annotated_gene_count", self.context.feature_names)
        self.assertIn("log_annotated_gene_count", self.context.feature_names)
        self.assertNotIn("pathway_size", self.context.feature_names)
        self.assertNotIn("log_size", self.context.feature_names)

    def test_xgboost_fold_predictions_are_deterministic(self) -> None:
        fold = pipeline.build_cv_fold_datasets(
            self.bundle,
            self.context,
            ratio=1,
            fast=True,
            analysis_name="test_xgb_determinism",
        )[0]
        predictions = []
        for _ in range(2):
            model = pipeline.xgb_model(
                scale_pos_weight=fold.scale_pos_weight,
                fast=True,
                random_state=fold.model_seed,
            )
            model.fit(fold.X_train, fold.y_train)
            predictions.append(pipeline.predict_scores(model, fold.X_validation))
        self.assertTrue(np.array_equal(predictions[0], predictions[1]))


if __name__ == "__main__":
    unittest.main()
