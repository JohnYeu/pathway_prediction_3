#!/usr/bin/env python3
"""Regression tests for split isolation and deterministic feature construction."""

from __future__ import annotations

import hashlib
import json
import random
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
        source_union = {gene for info in self.bundle.source_pathways.values() for gene in info["genes"]}
        group_union = {gene for info in self.bundle.pathways.values() for gene in info["genes"]}
        self.assertEqual(source_union, group_union)

    def test_no_inferred_pathway_category_metadata(self) -> None:
        retired_keys = {"family", "source_families", "mixed_family", "cross_same_family"}
        for info in self.bundle.source_pathways.values():
            self.assertFalse(retired_keys & set(info))
        for info in self.bundle.pathways.values():
            self.assertFalse(retired_keys & set(info))
        for record in self.context.train_records + self.context.test_records:
            self.assertFalse(retired_keys & set(record))

    def test_training_selection_has_fixed_dimension(self) -> None:
        self.assertEqual(pipeline.PRIMARY_MODEL_NAME, "Random Forest")
        self.assertEqual(len(self.context.selected_go), pipeline.N_GO_TERMS)
        self.assertEqual(len(self.context.feature_names), pipeline.N_TOTAL_FEATURES)
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

    def test_go_feature_count_candidates_share_one_fold_ranking(self) -> None:
        self.assertEqual(pipeline.GO_FEATURE_COUNT_CANDIDATES, (20, 40, 60, 80, 100))
        self.assertEqual(pipeline.N_GO_TERMS, 80)
        self.assertEqual(pipeline.N_TOTAL_FEATURES, 89)
        folds = pipeline.build_cv_fold_datasets(
            self.bundle,
            self.context,
            ratio=1,
            fast=True,
            analysis_name="test_go_feature_count_cv",
        )
        ranked_go, stages = pipeline.ranked_go_terms_for_fold(
            self.bundle,
            folds[0],
            max(pipeline.GO_FEATURE_COUNT_CANDIDATES),
        )
        self.assertEqual(len(ranked_go), 100)
        self.assertEqual(ranked_go[: pipeline.N_GO_TERMS], folds[0].selected_go)
        self.assertEqual(stages["mi_select"], 100)
        for count in pipeline.GO_FEATURE_COUNT_CANDIDATES:
            self.assertEqual(
                len(pipeline.feature_names_for(ranked_go[:count])),
                count + pipeline.N_ENGINEERED_FEATURES,
            )

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

    def test_jaccard_sampling_is_order_independent_and_local(self) -> None:
        genes = sorted(self.bundle.gene_go)[:40]
        selected = pipeline.select_genes_for_jaccard(genes)
        self.assertEqual(len(selected), pipeline.JACCARD_SAMPLE_SIZE)
        self.assertEqual(selected, pipeline.select_genes_for_jaccard(list(reversed(genes))))
        self.assertTrue(set(selected).issubset(genes))

        # Sampling uses a local Random instance and must not perturb negative
        # generation or any other caller that relies on Python's global RNG.
        random.seed(20260718)
        state_before = random.getstate()
        pipeline.select_genes_for_jaccard(genes)
        self.assertEqual(state_before, random.getstate())

    def test_jaccard_uses_all_genes_up_to_cap(self) -> None:
        genes = sorted(self.bundle.gene_go)[: pipeline.JACCARD_EXACT_MAX_GENES]
        self.assertEqual(pipeline.select_genes_for_jaccard(genes), genes)

    def test_large_gene_set_features_ignore_input_order(self) -> None:
        record = next(
            record
            for record in self.context.train_positive_records
            if sum(gene in self.bundle.gene_go for gene in record["genes"])
            > pipeline.JACCARD_EXACT_MAX_GENES
        )
        forward = pipeline.build_feature_vector(
            record["genes"], self.context.selected_go, self.bundle.gene_go
        )
        reverse = pipeline.build_feature_vector(
            list(reversed(record["genes"])), self.context.selected_go, self.bundle.gene_go
        )
        self.assertTrue(np.array_equal(forward, reverse))

    def test_random_forest_fold_predictions_are_deterministic(self) -> None:
        fold = pipeline.build_cv_fold_datasets(
            self.bundle,
            self.context,
            ratio=1,
            fast=True,
            analysis_name="test_rf_determinism",
        )[0]
        predictions = []
        for _ in range(2):
            model = pipeline.primary_model(fast=True, random_state=fold.model_seed)
            model.fit(fold.X_train, fold.y_train)
            predictions.append(pipeline.predict_scores(model, fold.X_validation))
        self.assertTrue(np.array_equal(predictions[0], predictions[1]))

    def test_model_comparison_catalog_includes_primary_model(self) -> None:
        factories = pipeline.model_factory_catalog(fast=True)
        self.assertIn(pipeline.PRIMARY_MODEL_NAME, factories)
        expected_count = 11 + int(pipeline.lgb is not None) + int(pipeline.cb is not None)
        self.assertEqual(len(factories), expected_count)
        model = factories[pipeline.PRIMARY_MODEL_NAME](1.0, pipeline.SEED)
        self.assertEqual(model.n_estimators, 120)
        self.assertEqual(model.max_depth, 10)
        self.assertEqual(model.class_weight, "balanced")

    def test_positive_class_shap_shape_normalization(self) -> None:
        class_zero = np.zeros((4, 3), dtype=float)
        class_one = np.arange(12, dtype=float).reshape(4, 3)
        from_list = pipeline.positive_class_shap_values([class_zero, class_one], 3)
        from_last_axis = pipeline.positive_class_shap_values(
            np.stack([class_zero, class_one], axis=-1), 3
        )
        self.assertTrue(np.array_equal(from_list, class_one))
        self.assertTrue(np.array_equal(from_last_axis, class_one))

    def test_heldout_diagnostics_use_supplied_scores(self) -> None:
        records = [
            {"id": "P1", "label": 1, "genes": ["G1"]},
            {"id": "P2", "label": 1, "genes": ["G2"]},
            {"id": "N1", "label": 0, "negative_type": "random_5_30", "genes": ["G3"]},
            {"id": "N2", "label": 0, "negative_type": "shuffled", "genes": ["G4"]},
            {"id": "N3", "label": 0, "negative_type": "partial_50_80", "genes": ["G5"]},
            {"id": "N4", "label": 0, "negative_type": "cross_pathway", "genes": ["G6"]},
        ]
        labels = np.array([1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.4, 0.1, 0.2, 0.7, 0.6])
        tables = pipeline.heldout_diagnostic_tables(records, labels, scores)

        confusion = tables["confusion"][0]
        self.assertEqual(confusion["true_positive"], 1)
        self.assertEqual(confusion["false_negative"], 1)
        self.assertEqual(confusion["true_negative"], 2)
        self.assertEqual(confusion["false_positive"], 2)
        type_rows = {row["gene_set_type"]: row for row in tables["score_by_type"]}
        self.assertEqual(type_rows["random_5_30"]["false_positive_count_at_0_5"], 0)
        self.assertEqual(type_rows["partial_50_80"]["false_positive_count_at_0_5"], 1)
        self.assertEqual(len(tables["negative_type_performance"]), 4)


if __name__ == "__main__":
    unittest.main()
