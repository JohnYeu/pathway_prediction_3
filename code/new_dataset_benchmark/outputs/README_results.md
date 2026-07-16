# New Dataset 13-Model Benchmark

This is an isolated benchmark on the new fixed curated 1:2 dataset. It does not replace the main PathwayML-Ath data, generated tables, or paper outputs.

## Dataset

- Input copy: `new_dataset_benchmark/outputs/input_training_dataset_ratio_1_2_curated.csv`
- SHA256: `b540c1f70aa6c98b6eae30e440092d276170661b136c2a87fba3146d3c06c8a4`
- Samples: 1608
- Positives: 536 ({'AraCyc': 383, 'KEGG': 153})
- Negatives: 1072
- Feature source: PathwayML-Ath/data_robustness cached GO feature set
- Feature matrix: 69 features = 60 selected GO terms + 9 engineered descriptors
- Split: seed 42, train 1286, test 322

## Output Files

- `model_metrics.csv`: final 13-model performance table.
- `model_test_predictions.csv`: held-out test scores for every model and sample.
- `feature_matrix.csv` and `feature_matrix.npy`: saved 69-dimensional feature matrix.
- `sample_metadata_with_go_coverage.csv`: parsed gene sets and GO annotation coverage.
- `train_test_split.csv`, `train_indices.npy`, `test_indices.npy`: deterministic split.
- `models/*.joblib`: trained model snapshots.
- `manifest.json`, `dataset_summary.json`: reproducibility metadata.

## Results Ranked By Test AUROC

| model | test_auroc | test_auprc | test_f1 | test_precision | test_recall | cv_auroc_mean |
| --- | --- | --- | --- | --- | --- | --- |
| AdaBoost | 0.840 | 0.663 | 0.652 | 0.610 | 0.701 | 0.831 |
| XGBoost | 0.834 | 0.660 | 0.636 | 0.576 | 0.710 | 0.836 |
| LightGBM | 0.834 | 0.652 | 0.656 | 0.584 | 0.748 |  |
| Gradient Boosting | 0.825 | 0.654 | 0.607 | 0.615 | 0.598 | 0.822 |
| CatBoost | 0.820 | 0.640 | 0.667 | 0.565 | 0.813 |  |
| Linear SVM | 0.819 | 0.609 | 0.462 | 0.606 | 0.374 | 0.792 |
| Logistic Regression | 0.816 | 0.596 | 0.580 | 0.600 | 0.561 | 0.799 |
| Random Forest | 0.811 | 0.631 | 0.565 | 0.560 | 0.570 | 0.817 |
| RBF SVM | 0.790 | 0.599 | 0.528 | 0.593 | 0.477 | 0.777 |
| MLP | 0.789 | 0.598 | 0.553 | 0.576 | 0.533 |  |
| Extra Trees | 0.787 | 0.585 | 0.520 | 0.546 | 0.495 | 0.788 |
| Gaussian Naive Bayes | 0.725 | 0.489 | 0.625 | 0.512 | 0.804 | 0.708 |
| k-NN (k=7) | 0.709 | 0.463 | 0.423 | 0.471 | 0.383 | 0.722 |
