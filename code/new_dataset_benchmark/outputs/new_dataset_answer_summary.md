Using the provided `training_dataset_ratio_1_2_curated.csv` as a fixed benchmark dataset, I extracted the same 69-dimensional PathwayML-Ath feature representation and evaluated 13 machine-learning models with an 80/20 stratified seed-42 split. The dataset contains 536 positive curated pathways and 1072 synthetic negatives.

Top held-out results by AUROC:

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

The strongest held-out AUROC was AdaBoost (0.840), followed by XGBoost and LightGBM (both 0.834). XGBoost achieved AUPRC 0.660. Full outputs and intermediate files are saved under `new_dataset_benchmark/outputs`.
