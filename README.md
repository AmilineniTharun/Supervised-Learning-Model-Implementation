# Supervised-Learning-Model-Implementation

## Project Overview

This project implements supervised machine learning models to classify breast cancer observations as malignant or benign using the Breast Cancer Wisconsin dataset available through Scikit-learn.

The objective is to demonstrate the complete machine learning workflow, including data inspection, preprocessing, model training, cross-validation, hyperparameter tuning, performance evaluation, visualization, and interpretation of results.


## Objectives

- Define a binary classification problem using a public dataset.
- Inspect the dataset and understand its features and target classes.
- Prepare the data using a stratified training and testing split.
- Build and compare Logistic Regression and Random Forest classifiers.
- Apply five-fold stratified cross-validation.
- Tune Logistic Regression hyperparameters using GridSearchCV.
- Evaluate performance using multiple classification metrics.
- Analyze prediction errors, visualizations, and feature importance.
- Document the strengths, limitations, and possible improvements.

## Dataset

The project uses the Breast Cancer Wisconsin dataset bundled with Scikit-learn.

- **Observations:** 569
- **Numerical features:** 30
- **Target classes:** Malignant and benign
- **Target encoding:** `0 = malignant`, `1 = benign`

The dataset is loaded directly using `sklearn.datasets.load_breast_cancer(as_frame=True)`, so a separate dataset download is not required.

## Models Implemented

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model. Standardization is applied because feature scales can affect its optimization. GridSearchCV evaluates different regularization strengths using cross-validation ROC-AUC.

### 2. Random Forest

Random Forest is included as a tree-based ensemble model that can learn nonlinear relationships and feature interactions. Feature standardization is not required for this model.

## Validation and Evaluation

The project uses an 80/20 stratified train/test split and five-fold stratified cross-validation on the training data. Hyperparameter tuning is performed without using the held-out test set.

Evaluation metrics include:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matthews correlation coefficient

The project also includes a confusion matrix, classification report, ROC curves, and permutation feature importance.

## Project Structure

```text
Week4_Strengthened_Project/
├── Week4_Supervised_Learning.ipynb
├── Week4_Supervised_Learning_Report.docx
├── Submission_Description_200_words.txt
└── outputs/
    ├── breast_cancer_dataset.csv
    ├── cross_validation_metrics.csv
    ├── held_out_test_metrics.csv
    ├── logistic_classification_report.csv
    ├── logistic_confusion_matrix.csv
    ├── test_set_predictions.csv
    ├── logistic_grid_search_results.csv
    ├── permutation_feature_importance.csv
    ├── experiment_summary.json
    └── figures/
        ├── 01_class_distribution.png
        ├── 02_cross_validation_comparison.png
        ├── 03_logistic_confusion_matrix.png
        ├── 04_roc_curves.png
        ├── 05_permutation_feature_importance.png
        └── 06_selected_feature_distributions.png
```

The `outputs/` directory is populated when the notebook is run from the project root.

## Requirements

Install the required Python packages:

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
```

## How to Run

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required packages.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Week4_Supervised_Learning.ipynb`.
6. Run all cells from top to bottom.
7. Check the generated files in the `outputs/` and `outputs/figures/` directories.

The notebook uses a fixed random seed to support reproducibility. Minor differences may occur with different software versions.

## Results

In the reported held-out test split, tuned Logistic Regression achieved approximately **98.25% accuracy** and **0.9954 ROC-AUC**. Random Forest achieved approximately **94.74% accuracy**.

These results apply to this experiment and split. They do not establish that the models will perform equally well on other datasets or real-world clinical data.

## Limitations and Future Improvements

- Validate the models using an independent external dataset.
- Investigate probability calibration and decision thresholds.
- Compare additional algorithms under the same validation strategy.
- Explore feature selection within cross-validation.
- Assess generalization and subgroup performance where appropriate.
- Conduct further domain-specific review before any real-world application.

## Conclusion

This project demonstrates a reproducible supervised learning workflow, from dataset inspection and preprocessing to model validation, tuning, evaluation, and interpretation. It compares two classification algorithms and documents their results, limitations, and potential improvements.

**Disclaimer:** The models are intended solely for educational use and must not be used for clinical diagnosis or medical decision-making.
