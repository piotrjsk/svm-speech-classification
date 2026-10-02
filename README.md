# English Alphabet Speech Signal Recognition (SVM Classification)

![R](https://img.shields.io/badge/Language-R-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Tidymodels-orange.svg)
![Model](https://img.shields.io/badge/Model-SVM%20(RBF)-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96.96%25-brightgreen.svg)

A comprehensive machine learning and digital signal processing project focused on automatically classifying **26 letters of the English alphabet** using numerical acoustic feature parameters.

---

## Performance Overview

The final model, based on **Support Vector Machines (SVM) with a Radial Basis Function (RBF) kernel**, achieved exceptional classification performance on the unseen test set:

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **96.96%** | Overall proportion of correctly classified speech signals |
| **Kappa (Cohen's)** | **0.9683** | High agreement metric adjusted for random chance |
| **Support Vectors** | **3551** | Key boundary data points defining decision hyperplanes |
| **Optimal Cost ($C$)** | `~31.91` | Determined via Bayesian optimization |
| **Optimal Sigma ($\sigma$)** | `~0.00119` | Wide kernel reach providing smooth decision boundaries |

---

## Architecture & Methodology

### 1. Data Splitting & Validation (`tidymodels`)
* **Input Features:** 618 numerical acoustic features representing speech signals.
* **Data Split:** 80% training set and 20% test set using class stratification (`strata = klasa`) to maintain identical letter distributions across splits[cite: 31].
* **Validation Strategy:** 5-fold cross-validation (`vfold_cv`) on the training set to prevent overfitting during hyperparameter selection[cite: 31].

### 2. Feature Engineering Pipeline (`recipes`)
* **`step_nzv()`**: Filters out Near-Zero Variance features to remove uninformative background noise.
* **`step_YeoJohnson()`**: Stabilizes variance and normalizes skewed distributions inherent to acoustic measurements.
* **`step_normalize()`**: Standardizes predictor scales (critical for Euclidean distance-based RBF kernel calculations).

### 3. Hyperparameter Tuning (`tune_bayes`)
* Replaced traditional grid search with **Bayesian Optimization**, probabilistically evaluating $C$ and $\sigma$ parameter spaces to efficiently discover global accuracy maxima.

---

## Model Comparison & Benchmark

Multiple algorithms were benchmarked during the experimental phase:
1. **SVM with RBF Kernel (Selected):** **96.96% Accuracy** – Best handled high dimensionality and non-linear decision boundaries.
2. **Logistic Regression (Elastic Net):** ~95.7% Accuracy.
3. **Random Forest:** ~94.9% Accuracy – Consistently lagged behind SVM by 1–3 percentage points.
4. **Dimensionality Reduction (PCA + SVM):** ~95.5% Accuracy – Rejected because unsupervised PCA discarded subtle discriminatory variance required to separate phonetically close letters (e.g., "B" vs "P").

---

## Repository Structure

```text
.
├── .gitignore              # Git exclusion rules
├── .Rprofile               # Automatic renv environment activation
├── renv.lock               # Reproducible package environment lockfile
│
├── data/
│   └── dane.csv            # Raw acoustic dataset
│
├── models/
│   └── model_svm.rds       # Trained SVM model artifact
│
├── reports/
│   ├── svm_speech_classification_report.Rmd   # RMarkdown report source
│   └── svm_speech_classification_report.pdf   # Compiled PDF report
│
├──renv/                   # Isolated virtual environment configuration
└── README.md
```

## Key Technical Skills Demonstrated
* **Data Processing & Feature Engineering:** Stratified data splitting, Near-Zero Variance filtering, Yeo-Johnson transformation, feature normalization.
* **Data Science & ML:** Support Vector Machines (SVM RBF), Bayesian hyperparameter optimization, cross-validation, model benchmarking & confusion matrix evaluation.
* **R Programming & Tidymodels:** Modular ML workflows (recipes, parsnip, workflows), reproducible environment management (renv), automated report rendering (RMarkdown).

## Installation
```bash
git clone https://github.com/piotrjsk/svm-speech-classification.git
cd svm-speech-classification

# Restore reproducible package dependencies
Rscript -e "renv::restore()"

# Render the executive report to HTML
Rscript -e "rmarkdown::render('reports/svm_speech_classification_report.Rmd', output_dir = 'reports')"
```

---
*Developed by Piotr Jasiak & Władysław Morawski | [LinkedIn Profile](https://www.linkedin.com/in/piotrjasiak)*