# UC-FCM Reproduction Project

## Unconstrained Fuzzy C-Means Clustering (TPAMI 2025)

### Project Overview

This project presents a Python implementation and experimental reproduction of the Unconstrained Fuzzy C-Means (UC-FCM) clustering algorithm proposed in the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI, 2025). The implementation compares UC-FCM with the classical Fuzzy C-Means (FCM) algorithm using multiple benchmark datasets and reproduces the major experimental evaluations presented in the paper.

The project includes implementations of both algorithms, evaluation metrics, visualization utilities, statistical analysis, and automated experiment execution.

---

## Project Structure

```
Project/
│
├── algorithms.py          # FCM and UC-FCM implementations
├── datasets.py            # Dataset loading and preprocessing
├── evaluation.py          # Clustering evaluation metrics and statistical tests
├── visualization.py       # Generation of figures and plots
├── run_experiments.py     # Main experiment pipeline
├── requirements.txt       # Python dependencies
│
├── data/                  # Benchmark datasets
├── figures/               # Generated plots
├── results/               # Experimental results
└── tables/                # Generated tables (CSV and LaTeX)
```

---

## Features

• Classical Fuzzy C-Means (FCM)

• Unconstrained Fuzzy C-Means (UC-FCM)

• Automatic parameter selection using grid search

• Multiple benchmark datasets

• Objective function comparison

• Clustering evaluation using:

* Accuracy (ACC)
* Normalized Mutual Information (NMI)
* Adjusted Rand Index (ARI)

• Statistical analysis using:

* Friedman Test
* Nemenyi Critical Difference
* Wilcoxon Signed-Rank Test

• Automatic visualization of:

* Convergence curves
* Parameter sensitivity analysis
* Runtime comparison
* PCA visualizations
* Confusion matrices
* Box plots

---

## Requirements

Python 3.10 or later

Required packages:

* numpy
* scipy
* pandas
* scikit-learn
* matplotlib

Install all dependencies using:

pip install -r requirements.txt

---

## Running the Project

Run all experiments:

python run_experiments.py

Run selected datasets:

python run_experiments.py --datasets digits yeast

Specify the number of runs:

python run_experiments.py --n_runs 20

Disable sensitivity analysis:

python run_experiments.py --no_sensitivity

---

## Output

After execution, the project automatically generates:

Results

* Experimental summary tables
* Objective function values
* Clustering evaluation metrics
* Statistical test results

Figures

* Convergence plots
* Sensitivity heatmaps
* Runtime comparison
* PCA visualizations
* Confusion matrices
* Box plots

Tables

* Table III reproduction
* CSV summary tables
* LaTeX tables

---

## Evaluation Metrics

The following metrics are reported:

• Objective Function Value

• Normalized Mutual Information (NMI)

• Clustering Accuracy (ACC)

• Adjusted Rand Index (ARI)

---

## Benchmark Datasets

The project supports multiple benchmark datasets, including:

* Digits
* Yeast
* Binary Alpha
* MSRA25
* JAFFE50

Additional datasets can be added through the dataset loader module.

---

## Reproducibility

The implementation follows a consistent experimental protocol to ensure reproducibility, including identical initialization strategies, common stopping criteria, multiple independent runs, and standardized evaluation metrics across both clustering algorithms.

---

## Author

M.Tech Computer Science and Engineering

Pattern Classification Course Project

Academic Reproduction of the TPAMI 2025 UC-FCM Clustering Algorithm
