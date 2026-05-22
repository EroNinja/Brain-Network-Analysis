# Brain Network Analysis — Machine Learning Pipeline

This project analyzes brain network connectivity data using classical machine learning models and graph neural networks. Each subject is represented by a structural brain connectivity matrix stored as a `.mat` file, along with subject metadata stored in an Excel file. The pipeline extracts edge-level, graph-theoretic, spectral, and hemispheric asymmetry features, then evaluates multiple predictive models using nested cross-validation.

## Project Goal

The goal of this project is to predict subject-level attributes from brain connectivity patterns. The notebook evaluates three classification tasks:

1. **Sex classification**
   - Female vs. Male

2. **Creativity classification**
   - Normal vs. Creative

3. **Mathematical ability classification**
   - Normal vs. High Math

The project compares different feature representations and machine learning models to determine which combination performs best for each task.

## Dataset

The notebook expects the following structure:

```text
brainnetworks/
├── smallgraphs/
│   ├── subject_1_fiber.mat
│   ├── subject_2_fiber.mat
│   └── ...
├── metainfo.xls
└── brain_network_analysis_v5.ipynb
```

Each `.mat` file should contain a variable named:

```text
fibergraph
```

The notebook is currently configured for **70 × 70 brain connectivity matrices**.

The metadata file should contain at least the following columns:

| Column | Description |
|---|---|
| `URSI` | Subject identifier used to match metadata with `.mat` files |
| `Sex` | Subject sex label |
| `Subject_type` | Group label used for creativity and math classification |

In the current run, the notebook loaded **114 matched subjects** after filtering metadata rows against the available `.mat` files.

## Feature Engineering

The pipeline builds several feature sets from each brain connectivity matrix.

### 1. Edge Features

The upper triangle of each 70 × 70 matrix is extracted to avoid duplicate symmetric edges.

```text
Edge features: 2,415
```

### 2. Graph and Spectral Features

The notebook computes graph-level statistics using NetworkX, including centrality and connectivity-related summaries. It also extracts spectral features from the normalized graph Laplacian.

```text
Graph + spectral statistics: 63
```

### 3. Hemispheric Asymmetry Features

The pipeline computes left-right hemispheric asymmetry features.

```text
Asymmetry features: 10
```

### 4. Combined Feature Set

All feature groups are concatenated into one representation.

```text
Total combined features: 2,488
```

## Models Used

The notebook evaluates the following models:

- Support Vector Machine with mutual information feature selection
- Support Vector Machine with PCA
- L1-regularized Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- Multi-Layer Perceptron
- XGBoost
- LightGBM
- Graph Convolutional Network
- Graph Isomorphism Network

The GNN models are only executed if PyTorch and PyTorch Geometric are installed.

## Evaluation Method

The project uses nested stratified cross-validation:

```text
Outer folds: 5
Inner folds: 3
Random seed: 42
```

The inner loop performs hyperparameter tuning with `GridSearchCV`, while the outer loop estimates generalization performance.

The main evaluation metrics are:

- Accuracy
- Balanced accuracy
- F1-score
- ROC-AUC
- Average precision

For imbalanced tasks, the pipeline uses class balancing techniques such as `BorderlineSMOTE` when `imbalanced-learn` is available. It also applies threshold tuning using Youden's J statistic.

## Results

Best models from the current notebook run:

| Task | Best Model | Feature Set | Accuracy | Balanced Accuracy | F1 |
|---|---|---|---:|---:|---:|
| Sex classification | Gradient Boosting | Edge upper triangle | 0.693 | 0.676 | 0.748 |
| Creativity classification | Gradient Boosting | Topology | 0.723 | 0.519 | 0.071 |
| High math classification | Extra Trees | Edge upper triangle | 0.793 | 0.550 | 0.182 |

Although creativity and high-math classification achieved relatively high accuracy, their balanced accuracy and F1-scores show that the class imbalance makes these tasks harder. Accuracy alone should not be treated as the final indicator of model quality.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib networkx statsmodels xlrd jupyter
```

Optional packages for additional models:

```bash
pip install imbalanced-learn xgboost lightgbm torch
```

PyTorch Geometric installation depends on the installed PyTorch and CUDA version. Follow the official PyTorch Geometric installation instructions if GNN support is required.

## Running the Notebook

1. Open the notebook:

```bash
jupyter notebook brain_network_analysis_v5.ipynb
```

2. Update the dataset paths in the configuration cell:

```python
MAT_DIR = "path/to/smallgraphs"
META_PATH = "path/to/metainfo.xls"
OUTPUT_DIR = "path/to/output_directory"
```

3. Run all notebook cells from top to bottom.

The notebook will:

- Load the brain connectivity matrices
- Match `.mat` files with subject metadata
- Normalize the connectivity graphs
- Extract edge, topology, spectral, and asymmetry features
- Build classification tasks
- Train and evaluate multiple models
- Print task-wise and overall best results

## Project Structure

```text
.
├── brain_network_analysis_v5.ipynb
├── README.md
└── data/
    ├── smallgraphs/
    │   └── *.mat
    └── metainfo.xls
```

## Notes

- The notebook expects each `.mat` file to contain a `fibergraph` variable.
- The current configuration assumes each brain network is a 70 × 70 matrix.
- If your matrices use a different number of nodes, update the `N_NODES` value in the configuration cell.
- XGBoost, LightGBM, imbalanced-learn, PyTorch, and PyTorch Geometric are optional. The notebook will skip or warn about unavailable optional libraries.
- Results may vary depending on available subject files, metadata filtering, package versions, and hardware.

## Summary

This project demonstrates an end-to-end brain network classification pipeline using structural connectivity matrices. It combines graph-based feature engineering, classical machine learning, imbalance-aware evaluation, and optional graph neural networks to study whether brain connectivity patterns can predict subject attributes such as sex, creativity group, and high mathematical ability.
