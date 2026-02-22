# Brain Network Analysis
*A Statistical Machine Learning Project*

## Overview
This project analyzes **structural brain connectivity networks** to study how differences in connectivity relate to individual traits such as **gender**, **cognitive ability (math)**, and **creativity**.

Each subject’s brain is represented as a **weighted graph** with 70 regions of interest (ROIs). Using these graphs, we extract features and train machine learning models to evaluate how well brain connectivity patterns predict subject traits.

**Dataset**
- 114 subjects  
- 70 × 70 weighted connectivity matrices  
- Metadata includes sex and subject type (normal, creative, high-math)

---

## Objectives
We address three classification tasks:

1. **Gender Classification (Male vs Female)**  
   Identify sex-specific patterns in brain connectivity.

2. **Cognitive Ability Classification (High Math vs Normal)**  
   Analyze whether structural connectivity reflects mathematical aptitude.

3. **Creativity Classification (Creative vs Normal)**  
   Evaluate whether creativity is encoded in structural brain networks.

---

## Methodology

### Data Processing
- Load brain connectivity graphs from `.mat` files
- Symmetrize adjacency matrices
- Normalize edge weights
- Align connectivity graphs with subject metadata

### Feature Representation
- Flatten full connectivity matrices (4900 features per subject)
- Handle high dimensionality using:
  - Feature selection (SelectKBest with ANOVA F-test)
  - Dimensionality reduction (PCA)

### Modeling
- Support Vector Machine (RBF kernel)
- Balanced class weights
- Stratified 5-fold cross-validation

### Evaluation Metrics
- Accuracy
- Balanced Accuracy
- F1-Score
- Confusion Matrix

---

## Results

| Task | Accuracy | Balanced Accuracy | F1-Score |
|----|----|----|----|
| Sex (Male/Female) | 0.667 | 0.655 | 0.716 |
| Math (Normal/High) | 0.655 | 0.496 | 0.211 |
| Creative (SelectKBest) | 0.468 | 0.350 | 0.074 |
| Creative (PCA) | 0.457 | 0.454 | 0.320 |

### Observations
- Gender classification shows the strongest separability in structural connectivity.
- Math classification exhibits moderate signal but is affected by class imbalance.
- Creativity classification performs weakly, suggesting creativity may not be strongly captured by structural connectivity alone.

---

## Visual Analysis
The project includes:
- Group-averaged connectivity heatmaps
- Difference matrices between classes
- Identification of top discriminative ROI-to-ROI connections

These visualizations support interpretability and highlight task-specific subnetworks.

---

## Dependencies
- Python 3.x
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn

---

## References 
- Duarte-Carvajalino, 2012  
- Alper et al., 2013  
- Brain Network Dataset:  
  https://www.andrew.cmu.edu/user/lakoglu/courses/95828/S17/projectsources/brainnetworks.rar

---

## Future Work
- Incorporate explicit graph-theoretic features (e.g., clustering coefficient, modularity, centrality) instead of relying solely on flattened connectivity matrices.
- Explore alternative models such as Graph Neural Networks (GNNs) to better capture structural dependencies in brain networks.
- Address class imbalance and label noise, particularly in the creativity classification task.
- Extend analysis to functional connectivity data or multimodal brain representations for improved predictive power.

---

## Notes
This project emphasizes **empirical analysis and interpretability**. Lower performance on some tasks (e.g., creativity) is treated as a meaningful finding rather than a failure.
