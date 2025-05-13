# AGE PREDICTION

**Date**: March 2025  
**Dataset**: [Kaggle - Age Prediction](https://www.kaggle.com/datasets/pooriamst/age-prediction)

## Overview

This project explores clustering techniques to analyze health-related data and identify distinct age-related health risk groups. The analysis focuses on glucose and insulin levels, using both K-Means and Hierarchical Clustering to form insights relevant to public health, medical researchers, and general users.

---

## Data Exploration

### Features Used:
- `ID` (Numeric)
- `Age_group` (String)
- `Age` (Numeric)
- `Gender` (Categorical)
- `PAQ605` (Numeric)
- `Body Mass Index` (Numeric)
- `Blood Glucose after fasting` (Numeric)
- `Diabetic or not` (Categorical)
- `Respondent's Oral` (Numeric)
- `Blood Insulin Levels` (Numeric)

---

## Feature Selection

- **Data Correlation** and **Hierarchical Clustering (Complete Linkage)** were used for dimensionality reduction.
- Selected Features for clustering:
  - `Blood Glucose after fasting`
  - `Blood Insulin Levels`

---

## Silhouette Score vs. Number of Clusters (K)

- **k = 2**: Highest Silhouette Score → indicates the best separation and cohesion.
- **k = 3**: Sharp drop in score → possible overlap and unclear groupings.
- **k = 4**: Slight improvement but not significant.
- **k = 5–9**: Gradual decline or plateau → additional clusters do not improve quality.

---

## Elbow Method for Optimal K

- WCSS drops significantly from k=1 to k=3.
- After k=4, the decrease slows, suggesting diminishing returns.
- **Elbow Point observed at k ≈ 3 or 4.**

---

## Hierarchical Clustering (Dendrogram)

- Shows a clear hierarchical structure with well-separated clusters.
- Uses **maximum linkage** to form equally-sized clusters.
- Suggests cutting at a high linkage distance → distinct 2-cluster separation.

---

## K-Means Clustering (with PCA reduction, k=2)

### Cluster 1 (Green):
- **Low Glucose**, **Moderate–High Insulin**
- Possibly insulin-sensitive individuals or early insulin resistance.
- May indicate healthy individuals or those with hyperinsulinemia.

### Cluster 2 (Blue):
- **High Glucose**, **Low Insulin**
- Indicative of Type 2 Diabetes or insulin deficiency.
- High-risk group for metabolic disorders.

---

## Clustering Performance

| Algorithm               | Silhouette Score (k=2) |
|------------------------|------------------------|
| **K-Means**            | 0.6661                 |
| **Hierarchical**       | 0.9167                 |

---

## Comparison & Discussion

### K-Means Clustering
-  Fast and scalable for large datasets.
-  Clear, well-separated clusters.
-  Requires pre-defined k value.
-  Not ideal for complex or non-convex shapes.

### Hierarchical Clustering
-  No need to specify k in advance.
-  Dendrograms reveal deeper relationships.
-  Suitable for multi-dimensional, complex health data.
-  Slower processing, sensitive to outliers.

---

## Conclusion

- **Hierarchical Clustering** provided more interpretable and meaningful groupings for age-related health segmentation, especially useful for medical professionals and researchers to understand hidden patterns.
- **K-Means Clustering** is suitable for faster, larger-scale applications such as categorizing patients by risk level.

---

## Benefits

-  **Medical professionals**: Support health risk stratification.
-  **Researchers**: Discover trends in insulin/glucose responses by age.
-  **General public**: Raise awareness of individual health profiles and potential risks.
