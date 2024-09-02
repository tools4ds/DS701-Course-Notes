---
title: "Clustering Topics with SKlearn"
output: html_document
---

# Clustering Related Topics at Scikit-learn

This is a summary of clustering related topics, classes and functions at
scikit-learn.

## Gaussian Mixture Models

### Algorithms

1. Gaussian Mixture via Expectation Maximization (GaussianMixture)
2. Variational Bayesian Gaussian Mixture (BayesianGaussianMixture)

### Examples

- [GMM Covariance Example Types](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html)
    - demonstration of several covariance types for GMMs
- [density estimation for a gaussian mixture](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py)
    - plot the density estimation of a mixture of two Gaussians
- [Gaussian Mixture Model Ellipsoids](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py)
    - plot the confidence ellipsoids of a mixture fo two Gaussians obtained with EM

## Manifold Learning


## Clustering


### Algorithms

1. K-means
    1. mini-batch k-means
2. Affinity Propagation
3. Mean Shift
4. Spectral Clustering
5. DBSCAN
6. HDBSCAN
7. OPTICS
8. Birch

### Clustering Performance Evaluation

1. Rand Index, ARI
2. Mutual Information Based Scores
3. Homogeneity, Completeness, V-measure
4. Fowlkes-Mallows Scores
5. Silhouette Coefficient
6. Calinski-Harabasz Index
7. Davies-Bouldin Index
8. Contingency Matrix
9. Pair Confusion Matrix

### Applications

1. Image Palletization
2. 20 Newsgroups
3. 