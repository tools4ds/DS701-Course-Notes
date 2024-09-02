#!/bin/bash

qmd_files=(
  "01-Intro-to-Python.qmd"
  "02A-Git.qmd"
  "02B-Pandas.qmd"
  "02C-Sklearn.qmd"
  "04-Linear-Algebra-Refresher.qmd"
  "03-Probability-and-Statistics-Refresher.qmd"
  "05-Distances-Timeseries.qmd"
  "05A-Learning.qmd"
  "06-Clustering-I-kmeans.qmd"
  "07-Clustering-II-in-practice.qmd"
  "08-Clustering-III-hierarchical.qmd"
  "09-Clustering-IV-GMM-EM.qmd"
  "13-Learning-From-Data.qmd"
  "14-Classification-I-Decision-Trees.qmd"
  "15-Classification-II-kNN.qmd"
  "16-Classification-III-NB-SVM.qmd"
  "10-Low-Rank-and-SVD.qmd"
  "11-Dimensionality-Reduction-SVD-II.qmd"
  "17-Regression-I-Linear.qmd"
  "18-Regression-II-Logistic.qmd"
  "19-Regression-III-More-Linear.qmd"
  "24-NN-II-Backprop.qmd"
  "25-NN-III-CNNs.qmd"
  "20-Recommender-Systems.qmd"
  "21-Networks-I.qmd"
  "22-Networks-II-Centrality-Clustering.qmd"
)

# Check if jupyter_notebooks directory exists, if not, create it
if [ ! -d "jupyter_notebooks" ]; then
  mkdir jupyter_notebooks
fi

# Convert all .qmd files to Jupyter notebooks and move to jupyter_notebooks directory
for file in "${qmd_files[@]}"; do
  echo "Converting $file"
  
  if [ -f "$file" ]; then
    quarto convert "$file"
    mv "${file%.qmd}.ipynb" jupyter_notebooks/
  else
    echo "File not found: $file"
  fi
done
