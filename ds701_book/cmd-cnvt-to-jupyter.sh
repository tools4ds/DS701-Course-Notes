#!/bin/bash

qmd_files=(
  "01-Intro-to-Python.qmd"
  "02B-Pandas.qmd"
  "02C-Sklearn.qmd"
  "04-Linear-Algebra-Refresher.qmd"
  "03-Probability-and-Statistics-Refresher.qmd"
  "05-Distances-Timeseries.qmd"
  "06-Clustering-I-kmeans.qmd"
  "07-Clustering-II-in-practice.qmd"
  "08-Clustering-III-hierarchical.qmd"
  "09-Clustering-IV-GMM-EM.qmd"
  "09B-Appendix-Cov-Matrix-in-MV-Gaussians.qmd"
  "13-Learning-From-Data.qmd"
  "14-Classification-I-Decision-Trees.qmd"
  "15-Classification-II-kNN.qmd"
  "16-Classification-III-NB-SVM.qmd"
  "10-Low-Rank-and-SVD.qmd"
  "11-Dimensionality-Reduction-SVD-II.qmd"
  "17-Regression-I-Linear.qmd"
  "18-Regression-II-Logistic.qmd"
  "18A-decision-tree-regression.qmd"
  "18B-gradient-boosting-methods.qmd"
  "19-Regression-III-More-Linear.qmd"
  "24-NN-II-Backprop.qmd"
  "25-NN-III-CNNs.qmd"
  "20-Recommender-Systems.qmd"
  "21-Networks-I.qmd"
  "22-Networks-II-Centrality-Clustering.qmd"
  "26-TimeSeries.qmd"
  "27-RNN.qmd"
  "28-NLP.qmd"
  "29-NN-IV-Scikit-Learn.qmd"
  "30-NN-consolidated.qmd"
)

# Check if jupyter_notebooks directory exists, if not, create it
if [ ! -d "jupyter_notebooks" ]; then
  mkdir jupyter_notebooks
fi

# Convert .qmd files to Jupyter notebooks if they are newer than existing .ipynb files
for file in "${qmd_files[@]}"; do
  echo "  "
  echo "Processing $file"
  
  if [ -f "$file" ]; then
    qmd_file="$file"
    ipynb_file="jupyter_notebooks/${file%.qmd}.ipynb"
    
    if [ ! -f "$ipynb_file" ] || [ "$qmd_file" -nt "$ipynb_file" ]; then
      # echo "Converting $file"
      quarto convert "$qmd_file" --profile slides
      mv "${qmd_file%.qmd}.ipynb" "$ipynb_file"
    else
      echo "Skipping $file (up to date)"
    fi
  else
    echo "File not found: $file"
  fi
done
