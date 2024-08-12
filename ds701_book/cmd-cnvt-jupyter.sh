#!/bin/bash

# Check if _jupyter directory exists, if not, create it
if [ ! -d "_jupyter" ]; then
  mkdir _jupyter
fi

# Convert all .qmd files to Jupyter notebooks and move to _jupyter directory
for file in *.qmd; do
  if [ -f "$file" ]; then
    quarto convert "$file"
    mv "${file%.qmd}.ipynb" _jupyter/
  fi
done
