#!/bin/bash

# Script to render all .qmd and .ipynb files to Reveal.js HTML slides.
# Note: this script is deprecated and you should use the `quarto render` command directly.

# Check if _revealjs directory exists, if not, create it
if [ ! -d "_revealjs" ]; then
  mkdir _revealjs
fi

# Render all .qmd files to HTML and Reveal.js
for file in *.qmd; do
  if [ -f "$file" ]; then
    quarto render "$file" --to revealjs --output-dir _revealjs
  fi
done

# Render all .ipynb files to HTML and Reveal.js
for file in *.ipynb; do
  if [ -f "$file" ]; then
    quarto render "$file" --to revealjs --output-dir _revealjs
  fi
done

# revealjs slideshow
#quarto render --to revealjs --output-dir _revealjs
