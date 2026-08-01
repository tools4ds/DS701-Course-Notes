# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DS701 ("Tools for Data Science") course notes: a set of Quarto `.qmd` lecture files that render to **three different outputs from the same source** — a website, reveal.js slides, and (deprecated) a book. The published site is at https://tools4ds.github.io/DS701-Course-Notes/. There is no application code to run — the deliverable is rendered content.

The Quarto project root is the **repository root**: the lecture `.qmd` files and the `_quarto-*.yml` configs live there, with `scripts/` (Python helper modules + build utilities), `jupyter_notebooks/`, `figs/`, `data/`, `drawio/`, and `lecture-graph/` as subdirectories. A leftover `ds701_book/` directory may exist locally but holds only git-ignored build artifacts — nothing under it is tracked.

## Environment setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # also: pip install jupyter
```

Tested with Python 3.12 and Quarto 1.5.55. `requirements.txt` maps packages to the lecture that needs them — when adding a Python dependency in a `.qmd`, add it there with a comment naming the lecture (matching the existing convention).

## Build / render commands

All rendering runs from the **repository root**. Large renders need a bigger V8 stack:

```sh
export QUARTO_DENO_V8_OPTIONS=--stack-size=8192

quarto render                              # website -> _site/ (default profile: web)
quarto render --profile slides             # all slides -> _revealjs/
quarto render 05-Distances-Timeseries.qmd --profile slides   # single lecture, slides
quarto preview                             # live preview of the website
```

Outputs (`_site/`, `_revealjs/`, `_book/`) are all git-ignored and never committed.

## Profiles and multi-output architecture

This is the central concept to understand before editing content. `_quarto.yml` sets `profile.default: web`. Each profile pulls in its own config, which defines its own render list, output dir, and format:

- `_quarto-web.yml` → website, `_site/`, `format: html`
- `_quarto-slides.yml` → slides, `_revealjs/`, `format: revealjs`
- `_quarto-book.yml` → book, `_book/` (**deprecated** — kept for PDF export)

A single `.qmd` therefore feeds multiple outputs. Content is shown/hidden per output using Quarto conditional divs, which appear throughout the lectures:

```
::: {.content-visible when-profile="slides"}   # slides only
::: {.content-hidden when-profile="slides"}     # everything except slides
::: {.incremental}   ::: {.fragment}            # slide-only reveal animations
```

When you add or rename a `.qmd`, you must add it to the `render:` list **and** the sidebar `contents:` in the relevant `_quarto-*.yml` files — the lists are maintained by hand and are not identical across profiles (e.g. `12-Anomaly-Detection-SVD-III.qmd` is excluded from web but present in slides).

`scripts/print_quarto_config.py` is a `pre-render` hook that merges and dumps the active config to `config_log.yml` for debugging what a profile actually resolves to.

## Jupyter notebook generation (must stay in sync)

`jupyter_notebooks/*.ipynb` are **generated from the `.qmd` files and committed** (they back the "Open in Colab" badges at the top of each lecture). Regenerate after editing any `.qmd` containing Python:

```sh
./cmd-cnvt-to-jupyter.sh    # from the repo root; converts only .qmd files newer than their .ipynb, commit the result
```

The list of converted files lives inside that script and is maintained separately from the profile render lists.

## Slide stripping (for presentation)

`scripts/strip-tags-with-profile.py` produces a `-stripped.qmd` with conditional-div wrappers resolved and `.fragment`/`.incremental` tags removed, for a cleaner presentation source:

```sh
./scripts/strip-tags-with-profile.py 11-Dimensionality-Reduction-SVD-II.qmd --profile slides
```

## Caching

Both website and slides configs set `execute: freeze: auto` and `cache: true`. Rendered/executed outputs are frozen under `_freeze/` (committed) so CI and other machines don't re-execute unchanged code. If a code change isn't reflected after render, the freeze/cache is likely stale.

## Deployment

`.github/workflows/publish.yml` renders the `web` profile at the repo root on every push to `main` (installs Quarto, GraphViz, FFmpeg, Python deps) and deploys `_site` to GitHub Pages. Merging to `main` publishes.

## Content conventions

- Every lecture `.qmd` has YAML front matter with `title:` and `jupyter: python3`, followed by a Colab badge linking to its generated notebook.
- Shared course URLs (Piazza, Gradescope, GitHub) live in `_variables.yml` and are referenced as Quarto variables — don't hardcode them.
- BibTeX citations go in `references.bib`; enable per-file via `bibliography: references.bib` in front matter (not enabled globally).
- Reusable Python for lectures lives in module files under `scripts/` (`laUtilities.py`, `slideUtilities.py`, `recommender_*.py`). Lectures import these as **top-level** modules (`import laUtilities as ut`), and `recommender_MF.py` imports its siblings the same way — see the `_environment` note below for how that resolves. Don't convert these to `from scripts.x import y`; it would break the intra-module imports.

## The `_environment` file

`_environment` at the repo root sets `PYTHONPATH=scripts` so the lectures' top-level imports of the helper modules resolve. Quarto applies it on every **project** render and passes it through to the Jupyter kernel.

Caveat: `_environment` is a project-level feature. It applies to `quarto render` and to `quarto render <lecture>.qmd` run from the repo root (both are project renders), but is ignored if a `.qmd` is rendered outside the project.

## Colab badge paths

Badges point at `.../blob/main/jupyter_notebooks/<lecture>.ipynb` (or `class_activity_notebooks/` for in-class exercises) — no `ds701_book/` segment. The same badge is embedded in the generated `.ipynb`, so a path change has to be applied to both the `.qmd` and the committed notebook.

Known dead badge: `12-Anomaly-Detection-SVD-III.qmd` links to a notebook that has never been generated — the lecture is excluded from the web profile and absent from `cmd-cnvt-to-jupyter.sh`, so the badge isn't reachable from the published site.

## grading-utils/

Separate, self-contained: `fa25-grading.ipynb` plus grade CSV/XLSX files for computing course grades. Unrelated to the book build.
