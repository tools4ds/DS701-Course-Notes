# DS701 Course Book

This is the repository for DS701. It contains the course book and course lecture slides. The slides are derived from the book. 

The book and slides have been created using [Quarto](https://quarto.org/). In order to build the book and slides you need to install Quarto. You can install quarto from this [link](https://quarto.org/docs/get-started/). Follow the instructions in the *Get Started* section for VSCode. For VSCode you need to also install the Quarto extension. This allows you to preview the content you have created in VSCode.

## Python environment

To executre the Python code used in the book requires several Python packages. This repository includes a conda environment with all the necessary pacakges. To set up this environment use the following terminal commands

```conda env create -f environment.yml```

```conda activate ds701_dev_env```

## Building the book

To build the book you need to be in the `ds701_book` directory. Since the book is many chapters you need to set the following environment variable using the terminal command:

```export QUARTO_DENO_EXTRA_OPTIONS=--v8-flags=--stack-size=8192```

Once this environment variable has been set you can render the entirebook using the terminal command:

```quarto render .```

The html files are all located in the `_books` directory. The `_books` directory is not committed in the repository.

## Previewing chapters

To preview and individual chapter using VSCode, open that chapter's qmd file in VSCode and run `Shift-Command-K` in the terminal.
