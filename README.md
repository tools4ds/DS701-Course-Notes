# DS701 Course Book

This is the repository for DS701. It contains the course book and course lecture slides. The slides are derived from the book. 

The book and slides have been created using [Quarto](https://quarto.org/). In order to build the book and slides you need to install Quarto. You can install quarto from this [link](https://quarto.org/docs/get-started/). Follow the instructions in the *Get Started* section for VSCode. For VSCode you need to also install the Quarto extension. This allows you to preview the content you have created in VSCode.

## Python environment

> Tested on MacOS Sonoma 14.5 with python 3.12.4.

To execute the Python code used in the book requires several Python packages. 

This repository includes a `venv` environment with all the necessary pacakges. To set up this environment use the following terminal commands

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Conda environment (deprecated)

> Deprecating the conda environment in favor of pip with a newer version of python and 
> perhaps better GitHub actions support.

This repository includes a conda environment with all the necessary packages. To set up this environment use the following terminal commands

```sh
conda env create -f environment.yml
```

```sh
conda activate ds701_dev_env
```

## Building the book

> Tested with Quarto 1.5.55 on MacOS Sonoma 14.5.

To build the book you need to be in the `ds701_book` directory. Since the book is many chapters you need to set the following environment variable using the terminal command:

```
export QUARTO_DENO_EXTRA_OPTIONS=--v8-flags=--stack-size=8192
```

If the render gives a DENO error with the above environment variable, try

```sh
export QUARTO_DENO_EXTRA_OPTIONS=

export QUARTO_DENO_V8_OPTIONS=--stack-size=8192
```

Once this environment variable has been set you can render the entirebook using the terminal commands:

```sh
cd ds701_book
quarto render .
```

The html files are all located in the `_books` directory. The `_books` directory is not committed in the repository.

## Previewing chapters

To preview and individual chapter using VSCode, open that chapter's qmd file in VSCode and run `Shift-Command-K` in the terminal.
