# DS701 Course Book

This is the repository for DS701 course notes and lecture slides. The slides are derived from the same source content. 

The site and slides are created using [Quarto](https://quarto.org/). In order to build the site and slides you need to install Quarto. You can install quarto from this [link](https://quarto.org/docs/get-started/). Follow the instructions in the *Get Started* section for VSCode. For VSCode you need to also install the Quarto extension. This allows you to preview the content you have created in VSCode.

## Python environment

> Tested on MacOS Sonoma 14.5 with python 3.12.4.

To execute the Python code used in the book requires several Python packages. 

We recommend using [`venv`](https://docs.python.org/3/library/venv.html) to
create a virtual environment with all the necessary pacakges. To set up this
environment use the following terminal commands

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quarto Project Selection

We use [Quarto Projects](https://quarto.org/docs/projects/quarto-projects.html)
to manage the YAML configurations specific to the 
output type (e.g. website or lecture slides) using 
[project profiles](https://quarto.org/docs/projects/profiles.html).

Per profiles documentation, we define YAML configuration files for each project
type:

* `_quarto-web.yml`  to create the website in output directory `_site`
* `_quarto-slides.yml` to create slides in output directory `_revealjs`
* `_quarto-book.yml` to create the HTML book in output directory `_book` (**deprecated**)

along with a base `_quarto.yml` configuration.


## Building the Site Locally

> Tested with Quarto 1.5.55 on MacOS Sonoma 14.5.

To build the site you need to be in the `ds701_book` directory. 

```sh
cd ds701_book
```

Since the site contains many lectures you may need to set the following
environment variable using the terminal command:

```sh
export QUARTO_DENO_V8_OPTIONS=--stack-size=8192
```

Once this environment variable has been set you can render the entire site using the terminal commands:

```sh
# From ds701_book/ dir
quarto render
```

The html files are all located in the `_site` directory. The `_site` directory is not committed in the repository.

### Previewing the Lecture Notes Website

To preview and individual chapter using VSCode, open that chapter's qmd file in 
VSCode and run `Shift-Command-K` in the terminal or click on the preview icon
on the upper right of the code window,.

Alternatively you can run `quarto preview` from a terminal prompt in the `ds701_book` directory.
To exit preview, hit `Ctl-c` in the same terminal window.

## Rendering Slides

To render slides for each lecture run
```sh
quarto render --profile slides
```
from `ds701_book/`. The resulting slides are writtein to `_revealjs` which
is ignored by git.

Any easy way to select slides to preview is to open the folder in a browser
such as `file:///<path-to-project-parent-folder>/DS701-Course-Notes/ds701_book/_revealjs/`.
Then you can just click on one of the `.html` files to view the slides.

You can render just one slide with a command like
```sh
quarto render 05-Distances-Timeseries.qmd --profile slides
```


## Creating Jupyter Notebooks

To create Jupyter notebook versions of each of the lecture
notes, run `./cmd-cnvt-to-jupyter.sh` from the `ds701_book`
folder. It renders the Jupyter notebooks into the `jupyter_notebooks` folder
if the associated .qmd file has been modified.

> Whenever you change in any `.qmd` file that has python in it, re-run
> `./cmd-cnvt-to-jupyter.sh` and commit `jupyter_notebooks`.

## (deprecated) Rendering the Book

> Rendering a Quarto book has been deprecated and replaced by rendering to
> a website. This might still be useful, for example to render to a PDF.

To render the the book run
```sh
quarto render --profile book
```

This will render an HTML book format to `_book` which is ignored by git.

## Citations and Bibliography

In many cases citations are directly referenced in the text, but in some cases
we use Quarto's support for [citations](https://quarto.org/docs/authoring/citations.html)
in the BibTeX format.

For bibtex citations, add entries to 
[`ds701_book/references.bib`](./ds701_book/references.bib) and cite them as
directed Quarto [citations](https://quarto.org/docs/authoring/citations.html).

As stated in the Quarto documentation, the list of works will be placed at the
end of the web page or the last slide. You can control the location by including
a div with id `refs` such as:

```
## References

::: {#refs}
:::
```

The WikiBook on [LaTeX Bibliography Management](https://en.wikibooks.org/wiki/LaTeX/Bibliography_Management#BibTeX)
is a good reference on BibTeX format.
