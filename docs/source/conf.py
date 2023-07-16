#
# Configuration file for the Sphinx documentation builder.
#

import os
import sys
from pathlib import Path

package_path = Path("../..").resolve()
sys.path.insert(0, str(package_path))
os.environ["PYTHONPATH"] = ":".join(
    (str(package_path), os.environ.get("PYTHONPATH", "")),
)
docs_path = Path("..").resolve()
sys.path.insert(1, str(docs_path))

import pipefunc  # noqa: E402, isort:skip

# -- Project information -----------------------------------------------------

project = "pipefunc"
copyright = "2023, Bas Nijholt"
author = "Bas Nijholt"

version = pipefunc.__version__
release = pipefunc.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_fontawesome",
]

source_parsers = {}
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "adaptivedoc"

# -- Extension configuration -------------------------------------------------

default_role = "autolink"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# myst-nb configuration
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True


def replace_named_emojis(input_file: Path, output_file: Path) -> None:
    """Replace named emojis in a file with unicode emojis."""
    import emoji

    with input_file.open("r") as infile:
        content = infile.read()
        content_with_emojis = emoji.emojize(content, language="alias")

        with output_file.open("w") as outfile:
            outfile.write(content_with_emojis)


def convert_notebook_to_md(input_file: Path, output_file: Path) -> None:
    """Convert a notebook to markdown."""
    import jupytext

    notebook = jupytext.read(input_file)
    notebook.metadata["jupytext"] = {"formats": "md:myst"}
    jupytext.write(notebook, output_file)


# Call the function to replace emojis in the README.md file

input_file = package_path / "README.md"
output_file = docs_path / "source" / "README.md"
replace_named_emojis(input_file, output_file)

# Add the example notebook to the docs
convert_notebook_to_md(
    package_path / "example.ipynb", docs_path / "source" / "tutorial.md"
)


def setup(app):
    pass
