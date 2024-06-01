#
# Configuration file for the Sphinx documentation builder.
#

import os
import shutil
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
author = "Bas Nijholt"
copyright = f"2024, {author}"

version = ".".join(pipefunc.__version__.split(".")[:3])
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

html_logo = "https://github.com/basnijholt/nijho.lt/raw/92b0aa820318f466388d828adf01120760255acf/content/project/pipefunc/featured.png"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "pipefuncdoc"

# -- Extension configuration -------------------------------------------------

default_role = "autolink"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

autodoc_member_order = "bysource"

# myst-nb configuration
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
myst_heading_anchors = 3


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
nb = package_path / "example.ipynb"
convert_notebook_to_md(nb, docs_path / "source" / "tutorial.md")

# Copy nb to docs/source/notebooks
nb_docs_folder = docs_path / "source" / "notebooks"
nb_docs_folder.mkdir(exist_ok=True)
shutil.copy(nb, nb_docs_folder)

# Group into single streams to prevent multiple output boxes
nb_merge_streams = True

def setup(app) -> None:
    pass
