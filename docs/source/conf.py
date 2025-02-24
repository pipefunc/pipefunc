#
# Configuration file for the Sphinx documentation builder.
#

import os
import sys
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from docutils.statemachine import StringList

PACKAGE_PATH = Path("../..").resolve()
sys.path.insert(0, str(PACKAGE_PATH))
os.environ["PYTHONPATH"] = ":".join(
    (str(PACKAGE_PATH), os.environ.get("PYTHONPATH", "")),
)
docs_path = Path("..").resolve()
sys.path.insert(1, str(docs_path))

import pipefunc  # noqa: E402, isort:skip

# -- Project information -----------------------------------------------------

project = "pipefunc"
author = "PipeFunc Developers"
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
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "notfound.extension",
    "sphinxcontrib.mermaid",
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
    "adaptive_scheduler": ("https://adaptive-scheduler.readthedocs.io/en/latest/", None),
}

autodoc_member_order = "bysource"

# myst-nb configuration
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
myst_heading_anchors = 4
myst_enable_extensions = [
    "html_admonition",
    "colon_fence",
]
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/pipefunc/pipefunc",
    "repository_branch": "main",
    "home_page_in_toc": False,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "navigation_with_keys": False,
    "analytics": {
        "plausible_analytics_domain": "pipefunc.readthedocs.io",
        "plausible_analytics_url": "https://plausible.nijho.lt/js/script.js",
    },
}


class TryNotebookWithUV(SphinxDirective):
    """Render a tip box with the opennb command to open the current page in a Jupyter notebook."""

    has_content = True
    optional_arguments = 1

    def run(self):
        source_path = os.path.relpath(self.get_source_info()[0], PACKAGE_PATH)
        notebook_url = self.arguments[0] if self.arguments else source_path

        lines = [
            ":::{admonition} Have [`uv`](https://docs.astral.sh/uv/)? ⚡",
            ":class: tip, dropdown",
            "",
            "If you have [`uv`](https://docs.astral.sh/uv/) installed, you can instantly open this page as a Jupyter notebook [using `opennb`](https://github.com/basnijholt/opennb):",
            "",
            "```bash",
            f'uvx --with "pipefunc[docs]" opennb pipefunc/pipefunc/{notebook_url}',
            "```",
            "",
            "This command creates an ephemeral environment with all dependencies and launches the notebook in your browser in 1 second - no manual setup needed! ✨.",
            "",
            "Alternatively, run:",
            "",
            "```bash",
            f"uv run https://raw.githubusercontent.com/pipefunc/pipefunc/refs/heads/main/get-notebooks.py",
            "```",
            "",
            "to download *all* documentation as Jupyter notebooks.",
            ":::",
        ]

        # Convert lines to StringList with proper source information
        source_info = self.get_source_info()
        string_list = StringList(lines, source=(source_info[0], source_info[1]))

        # Parse the content
        node = nodes.Element()
        self.state.nested_parse(string_list, 0, node)

        return node.children


def setup(app):
    app.add_directive("try-notebook", TryNotebookWithUV)


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


def _change_alerts_to_admonitions(input_text: str) -> str:
    # Splitting the text into lines
    lines = input_text.split("\n")

    # Placeholder for the edited text
    edited_text = []

    # Mapping of markdown markers to their new format
    mapping = {
        "IMPORTANT": "important",
        "NOTE": "note",
        "TIP": "tip",
        "WARNING": "caution",
    }

    # Variable to keep track of the current block type
    current_block_type = None

    for line in lines:
        # Check if the line starts with any of the markers
        if any(line.strip().startswith(f"> [!{marker}]") for marker in mapping):
            # Find the marker and set the current block type
            current_block_type = next(marker for marker in mapping if f"> [!{marker}]" in line)
            # Start of a new block
            edited_text.append("```{" + mapping[current_block_type] + "}")
        elif current_block_type and line.strip() == ">":
            # Empty line within the block, skip it
            continue
        elif current_block_type and not line.strip().startswith(">"):
            # End of the current block
            edited_text.append("```")
            edited_text.append(line)  # Add the current line as it is
            current_block_type = None  # Reset the block type
        elif current_block_type:
            # Inside the block, so remove '>' and add the line
            edited_text.append(line.lstrip("> ").rstrip())
        else:
            # Outside any block, add the line as it is
            edited_text.append(line)

    # Join the edited lines back into a single string
    return "\n".join(edited_text)


def change_alerts_to_admonitions(input_file: Path, output_file: Path) -> None:
    """Change markdown alerts to admonitions.

    For example, changes
    > [!NOTE]
    > This is a note.
    to
    ```{note}
    This is a note.
    ```
    """
    with input_file.open("r") as infile:
        content = infile.read()
    new_content = _change_alerts_to_admonitions(content)

    with output_file.open("w") as outfile:
        outfile.write(new_content)


def process_readme_for_sphinx_docs(readme_path: Path, docs_path: Path) -> None:
    """Process the README.md file for Sphinx documentation generation.

    Parameters
    ----------
    readme_path
        Path to the original README.md file.
    docs_path
        Path to the Sphinx documentation source directory.

    """
    # Step 1: Copy README.md to the Sphinx source directory and apply transformations
    output_file = docs_path / "source" / "README.md"
    replace_named_emojis(readme_path, output_file)
    change_alerts_to_admonitions(output_file, output_file)


def replace_rtd_links_with_local(input_file: str, output_file: str) -> None:
    """Replace ReadTheDocs links with local markdown links in a file.

    Parameters
    ----------
    input_file
        Path to the input file
    output_file
        Path to the output file where the modified content will be written
    """
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the RTD links with local markdown links
    base_url = "https://pipefunc.readthedocs.io/en/latest/"
    content = content.replace(f"{base_url}examples/", "./examples/")
    content = content.replace("/) page.", ".md) page.")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)


# Process the README.md file for Sphinx documentation
readme_path = PACKAGE_PATH / "README.md"
process_readme_for_sphinx_docs(readme_path, docs_path)

# Add the example notebook to the docs
nb = PACKAGE_PATH / "example.ipynb"
convert_notebook_to_md(nb, docs_path / "source" / "tutorial.md")
replace_rtd_links_with_local(
    docs_path / "source" / "tutorial.md",
    docs_path / "source" / "tutorial.md",
)

# Group into single streams to prevent multiple output boxes
nb_merge_streams = True


def setup(app) -> None:
    app.add_directive("try-notebook", TryNotebookWithUV)
    app.add_css_file("custom.css")
