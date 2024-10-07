#
# Configuration file for the Sphinx documentation builder.
#

import os
import re
import shutil
import sys
from pathlib import Path

from docutils import nodes
from docutils.nodes import Node
from docutils.statemachine import ViewList
from sphinx import addnodes
from sphinx.directives.other import TocTree
from sphinx.util.nodes import nested_parse_with_titles

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
}


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


def get_top_yaml_block(file_path: Path):
    with file_path.open(encoding="utf-8") as f:
        content = f.read()
    # Split the content by '---' and check for valid sections
    sections = content.split("---")
    if len(sections) >= 3:  # noqa: PLR2004
        # Return the content between the first and second '---'
        return "---\n" + sections[1].strip() + "\n---"
    msg = "Invalid YAML block in the file."
    raise ValueError(msg)


def split_notebook_by_headers(notebook_md: Path) -> list[str]:
    with notebook_md.open(encoding="utf-8") as f:
        content = f.read()

    folder = notebook_md.parent / "tutorial"
    folder.mkdir(exist_ok=True, parents=True)

    yaml_block = get_top_yaml_block(notebook_md)
    headers = [
        ("getting_started.md", "# Tutorial for Pipefunc Package"),
        ("advanced.md", "## Advanced features"),
        ("examples.md", "## Full Examples"),
    ]
    parts = []
    for i, (_, header) in enumerate(headers):
        assert header in content
        part, content = content.split(header, 1)
        content = header + content
        if i > 0:
            part = yaml_block + part
        parts.append(part)
    # Add the last part
    parts.append(yaml_block + content)
    for (filename, _), part in zip(headers, parts):
        with (folder / filename).open("w", encoding="utf-8") as f:
            f.write(part)

    return parts


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


class ExpandHeadersTocTree(TocTree):
    def run(self) -> list[Node]:
        # Make sure the directive is not set to hidden.
        assert self.options.get("hidden", False) == False

        # Initialize a root list for our toc
        root_list = nodes.bullet_list()

        # We expect a single document name (e.g., 'tutorial') as the content of this directive
        docname = self.content[0].strip()

        # Read the target document from Sphinx's environment
        env = self.state.document.settings.env
        doc_path = env.doc2path(docname)

        # Read and parse the document's contents to extract its section titles
        with open(doc_path, "r") as doc_file:
            doc_content = doc_file.readlines()

        # Create a string list from the document lines
        viewlist = ViewList(doc_content, docname)

        # Parse the document to get all section titles
        section_titles = []
        print("yolo", viewlist)
        for line in viewlist:
            if line.startswith("# ") or line.startswith("## "):
                # We assume that lines starting with '#' are section headers
                title_text = line.strip("#").strip()
                section_id = re.sub(r"\s+", "-", title_text.lower())
                section_titles.append((title_text, section_id))

        # Create list items for each section header
        for title, section_id in section_titles:
            node = addnodes.compact_paragraph()
            target = f"{docname}#{section_id}"
            xref_node = addnodes.pending_xref(
                target, reftype="doc", refdomain="std", reftarget=target
            )
            xref_node += nodes.inline(
                "", title, classes=["xref", "std", "std-doc"]
            )  # Corrected here
            node += xref_node
            root_list += nodes.list_item("", node)

        # Now we generate the actual hidden toctree to be used
        self.options["hidden"] = True
        toctree_output = super().run()

        # Append our dynamically generated list of section links to the wrapper node
        wrapper_node = toctree_output[-1]
        wrapper_node.append(root_list)

        return toctree_output


# Process the README.md file for Sphinx documentation
readme_path = package_path / "README.md"
process_readme_for_sphinx_docs(readme_path, docs_path)

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
    app.add_directive("expand_headers_toc", ExpandHeadersTocTree)
    return {"version": "1", "parallel_read_safe": True}
