import os
import sys
from importlib.metadata import version as _version

# Make the src/ layout visible to autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------

project = "torchscalers"
author = "Daniel Paukner"
copyright = "2026, Daniel Paukner"
release = _version("torchscalers")

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# sphinx-autodoc-typehints: render type annotations in the Parameters section
always_document_param_types = True
simplify_optional_unions = True

# autodoc: show type annotations, keep the same member order as the source
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
