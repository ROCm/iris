# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Iris"
copyright = "2025, Advanced Micro Devices, Inc."
author = "AMD Research and Advanced Development Team"
# Display "latest" in the docs header instead of a fixed version
release = "latest"
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "rocm_docs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    # Exclude removed sections from build to avoid toctree warnings
    "how-to/**",
    "tutorials/**",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "rocm_docs_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]

# Add any paths that contain extra files (such as images) here,
# relative to this directory. These files are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_extra_path = ["images"]

# Customize the HTML title shown in the top-left/header
html_title = "Iris Documentation"

# -- Extension configuration -------------------------------------------------

# Autodoc configuration for generating docs from docstrings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
}

# Show type hints in documentation
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Render objects without full module path (e.g., show "Iris" instead of "iris.iris.Iris")
add_module_names = False

# Mock heavy/runtime-only dependencies when building docs
autodoc_mock_imports = [
    "torch",
    "numpy",
    "iris._distributed_helpers",
    "iris.hip",
]

# Custom mocks that preserve docstrings for Triton Gluon


# Docstring-preserving decorator mock
class PreserveDocstringMock:
    """Mock decorator that preserves docstrings and function attributes."""

    def __call__(self, func):
        # Return the original function unchanged to preserve docstrings
        return func


# Mock triton.language first
triton_language_mock = MagicMock()
sys.modules["triton.language"] = triton_language_mock
sys.modules["triton.language.core"] = MagicMock()
sys.modules["triton.language.core"]._aggregate = lambda cls: cls  # Preserve class


# Mock triton modules with docstring-preserving jit decorator
class TritonMock:
    jit = PreserveDocstringMock()
    language = triton_language_mock


sys.modules["triton"] = TritonMock()


# Mock gluon with docstring-preserving jit
class GluonMock:
    jit = PreserveDocstringMock()


sys.modules["triton.experimental"] = MagicMock()
sys.modules["triton.experimental"].gluon = GluonMock()
sys.modules["triton.experimental.gluon"] = GluonMock()
sys.modules["triton.experimental.gluon"].language = MagicMock()

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_warnings = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# ROCm docs handles most configuration automatically

# Table of contents
external_toc_path = "./sphinx/_toc.yml"

# Theme options for AMD ROCm theme
html_theme_options = {
    "flavor": "instinct",
    "link_main_doc": True,
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_hide = False
copybutton_remove_prompts = True

# Force copy buttons to be generated
html_context = {
    "copybutton_prompt_text": copybutton_prompt_text,
    "copybutton_prompt_is_regexp": copybutton_prompt_is_regexp,
    "copybutton_line_continuation_character": copybutton_line_continuation_character,
    "copybutton_hide": copybutton_hide,
    "copybutton_remove_prompts": copybutton_remove_prompts,
}
