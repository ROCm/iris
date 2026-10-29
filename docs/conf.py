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
import shutil
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
    # Sphinx removed its bundled jQuery in 6.0, but four rocm-docs-core theme
    # scripts (code_word_breaks, renameVersionLinks, rdcMisc,
    # theme_mode_captions) still call $(). Without this every page throws
    # "$ is not defined" and those scripts silently do nothing.
    "sphinxcontrib.jquery",
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

# Render objects without full module path (e.g., show "Iris" instead of "iris.host.iris.Iris")
add_module_names = False

# Mock heavy/runtime-only dependencies when building docs
autodoc_mock_imports = [
    "torch",
    "numpy",
    "iris.host.distributed.helpers",
    "iris.host.platform.hip",
    "tritonblas",
    # rocm-xio. iris/host/iris.py imports it unconditionally at module load, so
    # without this every autodoc directive fails with ModuleNotFoundError and the
    # whole API reference renders empty. The docs CI installs only
    # docs/sphinx/requirements.txt, so xio is never present there either.
    "xio",
]

# Custom mocks that preserve docstrings for Triton Gluon


# Docstring-preserving decorator mock
class PreserveDocstringMock:
    """Mock decorator that preserves docstrings and function attributes."""

    def __call__(self, func=None, **kwargs):
        # Handle both @decorator and @decorator() usage patterns
        if func is None or not callable(func):
            return lambda f: f
        # Return the original function unchanged to preserve docstrings
        return func


# Mock triton.language first
triton_language_mock = MagicMock()


# These tl.* names are used as parameter ANNOTATIONS throughout the device APIs,
# so autodoc resolves them and prints their repr into the rendered signature. A
# bare MagicMock attribute reprs as
#     <MagicMock name='mock.constexpr' id='139779281861264'>
# which leaks into the parameter list, and the id changes on every build so the
# HTML churns too. Giving each one a real class makes it repr as its own name.
# The list is every `: tl.<name>` annotation appearing in iris/ -- if a new one
# is introduced it will show up as a MagicMock on the rendered page.
# The classes also take a value, as the real ones do: module-level constants such
# as `GATE_WORDS = tl.constexpr(8)` (iris/experimental/plan/triton.py) are built at
# import time, and a class that takes no arguments fails that import, so autodoc
# renders the module's whole reference page empty.
def _mock_type_init(self, value=None, *args, **kwargs):
    self.value = value


def _name_mock_types(mock, module, names):
    """Give a mocked module real classes for the names used as annotations."""
    for n in names:
        cls = type(n, (), {"__doc__": "Triton %s." % n, "__init__": _mock_type_init})
        cls.__module__ = module
        setattr(mock, n, cls)


_name_mock_types(triton_language_mock, "triton.language", ("constexpr", "tensor"))

sys.modules["triton.language"] = triton_language_mock
sys.modules["triton.language.core"] = MagicMock()
sys.modules["triton.language.core"]._aggregate = lambda cls: cls  # Preserve class
sys.modules["triton.language.extra"] = MagicMock()
sys.modules["triton.language.extra.hip"] = MagicMock()
sys.modules["triton.language.target_info"] = MagicMock()


# Mock triton modules with docstring-preserving jit decorator
class TritonMock:
    jit = PreserveDocstringMock()
    language = triton_language_mock
    constexpr_function = PreserveDocstringMock()


sys.modules["triton"] = TritonMock()


# Mock gluon with docstring-preserving jit
class GluonMock:
    jit = PreserveDocstringMock()
    constexpr_function = PreserveDocstringMock()


sys.modules["triton.experimental"] = MagicMock()
sys.modules["triton.experimental"].gluon = GluonMock()
sys.modules["triton.experimental.gluon"] = GluonMock()
_gluon_language_mock = MagicMock()
# Gluon annotates with gl.constexpr, which resolves through a *different* mock
# module than tl.constexpr -- so it needs the same treatment or it reprs as a
# MagicMock in the rendered signature.
_name_mock_types(_gluon_language_mock, "triton.experimental.gluon.language", ("constexpr", "tensor"))
sys.modules["triton.experimental.gluon"].language = _gluon_language_mock
sys.modules["triton.experimental.gluon.language"] = _gluon_language_mock

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

# ---------------------------------------------------------------------------
# Copy docs/images/ verbatim into the built site.
#
# Several conceptual pages reference images from raw HTML blocks, e.g.
#   <img class="dark-theme-img only-dark" src="../images/iris-model.png">
# which is how the light/dark theme variants are selected. Sphinx only tracks
# and copies images referenced through markdown or image directives, so every
# raw-HTML reference 404s on the published site -- 18 of them, across
# conceptual/programming-model, conceptual/taxonomy and index.
#
# Copying the directory wholesale fixes all of them without rewriting the
# theme-switching markup.
# ---------------------------------------------------------------------------


def _copy_images(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    src = os.path.join(os.path.dirname(__file__), "images")
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(app.outdir, "images"), dirs_exist_ok=True)


def setup(app):
    app.connect("build-finished", _copy_images)
