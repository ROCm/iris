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

# Mock triton but not the parts we're customizing
import sys
from unittest.mock import MagicMock

# First, create a basic mock for triton
sys.modules['triton'] = MagicMock()

# Mock triton.language with aggregate decorator
class AggregateMock:
    """Mock for @aggregate decorator that preserves class and method docstrings."""
    
    def __call__(self, cls):
        # Return the class unchanged to preserve docstrings
        return cls

class TritonLanguageMock:
    cast = MagicMock()

class TritonLanguageCoreMock:
    _aggregate = AggregateMock()

sys.modules['triton.language'] = TritonLanguageMock()
sys.modules['triton.language.core'] = TritonLanguageCoreMock()

class PreserveDocstringMock(MagicMock):
    """Mock that preserves __doc__ and other attributes from decorated functions."""
    
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            # This is being used as a decorator
            func = args[0]
            # Preserve the original function's attributes
            wrapper = MagicMock()
            wrapper.__doc__ = func.__doc__
            wrapper.__name__ = func.__name__
            wrapper.__module__ = func.__module__
            wrapper.__qualname__ = getattr(func, '__qualname__', func.__name__)
            wrapper.__annotations__ = getattr(func, '__annotations__', {})
            # Preserve the signature for autodoc
            wrapper.__signature__ = getattr(func, '__signature__', None)
            try:
                import inspect
                wrapper.__signature__ = inspect.signature(func)
            except:
                pass
            return wrapper
        return MagicMock(*args, **kwargs)

# Setup custom mocks for gluon
class GluonMock:
    jit = PreserveDocstringMock()

class GluonLanguageMock:
    constexpr = type
    tensor = type
    pointer_type = MagicMock()
    uint64 = type
    int8 = type
    
    @staticmethod
    def load(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def store(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def program_id(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def arange(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def BlockedLayout(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_add(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_sub(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_cas(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_xchg(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_xor(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_and(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_or(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_min(*args, **kwargs):
        return MagicMock()
    
    @staticmethod
    def atomic_max(*args, **kwargs):
        return MagicMock()

class TritonExperimentalMock:
    gluon = GluonMock()

# Inject custom mocks
sys.modules['triton.experimental'] = TritonExperimentalMock()
sys.modules['triton.experimental.gluon'] = GluonMock()
sys.modules['triton.experimental.gluon.language'] = GluonLanguageMock()

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
