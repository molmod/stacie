# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

from intersphinx_registry import get_intersphinx_mapping
from packaging.version import Version
from sphinx.ext.apidoc import main as main_api_doc

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Stacie"
copyright = "2024, Toon Verstraelen"  # noqa: A001
author = "Toon Verstraelen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Third-party extensions
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = get_intersphinx_mapping(packages={"python", "numpy", "scipy"})
nitpicky = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]

nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nb_execution_mode = "cache"
nb_merge_streams = True
exclude_patterns = ["conf.py"]
codeautolink_concat_default = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# Embedded SVG as recommended in Furo template.
# See https://pradyunsg.me/furo/customisation/footer/#using-embedded-svgs
with open("github.svg") as fh:
    GITHUB_ICON_SVG = fh.read().strip()
html_theme_options = {
    "source_repository": "https://github.com/molmod/stacie",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/molmod/stacie",
            "html": GITHUB_ICON_SVG,
            "class": "",
        },
    ],
}

# -- Options for LaTeX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/latex.html#module-latex
latex_engine = "xelatex"
latex_elements = {
    "preamble": r"\input{macros.txt}",
}
latex_additional_files = ["macros.txt"]

# -- Configuration for autodoc extensions ---------------------------------

autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "members": None,
    "inherited-members": True,
    "ignore-module-all": True,
}
napoleon_use_rtype = False
add_module_names = False


def autodoc_skip_member(_app, _what, name, _obj, skip, _options):
    """Decide which parts to skip when building the API doc."""
    if name == "__init__":
        return False
    return skip


def setup(app):
    """Set up sphinx."""
    app.connect("autodoc-skip-member", autodoc_skip_member)


# -- Configuration of mathjax extension -----------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "mean": r"\operatorname{E}",
            "var": r"\operatorname{VAR}",
            "cov": r"\operatorname{COV}",
            "gdist": r"\operatorname{Gamma}",
        }
    },
}


# -- Configuration of bibtex extension -----------------------------------

bibtex_bibfiles = ["references.bib"]

# -- Utility functions -------------------------------------------------------


def _get_version_info():
    """Get the version as defined in pyproject.toml"""
    from setuptools_scm import Configuration
    from setuptools_scm._get_version_impl import _get_version

    config = Configuration.from_file("../../pyproject.toml", "./")
    verinfo = Version(_get_version(config, force_write_version_files=False))
    return f"{verinfo.major}.{verinfo.minor}", str(verinfo)


def _pre_build():
    """Things to be executed before Sphinx builds the documentation"""
    os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(
        key for key, value in autodoc_default_options.items() if value is True
    )
    main_api_doc(
        [
            "--append-syspath",
            "--output-dir=apidocs/",
            "../../src/stacie/",
            "--separate",
            "--doc-project=API Reference",
        ]
    )


version, release = _get_version_info()
_pre_build()
html_title = f"{project} {version}"
