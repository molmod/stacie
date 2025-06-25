# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

import sphinx
import sphinx.builders.latex.transforms
from intersphinx_registry import get_intersphinx_mapping
from packaging.version import Version
from sphinx.ext.apidoc import main as main_api_doc

# -- Utility functions -------------------------------------------------------


def _get_version_info():
    """Get the version as defined in pyproject.toml"""
    from setuptools_scm import Configuration
    from setuptools_scm._get_version_impl import _get_version

    config = Configuration.from_file("../../pyproject.toml", "./")
    verinfo = Version(_get_version(config, force_write_version_files=False))
    return f"{verinfo.major}.{verinfo.minor}", str(verinfo)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "STACIE"
copyright = "2024--2025, Toon Verstraelen, Gozdenur Toraman"  # noqa: A001
author = "Toon Verstraelen, Gozdenur Toraman"
version, release = _get_version_info()

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
    "sphinx_tippy",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.inkscapeconverter",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = get_intersphinx_mapping(packages={"python", "numpy", "scipy"})
nitpicky = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


def load_footer_icons():
    """Include SVG footer icons as recommended in Furo template.

    See https://pradyunsg.me/furo/customisation/footer/#using-embedded-svgs
    """
    icon_links = [
        ("Center for Molecular Modeling", "cmm.svg", "https://molmod.ugent.be"),
        ("Soete Laboratory", "soete.svg", "https://www.ugent.be/ea/emsme/en/research/soete"),
        ("Ghent University", "ugent.svg", "https://ugent.be"),
        ("GitHub", "github.svg", "https://github.com/molmod/stacie"),
    ]
    footer_icons = []
    for name, path_svg, url in icon_links:
        with open(path_svg) as fh:
            svg = fh.read().strip()
        footer_icons.append({"name": name, "url": url, "html": svg, "class": ""})
    return footer_icons


html_theme = "furo"
html_static_path = ["static"]
html_title = f"{project} {version}"
html_css_files = ["custom.css"]
html_favicon = "static/stacie-logo-black.svg"
html_theme_options = {
    "dark_logo": "stacie-logo-white.svg",
    "light_logo": "stacie-logo-black.svg",
    "source_repository": "https://github.com/molmod/stacie",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": load_footer_icons(),
    "dark_css_variables": {
        "admonition-title-font-size": "1rem",
        "admonition-font-size": "1rem",
    },
    "light_css_variables": {
        "admonition-title-font-size": "1rem",
        "admonition-font-size": "1rem",
    },
}

# -- Options for LaTeX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/latex.html#module-latex
latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage[mathbf=sym,mathrm=sym]{unicode-math}
\usepackage{fontspec}
\setmainfont[Scale=1.2]{Libertinus Serif}
\setsansfont[Scale=1.2]{Libertinus Sans}
\setmonofont[Scale=0.85]{Cascadia Code}
\setmathfont[Scale=1.2]{XITS Math}
\setmathfont[Scale=1.2,range={\mathcal,\mathbfcal},StylisticSet=1]{XITS Math}
""",
    "fncychap": r"\usepackage[Sonny]{fncychap}",
    "papersize": "a4paper",
    "preamble": r"""
\input{macros.txt}
\usepackage[framemethod=TikZ]{mdframed}
\mdfdefinestyle{jupyquote}{
  usetwoside=false,
  topline=false,
  bottomline=false,
  rightline=false,
  innerleftmargin=12pt,
  leftmargin=12pt,
  innerrightmargin=0pt,
  rightmargin=0pt,
  innertopmargin=12pt,
  innerbottommargin=12pt,
  linewidth=1pt,
  linecolor=gray,
  skipabove=\topskip,
  skipbelow=\topskip
}
\renewenvironment{quote}{\begin{mdframed}[style=jupyquote]}{\end{mdframed}}
""",
    "sphinxsetup": "hmargin={2.2cm,2.2cm}, vmargin={3cm,3cm}",
}
latex_additional_files = ["macros.txt"]
latex_logo = "static/stacie-logo-black.pdf"


class DummyTransform(sphinx.builders.latex.transforms.BibliographyTransform):
    def run(self, **kwargs):
        pass


sphinx.builders.latex.transforms.BibliographyTransform = DummyTransform

# -- Configuration for myst-nb extensions -------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html
# https://myst-nb.readthedocs.io/en/v0.13.2/use/config-reference.html

myst_enable_extensions = [
    "amsmath",
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_merge_streams = True
exclude_patterns = ["conf.py"]
codeautolink_concat_default = True
nb_mime_priority_overrides = [("latex", "image/svg+xml", 15)]
myst_heading_anchors = 4

# -- Configuration for autodoc extensions -------------------------------------
# https://sphinx-autodoc2.readthedocs.io/en/latest/config.html
# https://github.com/tox-dev/sphinx-autodoc-typehints

add_module_names = False
autodoc_default_options = {
    "undoc-members": True,
    "members": None,
    "ignore-module-all": True,
}
autodoc_type_aliases = {
    "ArrayLike": ":py:class:`ArrayLike`",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
nitpick_ignore = [
    ("py:class", "ArrayLike"),
    ("py:class", "matplotlib.axes._axes.Axes"),
]
napoleon_use_rtype = False
napoleon_use_param = True

# -- Configuration of mathjax extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax

# These need to be synced with macros.tex
mathjax3_config = {
    "tex": {
        "macros": {
            "mean": r"\operatorname{E}",
            "var": r"\operatorname{VAR}",
            "std": r"\operatorname{STD}",
            "cov": r"\operatorname{COV}",
            "gdist": r"\operatorname{Gamma}",
        }
    },
}

# -- Configuration of bibtex extension ----------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#configuration

bibtex_bibfiles = ["references.bib"]

# -- Inform examples of data location -----------------------------------------
# This path is relative to the examples directory.
os.environ["DATA_ROOT"] = "../../data"

# -- Pre-build step to regenerate API documentation ---------------------------

# Note that autodoc2 is not used because it does not support NumPy style docstrings.
# See https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33


def _pre_build():
    """Things to be executed before Sphinx builds the documentation"""
    os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(
        key for key, value in autodoc_default_options.items() if value is True
    )
    main_api_doc(
        [
            "--output-dir=apidocs/",
            "../../src/stacie/",
            "--separate",
            "--force",
            "--remove-old",
            "--ext-autodoc",
            "--ext-intersphinx",
            "--ext-mathjax",
            "--ext-githubpages",
            "--doc-project=Application Programming Interface",
        ]
    )


_pre_build()
