# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import importlib.util
import inspect
import os
import pathlib

import sphinx
import sphinx.builders.latex.transforms
from intersphinx_registry import get_intersphinx_mapping
from packaging.version import Version
from sphinx.ext.apidoc import main as main_api_doc

# -- Utility functions -------------------------------------------------------


def _get_version_info():
    """Get the version as defined in pyproject.toml"""
    from setuptools_scm import get_version

    scm_version = get_version(root="../..", relative_to=__file__)
    verinfo = Version(scm_version)
    major_minor = f"{verinfo.major}.{verinfo.minor}"
    return major_minor, major_minor


def _get_source_ref():
    """Get the Git reference that the source links should point to."""
    from setuptools_scm import get_version

    scm_version = Version(get_version(root="../..", relative_to=__file__))
    if scm_version.local is not None:
        # A build from a working copy links to the exact commit,
        # because a branch keeps moving after the documentation is deployed.
        # The local segment of a setuptools-scm version is the commit hash,
        # prefixed with ``g`` and optionally followed by a dirty-tree marker.
        return scm_version.local.split(".")[0].removeprefix("g")
    if scm_version.is_devrelease:
        return "main"
    return f"v{scm_version.public}"


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

REPO_URL = "https://github.com/molmod/stacie"
REPO_BRANCH = "main"
REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

project = "STACIE"
copyright = "2024--2026, Gözdenur Toraman, Toon Verstraelen"  # noqa: A001
author = "Gözdenur Toraman, Toon Verstraelen"
version, release = _get_version_info()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # Third-party extensions
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_sitemap",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.inkscapeconverter",
    "sphinxext.opengraph",
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
# Sphinx emits a canonical link only when this is set,
# and the Open Graph configuration below derives the absolute page URLs from it.
html_baseurl = "https://molmod.github.io/stacie/"
# The documentation is deployed as a single unversioned and untranslated tree,
# so the sitemap URLs must not carry the language and version prefixes
# that sphinx-sitemap inserts by default.
sitemap_url_scheme = "{link}"
html_static_path = ["static"]
html_title = f"{project} {version}"
html_css_files = ["custom.css"]
html_extra_path = ["static/google7e7449498a5c0f0e.html"]
html_favicon = "static/stacie-logo-black.svg"
html_theme_options = {
    "dark_logo": "stacie-logo-white.svg",
    "light_logo": "stacie-logo-black.svg",
    "source_repository": REPO_URL,
    "source_branch": REPO_BRANCH,
    "source_directory": "docs/source/",
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

# -- Configuration of opengraph extension -------------------------------------
# https://sphinxext-opengraph.readthedocs.io/en/latest/

ogp_site_url = html_baseurl
ogp_site_name = f"{project} documentation"
# Setting an image also suppresses the per-page cards
# that the extension would otherwise generate with matplotlib.
ogp_image = html_baseurl + "_static/github_repo_card_light.png"
ogp_image_alt = (
    "Graphical summary of STACIE, "
    "listing the properties it estimates: diffusivity, ionic electrical conductivity, "
    "thermal conductivity, shear and bulk viscosity, "
    "exponential and integrated correlation times, and the error on the mean."
)
# Without this tag, X and Bluesky fall back to a small square preview.
ogp_custom_meta_tags = ['<meta name="twitter:card" content="summary_large_image">']
# The description is derived from the first content of each page,
# which on the theory and property pages is a LaTeX formula rather than prose.
# Such a description is acceptable in a social media card,
# where the title and the image carry the message,
# but as a search engine snippet it is worse than no description at all,
# because search engines tend to show this tag verbatim.
ogp_enable_meta_description = False

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
    "special-members": "__call__",
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
    ("py:class", "numpy._typing._array_like._ScalarT"),
    ("py:class", "numpy._typing._array_like.NDArray"),
    # A subscripted alias such as ``NDArray[float]`` nested inside another generic,
    # for example ``dict[str, NDArray[float]]``, is a ``types.GenericAlias``
    # whose ``__module__`` is the module defining the alias.
    # As of NumPy 2.5 with sphinx-autodoc-typehints 3.12, this is cross-referenced
    # under the name of its own type instead of the name of the alias.
    ("py:class", "numpy._typing._array_like.GenericAlias"),
    ("py:class", "numpy._typing.TypeAliasType"),
]
# Autodoc loads every module-level name, imported ones included,
# before it decides which to document,
# and sphinx-autodoc-typehints resolves the annotations of each name while doing so.
# The imports from NumPy therefore drag in annotations
# that reference names defined only under ``TYPE_CHECKING`` in NumPy,
# which cannot be resolved at build time.
# The downside of this setting is that it also hides genuinely broken forward references
# in STACIE's own annotations.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]
napoleon_use_rtype = False
napoleon_use_param = True

# -- Configuration of linkcode extension --------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

SOURCE_REF = _get_source_ref()


def linkcode_resolve(domain, info):
    """Get the URL of the source code of a documented Python object.

    Parameters
    ----------
    domain
        The language domain of the object, of which only ``"py"`` is resolved.
    info
        A dictionary with the keys ``"module"`` and ``"fullname"``,
        identifying the object to link to.

    Returns
    -------
    url
        The URL of the object on GitHub, or ``None`` when it has no source of its own.
    """
    if domain != "py" or not info["module"]:
        return None
    try:
        obj = importlib.import_module(info["module"])
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError):
        return None
    # A property hides its source behind the getter,
    # and a decorated function behind its wrapper.
    if isinstance(obj, property):
        obj = obj.fget
    obj = inspect.unwrap(obj) if callable(obj) else obj
    try:
        path = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        # Attributes and other objects without a source location of their own.
        return None
    if path is None:
        return None
    relpath = pathlib.Path(path).resolve().relative_to(REPO_ROOT.resolve())
    return f"{REPO_URL}/blob/{SOURCE_REF}/{relpath}#L{start}-L{start + len(lines) - 1}"


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

# -- Source links of the generated API pages ----------------------------------


def _get_api_module_path(pagename):
    """Get the repository path of the module documented by a generated API page.

    Parameters
    ----------
    pagename
        The name of the page being rendered.

    Returns
    -------
    relpath
        The path of the module relative to the root of the repository,
        or ``None`` when the page does not document a single module.
    """
    if not pagename.startswith("apidocs/"):
        return None
    try:
        spec = importlib.util.find_spec(pagename.removeprefix("apidocs/"))
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().relative_to(REPO_ROOT.resolve())


def _set_api_source_links(app, pagename, templatename, context, doctree):
    """Point the source buttons of the generated API pages at the documented module.

    The reStructuredText of these pages is written by ``sphinx-apidoc`` during the build
    and is not committed,
    so the links that the theme derives from the page source would be dead.
    The prose on such a page comes from the docstrings of one module,
    which makes that module the source a reader wants to view or edit.
    """
    relpath = _get_api_module_path(pagename)
    if relpath is None:
        if pagename.startswith("apidocs/"):
            # The theme renders neither button without a page source suffix.
            context["page_source_suffix"] = ""
        return
    context["theme_source_edit_link"] = f"{REPO_URL}/edit/{REPO_BRANCH}/{relpath}"
    context["theme_source_view_link"] = f"{REPO_URL}/blob/{REPO_BRANCH}/{relpath}?plain=true"


def setup(app):
    """Register the event handlers defined in this configuration file."""
    app.connect("html-page-context", _set_api_source_links)


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
