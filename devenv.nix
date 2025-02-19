{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    pre-commit
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
    # Enforce local paths with #egg=
    # See https://github.com/jazzband/pip-tools/issues/204#issuecomment-550051424
    venv.requirements = ''
      -e file:.#egg=stacie

      attrs>=21.3.0
      cattrs>=22.2.0
      matplotlib>=3.7.0
      msgpack>=1.0.3
      numpy>=1.23.3
      path>=16.14.0
      scipy>=1.11.1

      furo
      intersphinx-registry
      jupyter
      jupyter-cache
      jupytext
      myst-nb
      myst-parser
      numpydoc
      packaging
      setuptools_scm
      sphinx
      sphinx_autodoc_typehints
      sphinx-codeautolink
      sphinxcontrib-bibtex
      sphinxcontrib-svg2pdfconverter
      sphinx-copybutton
      sphinx-autobuild

      pytest
      pytest-cov
      pytest-regtest
      pytest-xdist
      numdifftools

      celerite2
    '';
  };
}
