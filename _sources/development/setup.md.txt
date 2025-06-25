# Development Setup

## Repository, Tests and Documentation Build

It is assumed that you have previously installed
[Python](https://www.python.org/),
[uv](https://docs.astral.sh/uv/),
[Git](https://git-scm.com/),
[pre-commit](https://pre-commit.com/) and
[direnv](https://direnv.net/).
A local installation for testing and development can be installed as follows:

```bash
git clone git@github.com:molmod/stacie.git
cd stacie
uv sync --extra=dev
echo 'source .venv/bin/activate' > .envrc
direnv allow
pre-commit install
```

Tests are implemented with [pytest](https://docs.pytest.org/).
Run them as follows:

```bash
pytest -vv
```

Documentation is built with [Sphinx](https://www.sphinx-doc.org/).
Rebuild the documentation as follows:

```bash
cd docs
./compile_html.sh
./compile_pdf.sh
```

## Documentation Live Preview

Edit the documentation Markdown files with a live preview
by running the following command *in the root* of the repository:

```bash
cd docs
./preview_html.sh
```

Keep this running.
This will print a URL in the terminal that you open in your browser to preview the documentation.
Now you can edit the documentation and see the result as soon as you save a file.

Please, use [Semantic Line Breaks](https://sembr.org/)
as it facilitates reviewing documentation changes.
