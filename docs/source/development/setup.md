# Development Setup

## Repository, Tests and Documentation Build

It is assumed that you have previously installed Python, Git, pre-commit and direnv.
A local installation for testing and development can be installed as follows:

```bash
git clone git@github.com:molmod/stacie.git
cd stacie
pre-commit install
python -m venv venv
echo 'source venv/bin/activate' > .envrc
direnv allow
pip install -U pip
pip install -e .[docs,tests]
pytest -vv
cd docs
./compile_html.sh
./compile_pdf.sh
```

## Documentation Live Preview

The documentation is created using [Sphinx](https://www.sphinx-doc.org/).

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
