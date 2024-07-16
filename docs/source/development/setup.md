# Development Setup

## Repository, Tests and Documentation Build

It is assumed that you have installed Python, Git, pre-commit and direnv before.
A local installation for testing and development can be installed as follows:

```bash
git clone git@github.com:molmod/stacie.git
cd stacie
pre-commit install
python -m venv venv
echo 'source venv/bin/activate' > .envrc
direnv allow
pip install -U pip
pip install -e .[docs,tests,tools]
pytest -vv
cd docs
make html
make latexpdf
```

## Documentation Live Preview

The documentation is created using [Sphinx](https://www.sphinx-doc.org/).

Edit the documentation Markdown files with a live preview by running the following command *in the root* of the repository:

```bash
sphinx-reload docs/ --watch docs/source/ src/stacie/
```

Keep this running.
Your browser will open a new tab with the preview.
Now you can edit the documentation and see the result as soon as you save a file.

Please, use [Semantic Line Breaks](https://sembr.org/)
because it facilitates reviewing documentation changes.
