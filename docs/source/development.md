# Developer Notes

## Development Install

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
pip install -e .[dev]
pytest -vv
cd docs
make html
make latexpdf
```

## Documentation

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


## How to Make a Release

- Mark the release in `docs/changelog.md`.
- Make a new commit and tag it with `vX.Y.Z`.
- Trigger the PyPI GitHub Action: `git push origin main --tags`.
