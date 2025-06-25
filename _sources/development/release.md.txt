# How to Make a Release

## Software packaging and deployment

To make a new release of STACIE on PyPI, take the following steps

- Mark the release in `docs/changelog.md`.
- Make a new commit and tag it with `vX.Y.Z`.
- Trigger the PyPI GitHub Action: `git push origin main --tags`.

After having verified that the PyPI release was successful, update the feedstock on `conda-forge` accordingly:
<https://github.com/conda-forge/stacie-feedstock>.

## Documentation build and deployment

Take the following steps, starting from the root of the repository:

```bash
cd docs
./release_docs.sh
```

Use this script with caution, as it will push changes to the `gh-pages` branch.
