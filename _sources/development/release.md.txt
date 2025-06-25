# How to Make a Release

## Software packaging and deployment

- Mark the release in `docs/changelog.md`.
- Make a new commit and tag it with `vX.Y.Z`.
- Trigger the PyPI GitHub Action: `git push origin main --tags`.

## Documentation build and deployment

Take the following steps, starting from the root of the repository:

```bash
cd docs
./release_docs.sh
```

Use this script with caution, as it will push changes to the `gh-pages` branch.
