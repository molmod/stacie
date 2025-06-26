# How to Make a Release

## Software packaging and deployment

- Mark the release in `docs/changelog.md`.
- Make a new commit and tag it with `vX.Y.Z`.
- Trigger the PyPI GitHub Action: `git push origin main --tags`.

## Documentation build and deployment

Take the following steps, starting from the root of the repository:

```bash
cd docs
./clean.sh
./compile_html.sh
cd ..
git checkout gh-pages
git rm -rf .
cp -r docs/build/html/* docs/build/html/.gitignore docs/build/html/.nojekyll .
git add .
git status
git commit --amend -m "Documentation update" -n
git push origin gh-pages --force
git checkout main
```
