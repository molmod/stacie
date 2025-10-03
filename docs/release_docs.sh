#!/usr/bin/env bash
# Use this script with caution.
# If something goes wrong, your repo may be left in confusing state.

# Exit upon the first error, to avoid uploading failed builds
set -e

# Locally build the html and pdf docs
./clean.sh
./compile_html.sh
./compile_pdf.sh

# Switch to the gh-pages branch and replace its contents with the latest docs
cd ..
git checkout gh-pages
git rm -rf .
cp -r docs/build/html/* docs/build/html/.gitignore docs/build/html/.nojekyll .
cp docs/build/latex/stacie.pdf documentation.pdf
git add .
git status
git commit --amend -m "Documentation update" -n

# Upload the amended single commit
git push origin gh-pages --force

# Switch back to main
git checkout main
