#!/usr/bin/env bash
sphinx-build -M html source build -W --keep-going --nitpicky

# Create a .gitignore file to avoid mistakes when deploying docs.
cat > build/html/.gitignore << EOF
.envrc
.local
.vscode
build
dist
docs
src
tests
tmp
tools
venv
EOF
