#!/usr/bin/env bash
# Use this script with caution.
# If something goes wrong, your repo may be left in confusing state.

# Exit upon the first error, to avoid uploading failed builds
set -e

./clean.sh
./compile_html.sh
./compile_pdf.sh
./upload_docs.sh
