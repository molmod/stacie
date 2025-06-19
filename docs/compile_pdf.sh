#!/usr/bin/env bash
inkscape source/static/stacie-logo-black.svg \
  --actions="select-by-element:svg;object-set-attribute:width,128;object-set-attribute:viewBox,-1.04 -1.3 2.133 2.5" \
  -o source/static/stacie-logo-black.pdf \
  --export-overwrite
sphinx-build -M latexpdf source build -W --keep-going --nitpicky
