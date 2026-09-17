#!/usr/bin/env bash
# Baut out/analyse.pdf. Bricht bei LaTeX-Fehlern ab.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out analyse.tex
echo "OK -> out/analyse.pdf"
