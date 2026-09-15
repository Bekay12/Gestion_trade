#!/usr/bin/env bash
# Baut out/hausarbeit.pdf. Bricht bei LaTeX-Fehlern ab.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out hausarbeit.tex
echo "OK -> out/hausarbeit.pdf"
