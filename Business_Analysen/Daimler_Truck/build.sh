#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out data
python3 scripts/pruefe_seiten.py
python3 scripts/rechnung_daimler.py
python3 scripts/check_literals.py sections/*.tex
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out analyse.tex > out/build.log 2>&1 || { tail -50 out/build.log; exit 1; }
echo "OK -> out/analyse.pdf"
