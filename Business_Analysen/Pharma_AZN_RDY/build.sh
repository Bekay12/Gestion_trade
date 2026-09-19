#!/usr/bin/env bash
# Baut out/analyse.pdf. Erzeugt zuvor die Datenschicht neu und bricht bei
# jedem Fehler ab - auch bei einem nicht belegten Zahlenwert.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out data
python3 scripts/kennzahlen.py
python3 scripts/pruefe_seiten.py
python3 scripts/reihe_azn.py > /dev/null
python3 scripts/reihe_rdy.py > /dev/null
python3 scripts/flow.py
python3 scripts/methode.py > /dev/null
python3 scripts/check_literals.py sections/*.tex
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out analyse.tex
echo "OK -> out/analyse.pdf"
