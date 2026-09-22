#!/usr/bin/env bash
# Baut out/analyse.pdf. Erzeugt die Zahlenschicht neu und bricht bei jedem
# Fehler ab: nicht belegter Wert oder Zahl im Fliesstext.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out data
python3 scripts/reihe.py > /dev/null
python3 scripts/kennzahlen_seite.py > /dev/null
python3 scripts/prognose.py > /dev/null
python3 scripts/pruefe_seiten.py
python3 scripts/rechnung_gea.py
python3 scripts/check_literals.py sections/*.tex
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out analyse.tex > out/build.log 2>&1 || { tail -40 out/build.log; exit 1; }
echo "OK -> out/analyse.pdf"
