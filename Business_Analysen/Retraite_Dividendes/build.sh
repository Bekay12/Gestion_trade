#!/usr/bin/env bash
# Regenere la couche de donnees, controle, compile. S'arrete au premier defaut.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/berkam/Projets/Gestion_trade/.venv_new/bin/python
mkdir -p out data
$PY scripts/hypotheses.py --pruefen
$PY scripts/rechnung_retraite.py
# Rafraichit la colonne "Valeur" de docs/figures.md depuis data/kennzahlen.tex fraichement
# regenere ci-dessus (correctif infrastructure, tache 10, second passage): sans cet appel,
# l'inventaire peut se reperimer silencieusement a chaque fois qu'un calcul change une
# valeur (constate par le controleur pour \ZielJahrBasis, documente "non atteint" alors
# qu'il valait deja 2071). Non bloquant en cas d'echec inattendu: un inventaire perime
# n'empeche pas la compilation du rapport.
$PY scripts/gen_figures_doc.py || echo "(non bloquant: docs/figures.md non rafraichi)"
$PY scripts/check_literals.py sections/*.tex
latexmk -pdf -synctex=1 -halt-on-error -interaction=nonstopmode -outdir=out analyse.tex > out/build.log 2>&1 || { tail -40 out/build.log; exit 1; }
cp out/analyse.pdf out/plan.pdf
# Gardes de notes de bas de page (tache 10): duplication sur une meme page imprimee
# (check_footnote_pages.py, \QH/\QP en usage reel depuis la redaction des chapitres)
# et ancres \QL/\QR deplacees par un saut de page (check_footnote_groups.py).
# Necessitent la PDF -synctex=1 ci-dessus, deja produite. Bloquantes (README.md,
# tableau "Gardes actives au build"): revue finale, constat 4 - le || echo qui les
# rendait non bloquantes datait d'avant la redaction des chapitres et n'a plus lieu
# d'etre, le commentaire "sections encore des squelettes" etait perime.
echo "--- check_footnote_pages.py ---"
$PY scripts/check_footnote_pages.py
echo "--- check_footnote_groups.py ---"
$PY scripts/check_footnote_groups.py
echo "OK -> out/plan.pdf"
