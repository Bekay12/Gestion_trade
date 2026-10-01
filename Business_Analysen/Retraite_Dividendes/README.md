# Retraite anticipée par les dividendes — plan pour Yann

Document de travail : peux-tu vivre de 3 500 € nets par mois (euros de 2026) grâce aux
dividendes, à quel âge et avec quelle épargne ? PDF en français, calculs reproductibles,
aucun chiffre tapé à la main dans le texte.

## Reconstruire le document

```bash
cd Business_Analysen/Retraite_Dividendes
./build.sh
```

`build.sh` enchaîne, et s'arrête au premier défaut : contrôle des hypothèses sourcées
(`scripts/hypotheses.py --pruefen`), calcul complet (`scripts/rechnung_retraite.py`, qui
régénère tous les fichiers sous `data/`, y compris `data/kennzahlen.tex` et
`data/zusammenfassung.json`), rafraîchissement de l'inventaire des figures
(`docs/figures.md`), garde « aucun chiffre tapé en dur » (`scripts/check_literals.py`),
compilation LaTeX (`latexmk`, journal dans `out/build.log`), gardes de notes de bas de page
(`check_footnote_pages.py`, `check_footnote_groups.py`). Le PDF final est
`out/plan.pdf` (copié depuis `out/analyse.tex`).

## Où sont les hypothèses

Toutes les valeurs sourcées (fiscalité, marché, salaire, rente légale) vivent dans
`data/quellen.json`, une entrée par clé avec `wert` (valeur), `quelle` (référence), `url`,
`abgerufen` (date de consultation) et `primaer` (source primaire ou secondaire). Rien
d'autre dans le document ne doit contenir un chiffre : `sections/*.tex` ne référence que des
macros de `data/kennzahlen.tex` ou d'autres fichiers `data/*.tex|csv`, vérifié par
`scripts/check_literals.py` à chaque build. Les lacunes de recherche (valeur introuvable,
repli utilisé) sont documentées dans `LUECKEN.md`, avec ce qui a été cherché et pourquoi le
repli a été retenu.

## Comment changer un paramètre

- **Le profil (scénario de référence)** : en tête de `scripts/rechnung_retraite.py`,
  `REFERENCE_STRATEGIE_NOM` (répartition, ex. `"MaisonSiebzig"` pour 70 % maison / 30 % ETF
  monde), `REFERENCE_QUOTE_NOM` (taux d'épargne, ex. `"Dreissig"` pour 30 %) et
  `REFERENCE_STARTKAPITAL_NOM` (capital de départ, ex. `"Mitte"`). Ce sont ces trois
  constantes qui alimentent toutes les macros `*Basis`/`*Yann` reprises au résumé ; les
  changer et relancer `./build.sh` recalcule tout le document en cohérence.
- **Une hypothèse chiffrée** (taux d'imposition, seuil, rendement, salaire…) : modifier
  l'entrée correspondante dans `data/quellen.json` (champ `wert`), avec la nouvelle source
  dans `quelle`/`url`/`abgerufen`, puis `./build.sh`.
- **Réserve d'urgence, taille du tampon Monte Carlo** : `RESERVE_URGENCE_MOIS` et
  `PUFFER_JAHRE_MACRO` dans `scripts/rechnung_retraite.py` (conventions du document, non
  sourcées, déclarées comme telles via `\annahme` dans le texte).

Après tout changement, relancer `./build.sh` en entier : les fichiers sous `data/` (macros,
CSV, tableaux `.tex`) sont régénérés, jamais édités à la main.

## Tests

```bash
PY=/home/berkam/Projets/Gestion_trade/.venv_new/bin/python
$PY -m pytest scripts/Test -q
```

ou, fichier par fichier (utile pour isoler une régression) :

```bash
for t in scripts/Test/test_*.py; do $PY "$t"; done
```

14 suites de tests, offline (aucun accès réseau, aucune clé API requise). À noter :
`test_rechnung.py` réimporte `scripts/rechnung_retraite.py` et régénère les fichiers `data/`
en conséquence ; relancer `./build.sh` après les tests si l'arbre de travail affiche des
différences dans `data/` avant un commit.

## La règle « aucun chiffre tapé »

Aucun nombre ne doit apparaître en dur dans `sections/*.tex` : chaque valeur citée dans le
texte est une macro LaTeX définie dans `data/kennzahlen.tex` (générée par
`rechnung_retraite.py` à partir du calcul, jamais reprise de mémoire) ou vient d'un tableau
`data/*.tex` généré de la même façon. `scripts/check_literals.py sections/*.tex` échoue la
compilation si un chiffre échappe à cette règle ; c'est la garde qui empêche un écart entre
le texte et le calcul qui le justifie.

## Gardes actives au build

| Garde | Rôle |
|---|---|
| `scripts/hypotheses.py --pruefen` | Vérifie que `data/quellen.json` est complet et cohérent avant tout calcul |
| `scripts/check_literals.py` | Refuse un chiffre tapé en dur dans `sections/*.tex` |
| `scripts/check_footnote_pages.py` | Refuse une citation de presse dupliquée sur une même page imprimée |
| `scripts/check_footnote_groups.py` | Refuse un groupe de notes de bas de page éclaté sur plusieurs pages par un saut de page |

`check_footnote_groups.py` ne s'applique qu'au mécanisme d'ancrage manuel `\QL{clé}` /
`\QR{clé}` (et `\label{fn:...}`, voir sa docstring) : regrouper deux citations identiques
sur la même page en une seule note, puis répéter la première via `\footnotemark`. Les
chapitres de ce document citent exclusivement via `\QH{clé}`/`\QP{id}`, qui n'utilisent
jamais ce mécanisme (chaque citation réimprime sa note intégralement, par construction) ;
le duplicata sur une même page que `\QH`/`\QP` peuvent produire est du ressort de
`check_footnote_pages.py`, pas de cette garde. Tant qu'aucune section n'utilise
`\QL`/`\QR`, `check_footnote_groups.py` rapporte donc normalement 0 groupe : ce n'est pas
un défaut de la garde, c'est un mécanisme que le document n'a pas eu besoin d'employer.

## Statut

Voir `MORGENBERICHT.md` pour le résumé en français des résultats, des écarts assumés au plan
initial, de ce qui manque encore et de la liste des commits de la nuit. Journal détaillé
tâche par tâche dans `PROGRESS.md` et lacunes de recherche dans `LUECKEN.md`.
