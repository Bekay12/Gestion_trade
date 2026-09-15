# Hausarbeit Aumann AG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produire `out/hausarbeit.pdf` — une Hausarbeit de Master en allemand analysant Aumann AG, répondant aux 6 parties de `Docu_Hskl/Hausarbeit SoSe2026 Aumann.pdf`, avec la prose évaluative isolée en bleu pour relecture par l'utilisateur.

**Architecture:** LaTeX modulaire (KOMA `scrartcl`) avec une couche de données à source unique — tout chiffre est déclaré une seule fois comme macro dans `data/kennzahlen.tex` et consommé par le texte, les tableaux et les graphiques. La partie 6 (analyse de cash-flow, 3 IRR) est calculée par un script Python testé qui émet les corps de tableaux LaTeX. Un script de garde (`check_literals.py`) interdit tout littéral numérique dans `sections/*.tex`.

**Tech Stack:** TeX Live (koma-script, pgfplots, pgfplotstable, booktabs, siunitx v3.0.46, babel/ngerman, eurosym, microtype, csquotes, latexmk) · Python 3.10 stdlib seule · `pdftotext` (poppler) pour l'extraction · `curl` pour l'acquisition des sources.

## Global Constraints

- **Langue du document : allemand.** Les commentaires LaTeX, les noms de fichiers, les scripts et leurs docstrings sont en allemand ou en anglais neutre — jamais de français dans le PDF.
- **Séparateur décimal : virgule.** Configuré via siunitx v3 (`output-decimal-marker={,}`, `group-separator={.}`). Les macros de `data/kennzahlen.tex` stockent la valeur **brute au format anglais** (`138.2`) ; le formatage est fait par `\num{}`/`\MioEUR{}`.
- **Aucun littéral numérique dans `sections/*.tex`** hors millésimes (`19xx`/`20xx`), numéros de section et arguments de `\label`/`\ref`. Vérifié par `scripts/check_literals.py`.
- **Aucun chiffre, aucune source inventés.** Toute valeur introuvable dans une source est signalée explicitement dans le document plutôt que comblée.
- **Citations : notes de bas de page, Zitierweise allemande**, via les macros `\quelleGB`, `\quelleQM`, `\quelleWeb` du préambule. Toute source web porte un **Abrufdatum**.
- **Zones bleues :** `\bk{...}` = prose évaluative rédigée en allemand ; le matériel factuel qui la fonde est placé **juste au-dessus en commentaires LaTeX `%`** sous l'en-tête `% FAKTENBASIS <section>`. Jamais de puces factuelles visibles dans le PDF.
- **Solde d'ouverture de la partie 6 : `138.2` Mio € (Nettofinanzliquidität 31.12.2024)**, imposé par l'énoncé.
- **Hypothèses de la partie 6 (énoncé, non négociables) :** A = 35 / 15 / +16 à partir de 2028 · B = 30 / 15 / +14 à partir de 2027 · C = 60 / 20 / +10 à partir de 2027 (Investition Mio € sur 3 ans / Kapitalbindung Mio € / EBITDA additionnel Mio €).
- **Horizon IRR : 2026–2035**, EBITDA additionnel constant à partir de son année de démarrage, Kapitalbindung libérée en fin d'horizon.
- **Build :** `./build.sh` doit produire `out/hausarbeit.pdf` sans erreur LaTeX.
- Échéance de rendu : **04.08.2026**.

---

### Task 1: Squelette LaTeX, préambule et build

**Files:**
- Create: `hausarbeit.tex`, `preamble.tex`, `build.sh`
- Create: `sections/00-deckblatt.tex`, `sections/01-unternehmen.tex`, `sections/02-jahresabschluss.tex`, `sections/03-strategie.tex`, `sections/04-aktienkurs.tex`, `sections/05-handlungsoptionen.tex`, `sections/06-cashflow.tex`, `sections/90-quellen.tex`, `sections/91-ki-nutzung.tex`, `sections/92-erklaerung.tex`
- Create: `data/kennzahlen.tex`
- Test: compilation via `./build.sh`

**Interfaces:**
- Consumes: rien.
- Produces: les macros de préambule que toutes les tâches suivantes utilisent —
  `\bk{<text>}` (prose bleue), `\MioEUR{<raw>}` (montant en Mio €), `\Pct{<raw>}` (pourcentage),
  `\EURje{<raw>}` (montant par action), `\quelleGB{<jahr>}{<seite>}`, `\quelleQM{<bezeichnung>}{<seite>}`,
  `\quelleWeb{<urheber und titel>}{<url>}{<abrufdatum>}`, `\annahme{<text>}` (encadré d'hypothèse).
  La variable `\bkactive` permet de neutraliser le bleu avant rendu.

- [ ] **Step 1: Écrire `preamble.tex`**

```latex
% ===================================================================
% preamble.tex — Praeambel der Hausarbeit "Finanzwirtschaft fuer Ingenieure"
% ===================================================================
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[ngerman]{babel}
\usepackage{csquotes}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{siunitx}
\usepackage{eurosym}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage[hidelinks]{hyperref}
\usepackage{url}

\pgfplotsset{compat=1.18}

% --- Zahlenformat: deutsches Dezimalkomma ---------------------------
\sisetup{
  output-decimal-marker = {,},
  group-separator       = {.},
  group-minimum-digits  = 4,
  detect-weight         = true,
  detect-family         = true
}

% --- Betragsmakros --------------------------------------------------
% Die Rohwerte stehen in data/kennzahlen.tex im englischen Format (138.2).
% Die Formatierung erfolgt ausschliesslich hier.
\newcommand{\MioEUR}[1]{\num{#1}\,Mio.\,\euro}
\newcommand{\Pct}[1]{\num{#1}\,\%}
\newcommand{\EURje}[1]{\num{#1}\,\euro}

% --- Zitierweise: deutsche Fussnotenzitation ------------------------
\newcommand{\quelleGB}[2]{\footnote{Aumann AG, Gesch\"aftsbericht #1, S.~#2.}}
\newcommand{\quelleQM}[2]{\footnote{Aumann AG, #1, S.~#2.}}
\newcommand{\quelleWeb}[3]{\footnote{#1, \url{#2} (abgerufen am #3).}}

% --- Blaue Zonen: vom Verfasser zu pruefende Wertungsprosa ----------
% \bkactive auf 0 setzen, um die Blaufaerbung vor der Abgabe zu entfernen.
\newcommand{\bkactive}{1}
\newcommand{\bk}[1]{{\color{blue}#1}}

% --- Annahmenkasten -------------------------------------------------
\newcommand{\annahme}[1]{%
  \par\medskip\noindent\fbox{\parbox{0.97\linewidth}{\small\textbf{Annahme:} #1}}\par\medskip}
```

- [ ] **Step 2: Écrire `hausarbeit.tex`**

```latex
\documentclass[a4paper,11pt,DIV=12,parskip=half]{scrartcl}
\input{preamble}
\input{data/kennzahlen}

\begin{document}
\input{sections/00-deckblatt}
\tableofcontents
\clearpage
\input{sections/01-unternehmen}
\input{sections/02-jahresabschluss}
\input{sections/03-strategie}
\input{sections/04-aktienkurs}
\input{sections/05-handlungsoptionen}
\input{sections/06-cashflow}
\clearpage
\input{sections/90-quellen}
\input{sections/91-ki-nutzung}
\input{sections/92-erklaerung}
\end{document}
```

- [ ] **Step 3: Écrire `sections/00-deckblatt.tex` avec des placeholders visibles**

```latex
\begin{titlepage}
\centering
\vspace*{2cm}
{\Large Hochschule Kaiserslautern\par}
\vspace{0.5cm}
{\large Fachbereich Betriebswirtschaft\par}
\vspace{2.5cm}
{\Huge\bfseries Hausarbeit\par}
\vspace{0.8cm}
{\LARGE Finanzwirtschaft f\"ur Ingenieure\par}
\vspace{0.5cm}
{\Large Sommersemester 2026\par}
\vspace{2cm}
{\Large\bfseries Finanzwirtschaftliche Analyse der Aumann AG\par}
\vspace{3cm}
\begin{tabular}{ll}
Verfasser:in    & \bk{[Yann Bertin Kamdem Bobda]}\\
Matrikelnummer: & \bk{[885893]}\\
Studiengang:    & \bk{[Master - Elektrotechnik und Informaonstechnik (PO2021)]}\\
Betreuung:      & Prof.\ Dr.\ J\"urgen Bott, Katharina Moor\\
Abgabedatum:    & 4.~August 2026\\
\end{tabular}
\vfill
\end{titlepage}
```

- [ ] **Step 4: Créer les 9 autres fichiers de section avec leur titre uniquement**

Chaque fichier contient sa section vide, par exemple `sections/01-unternehmen.tex` :

```latex
\section{Beschreibung des Unternehmens}
\label{sec:unternehmen}
```

Titres à utiliser : `01` → `Beschreibung des Unternehmens` · `02` → `Auswertung des Jahresabschlusses 2024` · `03` → `Strategische Ma{\ss}nahmen im Gesch\"aftsbericht 2024` · `04` → `Beurteilung des Verlaufs des Aktienkurses` · `05` → `Beurteilung ausgew\"ahlter Handlungsoptionen` · `06` → `Grobe Flow-Analyse` · `90` → `Quellenverzeichnis` · `91` → `Hinweis zur Nutzung von KI-Werkzeugen` · `92` → `Eidesstattliche Erkl\"arung`.
Labels : `sec:jahresabschluss`, `sec:strategie`, `sec:aktienkurs`, `sec:optionen`, `sec:cashflow`, `sec:quellen`, `sec:ki`, `sec:erklaerung`.

- [ ] **Step 5: Créer `data/kennzahlen.tex` vide avec son en-tête**

```latex
% ===================================================================
% data/kennzahlen.tex
% EINZIGE Quelle aller Zahlenwerte des Dokuments.
% Rohwerte im englischen Format (Punkt als Dezimaltrenner) -
% die Formatierung mit Dezimalkomma erfolgt in preamble.tex.
% Jedes Makro traegt seine Quelle und Seitenzahl im Kommentar.
% ===================================================================
```

- [ ] **Step 6: Écrire `build.sh`**

```bash
#!/usr/bin/env bash
# Baut out/hausarbeit.pdf. Bricht bei LaTeX-Fehlern ab.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=out hausarbeit.tex
echo "OK -> out/hausarbeit.pdf"
```

- [ ] **Step 7: Rendre exécutable et compiler**

Run: `chmod +x build.sh && ./build.sh`
Expected: `OK -> out/hausarbeit.pdf`, aucune erreur LaTeX. Le PDF contient la page de titre avec les placeholders en bleu, le sommaire, et 9 sections vides.

- [ ] **Step 8: Vérifier le rendu du séparateur décimal et de l'euro**

Run: `pdftotext out/hausarbeit.pdf - | head -40`
Expected: la page de titre apparaît en texte. Puis ajouter temporairement `\MioEUR{138.2}` dans `sections/01-unternehmen.tex`, recompiler, et vérifier :

Run: `./build.sh && pdftotext out/hausarbeit.pdf - | grep -c '138,2'`
Expected: `1` (virgule décimale, pas `138.2`). Retirer ensuite la ligne temporaire et recompiler.

- [ ] **Step 9: Commit**

```bash
git add preamble.tex hausarbeit.tex build.sh sections/ data/kennzahlen.tex
git commit -m "feat: LaTeX-Grundgeruest, Praeambel und Build-Skript"
```

---

### Task 2: Acquisition et manifeste des sources Aumann

**Files:**
- Create: `scripts/fetch_reports.sh`
- Create: `refs/MANIFEST.md` (généré)
- Create: `refs/*.pdf` (gitignorés)

**Interfaces:**
- Consumes: rien.
- Produces: `refs/gb2022.pdf`, `refs/gb2023.pdf`, `refs/gb2024.pdf`, `refs/gb2025.pdf`, `refs/q1-2025.pdf`, `refs/h1-2025.pdf`, `refs/q3-2025.pdf`, `refs/q1-2026.pdf` — noms canoniques utilisés par toutes les tâches d'extraction. `refs/MANIFEST.md` fournit URL + Abrufdatum + SHA-256 + nombre de pages, et alimente la section `90-quellen.tex` de la Task 12.

- [ ] **Step 1: Écrire `scripts/fetch_reports.sh`**

```bash
#!/usr/bin/env bash
# Laedt die Aumann-Finanzberichte nach refs/ und schreibt refs/MANIFEST.md.
# Quelle: https://www.aumann.com/en/investor-relations/financial-reports
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p refs
BASE="https://www.aumann.com/fileadmin/templates/downloads/finanzberichte"
ABRUF="$(date +%d.%m.%Y)"

declare -A DOCS=(
  [gb2022]="2022_aag-geschaeftsbericht-hp_englisch.pdf"
  [gb2023]="2023_aag-geschaeftsbericht-hp_englisch.pdf"
  [gb2024]="2024_aag-geschaeftsbericht-eng-hp.pdf"
  [gb2025]="2025_AAG_Gesch%C3%A4ftsbericht_FINAL_eng.pdf"
  [q1-2025]="20250513-aumann-quartalsmitteilung-q1-2025-eng.pdf"
  [h1-2025]="20250814-aumann-halbjahresfinanzbericht-h1-2025-eng.pdf"
  [q3-2025]="20251113-aumann-quartalsmitteilung-q3-2025-eng.pdf"
  [q1-2026]="20260512-aumann-quartalsmitteilung-q1-2026-eng.pdf"
)

{
  echo "# Manifest der Aumann-Quelldokumente"
  echo
  echo "Abrufdatum aller Dokumente: ${ABRUF}"
  echo
  echo "| Datei | Quelle (URL) | Seiten | SHA-256 |"
  echo "|---|---|---|---|"
} > refs/MANIFEST.md

for key in "${!DOCS[@]}"; do
  url="${BASE}/${DOCS[$key]}"
  out="refs/${key}.pdf"
  echo "-> ${key}"
  curl -sSL --fail --max-time 120 "$url" -o "$out"
  if ! head -c 5 "$out" | grep -q '%PDF-'; then
    echo "FEHLER: ${out} ist kein PDF" >&2; exit 1
  fi
  pages="$(pdfinfo "$out" | awk '/^Pages:/{print $2}')"
  sha="$(sha256sum "$out" | cut -c1-16)"
  echo "| \`refs/${key}.pdf\` | <${url}> | ${pages} | \`${sha}\` |" >> refs/MANIFEST.md
done

sort -o refs/MANIFEST.md -k1,1 -t'|' --stable refs/MANIFEST.md 2>/dev/null || true
echo "OK -> refs/MANIFEST.md"
```

- [ ] **Step 2: Exécuter le script**

Run: `chmod +x scripts/fetch_reports.sh && ./scripts/fetch_reports.sh`
Expected: 8 lignes `-> <clé>`, puis `OK -> refs/MANIFEST.md`, sans `FEHLER`.

- [ ] **Step 3: Vérifier que les 8 PDF sont exploitables en texte**

Run:
```bash
for f in refs/*.pdf; do printf "%-20s %5s pages %8s chars\n" "$f" \
  "$(pdfinfo "$f" | awk '/^Pages:/{print $2}')" \
  "$(pdftotext "$f" - | wc -c)"; done
```
Expected: 8 lignes, chacune avec un nombre de pages > 20 pour les Geschäftsberichte et un nombre de caractères > 20000. Si un fichier retourne ~0 caractère, il est scanné en image : le signaler immédiatement et ne pas poursuivre l'extraction automatique sur ce fichier.

- [ ] **Step 4: Vérifier que le GB 2024 contient bien les ancres attendues**

Run: `pdftotext -layout refs/gb2024.pdf - | grep -niE 'order intake|net liquidity|operating EBITDA|dividend' | head -20`
Expected: plusieurs correspondances avec numéros de ligne. Ce sont les points d'entrée de la Task 4.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_reports.sh refs/MANIFEST.md
git commit -m "feat: Skript zum Laden der Aumann-Finanzberichte und Quellenmanifest"
```

---

### Task 3: Garde anti-littéraux `check_literals.py`

**Files:**
- Create: `scripts/check_literals.py`
- Test: `scripts/test_check_literals.py`

**Interfaces:**
- Consumes: rien.
- Produces: `find_literals(text: str) -> list[tuple[int, str]]` — retourne `(numéro de ligne 1-indexé, extrait fautif)` pour chaque littéral numérique interdit. CLI : `python3 scripts/check_literals.py sections/*.tex` sort en code 1 s'il trouve quelque chose. Utilisé comme porte de vérification dans les Tasks 4, 7, 8, 9, 10, 12.

- [ ] **Step 1: Écrire le test qui échoue**

```python
# scripts/test_check_literals.py
"""
Tests fuer check_literals.py - Wachhund gegen hartkodierte Zahlen
in sections/*.tex. Reine Standardbibliothek.
"""
import unittest
from check_literals import find_literals


class TestFindLiterals(unittest.TestCase):
    def test_flags_decimal_number(self):
        hits = find_literals(r"Der Umsatz betrug 246,8 Mio. EUR.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 1)

    def test_flags_large_integer(self):
        hits = find_literals(r"Der Auftragseingang lag bei 2135 TEUR.")
        self.assertEqual(len(hits), 1)

    def test_allows_year(self):
        hits = find_literals(r"Im Gesch\"aftsjahr 2024 sowie 2026 und 1998.")
        self.assertEqual(hits, [])

    def test_allows_macro_call(self):
        hits = find_literals(r"Der Umsatz betrug \MioEUR{\UmsatzZFV}.")
        self.assertEqual(hits, [])

    def test_allows_label_and_ref(self):
        hits = find_literals(r"\label{tab:cf-a} siehe Tabelle \ref{tab:cf-a}")
        self.assertEqual(hits, [])

    def test_ignores_comment_lines(self):
        hits = find_literals("% FAKTENBASIS: Umsatz 246,8 Mio. EUR\nText ohne Zahl.")
        self.assertEqual(hits, [])

    def test_ignores_inline_comment_tail(self):
        hits = find_literals(r"Text ohne Zahl. % Beleg: 246,8 Mio. EUR")
        self.assertEqual(hits, [])

    def test_respects_escaped_percent(self):
        hits = find_literals(r"Die Marge stieg um 12,5 \% gegen\"uber dem Vorjahr.")
        self.assertEqual(len(hits), 1)

    def test_reports_correct_line_number(self):
        text = "Erste Zeile.\nZweite Zeile.\nDritte Zeile mit 246,8 Mio.\n"
        hits = find_literals(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 3)

    def test_allows_small_integers(self):
        # Aufzaehlungen wie "drei Optionen (A, B, C)" oder "Abschnitt 5.4"
        # sollen nicht anschlagen.
        hits = find_literals(r"Siehe Abschnitt 5.4 sowie die 3 Optionen.")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd scripts && python3 -m unittest test_check_literals -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'check_literals'`

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
#!/usr/bin/env python3
"""
check_literals.py - Wachhund gegen hartkodierte Zahlen in sections/*.tex.

Regel des Projekts: jeder Zahlenwert des Dokuments stammt aus einem Makro
in data/kennzahlen.tex oder aus einer generierten Tabelle in data/.
Erlaubt bleiben Jahreszahlen (19xx/20xx), Gliederungsnummern, kleine
Ganzzahlen (< 100 ohne Nachkommastelle) sowie alles in Kommentaren
und in \\label{}/\\ref{}-Argumenten.

Aufruf: python3 scripts/check_literals.py sections/*.tex
Rueckgabewert 1, wenn Treffer gefunden wurden.
"""
import re
import sys

# \label{...} und \ref{...} samt Argument entfernen
_REF = re.compile(r"\\(?:label|ref|eqref|cite|input|include|url)\{[^}]*\}")
# Gliederungsnummern wie "5.4" oder "6.1.2"
_GLIEDERUNG = re.compile(r"(?<![\d,.])\d{1,2}(?:\.\d{1,2}){1,2}(?![\d,])")
# Jahreszahlen
_JAHR = re.compile(r"(?<![\d,.])(?:19|20)\d{2}(?![\d,.])")
# Verdaechtig: Dezimalzahl mit Komma, oder Ganzzahl mit >= 3 Stellen
_LITERAL = re.compile(r"(?<![\d,.])\d+,\d+|(?<![\d,.])\d{3,}(?![\d,.])")


def _strip_comment(line: str) -> str:
    """Entfernt den Kommentarteil einer Zeile; \\% bleibt erhalten."""
    out = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def find_literals(text: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Findet unerlaubte Zahlenliterale in einem LaTeX-Quelltext.

    Inputs:
        text (str): Inhalt einer .tex-Datei.

    Outputs:
        hits (list[tuple[int, str]]): (Zeilennummer 1-basiert, Treffertext)
    --------------------------------------------------------------------------
    """
    hits = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        line = _REF.sub(" ", line)
        line = _GLIEDERUNG.sub(" ", line)
        line = _JAHR.sub(" ", line)
        for m in _LITERAL.finditer(line):
            hits.append((lineno, m.group(0)))
    return hits


def main(argv: list) -> int:
    total = 0
    for path in argv[1:]:
        with open(path, encoding="utf-8") as fh:
            hits = find_literals(fh.read())
        for lineno, frag in hits:
            print(f"{path}:{lineno}: hartkodierte Zahl {frag!r}")
            total += 1
    if total:
        print(f"\n{total} Treffer - Werte gehoeren nach data/kennzahlen.tex.")
        return 1
    print("OK - keine hartkodierten Zahlen in den geprueften Dateien.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd scripts && python3 -m unittest test_check_literals -v`
Expected: `Ran 10 tests` / `OK`

- [ ] **Step 5: Vérifier le CLI sur les sections actuelles**

Run: `python3 scripts/check_literals.py sections/*.tex`
Expected: `OK - keine hartkodierten Zahlen in den geprueften Dateien.` (code retour 0)

- [ ] **Step 6: Commit**

```bash
git add scripts/check_literals.py scripts/test_check_literals.py
git commit -m "feat: Wachhund gegen hartkodierte Zahlen in den Abschnitten"
```

---

### Task 4: Extraction des Kennzahlen et partie 2 (Jahresabschluss 2024)

**Files:**
- Modify: `data/kennzahlen.tex`
- Create: `data/basis.json`
- Modify: `sections/02-jahresabschluss.tex`

**Interfaces:**
- Consumes: `refs/gb2023.pdf`, `refs/gb2024.pdf`, `refs/gb2025.pdf` (Task 2) ; macros `\MioEUR`, `\Pct`, `\EURje`, `\quelleGB` (Task 1) ; `scripts/check_literals.py` (Task 3).
- Produces:
  - Macros dans `data/kennzahlen.tex` (noms exacts, chiffres romains pour l'année) :
    `\UmsatzZFIII`, `\UmsatzZFIV`, `\UmsatzZFV` (Umsatz 2023 / 2024 / 2025),
    `\AuftragseingangZFIII`, `\AuftragseingangZFIV`,
    `\EbitdaOpZFIV`, `\EbitdaOpZFV`, `\EbitdaMargeZFIV`,
    `\NettogewinnZFIII`, `\NettogewinnZFIV`,
    `\DividendeJeAktieZFIV`, `\KursSilvesterZFIV`, `\DividendenrenditeZFIV`,
    `\NettoliquiditaetZFIV`, `\NettoliquiditaetZFV`.
  - `data/basis.json` : `{"umsatz_2025": <float>, "ebitda_op_2025": <float>, "nettoliquiditaet_2024": 138.2, "nettoliquiditaet_2025": <float>}` — consommé par `scripts/cashflow.py` (Task 5).

- [ ] **Step 1: Extraire les valeurs du GB 2024 avec leurs numéros de page**

Run, pour chaque grandeur, la commande donnant le contexte **et la page** :

```bash
for p in $(seq 1 $(pdfinfo refs/gb2024.pdf | awk '/^Pages:/{print $2}')); do
  pdftotext -layout -f $p -l $p refs/gb2024.pdf - 2>/dev/null \
    | grep -niE 'revenue|order intake|operating EBITDA|net profit|net liquidity|dividend per share' \
    | sed "s/^/p${p}:/"
done | head -60
```

Noter pour chaque grandeur : **la valeur** et **le numéro de page** du GB 2024. Répéter avec `refs/gb2023.pdf` (comparatif 2023) et `refs/gb2025.pdf` (valeurs 2025 et Nettoliquidität 31.12.2025).

Règle : si une grandeur n'est pas trouvable dans le rapport, **ne pas l'inventer** — écrire la macro avec la valeur `\textbf{n.\,v.}` et documenter l'absence dans la section, puis le signaler à l'utilisateur en fin de tâche.

- [ ] **Step 2: Remplir `data/kennzahlen.tex`**

Format à respecter — une macro par ligne, **valeur brute au format anglais**, source et page en commentaire de fin de ligne :

```latex
% --- Aufgabe 2: Jahresabschluss ------------------------------------
\newcommand{\UmsatzZFIII}{<wert>}            % GB 2024, S. <seite>
\newcommand{\UmsatzZFIV}{<wert>}             % GB 2024, S. <seite>
\newcommand{\AuftragseingangZFIII}{<wert>}   % GB 2024, S. <seite>
\newcommand{\AuftragseingangZFIV}{<wert>}    % GB 2024, S. <seite>
\newcommand{\EbitdaOpZFIV}{<wert>}           % GB 2024, S. <seite>
\newcommand{\EbitdaMargeZFIV}{<wert>}        % berechnet: EbitdaOpZFIV / UmsatzZFIV
\newcommand{\NettogewinnZFIII}{<wert>}       % GB 2024, S. <seite>
\newcommand{\NettogewinnZFIV}{<wert>}        % GB 2024, S. <seite>
\newcommand{\DividendeJeAktieZFIV}{<wert>}   % GB 2024, S. <seite>
\newcommand{\KursSilvesterZFIV}{<wert>}      % GB 2024, S. <seite> (Schlusskurs 30.12.2024)
\newcommand{\DividendenrenditeZFIV}{<wert>}  % berechnet: Dividende / Kurs
\newcommand{\NettoliquiditaetZFIV}{138.2}    % GB 2024, S. <seite>
% --- Basiswerte 2025 (fuer Aufgabe 6) ------------------------------
\newcommand{\UmsatzZFV}{<wert>}              % GB 2025, S. <seite>
\newcommand{\EbitdaOpZFV}{<wert>}            % GB 2025, S. <seite>
\newcommand{\NettoliquiditaetZFV}{<wert>}    % GB 2025, S. <seite>
% --- Abgeleitete Veraenderungsraten --------------------------------
\newcommand{\UmsatzDeltaZFIV}{<wert>}        % berechnet: (U24-U23)/U23 in %
\newcommand{\AuftragseingangDeltaZFIV}{<wert>}  % berechnet, Aufgabe nennt -41,1 %
\newcommand{\NettogewinnDeltaZFIV}{<wert>}   % berechnet: (N24-N23)/N23 in %
```

- [ ] **Step 3: Contrôle de cohérence des taux calculés**

Run (remplacer les valeurs par celles saisies) :
```bash
python3 - <<'PY'
u23, u24 = <UmsatzZFIII>, <UmsatzZFIV>
a23, a24 = <AuftragseingangZFIII>, <AuftragseingangZFIV>
e24 = <EbitdaOpZFIV>
print("Umsatz-Delta   %", round((u24-u23)/u23*100, 1))
print("Auftrags-Delta %", round((a24-a23)/a23*100, 1))
print("EBITDA-Marge   %", round(e24/u24*100, 1))
PY
```
Expected: `Auftrags-Delta % -41.1` — le chiffre annoncé par l'énoncé. **Si l'écart dépasse 0,2 point, s'arrêter** : soit la valeur extraite est fausse, soit l'énoncé se réfère à un autre périmètre ; investiguer dans le GB avant de continuer. Reporter les trois résultats dans les macros `\...Delta...` de l'étape 2.

- [ ] **Step 4: Écrire `data/basis.json`**

```json
{
  "umsatz_2025": <wert>,
  "ebitda_op_2025": <wert>,
  "nettoliquiditaet_2024": 138.2,
  "nettoliquiditaet_2025": <wert>,
  "_quelle": "Aumann AG, Geschaeftsbericht 2025; Nettoliquiditaet 2024 laut Aufgabenstellung"
}
```

- [ ] **Step 5: Rédiger `sections/02-jahresabschluss.tex` — formule posée puis calcul déroulé**

Modèle à appliquer pour 2.1 à 2.6 (ici 2.3, à reproduire pour chaque sous-question) :

```latex
\subsection{H\"ohe der EBITDA-Marge (in Prozent vom Umsatz)}
\label{subsec:ebitda-marge}

Die EBITDA-Marge setzt das operative EBITDA ins Verh\"altnis zum Umsatz:
\begin{equation}
\label{eq:ebitda-marge}
\text{EBITDA-Marge} = \frac{\text{operatives EBITDA}}{\text{Umsatz}}
\end{equation}
Mit den Werten des Gesch\"aftsjahres 2024\quelleGB{2024}{<seite>} ergibt sich
\[
\text{EBITDA-Marge}_{2024}
= \frac{\MioEUR{\EbitdaOpZFIV}}{\MioEUR{\UmsatzZFIV}}
= \Pct{\EbitdaMargeZFIV}.
\]
```

Couvrir dans l'ordre : 2.1 Umsatz + Δ · 2.2 Auftragseingang + Δ · 2.3 EBITDA-Marge · 2.4 operatives EBITDA · 2.5 Nettogewinn + Δ · 2.6 Dividende et Dividendenrendite (= `\DividendeJeAktieZFIV` / `\KursSilvesterZFIV`, cours du 31.12.2024). La sous-section 2.7 (courbe de cours) est ajoutée en Task 6 — poser ici uniquement `\subsection{Entwicklung des Aktienkurses 2022 bis 2025}\label{subsec:kursverlauf}`.

Ajouter un tableau `booktabs` récapitulatif 2023 vs 2024 utilisant exclusivement les macros.

- [ ] **Step 6: Vérifier l'absence de littéraux**

Run: `python3 scripts/check_literals.py sections/02-jahresabschluss.tex`
Expected: `OK - keine hartkodierten Zahlen in den geprueften Dateien.`

- [ ] **Step 7: Compiler**

Run: `./build.sh`
Expected: `OK -> out/hausarbeit.pdf`, aucune erreur.

- [ ] **Step 8: Vérifier que les valeurs apparaissent bien dans le PDF**

Run: `pdftotext out/hausarbeit.pdf - | grep -E 'EBITDA-Marge|Dividendenrendite' -A2 | head -20`
Expected: les valeurs formatées avec virgule décimale apparaissent.

- [ ] **Step 9: Commit**

```bash
git add data/kennzahlen.tex data/basis.json sections/02-jahresabschluss.tex
git commit -m "feat: Kennzahlen 2023-2025 extrahiert und Aufgabe 2 ausgearbeitet"
```

---

### Task 5: `cashflow.py` — tableaux 6.1 et 6.3 calculés

**Files:**
- Create: `scripts/cashflow.py`
- Test: `scripts/test_cashflow.py`
- Create (générés): `data/cf_a.tex`, `data/cf_b.tex`, `data/cf_c.tex`, `data/vergleich2028.tex`, `data/irr_sensitivitaet.tex`

**Interfaces:**
- Consumes: `data/basis.json` (Task 4).
- Produces:
  - `schedule(key: str) -> list[dict]` — 3 lignes (2026, 2027, 2028) avec clés `jahr`, `investition`, `op_cf`, `liquiditaet`, `bemerkung`.
  - `full_cashflows(key: str, horizon_end: int = 2035) -> list[float]` — vecteur de flux 2026→2035, Kapitalbindung libérée la dernière année.
  - `npv(cashflows: list[float], rate: float) -> float`
  - `irr(cashflows: list[float]) -> float` — bissection, renvoie un taux décimal (0.12 = 12 %).
  - `ebitda_marge_2028(key: str, umsatz: float, ebitda_basis: float) -> float` — en %.
  - CLI `python3 scripts/cashflow.py` : écrit les 5 fichiers `data/*.tex`.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
# scripts/test_cashflow.py
"""
Tests fuer cashflow.py - grobe Cash-Flow-Analyse der Optionen A, B, C.
Die Erwartungswerte folgen unmittelbar aus den Annahmen der
Aufgabenstellung und dem Investitionsprofil des Spezifikationsdokuments.
Reine Standardbibliothek.
"""
import unittest
import cashflow as cf


class TestAnnahmen(unittest.TestCase):
    def test_investitionssummen(self):
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["A"]["investment"].values()), 35.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["B"]["investment"].values()), 30.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["C"]["investment"].values()), 60.0, places=2)

    def test_kapitalbindungssummen(self):
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["A"]["working_capital"].values()), 15.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["B"]["working_capital"].values()), 15.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["C"]["working_capital"].values()), 20.0, places=2)

    def test_ebitda_startjahre(self):
        self.assertEqual(cf.ASSUMPTIONS["A"]["ebitda_start"], 2028)
        self.assertEqual(cf.ASSUMPTIONS["B"]["ebitda_start"], 2027)
        self.assertEqual(cf.ASSUMPTIONS["C"]["ebitda_start"], 2027)

    def test_akquisition_zahlt_kaufpreis_bei_vollzug(self):
        # Option C: Kaufpreis 45 Mio. EUR im Vollzugsjahr 2026,
        # danach je 7,5 Mio. EUR Integration.
        self.assertAlmostEqual(cf.ASSUMPTIONS["C"]["investment"][2026], 45.0, places=2)


class TestSchedule(unittest.TestCase):
    def _liq(self, key):
        return [round(r["liquiditaet"], 1) for r in cf.schedule(key)]

    def test_liquiditaet_option_a(self):
        # 138,2 -11,7 = 126,5 | -11,7 -7,5 = 107,3 | -11,6 +16 -7,5 = 104,2
        self.assertEqual(self._liq("A"), [126.5, 107.3, 104.2])

    def test_liquiditaet_option_b(self):
        # 138,2 -10 -7,5 = 120,7 | -10 +14 -7,5 = 117,2 | -10 +14 = 121,2
        self.assertEqual(self._liq("B"), [120.7, 117.2, 121.2])

    def test_liquiditaet_option_c(self):
        # 138,2 -45 -20 = 73,2 | -7,5 +10 = 75,7 | -7,5 +10 = 78,2
        self.assertEqual(self._liq("C"), [73.2, 75.7, 78.2])

    def test_schedule_hat_drei_jahre(self):
        for key in ("A", "B", "C"):
            rows = cf.schedule(key)
            self.assertEqual([r["jahr"] for r in rows], [2026, 2027, 2028])

    def test_jede_zeile_hat_bemerkung(self):
        for key in ("A", "B", "C"):
            for row in cf.schedule(key):
                self.assertTrue(row["bemerkung"].strip())

    def test_startliquiditaet_ist_vorgabe(self):
        self.assertAlmostEqual(cf.OPENING_LIQUIDITY, 138.2, places=2)


class TestIRR(unittest.TestCase):
    def test_irr_einfacher_fall(self):
        self.assertAlmostEqual(cf.irr([-100.0, 110.0]), 0.10, places=4)

    def test_irr_zweiperiodig(self):
        # -100 heute, +60 und +60 -> IRR ca. 13,07 %
        self.assertAlmostEqual(cf.irr([-100.0, 60.0, 60.0]), 0.13066, places=4)

    def test_npv_am_irr_ist_null(self):
        for key in ("A", "B", "C"):
            flows = cf.full_cashflows(key)
            r = cf.irr(flows)
            self.assertAlmostEqual(cf.npv(flows, r), 0.0, places=6)

    def test_npv_bei_null_prozent_ist_summe(self):
        flows = [-10.0, 5.0, 8.0]
        self.assertAlmostEqual(cf.npv(flows, 0.0), 3.0, places=6)


class TestFullCashflows(unittest.TestCase):
    def test_laenge_entspricht_horizont(self):
        flows = cf.full_cashflows("A", horizon_end=2035)
        self.assertEqual(len(flows), 10)  # 2026..2035

    def test_kapitalbindung_wird_am_ende_freigesetzt(self):
        flows_a = cf.full_cashflows("A", horizon_end=2035)
        # Letztes Jahr: laufendes EBITDA 16 + freigesetzte Kapitalbindung 15
        self.assertAlmostEqual(flows_a[-1], 31.0, places=2)

    def test_erstes_jahr_entspricht_investition_option_b(self):
        # B 2026: -10 Investition -7,5 Kapitalbindung, kein EBITDA
        self.assertAlmostEqual(cf.full_cashflows("B")[0], -17.5, places=2)

    def test_alle_optionen_haben_positiven_irr_auf_zehn_jahren(self):
        for key in ("A", "B", "C"):
            self.assertGreater(cf.irr(cf.full_cashflows(key)), 0.0)


class TestEbitdaMarge(unittest.TestCase):
    def test_marge_formel(self):
        # (Basis-EBITDA 20 + Zuwachs 16) / Umsatz 300 = 12 %
        got = cf.ebitda_marge_2028("A", umsatz=300.0, ebitda_basis=20.0)
        self.assertAlmostEqual(got, 12.0, places=4)


class TestLatexAusgabe(unittest.TestCase):
    def test_tabellenkoerper_hat_drei_zeilen(self):
        body = cf.emit_schedule_table("A")
        self.assertEqual(body.count(r"\\"), 3)

    def test_tabellenkoerper_nutzt_dezimalpunkt_fuer_siunitx(self):
        # siunitx formatiert selbst; im Quelltext steht der englische Punkt.
        body = cf.emit_schedule_table("A")
        self.assertIn("126.5", body)
        self.assertNotIn("126,5", body)

    def test_vergleichstabelle_nennt_net_debt_nur_fuer_c(self):
        body = cf.emit_comparison_table(umsatz=300.0, ebitda_basis=20.0)
        self.assertEqual(body.count("nicht relevant"), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd scripts && python3 -m unittest test_cashflow -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cashflow'`

- [ ] **Step 3: Écrire l'implémentation**

```python
#!/usr/bin/env python3
"""
cashflow.py - Grobe Cash-Flow-Analyse der Optionen A, B und C (Aufgabe 6).

Erzeugt die Tabellenkoerper fuer Aufgabe 6.1 (je eine Tabelle pro Option),
die Vergleichstabelle 6.3 und die IRR-Sensitivitaet. Reine Standardbibliothek,
deterministische Ausgabe.

Sicht: inkrementell. Der ausgewiesene operative Cash-Flow ist der zusaetzliche
Cash-Flow der jeweiligen Option (zusaetzliches EBITDA abzueglich Aufbau der
Kapitalbindung), nicht der Konzern-Cash-Flow.
"""
import json
import os

# Nettofinanzliquiditaet zum 31.12.2024 laut Aufgabenstellung (Mio. EUR)
OPENING_LIQUIDITY = 138.2

# Horizont der Renditerechnung
HORIZON_START = 2026
HORIZON_END = 2035

ASSUMPTIONS = {
    "A": {
        "name": "Next Automation / Diversifikation",
        # 35 Mio. EUR gleichmaessig ueber drei Jahre
        "investment": {2026: 11.7, 2027: 11.7, 2028: 11.6},
        # Kapitalbindung 15 Mio. EUR, Aufbau vor dem Ergebnisbeitrag ab 2028
        "working_capital": {2027: 7.5, 2028: 7.5},
        "ebitda_delta": 16.0,
        "ebitda_start": 2028,
        "remarks": {
            2026: "Aufbau von Vertrieb und Applikationstechnik, erste Markterschliessung",
            2027: "Markterschliessung Clean Tech, Aerospace, Life Sciences; Aufbau Kapitalbindung",
            2028: "Erster voller EBITDA-Beitrag; Abschluss des Investitionsprogramms",
        },
    },
    "B": {
        "name": "Batterie- und Brennstoffzellen-Produktionstechnik",
        "investment": {2026: 10.0, 2027: 10.0, 2028: 10.0},
        # Ergebnisbeitrag ab 2027, Kapitalbindung entsprechend frueher
        "working_capital": {2026: 7.5, 2027: 7.5},
        "ebitda_delta": 14.0,
        "ebitda_start": 2027,
        "remarks": {
            2026: "Schwerpunkt F&E, Aufbau Elektroden- und MEA-Fertigungskompetenz",
            2027: "Markteintritt; erster EBITDA-Beitrag",
            2028: "Skalierung der Linien; voller EBITDA-Beitrag",
        },
    },
    "C": {
        "name": "Anorganisches Wachstum / gezielte Akquisition",
        # Kaufpreis im Vollzugsjahr, danach Integrationsaufwand
        "investment": {2026: 45.0, 2027: 7.5, 2028: 7.5},
        # Kapitalbindung des erworbenen Geschaefts faellt mit dem Vollzug an
        "working_capital": {2026: 20.0},
        "ebitda_delta": 10.0,
        "ebitda_start": 2027,
        "remarks": {
            2026: "Vollzug der Akquisition; Kaufpreiszahlung und Uebernahme des Umlaufvermoegens",
            2027: "Integration; erster EBITDA-Beitrag aus dem erworbenen Geschaeft",
            2028: "Abschluss der Integration; Hebung der Synergien",
        },
    },
}

_ORDER = ("A", "B", "C")


def _ebitda(key: str, year: int) -> float:
    """Zusaetzliches EBITDA der Option im gegebenen Jahr (Mio. EUR)."""
    opt = ASSUMPTIONS[key]
    return opt["ebitda_delta"] if year >= opt["ebitda_start"] else 0.0


def _working_capital(key: str, year: int) -> float:
    """Aufbau der Kapitalbindung im gegebenen Jahr (Mio. EUR, Mittelabfluss)."""
    return ASSUMPTIONS[key]["working_capital"].get(year, 0.0)


def _investment(key: str, year: int) -> float:
    """Investitionsauszahlung im gegebenen Jahr (Mio. EUR)."""
    return ASSUMPTIONS[key]["investment"].get(year, 0.0)


def operating_cf(key: str, year: int) -> float:
    """Inkrementeller operativer Cash-Flow: EBITDA-Zuwachs minus Kapitalbindung."""
    return _ebitda(key, year) - _working_capital(key, year)


def schedule(key: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Baut die Jahrestabelle 2026-2028 fuer eine Option (Aufgabe 6.1).

    Inputs:
        key (str): "A", "B" oder "C".

    Outputs:
        rows (list[dict]): je Jahr die Schluessel jahr, investition, op_cf,
                           liquiditaet, bemerkung.
    --------------------------------------------------------------------------
    """
    opt = ASSUMPTIONS[key]
    liq = OPENING_LIQUIDITY
    rows = []
    for year in (2026, 2027, 2028):
        inv = _investment(key, year)
        ocf = operating_cf(key, year)
        liq = liq - inv + ocf
        rows.append({
            "jahr": year,
            "investition": inv,
            "op_cf": ocf,
            "liquiditaet": liq,
            "bemerkung": opt["remarks"][year],
        })
    return rows


def full_cashflows(key: str, horizon_end: int = HORIZON_END) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Zahlungsreihe der Option ueber den Renditehorizont. Das zusaetzliche
        EBITDA laeuft ab seinem Startjahr konstant weiter; die Kapitalbindung
        wird im letzten Jahr freigesetzt.

    Inputs:
        key (str): "A", "B" oder "C".
        horizon_end (int): letztes Jahr des Horizonts.

    Outputs:
        flows (list[float]): ein Wert je Jahr von HORIZON_START bis horizon_end.
    --------------------------------------------------------------------------
    """
    total_wc = sum(ASSUMPTIONS[key]["working_capital"].values())
    flows = []
    for year in range(HORIZON_START, horizon_end + 1):
        flow = _ebitda(key, year) - _working_capital(key, year) - _investment(key, year)
        if year == horizon_end:
            flow += total_wc
        flows.append(flow)
    return flows


def npv(cashflows: list, rate: float) -> float:
    """Kapitalwert der Zahlungsreihe; das erste Element liegt in t = 0."""
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows))


def irr(cashflows: list, lo: float = -0.95, hi: float = 5.0, tol: float = 1e-10) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Interner Zinsfuss der Zahlungsreihe per Bisektion.

    Inputs:
        cashflows (list[float]): Zahlungsreihe ab t = 0.
        lo, hi (float): Klammer der Nullstellensuche.
        tol (float): Abbruchbreite des Intervalls.

    Outputs:
        rate (float): Dezimalsatz, 0.12 entspricht 12 %.
    --------------------------------------------------------------------------
    """
    f_lo, f_hi = npv(cashflows, lo), npv(cashflows, hi)
    if f_lo * f_hi > 0:
        raise ValueError("Kein Vorzeichenwechsel im Suchintervall - IRR nicht bestimmbar.")
    for _ in range(500):
        mid = (lo + hi) / 2.0
        f_mid = npv(cashflows, mid)
        if abs(f_mid) < 1e-12 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def ebitda_marge_2028(key: str, umsatz: float, ebitda_basis: float) -> float:
    """EBITDA-Marge Ende 2028 in Prozent, Umsatzbasis konstant gehalten."""
    return (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"]) / umsatz * 100.0


def net_debt_to_ebitda(key: str, ebitda_basis: float) -> float:
    """
    Net Debt / EBITDA Ende 2028. Positive Nettoliquiditaet ergibt einen
    negativen Verschuldungsgrad (Nettoguthaben).
    """
    liq = schedule(key)[-1]["liquiditaet"]
    return -liq / (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"])


# --- LaTeX-Ausgabe ---------------------------------------------------

def _n(value: float, digits: int = 1) -> str:
    """Zahl im englischen Format fuer siunitx (\\num formatiert spaeter)."""
    return f"{value:.{digits}f}"


def emit_schedule_table(key: str) -> str:
    """Tabellenkoerper fuer Aufgabe 6.1 (ohne \\toprule / \\bottomrule)."""
    lines = []
    for row in schedule(key):
        lines.append(
            f"{row['jahr']} & \\num{{{_n(row['investition'])}}} "
            f"& \\num{{{_n(row['op_cf'])}}} "
            f"& \\num{{{_n(row['liquiditaet'])}}} "
            f"& {row['bemerkung']} \\\\"
        )
    return "\n".join(lines) + "\n"


def emit_comparison_table(umsatz: float, ebitda_basis: float) -> str:
    """Tabellenkoerper fuer Aufgabe 6.3 (Vergleich Ende 2028)."""
    inv = {k: sum(ASSUMPTIONS[k]["investment"].values()) for k in _ORDER}
    rows = []
    rows.append("Kumulierte Investition (Mio.\\,\\euro) & "
                + " & ".join(f"\\num{{{_n(inv[k])}}}" for k in _ORDER) + " \\\\")
    rows.append("EBITDA-Zuwachs ab Jahr & "
                + " & ".join(
                    f"\\num{{{_n(ASSUMPTIONS[k]['ebitda_delta'])}}} ab {ASSUMPTIONS[k]['ebitda_start']}"
                    for k in _ORDER) + " \\\\")
    rows.append("EBITDA-Marge Ende 2028 & "
                + " & ".join(
                    f"\\num{{{_n(ebitda_marge_2028(k, umsatz, ebitda_basis))}}}\\,\\%"
                    for k in _ORDER) + " \\\\")
    rows.append("IRR (rd., Horizont 2026--2035) & "
                + " & ".join(
                    f"\\num{{{_n(irr(full_cashflows(k)) * 100.0)}}}\\,\\%" for k in _ORDER) + " \\\\")
    rows.append("Liquidit\\\"atsreserve Ende 2028 (Mio.\\,\\euro) & "
                + " & ".join(
                    f"\\num{{{_n(schedule(k)[-1]['liquiditaet'])}}}" for k in _ORDER) + " \\\\")
    rows.append("Net Debt/EBITDA & nicht relevant & nicht relevant & "
                + f"\\num{{{_n(net_debt_to_ebitda('C', ebitda_basis), 2)}}} \\\\")
    return "\n".join(rows) + "\n"


def emit_irr_sensitivity() -> str:
    """Tabellenkoerper der IRR-Sensitivitaet nach Horizontlaenge."""
    rows = []
    for end in (2030, 2035, 2040):
        cells = " & ".join(
            f"\\num{{{_n(irr(full_cashflows(k, horizon_end=end)) * 100.0)}}}\\,\\%"
            for k in _ORDER)
        rows.append(f"{end - HORIZON_START + 1} Jahre (bis {end}) & {cells} \\\\")
    return "\n".join(rows) + "\n"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    with open(os.path.join(root, "data", "basis.json"), encoding="utf-8") as fh:
        basis = json.load(fh)
    umsatz = float(basis["umsatz_2025"])
    ebitda_basis = float(basis["ebitda_op_2025"])

    outputs = {
        "cf_a.tex": emit_schedule_table("A"),
        "cf_b.tex": emit_schedule_table("B"),
        "cf_c.tex": emit_schedule_table("C"),
        "vergleich2028.tex": emit_comparison_table(umsatz, ebitda_basis),
        "irr_sensitivitaet.tex": emit_irr_sensitivity(),
    }
    header = "% Automatisch erzeugt von scripts/cashflow.py - nicht von Hand aendern.\n"
    for name, body in outputs.items():
        path = os.path.join(root, "data", name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + body)
        print(f"OK -> data/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd scripts && python3 -m unittest test_cashflow -v`
Expected: `Ran 22 tests` / `OK`

- [ ] **Step 5: Générer les fichiers de données**

Run: `python3 scripts/cashflow.py`
Expected: 5 lignes `OK -> data/<nom>.tex`

- [ ] **Step 6: Vérifier le contenu généré**

Run: `cat data/cf_c.tex`
Expected: 3 lignes, la première commençant par `2026 & \num{45.0} & \num{-20.0} & \num{73.2}` — le creux de liquidité de l'option C.

- [ ] **Step 7: Commit**

```bash
git add scripts/cashflow.py scripts/test_cashflow.py data/cf_a.tex data/cf_b.tex data/cf_c.tex data/vergleich2028.tex data/irr_sensitivitaet.tex
git commit -m "feat: Cash-Flow-Analyse der Optionen A, B, C mit IRR-Berechnung"
```

---

### Task 6: Série de cours et graphique 2022–2025

**Files:**
- Create: `scripts/fetch_kurs.py`
- Create: `data/aktienkurs.csv`
- Modify: `sections/02-jahresabschluss.tex` (sous-section 2.7)
- Modify: `data/kennzahlen.tex` (clôtures annuelles de contrôle)

**Interfaces:**
- Consumes: `refs/gb2022.pdf` … `refs/gb2025.pdf` (Task 2).
- Produces: `data/aktienkurs.csv` avec en-tête commenté portant la source et l'`Abrufdatum`, colonnes `datum,schluss` (`datum` au format `YYYY-MM-DD`). Macros `\KursSilvesterZFII`, `\KursSilvesterZFIII`, `\KursSilvesterZFV` (clôtures 2022, 2023, 2025 ; celle de 2024 existe déjà).

- [ ] **Step 1: Écrire `scripts/fetch_kurs.py`**

```python
#!/usr/bin/env python3
"""
fetch_kurs.py - Monatliche Schlusskurse der Aumann-Aktie (ISIN DE000A2DAM03,
Xetra-Kuerzel AAG) fuer den Zeitraum 2022-01 bis 2025-12.

Schreibt data/aktienkurs.csv mit Quellenangabe und Abrufdatum im Kopf.
Bricht mit Rueckgabewert 1 ab, wenn keine Quelle antwortet - in diesem Fall
ist auf die in den Geschaeftsberichten veroeffentlichten Kursdaten
zurueckzufallen (Rueckfallweg im Spezifikationsdokument).
Reine Standardbibliothek.
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

SYMBOL = "AAG.DE"
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    f"{SYMBOL}?period1=1640995200&period2=1735689600&interval=1mo"
)
UA = {"User-Agent": "Mozilla/5.0 (Hausarbeit HS Kaiserslautern; academic use)"}


def fetch() -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Laedt die monatlichen Schlusskurse und gibt sie als Liste zurueck.

    Inputs:
        keine

    Outputs:
        rows (list[tuple[str, float]]): (Datum ISO, Schlusskurs in EUR)
    --------------------------------------------------------------------------
    """
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    rows = []
    for ts, close in zip(stamps, closes):
        if close is None:
            continue
        day = datetime.datetime.utcfromtimestamp(ts).date().isoformat()
        rows.append((day, round(float(close), 2)))
    return rows


def main() -> int:
    try:
        rows = fetch()
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
        print(f"FEHLER: Kursquelle nicht erreichbar oder unlesbar: {exc}", file=sys.stderr)
        print("Rueckfall: Kursdaten aus den Geschaeftsberichten 2022-2025 verwenden.",
              file=sys.stderr)
        return 1
    if len(rows) < 40:
        print(f"FEHLER: nur {len(rows)} Datenpunkte erhalten, erwartet werden rund 48.",
              file=sys.stderr)
        return 1

    abruf = datetime.date.today().strftime("%d.%m.%Y")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "aktienkurs.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Aumann AG (ISIN DE000A2DAM03), monatliche Schlusskurse in EUR\n")
        fh.write(f"# Quelle: Yahoo Finance, Symbol {SYMBOL}\n")
        fh.write(f"# Abrufdatum: {abruf}\n")
        fh.write("datum,schluss\n")
        for day, close in rows:
            fh.write(f"{day},{close}\n")
    print(f"OK -> data/aktienkurs.csv ({len(rows)} Datenpunkte, Abruf {abruf})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Exécuter et gérer le repli**

Run: `python3 scripts/fetch_kurs.py`
Expected: `OK -> data/aktienkurs.csv (~48 Datenpunkte, Abruf <date>)`.

**Si le script sort en erreur** (fournisseur bloqué) : appliquer le repli prévu par le spec — extraire les données de cours publiées dans les Geschäftsberichte :
```bash
for y in 2022 2023 2024 2025; do echo "== GB$y =="; \
  pdftotext -layout refs/gb$y.pdf - | grep -niE 'share price|closing price|high|low|Xetra' | head -12; done
```
puis écrire `data/aktienkurs.csv` à la main avec les points disponibles (clôtures annuelles, plus haut, plus bas), en remplaçant l'en-tête `# Quelle:` par `# Quelle: Aumann AG, Geschaeftsberichte 2022-2025`.

- [ ] **Step 3: Contrôler la série contre les rapports**

Run:
```bash
grep -E '^(2022|2023|2024|2025)-12' data/aktienkurs.csv
pdftotext -layout refs/gb2024.pdf - | grep -niE 'closing price|share price at|31 December' | head -10
```
Expected: la clôture de décembre 2024 du CSV doit correspondre, à l'arrondi près, au cours de clôture publié dans le GB 2024 (macro `\KursSilvesterZFIV` de la Task 4). **Un écart supérieur à 2 % doit être investigué avant de continuer** — c'est le signe d'un mauvais symbole ou d'une place de cotation différente.

- [ ] **Step 4: Ajouter les clôtures annuelles à `data/kennzahlen.tex`**

```latex
% --- Aufgabe 2.7: Jahresschlusskurse (Xetra) -----------------------
\newcommand{\KursSilvesterZFII}{<wert>}   % GB 2022, S. <seite>
\newcommand{\KursSilvesterZFIII}{<wert>}  % GB 2023, S. <seite>
\newcommand{\KursSilvesterZFV}{<wert>}    % GB 2025, S. <seite>
```

- [ ] **Step 5: Écrire la sous-section 2.7 avec le graphique**

Ajouter dans `sections/02-jahresabschluss.tex`, sous `\label{subsec:kursverlauf}` :

```latex
Der Kursverlauf vom 1.~Januar 2022 bis zum 31.~Dezember 2025 ist in
Abbildung~\ref{fig:kursverlauf} dargestellt. Die Reihe umfasst monatliche
Schlusskurse; die Jahresschlusskurse wurden gegen die Angaben der
Gesch\"aftsberichte gepr\"uft.\quelleGB{2024}{<seite>}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.95\linewidth, height=6.5cm,
  date coordinates in=x,
  xticklabel={\month.\year},
  xlabel={Zeit}, ylabel={Schlusskurs in \euro},
  xtick distance=365, x tick label style={rotate=45, anchor=east},
  grid=major, grid style={gray!25},
  ymajorgrids=true, tick align=outside,
]
\addplot[blue!60!black, thick, mark=none]
  table[col sep=comma, x=datum, y=schluss, comment chars={\#}]
  {data/aktienkurs.csv};
\end{axis}
\end{tikzpicture}
\caption{Monatliche Schlusskurse der Aumann-Aktie 2022--2025.}
\label{fig:kursverlauf}
\end{figure}

\noindent Die Jahresschlusskurse betrugen \EURje{\KursSilvesterZFII} (2022),
\EURje{\KursSilvesterZFIII} (2023), \EURje{\KursSilvesterZFIV} (2024) und
\EURje{\KursSilvesterZFV} (2025).
```

- [ ] **Step 6: Compiler et vérifier le graphique**

Run: `./build.sh`
Expected: `OK -> out/hausarbeit.pdf` sans erreur. Si pgfplots signale `Package pgfplots Error: Sorry, I could not read the date`, vérifier que `\usepgfplotslibrary{dateplot}` est présent — l'ajouter dans `preamble.tex` juste après `\pgfplotsset{compat=1.18}` :
```latex
\usepgfplotslibrary{dateplot}
```

- [ ] **Step 7: Contrôler visuellement**

Run: `pdftotext out/hausarbeit.pdf - | grep -n 'Kursverlauf\|Schlusskurs' | head`
Expected: la légende de la figure apparaît. Ouvrir le PDF pour vérifier que la courbe est tracée, non vide, et que l'axe des dates est lisible.

- [ ] **Step 8: Vérifier l'absence de littéraux et committer**

```bash
python3 scripts/check_literals.py sections/*.tex
git add scripts/fetch_kurs.py data/aktienkurs.csv data/kennzahlen.tex sections/02-jahresabschluss.tex preamble.tex
git commit -m "feat: Kursreihe 2022-2025 und Kursverlaufsdiagramm (Aufgabe 2.7)"
```

---

### Task 7: Partie 1 — Beschreibung des Unternehmens

**Files:**
- Modify: `sections/01-unternehmen.tex`
- Modify: `data/kennzahlen.tex`

**Interfaces:**
- Consumes: `refs/gb2024.pdf`, `refs/gb2025.pdf` ; macros `\quelleGB`, `\quelleWeb`, `\MioEUR`, `\Pct` (Task 1) ; `\KursSilvesterZF*` (Task 6).
- Produces: macros `\AktienGesamt`, `\FreefloatAnteil`, `\MarktkapZFIV` dans `data/kennzahlen.tex`.

- [ ] **Step 1: Extraire les faits du GB 2024 et du GB 2025**

```bash
pdftotext -layout refs/gb2024.pdf - | grep -niE 'Executive Board|Management Board|Supervisory Board|CEO|CFO' | head -20
pdftotext -layout refs/gb2024.pdf - | grep -niE 'shareholder structure|free float|shares outstanding|treasury shares|% of' | head -20
pdftotext -layout refs/gb2024.pdf - | grep -niE 'analyst|coverage|research' | head -20
```

Noter : membres du Vorstand et de l'Aufsichtsrat avec fonction ; actionnaires principaux avec pourcentage ; free float ; nombre d'actions ; maisons d'analyse couvrant le titre.

- [ ] **Step 2: Compléter les analystes par une recherche web sourcée**

Consulter la page « Analysts » ou « Aktie » de l'espace Investor Relations :
```bash
curl -sL --max-time 30 "https://www.aumann.com/en/investor-relations/" \
  | grep -oiE '<a[^>]*href="[^"]*(analyst|share|aktie)[^"]*"' | sort -u | head
```
Suivre les liens pertinents, relever les maisons d'analyse et la date de consultation. **Chaque affirmation issue du web est citée avec `\quelleWeb{...}{...}{<Abrufdatum>}`.** Si aucune liste d'analystes n'est publiée, l'écrire explicitement dans le document plutôt que de nommer des maisons non vérifiées.

- [ ] **Step 3: Ajouter les macros chiffrées**

```latex
% --- Aufgabe 1: Unternehmensprofil ---------------------------------
\newcommand{\AktienGesamt}{<wert>}      % GB 2024, S. <seite> (Stueck)
\newcommand{\FreefloatAnteil}{<wert>}   % GB 2024, S. <seite> (in %)
\newcommand{\MarktkapZFIV}{<wert>}      % berechnet: AktienGesamt * KursSilvesterZFIV
```

- [ ] **Step 4: Rédiger `sections/01-unternehmen.tex`**

Structure : `\subsection{Handelnde Personen in f\"uhrender Position}` (1.1) · `\subsection{Eigent\"umerstruktur}` (1.2, avec tableau `booktabs` actionnaire / part) · `\subsection{Beurteilung durch die Kapitalm\"arkte}` (1.3, cours et capitalisation, renvoi à `\ref{fig:kursverlauf}`) · `\subsection{Analystenabdeckung}` (1.4) · `\subsection{Botschaften der Analysten}` (1.5).

Sections 1.1 à 1.4 : **prose factuelle noire**, chaque affirmation avec sa note de bas de page.
Section 1.5 : les messages des analystes sont **rapportés** factuellement en noir (« Analysehaus X nannte am <date> ein Kursziel von … »). Aucune interprétation ici — l'interprétation du cours est la zone bleue de la partie 4.

- [ ] **Step 5: Vérifier et compiler**

```bash
python3 scripts/check_literals.py sections/01-unternehmen.tex && ./build.sh
```
Expected: `OK - keine hartkodierten Zahlen` puis `OK -> out/hausarbeit.pdf`.

- [ ] **Step 6: Commit**

```bash
git add sections/01-unternehmen.tex data/kennzahlen.tex
git commit -m "feat: Aufgabe 1 - Unternehmensbeschreibung, Eigentuemerstruktur, Analysten"
```

---

### Task 8: Partie 3 — Strategische Maßnahmen (factuel)

**Files:**
- Modify: `sections/03-strategie.tex`
- Modify: `data/kennzahlen.tex`

**Interfaces:**
- Consumes: `refs/gb2024.pdf` (source principale), `refs/q1-2025.pdf`, `refs/h1-2025.pdf`, `refs/q3-2025.pdf`, `refs/gb2025.pdf`, `refs/q1-2026.pdf` (pour 3.6).
- Produces: macros `\EbitZFIV`, `\UmsatzEMobilityZFIV`, `\UmsatzNextAutomationZFIV`, `\PrognoseUmsatzZFVUnten`, `\PrognoseUmsatzZFVOben`, `\PrognoseEbitdaMargeZFV`.

- [ ] **Step 1: Extraire le contenu stratégique du GB 2024**

```bash
pdftotext -layout refs/gb2024.pdf - | grep -niE 'E-Mobility|Next Automation|Classic|Clean Tech|Aerospace|Life Sciences' | head -30
pdftotext -layout refs/gb2024.pdf - | grep -niE 'cell-to-pack|MEA|electrode|fuel cell|coating|lamination|acquisition' | head -30
pdftotext -layout refs/gb2024.pdf - | grep -niE 'China|United States|USA|site|location|internationalis' | head -20
pdftotext -layout refs/gb2024.pdf - | grep -niE 'share buyback|treasury|capital allocation|dividend' | head -20
pdftotext -layout refs/gb2024.pdf - | grep -niE 'outlook|guidance|forecast|expects|2025' | head -30
```

- [ ] **Step 2: Extraire le réalisé 2025 pour confronter le Ausblick (question 3.6)**

```bash
for f in q1-2025 h1-2025 q3-2025 gb2025 q1-2026; do
  echo "=== $f ==="
  pdftotext -layout refs/$f.pdf - | grep -niE 'guidance|outlook|revenue of|order intake|EBITDA margin' | head -10
done
```

- [ ] **Step 3: Ajouter les macros**

```latex
% --- Aufgabe 3: Strategie und Prognose -----------------------------
\newcommand{\EbitZFIV}{<wert>}                  % GB 2024, S. <seite>
\newcommand{\UmsatzEMobilityZFIV}{<wert>}       % GB 2024, S. <seite>
\newcommand{\UmsatzNextAutomationZFIV}{<wert>}  % GB 2024, S. <seite>
\newcommand{\PrognoseUmsatzZFVUnten}{<wert>}    % GB 2024, S. <seite> (Ausblick 2025)
\newcommand{\PrognoseUmsatzZFVOben}{<wert>}     % GB 2024, S. <seite>
\newcommand{\PrognoseEbitdaMargeZFV}{<wert>}    % GB 2024, S. <seite>
```

- [ ] **Step 4: Rédiger `sections/03-strategie.tex`**

Six sous-sections suivant l'énoncé, **toutes en noir, reformulées, sourcées, sans jugement** :
`\subsection{Wachstumssegment E-Mobility}` (3.1) · `\subsection{Neuausrichtung des Segments Next Automation}` (3.2) · `\subsection{Batterie- und Brennstoffzellen-Produktionstechnik}` (3.3, y compris la dernière acquisition) · `\subsection{Internationalisierung und Kapitalallokation}` (3.4) · `\subsection{Bisherige Wirkung auf operatives EBITDA und EBIT}` (3.5) · `\subsection{Ausblick 2025 und dessen Widerspiegelung in den Unternehmensberichten}` (3.6).

La sous-section 3.6 comporte un tableau `booktabs` à trois colonnes — `Kennzahl` / `Prognose GB 2024` / `Ist gemäß Berichten 2025` — utilisant uniquement des macros. Le tableau expose l'écart ; **aucun commentaire évaluatif** : la lecture de l'écart appartient aux zones bleues des parties 4 et 5.

Interdiction explicite dans cette section : les adjectifs de jugement (`erfolgreich`, `ambitioniert`, `überzeugend`, `enttäuschend`) et toute phrase au conditionnel prospectif. La partie 3 rapporte.

- [ ] **Step 5: Vérifier et compiler**

```bash
python3 scripts/check_literals.py sections/03-strategie.tex && ./build.sh
```
Expected: les deux commandes en succès.

- [ ] **Step 6: Commit**

```bash
git add sections/03-strategie.tex data/kennzahlen.tex
git commit -m "feat: Aufgabe 3 - strategische Massnahmen und Abgleich des Ausblicks 2025"
```

---

### Task 9: Partie 5 — structure et ancres factuelles

**Files:**
- Modify: `sections/05-handlungsoptionen.tex`

**Interfaces:**
- Consumes: toutes les macros de `data/kennzahlen.tex` (Tasks 4, 6, 7, 8) ; `\bk`, `\quelleGB` (Task 1).
- Produces: les 11 emplacements `\bk{}` de la partie 5, chacun précédé de son bloc `% FAKTENBASIS`, remplis en Task 11. Labels de tableaux : `tab:einschaetzung-a`, `tab:einschaetzung-b`, `tab:einschaetzung-c`, `tab:uebersicht-optionen`.

- [ ] **Step 1: Poser la structure et le contexte factuel commun**

```latex
\section{Beurteilung ausgew\"ahlter Handlungsoptionen}
\label{sec:optionen}

Ausgangslage laut Aufgabenstellung und Gesch\"aftsbericht 2024: Die Aumann~AG
verf\"ugt \"uber eine Nettofinanzliquidit\"at von \MioEUR{\NettoliquiditaetZFIV}
zum Jahresende 2024\quelleGB{2024}{<seite>}; der Auftragseingang ging 2024 um
\Pct{\AuftragseingangDeltaZFIV} zur\"uck\quelleGB{2024}{<seite>}.

\subsection{Diversifikation in Zukunftsbranchen (Option A)}
\label{subsec:option-a}
```

- [ ] **Step 2: Écrire les trois sous-sections .1 (Management-Summary)**

Pour A (répéter pour B en `\subsection{Ausbau der Produktionstechnik f\"ur Batterie- und Brennstoffzellenfertigung (Option B)}` et C en `\subsection{Anorganisches Wachstum durch gezielte Akquisition (Option C)}`) :

```latex
\subsubsection{Relevanz der Strategie (Management-Summary)}
% FAKTENBASIS 5.1.1
%  - Auftragseingang 2024: siehe \AuftragseingangZFIV, Rueckgang \AuftragseingangDeltaZFIV %
%  - Nettofinanzliquiditaet 31.12.2024: \NettoliquiditaetZFIV Mio. EUR
%  - Umsatz Segment Next Automation 2024: \UmsatzNextAutomationZFIV Mio. EUR
%  - Zielmaerkte laut GB 2024: Clean Tech, Aerospace, Life Sciences
%  - Treiber laut Aufgabenstellung: Automatisierungsbedarf, Reshoring, Demografie
%  - Investitionsannahme Aufgabe 6: 35 Mio. EUR ueber drei Jahre, +16 Mio. EUR EBITDA ab 2028
\bk{}
```

- [ ] **Step 3: Écrire les trois tableaux .2 (Faktor / Einschätzung)**

Les six lignes de facteurs sont reprises **littéralement** de l'énoncé. La colonne `Einschätzung` contient un `\bk{}` vide par ligne ; les ancres chiffrées disponibles sont posées en noir dans la colonne `Faktor` sous forme de parenthèse factuelle.

```latex
\subsubsection{Finanzwirtschaftliche Bewertung}
\begin{table}[htbp]
\centering
\caption{Finanzwirtschaftliche Einsch\"atzung der Option A.}
\label{tab:einschaetzung-a}
\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Faktor} & \textbf{Einsch\"atzung}\\
\midrule
Investitionsbedarf (F\&E / Markterschlie{\ss}ung) & \bk{}\\
Kapitalbindung (Anlagen, CapEx) & \bk{}\\
Liquidit\"atsbedarf (j\"ahrlich) & \bk{}\\
Zeithorizont bis Return & \bk{}\\
Risiko & \bk{}\\
Potenzial & \bk{}\\
\bottomrule
\end{tabularx}
\end{table}
```

Pour B, les libellés de l'énoncé sont `Investitionsbedarf (F\&E)` et `Kapitalbindung (Anlagen)`. Pour C : `Investitionsbedarf (Kaufpreis \& Integration)` et `Kapitalbindung (Umlaufverm\"ogen erworbenes Gesch\"aft)`.

- [ ] **Step 4: Écrire les trois sous-sections .3 (Statement) et l'appréciation supplémentaire de 5.3.3**

```latex
\subsubsection{Statement}
% FAKTENBASIS 5.1.3
%  - Umsetzungsdauer laut Annahme Aufgabe 6: Investition ueber drei Jahre (2026-2028)
%  - Ergebnisbeitrag erst ab 2028 -> Rentabilitaetswirkung (Eigenkapital) verzoegert
%  - Kapitalbindung 15 Mio. EUR belastet das Umlaufvermoegen (current assets)
%  - Liquiditaetsreserve Ende 2028 laut Tabelle \ref{tab:cf-a}
%  - EBITDA-Marge 2024: \EbitdaMargeZFIV %
%  - Lehrveranstaltung: Zielkonflikt Rentabilitaet (Eigenkapitalwirkung) vs.
%    Liquiditaet (Umlaufvermoegen)
\bk{}
```

Et, à la fin de 5.3.3 :

```latex
\paragraph{Verwendung der Nettoliquidit\"at.}
% FAKTENBASIS 5.3.3 (Zusatz)
%  - Nettofinanzliquiditaet 31.12.2024: \NettoliquiditaetZFIV Mio. EUR
%  - Nettofinanzliquiditaet 31.12.2025: \NettoliquiditaetZFV Mio. EUR
%  - Aktienrueckkaufprogramme und Dividende laut GB 2024, Abschnitt 3.4
%  - Dividende je Aktie 2024: \DividendeJeAktieZFIV EUR, Rendite \DividendenrenditeZFIV %
%  - Kapitalbedarf Option C laut Aufgabe 6: 60 Mio. EUR + 20 Mio. EUR Kapitalbindung
\bk{}
```

- [ ] **Step 5: Écrire 5.4 — matrice de synthèse et recommandation**

```latex
\subsection{Zusammenfassende Bewertung der drei Optionen}
\label{subsec:uebersicht}

\begin{table}[htbp]
\centering
\caption{\"Ubersicht der drei Handlungsoptionen.}
\label{tab:uebersicht-optionen}
\begin{tabularx}{\linewidth}{@{}lXXXXXX@{}}
\toprule
& \textbf{Investitions-} & \textbf{Kapital-} & \textbf{Liquidit\"ats-}
& \textbf{Risiko} & \textbf{Zeit-} & \textbf{Chancen}\\
& \textbf{bedarf} & \textbf{bindung} & \textbf{bedarf (j\"ahrl.)}
& & \textbf{horizont} & \\
\midrule
A -- Next Automation & \bk{} & \bk{} & \bk{} & \bk{} & \bk{} & \bk{}\\
B -- Batterie/Brennstoffzellen & \bk{} & \bk{} & \bk{} & \bk{} & \bk{} & \bk{}\\
C -- Akquisition & \bk{} & \bk{} & \bk{} & \bk{} & \bk{} & \bk{}\\
\bottomrule
\end{tabularx}
\end{table}

\subsubsection{Zusammenfassende Empfehlung}
% FAKTENBASIS 5.4.2
%  - Kumulierte Investitionen: A 35, B 30, C 60 Mio. EUR (Aufgabe 6)
%  - EBITDA-Zuwachs: A +16 ab 2028, B +14 ab 2027, C +10 ab 2027
%  - Liquiditaetsreserve Ende 2028 und IRR: siehe Tabelle \ref{tab:vergleich2028}
%  - Ausgangsliquiditaet: \NettoliquiditaetZFIV Mio. EUR
%  - Auftragseingangsrueckgang 2024: \AuftragseingangDeltaZFIV %
\bk{}
```

- [ ] **Step 6: Vérifier et compiler**

```bash
python3 scripts/check_literals.py sections/05-handlungsoptionen.tex && ./build.sh
```
Expected: `OK` puis `OK -> out/hausarbeit.pdf`. Les tableaux apparaissent avec la colonne `Einschätzung` vide — c'est attendu à ce stade.

- [ ] **Step 7: Compter les emplacements bleus de la partie 5**

Run: `grep -c '\\bk{}' sections/05-handlungsoptionen.tex`
Expected: `44` — 3 Management-Summary + 18 cellules de tableau (3 × 6) + 3 Statements + 1 Zusatz + 18 cellules de la matrice de synthèse (3 × 6) + 1 Empfehlung.

- [ ] **Step 8: Commit**

```bash
git add sections/05-handlungsoptionen.tex
git commit -m "feat: Aufgabe 5 - Struktur, Faktortabellen und Faktenbasis der Handlungsoptionen"
```

---

### Task 10: Partie 6 — section d'assemblage et hypothèses

**Files:**
- Modify: `sections/06-cashflow.tex`

**Interfaces:**
- Consumes: `data/cf_a.tex`, `data/cf_b.tex`, `data/cf_c.tex`, `data/vergleich2028.tex`, `data/irr_sensitivitaet.tex` (Task 5) ; `data/basis.json` ; macros `\annahme`, `\bk`, `\MioEUR` (Task 1).
- Produces: labels `tab:cf-a`, `tab:cf-b`, `tab:cf-c`, `tab:vergleich2028`, `tab:irr-sens` — référencés depuis la partie 5 (Task 9).

- [ ] **Step 1: Écrire la sous-section « Annahmen der Flow-Analyse »**

```latex
\section{Grobe Flow-Analyse}
\label{sec:cashflow}

\subsection{Annahmen der Flow-Analyse}
\label{subsec:annahmen}

Die Aufgabenstellung gibt die Investitionssummen, die Kapitalbindung und den
erwarteten EBITDA-Zuwachs vor, nicht jedoch deren zeitliche Verteilung und die
Definition des ausgewiesenen Cash-Flows. Die folgenden f\"unf Annahmen werden
daher offengelegt.

\annahme{\textbf{Anfangsbestand.} Die Rechnung setzt auf der Nettofinanz\-liquidit\"at
von \MioEUR{\NettoliquiditaetZFIV} zum 31.~Dezember 2024 auf, wie in der
Aufgabenstellung vorgegeben. Zum 31.~Dezember 2025 wies die Gesellschaft
tats\"achlich \MioEUR{\NettoliquiditaetZFV} aus\quelleGB{2025}{<seite>}; die
Vorgabe der Aufgabenstellung wird gleichwohl beibehalten.}

\annahme{\textbf{Investitionsprofil.} Die Investitionen der Optionen~A und~B
verteilen sich gleichm\"a{\ss}ig auf die Jahre 2026 bis 2028. Bei Option~C wird
der Kaufpreis im Vollzugsjahr 2026 f\"allig; die Integrationsaufwendungen
verteilen sich auf 2027 und 2028.}

\annahme{\textbf{Kapitalbindung.} Der Aufbau der Kapitalbindung erfolgt in dem
Jahr beziehungsweise den Jahren unmittelbar vor dem jeweiligen
Ergebnisbeitrag, da das Umlaufverm\"ogen den Umsatz vorfinanziert. Bei
Option~C f\"allt das Umlaufverm\"ogen des erworbenen Gesch\"afts mit dem
Vollzug an.}

\annahme{\textbf{Abgrenzung des operativen Cash-Flows.} Ausgewiesen wird der
\emph{zus\"atzliche} operative Cash-Flow der jeweiligen Option, das hei{\ss}t
der EBITDA-Zuwachs abz\"uglich des Aufbaus der Kapitalbindung. Der operative
Cash-Flow des bestehenden Kerngesch\"afts bleibt unber\"ucksichtigt, damit die
Wirkung der Entscheidung sichtbar bleibt.}

\annahme{\textbf{Renditehorizont.} Der interne Zinsfu{\ss} wird \"uber den
Horizont 2026 bis 2035 ermittelt. Der EBITDA-Zuwachs l\"auft ab seinem
Startjahr konstant weiter; die Kapitalbindung wird im letzten Jahr des
Horizonts freigesetzt. Die Sensitivit\"at gegen\"uber der Horizontl\"ange
zeigt Tabelle~\ref{tab:irr-sens}. Die EBITDA-Marge Ende 2028 bezieht den
EBITDA-Zuwachs auf das operative EBITDA und den Umsatz des Gesch\"aftsjahres
2025, die konstant fortgeschrieben werden.}
```

- [ ] **Step 2: Écrire les trois tableaux 6.1**

```latex
\subsection{Cash-Flow-Tabellen der Optionen}
\label{subsec:cf-tabellen}

\begin{table}[htbp]
\centering
\caption{Option~A -- Diversifikation \enquote{Next Automation}. Alle Betr\"age in Mio.\,\euro.}
\label{tab:cf-a}
\begin{tabularx}{\linewidth}{@{}l S[table-format=2.1] S[table-format=-2.1] S[table-format=3.1] X@{}}
\toprule
\textbf{Jahr} & {\textbf{Investition}} & {\textbf{Operativer CF}}
& {\textbf{Nettoliquidit\"at}} & \textbf{Bemerkung}\\
\midrule
\input{data/cf_a}
\bottomrule
\end{tabularx}
\end{table}
```

Reproduire à l'identique pour B (`\input{data/cf_b}`, label `tab:cf-b`, caption `Option~B -- Batterie- und Brennstoffzellen-Produktionstechnik`) et C (`\input{data/cf_c}`, label `tab:cf-c`, caption `Option~C -- Anorganisches Wachstum durch gezielte Akquisition`).

Note : les colonnes générées contiennent déjà `\num{...}` ; utiliser donc `c`/`r` plutôt que `S` si siunitx signale `Invalid number` — corriger en remplaçant `S[table-format=...]` par `r` dans les trois tableaux.

- [ ] **Step 3: Écrire 6.2 — trois zones bleues**

```latex
\subsection{Ergebnis der einzelnen Flow-Analysen}
\label{subsec:cf-ergebnis}

\subsubsection{Option A}
% FAKTENBASIS 6.2 A
%  - Kumulierte Investition 35 Mio. EUR, Kapitalbindung 15 Mio. EUR
%  - EBITDA-Zuwachs +16 Mio. EUR erst ab 2028 -> spaeteste Ergebniswirkung
%  - Liquiditaetsverlauf und Endbestand: Tabelle \ref{tab:cf-a}
%  - IRR und Vergleichswerte: Tabelle \ref{tab:vergleich2028}
\bk{}

\subsubsection{Option B}
% FAKTENBASIS 6.2 B
%  - Kumulierte Investition 30 Mio. EUR, Kapitalbindung 15 Mio. EUR
%  - EBITDA-Zuwachs +14 Mio. EUR bereits ab 2027
%  - Liquiditaetsverlauf und Endbestand: Tabelle \ref{tab:cf-b}
%  - Geringster Kapitalbedarf der drei Optionen
\bk{}

\subsubsection{Option C}
% FAKTENBASIS 6.2 C
%  - Kumulierte Investition 60 Mio. EUR, Kapitalbindung 20 Mio. EUR
%  - Kaufpreis 45 Mio. EUR im Vollzugsjahr 2026 -> tiefster Liquiditaetspunkt
%  - EBITDA-Zuwachs +10 Mio. EUR ab 2027, geringster absoluter Zuwachs
%  - Liquiditaetsverlauf und Endbestand: Tabelle \ref{tab:cf-c}
\bk{}
```

- [ ] **Step 4: Écrire 6.3 — comparaison 2028, sensibilité et Fazit**

```latex
\subsection{Vergleich der Ergebniswirkung (Ende 2028)}
\label{subsec:vergleich}

\begin{table}[htbp]
\centering
\caption{Vergleich der drei Optionen zum Jahresende 2028.}
\label{tab:vergleich2028}
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}
\toprule
\textbf{Kennzahl} & \textbf{Option A} & \textbf{Option B} & \textbf{Option C}\\
\midrule
\input{data/vergleich2028}
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[htbp]
\centering
\caption{Sensitivit\"at des internen Zinsfu{\ss}es gegen\"uber der Horizontl\"ange.}
\label{tab:irr-sens}
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}
\toprule
\textbf{Horizont} & \textbf{Option A} & \textbf{Option B} & \textbf{Option C}\\
\midrule
\input{data/irr_sensitivitaet}
\bottomrule
\end{tabularx}
\end{table}

\subsubsection{Fazit des Vergleichs}
% FAKTENBASIS 6.3
%  - Kumulierte Investition: A 35, B 30, C 60 Mio. EUR
%  - EBITDA-Zuwachs: A +16 ab 2028, B +14 ab 2027, C +10 ab 2027
%  - EBITDA-Marge Ende 2028, IRR, Liquiditaetsreserve: Tabelle \ref{tab:vergleich2028}
%  - IRR-Sensitivitaet nach Horizontlaenge: Tabelle \ref{tab:irr-sens}
%  - Net Debt/EBITDA nur fuer Option C relevant, da A und B die Nettoliquiditaet
%    nicht aufzehren
%  - Umsatzbasis der Margenrechnung: Geschaeftsjahr 2025, konstant fortgeschrieben
\bk{}
```

- [ ] **Step 5: Compiler et vérifier que les tableaux générés sont rendus**

Run: `./build.sh && pdftotext out/hausarbeit.pdf - | grep -E '73,2|78,2|121,2' | head`
Expected: les valeurs de liquidité calculées apparaissent avec la virgule décimale.

- [ ] **Step 6: Vérifier que la partie 6 ne contient aucun littéral non généré**

Run: `python3 scripts/check_literals.py sections/06-cashflow.tex`
Expected: `OK - keine hartkodierten Zahlen in den geprueften Dateien.` (les chiffres sont dans `data/*.tex`, pas dans la section).

- [ ] **Step 7: Commit**

```bash
git add sections/06-cashflow.tex
git commit -m "feat: Aufgabe 6 - Annahmen, Cash-Flow-Tabellen und Vergleich 2028"
```

---

### Task 11: Rédaction des 17 zones bleues

**Files:**
- Modify: `sections/04-aktienkurs.tex`, `sections/05-handlungsoptionen.tex`, `sections/06-cashflow.tex`

**Interfaces:**
- Consumes: tous les blocs `% FAKTENBASIS` posés aux Tasks 9 et 10, les tableaux calculés de la Task 5, les résultats des Tasks 4, 6, 7, 8.
- Produces: aucun `\bk{}` vide restant dans le document.

**Règles de rédaction** — s'appliquent à chacune des 17 zones :
- Allemand, registre Master, prose en phrases complètes ; jamais de puces dans le PDF.
- Chaque affirmation évaluative s'adosse à un chiffre déjà établi dans le document, cité par macro ou par renvoi de tableau (`Tabelle~\ref{...}`).
- Respecter la limite de longueur de l'énoncé (voir tableau ci-dessous). Contrôle : ½ page A4 ≈ 250 mots, ¼ page ≈ 125 mots, 1 page ≈ 500 mots.
- Les questions imposées par l'énoncé doivent être traitées explicitement : pour 5.x.3, la **durée de mise en œuvre**, l'effet sur la **Rentabilität (Eigenkapitalwirkung)**, l'effet sur la **Liquidität (Umlaufvermögen / current assets)**, et l'effet attendu sur le **Kerngeschäft et la marge**.
- Ne jamais introduire un chiffre qui n'existe pas déjà dans `data/kennzahlen.tex` ou dans une table générée.

- [ ] **Step 1: Rédiger la zone 1 — partie 4 (Beurteilung des Aktienkursverlaufs, max ½ A4)**

Dans `sections/04-aktienkurs.tex` :

```latex
\section{Beurteilung des Verlaufs des Aktienkurses}
\label{sec:aktienkurs}
% FAKTENBASIS 4
%  - Jahresschlusskurse: \KursSilvesterZFII, \KursSilvesterZFIII,
%    \KursSilvesterZFIV, \KursSilvesterZFV
%  - Kursverlauf: Abbildung \ref{fig:kursverlauf}
%  - Auftragseingang 2024: \AuftragseingangDeltaZFIV % -> Zyklusabhaengigkeit
%  - Umsatzentwicklung 2023/2024: \UmsatzDeltaZFIV %
%  - EBITDA-Marge 2024: \EbitdaMargeZFIV %
%  - Dividendenrendite 2024: \DividendenrenditeZFIV %
%  - Ausblick 2025 und Ist 2025: Abschnitt \ref{sec:strategie}
%  - Analystenbotschaften: Abschnitt \ref{sec:unternehmen}
\bk{...}
```

Relier explicitement le cours aux résultats des parties 2 et 3, comme l'exige l'énoncé (« bitte mit Ihren Ergebnissen begründen »).

- [ ] **Step 2: Rédiger les zones 2, 5, 8 — les trois Management-Summary (5.1.1, 5.2.1, 5.3.1, max ½ A4 chacune)**

Public visé imposé par l'énoncé : **non-ingénieurs**. Éviter le vocabulaire technique de production ; expliquer la logique économique.

- [ ] **Step 3: Rédiger les zones 3, 6, 9 — les 18 cellules `Einschätzung` des trois tableaux**

Chaque cellule : une à deux phrases, avec un ordre de grandeur chiffré lorsque l'énoncé en fournit un (partie 6). Exemple de facture attendue pour `Liquiditätsbedarf (jährlich)` de l'option A : montant annuel moyen tiré du profil d'investissement, puis appréciation.

- [ ] **Step 4: Rédiger les zones 4, 7, 10 — les trois Statements (5.1.3, 5.2.3, 5.3.3, max 1 A4 chacun)**

Traiter les quatre points imposés (durée, Rentabilität/Eigenkapital, Liquidität/Umlaufvermögen, effet sur Kerngeschäft et marge) dans cet ordre, en s'appuyant sur le cadre de la Veranstaltung.

- [ ] **Step 5: Rédiger la zone 11 — Wachstum vs. Rückgabe an die Aktionäre (max ½ A4)**

Prendre position explicitement, comme l'exige l'énoncé (« Würden Sie … ? Begründen Sie »), en s'appuyant sur la Nettoliquidität, les programmes de rachat et les besoins de financement des trois options.

- [ ] **Step 6: Rédiger la zone 12 — les 18 cellules de la matrice 5.4.1**

Cellules courtes et comparables entre les trois options : une échelle qualitative cohérente (`gering` / `mittel` / `hoch`) assortie du chiffre d'ancrage.

- [ ] **Step 7: Rédiger la zone 13 — Zusammenfassende Empfehlung (max ¼ A4)**

Une recommandation nette et hiérarchisée, cohérente avec les zones 2 à 12 et avec les chiffres de la partie 6.

- [ ] **Step 8: Rédiger les zones 14, 15, 16 — les trois Ergebnis de 6.2 (max ¼ page chacun)**

Traiter les trois points imposés par l'énoncé : Kapitalbedarf, erwartete Rendite, et mise en relation avec l'effet de liquidité.

- [ ] **Step 9: Rédiger la zone 17 — Fazit du comparatif 2028 (max 1 A4)**

- [ ] **Step 10: Vérifier qu'aucune zone bleue n'est restée vide**

Run: `grep -n '\\bk{}' sections/*.tex`
Expected: aucune sortie (hormis les placeholders du Deckblatt, qui contiennent du texte et ne matchent donc pas).

- [ ] **Step 11: Vérifier les longueurs**

Run:
```bash
python3 - <<'PY'
import re, glob
for path in sorted(glob.glob("sections/*.tex")):
    src = open(path, encoding="utf-8").read()
    src = "\n".join(l.split("%")[0] for l in src.splitlines())
    for m in re.finditer(r"\\bk\{", src):
        i = m.end(); depth = 1
        while i < len(src) and depth:
            if src[i] == "{": depth += 1
            elif src[i] == "}": depth -= 1
            i += 1
        body = src[m.end():i-1]
        words = len(body.split())
        if words > 20:
            print(f"{path}: {words} Woerter")
PY
```
Expected: aucune zone ne dépasse ~500 mots (1 page A4) ; les Management-Summary et l'appréciation de 5.3.3 restent sous ~250 mots ; les Ergebnis de 6.2 et l'Empfehlung sous ~125 mots. Raccourcir toute zone au-dessus de sa limite.

- [ ] **Step 12: Compiler et relire le PDF**

Run: `./build.sh`
Expected: `OK -> out/hausarbeit.pdf`. Ouvrir le PDF et vérifier que **tout le texte évaluatif apparaît en bleu** et que rien de factuel n'a été coloré par erreur.

- [ ] **Step 13: Commit**

```bash
git add sections/04-aktienkurs.tex sections/05-handlungsoptionen.tex sections/06-cashflow.tex
git commit -m "feat: Ausarbeitung der 17 Wertungsabschnitte (blaue Zonen)"
```

---

### Task 12: Quellen, KI-Nutzung, Erklärung et vérification finale

**Files:**
- Modify: `sections/90-quellen.tex`, `sections/91-ki-nutzung.tex`, `sections/92-erklaerung.tex`
- Create: `README.md`

**Interfaces:**
- Consumes: `refs/MANIFEST.md` (Task 2), l'ensemble des notes de bas de page du document.
- Produces: le PDF final vérifié.

- [ ] **Step 1: Rédiger `sections/90-quellen.tex`, scindé en deux blocs**

```latex
\section*{Quellenverzeichnis}
\addcontentsline{toc}{section}{Quellenverzeichnis}
\label{sec:quellen}

\subsection*{Berichte der Aumann AG}
\begin{enumerate}
\item Aumann AG: Gesch\"aftsbericht 2022. Beelen 2023.
      \url{https://www.aumann.com/fileadmin/templates/downloads/finanzberichte/2022_aag-geschaeftsbericht-hp_englisch.pdf}
      (abgerufen am <Abrufdatum aus refs/MANIFEST.md>).
% ... je ein Eintrag fuer GB 2023, GB 2024, GB 2025,
%     Quartalsmitteilung Q1 2025, Halbjahresfinanzbericht H1 2025,
%     Quartalsmitteilung Q3 2025, Quartalsmitteilung Q1 2026
\end{enumerate}

\subsection*{Externe Quellen und Analysten}
\begin{enumerate}
\item <Urheber>: <Titel>. <Datum>. \url{<URL>} (abgerufen am <Datum>).
\end{enumerate}
```

Les URL et les Abrufdaten sont repris **littéralement** de `refs/MANIFEST.md`.

Ajouter en tête du bloc « Berichte der Aumann AG » la note de limitation exigée par le spec :

```latex
\noindent\emph{Hinweis:} Die Gesch\"aftsberichte der Gesch\"aftsjahre 2022 bis 2025
werden von der Aumann~AG auf der Investor-Relations-Seite ausschlie{\ss}lich in
englischer Sprache ver\"offentlicht. Die Zahlenangaben dieser Arbeit sind diesen
englischsprachigen Fassungen entnommen; die Bezeichnungen wurden f\"ur die
vorliegende Arbeit ins Deutsche \"ubertragen.
\medskip
```

- [ ] **Step 2: Contrôler que chaque source citée en note figure dans le Quellenverzeichnis**

Run:
```bash
grep -ohE '\\quelle(GB|QM|Web)\{[^}]*\}' sections/*.tex | sort -u
```
Confronter la liste obtenue au Quellenverzeichnis : chaque `\quelleGB{<jahr>}` doit avoir son Geschäftsbericht listé, chaque `\quelleQM{<bezeichnung>}` son rapport trimestriel, chaque `\quelleWeb` son entrée externe. Corriger les manquants.

- [ ] **Step 3: Rédiger `sections/91-ki-nutzung.tex`**

```latex
\section*{Hinweis zur Nutzung von KI-Werkzeugen}
\addcontentsline{toc}{section}{Hinweis zur Nutzung von KI-Werkzeugen}
\label{sec:ki}

Bei der Erstellung dieser Arbeit wurden folgende KI-Werkzeuge eingesetzt:

\begin{itemize}
\item \textbf{Claude Code} (Anthropic), Modell Claude~Opus~5, eingesetzt im
      Zeitraum Juli~2026.
\end{itemize}

Der Einsatz beschr\"ankte sich auf das Auswerten und Analysieren der
ver\"offentlichten Gesch\"afts- und Quartalsberichte der Aumann~AG, auf die
Recherche und Pr\"ufung erg\"anzender Quellen sowie auf die abschlie{\ss}ende
Rechtschreib- und Grammatikpr\"ufung. Die zahlenm\"a{\ss}igen Auswertungen
wurden anhand der in Abschnitt~\ref{sec:quellen} genannten Prim\"arquellen
nachvollzogen und gepr\"uft.
```

**Note pour l'exécutant :** cette formulation doit correspondre à l'usage réel. Si l'utilisateur conserve tout ou partie de la prose des zones bleues telle que générée, la phrase de restriction doit être amendée en conséquence — c'est une décision de l'utilisateur, à lui soumettre explicitement à la fin de la tâche, **sans la trancher à sa place**.

- [ ] **Step 4: Rédiger `sections/92-erklaerung.tex`**

```latex
\section*{Eidesstattliche Erkl\"arung}
\addcontentsline{toc}{section}{Eidesstattliche Erkl\"arung}
\label{sec:erklaerung}

Ich versichere, dass ich die vorliegende Hausarbeit selbstst\"andig verfasst
und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe.
Alle Stellen, die w\"ortlich oder sinngem\"a{\ss} aus ver\"offentlichten oder
nicht ver\"offentlichten Schriften entnommen wurden, sind als solche kenntlich
gemacht. Der Einsatz von KI-Werkzeugen ist in Abschnitt~\ref{sec:ki} offengelegt.

\vspace{2cm}
\noindent\begin{tabular}{@{}p{6cm}p{6cm}@{}}
\hrulefill & \hrulefill\\
Ort, Datum & Unterschrift\\
\end{tabular}
```

- [ ] **Step 5: Écrire `README.md`**

```markdown
# Hausarbeit "Finanzwirtschaft für Ingenieure" — Aumann AG

Hochschule Kaiserslautern, Sommersemester 2026. Abgabe: 04.08.2026.

## Bauen

    ./scripts/fetch_reports.sh     # Quelldokumente nach refs/ laden
    python3 scripts/fetch_kurs.py  # Kursreihe nach data/aktienkurs.csv
    python3 scripts/cashflow.py    # Tabellen der Aufgabe 6 nach data/
    ./build.sh                     # -> out/hausarbeit.pdf

## Prüfen

    cd scripts && python3 -m unittest discover -p 'test_*.py' -v
    python3 scripts/check_literals.py sections/*.tex

## Aufbau

- `data/kennzahlen.tex` — einzige Quelle aller Zahlenwerte, je Makro mit Quellenangabe.
- `data/cf_*.tex`, `data/vergleich2028.tex` — von `scripts/cashflow.py` erzeugt, nicht von Hand ändern.
- `sections/` — ein Abschnitt je Aufgabenteil.
- Blaue Textstellen (`\bk{}`) sind die Wertungsabschnitte. Vor der Abgabe prüfen und
  anschließend in `preamble.tex` `\newcommand{\bk}[1]{#1}` setzen, um die Färbung zu entfernen.
```

- [ ] **Step 6: Lancer la suite de tests complète**

Run: `cd scripts && python3 -m unittest discover -p 'test_*.py' -v`
Expected: tous les tests passent (32 au total : 10 pour `check_literals`, 22 pour `cashflow`).

- [ ] **Step 7: Vérification finale de bout en bout**

Run:
```bash
python3 scripts/cashflow.py && python3 scripts/check_literals.py sections/*.tex && ./build.sh
pdfinfo out/hausarbeit.pdf | grep -E 'Pages|Page size'
grep -n '\\bk{}' sections/*.tex || echo "Keine leeren Wertungszonen."
grep -rn 'TODO\|TBD\|EINTRAGEN\|<wert>\|<seite>\|<Abrufdatum' sections/ data/kennzahlen.tex || echo "Keine Platzhalter."
```
Expected: le build réussit ; aucune zone bleue vide ; **`EINTRAGEN` ne doit subsister que dans les trois placeholders du Deckblatt** (nom, matricule, filière) — tout autre placeholder est un défaut à corriger avant de rendre la main.

- [ ] **Step 8: Vérifier la couverture des six parties de l'énoncé**

Run: `pdftotext out/hausarbeit.pdf - | grep -nE '^[0-9]+(\.[0-9]+)* ' | head -60`
Confronter à la numérotation de l'énoncé : 1.1–1.5, 2.1–2.7, 3.1–3.6, 4, 5.1.1–5.1.3, 5.2.1–5.2.3, 5.3.1–5.3.3, 5.4.1–5.4.2, 6.1–6.3. Toute sous-question absente est un défaut bloquant.

- [ ] **Step 9: Commit final**

```bash
git add sections/90-quellen.tex sections/91-ki-nutzung.tex sections/92-erklaerung.tex README.md
git commit -m "feat: Quellenverzeichnis, KI-Hinweis, Erklaerung und Projektdokumentation"
```

- [ ] **Step 10: Rendre la main à l'utilisateur**

Livrer à l'utilisateur, dans cet ordre :
1. Le chemin du PDF (`out/hausarbeit.pdf`) et son nombre de pages.
2. La liste des 17 zones bleues, dans l'ordre du document, avec leur section et leur limite de longueur.
3. Les trois placeholders du Deckblatt restant à renseigner.
4. Toute valeur qui n'a pas pu être extraite d'une source (macros marquées `n.\,v.`).
5. Le rappel : neutraliser le bleu via `\newcommand{\bk}[1]{#1}` dans `preamble.tex` avant l'envoi.
6. La question ouverte de l'étape 3 de cette tâche sur la formulation exacte du `Hinweis zur Nutzung von KI-Werkzeugen`.
