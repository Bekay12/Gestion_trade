# Anlageanalyse AstraZeneca PLC und Dr. Reddy's Laboratories Ltd.

Zwei Titel, zwei getrennte Voten, ein Dokument: `out/analyse.pdf` (43 Seiten).
Gebaut nach dem Muster in
`~/.claude/skills/startup-investment-analyzer/references/latex-project-pattern.md`.

## Bauen

```bash
./build.sh          # erzeugt die Datenschicht neu, prueft sie und baut das PDF
```

`build.sh` bricht ab, sobald ein Zahlenwert nicht auf der Seite steht, die er
angibt, oder eine Zahl im Fliesstext auftaucht, die nicht aus der Zahlenschicht
kommt. Nach jedem Bau, der Seitenumbrueche verschiebt:

```bash
python3 scripts/fussnoten_gruppen.py --schleife   # Fussnoten neu gruppieren
./build.sh
python3 scripts/check_footnote_pages.py
python3 scripts/check_footnote_groups.py
cd scripts && python3 -m unittest discover -p 'test_*.py'
```

## Quellen beschaffen

`refs/` ist gitignoriert (rund 300 MB). Einmalig:

```bash
python3 scripts/hole_quellen.py        # EDGAR-Einreichungen beider Emittenten
python3 scripts/seiten_zerlegen.py     # in gedruckte Seiten zerlegen
python3 scripts/kurse.py               # Kursreihen und Handelstage der Extrema
```

Die Geschaeftsberichte von AstraZeneca liegen zusaetzlich als PDF von
astrazeneca.com unter `refs/azn-ar-<jahr>.pdf`; die SEC-Fassung traegt keine
gedruckten Seitenzahlen und ist als Belegquelle unbrauchbar.

## Die fuenf Wachhunde

| Skript | faengt |
|---|---|
| `pruefe_seiten.py` | einen Wert, der nicht auf der zitierten Seite steht (355 geprueft) |
| `reihe_azn.py` / `reihe_rdy.py` | eine Mehrjahresreihe, die um eine Spalte verrutscht ist (148 Kettengleichungen) |
| `check_literals.py` | eine Zahl, die in die Prosa getippt statt aus der Zahlenschicht gelesen wurde |
| `check_footnote_pages.py` | dieselbe Quelle zweimal als eigene Fussnote auf einer Seite |
| `check_footnote_groups.py` | eine wiederverwendete Fussnote, deren Anker auf einer anderen Seite liegt |

## Aufbau

| Pfad | Inhalt |
|---|---|
| `analyse.tex`, `preamble.tex` | Rahmen, Satz, Zitiermakros |
| `sections/` | ein Abschnitt je Teil; enthaelt keine Zahl |
| `data/` | erzeugte Zahlenschicht und Tabellenkoerper; nie von Hand |
| `scripts/` | Beschaffung, Zahlenschicht, Rechnung, Wachhunde |
| `refs/` | Einreichungen, gitignoriert |
| `docs/` | Befunde, getroffene Entscheidungen, Rechercheberichte |
| `out/` | Bauergebnis, gitignoriert |

Firmenspezifisch sind `scripts/kennzahlen.py`, `scripts/flow.py`,
`scripts/pruefe_seiten.py`, `scripts/reihe_*.py` und `scripts/seiten_zerlegen.py`.
Alles Uebrige ist aus `Alamos/` uebernommen.

## Die beiden Voten in einem Satz

- **AstraZeneca**: beobachten, nicht kaufen zu 166,14 USD; Schwelle 152,85 USD
  auf zehn Jahre.
- **Dr. Reddy's**: meiden zu 12,19 USD, weil der Preis nicht pruefbar ist: Das
  Unternehmen beziffert den Beitrag des auslaufenden Lenalidomid-Geschaefts
  nirgends.

Begruendung, Annahmen und Gegenargumente stehen in Teil 7 des jeweiligen Teils.
