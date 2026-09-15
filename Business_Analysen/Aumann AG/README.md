# Hausarbeit „Finanzwirtschaft für Ingenieure" — Aumann AG

Hochschule Kaiserslautern, Sommersemester 2026. Abgabe: 04.08.2026.
Ergebnis des Baus: `out/hausarbeit.pdf`.

## Bauen

```bash
./scripts/fetch_reports.sh     # Quelldokumente nach refs/ laden (gitignoriert)
python3 scripts/cashflow.py    # Tabellen und Makros der Aufgabe 6 nach data/
./build.sh                     # -> out/hausarbeit.pdf
```

`scripts/fetch_kurs.py` bleibt als dokumentierter Fehlversuch im Repo: Am
29.07.2026 war keine Kursquelle erreichbar, weshalb die Kursreihe in
`data/aktienkurs.csv` aus den Geschäftsberichten stammt.

## Prüfen

```bash
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 53 Tests
python3 scripts/check_literals.py sections/*.tex            # muss mit 0 enden
```

## Aufbau

| Pfad | Inhalt |
|---|---|
| `sections/` | ein Abschnitt je Aufgabenteil, `00`–`06` plus Anhänge `90`–`92` |
| `data/kennzahlen.tex` | einzige Quelle aller von Hand erfassten Zahlen, je Makro mit Quelle und gedruckter Seite |
| `data/cf_*.tex`, `data/vergleich2028.tex`, `data/marge_sensitivitaet.tex`, `data/cf_makros.tex` | von `scripts/cashflow.py` erzeugt — **nicht von Hand ändern** |
| `refs/` | heruntergeladene Aumann-Berichte, gitignoriert; `MANIFEST.md` hält URL, Seitenzahl und SHA-256 fest |
| `docs/` | Spezifikation, Plan und das Befundprotokoll |

Jede Verzeichnisebene trägt eine eigene `CLAUDE.md` mit ihren Konventionen.

## Die drei Regeln des Projekts

1. **Schwarz / Blau.** Schwarz ist alles Belegbare, blau (`\bk{...}`) jede
   Wertung. Die Faktenbasis einer blauen Stelle steht als `%`-Kommentar
   darüber und erscheint nie im PDF.
2. **Keine Zahl direkt in `sections/*.tex`.** Jeder Wert kommt aus einem Makro
   in `data/`. Erzwungen durch `scripts/check_literals.py`.
3. **Nichts erfinden.** Fehlt ein Wert in den Quellen, wird die Lücke im
   Dokument benannt — siehe den Abschnitt „Nicht verfügbare Angaben" im
   Quellenverzeichnis.

## Vor der Abgabe

1. Alle blauen Stellen prüfen und in eigenen Worten überschreiben.
2. In `preamble.tex` `\newcommand{\bk}[1]{#1}` setzen — das entfernt die
   Blaufärbung, ohne den Text anzutasten.
3. Deckblattangaben und das Studiengangs-Label prüfen.
4. Den Hinweis zur Nutzung von KI-Werkzeugen an den tatsächlichen Gebrauch
   anpassen.
5. `./build.sh`, dann `out/hausarbeit.pdf` per Mail an juergen.bott@hs-kl.de
   **und** katharina.moor@hs-kl.de senden.
