# Hausarbeit „Finanzwirtschaft für Ingenieure" — Aumann AG

Master-Hausarbeit, HS Kaiserslautern, Sommersemester 2026.
**Abgabe: Dienstag, 04.08.2026, 23:59** per Mail an juergen.bott@hs-kl.de UND katharina.moor@hs-kl.de.
Verfasser: Yann Bertin Kamdem Bobda, Matrikelnummer 885893.

Aufgabenstellung: [Docu_Hskl/Hausarbeit SoSe2026 Aumann.pdf](Docu_Hskl/Hausarbeit%20SoSe2026%20Aumann.pdf) — 6 Teile, 100 Punkte.
Vorlesungsunterlage (Terminologie): [Docu_Hskl/FfE_251124_123712.pdf](Docu_Hskl/FfE_251124_123712.pdf), Prof. Dr. Jürgen Bott.

**Sprache des Dokuments: Deutsch.** Konversationssprache mit dem Verfasser: Französisch.

## Bauen und Prüfen

```bash
./build.sh                                   # -> out/hausarbeit.pdf
python3 scripts/check_literals.py sections/*.tex   # muss mit 0 enden
python3 scripts/check_footnote_pages.py      # Fussnoten-Duplikate je Seite
python3 scripts/check_footnote_groups.py     # Gruppen ueber einer Seitengrenze
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 76 Tests
python3 scripts/cashflow.py                  # erzeugt data/cf_*.tex neu
```

Alle LaTeX-Pakete sind installiert (KOMA, pgfplots, booktabs, siunitx 3.0.46,
babel/ngerman, eurosym, amsmath, amssymb, csquotes, microtype, footmisc, needspace,
placeins, latexmk). Python 3.10, nur stdlib.

## Die drei nicht verhandelbaren Regeln

1. **Schwarz / Blau.** Schwarz = Fakten, Zahlen, Tabellen, Struktur, Quellen. Blau
   (`\bk{...}`) = jede wertende oder interpretierende Prosa. Der Verfasser prüft und
   überschreibt die blauen Stellen vor der Abgabe. Details:
   [sections/CLAUDE.md](sections/CLAUDE.md).
2. **Keine Zahl direkt in `sections/*.tex`.** Jeder Wert kommt aus einem Makro in
   `data/kennzahlen.tex` oder aus einer generierten Tabelle in `data/`. Erzwungen durch
   `scripts/check_literals.py`. Details: [data/CLAUDE.md](data/CLAUDE.md).
3. **Nichts erfinden.** Keine Zahl, keine Quelle, keine Analystenaussage ohne Beleg.
   Fehlt ein Wert, wird die Lücke im Dokument benannt, nicht gefüllt. Seitenangaben
   werden gegen die **gedruckte** Seite geprüft: [refs/CLAUDE.md](refs/CLAUDE.md).

## Stand der Arbeit

| Teil | Inhalt | Status |
|---|---|---|
| 1 | Unternehmen, Organe, Eigentümer, Analysten | fertig (schwarz) |
| 2 | Jahresabschluss 2024 (2.1–2.7) + Kursdiagramm | fertig (schwarz) |
| 3 | Strategische Maßnahmen (3.1–3.6) | fertig (schwarz) |
| 4 | Beurteilung des Kursverlaufs | fertig (eine blaue Zone) |
| 5 | Handlungsoptionen A/B/C | fertig (44 blaue Zellen ausgeschrieben) |
| 6 | Flow-Analyse | fertig, mit Margen-Sensitivität |
| — | Quellen, KI-Hinweis, Erklärung | fertig |

Inhaltlich vollständig: 28 Seiten, `./build.sh` läuft ohne Fehler und ohne
Warnung. Offen ist allein die inhaltliche Prüfung der blauen Stellen durch den
Verfasser.

**Nachtrag 31.07.2026 — drei der vier offenen Lücken geschlossen.** Die
Jahreshöchst- und Jahrestiefstkurse 2021–2025 liegen vor (Yahoo Finance, für
2022–2025 gegen die Geschäftsberichte validiert), Abbildung 1 zeichnet den Kurspfad
über dreizehn Stützpunkte an ihrem tatsächlichen Handelstag, und die
Aktionärsstruktur ist auf den
tatsächlichen Stand der IR-Seite korrigiert — der Entwurf hatte zwei Fassungen der
Seite vermischt. Nicht geschlossen: Ratings und Kursziele der Analysten sowie ein
durch eine Primärquelle gedeckter Schlusskurs 2021. Details in
[docs/befunde-und-entscheidungen.md](docs/befunde-und-entscheidungen.md).

Ausführlicher Plan: [docs/superpowers/plans/2026-07-29-hausarbeit-aumann.md](docs/superpowers/plans/2026-07-29-hausarbeit-aumann.md)
(alle zwölf Tasks ausgeführt). Spezifikation:
[docs/superpowers/specs/2026-07-29-hausarbeit-aumann-design.md](docs/superpowers/specs/2026-07-29-hausarbeit-aumann-design.md).
**Befunde und getroffene Entscheidungen:**
[docs/befunde-und-entscheidungen.md](docs/befunde-und-entscheidungen.md) — dort stehen die
inhaltlichen Ergebnisse, die eine neue Sitzung sonst neu herleiten müsste.

## Verzeichnisse

| Pfad | Inhalt | Eigene CLAUDE.md |
|---|---|---|
| `sections/` | ein Abschnitt je Aufgabenteil | ja |
| `data/` | Zahlenschicht, einzige Quelle aller Werte | ja |
| `scripts/` | Wachhund, Cash-Flow-Rechnung, Beschaffung | ja |
| `refs/` | heruntergeladene Aumann-Berichte (gitignoriert) | ja |
| `docs/` | Spezifikation, Plan, Befunde | — |
| `out/` | Bauergebnis (gitignoriert) | — |

## Vor der Abgabe

1. Alle blauen Zonen prüfen und in eigenen Worten überschreiben.
2. `./build.sh`, dann `python3 scripts/check_footnote_pages.py` **und**
   `python3 scripts/check_footnote_groups.py` — das Überschreiben verschiebt
   Seitenumbrüche und kann Fußnoten-Duplikate auf derselben Seite neu entstehen lassen
   (erstes Skript) oder bestehende Zusammenführungen ungültig machen (zweites Skript);
   siehe `sections/CLAUDE.md`, Abschnitt „Fussnotennummer wiederverwenden“.
3. In `preamble.tex` `\newcommand{\bk}[1]{#1}` setzen — entfernt die Blaufärbung.
4. Deckblattangaben und das Studiengangs-Label prüfen.
5. Den Hinweis zur Nutzung von KI-Werkzeugen (`sections/91-ki-nutzung.tex`) an
   den tatsächlichen Gebrauch anpassen — die Formulierung ist eine Entscheidung
   des Verfassers, keine des Werkzeugs.
6. `./build.sh`, dann `out/hausarbeit.pdf` als Anhang versenden.

## Lokale Agenten- und Token-Strategie

Für diese Arbeit soll das lokale Modell bevorzugt werden, um Tokens zu sparen. Der lokale
Bridge ist über [scripts/ensure_ollama.sh](scripts/ensure_ollama.sh) verfügbar. Vor einer
lokalen Sitzung bitte einmal ausführen:

```bash
./scripts/ensure_ollama.sh
```

Weitere Details stehen in [docs/local-agent-workflow.md](docs/local-agent-workflow.md).
