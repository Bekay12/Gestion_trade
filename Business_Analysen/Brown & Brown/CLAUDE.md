# Anlageanalyse — Brown & Brown, Inc.

**Keine Prüfungsleistung.** Das Projekt entstand als Parallelarbeit zur Hausarbeit über
die Aumann AG (`../Aumann AG/`) und dient inzwischen der eigenen Anlageentscheidung des
Verfassers. Übernommen wurde die Methode, nicht der Prüfungsrahmen: kein Hochschulwappen,
keine Matrikelnummer, kein KI-Nutzungshinweis. An seine Stelle tritt der Abschnitt
„Methode und Grenzen“ (`sections/91-methode.tex`), der benennt, was das Dokument leistet
und was nicht.

Gegenstand ist eine bereits getroffene Entscheidung, nicht drei hypothetische Optionen:
die Übernahme der Accession Risk Management Group zum 1.8.2025 für 9,6 Mrd. USD. Deshalb
gibt es keine Handlungsoptionen A/B/C.

Sprache des Dokuments: Deutsch. Konversationssprache mit dem Verfasser: Französisch.
Basisjahr: Geschäftsjahr 2025 (Form 10-K, eingereicht 12.02.2026).

**Was eine gute Note belegt und was nicht.** Die Methode ist als Hausarbeit sehr gut
bewertet worden. Das validiert die Rigorosität — Belegpflicht, deklarierte Annahmen,
benannte Lücken. Es validiert nicht die Treffsicherheit der Anlageurteile: Eine
Hausarbeit wird auf die Qualität der Argumentation benotet, nicht auf die Rendite. Die
Skripte verhindern erfundene Zahlen; die Gewichtung der Szenarien bleibt eine
Entscheidung des Verfassers.

## Bauen und Prüfen

```bash
./build.sh                                    # -> out/analyse.pdf (20 Seiten)
python3 scripts/kennzahlen.py                 # erzeugt data/kennzahlen.tex neu
python3 scripts/flow.py                       # erzeugt data/flow_*.tex neu
python3 scripts/fetch_kurs.py                 # holt die Kursreihe neu (Nasdaq)
python3 scripts/pruefe_seiten.py              # jede Zahl gegen ihre gedruckte Seite
python3 scripts/check_literals.py sections/*.tex
python3 scripts/check_footnote_pages.py       # nach jedem Bauen
python3 scripts/check_footnote_groups.py      # nach jedem Bauen
python3 scripts/fussnoten_gruppen.py --schleife   # löst Befunde der beiden auf
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 35 Tests
```

Stand 28.08.2026: alle sechs Prüfungen grün, keine LaTeX-Warnung.

## Die vier nicht verhandelbaren Regeln

1. **Drei Farben, drei Verantwortlichkeiten.** Schwarz = Fakten, Zahlen, Tabellen,
   Quellen. Blau (`\bk{...}`) = Einschätzung im laufenden Text, vom Verfasser zu prüfen.
   Rot (`\vd{...}`) = das Votum in Teil 7, eine abgeleitete Empfehlung, ausdrücklich kein
   Beleg. Für den eigenen Gebrauch ist die Färbung wertvoller als für eine Abgabe: Sie
   zeigt auf einen Blick, wo das Dokument belegt, wo es einschätzt und wo es empfiehlt.
   Nur für eine Weitergabe an Dritte in `preamble.tex` `\newcommand{\bk}[1]{#1}` und
   `\newcommand{\vd}[1]{#1}` setzen.
2. **Keine Zahl direkt in `sections/*.tex`.** Jeder Wert kommt aus einem Makro in
   `data/kennzahlen.tex` (erzeugt von `scripts/kennzahlen.py`) oder aus einer generierten
   Tabelle in `data/`. Erzwungen durch `scripts/check_literals.py`.
3. **Jede Zahl trägt ihre gedruckte Seite.** `scripts/edgar_pages.py` zerlegt die
   EDGAR-HTML-Einreichung in gedruckte Seiten, `scripts/pruefe_seiten.py` prüft für jeden
   der 176 Rohwerte, dass er auf der zitierten Seite tatsächlich vorkommt. Beim ersten
   Lauf waren 23 Seitenangaben falsch — geraten, nicht gelesen.
4. **Nichts erfinden.** Fehlt ein Wert, wird die Lücke benannt (Abschnitt „Nicht
   verfügbare Angaben“), nicht gefüllt. Kein WACC, kein Konsensrating, keine
   Wachstumsschätzung des Verfassers.

## Aufbau des Dokuments

| Teil | Inhalt |
|---|---|
| 1 | Unternehmen, Organe, Eigentümer, Analystenabdeckung, Lücken |
| 2 | Jahresabschluss 2025, organisches Wachstum als Frühindikator, EBITDAC berichtet/bereinigt |
| 3 | Accession-Übernahme, Segmentneuordnung, Finanzierung, Kapitalallokation |
| 4 | Beurteilung des Kursverlaufs 2021–2026 |
| 5 | Flow-Analyse: Entschuldungspfad und Anleger-IRR über 5/10/15 Jahre |
| 6 | Anlageurteil, nicht verfügbare Angaben |
| 7 | **Verdikt** (rot): Votum, Begründung, Einstiegsschwelle, Auslöser, Grenzen |
| — | Quellenverzeichnis, Methode und Grenzen |

## Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `sections/` | ein Abschnitt je Teil |
| `data/` | Zahlenschicht, erzeugt; einzige Quelle aller Werte |
| `scripts/` | Beschaffung, Zahlenschicht, Flow-Rechnung, vier Wachhunde |
| `refs/` | SEC-Einreichungen als HTML und als seitenweiser Text (gitignoriert) |
| `docs/` | Befunde und getroffene Entscheidungen |
| `out/` | Bauergebnis (gitignoriert) |

## Teil 7 ist der feste Kern für jede weitere Firma

Der Verdikt-Teil hat bei jeder Analyse dieselben fünf Unterabschnitte, in derselben
Reihenfolge, damit zwei Berichte vergleichbar bleiben:

| 7.x | Inhalt |
|---|---|
| 7.1 | **Votum** — ein Satz: kaufen / beobachten / meiden, mit Kurs und Horizont |
| 7.2 | **Begründung aus den Befunden** — nur Größen, die oben schon stehen |
| 7.3 | **Einstiegsschwelle** — Kurs, bei dem der IRR die Hürde trifft, je Szenario und Horizont (`scripts/flow.py`, `schwellenkurs()`) |
| 7.4 | **Auslöser, die das Votum drehen** — beobachtbar, in beide Richtungen |
| 7.5 | **Was das Votum nicht wissen kann** — inklusive der Vorbehalte gegen das eigene Votum |

Die Hürde ist **nie** ein konstruierter WACC, sondern eine belegte Größe des Unternehmens
selbst (`flow.huerde()` = Eigenkapitalrendite auf das durchschnittliche Eigenkapital).
Ein IRR ohne Vergleichsmaßstab trägt kein Urteil.

Inhaltliche Ergebnisse, die eine neue Sitzung sonst neu herleiten müsste, stehen in
[docs/befunde-und-entscheidungen.md](docs/befunde-und-entscheidungen.md).
