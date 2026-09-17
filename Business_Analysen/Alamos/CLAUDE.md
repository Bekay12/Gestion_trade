# Anlageanalyse — Alamos Gold Inc. (TSX/NYSE: AGI)

**Keine Prüfungsleistung.** Das Projekt folgt der Methode der Hausarbeit über die
Aumann AG (`../Aumann AG/`) in der Fassung, die die Analyse über Brown & Brown
(`../Brown & Brown/`) daraus gemacht hat: Belegpflicht, deklarierte Annahmen,
benannte Lücken, drei Farben — aber kein Prüfungsrahmen. An die Stelle des
KI-Hinweises tritt der Abschnitt „Methode und Grenzen" (`sections/91-methode.tex`).

Gegenstand ist eine Eigentümerfrage, nicht ein Vergleich von Handlungsoptionen:
**AGI kaufen — und bis zu welchem Kurs?** Deshalb keine Optionen A/B/C.

Sprache des Dokuments: Deutsch. Basisjahr: Geschäftsjahr 2025 (Form 40-F,
eingereicht 26.03.2026), fortgeschrieben mit dem Zwischenbericht zum 30.06.2026.

**Was von Brown & Brown abweicht und warum.** Alamos hält mehr Kasse als
Finanzschulden. Ein Entschuldungspfad, die erste Rechnung des Musters, geht hier
ins Leere. An seine Stelle tritt die Finanzierungsfrage: Trägt der laufende
Cash-Flow das veröffentlichte Bauprogramm? Und die Szenarien laufen nicht über
das organische Wachstum, sondern über drei vom Unternehmen selbst **erzielte**
Goldpreise — bei einem Rohstoffproduzenten ist der Preis die tragende Größe.

## Bauen und Prüfen

```bash
./build.sh                                    # -> out/analyse.pdf (26 Seiten)
python3 scripts/hole_quellen.py               # EDGAR-Einreichungen nach refs/
python3 scripts/edgar_seiten.py               # zerlegt sie in gedruckte Seiten
python3 scripts/kennzahlen.py                 # erzeugt data/kennzahlen.tex neu
python3 scripts/flow.py                       # erzeugt data/flow_*.tex neu
python3 scripts/fetch_kurs.py                 # Kursreihe (Nasdaq)
python3 scripts/vergleich_kurs.py             # Aktie gegen Gold und Sektor
python3 scripts/pruefe_seiten.py              # jede Zahl gegen ihre gedruckte Seite
python3 scripts/reihe.py                      # Kettenprüfung der Zehnjahresreihe
python3 scripts/check_literals.py sections/*.tex
python3 scripts/check_footnote_pages.py       # nach jedem Bauen
python3 scripts/check_footnote_groups.py      # nach jedem Bauen
python3 scripts/fussnoten_gruppen.py --schleife   # löst Befunde der beiden auf
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 43 Tests
```

Stand 16.09.2026: alle fünf Prüfungen grün, 43 Tests grün, keine LaTeX-Warnung.

Für `hole_quellen.py` ist eine Kennung nötig, die die SEC verlangt:

```bash
export SEC_USER_AGENT='Vorname Nachname mail@example.com'
```

## Die vier nicht verhandelbaren Regeln

1. **Drei Farben, drei Verantwortlichkeiten.** Schwarz = Fakten, Zahlen, Tabellen,
   Quellen. Blau (`\bk{...}`) = Einschätzung im laufenden Text. Rot (`\vd{...}`) =
   das Votum in Teil 7, ausdrücklich kein Beleg. Für eine Weitergabe an Dritte in
   `preamble.tex` `\newcommand{\bk}[1]{#1}` und `\newcommand{\vd}[1]{#1}` setzen.
2. **Keine Zahl direkt in `sections/*.tex`.** Jeder Wert kommt aus einem Makro in
   `data/kennzahlen.tex` oder aus einer generierten Tabelle in `data/`. Erzwungen
   durch `scripts/check_literals.py`. Das gilt auch für scheinbar harmlose
   Prozentangaben unter 100, die der Wachhund durchließe.
3. **Jede Zahl trägt ihre gedruckte Seite.** `scripts/edgar_seiten.py` zerlegt die
   Einreichung, `scripts/pruefe_seiten.py` prüft für jeden der 203 Rohwerte, dass
   er auf der zitierten Seite vorkommt. Sechs Sekundärwerte (Analystenkonsens,
   Kursreihen) stehen in einem eigenen Block, werden nicht geprüft und sind im
   Dokument als Sekundärquelle gekennzeichnet.
4. **Nichts erfinden.** Fehlt ein Wert, wird die Lücke benannt (Abschnitt 6.3),
   nicht gefüllt. Kein WACC, keine eigene Goldpreisprognose. Aber: *nicht im
   Primärdokument* ist nicht *nicht beschaffbar* — Ratings und Kursziele stehen in
   keinem Emittentenbericht und wurden deshalb beschafft, nicht als Lücke geführt.

## Aufbau des Dokuments

| Teil | Inhalt |
|---|---|
| 1 | Unternehmen, Organe, Eigentümer, Analystenkonsens, Lücken |
| 2 | Jahresabschluss 2025, Marge je Unze als Frühindikator, Zehnjahresreihe |
| 3 | Island Gold Phase 3+, Lynn Lake, PDA, Reserven, Prognose gegen Ist |
| 4 | Kursverlauf 2021–2026 und Zerlegung des Einbruchs 2026 |
| 5 | Flow-Analyse: Finanzierungskapazität und Anleger-IRR über 5/10/15 Jahre |
| 6 | Anlageurteil, nicht verfügbare Angaben |
| 7 | **Verdikt** (rot): Votum, Begründung, Einstiegsschwelle, Auslöser, Grenzen |
| — | Quellenverzeichnis, Methode und Grenzen |

## Teil 7 hat dieselben fünf Unterabschnitte wie jede andere Analyse

| 7.x | Inhalt |
|---|---|
| 7.1 | **Votum** — ein Satz: kaufen / beobachten / meiden, mit Kurs und Horizont |
| 7.2 | **Begründung aus den Befunden** — nur Größen, die oben schon stehen |
| 7.3 | **Einstiegsschwelle** — Kurs, bei dem der IRR die Hürde trifft, je Szenario und Horizont |
| 7.4 | **Auslöser, die das Votum drehen** — beobachtbar, in beide Richtungen |
| 7.5 | **Was das Votum nicht wissen kann** — inklusive der Vorbehalte dagegen |

Die Hürde ist **nie** ein konstruierter WACC, sondern die Eigenkapitalrendite des
Unternehmens selbst — hier auf das **bereinigte** Ergebnis, weil das berichtete
eine Wertaufholung und einen Veräußerungsgewinn trägt (`flow.huerde()`).

## Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `sections/` | ein Abschnitt je Teil |
| `data/` | Zahlenschicht, erzeugt; einzige Quelle aller Werte |
| `scripts/` | Beschaffung, Zahlenschicht, Flow-Rechnung, fünf Wachhunde |
| `refs/` | EDGAR-Einreichungen als HTML und als seitenweiser Text (gitignoriert) |
| `quellen/`, `seiten/` | Vorgängerablage der leichtgewichtigen HTML-Auswertung |
| `docs/` | Befunde und getroffene Entscheidungen |
| `out/` | Bauergebnis (gitignoriert) |

Inhaltliche Ergebnisse, die eine neue Sitzung sonst neu herleiten müsste, stehen in
[docs/befunde-und-entscheidungen.md](docs/befunde-und-entscheidungen.md).
