# data/ — die Zahlenschicht

Hier steht **jeder** Zahlenwert des Dokuments genau einmal. Der Fließtext, die Tabellen und
die Diagramme lesen dieselbe Quelle, damit Text und Tabelle nie auseinanderlaufen können.

| Datei | Inhalt | Herkunft |
|---|---|---|
| `kennzahlen.tex` | alle Makros, je mit Quelle und gedruckter Seite im Kommentar | von Hand, aus `../refs/` |
| `basis.json` | Bezugswerte für die Cash-Flow-Rechnung | von Hand |
| `kursverlauf.csv` | die 13 Stützpunkte der Kursabbildung, nach Datum | gemischt, siehe unten |
| `cf_a.tex`, `cf_b.tex`, `cf_c.tex` | Tabellenkörper Aufgabe 6.1 | **generiert** |
| `vergleich2028.tex` | Tabellenkörper Aufgabe 6.3 | **generiert** |
| `irr_sensitivitaet.tex` | IRR nach Horizontlänge | **generiert** |
| `marge_sensitivitaet.tex` | EBITDA-Marge 2028 nach Umsatzbasis | **generiert** |
| `cf_makros.tex` | Ergebnisse der Rechnung als Makros für Teil 5 | **generiert** |

Die generierten Dateien niemals von Hand ändern — `python3 ../scripts/cashflow.py`
überschreibt sie.

### `kursverlauf.csv` — die einzige Datei mit gemischter Herkunft

Sie speist Abbildung 1 **und** Tabelle 6. Dreizehn Zeilen, **streng nach Datum geordnet**:
der Schlusskurs des letzten Handelstages 2021 als Ausgangspunkt, danach je Jahr 2022 bis 2025
das Jahreshoch, das Jahrestief und der Jahresschlusskurs — in der Reihenfolge ihres
Eintretens.

**Die Reihenfolge innerhalb eines Jahres ist nicht fest.** 2022 und 2024 kam zuerst das Hoch,
2023 und 2025 zuerst das Tief. Wer nach dem Schema Hoch–Tief sortiert, lässt die Linie
rückwärts laufen; pgfplots zeichnet das kommentarlos. `test_stuetzpunkte_sind_streng_nach_datum_geordnet`
hält die Invariante fest.

| Spalte | Herkunft |
|---|---|
| `kurs` bei `art=S`, 2022–2025 | Geschäftsberichte 2023 (S. 14), 2024 und 2025 (je S. 13) |
| `kurs` bei `art=S`, 2021 | **nur** Yahoo Finance — siehe Vorbehalt |
| `kurs` bei `art=H`/`art=T` | Yahoo Finance, Tagesbars, Intraday-Extrema |
| `datum`, `x` | Yahoo Finance, Tagesbars; `x` ist das Datum als Dezimaljahr |

**Warum die externe Quelle zitierfähig ist:** Für 2022 bis 2025 stimmen die Schlusskurse auf
den Cent mit den Geschäftsberichten überein. Die Gegenprobe läuft bei jedem Testlauf mit,
weil die `\KursSilvester…`-Makros die Werte der Berichte tragen.

**Vorbehalt 2021.** Der Geschäftsbericht 2022 enthält überhaupt keine Angabe zum
Jahresschlusskurs (geprüft am 31.07.2026), der Bericht 2023 führt als Vorjahreswert bereits
2022. Der Wert 13,68 € ist als einziger nicht gegengeprüft. Im Dokument steht das in
Abschnitt 2.7 und im Quellenverzeichnis.

**Was die Abbildung nicht leisten kann.** Hoch 2023 (28.12.), Schlusskurs 2023 (29.12.) und
Hoch 2024 (02.01.) liegen fünf Handelstage auseinander und fallen auf einer Achse über vier
Jahre zu einer einzigen Spitze zusammen; ebenso Schlusskurs 2022 (30.12.) und Tief 2023
(05.01.). Deshalb trägt Tabelle 6 den Handelstag jedes Extremwerts — sie ist die genaue
Lesart der Abbildung, nicht bloss deren Wiederholung.

**Die Doppelung mit `kennzahlen.tex` ist gewollt und abgesichert.** Dieselben Werte stehen
als Makro dort (für Fließtext und Tabelle) und als Spalte hier (für pgfplots, das keine
LaTeX-Makros lesen kann). `../scripts/test_data_consistency.py` prüft beide Richtungen: dass
jeder Stützpunkt zu seinem Makro passt, und dass kein Makro fehlt.

## Konventionen für `kennzahlen.tex`

- **Rohwert im englischen Format**, in voller veröffentlichter Genauigkeit: `312.346`.
  Die Rundung liegt in der Ausgabe, nicht im gespeicherten Wert. Sonst weichen die
  berechneten Veränderungsraten von denen des Geschäftsberichts ab — aus gerundeten
  Werten ergäbe sich 7,8 % statt der veröffentlichten 7,9 %.
- **Jede Zeile trägt Quelle und gedruckte Seite** im Zeilenkommentar. Berechnete Werte
  tragen stattdessen ihre Formel.
- **Jahresziffern römisch**: `ZFIII` = Geschäftsjahr 2023, `ZFIV` = 2024, `ZFV` = 2025.
- Beträge in Mio. EUR, sofern nicht anders vermerkt. Prozentwerte als Zahl ohne Zeichen.

## Ausgabemakros (definiert in `../preamble.tex`)

| Makro | Zweck | Beispiel |
|---|---|---|
| `\MioEUR{}` | Fließtext, eine Nachkommastelle | `312,3 Mio. €` |
| `\MioEURexakt{}` | Rechenwege, volle Genauigkeit | `312,346 Mio. €` |
| `\Pct{}` | Prozentwert | `11,7 %` |
| `\EURje{}` | Betrag je Aktie, zwei Nachkommastellen | `10,62 €` |
| `\num{}` | Tabellenzellen (siunitx) | |

## Zentrale Werte

Umsatz 2024 `312.346` (+7,9 %) · Auftragseingang `200.057` (−41,1 %) · operatives EBITDA
`36.417` (+71,0 %) · Konzernjahresüberschuss `21.506` (+124,4 %) · Dividende `0.22` €/Aktie
· Schlusskurs 31.12.2024 `10.62` € · Dividendenrendite `2.07` % · Nettoliquidität `138.2`
(2024) und `148.1` (2025).

**Operatives EBITDA = das bereinigte (adjusted) EBITDA.** Aumann weist beide Größen
nebeneinander aus; die Begründung dieser Wahl steht in
[../docs/befunde-und-entscheidungen.md](../docs/befunde-und-entscheidungen.md).

## `basis.json`

Speist `cashflow.py`. `umsatz_2025` = 203.985 und `ebitda_op_2025` = 27.278 bilden die
Bezugsbasis der EBITDA-Marge für 2028. **Achtung:** der Umsatz 2025 war um 34,7 %
eingebrochen; ihn konstant fortzuschreiben schmeichelt der Marge, weil der Nenner klein
ist. Abschnitt 6 legt das im Absatz „Vorbehalt zur EBITDA-Marge Ende 2028" offen und
zeigt die Sensitivität in `marge_sensitivitaet.tex`: Basisfall A 21,2 / B 20,2 / C 18,3 %,
bei +3 % Umsatzwachstum p. a. 19,4 / 18,5 / 16,7 %, auf dem Umsatzniveau 2024 nur noch
13,9 / 13,2 / 11,9 %. `umsatz_2024` in `basis.json` ist allein dieser Vergleichsnenner.

## Einlesen der generierten Tabellen

Im Dokument werden die Tabellenkörper mit `\tabellenkoerper{data/cf_a}` eingelesen,
**nicht** mit `\input`. `\input` hängt hinter den Dateiinhalt ein `\relax`; innerhalb
einer Tabelle beginnt dieses `\relax` bereits die nächste Zelle, sodass das folgende
`\bottomrule` mit „Misplaced \noalign" scheitert. Das Makro steht in `../preamble.tex`.
