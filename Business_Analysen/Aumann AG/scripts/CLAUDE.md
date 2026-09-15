# scripts/ — Werkzeuge

Reine Standardbibliothek, Python 3.10. Keine Abhängigkeit installieren.

| Datei | Zweck |
|---|---|
| `check_literals.py` | Wachhund gegen hartkodierte Zahlen in `../sections/*.tex` |
| `cashflow.py` | Cash-Flow-Analyse A/B/C, IRR, erzeugt die Tabellen in `../data/` |
| `fetch_reports.sh` | lädt die acht Aumann-Berichte nach `../refs/` |
| `fetch_kurs.py` | Versuch, eine monatliche Kursreihe zu beschaffen — schlägt fehl, siehe unten |
| `check_footnote_pages.py` | findet Quelle+Seite, die auf derselben PDF-Seite mehrfach zitiert wird, ohne die Fußnotennummer wiederzuverwenden |
| `check_footnote_groups.py` | findet den umgekehrten Fall: eine bestehende `\quelleMerken`/`\quelleErneut`-Gruppe, die über eine Seitengrenze gerutscht ist |
| `test_check_literals.py`, `test_cashflow.py`, `test_check_footnote_pages.py`, `test_check_footnote_groups.py`, `test_data_consistency.py` | 18 + 37 + 12 + 5 + 4 Tests |

```bash
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 76 Tests, muessen gruen sein
```

## `check_footnote_pages.py`

Fragt für jeden `\quelleGB`/`\quelleQM`/`\quelleWeb`-Aufruf in `../sections/*.tex` per
**SyncTeX** die tatsächlich gerenderte PDF-Seite ab — nicht geschätzt aus Zeilennummern,
sondern Ground Truth aus dem kompilierten Dokument. Meldet Fälle, in denen dieselbe
Quelle+Seite mehrfach auf derselben PDF-Seite auftaucht, ohne über `\quelleMerken`/
`\quelleErneut` (siehe `../preamble.tex` und `../sections/CLAUDE.md`, Abschnitt
„Fussnotennummer wiederverwenden“) zusammengeführt zu sein.

Voraussetzung: `./build.sh` mit `-synctex=1` (Standard, siehe `../build.sh`) wurde bereits
ausgeführt. `resolve_page()` ruft den `synctex`-Prozess auf und ist damit build-abhängig —
nur `parse_citations()` und `group_duplicates()` sind reine Funktionen und Teil der
schnellen, offline laufenden Testsuite.

**Der Schlüssel einer Webquelle ist Titel + Verzeichnisschlüssel** (das zweite Argument von
`\quelleWeb`, früher die URL). Ursprünglich vergab das Skript für jeden
`\quelleWeb`-Aufruf den festen Schlüssel `"Web Shares"` — solange das Dokument genau eine
Webquelle hatte, fiel das nicht auf. Mit mehreren Webquellen meldete es vier verschiedene
Belege auf derselben Seite als Duplikat. Wer diese Meldung befolgt hätte, hätte vier
unterschiedliche Fußnoten zu einer zusammengeführt und damit drei Quellen falsch
ausgewiesen. Zwei Tests halten das Verhalten fest.

Nebenwirkung für den Fließtext: Der Titel eines `\quelleWeb`-Aufrufs darf **keine
geschweiften Klammern enthalten** (also kein `\enquote{…}`), sonst bricht die Regex des
Prüfers am inneren `}` ab und liest die URL nicht mehr.

Der Fund ist pagination-abhängig: jede Änderung an Fließtext kann Seitenumbrüche
verschieben und damit bestehende Gruppen ungültig machen oder neue entstehen lassen — bei
der Ersteinführung ist genau das einmal passiert (eine Gruppe musste nach einer
Neukompilierung in zwei aufgeteilt werden, weil ihre Primärzitation auf eine andere Seite
gerutscht war als ihre spätere Wiederholung). Nach jeder inhaltlichen Änderung erneut
ausführen, insbesondere nach dem Überschreiben der blauen Zonen vor der Abgabe.

## `check_footnote_groups.py`

Deckt die Lücke, die `check_footnote_pages.py` offen lässt: dieses sieht nur die
`\quelle*`-Aufrufe und meldet daher **neue** Duplikate, nicht aber eine **bestehende**
Gruppe, deren `\quelleErneut` nach einem verschobenen Seitenumbruch auf einer anderen
Seite steht als ihr `\quelleMerken`. Dann verweist das `\footnotemark` auf eine Nummer,
deren Fußnotentext auf der Seite gar nicht abgedruckt ist — ein Fehler, den kein
LaTeX-Lauf meldet.

Ermittelt die Seiten wie `check_footnote_pages.py` per SyncTeX; `find_offset_groups()` ist
die reine, getestete Funktion. Bei der Umstellung des Fußnotenapparats auf `footmisc`
(para) sind so vier von zwölf Gruppen aufgefallen, die über eine Seitengrenze gerutscht
waren. Zusammen mit dem Duplikatstest ausführen:

```bash
./build.sh
python3 scripts/check_footnote_pages.py
python3 scripts/check_footnote_groups.py
```

## `test_data_consistency.py`

Sichert die Doppelung ab, die sich nicht vermeiden laesst: dreizehn Kurswerte stehen sowohl
als Makro in `../data/kennzahlen.tex` (fuer Fliesstext und Tabellen) als auch als Zeile in
`../data/kursverlauf.csv` (fuer pgfplots, das keine LaTeX-Makros lesen kann). Die Regel
„jeder Zahlenwert genau einmal" aus `../data/CLAUDE.md` ist hier technisch nicht einhaltbar.

Geprueft wird `kursverlauf.csv` in beide Richtungen: dass jeder Stuetzpunkt zu seinem Makro
passt, und dass kein Makro fehlt, das die Tabelle zeigt. Dazu die Invariante der Abbildung —
die Stuetzpunkte muessen streng nach Datum geordnet sein, und die x-Koordinate muss dieselbe
Ordnung tragen.

Der Schlusskurs-Test ist inhaltlich der wichtigste: Da die `\KursSilvester…`-Makros fuer 2022
bis 2025 die Werte der Geschaeftsberichte tragen (Quelle und gedruckte Seite im
Zeilenkommentar), ist er zugleich die Gegenprobe gegen die Primaerquelle — die einzige, die
die externe Kursquelle ueberhaupt zitierfaehig macht. Weicht die Yahoo-Reihe ab, faellt er.
Er deckt damit auch die Zeitzonenfalle ab.

Der Ordnungstest schliesst eine Luecke, die LaTeX nicht meldet: In 2023 und 2025 lag das
Jahrestief VOR dem Jahreshoch. Wer die Zeilen nach dem Schema Hoch-Tief sortiert, laesst die
Linie rueckwaerts laufen — pgfplots zeichnet das kommentarlos.

Jede Schleife prueft zuerst die Zeilenzahl. Ohne diese Zusicherung bestuenden die Tests auch
bei leerer CSV-Datei.

## `check_literals.py`

`find_literals(text) -> [(zeile, treffer)]`. CLI endet mit 1, wenn Treffer vorliegen.

Erlaubt: Jahreszahlen `19xx`/`20xx`, Gliederungsnummern, kleine Ganzzahlen, alles in
`%`-Kommentaren, Argumente von `\label`/`\ref`/`\cite`/`\input`/`\url` und von
`\tabellenkoerper` (dem Ersatz für `\input` innerhalb von Tabellen), sowie **jedes
Klammerargument, das mit `http://` oder `https://` beginnt**. Letzteres ist für
`\quelleWeb` nötig gewesen, solange dessen zweites Argument eine URL war. Seit die Adressen
nur noch im Quellenverzeichnis stehen, greift die Ausnahme dort nicht mehr — sie bleibt
aber sinnvoll für jedes künftige Klammerargument mit einer Adresse. Die Ausnahme entfernt
nur das Klammerargument, nicht die Zeile: eine echte Zahl daneben wird weiter gemeldet
(Test vorhanden).
Verboten: Kommazahlen, Ganzzahlen ab drei Stellen, **und deutsch gruppierte Zahlen**
(`12.345`, `246.800`, `1.234,5`) — letztere wurden zunächst übersehen und sind der
wichtigste Grund, den Wachhund nicht aufzuweichen.

Eine Datei mit der Zeile `% check-literals: skip` wird übersprungen. Genutzt nur vom
Deckblatt wegen der Matrikelnummer. **Keine Dateinamen-Whitelist im Skript** — der Marker
ist der Mechanismus.

Bekannter, bewusster Zielkonflikt: die Jahreszahl-Ausnahme deckt auch einen echten
vierstelligen Finanzwert ab, der mit 19 oder 20 beginnt.

## `cashflow.py`

Sicht: **inkrementell**. Ausgewiesen wird der zusätzliche Cash-Flow der Option
(EBITDA-Zuwachs minus Aufbau der Kapitalbindung), nicht der Konzern-Cash-Flow.

Vorgaben der Aufgabenstellung stehen in `ASSUMPTIONS`; die zeitliche Verteilung ist eine
offengelegte Annahme, keine Vorgabe. Bei Option C wird der Kaufpreis im Vollzugsjahr 2026
fällig (45 / 7,5 / 7,5), die Kapitalbindung des erworbenen Geschäfts ebenfalls 2026.

IRR per Bisektion über den **erklärten** Horizont 2026–2035; ohne Horizont ist ein IRR
nicht aussagefähig. `emit_irr_sensitivity()` zeigt 5 / 10 / 15 Jahre.

Ergebnisse: IRR A 29,6 % · B 38,3 % · C 6,3 %. Auf fünf Jahre ist C mit −9,9 % negativ.
Liquiditätsreserve Ende 2028: A 104,2 · B 121,2 · C 78,2 Mio. €, Tiefpunkt C 73,2 in 2026.

`emit_marge_sensitivity()` stellt der Marge des Basisfalls zwei größere Nenner gegenüber
(+3 % Umsatz p. a. und das Umsatzniveau 2024). Ohne diese Tabelle liest sich die hohe
Basismarge als Ertragskraft, obwohl sie zum großen Teil der Umsatzeinbruch von 2025 ist.

`emit_macros()` schreibt dieselben Ergebnisse als `\newcommand` nach
`../data/cf_makros.tex`. Ohne diese Datei müssten die Wertungsabschnitte der Aufgabe 5
die Zahlen der Aufgabe 6 hartkodieren und könnten von ihnen abweichen.

## `fetch_kurs.py` — der Fehlschlag war vorübergehend

Am 29.07.2026 war keine Kursquelle erreichbar (Yahoo HTTP 429 auf `query1` und `query2`,
stooq mit JavaScript-Verifikation, boerse-frankfurt und onvista ohne Ergebnis). **Am
31.07.2026 antwortete `query1` normal.** Daraus stammen die Jahreshöchst- und
Jahrestiefstkurse sowie der Schlusskurs 2021 in `../data/kursverlauf.csv`. Die exakten
Handelstage der Extrema erforderten einen zweiten Abruf mit `interval=1d`; die Monatsbars
geben nur den Monat her.

Die Abbildung hat zwei verworfene Zwischenstufen hinter sich, beide aus demselben Grund:
Eine monatliche Schlusskursreihe mit aufgesetzten Extremmarken liess die **Intraday**-Werte
frei neben der Kurve schweben; fuenf Jahrespunkte mit senkrechtem Spannenbalken behoben das,
zeigten aber nicht, WANN die Extrema eintraten. Die jetzige Form loest beides: dreizehn
Stuetzpunkte an ihrem tatsaechlichen Handelstag, ueber eine Hilfslinie verbunden. Jeder Wert
liegt damit auf der Linie.

Lehre für eine erneute Beschaffung: Ein 429 ist kein Beweis, dass eine Quelle unbrauchbar
ist — vor dem Eintrag einer Lücke in das Quellenverzeichnis mit Abstand erneut versuchen.

**Zwei Fallen beim Auswerten der Yahoo-Antwort**, beide hier schon einmal zugeschlagen:

1. *Zeitzone.* Die Monatsbars beginnen um 00:00 **CET**, der Zeitstempel ist Unix-UTC.
   `datetime.utcfromtimestamp()` schiebt jeden Bar in den Vormonat und verschiebt damit
   sämtliche Hoch-/Tiefzuordnungen um einen Monat. Vor der Umrechnung 12 h addieren.
2. *Kein Test ohne Gegenprobe.* Die Zuordnung ist erst dann belegt, wenn die vier
   Dezember-Schlusskurse mit den Geschäftsberichten übereinstimmen (11,48 / 18,58 / 10,62 /
   12,32). Genau diese Prüfung hat den Zeitzonenfehler aufgedeckt.

## Lokale Modelle über den Ollama-Bridge

Nur für beschreibenden Fließtext (Markdown, `%`-Kommentare) verwenden, **nie** um Zahlen
zu extrahieren, die zitiert werden.

Die frühere Angabe, `qwen3:14b` halte ein striktes Ausgabeformat ab etwa 14k Prompt-Tokens
nicht mehr ein, war falsch diagnostiziert. Ursache war Ollamas stille Kürzung des Prompts
auf `num_ctx` — vom **Anfang** her, wo die Formatvorgabe steht. Gemessen am 02.08.2026 mit
demselben Prompt von 15.403 Tokens: bei `num_ctx=4096` wurden 2.050 Tokens gelesen und
0 von 8 geforderten Zeilen geliefert, bei `num_ctx=16384` alle 15.403 und 8 von 8.
`num_ctx` daher immer explizit setzen, mindestens auf das Doppelte der Promptlänge.

Es gibt eine zweite, echte Grenze, und die ist modellabhängig: `qwen3:14b` liest nie mehr
als 20.482 Tokens, gleich welches `num_ctx` gesetzt ist, und antwortet darüber hinaus mit
Bruchstücken aus dem Fülltext. `gemma4:12b` und `qwen3.5:9b` haben 134k bzw. 135k Tokens
aufgenommen und korrekt geantwortet.

Zur Bildextraktion: **kein Bildmodell für eine zitierte Zahl.** An der Kennzahlentabelle
des GB 2024 (29 geprüfte Werte) lieferte `gemma4:12b` im Bildmodus 1 bis 4 richtige Werte
und erfand je Lauf 74 bis 78 Beträge, die im Bericht nicht vorkommen. `qwen3.5:9b` kam auf
29/29 ohne eine einzige Erfindung, `pdftotext -layout` ebenfalls. Siehe
[../docs/local-agent-workflow.md](../docs/local-agent-workflow.md).

Für eine robuste Nutzung im Workspace steht das Hilfsskript `scripts/ensure_ollama.sh`
bereit. Es startet Ollama automatisch, wartet auf die API und macht den Server für wiederholte
Aufrufe verfügbar:

```bash
./scripts/ensure_ollama.sh
./scripts/ensure_ollama.sh gemma4:12b "Say hello"
```

Bei Bedarf kann auch direkt der normale Aufruf über **stdin** verwendet werden:

```bash
ollama run gemma4:12b < prompt.txt   # stdin, nie als Argument
```
