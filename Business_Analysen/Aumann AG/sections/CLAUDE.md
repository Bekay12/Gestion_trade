# sections/ — ein Abschnitt je Aufgabenteil

Reihenfolge und Nummerierung folgen der Aufgabenstellung exakt. Eingebunden werden die
Dateien in [../hausarbeit.tex](../hausarbeit.tex).

| Datei | Aufgabenteil | Zustand |
|---|---|---|
| `00-deckblatt.tex` | Deckblatt | fertig, trägt `% check-literals: skip` (Matrikelnummer) |
| `01-unternehmen.tex` | 1.1–1.5 | fertig |
| `02-jahresabschluss.tex` | 2.1–2.7 | fertig |
| `03-strategie.tex` | 3.1–3.6 | fertig |
| `04-aktienkurs.tex` | 4 | fertig, eine blaue Zone |
| `05-handlungsoptionen.tex` | 5.1–5.4 | fertig, 44 blaue Zellen |
| `06-cashflow.tex` | 6.1–6.3 | fertig, vier blaue Zonen |
| `90-quellen.tex` | Quellenverzeichnis | fertig, unnummeriert |
| `91-ki-nutzung.tex` | KI-Hinweis | fertig — Formulierung vom Verfasser zu bestätigen |
| `92-erklaerung.tex` | Eidesstattliche Erklärung | fertig, unnummeriert |

Die Anhänge `90`–`92` und der Kennzahlenüberblick in `02` sind **unnummeriert**
(`\subsection*` bzw. `\section*` mit `\addcontentsline`). Grund: Die
Nummerierung des Dokuments muss der Aufgabenstellung entsprechen — mit einer
nummerierten Überblickstabelle trüge die Kursentwicklung, die Frage 2.7
beantwortet, die Nummer 2.8.

## Schwarz und Blau

**Schwarz** ist alles Belegbare: Zahlen, Tabellen, Struktur, referierte Aussagen der
Berichte. In den schwarzen Teilen sind wertende Adjektive verboten — kein „erfolgreich",
„vielversprechend", „enttäuschend", keine prospektiven Konjunktive. Teil 3 referiert und
verweist für die Bewertung auf Teil 5.

**Blau** ist jede Wertung, Einschätzung, Empfehlung oder Interpretation:

```latex
% FAKTENBASIS 5.1.1
%  - Auftragseingang 2024: \AuftragseingangDeltaZFIV %
%  - Nettofinanzliquiditaet 31.12.2024: \NettoliquiditaetZFIV Mio. EUR
\bk{Deutscher Fliesstext, vollstaendige Saetze, keine Aufzaehlung im PDF.}
```

Die Faktenbasis steht **als LaTeX-Kommentar** darüber, nie sichtbar im PDF. Jede wertende
Aussage stützt sich auf eine Zahl, die im Dokument bereits belegt ist — per Makro oder per
`Tabelle~\ref{...}`.

Längenvorgaben der Aufgabenstellung: ½ DIN-A4 ≈ 250 Wörter, ¼ ≈ 125, eine Seite ≈ 500.

## Kein Gedankenstrich im Fließtext

`--` (und `---`, `–`, `—`) als Einschub oder Betonung ist im gesamten Dokument
**verboten**; die Konstruktion gilt als deutlichstes Merkmal maschinell erzeugter Prosa.
Stattdessen Komma, Semikolon, Doppelpunkt, Klammern oder zwei Sätze. Auch in
Tabellenüberschriften und Zeilenköpfen: `Option~A: Diversifikation`, nicht
`Option~A -- Diversifikation`. Die Begründung und die Umschreibungstabelle stehen in
`~/.claude/rules/code-style.md`, Abschnitt „Prose punctuation“.

Erlaubt bleibt `--` allein als **Bis-Strich in Zahlen- und Jahresbereichen**
(`2021--2025`, `\num{a}--\num{b}`), als Bindestrich in Komposita
(`Batterie- und Brennstoffzellen`) und in wörtlich übernommenen Quellentiteln
(`Investor Relations -- Shares`).

Prüfen mit:

```bash
grep -rn -- '--' sections/*.tex | grep -v '}--\|[0-9]--[0-9]'
```

Jede Umschreibung verändert die Zeilenzahl: danach `./build.sh`, Seitenzahl und beide
Fußnotenskripte erneut prüfen.

## Gleitobjekte

Eine Tabelle, die den **gesamten** Rumpf einer Überschrift bildet, wandert als
Gleitobjekt auf die nächste Seite und lässt die Überschrift leer zurück — genau das war
bei 5.1.2, 5.2.2 und 6.3 der Fall. Zwei Regeln verhindern das:

1. **Jede Tabelle wird im Text angekündigt.** Unter der Überschrift steht mindestens ein
   Satz mit `Tabelle~\ref{...}`, bevor die `table`-Umgebung beginnt. Das ist ohnehin
   wissenschaftlicher Standard.
2. **`\FloatBarrier` nur hinter einer Gruppe mehrerer Tabellen** (Paket `placeins`), nicht
   hinter jeder einzelnen. Hinter einer **isolierten** Einzeltabelle erzwingt die Barriere,
   dass die Tabelle vor jedem nachfolgenden Text gesetzt wird — passt sie nicht mehr auf
   die laufende Seite, bleibt der Rest der Seite leer, statt dass der nachfolgende
   Fließtext nach oben rutscht (so geschehen bei den drei Einschätzungstabellen 5.1.2,
   5.2.2, 5.3.2 und der Übersichtstabelle 5.4.1 — dort wurde die Barriere wieder entfernt).
   Bei Tabellengruppen (z. B. die drei Cash-Flow-Tabellen in 6.1) bleibt sie nötig, um ein
   Auseinanderdriften über mehrere Abschnitte zu verhindern. Die Tabellen tragen zusätzlich
   `[!htbp]` statt `[htbp]`.

## Seitenumbrueche

Zwei Einstellungen in [../preamble.tex](../preamble.tex) und ein Makro halten die Umbrueche
sauber:

1. **`\clubpenalty` / `\widowpenalty` / `\displaywidowpenalty` auf 10000.** Die
   LaTeX-Vorgabe ist je 150 und erlaubt damit eine einzelne Absatzzeile am Seitenfuss oder
   -kopf. Im deutschen Satz gelten beide als Fehler.
2. **`\needspace{n\baselineskip}` vor einer gefaehrdeten Ueberschrift.** LaTeX verhindert
   von sich aus nur den Umbruch unmittelbar *nach* einer Ueberschrift; eine Ueberschrift
   plus eine einzelne Zeile am Seitenfuss bleibt erlaubt. Gesetzt ist es an zwei Stellen:
   vor dem Kennzahlenueberblick in `02` (dort gehoert die Tabelle mit dazu, deshalb
   `16\baselineskip` und nicht bloss Platz fuer die Ueberschrift) und vor 6.2 in `06`, wo
   unmittelbar die Unterunterueberschrift 6.2.1 folgt.

**Der Wert muss fuer alles bemessen sein, was zusammenbleiben soll.** Mit
`6\baselineskip` passte genau die Ueberschrift auf die Seite und die Tabelle blieb trotzdem
zurueck — die Reservierung war erfuellt und hat nichts bewirkt.

Pruefen laesst sich das nicht aus dem Quelltext, sondern nur am gesetzten PDF: die
Positionen der Ueberschriften aus `out/hausarbeit.toc` gegen ihre y-Koordinaten aus
`pdftotext -bbox` halten. Aussagekraeftig ist dabei nicht die Position allein, sondern wie
viele Textzeilen der Ueberschrift auf derselben Seite noch folgen — weniger als zwei ist
ein Fehlumbruch.

## Kopfzeile

Die Kopfzeile zeigt links Nummer **und** Titel des laufenden Abschnitts
(`\thesection\quad <Titel>`, gesetzt über `\sectionmark`) und rechts den festen
Dokumenttitel „Analyse Aumann AG“, getrennt durch eine Linie (`headsepline`).
Konfiguriert über `scrlayer-scrpage` in [../preamble.tex](../preamble.tex); die
Schriftgröße ist `\small`, damit der längste Abschnittstitel („Strategische Maßnahmen im
Geschäftsbericht 2024“) neben dem Dokumenttitel in eine Zeile passt. `\section*`
(Quellenverzeichnis, KI-Hinweis, Erklärung) löst `\sectionmark` nicht aus; die Kopfzeile
wird deshalb in [../hausarbeit.tex](../hausarbeit.tex) vor `\input{sections/90-quellen}`
manuell mit `\markright{}` geleert — dort bleibt nur der Dokumenttitel stehen.

## Zitieren

Deutsche Fußnotenzitierweise über drei Makros aus [../preamble.tex](../preamble.tex):

```latex
\quelleGB{2024}{13}                       % Geschaeftsbericht, Jahr, gedruckte Seite
\quelleQM{Quartalsmitteilung Q1 2025}{2}  % Zwischenbericht, Bezeichnung, gedruckte Seite
\quelleWeb{Urheber und Titel}{quelle:kurzname}{31.07.2026}
```

**Jeder Beleg ist ein Sprung ins Quellenverzeichnis.** Der Kurztitel in der Fußnote ist per
`\hyperref` mit seinem Eintrag verknüpft — ein Klick führt dorthin. Die Farbe
(`quellenlink` in [../preamble.tex](../preamble.tex)) macht die Verknüpfung sichtbar; ohne
sie wäre sie zwar anklickbar, aber nicht erkennbar, weil `hyperref` mit `hidelinks` läuft.
Für einen streng einfarbigen Ausdruck genügt es, diese Farbe auf `black` zu setzen — die
Verknüpfung bleibt bestehen.

Die Schlüssel folgen der Konvention `quelle:<kurzname>` und stehen als
`\item\label{quelle:...}` in [90-quellen.tex](90-quellen.tex):

| Makro | Schlüssel | woher |
|---|---|---|
| `\quelleGB{2024}{13}` | `quelle:gb2024` | aus dem Jahr abgeleitet |
| `\quelleQM{Quartalsmitteilung Q1 2025}{2}` | `quelle:Quartalsmitteilung Q1 2025` | aus der Bezeichnung abgeleitet (Label mit Leerzeichen, funktioniert) |
| `\quelleWeb{Titel}{quelle:shares}{31.07.2026}` | ausdrücklich angegeben | zweites Argument |

Bei `\quelleGB` und `\quelleQM` leitet sich der Schlüssel aus dem ersten Argument ab —
deshalb mussten die rund vierzig Aufrufe im Fließtext nicht angefasst werden. Bei
`\quelleWeb` ist das zweite Argument **kein Link mehr, sondern der Schlüssel**; damit steht
jede URL genau einmal im Dokument, dieselbe Regel wie für Zahlenwerte.

Eine neue Quelle braucht daher **zwei** Schritte: erst den Eintrag mit
`\item\label{quelle:...}` im Quellenverzeichnis, dann den Aufruf. Fehlt das Label, meldet
der erste LaTeX-Lauf „Hyper reference undefined" und im Satz steht ein toter Verweis.

Die Makros sind **nicht robust**. Niemals in ein `\caption{}` oder ein anderes bewegliches
Argument setzen — die Fußnote verschwindet oder verdoppelt sich. Zitate gehören in den
Fließtext.

Seitenzahlen sind die **gedruckten**, nicht die PDF-Seiten. Bei den Quartalsberichten
weichen beide voneinander ab. Prüfmethode: [../refs/CLAUDE.md](../refs/CLAUDE.md).

### Fussnotenapparat: laufender Satz

Die Fußnoten stehen nicht untereinander, sondern als **ein durchlaufender Absatz** am
Seitenfuß (`\usepackage[para]{footmisc}`). Auf belegdichten Seiten belegte der Apparat
untereinander bis zu einem Drittel des Satzspiegels; im laufenden Satz sind es ein bis
zwei Zeilen. Ein zweispaltiger Apparat (`dblfnote` aus `yafoot`) wäre die Alternative,
ist hier aber nicht einsetzbar: zusammen mit `pgfplots` und `hyperref` läuft er in eine
Endlosschleife der Ausgaberoutine (Details in
[../docs/befunde-und-entscheidungen.md](../docs/befunde-und-entscheidungen.md)).

Zwei Nebenwirkungen, die beim Überschreiben der blauen Zonen wichtig sind: der Apparat ist
niedriger, also passt **mehr Text auf jede Seite** — die Seitenumbrüche liegen anders als
vor der Umstellung, und die Fußnotengruppen unten müssen erneut geprüft werden. Und
`microtype` kann seinen `footnote`-Patch nicht anlegen, weshalb er in
[../preamble.tex](../preamble.tex) abgeschaltet ist; sonst warnt jeder Lauf.

### Fussnotennummer wiederverwenden

Zitiert ein Absatz dieselbe Quelle+Seite mehrfach auf derselben **gedruckten Seite der
Hausarbeit**, druckt jeder `\quelleGB`/`\quelleQM`/`\quelleWeb`-Aufruf sonst eine neue,
wortgleiche Fußnote — auf einer Seite standen zuvor bis zu sieben identische Fußnoten
„Aumann AG, Geschäftsbericht 2024, S. 10.“ untereinander. Zwei Zusatzmakros aus
[../preamble.tex](../preamble.tex) beheben das:

```latex
% Erste Zitation der Gruppe: normaler Aufruf, danach die Nummer merken.
Text.\quelleGB{2024}{10}\quelleMerken{p9-gb2024-10}

% Jede weitere Zitation derselben Quelle+Seite auf derselben Seite:
Text.\quelleErneut{p9-gb2024-10}
```

`\quelleErneut` setzt nur `\footnotemark` mit der gemerkten Nummer, ohne den Text erneut
zu drucken. Der Schlüssel ist frei wählbar, muss aber pro (PDF-Seite, Quelle+Seite)
eindeutig sein — Konvention: `p<PDF-Seite>-gb<jahr>-<seite>`.

**Nach jeder Änderung an Fließtext erneut prüfen:** ein verschobener Seitenumbruch kann
eine bestehende Gruppe ungültig machen (die Primärzitation landet auf einer anderen Seite
als ihre Wiederholung) oder eine neue Gruppe entstehen lassen, die noch niemand
zusammengeführt hat — genau das ist bei der Ersteinführung dieses Mechanismus passiert
(`git log`, Commit zur Fußnoten-Deduplizierung). Prüfmethode:

```bash
./build.sh                              # -synctex=1 ist bereits gesetzt
python3 ../scripts/check_footnote_pages.py    # neue, noch nicht zusammengefuehrte Duplikate
python3 ../scripts/check_footnote_groups.py   # bestehende Gruppen ueber einer Seitengrenze
```

Beide Skripte sind nötig: das erste sieht nur die `\quelle*`-Aufrufe und meldet daher
**neue** Duplikate; das zweite prüft, ob jede bestehende `\quelleErneut` noch auf der Seite
ihres `\quelleMerken` steht. Bei der Umstellung des Fußnotenapparats waren vier von zwölf
Gruppen über eine Seitengrenze gerutscht — repariert wird das, indem an der ersten Zitation
der neuen Seite wieder ein vollwertiges `\quelleGB`/`\quelleQM`/`\quelleWeb` mit eigenem
`\quelleMerken` steht und die Wiederholungen dieser Seite auf den neuen Schlüssel zeigen.

Das Skript fragt für jeden `\quelle*`-Aufruf per SyncTeX die tatsächlich gerenderte
PDF-Seite ab (keine Schätzung aus Zeilennummern) und meldet verbleibende Duplikate mit
Fundstelle. Muss vor der Abgabe — insbesondere nach dem Überschreiben der blauen Zonen —
sauber durchlaufen.

## Keine Zahlenliterale

`python3 ../scripts/check_literals.py *.tex` muss mit 0 enden. Erlaubt sind Jahreszahlen,
Gliederungsnummern und kleine Ganzzahlen. Alles andere gehört nach
[../data/kennzahlen.tex](../data/kennzahlen.tex) — auch scheinbar harmlose Werte wie eine
Beteiligungsquote von 100 %.

Zwei echte Fehlalarme sind so gelöst worden, ohne den Wachhund aufzuweichen: die ISIN
(`DE000A2DAM03`) liegt im Makro `\ISIN`, und pgfplots-Achsen bekommen explizite
`xticklabels` statt der Option `1000 sep`.
