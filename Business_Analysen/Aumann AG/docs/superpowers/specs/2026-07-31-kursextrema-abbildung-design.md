# Jahreshoch und Jahrestief in der Kursabbildung

**Datum:** 31.07.2026
**Betrifft:** `fig:kursverlauf` in [../../../sections/02-jahresabschluss.tex](../../../sections/02-jahresabschluss.tex)
**Vorgeschichte:** [../../befunde-und-entscheidungen.md](../../befunde-und-entscheidungen.md),
Abschnitt „Kursreihe — am 31.07.2026 doch monatlich beschafft"

## Ziel

Die Abbildung zeigt bisher allein die 48 Monatsschlusskurse 2022–2025. Ergänzt werden die
Jahreshöchst- und Jahrestiefstkurse als Marken, damit die Schwankungsbreite innerhalb der
Jahre unmittelbar sichtbar wird — insbesondere das Jahr 2024, in dem der Kurs vom
Januarhoch bis zum Novembertief um 50,3 % fiel, während Umsatz, EBITDA und
Jahresüberschuss Rekordwerte erreichten.

## Der Konflikt, der die Form bestimmt

Die Extremwerte sind **Intraday**-Werte, die gezeichnete Linie besteht aus
**Schlusskursen**. Eine Marke bei 18,96 € liegt deshalb zwangsläufig oberhalb der Linie.
Ohne Erklärung liest sich das als Zeichenfehler. Die Darstellung muss den Unterschied
benennen; sie darf ihn nicht durch Angleichung verstecken.

## Getroffene Entscheidungen

| Frage | Entscheidung | Begründung |
|---|---|---|
| Form | Punkte mit Markensymbol auf der bestehenden Linie | Kleinster Eingriff, schließt direkt an Tabelle~6 an |
| x-Position | Echter Monat des Extremwerts | Zeigt zusätzlich den **Zeitpunkt**; eine Jahresmitte wäre eine erfundene Position |
| Beschriftung | Keine Zahlen an den Marken | Das Hoch 2023 (18,98 €, Dezember) und das Hoch 2024 (18,96 €, Januar) liegen einen Monat und zwei Cent auseinander — zwei Beschriftungen stünden übereinander. Die Werte stehen unmittelbar darunter in Tabelle~6 |
| Unterscheidung | Über die Markenform, nicht über Farbe | Bleibt im Schwarzweißdruck lesbar |

Verworfen wurden: ein jahresweise getöntes Band von Tief bis Hoch (die Jahresgrenzen wirken
härter, als der Kurs es ist) und ein monatliches Hoch-Tief-Band als echtes Spannen-Diagramm
(methodisch am saubersten, aber die dichteste Darstellung und für die Aussage überdimensioniert).

## 1 · Datenschicht

Neue Datei `data/kursextrema.csv`:

```
jahr,x_hoch,hoch,x_tief,tief
2022,2022.2083,17.68,2022.7917,10.10
2023,2023.9583,18.98,2023.0417,11.28
2024,2024.0417,18.96,2024.8750,9.42
2025,2025.3750,14.66,2025.2083,9.87
```

`x` ist die Monatsmitte als Dezimaljahr — dieselbe Konvention wie in
`data/aktienkurs_monat.csv`, damit beide Reihen auf derselben Jahresachse liegen.
Die Monatszuordnung: Hoch 2022 März, Tief 2022 Oktober; Hoch 2023 Dezember, Tief 2023
Januar; Hoch 2024 Januar, Tief 2024 November; Hoch 2025 Mai, Tief 2025 März.

Der Kopfkommentar der Datei nennt, dem Muster von `aktienkurs_monat.csv` folgend:

- Herkunft: Yahoo Finance, Symbol `AAG.DE`, Handelsplatz XETRA, abgerufen am 31.07.2026
- dass es **Intraday**-Extrema sind, keine Extrema der Schlusskurse
- die Zeitzonenfalle: Die Monatsbars beginnen 00:00 CET, der Zeitstempel ist Unix-UTC;
  ohne Zuschlag von 12 h rutscht jeder Bar in den Vormonat und alle Monatszuordnungen
  verschieben sich um eine Position

## 2 · Konsistenzsicherung gegen die Doppelung

Die acht Werte stehen nach dieser Änderung zweimal im Repo: in `data/kursextrema.csv`
(für pgfplots) und als Makros `\KursHochZFII` … `\KursTiefZFV` in `data/kennzahlen.tex`
(für Fließtext und Tabelle). Das verstößt gegen die Regel in
[../../../data/CLAUDE.md](../../../data/CLAUDE.md), nach der jeder Zahlenwert genau einmal
auftritt.

Die Doppelung ist technisch erzwungen — pgfplots liest CSV, der Fließtext liest Makros —
und besteht bereits: `data/aktienkurs.csv` dupliziert ebenso die vier
`\KursSilvester…`-Makros.

**Statt sie hinzunehmen, wird sie abgesichert.** Neue Testdatei
`scripts/test_data_consistency.py`. Sie stellt eine Hilfsfunktion
`parse_macros(pfad) -> dict[str, str]` bereit, die die `\newcommand{\Name}{Wert}`-Paare aus
`kennzahlen.tex` einliest, und darauf **vier** Tests:

1. `test_parse_macros` — die Hilfsfunktion selbst, gegen einen kurzen Beispieltext mit
   Kommentarzeile und Zeilenkommentar hinter dem Makro.
2. `test_aktienkurs_csv_stimmt_mit_makros` — `data/aktienkurs.csv` gegen
   `\KursSilvesterZFII` … `\KursSilvesterZFV`.
3. `test_kursextrema_csv_stimmt_mit_makros` — `data/kursextrema.csv` gegen
   `\KursHochZFII` … `\KursTiefZFV`, alle acht Werte.
4. `test_monatsreihe_dezember_stimmt_mit_makros` — in `data/aktienkurs_monat.csv` muss die
   Dezember-Zeile jedes Jahres den Wert des zugehörigen `\KursSilvester…`-Makros tragen.
   Das ist zugleich die Validierung gegen die Geschäftsberichte, die die externe
   Kursquelle überhaupt zitierfähig macht.

Vergleich als Zeichenketten nach Normalisierung auf zwei Nachkommastellen. Läuft ohne
Netz, ohne LaTeX-Lauf und ohne SyncTeX; damit gehört die Prüfung in die schnelle Suite.

Erwartete Suitengröße danach: 72 + 4 = 76 Tests.

## 3 · Abbildung

In `sections/02-jahresabschluss.tex`, innerhalb der bestehenden `axis`-Umgebung, hinter dem
vorhandenen `\addplot`:

```latex
\addplot[blue!60!black, only marks, mark=triangle*, mark size=2.2pt]
  table[col sep=comma, x=x_hoch, y=hoch, comment chars={\#}]
  {data/kursextrema.csv};
\addlegendentry{Jahreshoch (Intraday)}
\addplot[blue!60!black, only marks, mark=triangle*, mark size=2.2pt,
         mark options={rotate=180}]
  table[col sep=comma, x=x_tief, y=tief, comment chars={\#}]
  {data/kursextrema.csv};
\addlegendentry{Jahrestief (Intraday)}
```

Die bestehende Kurve erhält `\addlegendentry{Monatsschlusskurs}`.

Ergänzungen an den `axis`-Optionen:

```latex
legend pos=south west,
legend style={font=\small, draw=none, fill=none, legend cell align=left},
```

Unten links ist der Bereich unter 9 € über die gesamte Breite leer — die Legende überdeckt
die Kurve nicht. `ymin=0`, `ymax=22`, `width` und `height` bleiben unverändert; der höchste
Wert liegt bei 18,98.

Neue Bildunterschrift:

> XETRA-Monatsschlusskurse der Aumann-Aktie 2022 bis 2025, mit Jahreshöchst- und
> Jahrestiefstkurs (Intraday) je Jahr.

## 4 · Text

Der Ankündigungssatz in Abschnitt 2.7 wird ergänzt. Bisher:

> Abbildung~\ref{fig:kursverlauf} zeigt die XETRA-Monatsschlusskurse der Jahre 2022 bis 2025.

Künftig zusätzlich: dass die Marken das Jahreshoch und das Jahrestief bezeichnen und die
zugehörigen Werte in Tabelle~\ref{tab:kursextrema} stehen. Schwarzer Text, keine Wertung.

Der bestehende Absatz „Hinweis zur Datengrundlage" erklärt den Intraday-Charakter bereits
und bleibt unverändert.

## 5 · Regeln, die eingehalten bleiben

- **Keine Zahl in `sections/*.tex`:** Alle Koordinaten kommen aus der CSV-Datei, alle
  Werte im Fließtext aus Makros. `check_literals.py` prüft nur `sections/`.
- **Schwarz/Blau:** Die Ergänzung ist reine Sachdarstellung — kein `\bk{}`.
- **Nichts erfinden:** Die x-Positionen sind die tatsächlichen Monate, keine gesetzten
  Werte. Der Intraday-Vorbehalt steht in Bildunterschrift und Fließtext.

## 6 · Prüfprotokoll

```bash
./build.sh
python3 scripts/check_literals.py sections/*.tex
python3 scripts/check_footnote_pages.py
python3 scripts/check_footnote_groups.py
cd scripts && python3 -m unittest discover -p 'test_*.py'
```

Alle fünf müssen sauber durchlaufen; die Suite auf 76 Tests.

**Erwartung zur Paginierung:** Die Abbildung wird nicht höher (die Marken liegen innerhalb
des bestehenden Wertebereichs), der Textzuwachs beträgt einen Halbsatz. Die Seitenumbrüche
sollten stehen bleiben.

**Falls doch nicht:** Fußnotengruppen nach dem in
[../../../sections/CLAUDE.md](../../../sections/CLAUDE.md) beschriebenen Verfahren neu
schneiden — an der ersten Zitation der neuen Seite ein vollwertiges
`\quelleGB`/`\quelleQM`/`\quelleWeb` mit eigenem `\quelleMerken`, die Wiederholungen dieser
Seite auf den neuen Schlüssel zeigen lassen. Das ist bei der vorigen Runde bei sechzehn
Gruppen nötig gewesen und ist erwartbares Handwerk, kein Fehler.

## 7 · Nicht Bestandteil dieser Änderung

- Die drei aufeinanderfolgenden Fußnotenmarken auf Seite 8, die sich als eine Zahl lesen
  („²³²²²⁴"). Vorbestehend, eigener Vorgang.
- Ein monatliches Hoch-Tief-Band. Verworfen, siehe oben.
- Änderungen an `data/aktienkurs.csv`. Die Datei bleibt als Prüfgröße gegen die
  Geschäftsberichte erhalten.

---

## Nachtrag 31.07.2026 — Form verworfen und ersetzt

Die oben spezifizierte Form (Marken auf einer Monatsschlusskurs-Linie) ist umgesetzt, im
Satz geprüft und **verworfen** worden. Der Grund ist genau der Konflikt, den Abschnitt „Der
Konflikt, der die Form bestimmt" benennt — er ließ sich durch Beschriftung nicht entschärfen:

> Die Extremwerte sind Intraday-Werte, die Linie besteht aus Schlusskursen. Eine Marke bei
> 18,96 € liegt deshalb zwangsläufig oberhalb der Linie.

Am gesetzten PDF wirkten die acht Dreiecke frei schwebend, ohne erkennbaren Bezug zur Kurve.
Die Bildunterschrift erklärt das zwar, aber eine Abbildung, die erst durch ihre Unterschrift
plausibel wird, verfehlt ihren Zweck.

**Ersetzt durch:** fünf Jahrespunkte (Schlusskurs zum 31.12.) mit einem senkrechten Balken
von Jahrestief bis Jahreshoch, technisch über pgfplots-Fehlerbalken (`spannenbalken` in
`preamble.tex`). Die Spanne hängt damit am Jahrespunkt, statt neben ihm zu liegen.

**Weitere Abweichungen von der Spezifikation:**

| Punkt | Spezifiziert | Umgesetzt |
|---|---|---|
| Zeitraum | 2022–2025 | **2021**–2025 |
| Datenhaltung | `kursextrema.csv` neben `aktienkurs_monat.csv` | eine Datei `aktienkurs.csv`; beide anderen entfernt |
| Monatsreihe | gezeichnet | nicht mehr gezeichnet, nicht mehr vorgehalten |
| Tests | 4, darunter die Dezember-Gegenprobe der Monatsreihe | 4, Gegenprobe wandert in den Schlusskurs-Test; neu: Fehlerbalken-Spalten |
| Markenstile | `hochmarke` / `tiefmarke` | `spannenbalken` |

**Neuer Vorbehalt, in der Spezifikation nicht vorgesehen:** Für 2021 existiert keine
Primärquelle. Der Geschäftsbericht 2022 enthält keine Angabe zum Jahresschlusskurs (geprüft
am 31.07.2026), der Bericht 2023 führt als Vorjahreswert bereits 2022. Der Wert 13,68 € ist
als einziger der fünf nicht gegengeprüft; das ist in Abschnitt 2.7 und im
Quellenverzeichnis benannt, statt es zu verschweigen.

## Zweiter Nachtrag 31.07.2026 — endgültige Form

Auch die Fassung mit fünf Jahrespunkten und Spannenbalken ist ersetzt worden. Sie behob das
Schweben der Marken, beantwortete aber nicht, **wann** die Extrema eintraten — der Balken
sass am Jahrespunkt, nicht am Tag des Ereignisses.

**Endgültige Form:** dreizehn Stützpunkte auf einer Datumsachse, verbunden durch eine
Hilfslinie. Ausgangspunkt ist der Schlusskurs des letzten Handelstages 2021; danach folgt je
Jahr 2022–2025 Jahreshoch, Jahrestief und Jahresschlusskurs, **geordnet nach dem Tag des
Eintretens**. Datei: `data/kursverlauf.csv`. Die Markenform (● / ▲ / ▼) kommt aus der Spalte
`art` über `scatter/classes` (Stil `kurspunkte` in `preamble.tex`).

**Drei Punkte, die aus der Umsetzung stammen und in keiner Planung standen:**

1. *Die Jahresreihenfolge ist nicht fest.* 2023 und 2025 lag das Tief **vor** dem Hoch. Eine
   feste Abfolge Hoch–Tief liesse die Linie rückwärts laufen; pgfplots zeichnet das
   kommentarlos. Ein Test hält die Datumsordnung jetzt fest.
2. *Der letzte Handelstag ist nie der 31.12.* Es ist der 29. oder 30. Dezember — die Börse
   ruht. Die Geschäftsberichte sprechen deshalb ebenfalls vom „letzten Handelstag".
3. *Drei Stützpunkte sind nicht auflösbar.* Hoch 2023 (28.12.), Schluss 2023 (29.12.) und
   Hoch 2024 (02.01.) liegen fünf Handelstage auseinander und fallen auf einer Vierjahresachse
   zu einer Spitze zusammen. Das ist keine Zeichenschwäche, sondern eine Eigenschaft der
   Daten. Konsequenz: Tabelle 6 trägt den Handelstag jedes Extremwerts und ist damit die
   genaue Lesart der Abbildung. Im Fließtext ist die Einschränkung benannt.

**Die exakten Handelstage** erforderten einen zweiten Yahoo-Abruf mit `interval=1d`; die
Monatsbars geben nur den Monat her. Beide Abrufe stimmen in den Extremwerten überein.
