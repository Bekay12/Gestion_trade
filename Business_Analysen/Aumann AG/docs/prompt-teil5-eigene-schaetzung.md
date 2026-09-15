# Auftrag: Teil 5 braucht eigene Schätzungen, nicht die Vorgaben aus Teil 6

Offener Befund vom 31.07.2026. In einer neuen Sitzung abzuarbeiten.
Den Abschnitt „Prompt" unten unverändert einfügen.

---

## Der Befund

Teil 5 liest derzeit die Makros aus `data/cf_makros.tex`, also die **von der
Aufgabenstellung für Teil 6 vorgegebenen** Beträge (A 35 / B 30 / C 60 Mio. €
Investition, 15 / 15 / 20 Mio. € Kapitalbindung). Das beantwortet die Frage nicht.

Zwei Stellen der Aufgabenstellung belegen das:

1. **5.x.2:** „Als Ergebnis Ihrer Analyse erstellen Sie bitte eine Tabelle, in der Sie
   **Ihre** (qualitativen und quantitativen) **Einschätzungen** darstellen."
2. **Teil 6:** „Gehen Sie dabei **(ggf. abweichend von Ihren vorausgegangenen Analysen)**
   von folgenden Annahmen aus:"

Die Klammer in Teil 6 ergibt nur dann einen Sinn, wenn Teil 5 eigene, aus der
Unternehmensanalyse hergeleitete Zahlen enthält, die von den Vorgaben abweichen dürfen.
Der jetzige Aufbau entwertet sie und verschenkt den Nachweis eigener
finanzwirtschaftlicher Herleitung — bei 40 von 100 Punkten.

## Umfang

Kleiner, als die 95 Makroaufrufe in `sections/05-handlungsoptionen.tex` vermuten lassen.

**Zu ersetzen — eigene Schätzung nötig:**

- die Zeilen **Investitionsbedarf**, **Kapitalbindung** und **Liquiditätsbedarf
  (jährlich)** der Sechs-Faktoren-Tabellen: 3 Optionen × 3 Zeilen = **9 Zellen**
- die Aussagen zum Kapitalbedarf in den Management-Summaries (5.x.1) und in den
  Statements (5.x.3), soweit sie eine Investitionshöhe behaupten

**Zu behalten — sind Rechenergebnisse aus Teil 6, kein Ersatz für eine Schätzung:**

- `\Irr*`, `\IrrFuenf*`, `\LiquiditaetEnde*`, `\LiquiditaetTief*`,
  `\MargeZweitausendachtundzwanzig*`
- Bedingung: im Satz muss stehen, dass sie **unter den vorgegebenen Annahmen** des
  Teils 6 gelten. Der Verweis `Abschnitt~\ref{sec:cashflow}` steht vor jeder Tabelle und
  bleibt.

**Nebenbei zu korrigieren:** Zeile 369 schreibt „Unter den in Abschnitt~\ref{sec:cashflow}
**vorgegebenen** Annahmen". Die Annahmen dort sind gemischt — die Beträge sind vorgegeben,
die zeitliche Verteilung ist selbst gesetzt. `offengelegten` oder `getroffenen` trifft es.

## Verfügbare Anker für eine eigene Herleitung

Alles bereits belegt in `data/kennzahlen.tex`:

| Größe | Makro | Wert |
|---|---|---|
| Konzernumsatz 2024 / 2025 | `\UmsatzZFIV` / `\UmsatzZFV` | 312,346 / 203,985 |
| Segment E-Mobility 2024 | `\UmsatzEMobilityZFIV` | 258,530 |
| Segment Next Automation 2023–2025 | `\UmsatzNextAutoZFIII/IV/V` | 60,5 / 53,8 / 40,2 |
| Nettoliquidität 2024 / 2025 | `\NettoliquiditaetZFIV/ZFV` | 138,2 / 148,1 |
| Eigenkapital 2024 | `\EigenkapitalZFIV` | 201,7 |
| operatives EBITDA 2024 / 2025 | `\EbitdaOpZFIV` / `\EbitdaOpZFV` | 36,417 / 27,278 |

Zusätzlich in den Berichten, noch nicht als Makro: die **aktivierten Entwicklungskosten**
(GB 2023 S. 14: 2,7 Mio. € = 1,5 % des Umsatzes; GB 2025 S. 13: 1,9 Mio. € = 0,8 %). Sie
sind der belastbarste Anker für den F&E-Anteil der Optionen A und B. Vor Verwendung gegen
die gedruckte Seite prüfen (Methode in `refs/CLAUDE.md`).

## Gewünschtes Ergebnis

1. Neun eigene Schätzwerte, **hergeleitet aus den Ankern oben**, nicht geraten. Die
   Herleitung steht sichtbar im Dokument (ein Satz je Zelle reicht: „bezogen auf den
   Segmentumsatz von … entspricht das …").
2. Eigene Makros in einer **eigenen Datei**, z. B. `data/schaetzung_teil5.tex`, damit sie
   nie mit den generierten Werten aus `data/cf_makros.tex` verwechselt werden. Jede Zeile
   mit ihrer Herleitung im Kommentar.
3. In Teil 6, im Absatz „Annahmen": **ein Satz zur Abweichung** zwischen eigener Schätzung
   und Vorgabe. Genau darauf zielt die Klammer der Aufgabenstellung — sie sichtbar zu
   bedienen bringt Punkte.
4. Die betroffenen Zellen sind **blaue Zonen** (`\bk{}`). Der Entwurf gehört dem Verfasser
   zur Überschreibung; er trifft die Größenordnung.

## Ebenfalls offen

Im Skill `~/.claude/skills/startup-investment-analyzer/references/report-template.md`
steckt derselbe Denkfehler: bei 5.x.2 muss stehen, dass die Schätzung **eigenständig** ist
und ein etwaiges vorgegebenes Szenario in Abschnitt 6 davon getrennt bleibt.

---

## Prompt

```
Im Projekt /home/berkam/Projets/Latex/Finanzwirtschaft_Für_Ingenieur ist ein Befund
offen. Lies zuerst docs/prompt-teil5-eigene-schaetzung.md vollständig — dort steht der
Auftrag mit Belegstellen, Umfang und den verfügbaren Ankern. Danach CLAUDE.md und
docs/befunde-und-entscheidungen.md für den Projektstand.

Kurz: Teil 5 (sections/05-handlungsoptionen.tex) übernimmt für Investitionsbedarf,
Kapitalbindung und Liquiditätsbedarf die Makros aus data/cf_makros.tex, also die von der
Aufgabenstellung für Teil 6 vorgegebenen Beträge. Gefragt ist dort aber meine eigene
Einschätzung — die Aufgabenstellung sagt zu Teil 6 ausdrücklich „ggf. abweichend von
Ihren vorausgegangenen Analysen". Neun Tabellenzellen plus die Kapitalbedarfs-Aussagen in
5.x.1 und 5.x.3 sind betroffen.

Leite die neun Schätzungen aus den Unternehmenszahlen her (Anker stehen im Auftragsdokument),
lege sie als eigene Makros in data/schaetzung_teil5.tex ab, mach die Herleitung im Text
sichtbar und ergänze in Teil 6 den Satz zur Abweichung. IRR- und Liquiditätsverweise
bleiben, sofern sie als Ergebnis der vorgegebenen Annahmen gekennzeichnet sind.

Regeln wie gehabt: kein Zahlenliteral in sections/*.tex (python3 scripts/check_literals.py
sections/*.tex muss 0 liefern), Seitenangaben gegen die gedruckte Seite prüfen, nichts
erfinden, ./build.sh muss durchlaufen, danach check_footnote_pages.py und
check_footnote_groups.py.

Die betroffenen Zellen sind blaue Zonen — liefere einen Entwurf, den ich überschreibe,
und sag mir am Ende, welche Größenordnungen du gesetzt hast und woraus.

Antworte mir auf Deutsch.
```
