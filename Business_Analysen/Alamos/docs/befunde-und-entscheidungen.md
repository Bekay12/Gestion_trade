# Befunde und getroffene Entscheidungen — Alamos Gold

Was eine neue Sitzung sonst neu herleiten müsste. Stand 16.09.2026.

## Der Fall in fünf Zahlen

| | |
|---|---|
| Kurs (Nasdaq, 14.09.2026) | 35,66 USD |
| Börsenwert | 14,98 Mrd. USD |
| Hürde (bereinigte Eigenkapitalrendite) | 14,62 % |
| Einstiegsschwelle, bestes Szenario, 15 Jahre | 23,16 USD |
| dieselbe mit dem günstigsten belegten Endwert | 25,55 USD |

Votum: **nicht kaufen, beobachten.** Kein Verkaufsvotum.

## Die sechs tragenden Befunde

1. **Der Zehnjahres-Median des freien Cash-Flows beträgt 4,45 Mio. USD.** In sieben
   von zehn Jahren erwirtschaftete Alamos praktisch nichts; 2025 ist mit 288,2 der
   höchste Wert der Reihe. Wer heute kauft, kauft auf dem besten Jahr der
   Unternehmensgeschichte. Das ist der wichtigste Befund und er schneidet gegen
   den Fall.
2. **Der Ertragssprung stammt vom Goldpreis, nicht aus dem Betrieb.** Produktion
   −3,8 %, AISC +21,7 %, erzielter Preis +41,7 %. Die Marge je Unze stieg um
   64,0 % — und im ersten Halbjahr 2026 um weitere 96,1 %.
3. **Der Einbruch 2026 war überwiegend kein Alamos-Ereignis.** Vom Hoch am
   02.03.2026 zum Tief am 31.07.2026: Aktie −49,7 %, Sektor (GDX) −35,8 %, Gold
   (GLD) −24,2 %. Der unternehmensspezifische Teil beträgt 13,9 Prozentpunkte —
   und ist mit 17,2 Punkten Rückstand auf den Sektor bis heute nicht aufgeholt.
4. **Alamos hat Nettokasse, nicht Nettoschulden.** 623,1 Kasse gegen 200,0
   Finanzschulden, 550,0 Kreditlinie unbeansprucht. Falle: Die Bilanzzeile *Total
   Liabilities* (1.938,8) sind die Gesamtverbindlichkeiten und **nicht** die
   Finanzschulden; die frühere Datei `../daten_agi.py` führte sie als
   `schulden_2025` und hätte die Verschuldungslage ins Gegenteil verkehrt.
5. **Das Bauprogramm ist finanzierbar — aber nicht bei jedem Preis.** Im mittleren
   Szenario fällt der Kassenbestand nie unter 716 Mio.; auf dem Preisniveau von
   2024 wird er mit −318,6 negativ und die Kreditlinie zur Finanzierungsquelle.
   Das ist die eigentliche Verwundbarkeit, und sie steht im Investitionsplan,
   nicht in der Bilanz.
6. **Das Urteil hält gegen die Annahme, die es entscheidet.** Ersetzt man den
   Buchwert (10,59 je Aktie) durch den vom Unternehmen selbst veröffentlichten
   Kapitalwert des Island-Gold-Distrikts (29,05 je Aktie, bei 4.500 $/oz), steigt
   die Schwelle im besten Szenario nur von 23,16 auf 25,55 — der Kurs liegt immer
   noch 39,6 % darüber.

## Getroffene Entscheidungen

**Kein Entschuldungspfad.** Das Muster (Brown & Brown) rechnet als erstes, wie
schnell der Konzern seine Übernahme abtragen kann. Bei Nettokasse ist die Frage
leer. An ihre Stelle tritt die Finanzierungskapazität des Bauprogramms.

**Szenarien über den Goldpreis, nicht über Mengenwachstum.** Drei vom Unternehmen
selbst berichtete *erzielte* Preise (2.379 / 3.372 / 4.660), konstant
fortgeschrieben. Damit muss keine Preisannahme erfunden werden.

**Menge und Kosten folgen dem veröffentlichten Dreijahresausblick.** 610 Tsd. Unzen
bei AISC 1.825 bis 2027, danach 795 Tsd. bei AISC 1.250 (−18 % gegenüber 2025).
Ohne diesen Schritt belastet die Rechnung fünfzehn Jahre mit Wachstumskapital und
erhält nie die Produktion, die es kauft — ein Fehler, keine Vorsicht. Vor der
Korrektur lag die Schwelle bei 3,44 statt 12,97.

**Hürde auf das bereinigte Ergebnis.** Berichtet wären 22,06 %; die Zahl trägt eine
Wertaufholung von 218,8 und einen Veräußerungsgewinn von 231,0. Bereinigt: 14,62 %.

**2027 wird nicht interpoliert.** Das Unternehmen veröffentlicht für 2027 keine
eigene Zahl. Das Jahr bleibt auf dem Stand von 2026; die Zurückhaltung fällt
zulasten der Rechnung.

**Kostenprognose in der angehobenen Fassung.** Der Zwischenbericht zum 30.06.2026
hob die AISC-Prognose von 1.500–1.600 auf 1.775–1.875. Verwendet wird die
angehobene; die ursprüngliche wäre die günstigere und damit die unehrlichere Wahl.

## Eichung des Flow-Modells

| Posten | Mio. USD |
|---|---|
| AISC-Marge 2025 (531.230 Unzen × (3.372 − 1.524)) | 981,7 |
| − Wachstumskapital (507,1 − 144,6 Erhaltung) | 362,5 |
| − laufende Ertragsteuern | 120,5 |
| **Modellwert** | **498,7** |
| Tatsächlicher freier Cash-Flow (795,3 − 507,1) | 288,2 |
| Differenz | 210,5 |
| davon Hedge-Ausbuchung 113,5, Vorauszahlung −50,0, Umlaufvermögen 129,0, Bauzinsen 17,1 | 209,6 |
| **Ungeklärter Rest** | **0,9** |

Das Modell bildet das Ist-Jahr ab, sobald man die Einmalposten benennt.

## Zwei Werkzeugfehler, die hier gefunden wurden

**Kollidierende Fußnotenschlüssel (betrifft auch Brown & Brown).**
`fussnoten_gruppen.py` bildete den Merkschlüssel mit
`re.sub(r"[^A-Za-z]", "", quelle.lower())` und entfernte damit genau die Ziffern,
die Jahr und Seite unterscheiden: Aus `MDA|2025|7` und `MDA|2025|13` wurde beide
Male `mda`. Jedes `\quelleErneut` verwies danach auf die zuletzt gesetzte Fußnote.
Hier korrigiert (Ziffern werden auf Buchstaben abgebildet) samt Regressionstest.
**In `../Brown & Brown/` ist der Fehler noch aktiv:** 7 Schlüssel mehrfach belegt,
12 betroffene `\quelleErneut`-Aufrufe; in `01-unternehmen.tex` verweisen drei
Fußnoten zur Geschäftsführung auf S. 28 des 10-K, die Seite mit der
Börsennotierung.

**Eine Reservenzahl als Seitenzahl gelesen.** In der Annual Information Form 2017
stand `452` in den letzten drei Zeilen eines Blocks und erfüllte jedes Muster einer
blanken Seitenzahl; gelesen als Seite wurde aus „Seiten 1–68" ein „Seiten 1–452".
Abwehr ist der Folgefilter in `edgar_seiten.py`: Es gilt nur, was die Seitenfolge
des Dokuments fortsetzt.

## Anerkannter Kettenbruch

`reihe.py` meldet einen Bruch: AISC 2024 steht im Bericht 2024 bei 1.281, im
Bericht 2025 als Vorjahr bei 1.252. Das ist eine Neuberechnung — das Unternehmen
nimmt die Marktbewertungseffekte der aktienbasierten Vergütung heraus (MD&A 2025,
S. 41). Eingetragen in `BEKANNT`, im Dokument benannt in Abschnitt 2.2.

## Offen

- **Der Investorentag vom 04.02.2026** enthält die ausführliche Fassung des
  Dreijahresausblicks. Die Präsentation ist keine eingereichte Unterlage und trägt
  keine gedruckte Seite; ausgewertet wurde nur, was im Lagebericht steht.
- **Die Zuordnung der Kursziele zu einzelnen Häusern** ist nicht beschafft; nur das
  Aggregat (13 Häuser, Mittel 46,25, Spanne 38–60) liegt vor.
- **Die Reserven sind nicht nachgerechnet.** Gehalte und Kapitalwerte sind Angaben
  des Unternehmens und seiner Sachverständigen.
