# Befunde und getroffene Entscheidungen

Was eine neue Sitzung sonst neu herleiten müsste. Stand: 18.09.2026.

## Inhaltliche Befunde

### AstraZeneca

- **Das Basisjahr ist der Spitzenwert.** Der freie Mittelzufluss je Aktie 2025
  (5,44 USD) ist der höchste der sieben erfassten Jahre — das 100. Perzentil.
  Jede Renditerechnung schreibt damit ein Spitzenjahr fort. Das ist die
  Umkehrung des Alamos-Falls (dort war das Basisjahr ein Tief) und muss im
  Votum stehen.
- **Die Notierung hat sich am 02.02.2026 geändert**: Bis dahin ADS über eine
  halbe Stammaktie an der Nasdaq, seither Direktnotierung der Stammaktie an
  der NYSE. Die Nasdaq-Kursreihe ist darauf zurückgerechnet (Schluss 2016
  53,87 USD statt rund 27). Ohne diesen Hinweis liest sich die Reihe falsch.
- **Prognosetreue vier Jahre, nicht fünf.** Die zuerst gegebene Prognose für
  FY2021 stünde im Geschäftsbericht 2020, dessen Fußzeile in einer Schrift
  ohne Unicode-Zuordnung gesetzt ist. Nicht belegbar, also nicht verwendet.
- **Der Verdikt-Preis hängt an der Endwertannahme.** Buchwert als Endwert:
  Schwelle 45,14 USD. Kurs von heute als Endwert: 96,34 USD. Konstantes
  Vielfaches bei der eigenen Umsatzambition (6,37 % p.a.): 152,85 USD. Nur die
  dritte liegt in der Größenordnung des Marktes; sie trägt das Votum.
- Das geforderte Wachstum bei konstantem Vielfachen ist **horizontunabhängig**
  (6,7 % über 5, 10 und 15 Jahre). Das ist eine Eigenschaft der Formel, kein
  Rundungseffekt, und steht so in der Tabellenbeschriftung.

### Dr. Reddy's

- **Die entscheidende Zahl ist nicht veröffentlicht.** Der Umsatz- und
  Ergebnisbeitrag des Lenalidomid-Geschäfts (generisches Revlimid) steht in
  keiner der fünf ausgewerteten Einreichungen. Gesucht wurde über sieben
  Geschäftsjahre. Belegbar sind nur: der Regalbestandsausgleich von
  4.530 Mio. INR, der Rückgang Nordamerikas um 31.427 Mio. INR und der
  Rückgang des Therapiegebiets Onkologie um 22.667 Mio. INR — zwischen den
  dreien stellt der Bericht keinen Zusammenhang her, und dieser Bericht tut es
  auch nicht.
- **Zwei Startjahre, zwei Vorzeichen.** Der freie Mittelzufluss je Aktie wuchs
  von FY2018 an um 12,77 % p.a. und von FY2019 an um −2,08 % p.a. Bei der
  höheren Rate (über der Hürde) gibt es keine Preisobergrenze mehr, bei der
  niedrigeren liegt die Schwelle bei 178,66 INR gegen einen Kurs von
  1.169,51 INR. Daraus folgt das Votum „meiden", und zwar wegen
  Unbestimmtheit, nicht wegen Überbewertung.
- **Kein Bewertungsereignis, sondern ein Ertragsereignis.** Das KGV lag am
  Jahreshoch bei 28,7 und liegt heute bei 22,7; gefallen ist überwiegend der
  Nenner. Bei AstraZeneca ist es umgekehrt.
- **Nettokasse, deshalb keine Entschuldungsrechnung.** Teil 5 rechnet
  stattdessen die Finanzierungsfrage. Das ist genau die Strukturwarnung aus
  `latex-project-pattern.md`.
- 1:5-Aktiensplit zum 28.10.2024, ADR-Verhältnis unverändert 1:1. Die
  Nasdaq-Reihe ist zurückgerechnet.

## Methodische Entscheidungen

- **Hürde**: Zehnjahres-Gesamtrendite des Dow Jones U.S. Select
  Pharmaceuticals Index (10,16 %) statt des ihn abbildenden Fonds (9,74 %) —
  die strengere Latte. Vom Auftraggeber als Branchenindex vorgegeben.
- **Drei Endwertannahmen statt einer.** Der Buchwert allein produziert für
  jedes Pharmaunternehmen eine Schwelle weit unter dem Markt und macht das
  Votum unbrauchbar. Die dritte Annahme (konstantes Vielfache, belegte
  Wachstumsrate) ist die, an der das Votum hängt; alle drei stehen im
  Dokument nebeneinander.
- **Keine Gegenüberstellung der beiden Titel.** Der Auftrag lautete auf zwei
  getrennte Voten. Die Reihen sind zudem nicht exakt gleich definiert (bei
  AstraZeneca ist die Leasingtilgung abgezogen, bei Dr. Reddy's weist der
  Abschluss sie nicht in derselben Gliederung aus).
- **Analystenkonsens** ist keine Lücke, sondern eine Sekundärquelle. Beide
  Konsenswerte bestehen den Staleness-Test: das aus Kursziel und Aufschlag
  zurückgerechnete Bezugsniveau trifft auf den Cent den Schlusskurs vom
  17.09.2026 (AZN 166,14 / 166,15; RDY 12,19 / 12,19).

## Werkzeugbefunde (für die nächste Analyse)

- **Moderne iXBRL-Einreichungen haben keine `<hr>`-Seitenumbrüche mehr.** Der
  Zerleger aus Alamos findet dort null Seiten. Dr. Reddy's setzt
  `break-after:page` bzw. `page-break-after:always` in einem `<div>`;
  AstraZeneca setzt überhaupt keine Paginierung und muss über die PDF-Fassung
  der Unternehmenswebseite belegt werden.
- **Den ergiebigsten Trenner wählen, nicht den ersten, der trifft.** Die
  Einreichungen FY2023 und FY2024 von Dr. Reddy's enthalten beide Formen,
  8-mal die eine und 209-mal die andere. Wer die erste nimmt, die überhaupt
  trifft, zerlegt das Dokument in acht Teile.
- **Fußzeilenformen wechseln je Jahrgang.** AstraZeneca setzt die Seitenzahl
  2025 hinter den Titel, 2019–2024 davor, auf manchen Seiten blank; die
  Jahrgänge 2019 und 2020 verlieren den Titel beim Auslesen ganz.
  `fuelle_luecken()` schließt nur, was die Zählung erzwingt (gleicher Versatz
  links und rechts der Lücke), und der Bericht weist je Datei aus, wie viele
  Seitenzahlen gelesen und wie viele ergänzt wurden.
- **Bei Dr. Reddy's steht die US-Dollar-Umrechnung VOR den Rupienspalten.**
  Gelesen wird deshalb von hinten (`von_hinten=True`, die letzten n Werte der
  Zeile) — robust gegen Notennummern und gegen die Dollarspalte zugleich. Die
  Kettenprüfung hat diesen Wechsel der Lesestrategie bestätigt: 148
  Gleichungen halten unter beiden.
- **Klammern negativer Beträge stehen bei Dr. Reddy's in eigenen Zeilen.** Wer
  sie überspringt statt sie zu lesen, verkehrt jedes Vorzeichen — und der
  Seitenwachhund schweigt dazu, weil der Betrag unverändert auf der Seite
  steht.
- **`\num` nicht verschachteln.** Die Zahlenschicht gibt blanke Werte aus; die
  Ausgabemakros legen das `\num` darum. Andernfalls bricht siunitx mit
  „Invalid number" ab.
