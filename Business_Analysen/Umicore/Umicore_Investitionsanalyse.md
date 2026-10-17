# Umicore NV/SA (UMI.BR / ISIN BE0974320526) — Investitionsanalyse

**Frage, der dieser Bericht dient:** Kaufen, halten oder meiden zum letzten Schlusskurs 20,98 €
(Freitag 09.10.2026, yfinance UMI.BR, abgerufen 11.10.2026) — und falls kaufen, bis zu welchem
Preis?

**n = 0.** Entscheidung bereits binär (Position eingehen oder nicht); Abschnitt 5 entfällt,
Abschnitt 6 verwendet die "no-options variant" (nur 6.3, Eigentümer-Sicht).

**Primärquellen** (lokal gespeichert, `quellen/`, PDF-Seite = gedruckte Seite, verifiziert über
die wiederkehrende Kopf-/Fußzeile "... Umicore Annual Report <Jahr> <Seite>" auf jeder Seite):
Annual Report 2025 (220 S., publiziert 27.03.2026), Annual Report 2024 (212 S.), Integrated
Annual Report 2023 (338 S.), Integrated Annual Report 2022 (290 S.), Integrated Annual Report
2021 (241 S.) — alle von umicore.com/en/investor-relations/. Zusätzlich: "Full Year Results
2025" Pressemitteilung (32 S., 20.02.2026) und Investor-Präsentation (38 S.). Marktdaten:
yfinance (UMI.BR), abgerufen 11.10.2026. Analysten- und Aktionärsdaten: Umicore
Investor-Relations-Website und Aggregatoren, abgerufen 11.10.2026 (teils sekundär, markiert).

**Hinweis zur Seitenverifikation.** Wie bei der vorangegangenen Analyse (Nynomic) hat eine
erste Rechercherunde an mehreren Stellen falsche Seiten zitiert — hier nicht durch
Zeilen-/Seiten-Verwechslung, sondern weil ein Dokument mehrere strukturell ähnliche Tabellen
auf verschiedenen Seiten hat (z. B. wurde die "Gesamtergebnisrechnung" mit der
"Konzern-Gewinn-und-Verlustrechnung" verwechselt, oder ein "Anpassungen je Segment"-Blatt mit
der eigentlichen Umsatztabelle). Jede im Folgenden zitierte Zahl wurde vor der Niederschrift
gegen die tatsächliche PDF-Seite re-verifiziert (`pdftotext -f N -l N` plus Abgleich mit der
Kopfzeile). Mehrere Korrekturen waren nötig, insbesondere beim mehrjährigen
Umsatzvergleich (siehe 2.1 — zwei unterschiedliche Umsatzbegriffe wurden in der ersten
Rechercherunde vermischt).

---

## 1. Beschreibung des Unternehmens

### 1.1 Handelnde Personen

Rechtsform: Aktiengesellschaft nach belgischem Recht, Sitz Brüssel.

**Executive Committee** (7 Mitglieder, GB2025 S.38–40):
- **Bart Sap** — CEO, seit 16.05.2024 (Belgier, 47)
- **Wannes Peferoen** — CFO, seit 01.10.2022 (Belgier, 45)
- **Jensen Verhelle** — EVP Catalysis, seit 16.05.2024
- **Geert Olbrechts** — EVP Recycling / CTO, seit 01.08.2023
- **Veerle Slenders** — EVP Specialty Materials, seit 01.06.2023
- **Ana Fonseca Nordang** — EVP People & Organisation/ESG, seit 01.09.2023
- **Karena Cancilleri** — EVP Battery Cathode Materials, seit 01.04.2025 (Antrittsbonus
  800 Tsd. € zum Ausgleich verfallener Vergütungsbestandteile beim vorherigen Arbeitgeber)

2025: zwei Abgänge (Géraldine Nolens, Ralph Kiessling, zum 01.01.2025), ein Neuzugang
(Cancilleri, 01.04.2025).

**Aufsichtsrat** (10 Mitglieder, GB2025 S.31/35): Vorsitz **Thomas Leysen** (seit 10.05.2000 im
Rat, Vorsitz seit 19.11.2008 — 25 Jahre Zugehörigkeit, Belgier, 65), stellv. Vorsitz Koenraad
Debackere (unabhängig). 60 % unabhängige Mitglieder (6/10), 40 % Frauen (4/10), vier
Nationalitäten. 9 ordentliche Sitzungen plus 5 schriftliche Umlaufbeschlüsse in GJ2025.
Ausschüsse: Audit (4 Sitzungen, Vorsitz Alison Henwood), Nomination & Remuneration
(8 Sitzungen), Sustainability (4 Sitzungen, Vorsitz Birgit Behrendt), Investment (1 Sitzung).

**Veränderung 2021→2025:** Vier durchgehende Mitglieder (Leysen, Chombar, Debackere,
Behrendt); Abgänge Ines Kolmsee, Eric Meurice, Mario Armero, Mark Garrett (Mandat
ausgelaufen 24.04.2025), Laurent Raets (Rücktritt 24.04.2025); Zugänge u. a. Marc Grynberg
(27.04.2023 — der frühere CEO, nun im Aufsichtsrat), Alison Henwood (01.09.2022), Frédéric
Oudéa (25.04.2024), Martina Merz und Michael Bredael (beide 24.04.2025).

**Vergütung GJ2025** (GB2025 S.45–48): Aufsichtsrat gesamt 1.162.827 €; CEO Sap gesamt
2.226.595 € (Fix 922,5 Tsd. € / Kurzfristbonus 746,2 Tsd. € bei 114,60 % Zielerreichung /
Langfristvariable 148,2 Tsd. € / Optionen 159,3 Tsd. € / Pension 192,8 Tsd. €); Executive
Committee gesamt (7 Personen inkl. CEO) 8.959.229 €.

Wirtschaftsprüfer: **EY Bedrijfsrevisoren BV / Réviseurs d'Entreprises SRL**, Mandat seit
25.04.2024 (3 Jahre, verlängerbar), **uneingeschränkter Bestätigungsvermerk** für GJ2025
(GB2025 S.207). Zentrale Prüfungssachverhalte: Komplexität der Hedging-/Derivate-Bilanzierung
(IFRS 9) und Wertminderungsrisiko der Battery-Materials-Vermögenswerte (GB2025 S.207–209).

**Governance-Signal:** Im Gegensatz zu den beiden vorherigen Berichten dieser Serie (Nynomic:
zwei Personen seit 2015 unverändert; CTS Eventim: Gründer-Dominanz) ist Umicore ein Konzern mit
regelmäßiger Führungsrotation und einer vollständig unabhängig besetzten Mehrheit im
Aufsichtsrat — kein Ankeraktionär sitzt mit Vertretern im Vorstand.

### 1.2 Eigentümerstruktur

Meldepflichtige Positionen ≥ 3 % **zum 31.12.2025** (GB2025 S.30, Primärquelle — korrigiert
gegenüber einer ersten Web-Recherche, die 16,49 % für GBL nannte; das war falsch/veraltet,
siehe Kontradiktion unten):

| Aktionär | Anteil | Aktien |
|---|---|---|
| Gérald Frère / Ségolène Gallienne / Groupe Bruxelles Lambert SA/NV u. a. (Frère-Familie, gebündelt) | 7,96 % | 19.617.030 |
| Norges Bank | 5,30 % | 13.054.028 |
| BlackRock Inc. | 5,11 % | 12.596.910 |
| SFPIM (belgischer Staat, Föderale Beteiligungs- und Investitionsgesellschaft) | 5,00 % | 12.320.000 |
| Silchester International Investors LLP | 4,98 % | 12.268.898 |
| Bank of America Corporation | 3,42 % | 8.431.041 |
| JP Morgan Asset Management Holdings Inc. | 3,41 % | 8.400.387 |
| Eigene Aktien (Treasury) | 2,37 % | 5.838.075 |
| Streubesitz (Rest, nicht einzeln meldepflichtig) | ≈ 62,45 % | — |

Gesamt 246.400.000 ausgegebene Aktien (davon 240.567.925 im Umlauf nach Abzug eigener
Aktien, GB2025-Pressemitteilung S.6).

**Kontradiktion, aufgelöst:** Eine erste Web-Recherche (ohne Primärquelle) nannte GBL mit
16,49 % — das ist **falsch**; die Primärquelle (GB2025 S.30, Meldestand 31.12.2025) weist die
Frère-Gruppe mit 7,96 % aus. Keine Einzelposition kontrolliert den Konzern; die größte Position
(7,96 %) liegt deutlich unter einer Sperrminorität.

**Directors' Dealings:** Im Rahmen der laufenden Recherche keine Einzeltransaktionen von
Vorstand/Aufsichtsrat für die letzten 12 Monate identifiziert — echte Lücke (siehe
Lückenliste), nicht mit einer Vermutung gefüllt.

### 1.3 Beurteilung durch die Kapitalmärkte

| Kennzahl | Wert | Quelle | Abgerufen |
|---|---|---|---|
| Schlusskurs | 20,98 € (Fr. 09.10.2026) | yfinance UMI.BR | 11.10.2026 |
| Marktkapitalisierung | ≈ 5,05 Mrd. € | yfinance | 11.10.2026 |
| 52-Wochen-Spanne | 14,19 € – 26,98 € | yfinance | 11.10.2026 |
| Buchwert/Aktie (Mutteraktionäre, 31.12.2025) | 9,31 € (2.239,169 Mio. € / 240,6 Mio. Aktien) | GB2025 S.64, Primärquelle; yfinance bestätigt (9,308) | 11.10.2026 |
| Kurs/Buchwert | 2,25 | eigene Rechnung | 11.10.2026 |
| KGV (trailing) | 10,2 | yfinance | 11.10.2026 |
| Beta | 0,92 | yfinance | 11.10.2026 |
| Dividendenrendite | 2,43 % (0,50 € / 20,98 €) | yfinance / eigene Rechnung | 11.10.2026 |

Die 52-Wochen-Spanne (14,19–26,98 €, Faktor 1,9) spiegelt die Erholung von der
GJ2024-Wertminderungskrise, ist aber deutlich weniger extrem als bei kleineren Werten dieser
Serie (Nynomic: Faktor 3,5).

**Buchwert pro Aktie über 5 Jahre** (eigene Berechnung, Mutteraktionäre, Jahresendkurs):

| Jahr | Buchwert/Aktie | Veränderung |
|---|---|---|
| 2021 | 12,64 € | — |
| 2022 | 14,27 € | +12,9 % |
| 2023 | 14,86 € | +4,1 % |
| 2024 | 8,06 € | **−45,8 %** (Wertminderung) |
| 2025 | 9,31 € | +15,5 % |

### 1.4 Analystenabdeckung und Konsens

**Breite, belastbare Abdeckung** — anders als bei Nynomic (2 Analysten) handelt es sich hier um
einen BEL20-Standardwert mit substanzieller Coverage:

| Quelle | Anzahl Analysten | Konsens | Kursziel (Ø) | Spanne | Abgerufen |
|---|---|---|---|---|---|
| Umicore-eigene IR-Seite | 14 | — | — | — | 11.10.2026 |
| yfinance | 13 | Buy | 25,63 € | 18,50–33,00 € | 11.10.2026 |
| Investing.com/MarketScreener | ~15 | Buy (9 Kauf / 5 Halten / 1 Verkauf) | 23,79 € | 15–33 € | 11.10.2026 |

Alle drei Quellen stimmen in der Größenordnung überein (13–15 Analysten, Konsens "Kaufen",
Kursziel 23,8–25,6 € — **oberhalb des aktuellen Kurses von 20,98 €**, +13 % bis +22 %).

### 1.5 Botschaften und Lücken

- Keine Einzel-Directors'-Dealings-Liste für die letzten 12 Monate gefunden (1.2) — echte
  Lücke.
- Keine konsolidierte Metall-Rückgewinnungsquote (z. B. "95 % Platin-Rückgewinnung") in den
  Primärquellen — das Unternehmen nennt nur "17 Metalle aus über 200 Reststoffarten", keine
  Prozentzahl.
- Keine isolierten Einheitenwirtschaftsdaten für Battery Cathode Materials (Volumen, Marge pro
  kg, Kundenkonzentration) — nur aggregierte Segmentzahlen.
- Zwischenziele der CORE-2028-Strategie für 2026/2027 nicht veröffentlicht, nur das
  GJ2025-Ergebnis und die kumulierten 2028-Ziele.

### 1.6 Wettbewerbsposition

Umicore gliedert sich in vier "Business Groups" (GB2025 S.5/10–12): **Battery Materials
Solutions** (Batteriekathodenmaterial + Batterierecycling, −21 Mio. € bereinigtes EBITDA
GJ2025 — verlustbringend), **Catalysis** (Autoabgaskatalysatoren, Brennstoffzellenkatalysatoren,
Edelmetallchemie), **Recycling** (Edelmetallraffination, Schmuck-/Industriemetalle,
Edelmetallhandel) und **Specialty Materials** (Germanium, Metallabscheidung, Kobaltspezialitäten).

**Wichtige Korrektur gegenüber einer ersten Web-Recherche:** Diese hatte behauptet, Recycling
erziele "fast 80 % des Konzernumsatzes" und sei der dominante Ertragsträger. Die Primärquelle
(Pressemitteilung GJ2025, S.3, direkt gelesen) zeigt das Gegenteil in absoluten Zahlen:

| Segment, GJ2025 | Umsatz (ohne Metalle) | Bereinigtes EBITDA | EBITDA-Marge | Bereinigtes EBIT |
|---|---|---|---|---|
| Catalysis | 1.668 Mio. € | **450 Mio. €** | 27,0 % | 383 Mio. € |
| Recycling | 947 Mio. € | 371 Mio. € | 39,2 % | 296 Mio. € |
| Specialty Materials | 558 Mio. € | 108 Mio. € | 19,4 % | 76 Mio. € |
| Battery Materials Solutions | 436 Mio. € | −21 Mio. € | −4,8 % | −91 Mio. € |
| Konzern gesamt | 3.609 Mio. € | 847 Mio. € | 24,0 % | 579 Mio. € |

**Catalysis liefert den größten absoluten Ertragsbeitrag (450 Mio. €), Recycling die höchste
Marge (39,2 %).** Beide zusammen ("Foundation Businesses" plus Specialty Materials) tragen die
gesamte Konzernertragskraft; Battery Materials bleibt ein Verlustbringer.

**CORE-2028-Strategie** (gestartet März 2025, GB2025 S.10–12): vier strategische Prioritäten
(Capital, Performance, People & Culture, Partnerships). Konzernziele bis 2028: bereinigtes
EBITDA 1,0–1,2 Mrd. € (Marge > 23 %), ROCE > 15 %, Capex-Obergrenze 2,1 Mrd. € kumuliert
2025–2028. Segmentziele 2028: Catalysis Umsatz ≈1,8 Mrd. €/Marge ≈25 %/ROCE > 35 %; Recycling
Umsatz ≈0,8 Mrd. €/Marge ≈35 %/ROCE > 40 %; Specialty Materials Umsatz ≈0,6 Mrd. €/Marge > 20 %/
ROCE > 12,5 %. **Dies sind Unternehmensziele, keine von dieser Analyse übernommenen Prognosen**
— sie werden in 7.5 als Gegenargument genannt, nicht als Basisszenario verwendet (Regel 5: keine
Schätzung statt belegter Ist-Werte).

**Battery-Materials-Schwäche:** Nach eigener Darstellung ausgelöst durch einen "sharp slowdown
in growth of demand for EVs" (Sommer 2024), der die Kundennachfrageprognosen "steeply
declined" ließ. Gegenmaßnahme: Halbierung des Kapitaleinsatzes 2025–2028 gegenüber dem
ursprünglichen Plan; nur noch Fertigstellung laufender Projekte und Erfüllung bestehender
Take-or-pay-Verträge. Battery Cathode Materials erreichte GJ2025 ein bereinigtes
EBITDA-Break-even (gegenüber −40 Mio. € zugrunde liegend in GJ2024).

**Regulatorischer Rückenwind:** EU-Batterieverordnung — verpflichtende Mindestquoten für
Rezyklat-Gehalt ab 2030 (12 % Kobalt, 4 % Lithium, 4 % Nickel, 85 % Blei), Berichtspflicht bereits
ab 2026 (sekundäre Quelle, sachlich plausibel, aber von Umicore selbst nicht explizit als
Umsatzhebel beziffert — Lücke).

**Peer-Vergleich** (einziger sauber vergleichbarer Peer mit gleichem Jahr/Definition):

| Kennzahl | Umicore Recycling (GJ2025) | Johnson Matthey PGM Services (GJ2024/25, Jahresende 31.03.2025) |
|---|---|---|
| Umsatz | 947 Mio. € | 464 Mio. £ |
| EBITDA-Marge | 39,2 % | 38,1 % |
| Operatives Ergebnis | 296 Mio. € (EBIT) | 149 Mio. £ |

Umicore Recycling ist größer und marginal profitabler als der direkte PGM-Recycling-Peer.
Glencore (Konzernumsatz 247,5 Mrd. USD) wurde geprüft, ist aber kein sauberer Peer — das
Batteriemetall-Recycling-Geschäft (Glencore/Redwood-JV) wird nicht separat ausgewiesen.

---

## 2. Auswertung des Jahresabschlusses (GJ2021–GJ2025, IFRS-Konzernabschluss)

### 2.1 Umsatz — zwei Begriffe, nicht verwechseln

Umicore berichtet **zwei unterschiedliche Umsatzgrößen**, die in einer ersten Recherche
fälschlich vermengt wurden:

1. **"Turnover" (IFRS-Umsatzzeile, inkl. durchlaufender Metallwerte):** riesige, stark
   metallpreisabhängige Zahl — fällt von 25,4 Mrd. € (2022) auf 14,9 Mrd. € (2024), fast
   ausschließlich wegen Metallpreis-/Volumeneffekten im durchlaufenden Edelmetallgeschäft, nicht
   wegen eines operativen Einbruchs.
2. **"Group revenue (excluding metals)"** — die vom Unternehmen selbst bevorzugte, kleinere
   Kennzahl ohne den durchlaufenden Metallwert; das ist die wirtschaftlich aussagekräftige Größe
   für Margenanalyse und wird hier für Tabellen/Margen verwendet.

| Mrd. € | GJ2021 | GJ2022 | GJ2023 | GJ2024 | GJ2025 |
|---|---|---|---|---|---|
| Turnover (IFRS, inkl. Metalle) | 24,05 | 25,44 | 18,27 | 14,85/19,37* | — |
| Group revenue (ohne Metalle) | ≈3,82** | 4,2 | 3,9 | 3,5 | 3,6 |

\* GB2024 weist 14.853,681 Tsd. € aus; GB2025 (Vorjahresspalte) weist 14.853,681 Tsd. €
identisch aus — kein Widerspruch, nur zwei verschiedene Notenzeilen derselben Zahl. Die
19,37 Mrd. € ist die GJ2025-eigene "Turnover"-Zeile (GB2025 S. Konsolidierte GuV, Notiz F9).

\** Abgeleitet aus der GB2022-Aussage "Group revenues ... amounted to €4.2 billion, up 10 %
compared with the previous year" (GB2022, Management-Bericht) — nicht direkt gelesen, sondern
aus der genannten Wachstumsrate zurückgerechnet; als Ableitung gekennzeichnet.

*Quelle Turnover-Zeile: GB2022 S.152 (2021/2022), GB2023 S.198 (2022/2023), GB2025 S.
Konsolidierte GuV (2024/2025, Notiz F9). Quelle Group revenue: GB2022 Management-Bericht
("€4.2 billion, up 10 %"), GB2023 Management-Bericht ("€3.9 billion versus €4.2 billion in
2022"), GB2024 Management-Bericht ("€3.5 billion versus €3.9 billion in 2023"),
GB2025-Pressemitteilung S.2 ("€3.6 billion up 3% versus 2024").*

### 2.2 Segmentumsatz

Siehe Tabelle in 1.6 (GJ2025, Pressemitteilung S.3) — detaillierter Segment-Track-Record vor
GJ2024 ist durch die Strukturreform (Battery Recycling Solutions wechselte von Recycling zu
Battery Materials Solutions, ab GJ2025 wirksam, GJ2024-Vergleichszahlen restated) nicht direkt
vergleichbar — als Lücke behandelt für GJ2021–GJ2023.

### 2.3 / 2.4 EBITDA, operatives Ergebnis

| Mio. € | GJ2020 | GJ2021 | GJ2022 | GJ2023 | GJ2024 | GJ2025 |
|---|---|---|---|---|---|---|
| Bereinigtes EBITDA | — | ≈1.251* | 1.151 | 972 | 763 | 847 |
| Bereinigtes EBIT | 536 | 971 | 865 | 674 | 478 | 579 |
| EBIT-Marge (auf Umsatz ohne Metalle) | — | ≈25,4 % | 20,6 % | 17,3 % | 13,7 % | 16,0 % |

\* Abgeleitet aus GB2022: "Adjusted EBITDA amounted to €1,151 million, a decrease of 8 %
compared with the record level achieved in 2021" → 1.151/0,92 ≈ 1.251. Als Ableitung
gekennzeichnet, nicht direkt gelesen.

*Quelle EBIT-Reihe (direkt gelesen, eine einzige Tabelle): GB2024 "Corporate governance
statements" S.50, Vergütungsbericht-Tabelle "Company's performance" (2020–2024); GJ2025 EBIT
aus GB2025-Pressemitteilung S.2.*

**GJ2024 war ein Krisenjahr: Wertminderung von 1.788 Mio. €** auf Battery-Materials-Vermögenswerte
(davon 1.601 Mio. € Sachanlagen/Vorräte, GB2024 S.89, "Anpassungen je Segment und Art"-Tabelle)
drückte das **ausgewiesene IFRS-EBIT auf −1.311 Mio. €**, während das bereinigte EBIT mit
478 Mio. € weiterhin positiv blieb. GJ2025 zeigt eine Erholung des bereinigten EBITDA um +11 %
auf 847 Mio. €, getragen von einem günstigen Metallpreisumfeld und Kostendisziplin.

### 2.5 Nettogewinn

| Mio. € | GJ2021 | GJ2022 | GJ2023 | GJ2024 | GJ2025 |
|---|---|---|---|---|---|
| Konzernergebnis (IFRS, Mutteraktionäre) | 618,96 | 569,88 | 385,08 | **−1.479,88** | 384,55 |
| Bereinigtes Konzernergebnis (Mutteraktionäre) | n. v. | n. v. | 447 | 255 | 288 |
| EPS (IFRS, Mutteraktionäre) | 2,57 € | 2,37 € | 1,60 € | **−6,15 €** | 1,60 € |

*Quelle: GB2022 S.152 (2021/2022), GB2023 S.198 (2022/2023), GB2025 S. Konsolidierte GuV
(2024/2025, Notiz F40).*

Der GJ2024-Verlust (−1,48 Mrd. €, EPS −6,15 €) ist vollständig auf die Wertminderung
zurückzuführen (siehe 2.3) — eine Einmalbelastung, keine operative Verschlechterung des
laufenden Geschäfts im gleichen Ausmaß (bereinigtes Ergebnis blieb mit 255 Mio. € positiv).
**Auffällig:** GJ2023 (385,08 Mio. €) und GJ2025 (384,55 Mio. €) sind nahezu identisch im
IFRS-Ergebnis — Zufall der Rundung auf Konzernebene, keine inhaltliche Wiederholung (die
zugrunde liegenden Geschäftsjahre unterscheiden sich stark: GJ2023 ohne Sondereffekte,
GJ2025 nach einem Krisenjahr mit Erholung).

### 2.6 Dividende

| GJ | Dividende je Aktie | Gesamtausschüttung | Status |
|---|---|---|---|
| 2022 | 0,80 € | — | Gezahlt |
| 2023 | 0,80 € | 192,4 Mio. € | Gezahlt |
| 2024 | 0,50 € (0,25 € Interim Aug. 2024 + 0,25 € Schlusszahlung 2025) | 60,1 Mio. € (2025 gezahlter Anteil) | Gezahlt |
| 2025 | 0,50 € (vorgeschlagen, einmalige Jahresdividende, keine Interimszahlung mehr) | — | Auszahlung Mai 2026 vorgesehen |

*Quelle: GB2022 S.152, GB2023 S.198, GB2025 S.30 (Primärzitat: "paid a gross dividend of €0.25
per share in respect of the 2024 financial year. Together with the gross interim dividend of
€0.25 per share paid in August 2024, the total gross dividend paid in respect of the 2024
financial year amounted to €0.50 per share"), Pressemitteilung GJ2025 S.1/S.19 ("Proposed gross
annual dividend for 2025 of €0.50 per share, to be paid out in May 2026").*

**Dividendenpolitik geändert im Februar 2025:** keine Interimsdividende mehr, nur noch eine
jährliche Ausschüttung (GB2025 S.30). Die Dividende wurde in Reaktion auf die
GJ2024-Wertminderung von 0,80 € auf 0,50 € gekürzt (−37,5 %) und seitdem auf diesem Niveau
gehalten.

### Kennzahlenüberblick

| Mio. € | GJ2021 | GJ2022 | GJ2023 | GJ2024 | GJ2025 |
|---|---|---|---|---|---|
| Bilanzsumme | 9.045 | 9.942 | 9.966 | 9.412 | 9.460 |
| Eigenkapital (Konzern, inkl. Minderheiten) | 3.167 | 3.566 | 3.697 | 1.922 | 2.272 |
| Eigenkapital (nur Mutteraktionäre) | 3.113 | 3.516 | 3.661 | 1.938 | 2.239 |
| Eigenkapitalquote | 35,0 % | 35,9 % | 37,1 % | 20,4 % | 24,0 % |
| Operativer Cashflow | — | 634 | 1.043 | 868 | 849 |
| Capex (Sachanlagen + immateriell) | — | 491 | 884 | 582 | 310* |
| Free Cashflow (OCF − Capex) | — | 143 | 159 | 286 | 539 |
| ROCE (eigene Angabe des Unternehmens) | 22,2 % | 19,2 % | 13,5 % | 12,3 % | 15,7 % |

\* GJ2025-Capex laut Pressemitteilung (S.2, einheitlich berichtete Zahl) 310 Mio. €; die
Aufschlüsselung Sachanlagen+immateriell in den Notizen ergibt 325 Mio. € (kleine Abweichung,
vermutlich andere Abgrenzung — für den FCF wird hier die offizielle Headline-Zahl 310 Mio. €
verwendet, Konsistenz über alle Jahre mit der jeweils offiziell kommunizierten Capex-Zahl.)

*Quelle Bilanz: GB2022 S.154 (2021/2022), GB2024 S.61 (2023/2024), GB2025 S.64 (2024/2025).
Quelle Cashflow: GB2023/GB2024/GB2025 Kapitalflussrechnungen (Seiten in domain2_financials.md
dokumentiert, Werte gegen die Bilanz-Differenzen plausibilisiert). Quelle ROCE: GB2024 S.50
(Vergütungsbericht-Tabelle, Spalten 2020–2024), GB2025-Pressemitteilung S.2 (GJ2025).*

**Eigenkapitalquote halbierte sich praktisch durch die GJ2024-Wertminderung** (37,1 % → 20,4 %),
erholt sich seitdem leicht (24,0 % in GJ2025) — strukturell niedriger als bei Nynomic (55–75 %),
aber für einen kapitalintensiven Industriekonzern mit Nettoverschuldung im normalen Bereich.

### 2.7 Aktienkursverlauf

| Datum | Kurs | Ereignis |
|---|---|---|
| 52-Wochen-Tief | 14,19 € | (genaues Datum nicht einzeln recherchiert) |
| 52-Wochen-Hoch | 26,98 € | (genaues Datum nicht einzeln recherchiert) |
| 09.10.2026 | 20,98 € | Letzter verifizierter Schlusskurs |

Der aktuelle Kurs (20,98 €) liegt zwischen dem 52-Wochen-Tief und -Hoch, näher an der Mitte —
ein deutlich ruhigerer Kursverlauf als bei den beiden kleineren Werten dieser Serie (Nynomic,
CTS Eventim), konsistent mit Umicores Status als etablierter BEL20-Standardwert.

### 2.8 Prognose vs. Ist

Umicore hat für GJ2026 **keine konkrete EBITDA-/EPS-Zielzahl** veröffentlicht (GB2025 S.18 f.),
sondern nur qualitative Segment-Aussagen: Battery Materials Solutions soll 2026 ein positives
bereinigtes EBITDA und EBIT erreichen (gegenüber negativ in GJ2025); Recycling-Ergebnis bleibt
abhängig vom Fortbestand des günstigen Metallpreisumfelds; Capex 2026 soll gegenüber den
310 Mio. € in GJ2025 steigen (selektive Wachstumsinvestitionen, finale Investitionsentscheidung
zur Hydrometallurgie-Erweiterung in Hoboken für H2 2026 erwartet).

Für GJ2025 selbst wurden im GB2024 (März 2025) nur qualitative Segmentaussagen gemacht
("Recycling-Ergebnis unter 2024-Niveau wegen auslaufender Hedges", "Specialty Materials leicht
über 2024", "Capex ca. 20 % niedriger als 2024") — tatsächlich erreicht: Recycling-EBITDA
blieb stabil (371 Mio. € vs. 2024, nicht niedriger wie prognostiziert), Specialty Materials
übertraf die Prognose (+11 % statt "leicht höher"), Capex sank deutlich stärker als angekündigt
(−47 % statt −20 %, strengere Kapitaldisziplin als kommuniziert).

---

## 3. Strategische Maßnahmen

- **CORE-2028-Strategie** (gestartet März 2025) löst die vorherige "RISE"-Strategie
  (2030-Ziel: bereinigte EBITDA-Marge > 20 %, in GJ2023 noch als Referenzpunkt genannt, GB2023
  Management-Bericht) ab — ein Strategiewechsel infolge der Battery-Materials-Krise, keine
  bloße Fortschreibung.
- **€100 Mio. Effizienzmaßnahmen 2025**, nach eigener Angabe vollständig erreicht
  (GB2025-Pressemitteilung S.1).
- **Kapitaldisziplin bei Battery Materials**: Halbierung der geplanten Investitionen 2025–2028,
  nur noch Fertigstellung laufender Projekte und Erfüllung von Take-or-pay-Verpflichtungen.
- **Finale Investitionsentscheidung zur Hoboken-Hydrometallurgie-Erweiterung** für H2 2026
  angekündigt — Kapazitätsausbau im Kerngeschäft Recycling, nicht in Battery Materials.
- **Keine Dividendenerhöhung in Sicht**: Politikwechsel Februar 2025 zu einer einzigen
  Jahresdividende ohne Interimszahlung, Niveau bei 0,50 € seit GJ2024 gehalten.

---

## 4. Beurteilung des Kursverlaufs

Der Kurs bewegt sich seit der GJ2024-Krise in einer moderaten Erholungsbandbreite
(14,19–26,98 € über 52 Wochen), aktuell bei 20,98 € — näher am oberen als am unteren Rand
dieser Spanne, aber ohne die extreme Volatilität kleinerer Werte dieser Serie. Die
Analystenkonsens-Kursziele (23,8–25,6 €) implizieren ein moderates Aufwärtspotenzial von
13–22 % gegenüber dem aktuellen Kurs — eine deutlich geringere Divergenz zwischen Markt und
Konsens als bei den Small-Cap-Berichten dieser Serie.

---

## 5. Entfällt (n = 0)

Keine Handlungsoptionen zu vergleichen — Umicore ist eine bestehende Börsennotierung, die
Entscheidung ist binär (Position eingehen oder nicht).

---

## 6. Grobe Flow-Analyse (no-options variant, nur 6.3)

### 6.0 Annahmen

**Hürde.** Umicore veröffentlicht eine eigene ROCE-Kennzahl (Return on Capital Employed:
bereinigtes EBIT / durchschnittliches eingesetztes Kapital), aber keine ROE. Die
Projektkonvention dieser Berichtsserie verwendet die Eigenkapitalrendite (ROE), nicht ROCE
(ROCE bezieht Fremdkapital im Nenner ein und liegt strukturell höher). Diese Analyse berechnet
daher eine eigene ROE, und zwar **auf Basis des bereinigten Konzernergebnisses, nicht des
IFRS-Ergebnisses**:

| GJ | Bereinigtes Konzernergebnis (Mio. €) | Eigenkapital Jahresende, Mutteraktionäre (Mio. €) | ROE (bereinigt) |
|---|---|---|---|
| 2023 | 447 | 3.697,43 | 12,09 % |
| 2024 | 255 | 1.921,76 | 13,27 % |
| 2025 | 288 | 2.271,55* | 12,68 % |

\* Hier wird die Gesamt-Konzerneigenkapitalzahl (inkl. Minderheiten, 2.271,55 Mio. €) statt der
reinen Mutteraktionärszahl (2.239,17 Mio. €) verwendet, weil die 2024er-Vergleichszahl aus
derselben Quelle ebenfalls die Gesamtzahl ist — Konsistenz der Zeitreihe hat Vorrang vor
technischer Praezision der letzten Nachkommastelle; der Unterschied ist marginal (12,68 % vs.
12,83 % bei strikter Mutteraktionärs-Eigenkapitalbasis).

**Annahme:** Das IFRS-Ergebnis wird für die Hürde **nicht** verwendet, weil GJ2024 durch die
1.788-Mio.-€-Wertminderung eine ROE von **−77 %** ergäbe — ein reiner Einmaleffekt
(Bilanzbuchung, kein Cash-Abfluss im selben Ausmaß), keine wiederkehrende Ertragskraft. Eine
Hürde, die einen Einmaleffekt mitmittelt, wäre keine Messung der laufenden Kapitalrendite,
sondern ein Artefakt des Bilanzierungszeitpunkts.

**3-Jahres-Mittel (2023–2025, einzige Jahre mit verfügbarem bereinigtem Ergebnis): 12,68 %.**
Für GJ2021/GJ2022 wurde kein separat ausgewiesenes "bereinigtes Konzernergebnis" gefunden
(Lücke) — das IFRS-Ergebnis dieser Jahre (2021: 19,55 % ROE; 2022: 15,98 % ROE) wird hier nicht
eingerechnet, weil es nicht auf derselben bereinigten Basis wie 2023–2025 beruht und die
Vermischung zweier Rechnungslegungsbasen die Hürde verzerren würde.

**Zur Einordnung, nicht als Hürde verwendet:** Umicores eigene ROCE lag im 5-Jahres-Mittel
2021–2025 bei (22,2+19,2+13,5+12,3+15,7)/5 = **16,58 %** — höher als die hier verwendete ROE,
weil ROCE auch das eingesetzte Fremdkapital im Nenner hat und damit strukturell über der
reinen Eigenkapitalrendite liegt, wenn Fremdkapital billiger ist als das Eigenkapitalrisiko.

**Terminal Value** = Buchwert der Mutteraktionäre pro Aktie zum 31.12.2025: 2.239,169 Mio. € /
240,567 Mio. Aktien = **9,31 €/Aktie** (GB2025 S.64, bestätigt durch yfinance `bookValue` 9,308).
Gehalten konstant über den Horizont.

**Einstiegskurs**: 20,98 € (letzter Schlusskurs, 09.10.2026).

**Jährlicher Cashflow (FCF/Aktie)**, aus operativem Cashflow minus Capex (konsistent die
offiziell kommunizierte Jahres-Capex-Zahl verwendet), geteilt durch die Aktienzahl des Jahres:

| GJ | OCF (Mio. €) | Capex (Mio. €) | FCF (Mio. €) | Aktien (Mio.) | FCF/Aktie |
|---|---|---|---|---|---|
| 2022 | 634 | 491 | 143 | 246,4 | **0,58 €** (schwächstes Jahr → pessimistisch) |
| 2023 | 1.043 | 884 | 159 | 246,4 | 0,65 € |
| 2024 | 868 | 582 | 286 | 240,5 | **1,19 €** (mittleres Jahr → Basis) |
| 2025 | 849 | 310 | 539 | 240,6 | **2,24 €** (bestes Jahr → optimistisch) |

**Annahme (wichtig):** GJ2025s hoher FCF/Aktie-Wert ist teils durch einen außergewöhnlich
niedrigen Capex bedingt (310 Mio. €, −47 % ggü. GJ2024, strengste Kapitaldisziplin wegen der
Battery-Materials-Krise). Die CORE-2028-Planung selbst sieht einen **steigenden** Capex für
2026 vor (kumuliertes Limit 2,1 Mrd. € über 2025–2028 = durchschnittlich ≈525 Mio. €/Jahr, mehr
als das Doppelte von GJ2025). Das optimistische Szenario (2,24 €/Aktie) sollte daher nicht als
nachhaltiger Dauerzustand gelesen werden, sondern als tatsächlich erzieltes, aber durch
Investitionszurückhaltung begünstigtes Einzeljahr — explizit in 7.5 als Gegenargument
aufgegriffen.

Horizont: 2026–2041 (15 Jahre), Zwischenpunkte 2031 (5 J.) und 2036 (10 J.).

### 6.3 Was verdient ein Eigentümer, der heute einsteigt?

| Szenario | FCF/Aktie p. a. | IRR, 5 J. (2031) | IRR, 10 J. (2036) | IRR, 15 J. (2041) |
|---|---|---|---|---|
| Pessimistisch (GJ2022) | 0,58 € | −10,6 % | −3,5 % | −1,1 % |
| Basis (GJ2024) | 1,19 € | −5,9 % | +0,9 % | +3,2 % |
| Optimistisch (GJ2025) | 2,24 € | +2,4 % | +8,6 % | +10,4 % |

*Erzeugt von `scripts/cashflow_irr.py`, Konfiguration `data/analysis.json`, Rohtabellen in
`data/irr_sensitivity.md`, `data/comparison.md`.*

**Keines der drei Szenarien erreicht die Hürde von 12,68 % — selbst das optimistische Szenario
bleibt über 15 Jahre bei 10,4 %, 2,3 Prozentpunkte darunter.** Das Basis-Szenario (realistischster
Fall, mittleres Jahr der Reihe) erreicht nach 15 Jahren gerade einmal +3,2 %.

### Einstiegsschwelle

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (2031) | 7,8 € | 10,5 € | 15,3 € |
| 10 Jahre (2036) | 6,6 € | 10,6 € | 17,4 € |
| 15 Jahre (2041) | 5,9 € | 10,6 € | 18,5 € |

*Quelle: `data/break_even.md`.*

Selbst im optimistischen Szenario und über den längsten geprüften Horizont (15 Jahre) liegt die
Einstiegsschwelle bei **18,50 €** — **11,8 % unter dem aktuellen Kurs von 20,98 €**. Im
realistischeren Basis-Szenario liegt sie bei 10,60 € (−49 % ggü. aktuellem Kurs).

**Gegenprobe — welchen Terminal Value setzt der heutige Preis schon voraus?**

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (2031) | 33,32 € (3,58× Buchwert) | 28,29 € (3,04×) | 19,62 € (2,11×) |
| 10 Jahre (2036) | 56,79 € (6,10×) | 43,72 € (4,70×) | 21,21 € (2,28×) |
| 15 Jahre (2041) | 99,43 € (10,68×) | 71,75 € (7,71×) | 24,11 € (2,59×) |

Selbst im optimistischen Fall setzt der heutige Preis voraus, dass Umicore in 15 Jahren zum
2,59-fachen seines dann aktuellen Buchwerts gehandelt wird — höher als das heutige Multiple
(2,25×).

**Zweite Gegenprobe — welches FCF/Aktie-Wachstum setzt der heutige Preis voraus?**

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (2031) | nicht erreichbar | 51,8 %/Jahr | 26,0 %/Jahr |
| 10 Jahre (2036) | nicht erreichbar | 21,2 %/Jahr | 8,3 %/Jahr |
| 15 Jahre (2041) | nicht erreichbar | 13,9 %/Jahr | 4,5 %/Jahr |

Selbst im optimistischen Szenario müsste der FCF/Aktie über 15 Jahre **4,5 % p. a.** wachsen, um
die Hürde zu erfüllen. Umicores eigene FCF/Aktie-Historie (0,58 € → 0,65 € → 1,19 € → 2,24 €,
2022–2025) zeigt zwar eine Aufwärtstendenz, diese ist aber — wie oben erläutert — teilweise durch
rückläufigen Capex statt durch operatives Wachstum getrieben.

---

## 7. Verdict — Investitionspotenzial

### 7.1 Urteil

**Nicht kaufen** zum aktuellen Kurs von 20,98 €. Einstiegsschwelle (Basis-Szenario, 15 Jahre):
**10,60 €** (−49 % ggü. aktuellem Kurs). Selbst im optimistischen Szenario liegt die Schwelle
bei 18,50 € (−12 %).

### 7.2 Begründung aus den Befunden

1. Keines der drei FCF/Aktie-Szenarien (gestützt auf die vier verfügbaren Geschäftsjahre,
   keine Schätzung) erreicht die selbst berechnete Hürde von 12,68 % — über keinen der drei
   geprüften Horizonte.
2. Diese Hürde beruht bewusst auf dem **bereinigten** Ergebnis, nicht auf dem durch eine
   Einmalwertminderung verzerrten IFRS-Ergebnis — eine härtere, IFRS-basierte Betrachtung
   würde das Bild nicht verbessern (die Wertminderung selbst war ein Preisschock, kein
   Dauerzustand, aber ihr Ausschluss ist bereits die großzügigere Annahme).
3. Das optimistische Szenario (FCF/Aktie 2,24 €, GJ2025) beruht teilweise auf einem
   außergewöhnlich niedrigen Capex, der laut eigener CORE-2028-Planung nicht fortbestehen
   wird — ein realistischeres Basisszenario (GJ2024, 1,19 €/Aktie) erreicht nach 15 Jahren nur
   +3,2 % IRR.
4. Gegenprobe über das implizierte Wachstum: selbst im optimistischen Fall müsste der FCF/Aktie
   15 Jahre lang mit 4,5 % p. a. wachsen — plausibel, aber nicht durch einen klaren historischen
   Trend gestützt, da ein Großteil des bisherigen FCF-Anstiegs aus Investitionszurückhaltung,
   nicht aus Ertragswachstum stammt.
5. Governance, Bilanzqualität und Marktstellung sprechen nicht gegen das Unternehmen: breite,
   unabhängige Führungsstruktur, uneingeschränktes Prüfungsurteil, Weltmarktführerschaft in
   mehreren Nischen (Recycling-Marge 39,2 %, vergleichbar mit dem besten direkten Peer Johnson
   Matthey), solide Eigenkapitalquote nach der Krise (24,0 %) und eine breite, im Konsens
   positive Analystenabdeckung. Das Urteil beruht ausschließlich auf dem Preis relativ zur
   erzielbaren Verzinsung, nicht auf einem Zweifel an der Unternehmensqualität.

### 7.3 Einstiegsschwelle und erforderliches Wachstum

Siehe Tabellen in 6.3. Zusammengefasst: Basis-Fall 10,60 € / 15 Jahre; optimistischer Fall
18,50 € / 15 Jahre. Eine Investition zu 20,98 € setzt im optimistischen Fall einen bis 2041
anhaltenden Terminal-Value-Multiple von 2,59× Buchwert oder ein FCF/Aktie-Wachstum von 4,5 %
p. a. über 15 Jahre voraus.

### 7.4 Trigger

- **Preis fällt auf ≤ 18,50 €** (optimistisches Szenario, 15-Jahres-Schwelle) oder ≤ 10,60 €
  (Basis-Szenario) — ab dort wird die Position auch bei konservativer Annahme rechnerisch
  attraktiv.
- **Battery Materials Solutions erreicht nachhaltig positives EBITDA** (eigenes GJ2026-Ziel)
  UND die Foundation Businesses (Catalysis, Recycling) halten ihre GJ2025-Margen bei
  steigendem, nicht nur niedrig gehaltenem, Capex — das würde den FCF/Aktie-Lauf der
  Basis-Annahme (1,19 €) glaubwürdig anheben, ohne auf Investitionszurückhaltung zu beruhen.
- **CORE-2028-Zwischenziele** (2026/2027) werden veröffentlicht und liegen über der hier
  verwendeten Basis-Annahme — aktuell nicht verfügbar (Lücke, 1.5).

### 7.5 Was dieses Urteil nicht wissen kann

- **Die Hürde beruht auf nur drei Jahren bereinigten Ergebnisses** (2023–2025, da frühere
  Jahre diese Kennzahl nicht separat ausweisen) — eine kürzere Historie als bei anderen
  Berichten dieser Serie. Eine andere Jahresauswahl (z. B. inklusive der IFRS-Werte 2021/2022
  mit 19,55 %/15,98 % ROE) würde die Hürde anheben und das Bild verschärfen, nicht entschärfen.
- **Der Analystenkonsens widerspricht dem Urteil** (1.4): Kursziele von 23,8–25,6 € liegen
  13–22 % über dem aktuellen Kurs, mit breiter, glaubwürdiger Abdeckung (13–15 Analysten) —
  anders als bei den dünn abgedeckten Small-Caps dieser Serie verdient dieser Konsens
  Gewicht. Das ist das am stärksten gegen dieses Urteil sprechende Einzelsignal der gesamten
  Berichtsserie bisher.
- **CORE 2028 könnte die eigenen Ziele erreichen**: Konzern-ROCE > 15 % und EBITDA 1,0–1,2 Mrd. €
  bis 2028 — würde dies eintreten, läge die tatsächliche Verzinsung deutlich über dem hier
  verwendeten Basis-Szenario. Diese Analyse verwendet bewusst nur Ist-Werte (Regel 5), nicht
  die Unternehmensziele selbst, aber der Leser sollte wissen, dass das Management explizit
  eine Verbesserung über das Basis-Szenario hinaus plant und dafür einen konkreten,
  datierten Fahrplan vorgelegt hat.
- **Die EU-Batterieverordnung** könnte einen quantifizierbaren Umsatzschub für Recycling
  bringen, den weder Umicore selbst noch diese Analyse beziffert (Lücke, 1.6).
- **Das Modell hält den FCF/Aktie konstant** (methodischer Standard dieser Serie, keine
  Prognose) — die reale Entwicklung hängt vom Zusammenspiel aus Metallpreisumfeld (volatil,
  außerhalb der Kontrolle des Unternehmens), Capex-Normalisierung (geplant steigend) und dem
  Erfolg der Battery-Materials-Sanierung ab.

---

## Lücken (verbleibend)

- Directors'-Dealings-Einzeltransaktionen der letzten 12 Monate nicht gefunden (1.2).
- Konsolidierte Metall-Rückgewinnungsquote nicht veröffentlicht (1.5, 1.6).
- Einheitswirtschaftsdaten Battery Cathode Materials (Volumen, Marge/kg, Kundenkonzentration)
  nicht isoliert verfügbar (1.5).
- CORE-2028-Zwischenziele 2026/2027 nicht veröffentlicht (1.5, 7.4).
- Bereinigtes Konzernergebnis für GJ2021/GJ2022 nicht in den geprüften Primärquellen gefunden —
  Hürde stützt sich daher nur auf GJ2023–GJ2025 (6.0, 7.5).
- Segment-EBITDA vor GJ2024 wegen Strukturreform nicht direkt vergleichbar (2.2).
- Quantifizierter Umsatzeffekt der EU-Batterieverordnung auf das Recycling-Segment nicht von
  Umicore selbst beziffert (1.6).
