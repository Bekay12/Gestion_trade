# Nynomic AG (M7U.DE / ISIN DE000A0MSN11) — Investitionsanalyse

**Frage, der dieser Bericht dient:** Kaufen, halten oder meiden zum letzten Schlusskurs 26,00 €
(Freitag 09.10.2026, yfinance M7U.DE, abgerufen 11.10.2026) — und falls kaufen, bis zu welchem
Preis?

**n = 0.** Entscheidung bereits binär (Position eingehen oder nicht); Abschnitt 5 entfällt,
Abschnitt 6 verwendet die "no-options variant" (nur 6.3, Eigentümer-Sicht).

**Primärquellen** (lokal gespeichert, `quellen/`, PDF-Seite = gedruckte Seite, verifiziert über
die wiederkehrende Kopf-/Fußzeile je Seite): Annual Report 2025 (publiziert 21.05.2026, zuletzt
überarbeitet, hier zitiert als "GB2025"), Annual Report 2024 ("GB2024"), Annual Report 2023
("GB2023"), Annual Report 2022 ("GB2022") — alle von nynomic.com/en/financial-reports/, echte
PDFs (76–80 Seiten) mit gedruckten Seitenzahlen. Halbjahresbericht 2026 ("HJB2026", 30 Seiten,
Stichtag 30.06.2026). Marktdaten: yfinance (M7U.DE), abgerufen 11.10.2026. Analysten- und
Aktionärsdaten: MarketScreener, abgerufen 11.10.2026 (sekundär, unten als solche markiert).

**Hinweis zur Seitenverifikation.** Eine erste Rechercherunde (drei parallele Teilagenten) hat an
mehreren Stellen Textzeilen-Nummern aus `pdftotext`-Ausgaben mit gedruckten Seitenzahlen
verwechselt (z. B. "S. 749" oder "S. 1032" in einem 80-seitigen Dokument — strukturell
unmöglich). Jede in diesem Bericht zitierte Seite wurde vor der Niederschrift gegen die
tatsächliche PDF-Seite re-verifiziert (`pdftotext -f N -l N` plus Abgleich mit der
Inhaltsverzeichnis-Angabe im jeweiligen Bericht). Die rohen Rechercheordner
(`domain1_governance.md`, `domain2_financials.md`, `domain3_market.md`) enthalten daher noch
unkorrigierte Seitenangaben und sollten nicht ungeprüft zitiert werden — die Zahlen selbst waren
in allen Stichproben korrekt, nur die Seitenverweise nicht.

---

## 1. Beschreibung des Unternehmens

### 1.1 Handelnde Personen

Rechtsform: Aktiengesellschaft nach deutschem Recht, Sitz Am Marienhof 2, 22880 Wedel.

**Vorstand** (unverändert seit 2015, GB2025 S.9, S.17):
- **Maik Müller** — CEO/Vorsitzender (Technologie, Operations, F&E)
- **Fabian Peters** — CFO (Finanzen, Controlling, Organisation, Investor Relations)

Vergütung des Vorstands für GJ2025 **nicht offengelegt** — Befreiung nach § 314 Abs. 3 i. V. m.
§ 286 Abs. 4 HGB in Anspruch genommen (GB2025 S.53). Das ist eine zulässige Erleichterung für
kleine/mittelgroße Konzerne, kein Hinweis auf eine Unregelmäßigkeit, aber eine echte Lücke: die
individuelle Vorstandsvergütung ist aus den Primärquellen nicht rekonstruierbar.

**Aufsichtsrat** (ebenfalls unverändert seit 2015, GB2025 S.15/17/53): Hans Wörmcke (Vorsitz,
Heist, Unternehmer), Dr. Sven Claussen (stellv. Vorsitz, Hamburg, Rechtsanwalt bei Weiland
Rechtsanwälte), Hartmut Harbeck (Wedel, Unternehmer). Zwei Sitzungen im GJ2025 (06.05. und
09.12.2025, GB2025 S.16). Gesamtvergütung Aufsichtsrat GJ2025: 83 TEUR (unverändert ggü. GJ2024,
GB2025 S.53).

**Governance-Signal:** Vorstand und Aufsichtsrat sind seit mindestens vier Geschäftsjahren
(GJ2022–GJ2025) identisch besetzt — keine Abgänge, keine Neuzugänge (GB2022 S.48 vs. GB2025
S.17: "There were no changes in the composition of either the Supervisory Board or the
Management Board during the 2025 financial year"). Das spricht für Kontinuität, aber auch für
eine sehr kleine Führungsspitze ohne sichtbare Nachfolgeplanung — zwei Vorstände, drei
Aufsichtsräte, für einen Konzern mit 24 konsolidierten Tochtergesellschaften (GB2025 S.36).

Wirtschaftsprüfer: Clauß Paal & Partner mbB, Münster (verantwortliche Prüfer Felix Hovelbernd,
Stefan Evers), uneingeschränkter Bestätigungsvermerk für GJ2025, Prüfungsbericht datiert
23.04.2026 (GB2025 S.74/77). Keine Prüferwechsel-Historie in den vier untersuchten Jahren
festgestellt.

### 1.2 Eigentümerstruktur

| Position | Anteil | Quelle | Abgerufen |
|---|---|---|---|
| Streubesitz (Free Float) | 81,36 % **oder** 85,44 % (Quellen widersprechen sich) | MarketScreener / aggregierte Sekundärquelle | 11.10.2026 |
| Kreissparkasse Biberach | 5,10 % | MarketScreener | 11.10.2026 |
| Paladin Asset Management | 3,60 % | MarketScreener | 11.10.2026 |
| Joh. Berenberg (Investment Management) | 3,54 % | MarketScreener | 11.10.2026 |
| FIVV Finanzinvestments | 1,44 % | MarketScreener | 11.10.2026 |
| Lupus alpha Asset Management | 1,25 % | MarketScreener | 11.10.2026 |
| rubicon equities | 1,14 % | MarketScreener | 11.10.2026 |
| MFI Asset Management | 0,82 % | MarketScreener | 11.10.2026 |
| Discover Capital | 0,76 % | MarketScreener | 11.10.2026 |
| Warburg Invest | 0,68 % | MarketScreener | 11.10.2026 |

**Lücke/Kontradiktion:** Die Geschäftsberichte selbst weisen keine Aktionärsstruktur mit
Prozentangaben aus (im 80-seitigen GB2025 kein Treffer für "shareholder structure",
"significant shareholding" oder § 33/34 WpHG-Meldungen außer einer allgemeinen Erwähnung von
§ 33 WpHG als Grund für den Scale-SME-Segmentstatus, S.34). Die Tabelle oben ist vollständig
sekundär (Aggregator-Daten, nicht aus einer Primärquelle nachvollziehbar) und die beiden
Free-Float-Zahlen widersprechen sich um vier Prozentpunkte — beide werden hier genannt, keine
gerundet oder stillschweigend bevorzugt. Entscheidend für die Bewertung: **kein Ankeraktionär
mit Kontrollblock.** Der größte benannte Einzelinvestor (Kreissparkasse Biberach) hält 5,10 %;
die Struktur ist breit gestreut, anders als etwa bei CTS Eventim (Gründer-Mehrheit).

**Aktienzahl-Entwicklung** (verifiziert, GB2022 S.24, GB2023 S.22, GB2024 S.28):

| Jahresende | Aktien ausstehend | Ereignis |
|---|---|---|
| 2022 | 5.931.200 | — |
| 2023 | 6.556.320 | Barkapitalerhöhung Mai 2023, explizit zur Finanzierung der Buy-and-Build-Strategie (GB2023 S.16) |
| 2024 | 6.568.820 | Ausübung von Aktienoptionen |
| 2025 | 6.568.820 | unverändert |

**Directors' Dealings (letzte 12 Monate):** April 2025 — Mitglieder von Vorstand und
Aufsichtsrat erwarben Aktien zu Kursen zwischen 11,70 € und 12,50 € in sechsstelligem
Gesamtvolumen (GB2025 S.26: "transactions with a total volume in the six-figure euro range at
prices between EUR 11.70 and EUR 12.50"). Zeitlich fällt dies in die Talsohle des Kursverlaufs
2025 (Schlusskurs 02.01.2025: 19,30 €; Tiefstkurs 30.12.2025: 9,84 €, beide GB2025 S.26). Eine genaue Stückzahl oder ein Datum pro Transaktion nennt der Bericht nicht —
nur die Preisspanne und das Gesamtvolumen in einer Bandbreite ("sechsstellig" = 100.000–999.999 €,
keine engere Eingrenzung möglich).

### 1.3 Beurteilung durch die Kapitalmärkte

| Kennzahl | Wert | Quelle | Abgerufen |
|---|---|---|---|
| Schlusskurs | 26,00 € (Fr. 09.10.2026, letzter Handelstag vor Abruf) | yfinance M7U.DE | 11.10.2026 |
| Marktkapitalisierung | ≈ 170,8 Mio. € (26,00 € × 6.568.820 Aktien) | yfinance / eigene Rechnung | 11.10.2026 |
| 52-Wochen-Spanne | 8,22 € – 28,40 € | yfinance | 11.10.2026 |
| Buchwert/Aktie (Mutteraktionäre, 30.06.2026) | 14,90 € | HJB2026 S.17 (97.922 TEUR / 6.568.820 Aktien), Primärquelle | 11.10.2026 |
| Kurs/Buchwert | 1,74 (26,00 / 14,90) | eigene Rechnung | 11.10.2026 |
| Beta | 1,02 | yfinance | 11.10.2026 |
| Dividendenrendite | 0 % (keine Dividende an Mutteraktionäre GJ2022–GJ2025) | GB2022–GB2025 Kapitalflussrechnungen | 11.10.2026 |

Die 52-Wochen-Spanne (8,22–28,40 €) spiegelt fast eine Verdreifachung innerhalb eines Jahres: der
Kurs brach während der Gewinnwarnungen 2025 bis auf 9,84 € ein (Schlusskurs 30.12.2025) und hat
sich seitdem mehr als verdoppelt — getrieben von der Erholung in H1 2026 (siehe 2.1) und den
angehobenen Analystenzielen (1.4).

### 1.4 Analystenabdeckung und Konsens

**Dünne, widersprüchlich berichtete Abdeckung** — selbst die Zahl der Analysten schwankt je nach
Quelle:

| Quelle | Anzahl Analysten | Konsens-Kursziel | Empfehlung | Abgerufen |
|---|---|---|---|---|
| MarketScreener (Stand Juni 2026, im GB2025 zitiert) | 2 (Montega, NuWays) | Montega 18,00 € / NuWays 21,00 € (März 2026) | Buy / — | GB2025 S.26/S.28 |
| MarketScreener (aktuelle Web-Abfrage) | 2 | Ø 29,00 € (Spanne 28,00–30,00 €) | Buy (Konsens) | 11.10.2026 |
| Web-Recherche (aktualisiert, Teilagent) | 2, mit neueren Einzelzielen Montega 27,00 € / NuWays 25,00 € | — | — | 11.10.2026 |
| Alternative Aggregator-Angabe (frühere Recherche dieses Berichts) | 6 Analysten, davon 5 Kauf / 1 Halten | Ø 21,25 € | Kauf (Mehrheit) | 11.10.2026 |

**Kontradiktion, nicht aufgelöst:** Je nach Aggregator werden 2 oder 6 Analysten gezählt, mit
Konsenszielen zwischen 21,25 € und 29,00 €. Beide Angaben werden hier genannt; **keine Stichprobe
> 3 Analysten gilt methodisch als tragfähiger Konsens**, und selbst die kleinere, im
Geschäftsbericht selbst zitierte Abdeckung (2 Häuser) revidierte ihre Ziele zwischen März und
Juni 2026 um 6–9 € nach oben — ein Hinweis auf hohe Volatilität der Einschätzung, nicht auf
Stabilität. Alle genannten Kursziele liegen über dem aktuellen Kurs von 26,00 €, in Spannen von
etwa −18 % (21,25 €) bis +15 % (30,00 €).

### 1.5 Botschaften und Lücken

Was die Primärquellen **nicht** enthalten und auch über die üblichen Sekundärkanäle nur
lückenhaft zu beschaffen war:
- Eine im Geschäftsbericht selbst ausgewiesene Aktionärsstruktur mit Prozentangaben (1.2) —
  echte Lücke der Primärquelle, kompensiert durch Sekundärdaten mit Kontradiktion.
- Eine belastbare, einheitliche Zählung der Analystenabdeckung (1.4) — Kontradiktion, nicht
  auflösbar ohne Zugriff auf eine kostenpflichtige Terminal-Datenbank.
- Eine eigene ROE-Kennzahl des Unternehmens (siehe 6.0) — Nynomic weist nur die
  Eigenkapitalquote aus (GB2025 S.4/14), nie eine Eigenkapitalrendite.
- Segment-EBIT (nur Segment-Umsatz wird offengelegt, GB2025 S.48) — echte Lücke, keine
  Schätzung vorgenommen.

### 1.6 Wettbewerbsposition

Nynomic positioniert sich selbst als **"The Photonics Group"**: ein Portfolio von zehn operativen
Tochtergesellschaften (von 24 konsolidierten Einheiten insgesamt, GB2025 S.20/S.36) entlang der
gesamten photonischen Wertschöpfungskette — von hochspezialisierten optischen Komponenten bis zu
kompletten Messsystemen. Strategie: **Buy & Build**, d. h. Wachstum primär durch Zukäufe
spezialisierter Nischenanbieter, nicht organisch (GB2025 S.22 f.). Benannte Akquisitionen seit
2017 (GB2025 S.61–63): LayTec AG (2017, Berlin, Messtechnik für Halbleiter/Photovoltaik),
Image Engineering GmbH & Co. KG (2021, 51 %-Anteil, Kerpen, Bildqualitätsmessung), NLIR ApS
(2023, zunächst ≈31 %, später vollständig, Farum/Dänemark, Mid-Infrarot-Hochkonversion), art
photonics GmbH (2023, 100 %, Berlin, Spezialfasern).

**Marktgröße:** Nynomic zitiert in seinen eigenen Berichten **keine** externe Marktgrößen- oder
Wachstumsrate für den Photonik-/Spektroskopie-/Sensormarkt (GB2025, keine Fundstelle für eine
benannte Marktstudie). Drittquellen (Precedence Research, Coherent Market Insights, Future
Market Insights, alle 11.10.2026 abgerufen, als Sekundärdaten markiert) nennen einen globalen
Photonikmarkt von 1,12–1,46 Bio. USD (2026) mit einer CAGR von 4,16–6,69 % je nach
Abgrenzung, und einen optischen Sensormarkt von 38,57 Mrd. USD mit 8,3 % CAGR (2026–2036) — diese
Zahlen stammen nicht vom Unternehmen und sind entsprechend zu gewichten.

**Peer-Vergleich** (einziger sauber vergleichbarer börsennotierter Peer; weitere genannte
Nischenanbieter wie Micro-Epsilon, Polytec, Gigahertz Optik sind privat oder ohne aktuelle
Einzelabschlusszahlen):

| Kennzahl, GJ2025 | Nynomic AG | Jenoptik AG |
|---|---|---|
| Umsatz | 92,6 Mio. € | 1.046 Mio. € (Jenoptik-Pressemitteilung 25.03.2026) |
| EBIT-Marge | 2,1 % | ≈ 10,9 % (berechnet: 114,5 / 1.046 Mio. €) |
| EBITDA-Marge | 8,6 % | 18,4 % |
| Marge-Entwicklung ggü. GJ2024 | 7,2 % → 2,1 % (−5,1 Pp) | 13,1 % → 10,9 % (−2,2 Pp) |

Beide Unternehmen litten 2024/2025 unter einem Margenrückgang in zyklischen, capex-getriebenen
Endmärkten (Halbleiter, Photovoltaik). Jenoptik ist elffach größer und bleibt trotz Rückgang
deutlich profitabler — Nynomics Margenerosion war relativ und absolut stärker. Eine echte
Multiple-Vergleichstabelle (KGV, EV/EBITDA) war mit den verfügbaren Primär- und frei
zugänglichen Sekundärquellen nicht belastbar herzustellen (Lücke, 1.5).

---

## 2. Auswertung des Jahresabschlusses (GJ2022–GJ2025, IFRS-Konzernabschluss)

### 2.1 Umsatz

| Mio. € | GJ2022 | GJ2023 | GJ2024 | GJ2025 | H1 2025 | H1 2026 |
|---|---|---|---|---|---|---|
| Konzernumsatz | 116,793 | 117,985 | 102,415 | 92,569 | 42,142 | 47,749 |
| Veränderung ggü. Vorjahr | +11 % | +1 % | −13 % | −10 % | — | +13 % |

*Quelle: GB2022 S.28, GB2023 S.26, GB2024 S.32, GB2025 S.32 (jeweils eigener Konzernabschluss,
nicht nur Vorjahresvergleichsspalte); H1-Werte HJB2026 S.18.*

Umsatz nach Segment, GJ2025 vs. GJ2024 (GB2025 S.48 — **nur Umsatz, kein Segment-EBIT
ausgewiesen**, siehe Lücke 1.5):

| Segment | GJ2025 | GJ2024 | Veränderung |
|---|---|---|---|
| Clean Tech | 59,852 | 68,657 | −13 % |
| Life Science | 11,989 | 13,130 | −9 % |
| Green Tech | 20,728 | 20,628 | 0 % |

H1 2026 nach Segment (HJB2026 S.4, nur Verifikation über Teilagenten-Digest, nicht erneut
Seite-für-Seite nachgeprüft — Wert plausibel, da konsistent mit Gesamtsumme 47,749): Clean Tech
31,165 (+15 %), Life Science 6,230 (+29 %), Green Tech 10,354 (+2 %). Alle drei Segmente wachsen
in H1 2026 wieder, am stärksten Life Science.

### 2.2 Auftragsbestand (statt Auftragseingang — Nynomic weist nur den Bestand aus)

Auftragsbestand zum 30.06.2026: **63,5 Mio. €**, ein Anstieg von **+46 %** gegenüber dem
Vorjahreswert (HJB2026 S.5). Management wertet dies explizit als Frühindikator für die
Umsatzerholung in H2 2026 und 2027 (HJB2026 S.5/S.24).

### 2.3 / 2.4 EBITDA-Marge, operatives Ergebnis

| Mio. € | GJ2022 | GJ2023 | GJ2024 | GJ2025 | H1 2025 | H1 2026 |
|---|---|---|---|---|---|---|
| EBITDA | 19,178 | 20,215 | 12,505 | 7,937 | 0,795 | 4,835 |
| EBIT | 15,083 | 15,432 | 7,386 | 1,961 | −2,042 | 1,804 |
| EBIT-Marge | 12,9 % | 13,1 % | 7,2 % | 2,1 % | −4,8 % | 3,8 % |

*Quelle: dieselben Seiten wie 2.1.*

**Der Einbruch ist real und massiv:** die EBIT-Marge fiel von ihrem Höchststand 13,1 % (GJ2023)
auf 2,1 % (GJ2025) — ein Rückgang von 11 Prozentpunkten in zwei Jahren, mit einem
EBIT-Tiefpunkt (negativ) im ersten Halbjahr 2025. H1 2026 zeigt eine Erholung auf 3,8 %, aber das
liegt immer noch unter der Mitte der FY2026-Prognosespanne (6–8 %, siehe 2.8) — die zweite
Jahreshälfte müsste überproportional stark ausfallen, um den Zielkorridor zu treffen.

### 2.5 Nettogewinn

| Mio. € | GJ2022 | GJ2023 | GJ2024 | GJ2025 | H1 2025 | H1 2026 |
|---|---|---|---|---|---|---|
| Konzernergebnis (gesamt) | 9,949 | 10,532 | −0,270 | −0,348 | −3,279 | 1,003 |
| davon Mutteraktionäre | 7,454 | 9,305 | −0,643 | −0,704 | −3,105 | 0,924 |
| EPS (Mutteraktionäre) | 1,26 € | 1,47 € | −0,10 € | −0,11 € | −0,47 € | 0,14 € |

GJ2024 enthält einen Sondereffekt: −4,0 Mio. € aus aufgegebenen Geschäftsbereichen (GB2025 S.32,
Vorjahresspalte: "Consolidated profit after tax from discontinued operations: −4,000"). Ohne
diesen Sondereffekt wäre GJ2024 für die Mutteraktionäre positiv ausgefallen. Identifiziert (GB2025
S.43, Anhangsangabe 6.2/6.3 nach IFRS 5.33): **Ausstieg aus dem finnischen Standort von Spectral
Engines** — die operativen Aktivitäten (NIRON, TactiScan, Purpl) wurden 2024 nach Wedel und
Steinbach verlagert, der finnische Standort geschlossen, und der darauf allokierte Goodwill von
4,0 Mio. € vollständig abgeschrieben. Die Transaktion wurde erst nachträglich (nicht bereits im
GB2024 selbst) als "discontinued operation" im Sinne von IFRS 5 eingestuft, daher die
Darstellung als Vorjahresanpassung im GB2025. In GJ2025 gab es laut Unternehmen keine
aufgegebenen Geschäftsbereiche (GB2025 S.43: "There were no discontinued operations in the 2025
financial year").

### 2.6 Dividende

**Keine Dividende an die Aktionäre der Nynomic AG in GJ2022–GJ2025.** Dividenden an
nicht beherrschende Anteile (Minderheitsgesellschafter von Tochtergesellschaften) wurden
gezahlt: 821 TEUR (GJ2024), 708 TEUR (GJ2025) (GB2024 S.60, GB2025 S.60). Ausschüttbarer
Bilanzgewinn der Nynomic AG (Einzelabschluss) zum GJ2025: 37,757 Mio. € (GB2025 S.52/59) — das
Unternehmen priorisiert explizit Reinvestition und Entschuldung statt Ausschüttung.

### 2.7 Aktienkursverlauf

| Datum | Kurs | Ereignis |
|---|---|---|
| 02.01.2025 | 19,30 € | Jahresbeginn (GB2025 S.26) |
| ~April 2025 | 11,70–12,50 € | Insiderkäufe (1.2) |
| 30.12.2025 | 9,84 € | Jahrestiefstkurs (GB2025 S.26) |
| 09.10.2026 | 26,00 € | Letzter verifizierter Schlusskurs (yfinance) |

Der Kurs hat sich von seinem Tiefpunkt (9,84 €) bis heute (26,00 €) um +164 % erholt — mehr als
verdreifacht gegenüber dem Krisentief, parallel zur operativen Erholung in H1 2026 (2.3/2.4).

### Kennzahlenüberblick

| Mio. € | GJ2022 | GJ2023 | GJ2024 | GJ2025 |
|---|---|---|---|---|
| Bilanzsumme | 138,445 | 151,512 | 139,526 | 133,836 |
| Eigenkapital (gesamt, inkl. NCI) | 77,027 | 103,934 | 102,977 | 100,668 |
| Eigenkapitalquote | 55,6 % | 68,6 % | 73,8 % | 75,2 % |
| Operativer Cashflow | 9,677 | 3,166 | 7,797 | 7,519 |
| Capex (immateriell + sachanlagen) | −2,572 | −8,715 | −3,879 | −2,986 |
| Free Cashflow (OCF − Capex) | 7,105 | −5,549 | 3,918 | 4,533 |
| Finanzierungs-Cashflow | −7,533 | +5,828 | −8,522 | −6,094 |
| Abschreibungen | 4,095 | 4,783 | 5,119 | 5,976 |

*Quelle: Bilanz GB2022 S.26 f., GB2023 S.24 f., GB2024 S.30 f., GB2025 S.30 f.; Kapitalflussrechnung
GB2022 S.56, GB2023 S.56, GB2024 S.60, GB2025 S.60.*

**Die Eigenkapitalquote steigt, während die absolute Profitabilität einbricht** — scheinbar
widersprüchlich, aber erklärbar: der Anstieg 2022→2023 (55,6 %→68,6 %) stammt überwiegend aus
der Barkapitalerhöhung Mai 2023 (+18,979 Mio. € Eigenkapitalzufluss, GB2024 S.60, explizit zur
Finanzierung von Zukäufen, GB2023 S.16), nicht aus einbehaltenen Gewinnen. GJ2023 ist damit das
Jahr mit dem höchsten EBIT der Reihe, aber dem schwächsten Free Cashflow (−5,549 Mio. €) — die
hohe Investitionstätigkeit (Capex 8,715 Mio. € plus zusätzlich 5,072 Mio. € in "Finanzanlagen",
akquisitionsnah, GB2024 S.60) fiel in dasselbe Jahr wie der operative Höhepunkt. **Diese
Asymmetrie ist wichtig für Abschnitt 6:** Ertragsstärke und Cashflow-Stärke fallen bei Nynomic
nicht zusammen.

### 2.8 Prognose vs. Ist

| Geschäftsjahr | Prognose (Zeitpunkt) | Tatsächliches Ergebnis |
|---|---|---|
| GJ2025 (ursprünglich) | Umsatz 105–110 Mio. €; EBIT 8,5–10,0 Mio. € (Jahresbeginn, GB2025 S.10/S.73-Kontext) | — |
| GJ2025 (angepasst, November 2025) | Umsatz 93–96 Mio. € (GB2025 S.63) | Umsatz 92,6 Mio. €; EBIT 1,961 Mio. € — leicht unter der angepassten Spanne |
| GJ2026 (Prognose) | Umsatz 100–105 Mio. €; EBIT-Marge 6–8 % (GB2025 S.73; bestätigt HJB2026 S.6/S.27) | H1 2026: 47,7 Mio. € Umsatz (annualisiert ≈ 95 Mio. €, am unteren Rand der Spanne), EBIT-Marge 3,8 % (unter der Zielspanne, muss in H2 deutlich zulegen) |

Die ursprüngliche GJ2025-Prognose wurde **im Jahresverlauf nach unten korrigiert** (von
105–110 Mio. € auf 93–96 Mio. € Umsatz) und dann nochmals knapp unterschritten. Begründung laut
Unternehmen: geopolitische Unsicherheiten, schwache Industrienachfrage, Lagerabbau bei Kunden,
US-Zölle, besondere Schwäche im Clean-Tech-Segment (GB2025 S.63 f.). Das **NyFIT2025**-Effizienzprogramm
(Fusion von Spectral Engines GmbH und APOS GmbH in die m-u-t GmbH, einmalige Kosten ≈1,5 Mio. €
in GJ2025, Zielersparnis 5,0–6,0 Mio. € p. a. ab GJ2026, GB2025 S.26) ist die vom Management
benannte Gegenmaßnahme.

---

## 3. Strategische Maßnahmen

- **Buy & Build fortgesetzt trotz Krise**: Die letzte große Akquisition (art photonics, 2023)
  fiel ins selbe Jahr wie die Kapitalerhöhung — das Unternehmen hat seine Wachstumsstrategie
  nicht pausiert, als die operative Marge zu bröckeln begann.
- **NyFIT2025**: Personalbezogene Maßnahmen, Straffung der Konzernstruktur (Fusion zweier
  Tochtergesellschaften in die m-u-t GmbH), Zielersparnis 5–6 Mio. € p. a. ab 2026 (GB2025
  S.26). Durchschnittliche Mitarbeiterzahl sank von 583 (GJ2024) auf 538 (GJ2025) (GB2025 S.48).
- **Auftragsbestand als Frühindikator**: +46 % auf 63,5 Mio. € zum 30.06.2026 wird vom
  Management explizit als Beleg für eine Nachfrageerholung angeführt (HJB2026 S.5).
- **Keine neue Großakquisition seit 2023** in den verfügbaren Quellen erkennbar — nach der
  kapitalintensiven 2023er-Phase scheint der Fokus auf interne Konsolidierung (NyFIT2025) statt
  auf weitere Zukäufe zu liegen. Das ist eine Beobachtung aus dem Fehlen neuer Meldungen, nicht
  eine vom Unternehmen verkündete Strategiewende — als solche gekennzeichnet.

---

## 4. Beurteilung des Kursverlaufs

Der Kurs folgt der operativen Entwicklung mit einer Verzögerung, aber durchaus plausibel: Der
Einbruch auf 9,84 € (30.12.2025) fällt mit der zweiten, verschärften Gewinnwarnung und dem
H1-2025-EBIT-Verlust zusammen; die Erholung auf 26,00 € (09.10.2026) läuft parallel zur
H1-2026-Rückkehr in die Gewinnzone und zum kräftigen Auftragsbestandswachstum. Die
Insiderkäufe im April 2025 (11,70–12,50 €, Abschnitt 1.2) kamen **vor** dem Jahrestief (9,84 €
im Dezember) — Vorstand und Aufsichtsrat lagen mit ihrem Kauf-Timing also nicht am exakten
Boden, aber deutlich unter dem heutigen Kurs, was für eine ex post richtige Einschätzung der
mittelfristigen Erholung spricht, auch wenn der Zeitpunkt nicht optimal war. (Quelle für
Jahresanfangs- und Jahrestiefstkurs: GB2025 S.26.)

**Kritischer Punkt:** Der Kurs hat sich bereits um +164 % von seinem Tief erholt, während die
operative Kennzahl, auf die sich diese Erholung stützt (EBIT-Marge H1 2026: 3,8 %), **noch unter
der Mitte der eigenen Unternehmensprognose für das Gesamtjahr (6–8 %)** liegt. Ein erheblicher
Teil der erwarteten Erholung ist damit bereits eingepreist, bevor sie in den Zahlen bestätigt
ist.

---

## 5. Entfällt (n = 0)

Keine Handlungsoptionen zu vergleichen — Nynomic ist eine bestehende Börsennotierung, die
Entscheidung ist binär (Position eingehen oder nicht).

---

## 6. Grobe Flow-Analyse (no-options variant, nur 6.3)

### 6.0 Annahmen

**Hürde.** Nynomic weist in keinem der vier untersuchten Geschäftsberichte eine eigene
Eigenkapitalrendite (ROE) aus — nur die Eigenkapitalquote (GB2025 S.4/14). Es gibt daher keine
"eigene Angabe" zum Zitieren (anders als bei CTS Eventim, wo das Unternehmen selbst eine ROE
veröffentlicht). Diese Analyse berechnet die ROE pro Jahr selbst:

| GJ | Jahresüberschuss Mutteraktionäre (TEUR) | Eigenkapital Jahresende (TEUR) | ROE |
|---|---|---|---|
| 2022 | 7.454 | 77.027 | 9,67 % |
| 2023 | 9.305 | 103.934 | 8,95 % |
| 2024 | −643 | 102.977 | −0,63 % |
| 2025 | −704 | 100.668 | −0,70 % |

*Quelle: siehe Kennzahlenüberblick (2.7) für die Einzelbelege.*

**Annahme:** Da die beiden jüngsten Jahre negative ROE zeigen (Krisenjahre) und ein negativer
Wert keine sinnvolle Mindestverzinsung darstellen kann, wird als Hürde der **4-Jahres-Durchschnitt
über den vollständigen verfügbaren Zyklus** verwendet: (9,67 % + 8,95 % − 0,63 % − 0,70 %) / 4 =
**4,32 %**. Das ist ausdrücklich eine schwache Hürde — niedriger als die ROE-basierten Hürden
anderer Berichte dieser Serie (Alamos 14,6 %, Yara 11,3 %, Accenture 26,7 %, CTS Eventim 22,3 %)
und sogar niedriger als die marktindex-basierten Hürden (9,08 % MSCI World long, verwendet bei
GEA/Daimler/Neste). Der Grund ist nicht eine methodische Lockerung, sondern eine Tatsache über
Nynomic selbst: **die Eigenkapitalrendite des Unternehmens war selbst im besten gemessenen Jahr
(GJ2022: 9,67 %) nicht höher als die verbreiteten marktindex-basierten Hürden anderer Berichte**,
und das Unternehmen finanziert sich mit sehr hoher Eigenkapitalquote (55,6–75,2 %) — viel
Eigenkapital, das auf eine bescheidene Ertragsbasis verteilt wird, drückt die Eigenkapitalrendite
strukturell. Eine härtere, realitätsfernere Hürde zu wählen wäre keine Vorsicht, sondern eine
Verzerrung in die andere Richtung.

**Terminal Value** = Buchwert der Mutteraktionäre pro Aktie zum 30.06.2026 (jüngster verfügbarer
Stichtag): 97.922 TEUR / 6.568.820 Aktien = **14,90 €/Aktie** (HJB2026 S.17, Primärquelle; die
yfinance-Kennzahl `bookValue` von 14,907 bestätigt diesen Wert sekundär). Gehalten konstant über
den gesamten Horizont (methodischer Standard dieser Serie), nicht wachsend.

**Einstiegskurs**: 26,00 € (letzter Schlusskurs, 09.10.2026).

**Jährlicher Cashflow (FCF/Aktie)**: Operativer Cashflow minus Capex (immateriell + Sachanlagen,
ohne Finanzanlagen-Investitionen, die als Akquisitionen separat behandelt werden), geteilt durch
die durchschnittliche Aktienzahl des jeweiligen Jahres:

| GJ | FCF (TEUR) | Ø Aktien | FCF/Aktie |
|---|---|---|---|
| 2022 | 7.105 | 5.904.323 | **1,203 €** (bestes Jahr → optimistisch) |
| 2023 | −5.549 | 6.313.975 | **−0,879 €** (schwächstes Jahr → pessimistisch) |
| 2024 | 3.918 | 6.558.164 | 0,598 € |
| 2025 | 4.533 | 6.568.820 | **0,690 €** (letztes volles GJ → Basis) |

**Annahme (wichtig):** GJ2023 ist zugleich das Jahr mit dem höchsten EBIT der Reihe (13,1 %
Marge) und dem negativsten FCF/Aktie — nicht wegen operativer Schwäche, sondern wegen der
akquisitionsnahen Investitionstätigkeit in genau diesem Jahr (2.7-Box). Es als "pessimistisches"
Szenario zu verwenden ist methodisch korrekt (schwächster tatsächlich erzielter Wert der
Reihe, keine Schätzung), sollte aber nicht als "Nynomic im schlechtesten operativen Jahr"
gelesen werden — es ist das Jahr mit der höchsten Investitionsintensität. **GJ2025**, das
tatsächliche Krisenjahr nach EBIT, hat paradoxerweise den zweithöchsten FCF/Aktie der Reihe
(0,690 €) — working-capital-Entlastung und reduzierte Investitionstätigkeit in der Krise.

Horizont: 2026–2041 (15 Jahre), mit Zwischenpunkten 2031 (5 J.) und 2036 (10 J.), analog zur
Methodik anderer Berichte dieser Serie.

### 6.3 Was verdient ein Eigentümer, der heute einsteigt?

| Szenario | FCF/Aktie p. a. | IRR, 5 J. (2031) | IRR, 10 J. (2036) | IRR, 15 J. (2041) |
|---|---|---|---|---|
| Pessimistisch (GJ2023) | −0,879 € | −15,4 % | −10,2 % | −8,4 % |
| Basis (GJ2025) | 0,690 € | −6,7 % | −1,7 % | −0,0 % |
| Optimistisch (GJ2022) | 1,203 € | −3,8 % | +1,0 % | +2,6 % |

*Erzeugt von `scripts/cashflow_irr.py` (Projektwerkzeug), Konfiguration in `data/analysis.json`,
Rohtabellen in `data/irr_sensitivity.md`, `data/comparison.md`.*

**Keines der drei Szenarien erreicht die Hürde von 4,32 % — auch nicht über 15 Jahre, und auch
nicht im optimistischen Fall (der das beste tatsächlich erzielte Jahr der letzten vier
verwendet).** Das optimistische Szenario kommt mit +2,6 % über 15 Jahre am nächsten heran, bleibt
aber 1,7 Prozentpunkte darunter. Erinnerung: Diese Hürde ist bereits die am Besten begründete, am
niedrigsten mögliche Hürde dieser Berichtsserie (6.0) — eine härtere, realistischere Hürde würde
das Bild nur verschärfen.

### Einstiegsschwelle

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (2031) | 7,3 € | 15,8 € | 18,6 € |
| 10 Jahre (2036) | 1,9 € | 16,0 € | 20,6 € |
| 15 Jahre (2041) | unter Hürde (kein Einstiegskurs erfüllt sie) | 16,1 € | 22,2 € |

*Quelle: `data/break_even.md`.*

Selbst im optimistischen Szenario und über den längsten geprüften Horizont (15 Jahre) liegt die
Einstiegsschwelle bei 22,2 € — **14,6 % unter dem aktuellen Kurs von 26,00 €**. Im realistischeren
Basis-Szenario liegt sie bei 16,1 € (−38 % ggü. aktuellem Kurs).

**Gegenprobe — welchen Terminal Value setzt der heutige Preis schon voraus?**

| Horizont | Pessimistisch | Basis | Optimistisch | Heutiges Kurs/Buchwert-Multiple |
|---|---|---|---|---|
| 5 Jahre (2031) | 38,00 € (2,55× Buchwert) | 27,51 € (1,85×) | 24,08 € (1,62×) | 1,74× (heute) |
| 10 Jahre (2036) | 51,74 € (3,47×) | 30,23 € (2,03×) | 23,19 € (1,56×) | — |
| 15 Jahre (2041) | 68,72 € (4,61×) | 33,58 € (2,25×) | 22,09 € (1,48×) | — |

Selbst im optimistischen Fall setzt der heutige Preis voraus, dass Nynomic in 15 Jahren
mindestens zum 1,48-fachen seines dann aktuellen Buchwerts gehandelt wird, nur um die (schwache)
Hürde zu erreichen — leicht unter dem heutigen Multiple (1,74×), aber nicht deutlich.

**Zweite Gegenprobe — welches Wachstum des FCF/Aktie setzt der heutige Preis voraus?**

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (2031) | nicht erreichbar (kein Wachstum rechtfertigt den Preis) | 57,4 %/Jahr | 35,1 %/Jahr |
| 10 Jahre (2036) | nicht erreichbar | 20,1 %/Jahr | 9,9 %/Jahr |
| 15 Jahre (2041) | nicht erreichbar | 11,3 %/Jahr | 4,4 %/Jahr |

Selbst im optimistischen Szenario und über 15 Jahre müsste der FCF/Aktie **4,4 % pro Jahr,
fünfzehn Jahre ununterbrochen**, wachsen, um die (ohnehin schwache) Hürde zu erfüllen. Nynomics
eigene FCF/Aktie-Historie (1,203 € → −0,879 € → 0,598 € → 0,690 €, 2022–2025) zeigt **keinen
erkennbaren Wachstumstrend** — im Gegenteil, eine hohe Schwankungsbreite ohne klare Richtung. Im
Basis-Szenario wären 11,3 % p. a. über 15 Jahre nötig; im pessimistischen Szenario rechtfertigt
*kein* Wachstum den heutigen Preis.

---

## 7. Verdict — Investitionspotenzial

### 7.1 Urteil

**Nicht kaufen** zum aktuellen Kurs von 26,00 €. Einstiegsschwelle (Basis-Szenario, 15 Jahre):
**16,10 €** (−38 % ggü. aktuellem Kurs). Selbst im optimistischen Szenario liegt die Schwelle bei
22,20 € (−15 %).

### 7.2 Begründung aus den Befunden

1. Keines der drei FCF/Aktie-Szenarien (gestützt auf die vier einzigen verfügbaren
   Geschäftsjahre, keine Schätzung) erreicht die selbst berechnete Hürde von 4,32 % — über
   keinen der drei geprüften Horizonte (5/10/15 Jahre).
2. Diese Hürde ist bereits die am wenigsten strenge in der gesamten Berichtsserie (6.0) —
   begründet durch Nynomics eigene, durchgängig niedrige historische Eigenkapitalrendite
   (9,67 % im besten Jahr, negativ in den beiden jüngsten), nicht durch eine methodische
   Lockerung.
3. Der Kurs hat sich bereits um +164 % von seinem Jahrestief erholt (1.3/4), während die
   zugrunde liegende operative Kennzahl (EBIT-Marge H1 2026: 3,8 %) noch unter der Mitte der
   eigenen Unternehmensprognose (6–8 %) liegt — ein erheblicher Teil der erwarteten Erholung ist
   bereits eingepreist, bevor sie bestätigt ist.
4. Die Gegenprobe über das implizierte Wachstum (6.3) zeigt: selbst im optimistischen Szenario
   müsste der freie Cashflow pro Aktie 15 Jahre lang mit 4,4 % p. a. wachsen, ohne dass
   Nynomics eigene Historie der letzten vier Jahre einen solchen Trend stützt.
5. Governance und Bilanz sprechen nicht gegen das Unternehmen (stabile Führung seit 2015, hohe
   und steigende Eigenkapitalquote, Netto-Cash-Position, Insiderkäufe 2025) — das Urteil beruht
   nicht auf einem Zweifel an der Unternehmensführung, sondern allein auf dem Preis relativ zur
   erzielbaren Verzinsung.

### 7.3 Einstiegsschwelle und erforderliches Wachstum

Siehe Tabellen in 6.3 ("Einstiegsschwelle" und beide Gegenproben). Zusammengefasst: Basis-Fall
16,10 € / 15 Jahre; optimistischer Fall 22,20 € / 15 Jahre. Eine Investition zu 26,00 € setzt
entweder einen bis 2041 anhaltenden Terminal-Value-Multiple von mindestens 1,48× Buchwert
(optimistisch) oder ein FCF/Aktie-Wachstum von 4,4 % p. a. über 15 Jahre voraus (optimistisch) —
beides liegt am oberen Rand oder außerhalb dessen, was die vier verfügbaren Geschäftsjahre an
Evidenz liefern.

### 7.4 Trigger

Was das Urteil ändern würde:
- **Preis fällt auf ≤ 22,20 €** (optimistisches Szenario, 15-Jahres-Schwelle) oder ≤ 16,10 €
  (Basis-Szenario) — ab dort wird die Position auch bei konservativer Annahme rechnerisch
  attraktiv.
- **H2 2026 bestätigt die obere Hälfte der EBIT-Margen-Prognose (nahe 8 % statt 6 %)** und damit
  einen höheren, nachhaltigeren FCF/Aktie-Lauf als die Basis-Annahme (0,690 €) — eine zweite
  Jahreshälfte, die die Prognosemitte deutlich übertrifft, würde die Basis-Annahme nach oben
  verschieben.
- **Auftragsbestandswachstum (+46 % zum 30.06.2026) setzt sich fort** und übersetzt sich in
  GJ2027-Zahlen, die über der aktuellen GJ2026-Prognose liegen.

### 7.5 Was dieses Urteil nicht wissen kann

- **Die Hürde selbst ist eine Modellentscheidung** (4-Jahres-Durchschnitts-ROE über einen
  Zyklus, der zufällig zwei gute und zwei schlechte Jahre enthält — bei einer anderen
  Jahresauswahl, z. B. nur den beiden Vor-Krisen-Jahren, läge sie bei 9,31 % und das Bild wäre
  noch schwächer; bei einer Gewichtung Richtung Erholung könnte sie niedriger ausfallen).
- **Der Analystenkonsens widerspricht dem Urteil in beiden zitierten Fassungen** (1.4):
  Kursziele von 21,25 € bis 30,00 € liegen überwiegend über dem aktuellen Kurs von 26,00 €. Die
  dünne Abdeckung (2–6 Häuser, je nach Zählung) relativiert dieses Signal, aber es widerspricht
  dem hier getroffenen Urteil in dieselbe Richtung wie bei den meisten anderen Berichten dieser
  Serie (Suivi_Verdicts.md: 8 von 9 "nicht kaufen"-Urteilen standen bislang im Gegensatz zum
  Analystenkonsens).
- **Das Modell hält den FCF/Aktie konstant** (keine Prognose, methodischer Standard dieser
  Serie) — wenn das NyFIT2025-Programm die versprochenen 5–6 Mio. € p. a. Einsparungen ab GJ2026
  tatsächlich vollständig realisiert UND der Umsatz die Prognosemitte (102,5 Mio. €) erreicht,
  wäre ein FCF/Aktie deutlich über dem hier verwendeten Basis-Wert plausibel — dies wird hier
  bewusst nicht vorweggenommen, sondern als Trigger (7.4) behandelt.
- **Directors' Dealings zeigen Vertrauen des Managements**, allerdings zu Kursen (11,70–12,50 €)
  weit unter dem heutigen — ihr Kaufsignal von 2025 sagt nichts über die Attraktivität des
  heutigen Kurses von 26,00 €.

---

## Lücken (verbleibend)

- Aktionärsstruktur mit Prozentangaben: keine Primärquelle, nur widersprüchliche
  Sekundärquellen (1.2).
- Einheitliche Analystenabdeckung/-zahl: Kontradiktion zwischen Aggregatoren nicht auflösbar
  (1.4).
- Segment-EBIT: nur Segment-Umsatz offengelegt (1.6, 2.1).
- Individuelle Vorstandsvergütung: Befreiung nach § 314 Abs. 3 HGB in Anspruch genommen (1.1).
- Belastbare KGV/EV-EBITDA-Multiple-Vergleichstabelle gegen Peers (1.6) — nur Jenoptik war mit
  verfügbaren Quellen sauber vergleichbar.
- Photecture Inc. (US-Tochter, gegründet 2023) wurde laut GB2025 S.63 "during the 2025
  financial year" eingestellt, aber nicht als "discontinued operation" nach IFRS 5 ausgewiesen
  und auch nicht separat quantifiziert — Größenordnung des Effekts auf GJ2025 unbekannt.
