# Accenture plc (NYSE: ACN) — Investitionsanalyse

**Frage, der dieser Bericht dient:** Soll man ACN am 06.10.2026 zum Schlusskurs des Vortags
(195,05 USD, Schluss 05.10.2026) kaufen, halten oder meiden — und falls kaufen, bis zu
welchem Preis?

**n = 0.** Es gibt keine zu vergleichenden Handlungsoptionen; die Entscheidung ist bereits
binär (Position eingehen / nicht eingehen, bzw. für einen Altaktionär: halten / verkaufen).
Abschnitt 5 entfällt daher; Abschnitt 6 verwendet die in `report-template.md` beschriebene
"no-options variant" mit zwei Fragen statt eines Optionsvergleichs.

**Methodischer Hinweis zu Seitenangaben.** Die SEC-10-K-Einreichungen von Accenture sind
Inline-XBRL-HTML-Dokumente. Sie tragen entgegen der ersten Vermutung tatsächlich gedruckte
Seitenzahlen — jede gedruckte Seite beginnt mit einer wiederholten Kopfzeile
"ACCENTURE 20XX FORM 10-K | <Item> | <Seite>" (Haupttext: 1, 2, 3 …; Abschluss ab Item 8:
F-1, F-2 …). Zitate unten verwenden diese gedruckte Seite. Wo eine Zahl aus einer
8-K-Ergebnismitteilung (Pressemitteilung, als Exhibit bei der SEC eingereicht) stammt, die
keine solche Kopfzeile trägt, wird stattdessen der Abschnittstitel der Mitteilung zitiert.

**Primärquellen** (lokal gespeichert, `quellen/`):
10-K FY2023 (GJ bis 31.08.2023, eingereicht 12.10.2023), 10-K FY2024 (bis 31.08.2024, eingereicht
10.10.2024), 10-K FY2025 (bis 31.08.2025, eingereicht 10.10.2025), DEF 14A 2025 (eingereicht
12.12.2025), 8-K-Ergebnismitteilung Q3 FY2026 (18.06.2026) und Q4/Gesamtjahr FY2026
(01.10.2026, die jüngste verfügbare Zahl — das FY2026-10-K selbst liegt bei Abgabe dieses
Berichts noch nicht vor). Marktdaten: yfinance, abgerufen 06.10.2026.

---

## 1. Beschreibung des Unternehmens

### 1.1 Handelnde Personen

Accenture plc ist in Irland registriert (1 Grand Canal Square, Dublin 2); Vorstandsvorsitzende
und CEO ist Julie Sweet (zugleich Chair des Board). CFO ist Angie Park (seit März 2025;
Vorgängerin KC McClure). Der Board besteht aus 10 Mitgliedern, davon 9 unabhängig (90 %);
unabhängiger Lead Director ist Arun Sarin (ehem. CEO Vodafone, seit Februar 2025), vier der
zehn Direktoren wurden in den letzten drei Jahren neu bestellt. Alle vier Ausschüsse (Audit,
Compensation, Finance, Nominating/Governance) sind zu 100 % unabhängig besetzt.
*Quelle: DEF 14A 2025, Abschnitt "Director Nominee Biographies"; 10-K FY2025, Item 1.*

Governance-Signal: Drei Direktorinnen (McKinstry, Nason, Travis) waren FY2025 bei
Organisationen beschäftigt, die Geschäfte mit Accenture machen, jeweils unter der
Meldeschwelle (1 % des konsolidierten Umsatzes der jeweiligen Partei). Kein
Interessenkonflikt mit einem Ankeraktionär, da keiner existiert (siehe 1.2).
*Quelle: DEF 14A 2025, "Director Independence".*

### 1.2 Eigentümerstruktur

| Aktionär | Class-A-Aktien | Anteil | Quelle |
|---|---|---|---|
| The Vanguard Group | 63.802.647 | 10,4 % | Schedule 13G/A, eingereicht 07.04.2025 |
| BlackRock, Inc. | 47.931.575 | 7,8 % | Schedule 13G/A, eingereicht 01.02.2024 |
| Accenture plc (eigene Aktien) | 44.997.383 | 6,8 % | DEF 14A 2025 |
| **Freefloat (Rest)** | **~93,2 %** | | abgeleitet |

Kein Ankeraktionär; Streubesitz ~93 %. Die Zeilen summieren sich nicht exakt auf 100 %, weil
Vanguard/BlackRock auf Basis der zuletzt eingereichten 13G/A (April 2025 bzw. Februar 2024)
berichtet werden, nicht auf denselben Stichtag wie der Aktienbestand.

**Insidergeschäfte, letzte 12 Monate (06.10.2025–06.10.2026).** **Lücke:** Die SEC-Form-4-Liste
deckt nur den Zeitraum 17.02.2026–08.09.2026 vollständig ab; die ersten gut vier Monate des
Fensters (Oktober 2025–Mitte Februar 2026) konnten mit dem verfügbaren Werkzeugsatz nicht
vollständig nachgezogen werden (Paginierung über 100 Einträge hinaus nötig). Für das
abgedeckte Teilfenster: 14 Form-4-Einreichungen, ausschließlich planmäßige RSU-Vesting- und
10b5-1-Verkäufe; **kein einziger offener Marktkauf (Transaktionscode "P") durch einen Director
oder Officer** in diesem Teilfenster. Das ist bei einem Kurs, der im selben Zeitraum rund
70 % unter sein Zwischenhoch gefallen war, ein neutral-bis-leicht-negatives Signal: Insider
haben den Einbruch nicht mit eigenem Geld gekauft — sie haben ihn auch nicht verstärkt
verkauft. *Quelle: SEC EDGAR Form-4-Index, abgerufen 06.10.2026; Lücke benannt, nicht
geschätzt.*

### 1.3 Beurteilung durch die Kapitalmärkte

Kurs 05.10.2026 (Schluss): **195,05 USD**. Marktkapitalisierung ~119,4 Mrd. USD (611,9 Mio.
Aktien, yfinance). Die Aktie fiel von einem Hoch von 386,45 USD (05.02.2026 intraday) auf ein
Tief von 118,15 USD (52-Wochen-Tief, yfinance) — ein Rückgang von **~69 %** — und hat sich seit
dem Tief auf 195,05 USD teilerholt (+65 % vom Tief, aber immer noch **−49,5 %** unter dem
Zwischenhoch). 52-Wochen-Hoch 291,09 USD. Details und Ursache in Abschnitt 4.
*Quelle: yfinance, Tageskurse, abgerufen 06.10.2026.*

### 1.4 Analystenabdeckung und Konsens

Das Unternehmen selbst nennt in seinen Berichten keine Ratings oder Kursziele (üblich, kein
Lückenbefund). Sekundärquelle: mittleres 12-Monats-Kursziel **223,03 USD** (Konsens, 24
Analysten, Quelle Yahoo Finance, abgerufen 06.10.2026); jüngstes Einzelrating Goldman Sachs
"Buy" (02.10.2026). **Lücke:** die vollständige Verteilung Buy/Hold/Sell wurde nicht
extrahiert — nur Mittelwert und ein Einzelrating liegen vor. Was vorliegt, reicht aber für
7.2: der Konsens-Mittelwert liegt 14,3 % über dem aktuellen Kurs, also klar auf der
"Kaufen"-Seite der Markteinschätzung.

### 1.5 Botschaften und Lücken

Nicht verfügbar und nicht beschaffbar innerhalb dieser Recherche: vollständige
Analysten-Ratingverteilung (nur Mittelwert vorliegend), IT-Dienstleistungs-Marktgröße
(Gartner/IDC/Statista kostenpflichtig), Kundenkonzentration in Prozent (10-K nennt nur
"eine kleine Zahl von Kunden", keinen Schwellenwert), Betriebsmargen der Peers Capgemini und
Cognizant (öffentlich nicht verfügbar; durch Nettomarge ersetzt, siehe 1.6).

### 1.6 Wettbewerbsposition

| Behauptung | Beleg, der sie zeigen würde | Zahl, Quelle | Trägt? |
|---|---|---|---|
| "Volle Service-Breite in der Branche" | Benannter Konkurrenzvergleich | 10-K Item 1: reine Eigenaussage, kein Vergleich | **nein** |
| Skalenvorteil | Marge gegen zwei benannte Peers, gleiches GJ | ACN 14,7 % operative Marge (GAAP, FY2025, 10-K) vs. Capgemini/Cognizant Nettomarge 5,9 %/10,3 % (Yahoo Finance) — **nicht dieselbe Kennzahl** | **nicht belegbar** (Margendefinition inkompatibel) |
| KI-Führungsposition | Marktanteil, Umsatzbeitrag KI-Lösungen | 77.000 KI-Praktiker, Ziel 80.000 (10-K Item 1); kein Umsatzbeitrag, kein Peer-Vergleich | **teilweise** |
| Kundenbindung / wiederkehrender Umsatz | Anteil Managed Services, Verlängerungsquote | Managed Services 50 % des Umsatzes FY2025 (10-K Item 7); keine Verlängerungsquote offengelegt | **teilweise** |
| Regulatorische Position | Sicherheitszertifizierungen, Vorfallsbilanz | ISO 27001/27701, NIST zitiert (10-K Item 1); keine Verstöße gemeldet | **ja (teilweise)** |

Größe ist der einzige Vorteil, der sich zweifelsfrei beziffern lässt: Umsatz 69,7 Mrd. USD
(FY2025) ist **3,0×** Capgemini (23,4 Mrd. USD TTM) und **3,3×** Cognizant (21,6 Mrd. USD TTM,
beide Yahoo Finance, TTM ~10/2025). Jede andere Moat-Behauptung im 10-K ist eine Aussage des
Managements ohne begleitende Kennzahl — das ist selbst ein Befund, keine Formsache: ein
Bericht, der eine Burggraben-Erzählung unwidersprochen übernimmt, ohne dass das Unternehmen
dafür eine Zahl liefert, übernimmt eine Behauptung, keinen Fakt.

---

## 2. Auswertung des Jahresabschlusses

Alle Werte GAAP, Geschäftsjahr bis 31. August. Quelle je Zeile: 10-K FY2023/2024/2025, Item 7
MD&A (Haupttextseiten) bzw. Item 8 (Abschluss, F-Seiten); FY2026 aus der 8-K-Ergebnismitteilung
vom 01.10.2026 (das 10-K FY2026 liegt noch nicht vor — ausdrücklich als vorläufige
Primärquelle gekennzeichnet).

### 2.1 Umsatz

| GJ | Umsatz (Mrd. USD) | Δ ggü. Vorjahr |
|---|---|---|
| 2023 | 64,112 | — |
| 2024 | 64,896 | +1,2 % |
| 2025 | 69,673 | +7,4 % |
| 2026 | 74,2 | +6,0 % |

*Quelle: 10-K FY2024 S. 33 (FY2023/24-Vergleich); 10-K FY2025 S. 31 (FY2024/25-Vergleich);
8-K 01.10.2026, "Fourth-Quarter and Full-Year Fiscal 2026 Results" (FY2026).*

### 2.2 Auftragseingang (New Bookings) — der Frühindikator

| GJ | New Bookings (Mrd. USD) | Δ |
|---|---|---|
| 2024 | 81,2 | — |
| 2025 | 80,6 | **−0,7 %** |
| 2026 | 84,5 | +4,8 % |

Innerhalb von FY2026 fiel der Auftragseingang im dritten Quartal (bis 31.05.2026) auf
19,3 Mrd. USD, gegenüber 19,7 Mrd. USD im Vorjahresquartal — ein Rückgang **in einem Quartal,
in dem Umsatz und Gewinn beide zweistellig wuchsen** (siehe Abschnitt 4).
*Quelle: 10-K FY2025 S. 32; 8-K 18.06.2026 "Q3 FY26 Financial Review"; 8-K 01.10.2026.*

### 2.3 / 2.4 Operative Marge und operatives Ergebnis

| GJ | Operatives Ergebnis (Mio. USD) | Marge GAAP | Marge adjusted (bereinigt um Business-Optimization-Kosten) |
|---|---|---|---|
| 2023 | 8.810 | 13,7 % | 15,4 % |
| 2024 | 9.596 | 14,8 % | 15,5 % |
| 2025 | 10.226 | 14,7 % | 15,6 % |
| 2026 | — | 15,4 % | 15,8 % |

Als operatives Ergebnis wird die **GAAP-Marge** fortgeführt (Zeile oben), weil sie die
tatsächlich verausgabten "Business Optimization"-Kosten (615 Mio. USD FY2025, davon 344 Mio.
USD Personalumbau und 271 Mio. USD Wertminderungen auf zwei abgestoßene Akquisitionen; 438
Mio. USD FY2024) enthält — diese Kosten sind real angefallenes Geld, keine Einmalbuchung ohne
Kasseneffekt. Bemerkenswert: Die bereinigte Marge lag FY2023 **über** der GAAP-Marge (15,4 %
vs. 13,7 %) und blieb es in jedem der folgenden Jahre; die Lücke zwischen beiden schließt sich
nicht, sie bleibt bei ~0,9–1,1 Prozentpunkten — ein struktureller, nicht ein einmaliger
Unterschied.
*Quelle: 10-K FY2024 S. 34 (FY2023-Zahl); 10-K FY2025 S. 31–32; 8-K 01.10.2026.*

### 2.5 Nettogewinn

| GJ | Nettogewinn, Accenture plc zurechenbar (Mio. USD) | Δ | Verdünntes EPS (USD) |
|---|---|---|---|
| 2023 | 6.872 | — | 10,77 |
| 2024 | 7.265 | +5,7 % | 11,44 |
| 2025 | 7.678 | +5,7 % | 12,15 |
| 2026 | 8.367 | +9,0 % | 13,56 |

*Quelle: 10-K FY2025, Item 8, Konzern-Gewinnrücklagenübersicht (Zeile "Net income attributable
to Accenture plc"), S. F-5/F-6; EPS-Überleitung Note 3, S. F-19; FY2026: 8-K 01.10.2026.*

### 2.6 Dividende und Dividendenrendite

| GJ | Dividende je Aktie (USD) | Gezahlte Gesamtdividende (Mrd. USD) | Aktienrückkäufe (Mrd. USD) | Gesamt an Aktionäre |
|---|---|---|---|---|
| 2023 | 4,48 | 2,83 | — | — |
| 2024 | 5,16 | 3,24 | — | — |
| 2025 | 5,92 | 3,70 | 4,6 | 8,3 |
| 2026 | — (gezahlt 6,45*) | 4,0 | 7,5 | **11,5 (Rekord, +38 %)** |

\* FY2026 bezahlte Dividendensumme je Aktie ungefähr aus 3,987 Mrd. USD / ~618 Mio.
gewichtete Aktien; die am 22.09.2026 **neu deklarierte** Quartalsdividende beträgt 1,71 USD
(+5 % ggü. der FY2026-Rate), also 6,84 USD auf Jahresbasis ab Q1 FY2027 — exakt der Wert, den
yfinance als `dividendRate` zeigt (Kreuzprobe bestanden). Bei Kurs 195,05 USD ergibt das eine
**Vorwärtsrendite von 3,51 %**.
*Quelle: 10-K FY2025, Item 8 Note 14, S. F-37; 8-K 01.10.2026, Abschnitt "Dividend".*

### 2.7 Aktienkursverlauf

Siehe Abschnitt 4 für die kausale Erklärung; die Zahlen: Geschäftsjahresendkurs FY2024
(30.08.2024) 328,99 USD, FY2025 (29.08.2025) 254,72 USD, aktueller Kurs (05.10.2026) 195,05
USD. **Lücke:** Kursreihen vor FY2023 lagen außerhalb des mit yfinance abgerufenen
3-Jahres-Fensters und wurden nicht nachbeschafft (geringe Priorität für diese Frage, da die
entscheidende Bewegung ohnehin in den letzten 20 Monaten liegt).

### Kennzahlenüberblick

| Kennzahl | FY2023 | FY2024 | FY2025 | FY2026 |
|---|---|---|---|---|
| Umsatz (Mrd. USD) | 64,1 | 64,9 | 69,7 | 74,2 |
| New Bookings (Mrd. USD) | — | 81,2 | 80,6 | 84,5 |
| Operative Marge GAAP | 13,7 % | 14,8 % | 14,7 % | 15,4 % |
| Verdünntes EPS (USD) | 10,77 | 11,44 | 12,15 | 13,56 |
| Free Cashflow (Mrd. USD) | 9,00 | 8,61 | 10,87 | 11,62 |
| FCF je Aktie (USD)¹ | 14,09 | 13,55 | 17,19 | 18,81 |
| Dividende je Aktie (USD) | 4,48 | 5,16 | 5,92 | 6,84² |
| Eigenkapital, Accenture plc (Mrd. USD) | 25,69 | 28,29 | 31,20 | 31,57 |
| Buchwert je Aktie (USD)³ | — | 45,24 | 50,17 | 51,59 |
| Eigenkapitalrendite (ROE)⁴ | 28,8 % | 26,9 % | 25,8 % | 26,7 % |

¹ FCF ÷ verdünnte gewichtete Aktienzahl des jeweiligen GJ (10-K Note 3 / 8-K-EPS-Überleitung).
² neu deklarierte Rate ab Q1 FY2027, nicht die FY2026-Istzahlung (siehe 2.6).
³ Accenture-plc-Eigenkapital (ohne Minderheitsanteile) ÷ ausstehende Class-A-Aktien
(ausgegeben minus eigene Aktien) am Bilanzstichtag; FY2026-Aktienzahl aus yfinance
(Stichtagszahl im 8-K nicht einzeln ausgewiesen — als Annahme gekennzeichnet).
⁴ Nettogewinn Accenture plc ÷ Durchschnitt des Accenture-plc-Eigenkapitals (Vorjahr/
Berichtsjahr).

**Durch-Zyklus-Einordnung (Regel 12b):** Der FCF je Aktie des letzten vollen GJ (FY2026,
18,81 USD) ist das **Maximum** der vier verfügbaren Jahre, nicht der Median — die Spannweite
reicht von 13,55 (FY2024) bis 18,81 (FY2026), Median der vier Jahre ≈ 15,64 USD. Das Basisszenario
in Abschnitt 6/7 sitzt damit am oberen Rand der jüngeren Historie, nicht in der Mitte; das wird
unten explizit benannt, nicht verschwiegen.

---

## 3. Strategische Maßnahmen

Rein faktisch; keine wertenden Adjektive. Die Maßnahmen werden hier berichtet, nicht
beurteilt — das leistet Abschnitt 7.

**3.1 Reorganisation "Reinvention Services" (01.09.2025).** Die fünf bisherigen operativen
Gruppen (Strategy & Consulting, Technology, Operations, Song, Industry X) wurden zu einer
integrierten "Reinvention Services"-Einheit zusammengeführt; Berichterstattung erfolgt seither
primär nach den drei geografischen Märkten. Die Umstellung fällt genau auf den Beginn von
FY2026 und erscheint im FY2025-10-K nur als Ankündigung (10-K FY2025, Item 1).

**3.2 Federal-Geschäft (Accenture Federal Services, AFS).** AFS macht 8 % des
Konzernumsatzes, 15 % des Americas-Umsatzes und 36 % des Health & Public Service-Umsatzes aus
(10-K FY2025, Item 1/Item 7). Die US-Bundesregierung reduziert unter dem "Department of
Government Efficiency" Ausgaben und Personal; das 10-K nennt explizit "Verzögerungen bei
Beschaffungen, Preis- und Umfangsreduktionen sowie Vertragskündigungen" als bereits
eingetretene, nicht nur befürchtete Effekte (10-K FY2025 S. 32, Item 7). Die Q3-FY2026-
Ergebnismitteilung bestätigt einen geschätzten **1-Prozentpunkt-Dämpfer** auf das
Jahresumsatzwachstum FY2026 allein aus dem Federal-Geschäft.

**3.3 Akquisitionen.** 23 strategische Akquisitionen für 1,5 Mrd. USD in FY2025 (10-K FY2025,
Item 1); in FY2026 zusätzlich die Mehrheitsübernahme von Dragos sowie vollständige Übernahmen
von runZero und NetRise (OT-Sicherheit), angekündigt mit der Q3-FY2026-Mitteilung.

**3.4 KI-Investition.** 3 Mrd. USD mehrjährige Investition (seit FY2023 deklariert), 77.000
KI/Daten-Praktiker Ende FY2025, Ziel 80.000 bis Ende FY2026 (10-K FY2025, Item 1). Kein
Umsatzbeitrag oder Kundenadoptions-Kennzahl offengelegt.

### Ausblick vs. Ist

| Kennzahl | Prognose (Q3-FY26-Mitteilung, 18.06.2026) | Ist (8-K 01.10.2026) | Position |
|---|---|---|---|
| Umsatzwachstum (lokale Währung, GJ) | 3–4 % (bzw. 4–5 % ohne Federal-Effekt) | 5 % (lokale Währung) | **innerhalb/über** oberem Bandende |
| Verdünntes EPS GAAP | 13,38–13,50 USD | 13,56 USD | **über** Bandende |
| Free Cashflow | 10,8–11,5 Mrd. USD | 11,62 Mrd. USD | **über** Bandende |

**Fünfjahres-Prognosetreffer (Lücke):** Eine vollständige Fünfjahres-Tabelle "Guidance wie
erstmals gegeben vs. Ist" würde das Nachziehen jeder einzelnen Quartalsmitteilung seit FY2022
erfordern; das wurde für diesen Bericht nicht geleistet. Der einzelne verfügbare Datenpunkt
(oben) zeigt: Das Unternehmen hat seine zuletzt gegebene Prognose für FY2026 **übertroffen**,
nicht verfehlt — relevant für Abschnitt 4, wo der Kurs trotzdem fiel.

---

## 4. Beurteilung des Kursverlaufs

**Der Kurs folgt dem Auftragseingang, nicht dem Gewinn — und das sogar innerhalb eines
einzelnen Quartals.** Am 18.06.2026, dem Tag der Q3-FY2026-Zahlen, fiel die Aktie **−18,0 %**
(von 154,16 auf 126,46 USD) — der größte Einzeltagesverlust im gesamten Beobachtungsfenster.
Die an diesem Tag veröffentlichten Zahlen selbst: Umsatz +6 % auf 18,7 Mrd. USD, verdünntes
EPS +9 % auf 3,80 USD, operative Marge +20 Basispunkte auf 17,0 %, freier Cashflow 3,6 Mrd.
USD — nach klassischer Lektüre ein gutes Quartal. Was im selben Bericht stand und den Kurs
bewegte: **New Bookings fielen von 19,7 auf 19,3 Mrd. USD** (Rückgang trotz wachsendem
Umsatz), und das Unternehmen bestätigte einen anhaltenden Dämpfer aus dem
US-Bundesgeschäft. Ein Kurs, der an einem Tag mit steigendem Gewinn fällt, ist nur über den
Frühindikator erklärbar — exakt das Muster, das diese Methode in jedem Fall zuerst prüft.

Der Gesamtverlauf: Hoch 386,45 USD am 05.02.2026 (unmittelbar vor der ersten
Anzeichen-Welle), dann eine Serie von Tagesverlusten >5 % (03.02.: −9,6 %; 23.02.: −6,6 %;
13.05.: −6,0 %; 17.06.: −5,7 %; 18.06.: −18,0 %) bis zum Tief 118,15 USD (52-Wochen-Tief,
Juni 2026). Der jüngste große Ausschlag ist positiv: **+15,8 % am 01.10.2026**, dem Tag der
Q4/Gesamtjahres-FY2026-Mitteilung — Umsatz über der oberen Guidance-Grenze, Rekord-Bookings
von 141 Großverträgen ≥100 Mio. USD, Rekordausschüttung 11,5 Mrd. USD (siehe Abschnitt 2/3).
Der Markt belohnt hier denselben Frühindikator in die andere Richtung: Q4-New-Bookings +4 %
in USD, +5 % in lokaler Währung — die erste klare Beschleunigung seit dem Bookings-Rückgang
von FY2025.

**Einordnung:** Der Kurs bei 195,05 USD liegt damit **49,5 % unter dem Zwischenhoch**, aber
bereits **+65 % über dem Tief** zwei Handelstage nach den besten Quartalszahlen seit FY2025.
Ob die jüngste Erholung eine Trendwende oder eine Gegenbewegung in einem strukturellen
Abwärtstrend ist, entscheidet sich an genau dem Frühindikator, der die ganze Bewegung
erklärt: hält der Bookings-Zuwachs aus Q4 FY2026 über zwei weitere Quartale an, oder war er
ein Einzelquartal. Das ist Abschnitt 7.4's erster Trigger.

---

## 5. Entfällt (n = 0)

Siehe Kopf des Berichts.

---

## 6. Grobe Flow-Analyse

### 6.0 Annahmen

1. **Sichtweise:** Eigentümer-Sicht (owner's view), nicht Konzern-Liquiditätssicht. Es wird
   NICHT modelliert, ob Accenture seine eigenen Schulden tragen kann (6.1-Frage aus der
   "no-options variant" entfällt hier inhaltlich, weil Accenture praktisch schuldenfrei im
   Nettosinn ist — Nettofinanzposition ist in den Primärquellen nicht als Nettoverschuldung
   ausgewiesen, sondern das Unternehmen führt strukturell Überschussliquidität an die
   Aktionäre zurück, siehe 2.6). Modelliert wird stattdessen ausschließlich 6.3: Was verdient
   ein Käufer, der heute einsteigt?
2. **Einstiegspreis:** 195,05 USD, Schlusskurs 05.10.2026 (einmaliger Mittelabfluss im Jahr
   2026, Jahr 0).
3. **Ausschüttung:** Der freie Cashflow je Aktie des Unternehmens wird 1:1 als jährliche
   Ausschüttung an den Eigentümer behandelt (Dividende + Aktienrückkauf-Äquivalent). Das
   entspricht der Realität der letzten vier Geschäftsjahre: Accenture führte FY2023–FY2026
   zwischen 93 % und 100 % des freien Cashflows als Dividende plus Rückkauf an die Aktionäre
   zurück (2,83+? / 8,3 / 11,5 Mrd. USD Ausschüttung gegen 9,0 / 10,9 / 11,6 Mrd. USD FCF in
   den jeweiligen Jahren).
4. **Working Capital / Kapitalbindung:** entfällt — ein Aktienkauf hat keinen
   Working-Capital-Aufbau im Sinn der Vorlage.
5. **Steuerliche Betrachtung:** Der freie Cashflow ist bereits **nach** Unternehmenssteuer
   (FCF = operativer Cashflow abzüglich Capex, beides aus dem bereits versteuerten
   Konzernergebnis). **Nicht** modelliert ist die persönliche Steuer des Aktionärs auf
   Dividenden oder Veräußerungsgewinne (Quellensteuer, Kapitalertragsteuer) — der tatsächliche
   Netto-Mittelrückfluss eines konkreten Anlegers liegt je nach Steuerstatus darunter.
6. **Renditehorizont:** 5, 10 und 15 Jahre (2031 / 2036 / 2041), ab Kaufjahr 2026.
7. **Endwert:** Buchwert je Aktie, **konstant gehalten** bei 51,59 USD (FY2026-Stand, siehe
   Kennzahlenüberblick), nicht fortgeschrieben. Das ist der von der Methodik empfohlene
   konservative Standardfall: ein Wert, der unabhängig vom Einstiegspreis ist und für jeden
   Titel, der über Buchwert gehandelt wird — wie ACN bei einem Kurs-Buchwert-Verhältnis von
   195,05 / 51,59 = **3,78×** — eine vorsichtige Untergrenze darstellt.
8. **Hürde:** 26,7 % — Accenture plcs eigene Eigenkapitalrendite GJ2026 (Nettogewinn
   8.367.056 Tsd. USD ÷ Durchschnitt des Accenture-plc-Eigenkapitals GJ2025/26, 10-K FY2025
   Item 8 Bilanz S. F-4 + 8-K 01.10.2026). **Das ist eine ungewöhnlich hohe Hürde, bewusst
   gewählt:** Die Methodik verbietet eine erfundene Kapitalkostenannahme; die
   eigene ROE des Unternehmens ist die einzige im Bericht bereits belegte Zahl, die als Hürde
   dienen kann (Regel 8). Die Konsequenz wird in 7.2/7.5 offen diskutiert.

**Drei Szenarien, aus der eigenen Historie, nicht geschätzt:**

| Szenario | FCF je Aktie (USD) | Herkunft |
|---|---|---|
| Pessimistisch | 13,55 | schwächstes der letzten vier GJ (FY2024) |
| Basis | 18,81 | letztes volles GJ (FY2026) — **zugleich das Maximum der Reihe, siehe 2-Kennzahlenüberblick** |
| Optimistisch | 20,50 | FY2026-Wert × Konzern-eigene EPS-Wachstumsobergrenze für FY2027 (+9 %, 8-K 01.10.2026, "Fiscal Year 2027 Business Outlook") |

### 6.3 Was verdient ein Eigentümer, der heute einsteigt?

**IRR über den Anlagehorizont** (`cashflow_irr.py`, Konfiguration `data/analysis.json`):

| Horizont | Pessimistisch | Basis | Optimistisch |
|---|---|---|---|
| 5 Jahre (bis 2031) | −10,0 % | −4,9 % | −3,3 % |
| 10 Jahre (bis 2036) | +0,5 % | +5,1 % | +6,5 % |
| 15 Jahre (bis 2041) | +3,9 % | +8,1 % | +9,4 % |

**Keines der neun Felder erreicht die Hürde von 26,7 %.** Das ist bei einem Kurs von 3,78×
Buchwert gegen eine Hürde in Höhe der eigenen Eigenkapitalrendite praktisch unvermeidlich und
unten (7.2/7.5) eingeordnet, nicht als Überraschung behandelt.

### 7.3-Vortabelle: Einstiegsschwelle (Preis, der die Hürde exakt erreicht)

| Szenario | 5 Jahre | 10 Jahre | 15 Jahre |
|---|---|---|---|
| Pessimistisch | 64,6 USD | 64,4 USD | 64,3 USD |
| Basis | 83,5 USD | 87,5 USD | 88,7 USD |
| Optimistisch | 89,6 USD | 94,9 USD | 96,6 USD |

Aktueller Kurs 195,05 USD liegt **102 % bis 203 %** über jeder dieser Schwellen — siehe
Abschnitt 7.3 für die Einordnung.

---

## 7. Verdict — Investitionspotenzial

### 7.1 Urteil

**Nicht kaufen zu 195,05 USD; für einen Altaktionär: Position nicht aufstocken, Halten
vertretbar, solange der Bookings-Trend aus Q4 FY2026 sich in Q1/Q2 FY2027 bestätigt (Trigger
7.4).** Für einen Neueinsteiger ohne bestehende Position gibt es bei dieser Hürde keinen
Kurs im beobachtbaren Bereich, der eine 15-Jahres-IRR über 26,7 % verspricht, ohne
Wachstumsannahmen zu treffen, die das Unternehmen noch nie geliefert hat (7.3). Für den
Altaktionär ist die Lage anders: er trägt kein Einstiegsrisiko zu 195,05 USD, sondern
entscheidet nur über Verkauf vs. Halten einer Position, deren Buchwert pro Jahr zuverlässig
wächst und die ~3,5 % Dividendenrendite plus Rückkäufe liefert, unabhängig vom Erreichen
einer 26,7-%-Hürde.

### 7.2 Begründung aus den Befunden

**Was dagegen spricht:**
- Der Kurs (195,05 USD) liegt beim Basisszenario über **jede** 7.3-Schwelle um 120–134 %
  (Abschnitt 6.3: Schwellen 83,5–88,7 USD je nach Horizont).
- Der Frühindikator Bookings ist über FY2025 gefallen (−0,7 %) und innerhalb von FY2026
  mehrfach volatil (Q3 FY2026 YoY-Rückgang bei gleichzeitig steigendem Umsatz, Abschnitt 4) —
  genau das Muster, das in der Referenzmethode den Kurs treibt, nicht die
  Gewinn-und-Verlust-Rechnung.
- Das Basisszenario (FY2026-FCF/Aktie) ist das **Maximum**, nicht der Median der letzten vier
  Jahre (2-Kennzahlenüberblick) — ein konservativeres Basisjahr (Median ≈ 15,64 USD) würde die
  7.3-Schwellen weiter senken, nicht anheben.
- Insider haben während der ~70-%-Korrektur **keinen** offenen Marktkauf getätigt (1.2) — kein
  Signal eigener Überzeugung bei den Personen mit dem besten Informationsstand.

**Was dafür spricht:**
- Das Unternehmen hat seine zuletzt gegebene Prognose für FY2026 in allen drei genannten
  Metriken übertroffen, nicht verfehlt (Abschnitt 3).
- Record-Ausschüttung 11,5 Mrd. USD FY2026 (+38 %), Dividende gerade um 5 % erhöht — das
  Geschäft erzeugt tatsächlich den Cashflow, der in Abschnitt 6 modelliert wird; nichts davon
  ist Projektion.
- Q4-FY2026-Bookings beschleunigen wieder (+4–5 %) nach dem Q3-Einbruch — falls das anhält,
  ist der Frühindikator, der den Kurs −49,5 % unter sein Hoch gedrückt hat, bereits am Drehen.
- Operative Marge GAAP steigt vier Jahre in Folge (13,7 %→14,8 %→14,7 %→15,4 %), trotz der
  AFS-Gegenwinde.

**Was für Abwarten statt Kaufen oder Verkaufen spricht:**
- Die föderale Belastung (AFS, 3.2) ist politisch, nicht zyklisch — ihre Dauer ist nicht aus
  den Primärquellen ableitbar und verdient ein bis zwei weitere Quartale Beobachtung, bevor
  man den Bookings-Rebound aus Q4 FY2026 als Trendwende liest.

**Wo dieses Urteil vom Konsens abweicht:** Der Analysten-Mittelwert (223,03 USD, Abschnitt
1.4) liegt **+14,3 %** über dem aktuellen Kurs — der Konsens tendiert "Buy", dieses Urteil
"nicht kaufen". Der Mechanismus, nicht nur die Richtung: ein 12-Monats-Kursziel bewertet das
Unternehmen über Gewinn-Multiplikatoren mit einem impliziten Fortführungswert; das hier
verwendete Modell bewertet eine 15-Jahres-Ausschüttungsreihe gegen einen zum Buchwert
gehaltenen, nicht fortgeschriebenen Endwert und gegen eine Hürde von 26,7 % — eine der
striktesten vertretbaren Hürden, weil sie der eigenen Kapitalrendite des Unternehmens
entspricht. Der Unterschied liegt im Bewertungsaufschlag für Fortführung, nicht in
unterschiedlichen operativen Zahlen: beide Seiten sehen dieselben Bookings-, Margen- und
EPS-Werte.

### 7.3 Einstiegsschwelle und erforderliches Wachstum

Schwellentabelle siehe 6.3 oben (identisch, hier nicht wiederholt). **Erforderliches
jährliches Wachstum der Ausschüttung**, das der aktuelle Preis von 195,05 USD bereits
voraussetzt (`required_growth()`):

| Szenario | 5 Jahre | 10 Jahre | 15 Jahre |
|---|---|---|---|
| Pessimistisch | 70,3 % p.a. | 34,4 % p.a. | 25,9 % p.a. |
| Basis | 55,0 % p.a. | 27,0 % p.a. | 20,5 % p.a. |
| Optimistisch | 51,1 % p.a. | 25,0 % p.a. | 19,0 % p.a. |

Das Unternehmen hat sein FCF je Aktie in den letzten vier Geschäftsjahren mit **durchschnittlich
10,1 % p.a.** gesteigert (von 14,09 auf 18,81 USD, inkl. des schwächeren FY2024-Jahres) — selbst
im günstigsten Fall (15 Jahre, optimistisches Szenario) verlangt der aktuelle Preis **fast das
Doppelte** dieser realisierten Rate, über dreimal so lange fortgeschrieben wie die verfügbare
Historie. Keine der neun Zellen ist eine plausible Fortschreibung der eigenen Vergangenheit des
Unternehmens.

**Was der Preis an Endwert bereits voraussetzt** (`required_terminal_value()`, als Vielfaches
des gehaltenen Buchwerts von 51,59 USD):

| Szenario | 5 Jahre | 10 Jahre | 15 Jahre |
|---|---|---|---|
| Pessimistisch | 9,3× Buchwert | 28,0× Buchwert | 89,2× Buchwert |
| Basis | 8,1× Buchwert | 23,2× Buchwert | 72,7× Buchwert |
| Optimistisch | 7,7× Buchwert | 21,7× Buchwert | 67,5× Buchwert |

Zum Vergleich: der Markt bewertet ACN **heute** mit 3,78× Buchwert. Jede Zelle oben verlangt ein
Vielfaches, das weit jenseits des heutigen eigenen Multiples liegt — ein Beleg dafür, dass die
26,7-%-Hürde in Kombination mit einem 3,78×-Buchwert-Einstieg eine fast unerreichbare Latte
legt, nicht ein spezifischer Mangel von Accenture. **Diese Tabelle ist der eigentliche Befund
von 7.3**, nicht die nackte IRR-Zahl: sie zeigt, dass das Urteil "nicht kaufen" bei dieser
Hürde für praktisch jedes Unternehmen gelten würde, das über Buchwert gehandelt wird — siehe
7.5.

**Multiple im Kontext:** KGV (trailing, eigene Berechnung aus FY2026-EPS) 195,05/13,56 =
**14,4×**. Das liegt deutlich **unter** den eigenen Geschäftsjahresend-Multiples der letzten
zwei Jahre (FY2024: 328,99/11,44 = 28,8×; FY2025: 254,72/12,15 = 21,0×) — die Aktie ist nach
ihrem eigenen jüngsten Maßstab historisch günstig bewertet, nicht teuer. Gegen die zwei
benannten Peers: Capgemini 13,5× und Cognizant 12,6× (beide trailing, Yahoo Finance) liegt ACN
mit 14,4–14,7× leicht darüber. **Lücke:** eine echte 10-Jahres-Multiple-Historie wurde nicht
beschafft (nur zwei Jahresendpunkte verfügbar); das Perzentil "historisch günstig" stützt sich
daher auf zwei Vergleichsjahre, nicht zehn.

### 7.4 Trigger, die das Urteil kippen würden

- **Aufwärts:** New Bookings bestätigen die Q4-FY2026-Beschleunigung (+4–5 %) auch in Q1 und
  Q2 FY2027 (nächste Meldungen üblicherweise Mitte Dezember 2026 und Mitte März 2027) —
  Accenture meldet New Bookings jedes Quartal, kostet nichts zu beobachten.
- **Aufwärts:** Ein offener Marktkauf durch einen Director oder Officer (Form-4-Code "P") —
  bisher im beobachteten Fenster nicht vorgekommen; jede Einzelmeldung ist öffentlich und
  sofort verfügbar.
- **Abwärts:** AFS-Umsatzanteil oder -Wachstum verschlechtert sich weiter in der nächsten
  10-Q/8-K (derzeit 8 % des Konzerns, 36 % von Health & Public Service) — eine Zahl, die
  Accenture jedes Quartal offenlegt.
- **Abwärts:** Kurs fällt unter das bisherige 52-Wochen-Tief von 118,15 USD — öffentlich,
  kostenlos beobachtbar, und ein Bruch des bisherigen Tiefs wäre ein Hinweis, dass die
  Oktober-Erholung tatsächlich nur eine Gegenbewegung war.

### 7.5 Was dieses Urteil nicht wissen kann

**Die Hürde entscheidet das Ergebnis fast allein, und das ist nicht verschleierbar.** Eine
Hürde von 26,7 % — die eigene Eigenkapitalrendite des Unternehmens — ist nach der Logik dieser
Methode die einzige im Bericht belegte, nicht erfundene Zahl, aber sie ist für einen externen
Käufer, der über Buchwert einsteigt, eine andere Größe als für das Unternehmen selbst, das auf
seinem EIGENEN Buchkapital arbeitet. Bei einem Kurs von 3,78× Buchwert verlangt diese Hürde
implizit, dass der externe Käufer densel­ben Ertrag auf einen ~3,78-fach teureren Einsatz
erzielt — arithmetisch nur über außergewöhnliches, hier nicht realisiertes Wachstum oder einen
drastisch höheren Endwert-Multiple möglich (7.3). **Argument gegen das eigene Urteil:** Mit
einer gebräuchlicheren Aktionärs-Hürde (z. B. 8–10 % nominale Eigenkapitalkosten statt 26,7 %
ROE) würden mehrere der 6.3-Zellen bereits beim Basis- oder Optimistisch-Szenario über 10–15
Jahre die Hürde erreichen — der Bericht hält bewusst an der strengeren, aber unbelegten-WACC-
vermeidenden Regel fest (Regel 8 der Methodik), nicht weil 26,7 % die einzig vertretbare Hürde
wäre, sondern weil sie die einzige ist, die ohne Erfindung einer Kapitalkostenzahl auskommt.
**Zweites Argument gegen das Urteil:** Der Analystenkonsens (7.2) sieht dieselben Zahlen und
kommt mit einer marktüblichen Diskontierung zu "Buy" — ein Leser, der eine klassische
Multiplikator-Bewertung für angemessener hält als diese Methode, sollte das Urteil
entsprechend relativieren. **Was nicht kalkulierbar ist:** die Dauer der föderalen
Umsatzbelastung (3.2) — eine politische, nicht eine zyklische Variable, zu der die
Primärquellen keine Prognose liefern.

---

## Nicht verfügbare Angaben (Lücken, gesamt)

- Form-4-Insiderhandel 06.10.2025–16.02.2026 (vier Monate des 12-Monats-Fensters fehlen).
- Vollständige Analysten-Ratingverteilung (nur Mittelwert + ein Einzelrating vorhanden).
- IT-Dienstleistungsmarktgröße/-wachstum (Gartner/IDC/Statista kostenpflichtig).
- Betriebsmargen von Capgemini und Cognizant (nur Nettomarge verfügbar, nicht
  vergleichbare Kennzahl).
- Kundenkonzentration in Prozent (10-K nennt keinen Schwellenwert).
- Fünfjahres-Prognosetreffer-Tabelle (nur ein Quartal nachgezogen, siehe Abschnitt 3).
- Zehnjahres-Multiple-Historie für den KGV-Kontext (nur zwei Jahresendpunkte verfügbar).
- Kopfzahlen FY2023 (nur FY2024/FY2025 sicher extrahiert).

## Quellenverzeichnis

**Unternehmensberichte:** 10-K FY2023 (eingereicht 12.10.2023), 10-K FY2024 (10.10.2024),
10-K FY2025 (10.10.2025), DEF 14A 2025 (12.12.2025), 8-K-Ergebnismitteilungen Q3 FY2026
(18.06.2026) und Q4/FY2026 (01.10.2026) — alle SEC EDGAR, CIK 0001467373.

**Externe Quellen (sekundär, mit Abrufdatum):** yfinance/Yahoo Finance (Kurse, Kennzahlen,
Analystenkonsens, Peer-Daten), abgerufen 06.10.2026.

---

*Erstellt 06.10.2026. Methodik: `startup-investment-analyzer`-Skill. Rechenmodell:
`data/analysis.json` + `cashflow_irr.py`; Governance- und Marktrecherche in
`domain1_governance.md` / `domain3_market.md` dokumentiert (jeweils mit eigenen Lücken- und
Widerspruchsabschnitten). Dieser Bericht ist ein Analyseinstrument, kein Anlagerat.*
