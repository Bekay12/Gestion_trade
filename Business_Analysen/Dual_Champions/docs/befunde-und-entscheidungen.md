# Befunde und Entscheidungen

## Huerde (Regel 8)
- **13,56 % p. a.**: MSCI World, Gross Returns (USD), annualisiert 10 Jahre, Stand 31.08.2026;
  Factsheet `refs/msci-world-factsheet.pdf`, S. 1 (Zeile "MSCI World", Spalte "10 Yr"),
  abgerufen 19.09.2026. Sekundaerquelle. Grund: eine gemeinsame Huerde fuer alle sechs Titel
  macht die Paare vergleichbar; die eigene Eigenkapitalrendite wuerde einem ertragsschwachen
  Titel eine niedrige Huerde schenken.
- Sensitivitaet: **9,08 % p. a.** seit 31.12.1987, gleiche Seite. 13,56 % spiegelt ein
  aussergewoehnliches Boersenjahrzehnt.
- Warnung: Die Zusammenfassung der Websuche nannte 10,07 % - die Zahl steht so nicht im
  Factsheet. Nur aus dem PDF uebernommen.

## OXY
- Berkshire Hathaway (Warren E. Buffett und verbundene Gesellschaften): 32,43 %
  einschliesslich 83,9 Mio. Aktien aus Optionsscheinen; DEF 14A 2026, S. 65.
  Dodge & Cox 8,38 %, Vanguard 8,09 % (ebd.).
- OxyChem an Berkshire verkauft, rund 9,7 Mrd. USD, Vollzug 02.01.2026 (DEF 14A 2026,
  Abschnitt Related Party Transactions). Folge: der Konzern-Cashflow 2025 enthaelt
  OxyChem; die Anlegerrechnung muss das Chemiegeschaeft herausrechnen oder die Luecke nennen.
- Vorzugsaktien (Berkshire): "Cash dividends paid on common and preferred stock" ist eine
  Summe; die Vorzugsdividende muss abgezogen werden, das Eigenkapital enthaelt die Vorzugsaktien.
- Directors' Dealings 12 Monate (Form 4): 2 Kaeufe (Klesse 16.12.2025, 5.000 zu 38,98;
  Jackson, CEO, 23.06.2026, 4.770 zu 52,38), 0 Verkaeufe. data/form4_oxy.json.
- DEF 14A: Inhaltsverzeichnis nennt S. 64 fuer "Security Ownership"; die gedruckte
  Fusszeile der Tabellenseite ist 65. Belegt wird die Fusszeile.

## TNK
- Teekay Corporation: 30,7 % aller Class-A- und Class-B-Aktien, 100 % der Class B
  (Stimmrechtskontrolle); Form 20-F 2025 (Seite folgt aus dossier).

## Kurse
- Yahoo Finance via yfinance, abgerufen 19.09.2026, Schlusskurse (splitbereinigt) seit
  01.12.2014: data/kurse_roh.json. TTE ueber die NYSE-Notiz (USD), weil TotalEnergies in
  USD berichtet - kein Waehrungsumrechnungsschritt je Aktie.
- TNK: Reverse Split 1:8 (2019) - EPS aus Berichten vor 2019 nur in restated Fassung nutzen.

## Directors' Dealings (1.2), Stand 19.09.2026, Zeitraum 12 Monate
- OXY: 2 Kaeufe (2 Personen), 0 Verkaeufe - SEC Form 4, data/form4_oxy.json.
- TTE: 0 Kaeufe, 8 Verkaeufe (4 Personen, u. a. CEO Pouyanne 31.008 Aktien zu 76,92 EUR am
  26.03.2026) - Aggregator der AMF-Meldungen (abcbourse.com), data/dd_tte.json. Ob die Verkaeufe
  Performance-Aktien nach Zuteilung betreffen, gibt die Quelle nicht an.
- GEA: 6 Kaeufe (4 Vorstaende) 27.03.-05.06.2026 zu 53,60-59,60 EUR, 0 Verkaeufe gefunden -
  EQS-Meldungen des Emittenten, data/dd_gea.json. Weitere Kaeufe Ende Juni laut Suche, ungeprueft.
- BESI: 31 Meldungen von 10 Personen seit 19.09.2025 im AFM-Register (MAR 19), Export
  refs/dd/besi_afm.csv; Richtung und Volumen stehen nur in der Einzelmeldung - nicht ausgewertet
  (Luecke, keine Deutung). Haeufung 23./24.04.2026 bei fast allen Personen: vermutlich
  Verguetungsvorgang, nicht belegt.
- FRO: eine Meldung identifiziert (O'Shaughnessy, Ausuebung 36.000 synthetischer Optionen,
  26.05.2026 - Verguetung); vollstaendige Liste ueber Oslo Newsweb nicht abrufbar (Luecke).
- TNK: Foreign Private Issuer, nur an der NYSE notiert - keine Meldepflicht (Form 4 entfaellt,
  keine EU-Notierung). Strukturelle Luecke, keine unvollendete Suche.
