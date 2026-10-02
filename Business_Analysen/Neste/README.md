# Neste Oyj — Anlageanalyse mit zeitlicher Reichweite (Vollbericht)

Vollbericht nach dem Muster des Skills `startup-investment-analyzer` (sieben Teile,
Seitenpruefung, keine Zahl von Hand im Fliesstext), aufgebaut wie `../GEA_Group/`.
Ergebnis: `out/analyse.pdf`, 26 Seiten, **326 von 326 Primaerwerten auf ihrer gedruckten
Seite belegt**, 41 Sekundaerwerte getrennt (`data/extern.json`). Stand 27.09.2026.

Vorlaeufer: `Neste_Investitionsanalyse.html` (15.09.2026, 50 Werte, Huerde = ROACE 5,3 %).
Dieser Bericht ersetzt ihn; die Huerde ist jetzt dieselbe wie bei GEA und den Dual Champions.

## Bauen

```bash
./build.sh                                   # -> out/analyse.pdf
python3 scripts/check_footnote_pages.py      # nach jeder Textaenderung
python3 scripts/check_footnote_groups.py
cd scripts && python3 -m unittest discover -p 'test_*.py'   # 49 Tests, offline
```

`build.sh` bricht ab bei einem Kettenbruch der Reihe, einem nicht belegten Wert oder einer
Zahl im Fliesstext.

## Woher die Zahlen kommen

| Schritt | Skript | Ausgabe |
|---|---|---|
| Seiten zerlegen (gedruckte Seite aus der Fusszeile, Folgefilter) | `seiten.py` | `refs/*.txt` |
| Zellen lesen: Kennzahlenseite nach Reihenfolge, Kapitalflussrechnung nach Spaltenstellung | `lesen.py` | – |
| Mehrjahresreihe 2017–2025 mit Kettenpruefung ueber die Berichte | `reihe.py` | `data/reihen.json` |
| Seitenwachhund ueber alle Rohwerte und Reihenwerte | `pruefe_seiten.py` | `data/pruefung.tex` |
| Rechnungen, Makros, Tabellenkoerper | `rechnung_neste.py` | `data/kennzahlen.tex`, `data/*.tex` |

## Ergebnis

Votum **nicht kaufen** zu 34,01 EUR (25.09.2026), auf allen drei Horizonten:

| Horizont | Massstab | Wert je Aktie |
|---|---|---|
| bis 12 Monate | Konsensziel / Median-Multiple (10,8x) auf LTM-EBITDA / auf Median-EBITDA | 34,26 / 40,23 / 22,42 |
| 1–3 Jahre | Median-Multiple auf Median-EBITDA, skaliert auf 6,8 Mio. t (Annahme) | 28,83 |
| 10 Jahre | Schwellenkurs Basis / LTM ohne Umlaufvermoegen (13,56 %); bei 9,08 % | 7,36 / 15,27; 19,30 |

Halter: reduzieren, Pruefpunkt Q3-Bericht 29.10.2026. Nichthalter: Kaufmarken 22,42 EUR
(bis drei Jahre) und 15,27 / 19,30 EUR (zehn Jahre).

Kernbefund: Selbst zehn Jahre Zufluss auf LTM-Niveau ergeben bei Ausstieg zum heutigen
Boersenwert nur 4,4 % p. a.; die Huerde verlangt 3.543 Mio. EUR jaehrlich, das beste Jahr
der Reihe brachte 1.180 (2020). Konsens: 19 Analysten Kaufen, Ziel 34,26 = Kurs;
Konsens-EPS 2027 22 % unter 2026.

## Befunde am Material, die eine spaetere Sitzung nicht neu herleiten muss

- **Kapitalflussrechnung GB 2025 (S. 154):** rechte Tabelle hat eine Randspalte ("3", "4",
  "18") hinter den Werten und eine Notenspalte davor; Lesen nach "letzten n Zahlen" liefert
  falsche Werte. Werte der rechten Tabelle enden bis zu 8 Zeichen hinter ihrem Kopf.
  `lesen.werte_spalten()` mit Fenster, Regressionstest `test_lesen.py`.
- **GB 2021 S. 170** druckt im rechten Jahreskopf "1 Jan–31 Dec 2022" statt 2020 (Satzfehler
  des Berichts); die Werte selbst sind 2020 (Kette mit GB 2020 bestaetigt).
- **Kennzahlenseite:** die Nachbarzelle "- of weighted average number of shares" beginnt mit
  Bindestrich und wurde in Verschuldungsgrad und ROACE gelesen; die Kettenpruefung fing es.
- **Leasing 2019/2020** hat keine eigene Zeile; Gesamtabfluss im Anhang GB 2020 S. 193 (68/115).
- **Neudarstellungen (BEKANNT in reihe.py):** CFO 2024 (1.183 -> 1.154), Ergebnis 2018
  (779 -> 775), ROACE ab 2020 "Comparable", Boersenwert 2022 (33.063 vs 33.091), RP-Marge
  2022 (804 -> 779, neue Formel ab 2023).
- **Vergleichbares EBITDA** in heutiger Abgrenzung erst ab 2019 (2019: 2.452, nicht
  vergl. EBIT 1.962 + Abschreibungen 502).
- **Instandhaltungsinvestitionen** gibt es 2017–2024 im Lagebericht; GB 2025 teilt nicht mehr auf.
- **Rotterdam:** GB 2024 S. 91: Start 2026 -> 2027, Kosten 1,9 -> 2,5 Mrd., danach ~0,5 Mrd./J.
  Investitionen. Der Governance-Subagent meldete den veralteten Stand (1,9 Mrd., H1 2026) –
  Webseiten von Neste zum Projekt sind nicht aktualisiert.
- **Fussnoten in Tabellen gehen in Gleitumgebungen verloren:** dort `\QI{Dok}{Seite}`
  (Inline-Beleg). Mehrfachzitat auf einer Seite: `\QL{key}{Dok}{Seite}` + `\QR{key}`.
- **Sekundaerquellen:** OP Corporate Bank "40 -> 43" widerspricht der eigenen URL (33 -> 34),
  RBC-Datum ungeklaert – beide nicht aufgenommen.
