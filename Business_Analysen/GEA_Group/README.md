# GEA Group — Anlageanalyse (Vollbericht)

Vollbericht nach dem Muster des Skills `startup-investment-analyzer`
(sieben Teile, Seitenprüfung, keine Zahl von Hand im Fließtext).
Ergebnis: `out/analyse.pdf`, 19 Seiten, **217 von 217 Werten auf ihrer
gedruckten Seite belegt**.

Vorläufer war der vertiefte Filter in `../Dual_Champions/` (sechs Titel,
drei Paare); dieser Bericht behandelt GEA allein und in voller Tiefe.

## Bauen

```bash
./build.sh          # -> out/analyse.pdf
```

`build.sh` bricht ab bei einem nicht belegten Wert (`pruefe_seiten.py`) oder
einer hartkodierten Zahl im Fließtext (`check_literals.py`).

## Woher die Zahlen kommen

| Schritt | Skript | Ausgabe |
|---|---|---|
| Seiten zerlegen (gedruckte Seite, nicht PDF-Seite) | `seiten.py`, `lies.py` | `refs/*.txt` |
| Abschlusszeilen benennen (lokales Modell) und lesen (Code) | `abschluss_extrakt.py`, `reihe.py` | `data/reihen.json` |
| Kennzahlenseite (S. 2 jedes Berichts) | `kennzahlen_seite.py` | `data/kennzahlen_seite.json` |
| Prognose-Ist-Vergleich, Prognose *wie zuerst gegeben* | `prognose.py` | `data/prognose.json` |
| Seitenwachhund über alle gespeicherten Werte | `pruefe_seiten.py` | `data/pruefung.tex` |
| Rechnungen, Makros und Tabellenkörper | `rechnung_gea.py` | `data/kennzahlen.tex`, `data/*.tex` |

Das lokale Modell benennt nur Zeilenetiketten; **die Werte liest Code aus dem
Textlayer derselben Seite**. Grund: `gemma4:12b` erfand auf einer dichten
Finanztabelle 39 Beträge, `pdftotext -layout` las alle 29 korrekt.

## Ergebnis

Votum **nicht kaufen** zu 64,60 EUR (16.09.2026). Schwellenkurs auf zehn Jahre
17,28 EUR bei der Hürde von 13,56 % (MSCI World, 10 J.), 22,04 EUR bei 9,08 %.
Der Kurs verlangt 31,6 % Wachstum des freien Zuflusses p. a.; Mission 30
impliziert 6,8 %.

Gegenargumente stehen in 7.5: der auf den Buchwert begrenzte Endwert und
sechs Organkäufe ohne einen Verkauf.

## Zwei Befunde am Bericht selbst

- **Widerspruch im Geschäftsbericht 2025.** Der Prognose-Ist-Vergleich auf
  S. 40 nennt als EBITDA-Marge 2025 16,2 %, Kennzahlenseite und Prosa nennen
  16,5 %. Aus den ebenfalls gedruckten 907,4 / 5.495,4 folgt 16,51 %. Die
  16,2 % sind zugleich die Untergrenze der zuletzt angepassten Prognose.
  Im Bericht als Widerspruch festgehalten, nicht aufgelöst.
- **Dividendenzahlung 2019 = 2020 = 153.418 TEUR.** Kein Lesefehler: zwei
  Geschäftsberichte drucken dasselbe Paar, und 0,85 EUR × 180,5 Mio. Aktien
  bestätigt es.

## Grenzen

Die Seitenprüfung fängt eine Zahl, die *nicht* auf der genannten Seite steht.
Eine dort stehende, aber falsch zugeordnete Zahl fängt sie nicht; dagegen
stehen die Kettenprüfung über zwei Berichte und der Abgleich von
Kennzahlenseite und Kapitalflussrechnung.
