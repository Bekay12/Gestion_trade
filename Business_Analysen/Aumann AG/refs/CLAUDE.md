# refs/ — die Quelldokumente

Die PDF-Dateien sind **gitignoriert**. Nach einem frischen Klon neu laden:

```bash
./scripts/fetch_reports.sh
```

Das Skript schreibt zugleich `MANIFEST.md` mit URL, Seitenzahl, SHA-256-Präfix und
Abrufdatum je Dokument. Aus dieser Datei wird das Quellenverzeichnis in
`../sections/90-quellen.tex` **wörtlich** übernommen.

| Datei | Dokument | Rolle |
|---|---|---|
| `gb2024.pdf` | Geschäftsbericht 2024 | Hauptquelle für Teil 2, 3, 5, 6 |
| `gb2025.pdf` | Geschäftsbericht 2025 | Ist 2025, Organe, Eigentümer, Bezugsbasis Teil 6 |
| `gb2023.pdf`, `gb2022.pdf` | Geschäftsberichte 2023 / 2022 | Vorjahresvergleich, Kurse |
| `q1-2025.pdf`, `h1-2025.pdf`, `q3-2025.pdf` | Zwischenberichte 2025 | Verfolgung der Prognose (3.6) |
| `q1-2026.pdf` | Quartalsmitteilung Q1 2026 | jüngster Stand |

Alle Berichte sind **nur auf Englisch** veröffentlicht. Die Arbeit ist deutsch und zitiert
englische Quellen; darauf weist das Quellenverzeichnis hin.

## Seitenzahlen prüfen — nicht raten

Zitiert wird die **gedruckte** Seite, nicht die PDF-Seite. Nachgeprüft am 29.07.2026 über
den Fußzeilenmarker `Page N`: Bei den Geschäftsberichten 2023–2025 sind beide gleich; bei
**allen drei** Zwischenberichten gilt durchgängig PDF = gedruckt + 1.

Seitenzahlen aus den Zeilennummern eines `pdftotext`-Volltexts abzuleiten ist **falsch**
und hat bereits fünf fehlerhafte Zitate erzeugt. Richtige Methode: seitenweise extrahieren
und die Fußzeile lesen.

```bash
python3 - <<'PY'
import subprocess, re
pdf, needle = "refs/gb2025.pdf", "Sebastian Roll, Business economist"
n = int(re.search(r'Pages:\s+(\d+)',
    subprocess.run(['pdfinfo',pdf],capture_output=True,text=True).stdout).group(1))
for p in range(1, n+1):
    t = subprocess.run(['pdftotext','-layout','-f',str(p),'-l',str(p),pdf,'-'],
                       capture_output=True,text=True).stdout
    if needle in t:
        pr = re.findall(r'Page (\d+)', t)
        print(f"PDF {p} / gedruckt {pr[-1] if pr else '?'}"); break
PY
```

## Belegte Fundstellen

**GB 2024:** S. 2 Kennzahlenübersicht („Aumann in figures", Konzern und bereinigt) ·
S. 5 Welcome Note (Nettoliquidität 138,2, Eigenkapital 201,7, Prognose 2025, Dividende,
Rückkaufangebot 1.434.523 zu 12,37 €) · S. 6 Aumann Lauchheim GmbH · S. 7 Reorganisation
China · S. 8 Gewinnverwendungsvorschlag 0,22 € · S. 9 Geschäftsmodell und sechs Standorte ·
S. 10 Produktspektrum, Elektroden-/MEA-Anlagen, Umbenennung Next Automation, 80 %-Anteil ·
S. 13 Segmentzahlen, Kursangaben · S. 14 Elektrodenfolien · S. 17 Bereinigung
Personalaufwand · S. 18 Definition der Steuerungskennzahlen · S. 52 Beteiligungsliste.

**GB 2025:** S. 2 Kennzahlen · S. 4 Nettoliquidität 148,1 · S. 13 Segment Next Automation,
Schlusskurs 12,32 · S. 29 keine eigenen Aktien zum Stichtag · S. 52 Kapitalherabsetzungen ·
S. 90 Organe · S. 93 MBB SE als Mutterunternehmen, Mitarbeitende.

**GB 2023:** S. 14 Schlusskurse 2023 (18,58) und 2022 (11,48).

**Zwischenberichte 2025** (gedruckte Seiten): Q1 S. 2, H1 S. 3, Q3 S. 3 — jeweils die
unveränderte Bestätigung der Prognose.

## Nicht in den Berichten enthalten

Höchst- und Tiefstkurse, Kursangaben in den Quartalsberichten, der genaue Anteil der
MBB SE, Analystenratings und Kursziele. Was fehlt, wird im Dokument benannt, nicht ergänzt.
Aktionärsstruktur, Analystennamen, **ISIN und WKN** stammen von
<https://www.aumann.com/en/investor-relations/shares> (Abruf 29.07.2026). Die ISIN
`DE000A2DAM03` kommt in **keinem** der acht Berichte vor — eine frühere Zuschreibung an
GB 2024 S. 13 war falsch und ist korrigiert. Die Seite 13 belegt lediglich die Notierung
im Prime Standard seit März 2017 sowie die Kursangaben.

## Ergebnis der Schlussprüfung (29.07.2026)

Alle 64 mit einer Seitenangabe versehenen Makros aus `../data/kennzahlen.tex` wurden
seitenweise gegen den Bericht geprüft; alle sind belegt. Zwei Zuschreibungen waren falsch
und sind korrigiert: die ISIN (siehe oben) und der Segmentumsatz E-Mobility 2024
(`258.530` steht auf S. 2 unter „thereof E-mobility", nicht auf S. 13).

Zwei veröffentlichte Werte lassen sich aus den ebenfalls veröffentlichten, gerundeten
Eingangsgrößen nicht exakt nachrechnen — `\KursDeltaZFIV` (−42,9 % gegenüber −42,8 % aus
10,62/18,58) und `\EbitdaMargeNextAutoZFV` (12,8 % gegenüber 12,7 % aus 5,1/40,2). In
beiden Fällen ist der hinterlegte Wert der **im Bericht gedruckte**; das ist so gewollt.
