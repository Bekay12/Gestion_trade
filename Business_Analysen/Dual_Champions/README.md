# Vertiefter Filter: sechs Dual Champions in drei Paaren

`out/analyse.pdf` (26 Seiten, Deutsch). Titel aus `Magasines/Combined_scan_recos_2026-09-19.csv`:
OXY/TTE (Oel), TNK/FRO (Tanker), GEA/BESI (Ausruester). Muster: `startup-investment-analyzer`
(Teile 1.2, 1.6, 2, 4, 7; Teil 6 nur als Anlegerrechnung).

## Bauen

```bash
./build.sh     # xbrl -> reihe (Kettenpruefung) -> pruefe_seiten -> rechnung -> check_literals -> LaTeX
```

Bricht ab bei einem unerklaerten Kettenbruch, einem Wert, der nicht auf seiner gedruckten
Seite steht (512 geprueft), oder einer Zahl im Fliesstext.

## Quellen neu beschaffen (refs/ ist gitignoriert)

```bash
python3 scripts/hole_quellen.py      # SEC: 10-K/20-F 2016/2019/2022/2024/2025 + companyfacts
python3 scripts/seiten.py            # gedruckte Seiten (20-F: F-Seiten als eigene Folge)
python3 scripts/abschluss_extrakt.py # qwen3.5:9b nennt Etiketten, Code liest Werte
python3 scripts/dossier.py           # 1.2/1.6-Kandidaten mit woertlichem Zitat
```

GEA/BESI/TTE-URD/MSCI-Factsheet: PDF-Adressen in `docs/progress.md`.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `scripts/reihe.py` | Mehrjahresreihen, Kettenpruefung, XBRL-Abgleich, erklaerte Anpassungen (`ERKLAERT`) |
| `scripts/rechnung.py` | Zahlenschicht (ROH mit Seite, EXTERN), Anlegerrechnung mit `cashflow_irr.py` |
| `scripts/pruefe_seiten.py` | Seitenwachhund |
| `sections/` | Prosa, keine Zahl |
| `docs/befunde-und-entscheidungen.md` | Huerde, Eigentuemer, Directors' Dealings, Lesefallen |

## Voten in einem Satz

Kein Titel ist zum Kurs vom 16./18.09.2026 zu kaufen (Huerde 13,56 %, MSCI World 10 J.);
GEA beobachten (Schwelle 22,04 EUR bei 9,08 %), die uebrigen nicht kaufen.
