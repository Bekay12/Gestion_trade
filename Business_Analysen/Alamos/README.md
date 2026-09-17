# Alamos Gold Inc. — Anlageanalyse

LaTeX-Quelle einer belegpflichtigen Unternehmensanalyse aus Primärdokumenten
(SEC-Einreichungen Form 40-F und 6-K). Keine Prüfungsleistung: Das Dokument dient
der eigenen Anlageentscheidung. Jede Zahl aus einer Primärquelle ist gegen die
gedruckte Seite ihrer Quelle geprüft; Sekundärquellen sind als solche
gekennzeichnet.

```bash
export SEC_USER_AGENT='Vorname Nachname mail@example.com'
python3 scripts/hole_quellen.py && python3 scripts/edgar_seiten.py
./build.sh          # -> out/analyse.pdf
```

Alles Weitere in [CLAUDE.md](CLAUDE.md); inhaltliche Befunde in
[docs/befunde-und-entscheidungen.md](docs/befunde-und-entscheidungen.md).
