# Fortschritt

Rahmen (2026-09-17): Sprache Deutsch; LaTeX-Projekt nach dem Brown-&-Brown-Muster;
zwei getrennte Voten (AZN, RDY), keine Mittelkonkurrenz; Huerde = Branchenindex
(Sekundaerquelle, mit Abrufdatum).

## Quellen
- AZN: Annual Report & Form 20-F Information 2019-2025 (PDF, astrazeneca.com),
  Halbjahresbericht 2026 (Form 6-K, 27.07.2026). Seiten gelesen, nicht geschaetzt.
  AR 2019/2020 nur 3 bzw. 5 Fusszeilen lesbar (Schrift ohne Unicode-Zuordnung) -
  NICHT als Belegquelle verwenden; Reihe beginnt deshalb bei FY2019 aus AR 2021.
- RDY: Form 20-F FY2019-FY2026 (EDGAR), Quartalsbericht zum 30.06.2026 (Form 6-K).
- Zerlegung: scripts/seiten_zerlegen.py, Bericht "gelesen/ergaenzt" je Datei.
  rdy-20f-2023 (60 ergaenzt) und rdy-20f-2024 (102 ergaenzt) nur nachrangig belegen.

## Stand
- Quellenbeschaffung: fertig
- Seitenzerlegung: fertig, alle Dateien lueckenlos
- Recherche: 4 Teilgebiete fertig (docs/azn-unternehmen.md, azn-strategie.md,
  rdy-unternehmen.md, rdy-strategie.md)
- Zahlenschicht: 238 Rohwerte, alle auf ihrer gedruckten Seite belegt (RC 0)
- Kettenpruefung: AZN 88 Gleichungen, RDY 60 Gleichungen, 0 Brueche
- Teil 6 gerechnet: Kapazitaet + Anlegerrechnung je Titel, zwei Endwertvarianten
- GAP UEBERNOMMEN: RDY nennt den Umsatzanteil des Lenalidomid-Geschaefts
  nirgends. Die zentrale Groesse des Falls ist nicht offengelegt; Teil 7 muss das
  ausdruecklich tragen.
- GAP UEBERNOMMEN: AZN FY2021-Prognose nicht belegbar (stuende im AR2020, dessen
  Fusszeilen unlesbar sind). Prognosetreue deshalb ueber vier statt fuenf Jahre.
- Dokument fertig: 43 Seiten, `out/analyse.pdf`
- Alle fuenf Wachhunde gruen: 355 Werte belegt, 148 Kettengleichungen,
  92 Fussnotenzitationen ohne Duplikat, keine Zahl im Fliesstext, 43 Tests
- Voten: AZN beobachten (Schwelle 152,85 USD), RDY meiden (nicht bewertbar)
- Inhaltliche Befunde: docs/befunde-und-entscheidungen.md
