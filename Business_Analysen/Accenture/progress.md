# Progress — Accenture plc (ACN) Investitionsanalyse

Frame: n = 0 (bereits haltbare/kaufbare Position, keine Handlungsoptionen zum Vergleich).
Entscheidung: Kauf / Halten / Meiden für ACN heute (06.10.2026), Alleineigner-Perspektive.
Hurdle: Accenture eigene ROE (Quelle: 10-K FY2025), siehe data/kennzahlen.

Primärquellen (lokal gepinnt, quellen/):
- acn-10k-2025.htm/.md — 10-K FY2025 (GJ bis 31.08.2025, eingereicht 10.10.2025)
- acn-10k-2024.htm/.md — 10-K FY2024
- acn-10k-2023.htm/.md — 10-K FY2023
- acn-def14a-2025.htm/.md — DEF 14A, eingereicht 12.12.2025
- form4_list.txt — erste 100 Form-4-Einreichungen (deckt nur bis 17.02.2026 zurück; nicht die vollen 12 Monate)

Hinweis Seitenangaben: dieses 10-K ist eine Inline-XBRL-HTML-Einreichung ohne gedruckte
Seitenfußzeile im Quelltext (verifiziert: 101 CSS-page-break-Marker ohne begleitende
Seitenzahl). Zitate verwenden daher Item/Abschnitt statt Seitenzahl — siehe Methodik-Hinweis
im Bericht.

## Domains
- domain2_financials: complete -> extrahiert im Hauptthread aus acn-10k-2023/24/25.md + acn-q3fy26-8k-exhibit.md + acn-q4fy26-8k-exhibit.md (0 Luecken bei Kernzahlen, FY2023-Kopfzahl als einzige kleine Luecke)
- domain1_governance: complete -> domain1_governance.md (21 Fakten, 2 Luecken: Form-4-Fenster 06.10.25-16.02.26 fehlt, Detailbeteiligung Direktoren unvollstaendig), confidence high/medium
- domain3_market: complete -> domain3_market.md (20 Fakten, 4 Luecken: Peer-Betriebsmargen, Marktgroesse, Konsens-Verteilung, Segment-Marge), confidence medium
- Rechenmodell: data/analysis.json + cashflow_irr.py, Hurdle 26,7% (ROE FY2026, sourced), break_even/required_growth/required_terminal_value alle berechnet
- Bericht fertig: Accenture_Investitionsanalyse.md, Urteil: nicht kaufen zu 195,05 USD / fuer Altaktionaer halten vertretbar
- GAP CARRIED (naechste Sitzung): Form-4-Fenster Okt25-Feb26 nachbeschaffen falls Insiderkauf-Frage erneut relevant wird; 10-Jahres-Multiple-Historie fuer 7.3 nachbeschaffen
