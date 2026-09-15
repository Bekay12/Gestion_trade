"""Valeurs vérifiées d'Alamos Gold Inc. Une valeur, une source, une page IMPRIMÉE.

Exercice civil. Documents : rapport de gestion annuel 2025 (déposé le 18.02.2026),
rapport de gestion du deuxième trimestre 2026 et états financiers trimestriels
(déposés le 30.07.2026). Montants en millions de dollars américains, coûts et
prix en dollars par once. Page imprimée = numéro de bloc (versant mesuré à 0).
"""
DOKUMENTE = {'MDA25': ('agi-mda-2025', 0), 'MDA26': ('agi-mda-q2-2026', 0),
             'FS26':  ('agi-fs-q2-2026', 0)}

WERTE = {
 # --- Exercice 2025 contre 2024 (rapport de gestion 2025, page 4) ---
 'umsatz_2025': (1808.8, 'MDA25', 4), 'umsatz_2024': (1346.9, 'MDA25', 4),
 'nettoergebnis_2025': (885.8, 'MDA25', 4), 'nettoergebnis_2024': (284.3, 'MDA25', 4),
 'bereinigt_2025': (587.1, 'MDA25', 4), 'bereinigt_2024': (328.9, 'MDA25', 4),
 'fcf_2025': (351.7, 'MDA25', 4), 'fcf_2024': (272.3, 'MDA25', 4),
 'produktion_2025': (545400, 'MDA25', 4), 'produktion_2024': (567000, 'MDA25', 4),
 'goldpreis_2025': (3372, 'MDA25', 4), 'goldpreis_2024': (2379, 'MDA25', 4),
 'cash_kosten_2025': (1077, 'MDA25', 4), 'cash_kosten_2024': (927, 'MDA25', 4),
 'aisc_2025': (1524, 'MDA25', 4), 'aisc_2024': (1252, 'MDA25', 4),
 'op_cashflow_2025': (795.3, 'MDA25', 4), 'op_cashflow_2024': (661.1, 'MDA25', 4),
 # --- Premier semestre 2026 contre 2025 (rapport T2 2026, page 4) ---
 'umsatz_1h2026': (1190.8, 'MDA26', 4), 'umsatz_1h2025': (771.2, 'MDA26', 4),
 'nettoergebnis_1h2026': (461.8, 'MDA26', 4), 'nettoergebnis_1h2025': (174.6, 'MDA26', 4),
 'bereinigt_1h2026': (479.6, 'MDA26', 4), 'bereinigt_1h2025': (203.9, 'MDA26', 4),
 'fcf_1h2026': (245.2, 'MDA26', 4), 'fcf_1h2025': (64.5, 'MDA26', 4),
 'produktion_1h2026': (254500, 'MDA26', 4), 'produktion_1h2025': (262200, 'MDA26', 4),
 'goldpreis_1h2026': (4660, 'MDA26', 4), 'goldpreis_1h2025': (3027, 'MDA26', 4),
 'cash_kosten_1h2026': (1268, 'MDA26', 4), 'cash_kosten_1h2025': (1114, 'MDA26', 4),
 'aisc_1h2026': (1793, 'MDA26', 4), 'aisc_1h2025': (1565, 'MDA26', 4),
 'op_cashflow_1h2026': (474.3, 'MDA26', 4), 'op_cashflow_1h2025': (279.1, 'MDA26', 4),
 # --- Deuxième trimestre 2026 seul ---
 'umsatz_q2_2026': (594.1, 'MDA26', 4), 'umsatz_q2_2025': (438.2, 'MDA26', 4),
 'nettoergebnis_q2_2026': (270.4, 'MDA26', 4),
 'produktion_q2_2026': (130600, 'MDA26', 4), 'produktion_q2_2025': (137200, 'MDA26', 4),
 'goldpreis_q2_2026': (4504, 'MDA26', 4), 'aisc_q2_2026': (1728, 'MDA26', 4),
 'fcf_q2_2026': (143.5, 'MDA26', 4),
 # --- Prévision 2026, relevée (rapport T2 2026, page 6) ---
 'prognose_cash_unten': (1175, 'MDA26', 6), 'prognose_cash_oben': (1275, 'MDA26', 6),
 'prognose_aisc_unten': (1775, 'MDA26', 6), 'prognose_aisc_oben': (1875, 'MDA26', 6),
 # --- Bilan au 30 juin 2026 (états financiers, page 2) ---
 'aktiva_1h2026': (6712.9, 'FS26', 2), 'aktiva_2025': (6384.6, 'FS26', 2),
 'eigenkapital_1h2026': (4804.9, 'FS26', 2), 'eigenkapital_2025': (4445.8, 'FS26', 2),
 'liquiditaet_1h2026': (636.9, 'FS26', 2), 'liquiditaet_2025': (623.1, 'FS26', 2),
 'schulden_1h2026': (1908.0, 'FS26', 2), 'schulden_2025': (1938.8, 'FS26', 2),
}
EXTERN = {
 'kurs': (35.21, 'Yahoo Finance, AGI, Schlusskurs 15.09.2026'),
 'aktien_mio': (418.6, 'Yahoo Finance, sharesOutstanding, abgerufen 15.09.2026'),
 'hoch_3j': (55.33, 'Yahoo Finance, plus haut du 02.03.2026'),
 'tief_3j': (10.92, 'Yahoo Finance, plus bas du 02.10.2023'),
}
