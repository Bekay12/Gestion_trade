"""Valeurs vérifiées de Südzucker AG. Une valeur, une source, une page IMPRIMÉE.

Exercice décalé : 2025/26 court du 1er mars 2025 au 28 février 2026 ; le premier
trimestre 2026/27 va du 1er mars au 31 mai 2026.

VERSANT DE PAGINATION, mesuré sur les 278 pages du rapport annuel :
  page imprimée = page PDF − 2   (couverture 100 %)
  rapport trimestriel : page imprimée = page PDF (couverture 95 %)
Montants en millions d'euros sauf mention.
"""
DOKUMENTE = {'GB': ('szu-annual-report-2025-26', 2), 'Q1': ('szu-q1-2026-27', 0)}

WERTE = {
 # --- Tableau de bord quadriennal (GB, page imprimée 4) ---
 'umsatz_2526': (8352, 'GB', 4), 'umsatz_2425': (9694, 'GB', 4),
 'umsatz_2324': (10289, 'GB', 4), 'umsatz_2223': (9498, 'GB', 4),
 'op_ebitda_2526': (535, 'GB', 4), 'op_ebitda_2425': (723, 'GB', 4),
 'op_ebitda_2324': (1318, 'GB', 4), 'op_ebitda_2223': (1070, 'GB', 4),
 'op_marge_2526': (6.4, 'GB', 4), 'op_marge_2324': (12.8, 'GB', 4),
 'op_ergebnis_2526': (163, 'GB', 4), 'op_ergebnis_2425': (350, 'GB', 4),
 'op_ergebnis_2324': (947, 'GB', 4),
 'nettoergebnis_2526': (-378, 'GB', 4), 'nettoergebnis_2425': (-86, 'GB', 4),
 'nettoergebnis_2324': (648, 'GB', 4), 'nettoergebnis_2223': (529, 'GB', 4),
 'op_cashflow_2526': (462, 'GB', 4), 'op_cashflow_2425': (906, 'GB', 4),
 'op_cashflow_2324': (1073, 'GB', 4), 'op_cashflow_2223': (244, 'GB', 4),
 'investitionen_2526': (450, 'GB', 4), 'investitionen_2425': (574, 'GB', 4),
 'investitionen_2324': (546, 'GB', 4), 'investitionen_2223': (400, 'GB', 4),
 'capital_employed_2526': (6019, 'GB', 4), 'roce_2526': (2.7, 'GB', 4),
 'roce_2425': (5.2, 'GB', 4), 'roce_2324': (13.2, 'GB', 4), 'roce_2223': (9.9, 'GB', 4),
 'bilanzsumme_2526': (8398, 'GB', 4),
 'nettoschuld_2526': (1750, 'GB', 4), 'nettoschuld_2425': (1654, 'GB', 4),
 'nettoschuld_2324': (1795, 'GB', 4),
 'verschuldungsgrad_2526': (3.3, 'GB', 4), 'verschuldungsgrad_2324': (1.4, 'GB', 4),
 'ek_quote_2526': (41.7, 'GB', 4),
 'marktkap_2526': (2068, 'GB', 4), 'kurs_ende_2526': (10.13, 'GB', 4),
 'eps_2526': (-1.92, 'GB', 4), 'eps_2425': (-0.54, 'GB', 4), 'eps_2324': (2.72, 'GB', 4),
 'dividende_2526': (0.00, 'GB', 4), 'dividende_2425': (0.20, 'GB', 4),
 'dividende_2324': (0.90, 'GB', 4),
 'mitarbeiter_2526': (18188, 'GB', 4),
 # --- Bilan (GB, page imprimée 35) ---
 'ek_aktionaere_2526': (2147, 'GB', 35), 'hybridkapital_2526': (715, 'GB', 35),
 'minderheiten_2526': (643, 'GB', 35), 'ek_gesamt_2526': (3505, 'GB', 35),
 'aktiva_gesamt_2526': (8398, 'GB', 35),
 # --- Premier trimestre 2026/27 (Q1, page imprimée 3) ---
 'umsatz_q1_2627': (2058, 'Q1', 3), 'umsatz_q1_2526': (2153, 'Q1', 3),
 'op_ebitda_q1_2627': (135, 'Q1', 3), 'op_ebitda_q1_2526': (96, 'Q1', 3),
 'op_marge_q1_2627': (6.6, 'Q1', 3), 'op_marge_q1_2526': (4.5, 'Q1', 3),
 'op_ergebnis_q1_2627': (62, 'Q1', 3), 'op_ergebnis_q1_2526': (22, 'Q1', 3),
 'nettoergebnis_q1_2627': (20, 'Q1', 3), 'nettoergebnis_q1_2526': (-35, 'Q1', 3),
 'nettoschuld_q1_2627': (1855, 'Q1', 3), 'ek_q1_2627': (3583, 'Q1', 3),
 # --- Prévision 2026/27 (Q1, page imprimée 1) ---
 'prognose_ebitda_unten': (480, 'Q1', 1), 'prognose_ebitda_oben': (680, 'Q1', 1),
}
EXTERN = {
 'kurs': (12.58, 'Yahoo Finance, SZU.DE, Schlusskurs 15.09.2026'),
 'aktien_mio': (204.2, 'Marktkapitalisierung 2.068 / Kurs 10,13 zum 28.02.2026, GB S. 4'),
 'beta': (0.20, 'Yahoo Finance, abgerufen 15.09.2026'),
}
