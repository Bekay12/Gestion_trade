"""Valeurs vérifiées des trois inflexions. Une valeur, une source, une page IMPRIMÉE.

Unités : Honeywell en millions USD, ESAB et Pool Corp en milliers USD (unité du
dépôt, jamais convertie à la volée).
"""
DOKUMENTE = {
 'Honeywell': {'10K2025': ('hon-10k-2025', 0), '10Q1H26': ('hon-10q-1h2026', 0)},
 'ESAB':      {'10K2025': ('esab-10k-2025', 0), '10Q1H26': ('esab-10q-1h2026', 0)},
 'Pool_Corp': {'10K2025': ('pool-10k-2025', 0), '10Q1H26': ('pool-10q-1h2026', 0)},
}
# bloc EDGAR de chaque page imprimée citée
BLOCS = {
 'Honeywell': {55: 57, 57: 59, 58: 60, 3: 5, 5: 7, 6: 8},
 'ESAB':      {49: 51, 51: 53, 53: 55, 2: 3, 4: 5, 7: 8},
 'Pool_Corp': {50: 53, 52: 55, 53: 56, 13: 16, 14: 17},
}
WERTE = {
'Honeywell': {   # millions USD
  'umsatz_2025': (37442, '10K2025', 55), 'umsatz_2024': (34717, '10K2025', 55),
  'umsatz_2023': (33009, '10K2025', 55),
  'nettoergebnis_2025': (4729, '10K2025', 55), 'nettoergebnis_2024': (5705, '10K2025', 55),
  'nettoergebnis_2023': (5658, '10K2025', 55),
  'goodwill_abschreibung_2025': (724, '10K2025', 55),
  'op_cashflow_2025': (6408, '10K2025', 58), 'op_cashflow_2024': (6097, '10K2025', 58),
  'op_cashflow_2023': (5340, '10K2025', 58),
  'investitionen_2025': (986, '10K2025', 58), 'investitionen_2024': (871, '10K2025', 58),
  'investitionen_2023': (741, '10K2025', 58),
  'dividenden_2025': (2976, '10K2025', 58),
  'eigenkapital_2025': (15030, '10K2025', 57), 'eigenkapital_2024': (19154, '10K2025', 57),
  'liquiditaet_2025': (12487, '10K2025', 57), 'geldmarkt_2025': (5893, '10K2025', 57),
  'faellige_schuld_2025': (1546, '10K2025', 57), 'langfrist_schuld_2025': (27141, '10K2025', 57),
  'eps_verwaessert_2025': (7.36, '10K2025', 55), 'eps_verwaessert_2024': (8.71, '10K2025', 55),
  'eps_verwaessert_2023': (8.47, '10K2025', 55),
  'umsatz_1h2026': (18862, '10Q1H26', 3), 'umsatz_1h2025': (18247, '10Q1H26', 3),
  'nettoergebnis_1h2026': (6481, '10Q1H26', 3), 'nettoergebnis_1h2025': (3036, '10Q1H26', 3),
  'op_cashflow_1h2026': (626, '10Q1H26', 6), 'op_cashflow_1h2025': (1916, '10Q1H26', 6),
  'investitionen_1h2026': (538, '10Q1H26', 6),
  'eigenkapital_1h2026': (18857, '10Q1H26', 5), 'liquiditaet_1h2026': (8751, '10Q1H26', 5),
  'langfrist_schuld_1h2026': (26228, '10Q1H26', 5),
},
'ESAB': {        # milliers USD
  'umsatz_2025': (2842555, '10K2025', 49), 'umsatz_2024': (2740803, '10K2025', 49),
  'umsatz_2023': (2774766, '10K2025', 49),
  'nettoergebnis_2025': (226766, '10K2025', 49), 'nettoergebnis_2024': (264842, '10K2025', 49),
  'nettoergebnis_2023': (205285, '10K2025', 49),
  'op_cashflow_2025': (260567, '10K2025', 53), 'op_cashflow_2024': (355399, '10K2025', 53),
  'op_cashflow_2023': (330494, '10K2025', 53),
  'investitionen_2025': (47287, '10K2025', 53), 'investitionen_2024': (51779, '10K2025', 53),
  'investitionen_2023': (48178, '10K2025', 53),
  'eigenkapital_2025': (2166040, '10K2025', 51), 'eigenkapital_2024': (1769004, '10K2025', 51),
  'liquiditaet_2025': (185863, '10K2025', 51), 'langfrist_schuld_2025': (1232540, '10K2025', 51),
  'umsatz_1h2026': (1553224, '10Q1H26', 2), 'umsatz_1h2025': (1393724, '10Q1H26', 2),
  'nettoergebnis_1h2026': (79998, '10Q1H26', 2), 'nettoergebnis_1h2025': (134246, '10Q1H26', 2),
  'op_cashflow_1h2026': (80338, '10Q1H26', 7), 'op_cashflow_1h2025': (82037, '10Q1H26', 7),
  'investitionen_1h2026': (31448, '10Q1H26', 7),
  'eigenkapital_1h2026': (2538681, '10Q1H26', 4), 'liquiditaet_1h2026': (217491, '10Q1H26', 4),
  'langfrist_schuld_1h2026': (2391350, '10Q1H26', 4),
},
'Pool_Corp': {   # milliers USD
  'umsatz_2025': (5289396, '10K2025', 50), 'umsatz_2024': (5310953, '10K2025', 50),
  'umsatz_2023': (5541595, '10K2025', 50),
  'bruttoergebnis_2025': (1572458, '10K2025', 50),
  'betriebsergebnis_2025': (580204, '10K2025', 50), 'betriebsergebnis_2024': (617204, '10K2025', 50),
  'betriebsergebnis_2023': (746567, '10K2025', 50),
  'nettoergebnis_2025': (406404, '10K2025', 50), 'nettoergebnis_2024': (434325, '10K2025', 50),
  'nettoergebnis_2023': (523229, '10K2025', 50),
  'op_cashflow_2025': (365850, '10K2025', 53), 'op_cashflow_2024': (659186, '10K2025', 53),
  'op_cashflow_2023': (888229, '10K2025', 53),
  'investitionen_2025': (56334, '10K2025', 53), 'investitionen_2024': (59476, '10K2025', 53),
  'investitionen_2023': (60096, '10K2025', 53),
  'eigenkapital_2025': (1185229, '10K2025', 52), 'eigenkapital_2024': (1273465, '10K2025', 52),
  'liquiditaet_2025': (104963, '10K2025', 52),
  'umsatz_1h2026': (2960952, '10Q1H26', 13), 'umsatz_1h2025': (2856056, '10Q1H26', 13),
},
}
EXTERN = {
 # ATTENTION : yfinance donnait 316,9 M d'actions (capitalisation 63,83 Md / cours).
# La page de garde du 10-K 2025 indique 635 675 701 actions, et le résultat par
# action dilué de 7,36 USD sur 4 729 M USD confirme ~642 M. Le chiffre de
# yfinance est faux d'un facteur deux ; c'est celui du dépôt qui est retenu.
'Honeywell': {'kurs': (201.38, 'Yahoo Finance, HON, Schlusskurs 14.09.2026'),
              'aktien_mio': (635.68, "10-K 2025, page de garde : 635.675.701 actions")},
 'ESAB':      {'kurs': (67.52, 'Yahoo Finance, ESAB, 14.09.2026'), 'aktien_mio': (62.1, 'Yahoo Finance')},
 'Pool_Corp': {'kurs': (171.76, 'Yahoo Finance, POOL, 14.09.2026'), 'aktien_mio': (36.3, 'Yahoo Finance')},
}
