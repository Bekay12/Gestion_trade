"""Valeurs vérifiées d'EVN AG. Une valeur, une source, une page IMPRIMÉE.

Exercice décalé : l'exercice 2024/25 court du 1er octobre 2024 au 30 septembre
2025 ; les neuf premiers mois de 2025/26 vont du 1er octobre 2025 au 30 juin 2026.
Page imprimée = page PDF pour les trois documents (couverture mesurée à 100 %).
Montants en millions d'euros sauf mention.
"""
DOKUMENTE = {'JFB': ('evn-jahresfinanzbericht-2024-25', 0),
             'IR':  ('evn-ergebnis-2024-25', 0),
             'AB':  ('evn-aktionaersbrief-q1-3-2025-26', 0)}

WERTE = {
 # --- Compte de résultat 2024/25 (JFB p. 23) ---
 'umsatz_2425':        (3000.0, 'JFB', 23), 'umsatz_2324':        (2889.2, 'JFB', 23),
 'ebitda_2425':        (909.1,  'JFB', 23), 'ebitda_2324':        (762.9,  'JFB', 23),
 'abschreibungen_2425':(360.1,  'JFB', 23), 'wertminderung_2425': (58.2,   'JFB', 23),
 'ebit_2425':          (490.9,  'JFB', 23), 'ebit_2324':          (404.3,  'JFB', 23),
 'finanzergebnis_2425':(83.6,   'JFB', 23), 'finanzergebnis_2324':(145.6,  'JFB', 23),
 'ebt_2425':           (574.4,  'JFB', 23), 'ebt_2324':           (549.9,  'JFB', 23),
 'steuern_2425':       (65.6,   'JFB', 23), 'steuern_2324':       (32.1,   'JFB', 23),
 # --- Bilan au 30.09.2025 (JFB p. 24) ---
 'ek_aktionaere_2425': (6328.3, 'JFB', 24), 'ek_aktionaere_2324': (6414.8, 'JFB', 24),
 'ek_gesamt_2425':     (6658.8, 'JFB', 24), 'ek_gesamt_2324':     (6730.6, 'JFB', 24),
 'bilanzsumme_2425':   (11030.7,'JFB', 24), 'bilanzsumme_2324':   (10913.6,'JFB', 24),
 'finanzschuld_lang':  (1199.9, 'JFB', 24), 'finanzschuld_kurz':  (22.9,   'JFB', 24),
 'liquiditaet_2425':   (89.8,   'JFB', 24),
 # --- Flux de trésorerie (JFB p. 26) ---
 'op_cashflow_2425':   (935.2,  'JFB', 26), 'op_cashflow_2324':   (1166.7, 'JFB', 26),
 # --- Communiqué de résultats (IR p. 2) ---
 'konzernergebnis_2425': (436.7, 'IR', 2),
 'nettoverschuldung_2425': (1155.9, 'IR', 2), 'nettoverschuldung_2324': (1129.3, 'IR', 2),
 'dividende_2425':     (0.90,   'IR', 2),
 'verbund_div_2024':   (2.80,   'IR', 2),    'verbund_div_2023':   (4.15,   'IR', 2),
 'prognose_unten':     (430,    'IR', 3),    'prognose_oben':      (480,    'IR', 3),
 'erneuerbar_mw':      (980,    'IR', 3),
 'stromerzeugung_gwh': (2915,   'IR', 3),
 # --- Neuf premiers mois 2025/26 (AB p. 2) ---
 'umsatz_9m2526':      (2433.1, 'AB', 2),  'umsatz_9m2425':      (2360.4, 'AB', 2),
 'ebitda_9m2526':      (748.5,  'AB', 2),  'ebitda_9m2425':      (713.6,  'AB', 2),
 'ebit_9m2526':        (460.4,  'AB', 2),  'ebit_9m2425':        (447.1,  'AB', 2),
 'ebt_9m2526':         (576.3,  'AB', 2),  'ebt_9m2425':         (540.5,  'AB', 2),
 'konzernergebnis_9m2526': (525.1,'AB', 2), 'konzernergebnis_9m2425': (434.7,'AB', 2),
 'eps_9m2526':         (2.94,   'AB', 2),  'eps_9m2425':         (2.44,   'AB', 2),
 'ebit_q3_2526':       (97.5,   'AB', 2),  'ebit_q3_2425':       (111.6,  'AB', 2),
 'ebitda_q3_2526':     (195.1,  'AB', 2),  'ebitda_q3_2425':     (200.9,  'AB', 2),
}
EXTERN = {
 'kurs': (28.45, 'Yahoo Finance, EVN.VI, Schlusskurs 14.09.2026'),
 'aktien_mio': (178.3, 'Yahoo Finance, sharesOutstanding, abgerufen 15.09.2026'),
 'beta': (0.65, 'Yahoo Finance, abgerufen 15.09.2026'),
}
