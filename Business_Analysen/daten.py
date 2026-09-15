"""Gepruefte Werte der drei Dual Champions. Einzige Wahrheitsquelle.

Jeder Wert: (Wert, Quelle, GEDRUCKTE Seite). werkzeug.pruefe() laeuft darueber
und scheitert, wenn ein Wert im Text der zitierten Seite fehlt.

SEITENVERSATZ, gemessen ueber alle Seiten jedes Dokuments:

  Neste GB 2025        gedruckt = PDF            (99 % der Seiten belegt)
  Neste H1 2026        gedruckt = PDF - 1        (96 %)
  GEA GB 2025          gedruckt = PDF            (64 %, Rest ESRS-Nummerierung)
  GEA H1 2026          gedruckt = PDF
  Prysmian GB 2025     KEINE gedruckten Nummern im Textlayer -> zitiert wird
                       die PDF-Seite, hier ausdruecklich erklaert
  Prysmian H1 2026     gedruckt = PDF - 1        (98 %)
  Prysmian Q2-Mitteilung  keine verlaessliche Nummerierung -> PDF-Seite
"""

# Quelle -> (Dateistamm im Seitenindex, Versatz PDF minus gedruckt)
DOKUMENTE = {
    'Neste': {
        'GB2025':  ('neste-annual-report-2025', 0),
        'H1-2026': ('neste-h1-2026-report', 1),
    },
    'GEA_Group': {
        'GB2025':  ('gea-annual-report-2025', 0),
        'H1-2026': ('gea-h1-2026-report', 0),
    },
    'Prysmian': {
        'GB2025':  ('prysmian-annual-report-2025', 0),   # PDF-Seite = Zitat
        'H1-2026': ('prysmian-h1-2026-report', 1),
        'Q2-Mitteilung': ('prysmian-q2-1h-2026-pressemitteilung', 0),
    },
}

WERTE = {
'Neste': {
    'umsatz_2025':        (19016, 'GB2025', 240),
    'umsatz_2024':        (20635, 'GB2025', 240),
    'ebitda_2025':        (1438,  'GB2025', 240),
    'ebitda_2024':        (1005,  'GB2025', 240),
    'ebitda_verg_2025':   (1683,  'GB2025', 240),
    'ebitda_verg_2024':   (1252,  'GB2025', 240),
    'betriebsergebnis_2025': (503, 'GB2025', 240),
    'betriebsergebnis_2024': (25,  'GB2025', 240),
    'nettoergebnis_2025': (144,   'GB2025', 240),
    'nettoergebnis_2024': (-95,   'GB2025', 240),
    'eps_2025':           (0.19,  'GB2025', 240),
    'roe_2025':           (2.0,   'GB2025', 240),
    'roace_2025':         (5.3,   'GB2025', 240),
    'roace_2024':         (2.5,   'GB2025', 240),
    'eigenkapital_2025':  (7314,  'GB2025', 240),
    'eigenkapital_2024':  (7417,  'GB2025', 240),
    'ek_je_aktie_2025':   (9.52,  'GB2025', 240),
    'nettoschuld_2025':   (3817,  'GB2025', 240),
    'nettoschuld_2024':   (4192,  'GB2025', 240),
    'verschuldungsgrad_2025': (34.3, 'GB2025', 240),
    'op_cashflow_2025':   (1747,  'GB2025', 240),
    'op_cashflow_2024':   (1154,  'GB2025', 240),
    'investitionen_2025': (1253,  'GB2025', 240),
    'dividende_2025':     (0.20,  'GB2025', 240),
    'ausschuettungsquote_2025': (106.6, 'GB2025', 240),
    'kurs_ende_2025':     (19.41, 'GB2025', 240),
    'kurs_tief_2025':     (6.79,  'GB2025', 240),
    'kurs_hoch_2025':     (20.22, 'GB2025', 240),
    'marktkap_ende_2025': (14930, 'GB2025', 240),
    'freier_cashflow_2025': (759, 'GB2025', 79),
    'freier_cashflow_2024': (-341,'GB2025', 79),
    'programm_lauf_2025': (376,   'GB2025', 79),
    # Halbjahr 2026
    'umsatz_1h2026':      (11150, 'H1-2026', 3),
    'umsatz_1h2025':      (9528,  'H1-2026', 3),
    'ebitda_verg_1h2026': (2064,  'H1-2026', 3),
    'ebitda_verg_1h2025': (551,   'H1-2026', 3),
    'nettoergebnis_1h2026': (1298,'H1-2026', 3),
    'eps_1h2026':         (1.69,  'H1-2026', 3),
    'eigenkapital_1h2026':(8458,  'H1-2026', 3),
    'nettoschuld_1h2026': (3613,  'H1-2026', 3),
    'roace_ltm_2026':     (18.0,  'H1-2026', 3),
    'ek_je_aktie_1h2026': (11.01, 'H1-2026', 3),
    'verschuldungsgrad_1h2026': (29.9, 'H1-2026', 3),
    'op_cashflow_1h2026': (993,   'H1-2026', 3),
    'investitionen_1h2026': (405, 'H1-2026', 3),
    'fcf_1h2026':         (450,   'H1-2026', 1),
    'fcf_1h2025':         (0,     'H1-2026', 1),
    'programm_lauf_2026': (594,   'H1-2026', 1),
    'marge_rp_q2_2026':   (1223,  'H1-2026', 1),
    'marge_rp_q2_2025':   (361,   'H1-2026', 1),
},
'GEA_Group': {
    'auftragseingang_2025': (5924.1, 'GB2025', 2),
    'auftragseingang_2024': (5553.0, 'GB2025', 2),
    'auftragsbestand_2025': (3339.2, 'GB2025', 2),
    'book_to_bill_2025':  (1.08,  'GB2025', 2),
    'umsatz_2025':        (5495.4,'GB2025', 2),
    'umsatz_2024':        (5422.1,'GB2025', 2),
    'ebitda_vor_rest_2025': (907.4,'GB2025', 2),
    'ebitda_vor_rest_2024': (837.3,'GB2025', 2),
    'ebitda_2025':        (859.0, 'GB2025', 2),
    'ebitda_2024':        (776.7, 'GB2025', 2),
    'nettoergebnis_2025': (414.0, 'GB2025', 2),
    'nettoergebnis_2024': (385.0, 'GB2025', 2),
    'roce_2025':          (36.2,  'GB2025', 2),
    'roce_2024':          (33.8,  'GB2025', 2),
    'op_cashflow_2025':   (726.7, 'GB2025', 2),
    'invest_cashflow_2025': (-214.8,'GB2025', 2),
    'freier_cashflow_2025': (511.8,'GB2025', 2),
    'freier_cashflow_2024': (504.8,'GB2025', 2),
    'eigenkapital_2025':  (2453.4,'GB2025', 2),
    'eigenkapital_2024':  (2424.1,'GB2025', 2),
    'eigenkapitalquote_2025': (40.3,'GB2025', 2),
    'nettoliquiditaet_2025': (378.9,'GB2025', 2),
    'nettoliquiditaet_2024': (343.5,'GB2025', 2),
    'eps_2025':           (2.54,  'GB2025', 2),
    'eps_2024':           (2.30,  'GB2025', 2),
    'marktkap_ende_2025': (9.4,   'GB2025', 2),
    'kurs_ende_2025':     (57.80, 'GB2025', 2),
    'kurs_ende_2024':     (47.82, 'GB2025', 2),
    'mitarbeiter_2025':   (18628, 'GB2025', 2),
    # Halbjahr 2026
    'auftragseingang_1h2026': (2949.0,'H1-2026', 2),
    'auftragseingang_1h2025': (2724.0,'H1-2026', 2),
    'auftragsbestand_1h2026': (3540.4,'H1-2026', 2),
    'umsatz_1h2026':      (2715.6,'H1-2026', 2),
    'umsatz_1h2025':      (2570.2,'H1-2026', 2),
    'ebitda_vor_rest_1h2026': (456.5,'H1-2026', 2),
    'ebitda_vor_rest_1h2025': (415.0,'H1-2026', 2),
    'nettoergebnis_1h2026': (221.5,'H1-2026', 2),
    'nettoergebnis_1h2025': (201.4,'H1-2026', 2),
    'roce_1h2026':        (36.8,  'H1-2026', 2),
    'freier_cashflow_1h2026': (-39.2,'H1-2026', 2),
    'freier_cashflow_1h2025': (-10.8,'H1-2026', 2),
    'eigenkapital_1h2026':(2503.0,'H1-2026', 2),
    'eps_1h2026':         (0.75,  'H1-2026', 2),
    'nettoliquiditaet_1h2026': (70.9,'H1-2026', 2),
},
'Prysmian': {
    'umsatz_2025':        (19650, 'GB2025', 14),
    'umsatz_2024':        (17026, 'GB2025', 14),
    'ebitda_ber_2025':    (2398,  'GB2025', 14),
    'ebitda_ber_2024':    (1927,  'GB2025', 14),
    'nettoergebnis_2025': (1294,  'GB2025', 14),
    'nettoergebnis_2024': (748,   'GB2025', 14),
    'eps_verw_2025':      (4.30,  'GB2025', 14),
    'eps_verw_2024':      (2.52,  'GB2025', 14),
    'nettoschuld_2025':   (3097,  'GB2025', 14),
    'nettoschuld_2024':   (4296,  'GB2025', 14),
    'investiertes_kapital_2025': (10056,'GB2025', 14),
    'ebitda_2025':        (2688,  'GB2025', 365),
    'ebitda_2024':        (1754,  'GB2025', 365),
    'op_cashflow_2025':   (2165,  'GB2025', 365),
    'op_cashflow_2024':   (1933,  'GB2025', 365),
    'akquisitionen_2025': (-1069, 'GB2025', 365),
    'akquisitionen_2024': (-4126, 'GB2025', 365),
    'fcf_levered_2025':   (773,   'GB2025', 365),
    'fcf_levered_2024':   (-3120, 'GB2025', 365),
    'fcf_levered_2023':   (720,   'GB2025', 365),
    'eigenkapital_2025':  (6474,  'GB2025', 400),
    'eigenkapital_2024':  (5087,  'GB2025', 400),
    'eigenkapital_gesamt_2025': (6680,'GB2025', 400),
    # Halbjahr 2026
    'umsatz_1h2026':      (11239, 'H1-2026', 12),
    'umsatz_1h2025':      (9654,  'H1-2026', 12),
    'ebitda_ber_1h2026':  (1331,  'H1-2026', 12),
    'ebitda_ber_1h2025':  (1132,  'H1-2026', 12),
    'nettoergebnis_1h2026': (584, 'H1-2026', 12),
    'nettoergebnis_1h2025': (435, 'H1-2026', 12),
    'nettoschuld_1h2026': (4079,  'H1-2026', 12),
    'nettoschuld_1h2025': (4694,  'H1-2026', 12),
    'investiertes_kapital_1h2026': (11663,'H1-2026', 12),
    'fcf_ltm_2026':       (978,   'Q2-Mitteilung', 2),
    'fcf_ltm_2025':       (979,   'Q2-Mitteilung', 2),
},
}

# Externe Quellen: Kurse und Analystenkonsens. KEINE Seitenpruefung moeglich,
# deshalb getrennt gefuehrt (Regel 5: Sekundaerquellen nie mit den
# seitengeprueften Primaerwerten vermischen).
EXTERN = {
'Neste': {
    'kurs': (32.89, 'Yahoo Finance, NESTE.HE, Schlusskurs 11.09.2026'),
    'aktien_mio': (769.2, 'abgeleitet: Marktkapitalisierung 14.930 / Kurs 19,41, beide GB2025 S. 240'),
    'konsens_anzahl': (19, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_urteil': ('Outperform', 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_ziel': (33.34, 'MarketScreener, Mittelwert, Schlusskurs der Seite 31,72'),
    'konsens_hoch': (40.00, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_tief': (18.40, 'MarketScreener, abgerufen 15.09.2026'),
},
'GEA_Group': {
    'kurs': (63.85, 'Yahoo Finance, G1A.DE, Schlusskurs 11.09.2026'),
    'aktien_mio': (162.6, 'abgeleitet: Marktkapitalisierung 9,4 Mrd / Kurs 57,80, beide GB2025 S. 2'),
    'konsens_anzahl': (17, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_urteil': ('Outperform', 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_ziel': (70.26, 'MarketScreener, Mittelwert, Schlusskurs der Seite 63,55'),
    'konsens_hoch': (78.00, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_tief': (53.00, 'MarketScreener, abgerufen 15.09.2026'),
},
'Prysmian': {
    'kurs': (127.25, 'Yahoo Finance, PRY.MI, Schlusskurs 11.09.2026'),
    'aktien_mio': (271.6, 'Yahoo Finance, Marktkapitalisierung 34,56 Mrd / Kurs 127,25, abgerufen 15.09.2026'),
    'konsens_anzahl': (16, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_urteil': ('Buy', 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_ziel': (156.06, 'MarketScreener, Mittelwert, Schlusskurs der Seite 118,25'),
    'konsens_hoch': (181.00, 'MarketScreener, abgerufen 15.09.2026'),
    'konsens_tief': (90.00, 'MarketScreener, abgerufen 15.09.2026'),
},
}
