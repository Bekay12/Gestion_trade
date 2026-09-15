"""Einzige Wahrheitsquelle fuer alle Zahlen der Yara-Analyse.

Jeder Wert steht genau einmal, mit Quelle und GEDRUCKTER Seite. pruefe.py laeuft
ueber diese Tabelle und scheitert, wenn ein Wert auf der zitierten Seite nicht
vorkommt (Regel 1 der Skill: 23 von 176 Zitaten waren im Referenzfall beim
ersten Durchgang falsch, ausnahmslos die aus dem Kontext erschlossenen).

Schluessel -> (Wert, Quelle, gedruckte Seite)
Quelle 'GB2025' = Yara Annual Report 2025 ; 'Q2-2026' = Yara Second-Quarter Report 2026
"""

WERTE = {
    # --- Erfolgsrechnung Konzern -------------------------------------------
    'umsatz_2025':            (15715, 'GB2025', 17),
    'umsatz_2024':            (13934, 'GB2025', 17),
    'ebitda_2025':            (2754,  'GB2025', 17),
    'ebitda_2024':            (1889,  'GB2025', 17),
    'ebitda_bsp_2025':        (2803,  'GB2025', 17),
    'ebitda_bsp_2024':        (2051,  'GB2025', 17),
    'betriebsergebnis_2025':  (1571,  'GB2025', 17),
    'betriebsergebnis_2024':  (686,   'GB2025', 17),
    'nettoergebnis_2025':     (1372,  'GB2025', 17),
    'nettoergebnis_2024':     (15,    'GB2025', 17),
    'eps_2025':               (5.37,  'GB2025', 17),
    'eps_ber_2025':           (4.42,  'GB2025', 17),
    'eps_ber_2024':           (1.73,  'GB2025', 17),
    'waehrungsgewinn_2025':   (383,   'GB2025', 17),
    'waehrungsverlust_2024':  (321,   'GB2025', 17),
    'cf_operativ_2025':       (1894,  'GB2025', 17),
    'cf_operativ_2024':       (1286,  'GB2025', 17),
    'cf_invest_2025':         (906,   'GB2025', 17),
    'cf_invest_2024':         (1080,  'GB2025', 17),
    'roic_2025':              (10.7,  'GB2025', 17),
    'roic_2024':              (5.0,   'GB2025', 17),
    'nettoschuld_ebitda_2025':(1.17,  'GB2025', 17),
    'nettoschuld_ek_2025':    (0.37,  'GB2025', 17),

    # --- Mengen und Energie -------------------------------------------------
    'lieferungen_2025':       (32061, 'GB2025', 17),
    'lieferungen_2024':       (31159, 'GB2025', 17),
    'gaskosten_global_2025':  (10.0,  'GB2025', 17),
    'gaskosten_europa_2025':  (13.2,  'GB2025', 17),

    # --- Bilanz -------------------------------------------------------------
    'eigenkapital_2025':      (8743,  'GB2025', 192),
    'eigenkapital_2024':      (7003,  'GB2025', 192),
    'ek_anteilseigner_2025':  (8724,  'GB2025', 192),
    'ek_anteilseigner_2024':  (6988,  'GB2025', 192),
    'aktien_ausstehend':      (254725627, 'GB2025', 192),
    'nettoschuld_2025':       (3271,  'GB2025', 319),
    'nettoschuld_2024':       (3730,  'GB2025', 319),

    # --- Aktie --------------------------------------------------------------
    'kurs_ende_2025':         (414.00, 'GB2025', 24),
    'kurs_ende_2024':         (301,    'GB2025', 24),
    'kurs_hoch_2025':         (416.90, 'GB2025', 24),
    'kurs_tief_2025':         (297.10, 'GB2025', 24),
    'dividende_2025':         (22,     'GB2025', 24),
    'dividende_2024':         (5,      'GB2025', 24),
    'dividendenrendite_2025': (5.3,    'GB2025', 24),
    'tsr_2025':               (57.31,  'GB2025', 24),
    'marktkap_ende_2025':     (105.5,  'GB2025', 24),
    'adr_kurs_ende_2025':     (20.40,  'GB2025', 25),
    'staatsanteil':           (36.2,   'GB2025', 25),
    'folketrygdfondet':       (7.9,    'GB2025', 25),
    'aktionaere_anzahl':      (53625,  'GB2025', 25),
    'auslandsanteil':         (37.1,   'GB2025', 25),
    'analysten_anzahl':       (23,     'GB2025', 26),

    # --- Quartal 2026 -------------------------------------------------------
    'umsatz_1h2026':          (8687,  'Q2-2026', 1),
    'umsatz_1h2025':          (7595,  'Q2-2026', 1),
    'ebitda_bsp_1h2026':      (1802,  'Q2-2026', 1),
    'ebitda_bsp_1h2025':      (1290,  'Q2-2026', 1),
    'ebitda_bsp_q2_2026':     (906,   'Q2-2026', 1),
    'ebitda_bsp_q2_2025':     (652,   'Q2-2026', 1),
    'nettoergebnis_1h2026':   (872,   'Q2-2026', 1),
    'nettoergebnis_1h2025':   (708,   'Q2-2026', 1),
    'cf_operativ_1h2026':     (1206,  'Q2-2026', 1),
    'cf_invest_1h2026':       (339,   'Q2-2026', 1),
    'roic_1h2026':            (14.3,  'Q2-2026', 1),
    'roic_1h2025':            (7.0,   'Q2-2026', 1),
    'lieferungen_q2_2026':    (7108,  'Q2-2026', 1),
    'lieferungen_q2_2025':    (8269,  'Q2-2026', 1),
    'gaskosten_europa_q2_2026':(15.8, 'Q2-2026', 1),
    'nettoschuld_1h2026':     (3067,  'Q2-2026', 34),
    'gca_kaufpreis':          (1.3,   'Q2-2026', 26),
    'gca_kreditlinie':        (1400,  'Q2-2026', 26),
    'gca_kapazitaet':         (1.3,   'Q2-2026', 26),
    'programm_2027':          (200,   'Q2-2026', 8),
    'programm_2030':          (150,   'Q2-2026', 8),
}

# Werte ohne Seitenbeleg: externe Quellen mit Abrufdatum (Regel 1, zweiter Teil).
EXTERN = {
    'kurs_aktuell_nok': (460.90, 'Yahoo Finance, YAR.OL, Schlusskurs', '14.09.2026'),
    'usd_nok_aktuell':  (9.3237, 'Yahoo Finance, NOK=X, Schlusskurs',  '14.09.2026'),
    # Analystenkonsens: SEKUNDAERquellen. Yara selbst veroeffentlicht weder
    # Namen noch Ratings noch Kursziele (GB2025 S. 26 nennt nur die Anzahl).
    # MarketScreener und Investing.com stimmen auf zwei Nachkommastellen
    # ueberein, sie beziehen also mutmasslich denselben Datenlieferanten; das
    # sind KEINE zwei unabhaengigen Bestaetigungen.
    'konsens_anzahl':   (19, 'MarketScreener + Investing.com', '14.09.2026'),
    'konsens_ziel':     (474.47, 'MarketScreener + Investing.com, Mittelwert NOK', '14.09.2026'),
    'konsens_hoch':     (580.00, 'MarketScreener + Investing.com, hoechstes Ziel NOK', '14.09.2026'),
    'konsens_tief':     (380.00, 'MarketScreener + Investing.com, niedrigstes Ziel NOK', '14.09.2026'),
    'konsens_kauf':     (5,  'Investing.com, Empfehlungen Kaufen', '14.09.2026'),
    'konsens_halten':   (9,  'Investing.com, Empfehlungen Halten', '14.09.2026'),
    'konsens_verkauf':  (5,  'Investing.com, Empfehlungen Verkaufen', '14.09.2026'),
}
