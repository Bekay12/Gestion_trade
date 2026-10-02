#!/usr/bin/env python3
"""
rechnung.py - Vertiefung der beiden Insider-Kaeufe (GameStop, Salesforce), Stand 28.09.2026.

GME: Summe der Teile je Aktie (Kasse, eBay-Beteiligung, Bitcoin, Schulden nach dem Anleihe-
     tausch vom 03.09.2026) und der Rest, den der Kurs dem Einzelhandel zuschreibt.
CRM: Eigentuemerrechnung wie im Neste-Bericht: Einstieg = Boersenwert, Zufluss konstant,
     Endwert = Buchwert; Schwellenkurs bei 13,56 % / 9,08 % (MSCI World, Factsheet 31.08.2026),
     verlangtes Wachstum. Vorsteuer auf Anlegerebene.
Quellen: XBRL der 10-K/10-Q (Akte je Wert in data/xbrl_*.json), GME-10-Q Q2 FY2026 (Text),
Kurse yfinance 28.09.2026.
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cashflow_irr as ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = json.load(open(os.path.join(ROOT, "data", "kurse.json")))
HUERDE, HUERDE_LANG = 13.56, 9.08
QUELLE = "MSCI World, Gross Returns (USD), 10 Jahre annualisiert, Factsheet 31.08.2026"

# ---------------- GameStop ----------------
gme = {
    "kasse_0108": 4854.3, "wertpapiere_0108": 206.0,           # 10-Q Q2 FY26, Bilanz 01.08.2026
    "ebay_stueck": 43390383, "bitcoin": 4710,                   # 10-Q Note 10, Note 11
    "schulden_0108": 4167.8,                                     # 10-Q Note 5
    "tausch_nominal": 1400.0, "tausch_bar": 358.4, "tausch_aktien_mio": 55.5,  # Note 15, geschlossen 03.09.2026
    "aktien_0309": 504.500990,                                   # Deckblatt 10-Q (dei), 03.09.2026
    "ebit_fy25": 232.1, "ebit_h1_25": 55.6, "ebit_h1_26": 303.5, # OperatingIncomeLoss
    "optionsscheine_mio": 59.1, "optionsschein_preis": 32.0,     # 10-Q, Verfall 30.10.2026
    "angebot_ebay": 125.0,
}
kurs_g, ebay, btc = K["GME"]["letzter"][1], K["EBAY"]["letzter"][1], K["BTC-USD"]["letzter"][1]
teile = {
    "Kasse und Wertpapiere nach Barteil des Tauschs": gme["kasse_0108"] + gme["wertpapiere_0108"] - gme["tausch_bar"],
    f"eBay-Beteiligung ({gme['ebay_stueck']/1e6:.1f} Mio. Aktien zu {ebay:.2f} $)": gme["ebay_stueck"] * ebay / 1e6,
    f"Bitcoin ({gme['bitcoin']} zu {btc:,.0f} $, an Coinbase verpfaendet)": gme["bitcoin"] * btc / 1e6,
    "Wandelanleihen nach Tausch (Buchwert, Naeherung)": -(gme["schulden_0108"] - gme["tausch_nominal"]),
}
netto = sum(teile.values())
n = gme["aktien_0309"]
ebit_ltm = gme["ebit_fy25"] - gme["ebit_h1_25"] + gme["ebit_h1_26"]
rest = kurs_g * n - netto
print(f"GME  Kurs {kurs_g}  Aktien {n:.1f} Mio  Boersenwert {kurs_g*n:,.0f} Mio $")
for k, v in teile.items():
    print(f"   {k:62s} {v:>9,.0f} Mio $  {v/n:6.2f} $/Aktie")
print(f"   {'Netto-Finanzvermoegen':62s} {netto:>9,.0f} Mio $  {netto/n:6.2f} $/Aktie")
print(f"   Rest fuer das operative Geschaeft: {rest:,.0f} Mio $ = {rest/n:.2f} $/Aktie; EBIT LTM {ebit_ltm:.1f} -> {rest/ebit_ltm:.1f}x")
for p in (125.0, ebay * 0.8):
    print(f"   eBay zu {p:.2f}: Netto je Aktie {(netto + gme['ebay_stueck']*(p-ebay)/1e6)/n:.2f}")
print(f"   Cohen-Kaeufe 10.09. zu 20,38 und 21.09. zu 22,94 -> Aufschlag auf Netto-Finanzvermoegen "
      f"{20.38/(netto/n)-1:+.0%} / {22.94/(netto/n)-1:+.0%}")

# ---------------- Salesforce ----------------
crm = {"cfo_fy26": 14996.0, "cfo_h1_27": 7970.0, "cfo_h1_26": 7216.0,
       "capex_fy26": 594.0, "capex_h1_27": 316.0, "capex_h1_26": 314.0,
       "sbc_fy26": 3509.0, "sbc_h1_27": 1763.0, "sbc_h1_26": 1607.0,
       "zins_q2_27": 473.0, "zins_h1_27": 790.0, "zins_h1_26": 135.0,
       "eigenkapital": 38378.0, "schulden": 39288.0, "kasse": 8310.0, "wertpapiere": 3093.0,
       "aktien": 823.0, "rueckkauf_h1_27": 27332.0, "anleihen_h1_27": 24842.0}
fcf_reihe = {2018: 2204, 2019: 2803, 2020: 3688, 2021: 4091, 2022: 5283, 2023: 6313, 2024: 9498, 2025: 12434, 2026: 14402}
cfo_ltm = crm["cfo_fy26"] - crm["cfo_h1_26"] + crm["cfo_h1_27"]
capex_ltm = crm["capex_fy26"] - crm["capex_h1_26"] + crm["capex_h1_27"]
sbc_ltm = crm["sbc_fy26"] - crm["sbc_h1_26"] + crm["sbc_h1_27"]
fcf_ltm = cfo_ltm - capex_ltm
# Annahme: im Vorjahr lag der Zins je Quartal auf dem Niveau des Vorjahreshalbjahrs (135/2).
zins_ltm = 2 * crm["zins_h1_26"] - crm["zins_h1_26"] + crm["zins_h1_27"]
zins_voll = 4 * crm["zins_q2_27"]
fcf_zins = fcf_ltm - (zins_voll - zins_ltm)
kurs_c = K["CRM"]["letzter"][1]
mcap = kurs_c * crm["aktien"]
nd = crm["schulden"] - crm["kasse"] - crm["wertpapiere"]
szen = {"LTM (07/2026)": fcf_ltm, "LTM, volle neue Zinslast": fcf_zins,
        "LTM abzgl. Aktienverguetung": fcf_ltm - sbc_ltm, "Median FY2018-2026": statistics.median(fcf_reihe.values())}
print(f"\nCRM  Kurs {kurs_c}  Aktien {crm['aktien']} Mio  Boersenwert {mcap:,.0f}  Nettoschulden {nd:,.0f}  EV {mcap+nd:,.0f}  KBV {mcap/crm['eigenkapital']:.2f}")
print(f"   FCF LTM {fcf_ltm:,.0f} (CFO {cfo_ltm:,.0f}, Capex {capex_ltm:,.0f}), SBC LTM {sbc_ltm:,.0f}, Zins LTM ~{zins_ltm:,.0f} vs Laufrate {zins_voll:,.0f}")
print(f"   Rueckkauf H1 {crm['rueckkauf_h1_27']:,.0f} finanziert mit Anleihen {crm['anleihen_h1_27']:,.0f}; Nettoschulden/FCF LTM {nd/fcf_ltm:.1f}x")
w = list(fcf_reihe.values())
js = sorted(fcf_reihe); a3 = statistics.mean(fcf_reihe[j] for j in js[:3]); e3 = statistics.mean(fcf_reihe[j] for j in js[-3:])
print(f"   FCF-Wachstum realisiert (Mittel erste/letzte drei Jahre) {((e3/a3)**(1/(js[-2]-js[1]))-1):.1%} p.a.")
def konfig(ein, z, end, h):
    return {"currency": "Mio. USD", "opening_liquidity": 0.0, "horizon_start": 2026, "horizon_end": 2041,
            "schedule_years": [2026], "sensitivity_horizons": [2031, 2036, 2041], "revenue_base": 1.0,
            "earnings_base": 1.0, "hurdle": h, "hurdle_source": QUELLE, "revenue_scenarios": [],
            "options": {"X": {"name": "CRM", "investment": {"2026": ein}, "earnings_uplift": z,
                              "uplift_start": 2027, "terminal_value": end}}}
ergebnis = {}
print(f"   {'Szenario':34s} {'Zufluss':>8s} {'Rendite':>8s} {'S5':>8s} {'S10':>8s} {'S15':>8s} {'S10@9,08':>9s} {'g10':>7s}")
for name, z in szen.items():
    c = konfig(mcap, z, crm["eigenkapital"], HUERDE); cl = konfig(mcap, z, crm["eigenkapital"], HUERDE_LANG)
    s = [ci.break_even_investment(c, "X", 2026 + h) for h in (5, 10, 15)]
    sl = ci.break_even_investment(cl, "X", 2036)
    g = ci.required_growth(c, "X", 2036)
    ergebnis[name] = {"zufluss": z, "schwellen": [x / crm["aktien"] if x else None for x in s], "schwelle_lang": sl / crm["aktien"] if sl else None, "wachstum10": g}
    f = lambda x: f"{x/crm['aktien']:8.2f}" if x else "     n.b"
    print(f"   {name:34s} {z:8,.0f} {z/mcap:8.1%} {f(s[0])} {f(s[1])} {f(s[2])} {f(sl):>9s} {g:7.1%}" if g is not None else f"   {name} n.b.")
print(f"   Rang LTM-FCF in eigener Reihe: {100*sum(1 for x in w if x < fcf_ltm)/len(w):.0f} %")
json.dump({"gme": {"teile": teile, "netto": netto, "je_aktie": netto / n, "rest_je_aktie": rest / n, "ebit_ltm": ebit_ltm},
           "crm": {"szenarien": ergebnis, "mcap": mcap, "nettoschulden": nd, "fcf_ltm": fcf_ltm}},
          open(os.path.join(ROOT, "data", "rechnung.json"), "w"), indent=1)
