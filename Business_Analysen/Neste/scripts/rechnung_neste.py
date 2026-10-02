#!/usr/bin/env python3
"""
rechnung_neste.py - Zahlenschicht und Rechnungen des Vollberichts Neste.

Bloecke wie im Muster (GEA_Group/scripts/rechnung_gea.py):
    ROH      auf einer gedruckten Seite belegt; pruefe_seiten.py prueft jeden Wert
    REIHEN   aus data/reihen.json (reihe.py, mit Kettenpruefung)
    EXTERN   Sekundaerquellen ohne Seite: Kurse, Konsens, Directors' Dealings, Peers

Gerechnet werden:
  * Teil 2  Ergebnis- und Zahlungsreihe 2017-2025, Perzentil des Basisfalls
  * Teil 3  Prognosetreue der Investitionen 2021-2025, Prognose WIE ZUERST GEGEBEN
  * Teil 4  Kurs gegen RP-Verkaufsmarge (fuehrender Indikator)
  * Teil 5  drei Handlungsoptionen des Anlegers
  * Teil 6  Anlegerrechnung (unternehmensweit, Mio. EUR), vier Zuflussszenarien
  * Teil 7  Schwellenkurse, Umkehrungen, EV/EBITDA im Zeitverlauf, Horizontmatrix

Die zeitliche Reichweite ist das Thema dieses Berichts. Sie steckt in drei Rechnungen:
  (a) Schwellenkurse auf 5/10/15 Jahre (Cashflow gegen Huerde),
  (b) Dauer der Sondermarge: Zufluss auf LTM-Niveau fuer k Jahre, danach Median,
  (c) kurzer Horizont: Kurs bei historischem Median-Multiple je EBITDA-Niveau.

Aufruf: python3 scripts/rechnung_neste.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cashflow_irr as ci                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START, HORIZONTE = 2026, (5, 10, 15)
HUERDE, HUERDE_LANG = 13.56, 9.08
HUERDE_QUELLE = "MSCI World, Gross Returns (USD), 10 Jahre annualisiert, Factsheet 31.08.2026"
NB = "n.\\,b."
MAL = "\\,$\\times$"
JAHRE = list(range(2017, 2026))

# --------------------------------------------------------------------------
# ROH: (Wert wie gedruckt, Faktor auf die Einheit der Rechnung, Dokument, Seite)
# --------------------------------------------------------------------------
ROH = {
    # Aktie und Bilanz
    "Aktien": (768274059, 1e-6, "neste-ar-2025", "146"),
    "EkJeAktieHj": (11.01, 1, "neste-hj-2026", "3"),
    "EigenkapitalHj": (8458, 1, "neste-hj-2026", "3"),
    "NettoschuldenHj": (3613, 1, "neste-hj-2026", "3"),
    "VerschuldungHj": (29.9, 1, "neste-hj-2026", "3"),
    "RoaceLtm": (18.0, 1, "neste-hj-2026", "3"),
    "StaatAnteil": (44.2, 1, "neste-ar-2025", "239"),
    "AuslandAnteil": (26.4, 1, "neste-ar-2025", "239"),
    "InstitutionenAnteil": (17.7, 1, "neste-ar-2025", "239"),
    "HaushalteAnteil": (11.7, 1, "neste-ar-2025", "239"),
    "Aktionaere": (194382, 1, "neste-ar-2025", "238"),
    "DividendeVorschlag": (0.20, 1, "neste-ar-2025", "238"),
    "AusschuettungsQuoteNeste": (106.6, 1, "neste-ar-2025", "146"),
    # Halbjahr 2026 (Kennzahlenseite S. 3; Quartal, Halbjahr, Vorjahr, Gesamtjahr 2025)
    "HjUmsatz": (11150, 1, "neste-hj-2026", "3"),
    "HjUmsatzVj": (9528, 1, "neste-hj-2026", "3"),
    "HjVerglEbitda": (2064, 1, "neste-hj-2026", "3"),
    "HjVerglEbitdaVj": (551, 1, "neste-hj-2026", "3"),
    "QzVerglEbitda": (1203, 1, "neste-hj-2026", "3"),
    "QzVerglEbitdaVj": (341, 1, "neste-hj-2026", "3"),
    "HjNettogewinn": (1298, 1, "neste-hj-2026", "3"),
    "HjEps": (1.69, 1, "neste-hj-2026", "3"),
    "HjEpsVj": (-0.10, 1, "neste-hj-2026", "3"),
    "HjCfo": (993, 1, "neste-hj-2026", "3"),
    "HjCfoVj": (476, 1, "neste-hj-2026", "3"),
    "HjInvest": (404, 1, "neste-hj-2026", "15"),
    "HjInvestVj": (487, 1, "neste-hj-2026", "15"),
    "HjLeasing": (122, 1, "neste-hj-2026", "15"),
    "HjLeasingVj": (127, 1, "neste-hj-2026", "15"),
    "HjNwcAufbau": (936, 1, "neste-hj-2026", "15"),
    "QzNwcAufbau": (842, 1, "neste-hj-2026", "2"),
    "HjPreiseffekt": (2.4, 1, "neste-hj-2026", "3"),
    "RedDreiNachfrage": (1.5, 1, "neste-hj-2026", "1"),
    "HjNwcVj": (37, 1, "neste-hj-2026", "15"),
    "NwcVoll": (364, 1, "neste-ar-2025", "154"),
    "HjRpMarge": (1054, 1, "neste-hj-2026", "6"),
    "HjRpMargeVj": (338, 1, "neste-hj-2026", "6"),
    "QzRpMarge": (1223, 1, "neste-hj-2026", "1"),
    "QzRpMargeVj": (361, 1, "neste-hj-2026", "1"),
    "QzRpEbitda": (859, 1, "neste-hj-2026", "1"),
    "QzRpEbitdaVj": (174, 1, "neste-hj-2026", "1"),
    "QzOpEbitda": (334, 1, "neste-hj-2026", "1"),
    "QzOpMarge": (25.8, 1, "neste-hj-2026", "1"),
    "QzOpMargeVj": (10.0, 1, "neste-hj-2026", "1"),
    "ProgrammLaufrateHj": (594, 1, "neste-hj-2026", "1"),
    "ProgrammQuartal": (118, 1, "neste-hj-2026", "1"),
    "ProgrammZiel": (350, 1, "neste-ar-2025", "80"),
    "ProgrammLaufrate": (376, 1, "neste-ar-2025", "80"),
    "InvestPrognoseHj": (1.2, 1, "neste-hj-2026", "2"),
    "StillstandPorvoo": (8, 1, "neste-hj-2026", "2"),
    "StillstandRotterdam": (8, 1, "neste-hj-2026", "2"),
    "StillstandSingapur": (11, 1, "neste-hj-2026", "2"),
    # Gesamtjahr 2025 und Strategie
    "RpMarge": (411, 1, "neste-ar-2025", "81"),
    "RpMargeVj": (377, 1, "neste-ar-2025", "81"),
    "RpReferenzmarge": (435, 1, "neste-ar-2025", "81"),
    "RpReferenzmargeVj": (460, 1, "neste-ar-2025", "81"),
    "Abfallanteil": (95, 1, "neste-ar-2025", "81"),
    "AbfallanteilVj": (90, 1, "neste-ar-2025", "81"),
    "RoaceZielAlt": (15, 1, "neste-ar-2022", "149"),
    "RpEbitda": (764, 1, "neste-ar-2025", "79"),
    "RpEbitdaVj": (514, 1, "neste-ar-2025", "79"),
    "RpAbsatz": (4.1, 1, "neste-ar-2025", "79"),
    "RpAbsatzVj": (3.7, 1, "neste-ar-2025", "79"),
    "Kapazitaet": (5.5, 1, "neste-ar-2025", "14"),
    "KapazitaetZiel": (6.8, 1, "neste-ar-2024", "8"),
    "RotterdamKostenAlt": (1.9, 1, "neste-ar-2024", "91"),
    "RotterdamKostenNeu": (2.5, 1, "neste-ar-2024", "91"),
    "InvestNachRotterdam": (0.5, 1, "neste-ar-2024", "91"),
    "Verschuldungsziel": (40, 1, "neste-ar-2025", "80"),
    "FreierCfNeste": (759, 1, "neste-ar-2025", "80"),
    "FreierCfNesteVj": (-341, 1, "neste-ar-2025", "80"),
    "BauzinsenAktiviert": (69, 1, "neste-ar-2025", "210"),
    # Instandhaltungsinvestitionen (Mio. EUR, Kassenabfluss) je Jahr
    "Inst2017": (214, 1, "neste-fsr-2018", "7"),
    "Inst2018": (253, 1, "neste-fsr-2018", "7"),
    "Inst2019": (260, 1, "neste-ar-2019", "107"),
    "Inst2020": (190, 1, "neste-ar-2020", "114"),
    "Inst2021": (411, 1, "neste-ar-2021", "145"),
    "Inst2022": (249, 1, "neste-ar-2022", "149"),
    "Inst2023": (305, 1, "neste-ar-2023", "148"),
    "Inst2024": (579, 1, "neste-ar-2024", "92"),
    # Investitionen ohne M&A: Prognose wie zuerst gegeben (Mrd. EUR) und Ist (Mio. EUR)
    "Prog2021": (1.2, 1, "neste-ar-2020", "128"),
    "Prog2022": (1.1, 1, "neste-ar-2021", "161"),
    "Prog2023": (1.8, 1, "neste-ar-2022", "171"),
    "Prog2024Von": (1.4, 1, "neste-ar-2023", "170"),
    "Prog2024Bis": (1.6, 1, "neste-ar-2023", "170"),
    "Prog2025Von": (1.1, 1, "neste-ar-2024", "99"),
    "Prog2025Bis": (1.3, 1, "neste-ar-2024", "99"),
    "Prog2026Von": (1.0, 1, "neste-ar-2025", "85"),
    "Prog2026Bis": (1.2, 1, "neste-ar-2025", "85"),
    "Ist2021": (976, 1, "neste-ar-2021", "145"),
    "Ist2022": (990, 1, "neste-ar-2022", "149"),
    "Ist2023": (1431, 1, "neste-ar-2023", "148"),
    "Ist2024": (1552, 1, "neste-ar-2024", "92"),
    "Ist2025": (923, 1, "neste-ar-2025", "80"),
    "Prog2021Angepasst": (1.1, 1, "neste-ar-2021", "152"),
    # RP-Verkaufsmarge (USD/t) je Jahr, jeweils juengste Darstellung
    "Marge2019": (733, 1, "neste-ar-2020", "115"),
    "Marge2020": (703, 1, "neste-ar-2021", "146"),
    "Marge2021": (715, 1, "neste-ar-2022", "150"),
    "Marge2022": (779, 1, "neste-ar-2023", "148"),
    "Marge2022Alt": (804, 1, "neste-ar-2022", "150"),
    "Marge2023": (863, 1, "neste-ar-2023", "148"),
    "Marge2024": (377, 1, "neste-ar-2025", "81"),
    "Marge2025": (411, 1, "neste-ar-2025", "81"),
}
# Nachkommastellen, wo Python die gedruckte Schreibweise nicht kennt (0.20 -> "0.2").
NACHKOMMA = {"DividendeVorschlag": 2, "EkJeAktieHj": 2, "HjEps": 2, "HjEpsVj": 2,
             "VerschuldungHj": 1, "RoaceLtm": 1, "QzOpMarge": 1, "QzOpMargeVj": 1,
             "InvestPrognoseHj": 1, "RpAbsatz": 1, "RpAbsatzVj": 1, "Kapazitaet": 1,
             "KapazitaetZiel": 1, "RotterdamKostenAlt": 1, "RotterdamKostenNeu": 1,
             "InvestNachRotterdam": 1, "RedDreiNachfrage": 1, "HjPreiseffekt": 1, "StaatAnteil": 1, "AuslandAnteil": 1,
             "InstitutionenAnteil": 1, "HaushalteAnteil": 1,
             **{k: 1 for k in ROH if k.startswith("Prog2")}}

# EXTERN: Sekundaerquellen ohne gedruckte Seite; Belege in docs/research/*.md
EXTERN = {}
EXTERN_PFAD = os.path.join(DATA, "extern.json")
if os.path.exists(EXTERN_PFAD):
    EXTERN = json.load(open(EXTERN_PFAD))

R = json.load(open(os.path.join(DATA, "reihen.json")))
KURSE = json.load(open(os.path.join(DATA, "kurse_roh.json")))["NESTE.HE"]


def roh(n):
    w, f, _, _ = ROH[n]
    return w * f


def reihe(posten):
    return {int(j): v["wert"] for j, v in R.get(posten, {}).items()}


def de(x, stellen=0) -> str:
    if x is None:
        return "--"
    s = f"{abs(x):,.{stellen}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("$-$" if round(x, stellen) < 0 else "") + s


def pz(x, stellen=1) -> str:
    return "--" if x is None else de(x, stellen) + "\\,\\%"


def perzentil(werte, p):
    s = sorted(werte)
    k = (len(s) - 1) * p
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


def rang(werte, x) -> float:
    """Anteil der Jahre (in %), deren Wert unter x liegt."""
    return 100.0 * sum(1 for w in werte if w < x) / len(werte)


# --------------------------------------------------------------------------
# Zahlungsreihe
# --------------------------------------------------------------------------
def investitionen() -> dict:
    """Sach- und immaterielle Investitionen (negativ); 2017 als eine Zeile gedruckt."""
    sa, im, ges = reihe("sachanlagen"), reihe("immateriell"), reihe("investitionen_gesamt")
    return {j: (ges[j] if j in ges else sa[j] + im[j]) for j in JAHRE}


def leasing() -> dict:
    """Leasingtilgung (negativ). Vor 2019 (IAS 17) keine aktivierten Leasingverhaeltnisse."""
    le = reihe("leasing")
    return {j: le.get(j, 0.0) for j in JAHRE}


def fcf() -> dict:
    """Ausschuettungsfaehiger freier Zufluss: operativer CF + Investitionen + Leasingtilgung."""
    cfo, inv, le = reihe("operativer_cf"), investitionen(), leasing()
    return {j: cfo[j] + inv[j] + le[j] for j in JAHRE}


def erntezufluss() -> dict:
    """Zufluss ohne Wachstumsinvestitionen: operativer CF - Instandhaltung - Leasing."""
    cfo, le = reihe("operativer_cf"), leasing()
    return {j: cfo[j] - roh(f"Inst{j}") + le[j] for j in JAHRE if f"Inst{j}" in ROH}


def konfig(einstieg: float, zufluss: float, endwert: float, huerde: float) -> dict:
    return {"currency": "Mio. EUR", "opening_liquidity": 0.0, "horizon_start": START,
            "horizon_end": START + max(HORIZONTE), "schedule_years": [START, START + 1],
            "sensitivity_horizons": [START + h for h in HORIZONTE], "revenue_base": 1.0,
            "earnings_base": 1.0, "hurdle": huerde, "hurdle_source": HUERDE_QUELLE,
            "revenue_scenarios": [],
            "options": {"X": {"name": "Anlegerrechnung", "investment": {str(START): einstieg},
                              "earnings_uplift": zufluss, "uplift_start": START + 1,
                              "terminal_value": endwert}}}


def reihe_irr(einstieg, zufluesse, endwert):
    """IRR einer frei vorgegebenen Zuflussfolge (Jahr 1..n), Endwert im letzten Jahr."""
    fl = [-einstieg] + list(zufluesse)
    fl[-1] += endwert
    return ci.irr(fl)


def kurzbeleg(dok: str) -> str:
    """Dateiname -> Kurzbeleg der Tabellen: neste-ar-2019 -> GB~2019."""
    art, jahr = dok.split("-")[1:]
    return {"ar": "GB", "fsr": "FSR", "hj": "HJ"}[art] + "~" + jahr


def makroname(k: str) -> str:
    """LaTeX-Makronamen duerfen keine Ziffern tragen: Jahrgang 2017 -> XVII."""
    import re
    roem = {17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI", 22: "XXII", 23: "XXIII",
            24: "XXIV", 25: "XXV", 26: "XXVI"}
    aus = re.sub(r"20(\d\d)", lambda m: roem[int(m.group(1))], k)
    if re.search(r"\d", aus):
        raise ValueError(f"Makroname mit Ziffer: {k}")
    return aus


def main() -> int:
    M, T = {}, {}
    n_akt = roh("Aktien")
    kurs = KURSE["letzter"][1]
    mcap = kurs * n_akt
    bw = roh("EkJeAktieHj") * n_akt
    nd = roh("NettoschuldenHj")

    for name in ROH:
        w, fak, _, _ = ROH[name]
        stellen = len(str(w).split(".")[1]) if isinstance(w, float) and w % 1 else 0
        M[name] = de(w * fak, 2 if fak != 1 else NACHKOMMA.get(name, stellen))
    for k, v in EXTERN.items():
        if isinstance(v, list):
            continue
        if isinstance(v, (int, float)):
            M[k] = de(v, 2 if isinstance(v, float) and v % 1 else 0)
        else:
            M[k] = str(v)

    # ---------------- Teil 2: Reihen ----------------
    f, e = fcf(), erntezufluss()
    inv, le, cfo = investitionen(), leasing(), reihe("operativer_cf")
    fw, ew = [f[j] for j in JAHRE], list(e.values())
    med, pes = statistics.median(fw), perzentil(fw, 0.25)
    ltm_cfo = cfo[2025] - roh("HjCfoVj") + roh("HjCfo")
    ltm_inv = -inv[2025] - roh("HjInvestVj") + roh("HjInvest")
    ltm_le = -le[2025] - roh("HjLeasingVj") + roh("HjLeasing")
    ltm = ltm_cfo - ltm_inv - ltm_le
    # Umlaufvermoegen: 2025 freigesetzt (+364), im 1. Hj. 2026 fuer die Stillstaende aufgebaut
    # (-936). Ohne diese Bewegung zeigt die LTM-Zeile die Ertragskraft der Sondermarge selbst.
    ltm_nwc = roh("NwcVoll") - roh("HjNwcVj") - roh("HjNwcAufbau")
    ltm_ohne = ltm - ltm_nwc
    ernte = statistics.median(cfo[j] for j in JAHRE) - 1000 * roh("InvestNachRotterdam") + le[2025]
    ve = reihe("vergl_ebitda")
    ltm_ebitda = ve[2025] - roh("HjVerglEbitdaVj") + roh("HjVerglEbitda")
    ve_jahre = sorted(ve)
    M.update({
        "Kurs": de(kurs, 2), "KursDatum": ".".join(reversed(KURSE["letzter"][0].split("-"))),
        "AktienMio": de(n_akt, 1), "Boersenwert": de(mcap), "BuchwertJeAktie": de(bw / n_akt, 2),
        "Buchwert": de(bw), "KBV": de(mcap / bw, 2), "Huerde": de(HUERDE, 2),
        "HuerdeLang": de(HUERDE_LANG, 2), "HuerdeQuelle": HUERDE_QUELLE,
        "Hoch": de(KURSE["hoch_52w"], 2), "Tief": de(KURSE["tief_52w"], 2),
        "FcfMedian": de(med), "FcfPes": de(pes), "FcfLtm": de(ltm), "FcfErnte": de(ernte),
        "FcfLtmOhne": de(ltm_ohne), "NwcLtm": de(ltm_nwc), "FcfRenditeLtmOhne": pz(100 * ltm_ohne / mcap),
        "FcfMin": de(min(fw)), "FcfMax": de(max(fw)), "FcfNegativJahre": str(sum(1 for x in fw if x < 0)),
        "FcfRangLtm": de(rang(fw, ltm), 0), "FcfRangVoll": de(rang(fw, f[2025]), 0),
        "FcfVoll": de(f[2025]), "FcfSumme": de(sum(fw)), "DividendenSumme": de(-sum(reihe("dividenden")[j] for j in JAHRE)),
        "ErnteMedian": de(statistics.median(ew)), "ErnteJahre": f"{min(e)}--{max(e)}",
        "InstMedian": de(statistics.median(roh(f"Inst{j}") for j in e)),
        "CfoMedian": de(statistics.median(cfo[j] for j in JAHRE)), "CfoLtm": de(ltm_cfo),
        "InvestLtm": de(ltm_inv), "LeasingLtm": de(ltm_le), "LeasingVoll": de(-le[2025]),
        "FcfRenditeLtm": pz(100 * ltm / mcap), "FcfRenditeMed": pz(100 * med / mcap),
        "EbitdaLtm": de(ltm_ebitda), "EbitdaMedian": de(statistics.median(ve.values())),
        "EbitdaJahre": f"{ve_jahre[0]}--{ve_jahre[-1]}",
        "EbitdaRangLtm": de(rang(list(ve.values()), ltm_ebitda), 0),
        "HjEbitdaFaktor": de(roh("HjVerglEbitda") / roh("HjVerglEbitdaVj"), 1),
        "QzEbitdaFaktor": de(roh("QzVerglEbitda") / roh("QzVerglEbitdaVj"), 1),
        "HjUmsatzDelta": pz(100 * (roh("HjUmsatz") / roh("HjUmsatzVj") - 1)),
        "Eigenkapital": de(bw),
    })
    # Ergebnisreihe
    T["reihe_ergebnis"] = " \\\\\n".join(
        f"{j} & {de(reihe('umsatz')[j])} & {de(ve.get(j))} & {de(reihe('ebit')[j])} & "
        f"{de(reihe('ergebnis')[j])} & {de(reihe('eps')[j], 2)} & {de(reihe('dps')[j], 2)} & "
        f"{de(reihe('roace')[j], 1)} & {de(reihe('nettoschulden')[j])} & {de(reihe('verschuldungsgrad')[j], 1)}"
        for j in JAHRE) + " \\\\"
    # Zahlungsreihe
    zeilen = []
    for j in JAHRE:
        q = R["operativer_cf"][str(j)]
        zeilen.append(f"{j} & {de(cfo[j])} & {de(inv[j])} & {de(le[j]) if le[j] else '--'} & "
                      f"{de(f[j])} & {de(roh('Inst' + str(j))) if 'Inst' + str(j) in ROH else '--'} & "
                      f"{de(e.get(j))} & {de(reihe('dividenden')[j])} & "
                      f"{kurzbeleg(q['bericht'])}, S.~{q['seite']}")
    zeilen.append(f"LTM 06/26 & {de(ltm_cfo)} & {de(-ltm_inv)} & {de(-ltm_le)} & {de(ltm)} & -- & -- & -- & HJ 2026, S.~15")
    T["reihe_zahlung"] = " \\\\\n".join(zeilen) + " \\\\"
    # Ueberblick 2025 / 2024 und Halbjahr
    ueb = [("Umsatz", "umsatz", 0), ("Vergleichbares EBITDA", "vergl_ebitda", 0),
           ("Betriebsergebnis", "ebit", 0), ("Periodenergebnis", "ergebnis", 0),
           ("Operativer Mittelzufluss", "operativer_cf", 0), ("Nettofinanzschulden", "nettoschulden", 0),
           ("Verschuldungsgrad in \\%", "verschuldungsgrad", 1), ("Vergl. ROACE in \\%", "roace", 1),
           ("Ergebnis je Aktie (EUR)", "eps", 2), ("Dividende je Aktie (EUR)", "dps", 2)]
    T["ueberblick"] = " \\\\\n".join(
        f"{lab} & {de(reihe(p)[2025], s)} & {de(reihe(p)[2024], s)} & {de(reihe(p)[2025] - reihe(p)[2024], max(s, 0))}"
        for lab, p, s in ueb) + " \\\\"

    # ---------------- Teil 3: Prognosetreue Investitionen ----------------
    zeilen, unter = [], 0
    for j in range(2021, 2026):
        if f"Prog{j}" in ROH:
            lo = hi = 1000 * roh(f"Prog{j}")
            zelle = f"ca. {de(roh(f'Prog{j}'), 1)} Mrd."
        else:
            lo, hi = 1000 * roh(f"Prog{j}Von"), 1000 * roh(f"Prog{j}Bis")
            zelle = f"{de(roh(f'Prog{j}Von'), 1)}--{de(roh(f'Prog{j}Bis'), 1)} Mrd."
        ist = roh(f"Ist{j}")
        # "ca." wird als +/- 5 % gelesen; ausserhalb gilt darunter/darueber
        if lo == hi:
            lo, hi = lo * 0.95, hi * 1.05
        lage = "darunter" if ist < lo else ("darüber" if ist > hi else "in der Spanne")
        unter += lage == "darunter"
        dok, s = ROH[f"Prog{j}" if f"Prog{j}" in ROH else f"Prog{j}Von"][2:]
        zeilen.append(f"{j} & {zelle} & {de(ist)} & {pz(100 * (ist / ((lo + hi) / 2) - 1), 0)} & {lage} & "
                      f"GB~{int(dok[-4:])}, S.~{s}")
    T["prognose"] = " \\\\\n".join(zeilen) + " \\\\"
    M["PrognoseUnter"] = str(unter)

    # ---------------- Teil 4: Kurs gegen RP-Marge ----------------
    marge = {j: roh(f"Marge{j}") for j in range(2019, 2026)}
    ye = reihe("kurs_ende")
    zeilen = []
    for j in range(2019, 2026):
        dm = 100 * (marge[j] / marge[j - 1] - 1) if j - 1 in marge else None
        dk = 100 * (ye[j] / ye[j - 1] - 1)
        zeilen.append(f"{j} & {de(marge[j])} & {de(dm, 1)} & {de(ve.get(j))} & {de(ye[j], 2)} & {de(dk, 1)}")
    zeilen.append(f"1.~Hj.~2026 & {de(roh('HjRpMarge'))} & -- & {de(roh('HjVerglEbitda'))} & {de(kurs, 2)} & "
                  f"{de(100 * (kurs / ye[2025] - 1), 1)}")
    T["marge"] = " \\\\\n".join(zeilen) + " \\\\"
    for j in range(2020, 2026):
        M[f"MargeDelta{j}"] = pz(100 * (marge[j] / marge[j - 1] - 1), 0)
        M[f"KursDelta{j}"] = pz(100 * (ye[j] / ye[j - 1] - 1), 0)
    mw = list(marge.values())
    M.update({"MargeMedian": de(statistics.median(mw)), "MargeMax": de(max(mw)),
              "MargeRangHj": de(rang(mw, roh("HjRpMarge")), 0),
              "MargeFaktorHj": de(roh("HjRpMarge") / statistics.median(mw), 1),
              "KursSeitJahresende": pz(100 * (kurs / ye[2025] - 1), 0),
              "KursTiefZuHeute": de(kurs / KURSE["tief_52w"], 1)})

    # ---------------- Teil 6/7: Anlegerrechnung ----------------
    szen = {"PES": ("pessimistisch (25.~Perzentil)", pes), "MED": ("Basis (Median 2017--2025)", med),
            "ERN": ("Erntephase nach Rotterdam", ernte), "LTM": ("letzte zwölf Monate", ltm),
            "LTMO": ("LTM ohne Umlaufvermögen", ltm_ohne)}
    zs, zw, ze = [], [], []
    for k, (name, z) in szen.items():
        cfg, cfgl = konfig(mcap, z, bw, HUERDE), konfig(mcap, z, bw, HUERDE_LANG)
        s = [ci.break_even_investment(cfg, "X", START + h) for h in HORIZONTE]
        sl = ci.break_even_investment(cfgl, "X", START + 10)
        g = [ci.required_growth(cfg, "X", START + h) for h in HORIZONTE]
        t = [ci.required_terminal_value(cfg, "X", START + h) for h in HORIZONTE]
        irr10 = ci.irr(ci.full_cashflows(cfg, "X", START + 10))
        zs.append(f"{name} & {de(z)} & " + " & ".join(de(x / n_akt, 2) if x else NB for x in s)
                  + " & " + (de(sl / n_akt, 2) if sl else NB))
        zw.append(f"{name} & " + " & ".join(pz(100 * x) if x is not None else NB for x in g))
        ze.append(f"{name} & " + " & ".join((de(x / bw, 1) + MAL) if x is not None else NB for x in t))
        M[f"Schwelle{k}Fuenf"] = de(s[0] / n_akt, 2) if s[0] else NB
        M[f"Schwelle{k}Zehn"] = de(s[1] / n_akt, 2) if s[1] else NB
        M[f"Schwelle{k}Fuenfzehn"] = de(s[2] / n_akt, 2) if s[2] else NB
        M[f"Schwelle{k}ZehnLang"] = de(sl / n_akt, 2) if sl else NB
        M[f"Wachstum{k}Zehn"] = pz(100 * g[1]) if g[1] is not None else NB
        M[f"Endwert{k}Zehn"] = de(t[1] / bw, 1) if t[1] is not None else NB
        M[f"Irr{k}Zehn"] = pz(100 * irr10)
        M[f"Abstand{k}"] = pz(100 * (kurs / (s[1] / n_akt) - 1), 0) if s[1] else NB
    T["schwelle"] = " \\\\\n".join(zs) + " \\\\"
    T["wachstum"] = " \\\\\n".join(zw) + " \\\\"
    T["endwert"] = " \\\\\n".join(ze) + " \\\\"
    M["EndwertHeute"] = de(mcap / bw, 1)

    # Realisiertes Wachstum der eigenen Reihen (Durchschnitt erste drei gegen letzte drei Jahre)
    def real(reihe_d):
        js = sorted(reihe_d)
        a3 = statistics.mean(reihe_d[j] for j in js[:3])
        e3 = statistics.mean(reihe_d[j] for j in js[-3:])
        ab = js[-2] - js[1]
        return (e3 / a3) ** (1 / ab) - 1 if a3 > 0 and e3 > 0 else None
    M["CfoWachstumReal"] = pz(100 * real({j: cfo[j] for j in JAHRE}))
    M["EbitdaWachstumReal"] = pz(100 * real(ve))
    r_f = real(f)
    M["FcfWachstumReal"] = pz(100 * r_f) if r_f is not None else NB

    # (b) Dauer der Sondermarge: LTM-Zufluss k Jahre, danach Median; zehn Jahre
    zeilen = []
    wort = {0: "Null", 1: "Eins", 2: "Zwei", 3: "Drei", 5: "Fuenf", 10: "Zehn"}
    for k in (0, 1, 2, 3, 5, 10):
        zfl = [ltm] * k + [med] * (10 - k)
        a = reihe_irr(mcap, zfl, bw)
        b = reihe_irr(mcap, zfl, mcap)
        zeilen.append(f"{k} & {pz(100 * a)} & {pz(100 * b)}")
        M[f"DauerIrrBuch{wort[k]}"] = pz(100 * a)
        M[f"DauerIrrMarkt{wort[k]}"] = pz(100 * b)
    T["dauer"] = " \\\\\n".join(zeilen) + " \\\\"

    # (c) EV/EBITDA im Zeitverlauf und kurzer Horizont
    bwert = reihe("boersenwert")
    ev = {j: (bwert[j] + reihe("nettoschulden")[j]) / ve[j] for j in ve_jahre}
    ev_med = statistics.median(ev.values())
    ev_heute_ltm = (mcap + nd) / ltm_ebitda
    ev_heute_voll = (mcap + nd) / ve[2025]
    T["evebitda"] = " \\\\\n".join(
        f"{j} & {de(bwert[j])} & {de(reihe('nettoschulden')[j])} & {de(ve[j])} & {de(ev[j], 1)}"
        for j in ve_jahre) + f" \\\\\nheute (LTM) & {de(mcap)} & {de(nd)} & {de(ltm_ebitda)} & {de(ev_heute_ltm, 1)} \\\\"
    for j in ve_jahre:
        M[f"Ev{j}"] = de(ev[j], 1)
    M.update({"EvMedian": de(ev_med, 1), "EvHeuteLtm": de(ev_heute_ltm, 1),
              "EvHeuteVoll": de(ev_heute_voll, 1), "EvRangLtm": de(rang(list(ev.values()), ev_heute_ltm), 0),
              "EvMin": de(min(ev.values()), 1), "EvMax": de(max(ev.values()), 1)})
    niveaus = [("LTM bis 06/2026", ltm_ebitda, "LTM"), ("Median 2019--2025", statistics.median(ve.values()), "MED"),
               ("Geschäftsjahr 2025", ve[2025], "VOLL"), ("Minimum (2024)", min(ve.values()), "MIN")]
    zeilen = []
    for name, eb, k in niveaus:
        p = (ev_med * eb - nd) / n_akt
        zeilen.append(f"{name} & {de(eb)} & {de(p, 2)} & {pz(100 * (p / kurs - 1), 0)}")
        M[f"Kurzfrist{k}"] = de(p, 2)
        M[f"KurzfristAbstand{k}"] = pz(100 * (p / kurs - 1), 0)
    # Mittlerer Horizont (Annahme, im Text erklaert): Median-EBITDA skaliert mit der Kapazitaet
    # nach Rotterdam, bewertet zum Median-Multiple. Keine Prognose, eine Rechnung auf zwei Belegen.
    eb_rot = statistics.median(ve.values()) * roh("KapazitaetZiel") / roh("Kapazitaet")
    p_rot = (ev_med * eb_rot - nd) / n_akt
    zeilen.append(f"Median, skaliert auf {de(roh('KapazitaetZiel'), 1)} Mio.\\,t & {de(eb_rot)} & {de(p_rot, 2)} & "
                  f"{pz(100 * (p_rot / kurs - 1), 0)}")
    M["KurzfristROT"] = de(p_rot, 2); M["KurzfristAbstandROT"] = pz(100 * (p_rot / kurs - 1), 0)
    M["EbitdaRot"] = de(eb_rot)
    T["kurzfrist"] = " \\\\\n".join(zeilen) + " \\\\"
    # Zufluss, den der Kurs bei Ausstieg zum heutigen Boersenwert verlangt: IRR = Zufluss / Preis
    M["ZuflussBedarf"] = de(HUERDE / 100 * mcap)
    M["ZuflussBedarfLang"] = de(HUERDE_LANG / 100 * mcap)
    M["FcfBestesJahr"] = de(max(fw)); M["FcfBestesJahrJahr"] = str(max(f, key=f.get))
    # EBITDA, das der Kurs beim Median-Multiple verlangt
    M["EbitdaImpliziert"] = de((mcap + nd) / ev_med)

    # ---------------- Teil 5: Optionen des Anlegers ----------------
    cfg_med = konfig(mcap, med, bw, HUERDE)
    schw = ci.break_even_investment(cfg_med, "X", START + 10)
    optionen = {"A": ("Kauf heute", mcap), "B": ("Kauf zum Schwellenkurs (Basis)", schw)}
    zeilen = []
    for k, (bez, ein) in optionen.items():
        w = [ci.irr(ci.full_cashflows(konfig(ein, z, bw, HUERDE), "X", START + 10))
             for _, z in szen.values()]
        zeilen.append(f"{bez} & {de(ein / n_akt, 2)} & " + " & ".join(pz(100 * x) for x in w))
        M[f"Option{k}Kurs"] = de(ein / n_akt, 2)
        for sk, x in zip(szen, w):
            M[f"Option{k}Irr{sk}"] = pz(100 * x)
    irr_c = ci.irr(ci.full_cashflows(konfig(mcap, 0.0, mcap * (1 + HUERDE / 100) ** 10, HUERDE), "X", START + 10))
    zeilen.append("Verzicht, Indexanlage & " + de(kurs, 2) + " & " + " & ".join([pz(100 * irr_c)] * len(szen)))
    T["optionen"] = " \\\\\n".join(zeilen) + " \\\\"
    M["OptionCIrr"] = pz(100 * irr_c)
    M["KapazitaetPlus"] = de(roh("KapazitaetZiel") - roh("Kapazitaet"), 1)
    M["KapazitaetPlusPz"] = pz(100 * (roh("KapazitaetZiel") / roh("Kapazitaet") - 1), 0)
    M["RotterdamMehrkosten"] = pz(100 * (roh("RotterdamKostenNeu") / roh("RotterdamKostenAlt") - 1), 0)
    M["ReaktionHj"] = pz(100 * (EXTERN["KursReaktionNach"] / EXTERN["KursReaktionVor"] - 1)) if EXTERN else NB
    M["MargeAbstandRef"] = de(roh("RpReferenzmarge") - roh("RpMarge"))
    M["MargeAbstandRefVj"] = de(roh("RpReferenzmargeVj") - roh("RpMargeVj"))
    M["OptionBAbschlag"] = pz(100 * (1 - schw / mcap), 0)

    # KGV: Jahresende (Kennzahlenseite) ueber Ergebnis je Aktie desselben Jahres, nur positive
    eps, ye_k = reihe("eps"), reihe("kurs_ende")
    kgv = {j: ye_k[j] / eps[j] for j in JAHRE if eps[j] > 0}
    eps_ltm = eps[2025] - roh("HjEpsVj") + roh("HjEps")
    M.update({"KgvMedian": de(statistics.median(kgv.values()), 1), "EpsLtm": de(eps_ltm, 2),
              "KgvLtm": de(kurs / eps_ltm, 1), "KgvRangLtm": de(rang(list(kgv.values()), kurs / eps_ltm), 0),
              "KgvJahre": str(len(kgv))})
    if "KonsensEpsXXVI" in EXTERN:
        M["KgvXXVI"] = de(kurs / EXTERN["KonsensEpsXXVI"], 1)
        M["KgvXXVII"] = de(kurs / EXTERN["KonsensEpsXXVII"], 1)
        M["EpsRueckgangXXVII"] = pz(100 * (1 - EXTERN["KonsensEpsXXVII"] / EXTERN["KonsensEpsXXVI"]), 0)
    T["kgv"] = " \\\\\n".join(f"{j} & {de(ye_k[j], 2)} & {de(eps[j], 2)} & {de(v, 1)}"
                              for j, v in kgv.items()) + " \\\\"
    # Kursreihe fuer die Abbildung (Monatsende, Yahoo Finance) als CSV fuer pgfplots
    with open(os.path.join(DATA, "kursreihe.csv"), "w") as fo:
        fo.write("datum,kurs\n")
        for m, v in sorted(KURSE["monatsende"].items()):
            fo.write(f"{m}-15,{v}\n")
    M["VerglEbitdaVoll"] = de(ve[2025])
    M["ErgebnisVoll"] = de(reihe("ergebnis")[2025])

    # Teil 2: ausgewiesenes gegen vergleichbares EBITDA; das Vorzeichen der Differenz wechselt
    eb = reihe("ebitda")
    T["ebitda_abgleich"] = " \\\\\n".join(
        f"{j} & {de(eb[j])} & {de(ve[j])} & {de(eb[j] - ve[j])} & {'ausgewiesen höher' if eb[j] > ve[j] else 'vergleichbar höher'}"
        for j in ve_jahre) + " \\\\"
    M["EbitdaHoeherJahre"] = ", ".join(str(j) for j in ve_jahre if eb[j] > ve[j])
    um = reihe("umsatz")
    M["UmsatzDelta"] = pz(100 * (um[2025] / um[2024] - 1))
    M["UmsatzMax"] = de(max(um.values())); M["UmsatzMaxJahr"] = str(max(um, key=um.get))
    M["EbitdaDeltaVoll"] = pz(100 * (ve[2025] / ve[2024] - 1))
    M["DivRenditeHeute"] = pz(100 * roh("DividendeVorschlag") / kurs)
    M["DivRenditeJahresende"] = pz(100 * roh("DividendeVorschlag") / ye[2025])
    M["LeasingSummeFuenf"] = de(-sum(le[j] for j in range(2021, 2026)))
    M["FinanzierungsLuecke"] = de(-sum(cfo[j] + inv[j] + le[j] + reihe("dividenden")[j] for j in range(2021, 2026)))
    M["DpsMax"] = de(max(reihe("dps").values()), 2); M["DpsMaxJahr"] = str(max(reihe("dps"), key=reihe("dps").get))
    M["KursMax"] = de(max(reihe("kurs_hoch").values()), 2)
    M["KursMaxJahr"] = str(max(reihe("kurs_hoch"), key=reihe("kurs_hoch").get))
    M["KursMin"] = de(min(reihe("kurs_tief").values()), 2)
    M["KursMinJahr"] = str(min(reihe("kurs_tief"), key=reihe("kurs_tief").get))
    M["KursVomHoch"] = pz(100 * (kurs / max(reihe("kurs_hoch").values()) - 1), 0)
    M["NettoschuldenAnstieg"] = de(reihe("nettoschulden")[2024] - reihe("nettoschulden")[2021])
    M["InvestSumme"] = de(-sum(inv[j] for j in range(2021, 2026)))
    M["CfoSumme"] = de(sum(cfo[j] for j in range(2021, 2026)))
    M["DivSummeFuenf"] = de(-sum(reihe("dividenden")[j] for j in range(2021, 2026)))
    # Directors' Dealings (Sekundaerquelle, neste.com, abgerufen 27.09.2026)
    T["dd"] = " \\\\\n".join(
        f"{d} & {p} & {a} & {de(n)} & {de(pr, 2) if pr else '--'} & {de(n * pr) if pr else '--'}"
        for d, p, a, n, pr in EXTERN.get("dd", [])) + " \\\\"
    # Ausgabe
    with open(os.path.join(DATA, "kennzahlen.tex"), "w") as fo:
        fo.write("% erzeugt von scripts/rechnung_neste.py - nicht von Hand aendern\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{makroname(k)}}}{{{M[k]}}}\n")
    for name, inhalt in T.items():
        open(os.path.join(DATA, f"{name}.tex"), "w").write(inhalt + "\n")
    print(f"[RECHNUNG] {len(M)} Makros, {len(T)} Tabellen")
    print(f"  Kurs {M['Kurs']} ({M['KursDatum']}), Boersenwert {M['Boersenwert']}, KBV {M['KBV']}")
    print(f"  Zufluss: PES {M['FcfPes']} MED {M['FcfMedian']} ERN {M['FcfErnte']} LTM {M['FcfLtm']} "
          f"(Rang LTM {M['FcfRangLtm']} %), Ernte-Median Instandhaltung {M['ErnteMedian']}")
    for k in szen:
        print(f"  {k}: Schwelle 5/10/15 {M[f'Schwelle{k}Fuenf']} / {M[f'Schwelle{k}Zehn']} / "
              f"{M[f'Schwelle{k}Fuenfzehn']} (10J bei {HUERDE_LANG}: {M[f'Schwelle{k}ZehnLang']}) | "
              f"Wachstum 10J {M[f'Wachstum{k}Zehn']} | Endwert {M[f'Endwert{k}Zehn']}x | IRR10 {M[f'Irr{k}Zehn']}")
    print(f"  EV/EBITDA median {M['EvMedian']} heute LTM {M['EvHeuteLtm']} (2025: {M['EvHeuteVoll']}), "
          f"Kurzfrist LTM {M['KurzfristLTM']} MED {M['KurzfristMED']} VOLL {M['KurzfristVOLL']}")
    print("  Dauer:", T["dauer"].replace("\\\\\n", " | ").replace("\\,\\%", "%"))
    print(f"  Prognose unter: {M['PrognoseUnter']}/5; Marge Rang HJ {M['MargeRangHj']} %, Faktor {M['MargeFaktorHj']}")
    print(f"  Real: CFO {M['CfoWachstumReal']} EBITDA {M['EbitdaWachstumReal']} FCF {M['FcfWachstumReal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
