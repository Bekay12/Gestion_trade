#!/usr/bin/env python3
"""
rechnung_daimler.py - Zahlenschicht und Rechnungen des vertieften Filters Daimler Truck.

Tiefe wie der urspruengliche Dual-Champions-Filter (Eigentuemer, Wettbewerbsposition,
Mehrjahresreihe, Kursverlauf, Votum mit Schwellen- und Umkehrtabellen), nicht der
siebenteilige Vollbericht. Vier Geschaeftsjahre (2022-2025, erstes Jahr nach dem
Spin-off von der Mercedes-Benz Group im Dezember 2021), jedes Jahr aus zwei Berichten
gelesen (Kettenpruefung: der juengere Bericht druckt das Vorjahr erneut).

Abweichend von GEA: Daimler Truck weist einen eigenen "Free cash flow of the Industrial
Business" aus (ohne das Segment Financial Services). Dieser Wert wird uebernommen statt
selbst aus operativem Mittelzufluss und Investitionen gerechnet zu werden, weil der
operative Mittelzufluss der GESAMTEN Gruppe durch das Finanzdienstleistungsgeschaeft
(wachsende Forderungen aus Absatzfinanzierung) stark schwankt und damit kein Mass fuer
den Zufluss an den Aktionaer waere - siehe die Annahme im Rahmenabschnitt.

Aufruf: python3 scripts/rechnung_daimler.py
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
NB = "n.\\,b."
MAL = "\\,$\\times$"
HUERDE_QUELLE = "MSCI World, Gross Returns (USD), 10 Jahre annualisiert, Factsheet 31.08.2026"

# --------------------------------------------------------------------------
# ROH: (Wert wie gedruckt, Faktor auf Mio. EUR bzw. Einheit der Rechnung, Dokument, Seite)
# --------------------------------------------------------------------------
ROH = {
    "Aktien": (766, 1, "daimler-ar-2025", "281"),
    "Eigenkapital": (21552, 1, "daimler-ar-2025", "201"),
    "EigenkapitalVorjahr": (22205, 1, "daimler-ar-2025", "201"),
    "NettoliquiditaetInd": (7670, 1, "daimler-ar-2025", "2"),
    "Umsatz": (49387, 1, "daimler-ar-2025", "2"),
    "UmsatzInd": (45911, 1, "daimler-ar-2025", "2"),
    "Ebit": (2945, 1, "daimler-ar-2025", "2"),
    "EbitBereinigt": (3778, 1, "daimler-ar-2025", "2"),
    "EbitBereinigtInd": (3596, 1, "daimler-ar-2025", "2"),
    "UmsatzrenditeInd": (6.1, 1, "daimler-ar-2025", "2"),
    "UmsatzrenditeBereinigtInd": (7.8, 1, "daimler-ar-2025", "2"),
    "RoceInd": (26.3, 1, "daimler-ar-2025", "2"),
    "Jahresueberschuss": (2033, 1, "daimler-ar-2025", "2"),
    "Eps": (2.56, 1, "daimler-ar-2025", "2"),
    "FcfInd": (1824, 1, "daimler-ar-2025", "2"),
    "FcfBereinigtInd": (2182, 1, "daimler-ar-2025", "2"),
    "Absatz": (422510, 1e-3, "daimler-ar-2025", "2"),
    "AbsatzNull": (6726, 1e-3, "daimler-ar-2025", "2"),
    "DividendeSumme": (1462, 1, "daimler-ar-2025", "202"),
    "RueckkaufSumme": (616, 1, "daimler-ar-2025", "202"),
    "Betriebsergebnis": (4336, 1, "daimler-ar-2025", "202"),
    "InvestitionenSachanlagen": (1117, 1, "daimler-ar-2025", "202"),
    "InvestitionenImmateriell": (358, 1, "daimler-ar-2025", "202"),
    # Eigentuemer, Streubesitz (Stand 31.12.2025)
    "MbAnteil": (30.01, 1, "daimler-ar-2025", "4"),
    "MbPensionAnteil": (5.19, 1, "daimler-ar-2025", "4"),
    "KiaAnteil": (4.98, 1, "daimler-ar-2025", "4"),
    "InstitutionelleAnteil": (44.57, 1, "daimler-ar-2025", "4"),
    "PrivatanlegerAnteil": (15.25, 1, "daimler-ar-2025", "4"),
    "KursJahreshoch": (45.05, 1, "daimler-ar-2025", "4"),
    "KursJahrestief": (31.75, 1, "daimler-ar-2025", "4"),
    "AktienStueckChart": (765600, 1e-3, "daimler-ar-2025", "4"),
}
# EXTERN: Sekundaerquellen ohne gedruckte Seite
EXTERN = {
    "KonsensAnalysten": 18, "KonsensZiel": 51.26,
    "DdPhantomDatum": None,  # Text, kein Zahlenmakro
}

# Reihen aus den vier Jahresberichten, je Groesse als {jahr: (wert, bericht, seite)}.
REIHE_ROH = {
    "umsatz": {2022: (50945, "daimler-ar-2023", "2"), 2023: (55890, "daimler-ar-2024", "2"),
               2024: (54077, "daimler-ar-2025", "2"), 2025: (49387, "daimler-ar-2025", "2")},
    "umsatz_ind": {2022: (49186, "daimler-ar-2023", "2"), 2023: (53216, "daimler-ar-2024", "2"),
                   2024: (50743, "daimler-ar-2025", "2"), 2025: (45911, "daimler-ar-2025", "2")},
    "ebit_bereinigt": {2022: (3959, "daimler-ar-2023", "2"), 2023: (5489, "daimler-ar-2024", "2"),
                       2024: (4667, "daimler-ar-2025", "2"), 2025: (3778, "daimler-ar-2025", "2")},
    "umsatzrendite_bereinigt": {2022: (7.7, "daimler-ar-2023", "2"), 2023: (9.9, "daimler-ar-2024", "2"),
                                2024: (8.9, "daimler-ar-2025", "2"), 2025: (7.8, "daimler-ar-2025", "2")},
    "roce_ind": {2022: (28.9, "daimler-ar-2023", "2"), 2023: (44.6, "daimler-ar-2024", "2"),
                 2024: (31.1, "daimler-ar-2025", "2"), 2025: (26.3, "daimler-ar-2025", "2")},
    "eps": {2022: (3.24, "daimler-ar-2023", "2"), 2023: (4.62, "daimler-ar-2024", "2"),
            2024: (3.64, "daimler-ar-2025", "2"), 2025: (2.56, "daimler-ar-2025", "2")},
    "fcf_ind": {2022: (1746, "daimler-ar-2023", "2"), 2023: (2811, "daimler-ar-2024", "2"),
                2024: (3152, "daimler-ar-2025", "2"), 2025: (1824, "daimler-ar-2025", "2")},
    "nettoliquiditaet_ind": {2022: (7530, "daimler-ar-2023", "2"), 2023: (8322, "daimler-ar-2024", "2"),
                             2024: (8558, "daimler-ar-2025", "2"), 2025: (7670, "daimler-ar-2025", "2")},
    "absatz": {2022: (520291, "daimler-ar-2023", "2"), 2023: (526053, "daimler-ar-2024", "2"),
               2024: (460409, "daimler-ar-2025", "2"), 2025: (422510, "daimler-ar-2025", "2")},
    "cfo": {2022: (-523, "daimler-ar-2023", "176"), 2023: (386, "daimler-ar-2023", "176"),
            2024: (1555, "daimler-ar-2025", "202"), 2025: (4336, "daimler-ar-2025", "202")},
    "dividenden": {2022: (0, "daimler-ar-2023", "176"), 2023: (-1070, "daimler-ar-2023", "176"),
                   2024: (-1528, "daimler-ar-2025", "202"), 2025: (-1462, "daimler-ar-2025", "202")},
    "rueckkauf": {2022: (0, "daimler-ar-2023", "176"), 2023: (-557, "daimler-ar-2023", "176"),
                  2024: (-850, "daimler-ar-2025", "202"), 2025: (-616, "daimler-ar-2025", "202")},
}
JAHRE = (2022, 2023, 2024, 2025)
KURSE = {2022: 28.945, 2023: 34.02, 2024: 36.85, 2025: 37.32}  # Yahoo Finance, Jahresendkurse


def roh(n):
    w, f, _, _ = ROH[n]
    return w * f


def reihe(posten):
    return {j: v[0] for j, v in REIHE_ROH[posten].items()}


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


def konfig(einstieg: float, zufluss: float, endwert: float, huerde: float) -> dict:
    return {"currency": "Mio. EUR", "opening_liquidity": 0.0, "horizon_start": START,
            "horizon_end": START + max(HORIZONTE), "schedule_years": [START, START + 1],
            "sensitivity_horizons": [START + h for h in HORIZONTE], "revenue_base": 1.0,
            "earnings_base": 1.0, "hurdle": huerde, "hurdle_source": HUERDE_QUELLE,
            "revenue_scenarios": [],
            "options": {"X": {"name": "Anlegerrechnung", "investment": {str(START): einstieg},
                              "earnings_uplift": zufluss, "uplift_start": START + 1,
                              "terminal_value": endwert}}}


def main() -> int:
    M, T = {}, {}
    n_akt = roh("Aktien")
    kurs = 43.49  # Yahoo Finance, 18.09.2026
    mcap = kurs * n_akt
    bw = roh("Eigenkapital")
    f = reihe("fcf_ind")
    werte = list(f.values())
    med, pes, opt = statistics.median(werte), perzentil(werte, 0.25), perzentil(werte, 0.75)

    for name in ROH:
        w, fak, _, _ = ROH[name]
        stellen = 2 if fak != 1 else (2 if name in ("KursJahreshoch", "KursJahrestief") else 0)
        M[name] = de(w * fak, stellen)
    M["KonsensAnalysten"] = str(EXTERN["KonsensAnalysten"])
    M["KonsensZiel"] = de(EXTERN["KonsensZiel"], 2)
    M["KonsensAufschlag"] = pz(100 * (EXTERN["KonsensZiel"] / kurs - 1))

    M.update({
        "Kurs": de(kurs, 2), "KursDatum": "18.09.2026",
        "AktienMio": de(n_akt), "Boersenwert": de(mcap),
        "BuchwertJeAktie": de(bw / n_akt, 2), "KBV": de(mcap / bw, 2),
        "Huerde": de(HUERDE, 2), "HuerdeLang": de(HUERDE_LANG, 2),
        "FcfMedian": de(med), "FcfPes": de(pes), "FcfOpt": de(opt),
        "FcfJahre": f"{min(f)}--{max(f)}", "FcfMinJahr": str(min(f, key=f.get)),
        "FcfMaxJahr": str(max(f, key=f.get)), "FcfMin": de(min(werte)), "FcfMax": de(max(werte)),
        "FcfRendite": pz(100 * f[2025] / mcap),
        "AusschuettungAnteil": pz(100 * (roh("DividendeSumme") + roh("RueckkaufSumme")) / f[2025])
        if f[2025] > 0 else NB,
        "FcfHoch": de(f[2023]), "KursStart": de(KURSE[2022], 2),
        "KursZuwachs": pz(100 * (kurs / KURSE[2022] - 1), 0),
        "EpsHoch": de(reihe("eps")[2023], 2), "EpsHeute": de(reihe("eps")[2025], 2),
    })

    # Mehrjahresreihe
    posten_liste = [("umsatz_ind", 0), ("ebit_bereinigt", 0), ("umsatzrendite_bereinigt", 1),
                    ("roce_ind", 1), ("fcf_ind", 0), ("eps", 2), ("absatz", 0)]
    zeilen = []
    for j in JAHRE:
        q = REIHE_ROH["fcf_ind"][j]
        vals = " & ".join(de(REIHE_ROH[p][j][0], st) for p, st in posten_liste)
        zeilen.append(f"{j} & {vals} & {q[1][-4:]}, S.~{q[2]}")
    T["reihe"] = " \\\\\n".join(zeilen) + " \\\\"

    # Kurs gegen Absatz (fuehrender Indikator umgekehrt: hier FALLENDER Absatz, STEIGENDER Kurs)
    absatz = reihe("absatz")
    zeilen = []
    for j in JAHRE[1:]:
        da = 100 * (absatz[j] / absatz[j - 1] - 1)
        dk = 100 * (KURSE[j] / KURSE[j - 1] - 1)
        zeilen.append(f"{j} & {de(absatz[j])} & {de(da, 1)} & {de(KURSE[j], 2)} & {de(dk, 1)}")
    T["kurs"] = " \\\\\n".join(zeilen) + " \\\\"

    # KGV-Reihe
    eps = reihe("eps")
    kgv = {j: KURSE[j] / eps[j] for j in JAHRE}
    M["KgvHeute"] = de(kurs / eps[2025], 1)
    M["KgvMedian"] = de(statistics.median(kgv.values()), 1)
    M["KgvJahre"] = f"{min(kgv)}--{max(kgv)}"
    M["KgvRang"] = de(100 * sum(1 for v in kgv.values() if v < kurs / eps[2025]) / len(kgv), 0)
    T["kgv"] = " \\\\\n".join(f"{j} & {de(KURSE[j], 2)} & {de(eps[j], 2)} & {de(v, 1)}"
                              for j, v in kgv.items()) + " \\\\"

    # Cash-Flow-Details (Ausschuettung)
    div, rk = reihe("dividenden"), reihe("rueckkauf")
    T["ausschuettung"] = " \\\\\n".join(
        f"{j} & {de(abs(div[j])) if div[j] else '--'} & {de(abs(rk[j])) if rk[j] else '--'} & "
        f"{de(f[j])} & {pz(100 * (abs(div[j]) + abs(rk[j])) / f[j], 0) if f[j] > 0 else NB}"
        for j in JAHRE) + " \\\\"

    # Schwelle, Umkehrungen (drei Szenarien der eigenen FCF-Ind.-Reihe)
    szenarien = {"PES": ("pessimistisch (25.~Perzentil)", pes), "MED": ("Basis (Median)", med),
                 "OPT": ("optimistisch (75.~Perzentil)", opt)}
    zeilen_s, zeilen_w, zeilen_e = [], [], []
    for schl, (name, zufluss) in szenarien.items():
        cfg = konfig(mcap, zufluss, bw, HUERDE)
        cfgl = konfig(mcap, zufluss, bw, HUERDE_LANG)
        s = [ci.break_even_investment(cfg, "X", START + h) for h in HORIZONTE]
        sl = ci.break_even_investment(cfgl, "X", START + 10)
        g = [ci.required_growth(cfg, "X", START + h) for h in HORIZONTE]
        e = [ci.required_terminal_value(cfg, "X", START + h) for h in HORIZONTE]
        zeilen_s.append(f"{name} & {de(zufluss)} & "
                        + " & ".join(de(x / n_akt, 2) if x else NB for x in s)
                        + " & " + (de(sl / n_akt, 2) if sl else NB))
        zeilen_w.append(f"{name} & " + " & ".join(pz(100 * x) if x is not None else NB for x in g))
        zeilen_e.append(f"{name} & " + " & ".join((de(x / bw, 1) + MAL) if x is not None else NB
                                                 for x in e))
        M[f"Schwelle{schl}Zehn"] = de(s[1] / n_akt, 2) if s[1] else NB
        M[f"Schwelle{schl}ZehnLang"] = de(sl / n_akt, 2) if sl else NB
        M[f"Wachstum{schl}Zehn"] = pz(100 * g[1]) if g[1] is not None else NB
        M[f"Endwert{schl}Zehn"] = de(e[1] / bw, 1) if e[1] is not None else NB
        M[f"Vielfaches{schl}"] = de(kurs / (s[1] / n_akt), 1) if s[1] and s[1] > 0 else NB
    T["schwelle"] = " \\\\\n".join(zeilen_s) + " \\\\"
    T["wachstum"] = " \\\\\n".join(zeilen_w) + " \\\\"
    T["endwert"] = " \\\\\n".join(zeilen_e) + " \\\\"

    with open(os.path.join(DATA, "kennzahlen.tex"), "w") as fo:
        fo.write("% erzeugt von scripts/rechnung_daimler.py - nicht von Hand aendern\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for name, inhalt in T.items():
        open(os.path.join(DATA, f"{name}.tex"), "w").write(inhalt + "\n")
    print(f"[RECHNUNG] {len(M)} Makros, {len(T)} Tabellen")
    print(f"  Kurs {M['Kurs']} | FCF-Reihe {M['FcfJahre']} ({M['FcfMin']}..{M['FcfMax']}) | "
          f"Schwelle Basis 10J {M['SchwelleMEDZehn']} | Wachstum verlangt {M['WachstumMEDZehn']}")
    print(f"  KGV {M['KgvHeute']} (Median {M['KgvMedian']}, Rang {M['KgvRang']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
