#!/usr/bin/env python3
"""
rechnung_gea.py - Zahlenschicht und Rechnungen des Vollberichts GEA Group.

Bloecke wie im Muster: ROH (auf gedruckter Seite belegt), REIHEN (aus reihe.py und
kennzahlen_seite.py), EXTERN (Kurse, Analystenkonsens - Sekundaerquellen).

Gerechnet werden:
  * Teil 2  Mehrjahresreihe und Kennzahlenueberblick
  * Teil 3  Prognosetreue 2021-2025: Prognose WIE ZUERST GEGEBEN gegen das Ergebnis der
            eigenen Reihe; die Einordnung (unter/in/ueber) rechnet dieses Skript, sie
            wird nicht aus der Ergebnisspalte des Berichts uebernommen
  * Teil 4  Kurs gegen Auftragseingang (fuehrender Indikator)
  * Teil 5  drei Handlungsoptionen des Anlegers, je mit interner Verzinsung
  * Teil 6  Anlegerrechnung (unternehmensweit, Mio. EUR)
  * Teil 7  Schwellenkurs, erforderliches Wachstum, erforderlicher Endwert, Multiple

Aufruf: python3 scripts/rechnung_gea.py
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
NB = "n.\\,b."                      # LaTeX-Literale; in f-Ausdruecken ist kein \ erlaubt
MAL = "\\,$\\times$"
PROZ = "\\%"
HUERDE_QUELLE = "MSCI World, Gross Returns (USD), 10 Jahre annualisiert, Factsheet 31.08.2026"

# --------------------------------------------------------------------------
# ROH: (Wert wie gedruckt, Faktor auf die Einheit der Rechnung, Dokument, Seite)
# --------------------------------------------------------------------------
ROH = {
    "Aktien": (162801664, 1e-6, "gea-ar-2025", "16"),
    "AktienVorjahr": (172331076, 1e-6, "gea-ar-2025", "14"),
    "Eigenkapital": (2453007, 1e-3, "gea-ar-2025", "312"),
    "Nettoliquiditaet": (378.9, 1, "gea-ar-2025", "2"),
    "Auftragseingang": (5924.1, 1, "gea-ar-2025", "2"),
    "Auftragsbestand": (3339.2, 1, "gea-ar-2025", "44"),
    "Umsatz": (5495.4, 1, "gea-ar-2025", "2"),
    "EbitdaVr": (907.4, 1, "gea-ar-2025", "2"),
    "EbitdaMarge": (16.5, 1, "gea-ar-2025", "2"),
    "Roce": (36.2, 1, "gea-ar-2025", "2"),
    "FreierCashflowGea": (511.8, 1, "gea-ar-2025", "2"),
    "Eps": (2.54, 1, "gea-ar-2025", "2"),
    "Mitarbeiter": (18628, 1, "gea-ar-2025", "2"),
    "Serviceanteil": (40.0, 1, "gea-ar-2025", "2"),
    "Dividende": (1.30, 1, "gea-ar-2025", "2"),
    "DividendeVorjahr": (1.15, 1, "gea-ar-2025", "2"),
    "Streubesitz": (89.5, 1, "gea-ar-2025", "14"),
    "Grossaktionaere": (10.5, 1, "gea-ar-2025", "14"),
    "Privatanleger": (9.4, 1, "gea-ar-2025", "14"),
    "IdentifiziertAnteil": (97.2, 1, "gea-ar-2025", "14"),
    "RueckkaufStueckAlt": (9529412, 1e-6, "gea-ar-2025", "14"),
    "RueckkaufPreisAlt": (41.98, 1, "gea-ar-2025", "14"),
    "KursJahresende": (57.80, 1, "gea-ar-2025", "14"),
    "AnalystPositiv": (37, 1, "gea-ar-2025", "15"),
    "AnalystNeutral": (42, 1, "gea-ar-2025", "15"),
    "AnalystNegativ": (21, 1, "gea-ar-2025", "15"),
    "AusschuettungsZiel": (50, 1, "gea-ar-2025", "16"),
    # Widerspruch im Bericht: der Prognose-Ist-Vergleich auf S. 40 nennt als Ergebnis
    # 16,2 %, die Kennzahlenseite und die Prosa nennen 16,5 %. Beide Werte sind gedruckt.
    "MargeVergleich": (16.2, 1, "gea-ar-2025", "40"),
    "Institutionelle": (77.3, 1, "gea-ar-2025", "14"),
    "MissionWachstum": (5, 1, "gea-ar-2025", "23"),
    "MissionMargeVon": (17, 1, "gea-ar-2025", "23"),
    "MissionMargeBis": (19, 1, "gea-ar-2025", "23"),
    "MissionRoce": (45, 1, "gea-ar-2025", "23"),
    "PrognoseWachstumVon": (5.0, 1, "gea-ar-2025", "2"),
    "PrognoseWachstumBis": (7.0, 1, "gea-ar-2025", "2"),
    "PrognoseMargeVon": (16.6, 1, "gea-ar-2025", "2"),
    "PrognoseMargeBis": (17.2, 1, "gea-ar-2025", "2"),
    # Halbjahresbericht 2026
    "HjAuftragseingang": (2949.0, 1, "gea-hj-2026", "2"),
    "HjUmsatz": (2715.6, 1, "gea-hj-2026", "2"),
    "HjEbitdaVr": (456.5, 1, "gea-hj-2026", "2"),
    "HjAuftragsbestand": (3540.4, 1, "gea-hj-2026", "2"),
    "HjAuftragseingangVorjahr": (2724.0, 1, "gea-hj-2026", "2"),
    "HjEbitdaVrVorjahr": (415.0, 1, "gea-hj-2026", "2"),
    "HjWachstumVon": (6.0, 1, "gea-hj-2026", "7"),
    "HjWachstumBis": (8.0, 1, "gea-hj-2026", "7"),
    "HjMargeVon": (17.0, 1, "gea-hj-2026", "7"),
    "HjMargeBis": (17.4, 1, "gea-hj-2026", "7"),
    "RueckkaufProgramm": (400, 1, "gea-hj-2026", "44"),
    "RueckkaufTrancheZwei": (250, 1, "gea-hj-2026", "44"),
    "RueckkaufStueck": (5378681, 1e-6, "gea-hj-2026", "44"),
}
# EXTERN: Sekundaerquellen ohne gedruckte Seite
EXTERN = {
    "KonsensAnalysten": 17, "KonsensZiel": 70.26, "KonsensHoch": 78.00, "KonsensTief": 53.00,
    "KonsensKurs": 64.50, "KonsensAufschlag": 8.94,
    "DdKaeufe": 6, "DdPersonen": 4, "DdPreisVon": 53.60, "DdPreisBis": 59.60,
    "DdKlebertVolumen": 597350.0,
}

# Nachkommastellen, wo Python die gedruckte Schreibweise nicht kennt: 1.30 wird als
# "1.3" gespeichert, die Seite druckt "1.30". Die Stelle ist Darstellung, nicht Wert.
NACHKOMMA = {"Dividende": 2, "DividendeVorjahr": 2, "KursJahresende": 2, "RueckkaufPreisAlt": 2,
             "Serviceanteil": 1, "HjMargeVon": 1, "HjMargeBis": 1, "HjWachstumVon": 1,
             "HjWachstumBis": 1, "HjAuftragseingang": 1, "HjAuftragseingangVorjahr": 1,
             "HjEbitdaVrVorjahr": 1, "PrognoseWachstumVon": 1, "PrognoseWachstumBis": 1}

R = json.load(open(os.path.join(DATA, "reihen.json")))["gea"]
K = json.load(open(os.path.join(DATA, "kennzahlen_seite.json")))
P = json.load(open(os.path.join(DATA, "prognose.json")))
KURSE = json.load(open(os.path.join(DATA, "kurse_roh.json")))["G1A.DE"]
JAHRE = list(range(2019, 2026))


def roh(n):
    w, f, _, _ = ROH[n]
    return w * f


def reihe(posten):
    return {int(j): v["wert"] for j, v in R.get(posten, {}).items()}


def kennzahl(posten):
    return {int(j): v[posten]["wert"] for j, v in K.items() if posten in v}


def kz(posten, jahr):
    return K.get(str(jahr), {}).get(posten, {}).get("wert")


def fcf() -> dict:
    """Ausschuettungsfaehiger freier Mittelzufluss: operativ + Investitionen + Leasingtilgung."""
    cfo, cap, lea = reihe("operativer_cf"), reihe("investitionen"), reihe("leasing")
    return {j: cfo[j] + cap[j] + lea.get(j, 0.0) for j in sorted(cfo) if j in cap}


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


def lage(ergebnis, zelle: str) -> str:
    """Einordnung des Ergebnisses in die zuerst gegebene Prognose (unter / in / ueber)."""
    zahlen = [float(x) for x in __import__("re").findall(r"-?\d+\.?\d*", zelle.replace(",", ""))]
    if ergebnis is None or len(zahlen) < 1 or "rising" in zelle or "unchanged" in zelle:
        return "nicht quantifiziert"
    lo, hi = (zahlen[0], zahlen[-1]) if len(zahlen) > 1 else (zahlen[0], None)
    if hi is None:
        return "erreicht" if ergebnis >= lo else "verfehlt"
    return "darunter" if ergebnis < lo else ("darueber" if ergebnis > hi else "in der Spanne")


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
    f = fcf()
    n_akt = roh("Aktien")
    kurs = KURSE["letzter"][1]
    mcap = kurs * n_akt
    bw = roh("Eigenkapital")
    werte = list(f.values())
    med, pes, opt = statistics.median(werte), perzentil(werte, 0.25), perzentil(werte, 0.75)

    for name in ROH:
        w, fak, _, _ = ROH[name]
        stellen = len(str(w).split(".")[1]) if isinstance(w, float) and w % 1 else 0
        M[name] = de(w * fak, 2 if fak != 1 else NACHKOMMA.get(name, stellen))
    for k, v in EXTERN.items():
        M[k] = de(v, 2 if isinstance(v, float) and v % 1 else 0)

    M.update({
        "Kurs": de(kurs, 2), "KursDatum": ".".join(reversed(KURSE["letzter"][0].split("-"))),
        "AktienMio": de(n_akt, 1), "Boersenwert": de(mcap), "BuchwertJeAktie": de(bw / n_akt, 2),
        "KBV": de(mcap / bw, 2), "Huerde": de(HUERDE, 2), "HuerdeLang": de(HUERDE_LANG, 2),
        "FcfMedian": de(med), "FcfPes": de(pes), "FcfOpt": de(opt), "FcfVoll": de(f[2025]),
        "FcfMin": de(min(werte)), "FcfMax": de(max(werte)),
        "FcfMinJahr": str(min(f, key=f.get)), "FcfMaxJahr": str(max(f, key=f.get)),
        "FcfJahre": f"{min(f)}--{max(f)}", "FcfRendite": pz(100 * f[2025] / mcap),
        "DividendeSumme": de(abs(reihe("dividenden")[2025])),
        "RueckkaufSumme": de(abs(reihe("rueckkauf")[2025])),
        "AusschuettungAnteil": pz(100 * (abs(reihe("dividenden")[2025]) + abs(reihe("rueckkauf")[2025])) / f[2025]),
    })

    # Teil 2: Reihe und Kennzahlenueberblick
    zeilen = []
    for j in JAHRE:
        q = R["operativer_cf"][str(j)]
        zeilen.append(f"{j} & {de(kz('auftragseingang', j))} & {de(kz('umsatz', j))} & "
                      f"{de(kz('ebitda_vr', j))} & {de(kz('ebitda_marge', j), 1)} & "
                      f"{de(kz('roce', j), 1)} & {de(f[j])} & {de(reihe('eps_verwaessert')[j], 2)} & "
                      f"{q['bericht'][-4:]}, S.~{q['seite']}")
    T["reihe"] = " \\\\\n".join(zeilen) + " \\\\"

    ueberblick = [("Auftragseingang", "auftragseingang", 0), ("Umsatz", "umsatz", 0),
                  ("EBITDA vor Restrukturierung", "ebitda_vr", 0), ("EBITDA-Marge in \\%", "ebitda_marge", 1),
                  ("ROCE in \\%", "roce", 1), ("Freier Cashflow (GEA)", "freier_cashflow", 0),
                  ("Nettoliquiditaet", "nettoliquiditaet", 0), ("Serviceanteil in \\%", "serviceanteil", 1),
                  ("Ergebnis je Aktie in EUR", "eps", 2), ("Mitarbeiter (FTE)", "mitarbeiter", 0)]
    zeilen = []
    for label, posten, st in ueberblick:
        a, b = kz(posten, 2025), kz(posten, 2024)
        d = (a - b) if a is not None and b is not None else None
        zeilen.append(f"{label} & {de(a, st)} & {de(b, st)} & {de(d, max(st, 1))}")
    T["ueberblick"] = " \\\\\n".join(zeilen) + " \\\\"

    # Teil 3: Prognosetreue, Prognose wie zuerst gegeben
    ist = {"umsatz": kennzahl("organisches_wachstum"), "ebitda": kennzahl("ebitda_vr"),
           "marge": kennzahl("ebitda_marge"), "roce": kennzahl("roce")}
    bez = {"umsatz": "Organisches Umsatzwachstum", "ebitda": "EBITDA vor Restrukturierung (Mio. EUR)",
           "marge": "EBITDA-Marge vor Restrukturierung", "roce": "ROCE"}
    zeilen, treffer, quantifiziert = [], 0, 0
    for b in P:
        for posten, v in b["posten"].items():
            e = ist.get(posten, {}).get(b["jahr"])
            l = lage(e, v["erst"])
            if l not in ("nicht quantifiziert",):
                quantifiziert += 1
                treffer += l in ("in der Spanne", "darueber", "erreicht")
            zelle = v["erst"].replace("%", PROZ)
            ergebnis = de(e, 1) if e is not None else "--"
            zeilen.append(f"{b['jahr']} & {bez[posten]} & {zelle} & "
                          f"{ergebnis} & {l} & S.~{b['seite']}")
    T["prognose"] = " \\\\\n".join(zeilen) + " \\\\"
    M["PrognoseTreffer"] = str(treffer)
    M["PrognoseQuantifiziert"] = str(quantifiziert)

    # Teil 4: Kurs gegen Auftragseingang
    ye = {int(j): p for j, p in KURSE["jahresende"].items()}
    zeilen = []
    for j in JAHRE[1:]:
        ae, ae_v = kz("auftragseingang", j), kz("auftragseingang", j - 1)
        dk = 100 * (ye[j] / ye[j - 1] - 1) if j in ye and j - 1 in ye else None
        da = 100 * (ae / ae_v - 1) if ae and ae_v else None
        zeilen.append(f"{j} & {de(ae)} & {de(da, 1)} & {de(ye.get(j), 2)} & {de(dk, 1)}")
    T["kurs"] = " \\\\\n".join(zeilen) + " \\\\"

    # Teile 5-7: Anlegerrechnung je Szenario, Optionen, Umkehrungen
    # Drei Szenarien aus der eigenen Reihe 2019--2025, nicht aus einer Schaetzung. Das
    # Geschaeftsjahr 2025 (400,4) faellt mit dem Median zusammen und ist deshalb kein
    # eigenes Szenario; der optimistische Fall ist das 75. Perzentil derselben Reihe.
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
        M[f"Vielfaches{schl}"] = de(kurs / (s[1] / n_akt), 1) if s[1] else NB
    T["schwelle"] = " \\\\\n".join(zeilen_s) + " \\\\"
    T["wachstum"] = " \\\\\n".join(zeilen_w) + " \\\\"
    T["endwert"] = " \\\\\n".join(zeilen_e) + " \\\\"

    # Teil 5: Optionen des Anlegers - interne Verzinsung auf zehn Jahre je Szenario
    optionen = {"A": ("Kauf heute", mcap), "B": ("Kauf zur Schwelle (Basis)", None)}
    schwelle_basis = ci.break_even_investment(konfig(mcap, med, bw, HUERDE), "X", START + 10)
    optionen["B"] = ("Kauf zum Schwellenkurs (Basis)", schwelle_basis)
    indexwert = mcap * (1 + HUERDE / 100) ** 10
    zeilen = []
    for schl, (bez_o, einstieg) in optionen.items():
        werte_irr = [ci.irr(ci.full_cashflows(konfig(einstieg, z, bw, HUERDE), "X", START + 10))
                     for _, z in szenarien.values()]
        zeilen.append(f"{bez_o} & {de(einstieg / n_akt, 2)} & " +
                      " & ".join(pz(100 * x) for x in werte_irr))
        M[f"Option{schl}Kurs"] = de(einstieg / n_akt, 2)
        M[f"Option{schl}IrrMed"] = pz(100 * werte_irr[1])
    irr_c = ci.irr(ci.full_cashflows(
        {**konfig(mcap, 0.0, indexwert, HUERDE)}, "X", START + 10))
    zeilen.append("Verzicht, Indexanlage & " + de(kurs, 2) + " & " + " & ".join([pz(100 * irr_c)] * 3))
    M["OptionCIrr"] = pz(100 * irr_c)
    T["optionen"] = " \\\\\n".join(zeilen) + " \\\\"

    # Teil 7: Multiple im Kontext
    eps = reihe("eps_verwaessert")
    kgv = {j: ye[j] / eps[j] for j in sorted(eps) if j in ye and eps[j] > 0}
    M["KgvHeute"] = de(kurs / eps[2025], 1)
    M["KgvMedian"] = de(statistics.median(kgv.values()), 1)
    M["KgvJahre"] = f"{min(kgv)}--{max(kgv)}"
    M["KgvRang"] = de(100 * sum(1 for v in kgv.values() if v < kurs / eps[2025]) / len(kgv), 0)
    T["kgv"] = " \\\\\n".join(f"{j} & {de(ye[j], 2)} & {de(eps[j], 2)} & {de(v, 1)}"
                              for j, v in kgv.items()) + " \\\\"

    # Dividende je Aktie aus der eigenen Reihe: Zahlung des Jahres durch die Aktienzahl
    # desselben Jahres. Die Reihe bestaetigt die Politik der Seite 16 (jaehrlich 5 Cent
    # mehr bis 2024, danach zweimal 15 Cent) aus den Zahlungsstroemen heraus.
    div, akt = reihe("dividenden"), reihe("aktien_verwaessert")
    dps = {j: abs(div[j]) / akt[j] for j in sorted(div) if j in akt}
    T["dividende"] = " \\\\\n".join(
        f"{j} & {de(abs(div[j]))} & {de(akt[j], 1)} & {de(dps[j], 2)} & "
        f"{de(dps[j] - dps[j - 1], 2) if j - 1 in dps else '--'}" for j in dps) + " \\\\"
    M["DpsRechnerisch"] = de(dps[2025], 2)
    M["DpsSchritt"] = de(dps[2025] - dps[2024], 2)
    ergebnis25 = reihe("ergebnis")[2025]
    M["AusschuettungsQuote"] = pz(100 * roh("Dividende") * n_akt / ergebnis25)
    M["Jahresergebnis"] = de(ergebnis25)
    M["DividendeSummeKuenftig"] = de(roh("Dividende") * n_akt)
    M["MargeBerechnet"] = pz(100 * roh("EbitdaVr") / roh("Umsatz"), 2)
    M["RueckkaufVolumenAlt"] = de(roh("RueckkaufStueckAlt") * roh("RueckkaufPreisAlt"))
    M["RueckkaufAnteil"] = pz(100 * roh("RueckkaufStueck") / n_akt)
    M["AktienRueckgang"] = pz(100 * (1 - n_akt / roh("AktienVorjahr")))
    M["EkQuote"] = pz(kz("eigenkapitalquote", 2025))
    # Mission 30: implizites EBITDA-Wachstum aus den eigenen Zielen
    marge_ziel = (roh("MissionMargeVon") + roh("MissionMargeBis")) / 2
    implizit = ((1 + roh("MissionWachstum") / 100) ** 5 * marge_ziel / roh("EbitdaMarge")) ** (1 / 5) - 1
    M["MissionImplizit"] = pz(100 * implizit)
    M["MissionMargeMitte"] = de(marge_ziel, 1)
    # Halbjahr 2026
    M["HjMarge"] = pz(100 * roh("HjEbitdaVr") / roh("HjUmsatz"))
    M["HjAuftragWachstum"] = pz(100 * (roh("HjAuftragseingang") / roh("HjAuftragseingangVorjahr") - 1))
    M["Auftragsreichweite"] = de(12 * roh("Auftragsbestand") / roh("Umsatz"), 1)

    with open(os.path.join(DATA, "kennzahlen.tex"), "w") as fo:
        fo.write("% erzeugt von scripts/rechnung_gea.py - nicht von Hand aendern\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for name, inhalt in T.items():
        open(os.path.join(DATA, f"{name}.tex"), "w").write(inhalt + "\n")
    print(f"[RECHNUNG] {len(M)} Makros, {len(T)} Tabellen")
    print(f"  Kurs {M['Kurs']} | Schwelle Basis 10J {M['SchwelleMEDZehn']} (bei {M['HuerdeLang']}: "
          f"{M['SchwelleMEDZehnLang']}) | Wachstum verlangt {M['WachstumMEDZehn']} | "
          f"Mission 30 impliziert {M['MissionImplizit']}")
    print(f"  KGV {M['KgvHeute']} (Median {M['KgvMedian']}, Rang {M['KgvRang']}) | "
          f"Prognosetreue {M['PrognoseTreffer']}/{M['PrognoseQuantifiziert']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
