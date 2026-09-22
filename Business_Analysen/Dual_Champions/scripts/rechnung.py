#!/usr/bin/env python3
"""
rechnung.py - Zahlenschicht und Anlegerrechnung (Teil 7) fuer die sechs Titel.

Erzeugt data/kennzahlen.tex (die einzige Zahlenquelle der Abschnitte) und die
Tabellenkoerper data/<titel>_*.tex. Drei Bloecke, wie im Muster (Pharma_AZN_RDY):

    ROH      auf einer gedruckten Seite belegt; pruefe_seiten.py prueft jeden Wert
    REIHEN   aus data/reihen.json (reihe.py: Kettenpruefung und XBRL-Abgleich)
    EXTERN   Sekundaerquellen: Kurse (Yahoo Finance), Huerde (MSCI-Factsheet)

Die Anlegerrechnung ist unternehmensweit gerechnet (Mio. in Berichtswaehrung):
Einstieg = Boersenwert heute, Zufluss = ausschuettungsfaehiger freier Mittel-
zufluss, konstant gehalten, Endwert = Buchwert des Stammeigenkapitals 2025. Der
Schwellenkurs je Aktie ist der Schwellen-Boersenwert geteilt durch die Aktienzahl.
Die Rechnung ist vor Steuern des Anlegers (report-template.md, Regel 9).

Aufruf: python3 scripts/rechnung.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cashflow_irr as ci                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START = 2026
HORIZONTE = (5, 10, 15)
HUERDE, HUERDE_LANG = 13.56, 9.08
HUERDE_QUELLE = "MSCI World, Gross Returns (USD), 10 Jahre annualisiert, Factsheet Stand 31.08.2026, S. 1"
KURS_QUELLE = "Yahoo Finance (yfinance), abgerufen 19.09.2026"

TITEL = ["Oxy", "Tte", "Tnk", "Fro", "Gea", "Besi"]
KUERZEL = {"Oxy": "oxy", "Tte": "tte", "Tnk": "tnk", "Fro": "fro", "Gea": "gea", "Besi": "besi"}
TICKER = {"Oxy": "OXY", "Tte": "TTE", "Tnk": "TNK", "Fro": "FRO", "Gea": "G1A.DE", "Besi": "BESI.AS"}
WAEHRUNG = {"Oxy": "USD", "Tte": "USD", "Tnk": "USD", "Fro": "USD", "Gea": "EUR", "Besi": "EUR"}
PAARE = [("Oxy", "Tte"), ("Tnk", "Fro"), ("Gea", "Besi")]
NAME = {"Oxy": "Occidental", "Tte": "TotalEnergies", "Tnk": "Teekay Tankers", "Fro": "Frontline",
        "Gea": "GEA", "Besi": "BESI"}

# --------------------------------------------------------------------------
# ROH: Wert wie gedruckt, Faktor auf Mio. (bzw. Stueck/Prozent), Dokument, Seite.
# Das Dokument muss in pruefe_seiten.DOKUMENTE stehen; "?" = Deckblatt ohne Seitenzahl.
# --------------------------------------------------------------------------
ROH = {
    "OxyAktien": (986266656, 1e-6, "oxy-10-k-2025", "?"),
    "OxyEigenkapital": (36034, 1, "oxy-10-k-2025", "59"),
    "OxyVorzugsaktien": (8287, 1, "oxy-10-k-2025", "59"),
    "OxyBerkshireAnteil": (32.43, 1, "oxy-def14a-2026", "65"),
    "OxyBerkshireOptionsscheine": (83911942, 1e-6, "oxy-def14a-2026", "65"),
    "OxyDodgeAnteil": (8.38, 1, "oxy-def14a-2026", "65"),
    "OxyVanguardAnteil": (8.09, 1, "oxy-def14a-2026", "65"),
    "TteAktien": (2206585543, 1e-6, "tte-20-f-2025", "?"),
    "TteEigenkapital": (114883, 1, "tte-20-f-2025", "F-11"),
    "TteBlackrockAnteil": (6.5, 1, "tte-urd-2025", "430"),
    "TteMitarbeiterAnteil": (8.9, 1, "tte-urd-2025", "430"),
    "TteFcpeAnteil": (5.5, 1, "tte-urd-2025", "430"),
    "TteAfrikaAnteil": (15, 1, "tte-urd-2025", "127"),
    "TnkAktienA": (29921732, 1e-6, "tnk-20-f-2025", "2"),
    "TnkAktienB": (4625997, 1e-6, "tnk-20-f-2025", "66"),
    "TnkEigenkapital": (2043616, 1e-3, "tnk-20-f-2025", "F-5"),
    "TnkTeekayAnteil": (30.7, 1, "tnk-20-f-2025", "66"),
    "FroAktien": (222622889, 1e-6, "fro-20-f-2025", "?"),
    "FroEigenkapital": (2511350, 1e-3, "fro-20-f-2025", "F-6"),
    "FroHemenAktien": (79145703, 1e-6, "fro-20-f-2025", "19"),
    "FroHemenAnteil": (35.6, 1, "fro-20-f-2025", "19"),
    "GeaAktien": (162801664, 1e-6, "gea-ar-2025", "16"),
    "GeaEigenkapital": (2453007, 1e-3, "gea-ar-2025", "312"),
    "GeaStreubesitz": (89.5, 1, "gea-ar-2025", "14"),
    "GeaServiceAnteil": (40.0, 1, "gea-ar-2025", "2"),
    "BesiAktien": (79.3, 1, "besi-ar-2025", "53"),
    "BesiEigenkapital": (416397, 1e-3, "besi-ar-2025", "193"),
    "BesiAmatAnteil": (9.00, 1, "besi-ar-2025", "159"),
    "BesiBlackrockAnteil": (8.68, 1, "besi-ar-2025", "159"),
    "BesiFmrAnteil": (6.00, 1, "besi-ar-2025", "159"),
    "BesiAdvPackAnteil": (70, 1, "besi-ar-2025", "31"),
    "BesiKiAuftraege": (50, 1, "besi-ar-2025", "31"),
    "OxyChemPreis": (9.7, 1, "oxy-def14a-2026", "32"),
    "OxyCfoFortgefuehrt": (9606, 1, "oxy-10-k-2025", "63"),
    "OxyCfoAufgegeben": (926, 1, "oxy-10-k-2025", "63"),
}

# EXTERN: Sekundaerquellen ohne gedruckte Seite (Directors' Dealings, Stand 19.09.2026).
# Quelle je Titel in docs/befunde-und-entscheidungen.md und data/form4_oxy.json,
# data/dd_tte.json, data/dd_gea.json, refs/dd/besi_afm.csv.
EXTERN = {
    "OxyDdKaeufe": 2, "OxyDdVerkaeufe": 0, "OxyKlesseStueck": 5000, "OxyKlessePreis": 38.98,
    "OxyJacksonStueck": 4770, "OxyJacksonPreis": 52.38,
    "TteDdKaeufe": 0, "TteDdVerkaeufe": 8, "TteDdPersonen": 4,
    "TtePouyanneStueck": 31008, "TtePouyannePreis": 76.92,
    "GeaDdKaeufe": 6, "GeaDdPersonen": 4, "GeaDdPreisVon": 53.60, "GeaDdPreisBis": 59.60,
    "GeaKlebertVolumen": 597350.0,
    "BesiDdMeldungen": 31, "BesiDdPersonen": 10,
    "FilterTitel": 190,
}


def roh(name: str) -> float:
    w, f, _, _ = ROH[name]
    return w * f


R = json.load(open(os.path.join(DATA, "reihen.json")))
KURSE = json.load(open(os.path.join(DATA, "kurse_roh.json")))


def reihe(t: str, posten: str) -> dict:
    return {int(j): v["wert"] for j, v in R[KUERZEL[t]].get(posten, {}).items()}


def fcf(t: str) -> dict:
    """
    Ausschuettungsfaehiger freier Mittelzufluss je Jahr (Mio.):
    operativer Mittelzufluss + Investitionen (negativ) + Leasingtilgung (negativ, IFRS)
    - Vorzugsdividende (OXY). OXY ab 2023 ohne OxyChem: fortgefuehrter operativer
    Mittelzufluss, Investitionen gesamt abzueglich Investitions-CF der aufgegebenen
    Taetigkeit (Annahme, da der 10-K die Investitionen nicht nach Taetigkeit trennt).
    """
    cfo, cap, lea = reihe(t, "operativer_cf"), reihe(t, "investitionen"), reihe(t, "leasing")
    vz = reihe(t, "vorzugsdividende")
    aus = {}
    # Reihe ab 2014 (hoechstens zwoelf Jahre): XBRL reicht bei OXY und FRO bis 2007
    # zurueck, das Muster verlangt sieben bis zehn Jahre, nicht so viele wie verfuegbar.
    for j in sorted(j for j in set(cfo) & set(cap) if j >= 2014):
        wert = cfo[j] + cap[j] + lea.get(j, 0.0) - abs(vz.get(j, 0.0))
        if t == "Oxy" and j >= 2023:
            fort = reihe(t, "operativer_cf_fortgef")[j]
            aufg = reihe(t, "invest_cf_aufgegeben").get(j, 0.0)
            wert = fort + (cap[j] - aufg) - abs(vz.get(j, 0.0))
        aus[j] = wert
    return aus


def perzentil(werte: list, p: float) -> float:
    s = sorted(werte)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rang(werte: list, x: float) -> float:
    """Anteil der Jahre der eigenen Reihe (0-100 %), deren Wert unter x liegt."""
    return 100.0 * sum(1 for w in werte if w < x) / len(werte) if werte else 50.0


# --------------------------------------------------------------------------
# Deutsche Zahlformate (kein siunitx: siehe sources-and-typesetting.md)
# --------------------------------------------------------------------------
def de(x, stellen=0) -> str:
    if x is None:
        return "--"
    s = f"{abs(x):,.{stellen}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("$-$" if round(x, stellen) < 0 else "") + s   # kein "-0" bei gerundeter Null


def prozent(x, stellen=1) -> str:
    return "--" if x is None else de(100 * x, stellen) + "\\,\\%"


# --------------------------------------------------------------------------
# Rechnung je Titel
# --------------------------------------------------------------------------
def aktien(t: str) -> float:
    return roh("TnkAktienA") + roh("TnkAktienB") if t == "Tnk" else roh(f"{t}Aktien")


def buchwert(t: str) -> float:
    ek = roh(f"{t}Eigenkapital")
    return ek - roh("OxyVorzugsaktien") if t == "Oxy" else ek


def konfig(t: str, szen: dict, huerde: float) -> dict:
    kurs = KURSE[TICKER[t]]["letzter"][1]
    return {
        "currency": f"Mio. {WAEHRUNG[t]}", "opening_liquidity": 0.0, "horizon_start": START,
        "horizon_end": START + max(HORIZONTE), "schedule_years": [START, START + 1, START + 2],
        "sensitivity_horizons": [START + h for h in HORIZONTE], "revenue_base": 1.0,
        "earnings_base": 1.0, "hurdle": huerde, "hurdle_source": HUERDE_QUELLE, "revenue_scenarios": [],
        "options": {k: {"name": name, "investment": {str(START): kurs * aktien(t)},
                        "earnings_uplift": wert, "uplift_start": START + 1,
                        "terminal_value": buchwert(t)} for k, (name, wert) in szen.items()},
    }


def szenarien(t: str) -> dict:
    f = fcf(t)
    werte = list(f.values())
    return {
        "PES": ("pessimistisch (25.~Perzentil)", perzentil(werte, 0.25)),
        "MED": ("Basis (Median)", statistics.median(werte)),
        "VOLL": ("Geschäftsjahr 2025", f[2025]),
    }


def kgv_reihe(t: str) -> dict:
    """Jahresend-KGV: Schlusskurs Jahresende / verwaessertes Ergebnis je Aktie (nur positive)."""
    eps = reihe(t, "eps_verwaessert")
    ye = {int(j): p for j, p in KURSE[TICKER[t]]["jahresende"].items()}
    ab = 2017 if t == "Tnk" else 0          # TNK: 1:8-Zusammenlegung 2019, EPS vor 2017 nicht angepasst
    return {j: ye[j] / eps[j] for j in sorted(eps) if j in ye and eps[j] > 0 and j >= ab}


def main() -> int:
    M = {}                       # Makros
    tabellen = {}
    for t in TITEL:
        f = fcf(t)
        sz = szenarien(t)
        cfg = konfig(t, sz, HUERDE)
        cfg_lang = konfig(t, sz, HUERDE_LANG)
        n_akt = aktien(t)
        kurs = KURSE[TICKER[t]]["letzter"][1]
        mcap = kurs * n_akt
        bw = buchwert(t)
        werte = list(f.values())
        M.update({
            f"{t}Kurs": de(kurs, 2), f"{t}KursDatum": ".".join(reversed(KURSE[TICKER[t]]["letzter"][0].split("-"))),
            f"{t}AktienMio": de(n_akt, 1), f"{t}Boersenwert": de(mcap), f"{t}Buchwert": de(bw),
            f"{t}BuchwertJeAktie": de(bw / n_akt, 2), f"{t}KBV": de(mcap / bw, 2),
            f"{t}FcfJahre": f"{min(f)}--{max(f)}", f"{t}FcfAnzahl": str(len(f)),
            f"{t}FcfMedian": de(statistics.median(werte)), f"{t}FcfPes": de(perzentil(werte, 0.25)),
            f"{t}FcfVoll": de(f[2025]), f"{t}FcfRang": de(rang(werte, f[2025]), 0),
            f"{t}FcfRendite": prozent(f[2025] / mcap), f"{t}FcfRenditeMed": prozent(statistics.median(werte) / mcap),
            f"{t}Waehrung": WAEHRUNG[t],
        })
        for name in ROH:
            if name.startswith(t):
                w, fak, _, _ = ROH[name]
                # so viele Nachkommastellen wie gedruckt; Stueckzahlen in Mio. mit zwei
                gedruckt = len(str(w).split(".")[1]) if isinstance(w, float) and w % 1 else 0
                M[name] = de(w * fak, 2 if fak != 1 else gedruckt)
        # Realisiertes Wachstum: Durchschnitt der ersten drei gegen den der letzten drei Jahre,
        # annualisiert ueber den Abstand der Mittelpunkte. Ein einzelnes Anfangs- oder Endjahr
        # waere bei zyklischen Reihen ein Zufallswert; bei negativem Durchschnitt keine Rate.
        jahre = sorted(f)
        a3, e3 = statistics.mean(f[j] for j in jahre[:3]), statistics.mean(f[j] for j in jahre[-3:])
        abstand_j = jahre[-2] - jahre[1]
        real = (e3 / a3) ** (1 / abstand_j) - 1 if a3 > 0 and e3 > 0 and abstand_j > 0 else None
        M[f"{t}FcfWachstumReal"] = prozent(real) if real is not None else "n.\\,b."
        M[f"{t}FcfWachstumSpanne"] = f"{jahre[1]}--{jahre[-2]}"
        # Teil 7.3: Schwelle je Aktie, erforderliches Wachstum, erforderlicher Endwert
        zeilen_s, zeilen_w, zeilen_e = [], [], []
        for k, (name, wert) in sz.items():
            s = [ci.break_even_investment(cfg, k, START + h) for h in HORIZONTE]
            s_lang = [ci.break_even_investment(cfg_lang, k, START + h) for h in HORIZONTE]
            g = [ci.required_growth(cfg, k, START + h) for h in HORIZONTE]
            e = [ci.required_terminal_value(cfg, k, START + h) for h in HORIZONTE]
            zeilen_s.append(f"{name} & {de(wert)} & " + " & ".join(
                de(x / n_akt, 2) if x else "n.\\,b." for x in s) + " & " + de(s_lang[1] / n_akt, 2)
                if s_lang[1] else f"{name} & {de(wert)} & " + " & ".join(
                de(x / n_akt, 2) if x else "n.\\,b." for x in s) + " & n.\\,b.")
            zeilen_w.append(f"{name} & " + " & ".join(prozent(x) if x is not None else "n.\\,b." for x in g))
            zeilen_e.append(f"{name} & " + " & ".join(
                (de(x / bw, 1) + "\\,$\\times$") if x is not None else "n.\\,b." for x in e))
            M[f"{t}Schwelle{k}Zehn"] = de(s[1] / n_akt, 2) if s[1] else "n.\\,b."
            M[f"{t}Schwelle{k}ZehnLang"] = de(s_lang[1] / n_akt, 2) if s_lang[1] else "n.\\,b."
            M[f"{t}Abstand{k}"] = (prozent(kurs / (s[1] / n_akt) - 1, 0) if s[1] else "n.\\,b.")
            M[f"{t}Vielfaches{k}"] = (de(kurs / (s[1] / n_akt), 1) if s[1] else "n.\\,b.")
            M[f"{t}Wachstum{k}Zehn"] = prozent(g[1]) if g[1] is not None else "n.\\,b."
            M[f"{t}Endwert{k}Zehn"] = de(e[1] / bw, 1) if e[1] is not None else "n.\\,b."
        tabellen[f"{KUERZEL[t]}_schwelle"] = " \\\\\n".join(zeilen_s) + " \\\\"
        tabellen[f"{KUERZEL[t]}_wachstum"] = " \\\\\n".join(zeilen_w) + " \\\\"
        tabellen[f"{KUERZEL[t]}_endwert"] = " \\\\\n".join(zeilen_e) + " \\\\"
        # Teil 2: Reihe (Mio.)
        cfo, cap, lea = reihe(t, "operativer_cf"), reihe(t, "investitionen"), reihe(t, "leasing")
        div, eps = reihe(t, "dividenden"), reihe(t, "eps_verwaessert")
        if t == "Oxy":   # Summe Stamm- und Vorzugsdividende; Stammanteil = Summe - Vorzugsdividende
            vzd = reihe(t, "vorzugsdividende")
            div = {j: d + abs(vzd.get(j, 0.0)) for j, d in div.items()}
        quelle = {int(j): v for j, v in R[KUERZEL[t]].get("operativer_cf", {}).items()}
        zeilen = []
        for j in sorted(f):
            q = quelle.get(j, {})
            beleg = "XBRL" if q.get("seite") is None else f"{q['bericht'].split('-')[-1]}, S.~{q['seite']}"
            zeilen.append(f"{j} & {de(cfo[j])} & {de(cap[j])} & {de(lea.get(j))} & {de(f[j])} & "
                          f"{de(div.get(j))} & {de(eps.get(j), 2)} & {beleg}")
        tabellen[f"{KUERZEL[t]}_reihe"] = " \\\\\n".join(zeilen) + " \\\\"
        # Teil 7.3: Multiple im Kontext
        kgv = kgv_reihe(t)
        eps25 = eps.get(2025)
        kgv_heute = kurs / eps25 if eps25 and eps25 > 0 else None
        if kgv:
            M[f"{t}KgvMedian"] = de(statistics.median(kgv.values()), 1)
            M[f"{t}KgvJahre"] = f"{min(kgv)}--{max(kgv)}"
            M[f"{t}KgvAnzahl"] = str(len(kgv))
        M[f"{t}KgvHeute"] = de(kgv_heute, 1) if kgv_heute else "n.\\,b."
        M[f"{t}KgvRang"] = de(rang(list(kgv.values()), kgv_heute), 0) if kgv and kgv_heute else "n.\\,b."
        tabellen[f"{KUERZEL[t]}_kgv"] = " \\\\\n".join(
            f"{j} & {de(KURSE[TICKER[t]]['jahresende'][str(j)], 2)} & {de(eps[j], 2)} & {de(v, 1)}"
            for j, v in kgv.items()) + " \\\\"
    # Paarvergleich: KGV heute und 10-Jahres-Median beider Titel
    for a, b in PAARE:
        tabellen[f"paar_{KUERZEL[a]}_{KUERZEL[b]}"] = " \\\\\n".join(
            f"{NAME[x]} & \\{x}KgvHeute & \\{x}KgvMedian & \\{x}KgvRang & \\{x}FcfRendite & \\{x}KBV"
            for x in (a, b)) + " \\\\"
    for k, w in EXTERN.items():
        M[k] = de(w, 2 if isinstance(w, float) and w % 1 else 0)
    M["Huerde"] = de(HUERDE, 2)
    M["HuerdeLang"] = de(HUERDE_LANG, 2)
    M["HuerdeQuelle"] = HUERDE_QUELLE
    M["KursQuelle"] = KURS_QUELLE
    # Ausgabe
    with open(os.path.join(DATA, "kennzahlen.tex"), "w") as fo:
        fo.write("% erzeugt von scripts/rechnung.py - nicht von Hand aendern\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for name, inhalt in tabellen.items():
        open(os.path.join(DATA, f"{name}.tex"), "w").write(inhalt + "\n")
    print(f"[RECHNUNG] {len(M)} Makros, {len(tabellen)} Tabellen")
    for t in TITEL:
        print(f"  {t}: Kurs {M[t + 'Kurs']} | Schwelle MED 10J {M[t + 'SchwelleMEDZehn']} | "
              f"VOLL {M[t + 'SchwelleVOLLZehn']} | Wachstum MED {M[t + 'WachstumMEDZehn']} | "
              f"Endwert MED {M[t + 'EndwertMEDZehn']}x Buch | KGV {M[t + 'KgvHeute']} (Median "
              f"{M.get(t + 'KgvMedian')}, Rang {M[t + 'KgvRang']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
