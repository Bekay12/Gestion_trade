#!/usr/bin/env python3
"""
flow.py - Teil 6 des Berichts: Kapazitaetsrechnung und Anlegerrechnung
fuer beide Emittenten.

Es gibt hier keine Handlungsoptionen zu vergleichen. Die Frage lautet je
Titel: lohnt der Einstieg zum heutigen Kurs? Deshalb die Variante ohne
Optionen aus dem Muster, und deshalb zwei Rechnungen, die ALTERNATIVEN sind
und nicht addiert werden duerfen - derselbe Euro kann nicht zweimal
ausgegeben werden:

    6.1  Kapazitaet   Wohin traegt der freie Mittelzufluss?
                      AZN: Entschuldung (Nettoverschuldung 23,4 Mrd. USD).
                      RDY: Nettokasse - die Entschuldungsfrage ist leer und
                      wird durch die Finanzierungsfrage ersetzt: traegt der
                      operative Mittelzufluss das eigene Investitionsprogramm?
    6.3  Anleger      Was verdient, wer heute kauft? Einstiegspreis bei t=0,
                      ausschuettungsfaehiger Mittelzufluss je Aktie in jedem
                      Jahr, Buchwert je Aktie am Ende des Horizonts.

Der Endwert ist der Buchwert je Aktie. Er ist belegt, er haengt nicht vom
Einstiegspreis ab, und er ist fuer beide Titel konservativ. Ein Bewertungs-
vielfaches waere genau der plausibel aussehende Platzhalter, den die
Rigorosregeln verbieten. Weil der Endwert das Ergebnis dominiert, wird die
Umkehrung mitgedruckt: welchen Endwert der HEUTIGE Kurs schon voraussetzt.

Alle Rechnungen sind VORSTEUERrechnungen auf Ebene des Unternehmens: der
ausschuettungsfaehige Mittelzufluss ist bereits nach gezahlten Ertragsteuern
des Konzerns, aber die Steuer des Anlegers auf Dividende und Kursgewinn
bleibt unberuecksichtigt. Quellensteuer auf indische Dividenden und
US-Quellensteuer sind damit nicht abgebildet.

Aufruf: python3 scripts/flow.py
Ausgabe: data/flow_*.tex
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cashflow_irr as ci                                    # noqa: E402
from kennzahlen import rechne, ROH, EXTERN                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONTE = (5, 10, 15)
START = 2026

# --------------------------------------------------------------------------
# Zwischenwerte, die aus der Zahlenschicht folgen. Sie werden hier gerechnet
# und nicht getippt; jeder Bestandteil traegt in kennzahlen.py seine Seite.
# --------------------------------------------------------------------------
T = rechne()

# Freier Mittelzufluss = operativer Mittelzufluss abzueglich Sachinvestitionen,
# abzueglich immaterieller Investitionen zuzueglich ihrer Abgaenge, abzueglich
# Leasingtilgung. Die Leasingtilgung gehoert dazu, weil sie wirtschaftlich
# Miete ist und dem Anleger nicht zur Verfuegung steht.
AZN_JAHRE = list(range(2019, 2026))
RDY_JAHRE = list(range(2018, 2027))


def _r(n: int) -> str:
    from kennzahlen import roemisch
    return roemisch(n)


def azn_fcf(jahr: int) -> float:
    r = _r(jahr)
    return (T[f"AznOperativerCF{r}"] + T[f"AznSachInvestition{r}"]
            + T[f"AznImmatInvestition{r}"] + T[f"AznImmatAbgang{r}"]
            + T[f"AznLeasingTilgung{r}"])


def rdy_fcf(jahr: int) -> float:
    r = _r(jahr)
    return (T[f"RdyOperativerCF{r}"] + T[f"RdySachInvestition{r}"]
            + T[f"RdyImmatInvestition{r}"] + T[f"RdySachAbgang{r}"])


# Halbjahr bzw. Quartal, um die letzten zwoelf Monate zu bilden. Die Werte
# kommen aus der Zahlenschicht und tragen dort ihre gedruckte Seite; hier
# steht nur, welche Posten zusammengehoeren.
AZN_HJ = {
    2026: ("AznOperativerCfHJ", "AznSachInvestitionHJ", "AznImmatInvestitionHJ",
           "AznImmatAbgangHJ", "AznLeasingTilgungHJ"),
    2025: ("AznOperativerCfHJVJ", "AznSachInvestitionHJVJ", "AznImmatInvestitionHJVJ",
           "AznImmatAbgangHJVJ", "AznLeasingTilgungHJVJ"),
}
RDY_Q1 = {
    2027: ("RdyQuartalCf", "RdyQuartalInvest"),
    2026: ("RdyQuartalCfVJ", "RdyQuartalInvestVJ"),
}


def _summe(namen) -> float:
    return sum(T[n] for n in namen)


def azn_reihen() -> dict:
    jaktie = {j: azn_fcf(j) / T["AznAktienGewichtet" + _r(j)] for j in AZN_JAHRE}
    hj26 = _summe(AZN_HJ[2026])
    hj25 = _summe(AZN_HJ[2025])
    ltm = (azn_fcf(2025) - hj25 + hj26) / (T["AznAktienAusgegeben"] / 1e6)
    return {"jeAktie": jaktie, "ltm": ltm,
            "median": statistics.median(jaktie.values()),
            "letztes": jaktie[2025],
            "buchwert": T["AznBuchwertJeAktie"], "kurs": T["AznKurs"]}


def rdy_reihen() -> dict:
    aktien = T["RdyAktienAusgegeben"] / 1e6
    jaktie = {j: rdy_fcf(j) / aktien for j in RDY_JAHRE}
    ltm = (rdy_fcf(2026) - _summe(RDY_Q1[2026]) + _summe(RDY_Q1[2027])) / aktien
    return {"jeAktie": jaktie, "ltm": ltm,
            "median": statistics.median(jaktie.values()),
            "letztes": jaktie[2026],
            "buchwert": T["RdyBuchwertJeAktie"], "kurs": T["RdyKursRupien"]}


def konfig(reihen: dict, waehrung: str, quelle_huerde: str) -> dict:
    """Anlegerrechnung als Konfiguration fuer cashflow_irr."""
    szenarien = {
        "LTM": ("letzte zw\\\"olf Monate", reihen["ltm"]),
        "MED": ("Median der Reihe", reihen["median"]),
        "VOLL": ("letztes volles Gesch\\\"aftsjahr", reihen["letztes"]),
    }
    return {
        "currency": waehrung,
        "opening_liquidity": 0.0,
        "horizon_start": START,
        "horizon_end": START + max(HORIZONTE),
        "schedule_years": [START, START + 1, START + 2],
        "sensitivity_horizons": [START + h for h in HORIZONTE],
        "revenue_base": 1.0,
        "earnings_base": 1.0,
        "hurdle": T["Huerde"],
        "hurdle_source": quelle_huerde,
        "revenue_scenarios": [],
        "options": {
            schluessel: {
                "name": bezeichnung,
                "investment": {str(START): reihen["kurs"]},
                "earnings_uplift": wert,
                "uplift_start": START + 1,
                "terminal_value": reihen["buchwert"],
            }
            for schluessel, (bezeichnung, wert) in szenarien.items()
        },
    }


def _num(w, stellen=2) -> str:
    return "--" if w is None else f"\\num{{{w:.{stellen}f}}}"


def tabelle_irr(cfg: dict) -> str:
    zeilen = []
    for s in ("LTM", "MED", "VOLL"):
        werte = [ci.irr(ci.full_cashflows(cfg, s, START + h)) for h in HORIZONTE]
        zeilen.append(" & ".join([cfg["options"][s]["name"]]
                                 + [_num(w * 100, 1) if w is not None else "--"
                                    for w in werte]) + r" \\")
    return "\n".join(zeilen)


def tabelle_schwelle(cfg: dict) -> str:
    zeilen = []
    for s in ("LTM", "MED", "VOLL"):
        werte = [ci.break_even_investment(cfg, s, START + h) for h in HORIZONTE]
        zeilen.append(" & ".join(
            [cfg["options"][s]["name"], _num(cfg["options"][s]["earnings_uplift"])]
            + [_num(w) for w in werte]) + r" \\")
    return "\n".join(zeilen)


def tabelle_endwert(cfg: dict, buchwert: float) -> str:
    zeilen = []
    for s in ("LTM", "MED", "VOLL"):
        werte = [ci.required_terminal_value(cfg, s, START + h) for h in HORIZONTE]
        zeilen.append(" & ".join(
            [cfg["options"][s]["name"]]
            + [f"{_num(w)} ({w / buchwert:.2f}x)" if w and w > 0 else _num(w)
               for w in werte]) + r" \\")
    return "\n".join(zeilen)


def tabelle_wachstum(cfg: dict) -> str:
    zeilen = []
    for s in ("LTM", "MED", "VOLL"):
        werte = [ci.required_growth(cfg, s, START + h) for h in HORIZONTE]
        zeilen.append(" & ".join(
            [cfg["options"][s]["name"]]
            + [_num(w * 100, 1) if w is not None else "nicht erreichbar"
               for w in werte]) + r" \\")
    return "\n".join(zeilen)


def tabelle_reihe(reihen: dict, stellen=2) -> str:
    zeilen = []
    werte = reihen["jeAktie"]
    rang = sorted(werte.values())
    for j in sorted(werte):
        p = 100.0 * rang.index(werte[j]) / (len(rang) - 1)
        zeilen.append(f"{j} & {_num(werte[j], stellen)} & {_num(p, 0)} \\%" + r" \\")
    return "\n".join(zeilen)


def tabelle_reihe_voll(jahre: list, umsatz, ergebnis, opcf, fcf, aktien) -> str:
    """Mehrjahresreihe: Umsatz, Ergebnis, operativer und freier Mittelzufluss.

    Die Reihe ist der Grund, aus dem Teil 6 ueberhaupt etwas ueber die Lage
    im Zyklus sagen kann. Ein einzelnes Jahr ist ein Niveau und traegt keine
    Auskunft darueber, wo es in der eigenen Verteilung steht.
    """
    zeilen = []
    for j in jahre:
        zeilen.append(" & ".join([
            str(j), _num(umsatz(j), 0), _num(ergebnis(j), 0), _num(opcf(j), 0),
            _num(fcf(j), 0), _num(fcf(j) / aktien(j), 2)]) + r" \\")
    return "\n".join(zeilen)


def tabelle_vielfach(reihen: dict) -> str:
    """Gefordertes Wachstum bei konstantem Bewertungsvielfachen, drei Horizonte."""
    zeilen = []
    for name, wert in (("letzte zw\\\"olf Monate", reihen["ltm"]),
                       ("Median der Reihe", reihen["median"]),
                       ("letztes volles Gesch\\\"aftsjahr", reihen["letztes"])):
        raten = [wachstum_bei_festem_vielfachen(reihen["kurs"], wert,
                                                T["Huerde"] / 100, h)
                 for h in HORIZONTE]
        zeilen.append(" & ".join([name] + [_num(r * 100, 1) if r is not None
                                           else "nicht erreichbar" for r in raten])
                      + r" \\")
    return "\n".join(zeilen)


def kapazitaet_azn() -> str:
    """Entschuldungskapazitaet: rechnerische Jahre bis zur Schuldenfreiheit."""
    fcf = azn_fcf(2025)
    dividende = -T["AznDividendeGezahltXXV"]
    ueberschuss = fcf - dividende
    jahre = T["AznNettoverschuldung"] / ueberschuss if ueberschuss > 0 else None
    return "\n".join([
        rf"Freier Mittelzufluss 2025 & {_num(fcf, 0)} \\",
        rf"abz\"uglich gezahlter Dividende & {_num(-dividende, 0)} \\",
        rf"verbleibender \"Uberschuss & {_num(ueberschuss, 0)} \\",
        rf"Nettoverschuldung 31.12.2025 & {_num(T['AznNettoverschuldung'], 0)} \\",
        rf"rechnerische Jahre bis zur Tilgung & {_num(jahre, 1)} \\",
    ])


def kapazitaet_rdy() -> str:
    """Finanzierungsfrage: traegt der operative Zufluss das eigene Programm?"""
    op = T["RdyOperativerCFXXVI"]
    invest = -(T["RdySachInvestitionXXVI"] + T["RdyImmatInvestitionXXVI"])
    dividende = T["RdyDividendeJeAktie"] * T["RdyAktienAusgegeben"] / 1e6
    rest = op - invest - dividende
    return "\n".join([
        rf"Operativer Mittelzufluss FY2026 & {_num(op, 0)} \\",
        rf"abz\"uglich Investitionen (Sach und immateriell) & {_num(-invest, 0)} \\",
        rf"abz\"uglich Dividende & {_num(-dividende, 0)} \\",
        rf"verbleibender \"Uberschuss & {_num(rest, 0)} \\",
        rf"Nettokasse 31.03.2026 & {_num(T['RdyNettokasse'], 0)} \\",
    ])


def variante(reihen: dict, waehrung: str, quelle: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Dieselbe Anlegerrechnung mit einem anderen Endwert: die Aktie notiert
        am Ende des Horizonts nominal zum HEUTIGEN Kurs. Der Buchwert ist der
        belegte und der konservative Endwert, und fuer ein Pharmaunternehmen,
        dessen Vermoegen zu grossen Teilen nicht bilanziert ist, ist er sehr
        konservativ - AstraZeneca steht mit dem Fuenffachen seines Buchwerts
        im Markt. Ohne eine zweite Rechnung liest sich die erste, als sei der
        Markt um den Faktor drei im Irrtum. Mit ihr laesst sich trennen, was
        am Ergebnis am Ausschuettungsstrom haengt und was am Endwert.

        Der Kurs am Ende ist eine ANNAHME und wird als solche ausgewiesen; er
        ist keine Prognose, sondern die Aussage "kein Bewertungsgewinn, kein
        Bewertungsverlust".

    Inputs:
        reihen (dict): Ausgabe von azn_reihen bzw. rdy_reihen
        waehrung (str): Einheit je Aktie
        quelle (str): Herkunft der Huerde

    Outputs:
        cfg (dict): Konfiguration mit Endwert = heutiger Kurs
    --------------------------------------------------------------------------
    """
    cfg = konfig(reihen, waehrung, quelle)
    for opt in cfg["options"].values():
        opt["terminal_value"] = reihen["kurs"]
    return cfg


def wachstum_bei_festem_vielfachen(kurs: float, fluss: float, huerde: float,
                                   jahre: int) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Die dritte Umkehrung, und die fairste. Die beiden anderen halten den
        Endwert fest: einmal auf dem Buchwert, einmal auf dem heutigen Kurs.
        Beide bestrafen damit ein wachsendes Unternehmen doppelt - der
        Mittelzufluss darf wachsen, der Wert des Anteils nicht. Hier waechst
        der Endwert mit dem Mittelzufluss, das Bewertungsvielfache bleibt
        also konstant. Das ist eine Annahme und keine Prognose, und sie ist
        genau die, die ein Leser im Kopf hat, wenn er sagt, eine wachsende
        Firma sei ihren Preis wert.

        Gesucht ist die Rate g, bei der gilt:
            -P + sum_t F(1+g)^t/(1+h)^t + P(1+g)^n/(1+h)^n = 0

    Inputs:
        kurs (float): Einstiegspreis je Aktie
        fluss (float): ausschuettungsfaehiger Mittelzufluss je Aktie, Jahr 1
        huerde (float): geforderte Rendite als Dezimalzahl
        jahre (int): Horizont in Jahren

    Outputs:
        g (float): Wachstumsrate als Dezimalzahl, oder None wenn selbst
            200 Prozent nicht genuegen
    --------------------------------------------------------------------------
    """
    def barwert(g: float) -> float:
        w = -kurs
        for t in range(1, jahre + 1):
            w += fluss * (1 + g) ** t / (1 + huerde) ** t
        w += kurs * (1 + g) ** jahre / (1 + huerde) ** jahre
        return w

    lo, hi = -0.5, 2.0
    if barwert(hi) < 0:
        return None
    for _ in range(200):
        mitte = (lo + hi) / 2
        if barwert(mitte) < 0:
            lo = mitte
        else:
            hi = mitte
    return (lo + hi) / 2


def schwelle_bei_wachstum(fluss: float, huerde: float, wachstum: float,
                          jahre: int) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Einstiegspreis, bei dem die Huerde erreicht wird, wenn der
        Mittelzufluss mit einer VORGEGEBENEN Rate waechst und das
        Bewertungsvielfache konstant bleibt. Das ist die Gegenrichtung zu
        wachstum_bei_festem_vielfachen: dort ist der Preis gegeben und die
        Rate gesucht, hier die Rate gegeben und der Preis gesucht.

        Die Rate muss BELEGT sein, sonst ist der Preis eine Zahl, die auf
        nichts steht. Zulaessig sind die eigene Umsatzambition des
        Unternehmens und die von ihm selbst berichteten historischen Raten -
        also Groessen, die schon im Bericht stehen.

    Inputs:
        fluss (float): Mittelzufluss je Aktie im ersten Jahr
        huerde (float): geforderte Rendite als Dezimalzahl
        wachstum (float): unterstellte Wachstumsrate als Dezimalzahl
        jahre (int): Horizont

    Outputs:
        preis (float): Einstiegspreis je Aktie
    --------------------------------------------------------------------------
    """
    k = (1.0 + wachstum) / (1.0 + huerde)
    if k >= 1.0:
        # Waechst der Mittelzufluss schneller als die Huerde verlangt, gibt es
        # keine obere Preisgrenze mehr: Der Endwert allein uebersteigt dann den
        # Einstiegspreis, und jeder Preis erreicht die Huerde. Das ist kein
        # Rechenfehler, sondern die Aussage, dass die unterstellte Rate die
        # Frage aufhebt - und deshalb wird hier None zurueckgegeben statt
        # eines negativen Preises, der wie ein Ergebnis aussaehe.
        return None
    if abs(1.0 - k ** jahre) < 1e-12:
        return None
    return fluss * sum(k ** t for t in range(1, jahre + 1)) / (1.0 - k ** jahre)


def cagr(werte: dict) -> float:
    """Jaehrliche Rate zwischen erstem und letztem Jahr der Reihe, in Prozent."""
    jahre = sorted(werte)
    n = jahre[-1] - jahre[0]
    return ((werte[jahre[-1]] / werte[jahre[0]]) ** (1.0 / n) - 1.0) * 100.0


def main() -> int:
    azn, rdy = azn_reihen(), rdy_reihen()
    quelle = ("Zehnjahresrendite des Dow Jones U.S. Select Pharmaceuticals Index "
              "zum 30.06.2026, ishares.com, abgerufen 17.09.2026")
    cfg_azn = konfig(azn, "USD je Aktie", quelle)
    cfg_rdy = konfig(rdy, "INR je Aktie", quelle)
    aus = {
        "flow_azn_irr": tabelle_irr(cfg_azn),
        "flow_azn_schwelle": tabelle_schwelle(cfg_azn),
        "flow_azn_endwert": tabelle_endwert(cfg_azn, azn["buchwert"]),
        "flow_azn_wachstum": tabelle_wachstum(cfg_azn),
        "flow_azn_reihe": tabelle_reihe(azn),
        "flow_azn_kapazitaet": kapazitaet_azn(),
        "flow_rdy_irr": tabelle_irr(cfg_rdy),
        "flow_rdy_schwelle": tabelle_schwelle(cfg_rdy),
        "flow_rdy_endwert": tabelle_endwert(cfg_rdy, rdy["buchwert"]),
        "flow_rdy_wachstum": tabelle_wachstum(cfg_rdy),
        "flow_rdy_reihe": tabelle_reihe(rdy),
        "flow_rdy_kapazitaet": kapazitaet_rdy(),
        "flow_azn_schwellevariante": tabelle_schwelle(variante(azn, "USD je Aktie", quelle)),
        "flow_rdy_schwellevariante": tabelle_schwelle(variante(rdy, "INR je Aktie", quelle)),
        "flow_azn_irrvariante": tabelle_irr(variante(azn, "USD je Aktie", quelle)),
        "flow_azn_vielfach": tabelle_vielfach(azn),
        "flow_rdy_vielfach": tabelle_vielfach(rdy),
        "flow_azn_reihevoll": tabelle_reihe_voll(
            AZN_JAHRE,
            lambda j: T["AznUmsatz" + _r(j)],
            lambda j: T["AznJahresueberschuss" + _r(j)],
            lambda j: T["AznOperativerCF" + _r(j)],
            azn_fcf,
            lambda j: T["AznAktienGewichtet" + _r(j)]),
        "flow_rdy_reihevoll": tabelle_reihe_voll(
            RDY_JAHRE,
            lambda j: T["RdyUmsatz" + _r(j)],
            lambda j: T["RdyJahresueberschuss" + _r(j)],
            lambda j: T["RdyOperativerCF" + _r(j)],
            rdy_fcf,
            lambda j: T["RdyAktienAusgegeben"] / 1e6),
        "flow_rdy_irrvariante": tabelle_irr(variante(rdy, "INR je Aktie", quelle)),
    }
    for name, inhalt in aus.items():
        open(os.path.join(ROOT, "data", f"{name}.tex"), "w").write(inhalt + "\n")
    # Einzelgroessen, die die Prosa braucht
    makros = {
        "AznFcfLtm": azn["ltm"], "AznFcfMedian": azn["median"],
        "AznFcfVoll": azn["letztes"], "AznFcfXXV": azn_fcf(2025),
        "AznFcfXXIV": azn_fcf(2024),
        "AznFcfWachstum": (azn_fcf(2025) / azn_fcf(2024) - 1) * 100,
        "AznFcfJeAktieXIX": azn["jeAktie"][2019],
        "RdyFcfXXV": rdy_fcf(2025),
        "RdyFcfWachstum": (rdy_fcf(2026) / rdy_fcf(2025) - 1) * 100,
        "RdyFcfJeAktieXVIII": rdy["jeAktie"][2018],
        "RdyFcfLtm": rdy["ltm"], "RdyFcfMedian": rdy["median"],
        "RdyFcfVoll": rdy["letztes"], "RdyFcfXXVI": rdy_fcf(2026),
        "AznSchwelleZehn": ci.break_even_investment(cfg_azn, "VOLL", START + 10),
        "AznSchwelleMedZehn": ci.break_even_investment(cfg_azn, "MED", START + 10),
        "RdySchwelleZehn": ci.break_even_investment(cfg_rdy, "VOLL", START + 10),
        "RdySchwelleMedZehn": ci.break_even_investment(cfg_rdy, "MED", START + 10),
        "AznSchwelleVarZehn": ci.break_even_investment(
            variante(azn, "USD je Aktie", quelle), "VOLL", START + 10),
        "RdySchwelleVarZehn": ci.break_even_investment(
            variante(rdy, "INR je Aktie", quelle), "VOLL", START + 10),
        "AznWachstumZehn": ci.required_growth(cfg_azn, "VOLL", START + 10) * 100,
        "RdyWachstumZehn": ci.required_growth(cfg_rdy, "VOLL", START + 10) * 100,
        "AznFcfRendite": azn["letztes"] / azn["kurs"] * 100,
        "AznAbstandSchwelle": (azn["kurs"] / schwelle_bei_wachstum(
            azn["letztes"], T["Huerde"] / 100, T["AznAmbitionRate"] / 100, 10) - 1) * 100,
        "RdyFcfRendite": rdy["letztes"] / rdy["kurs"] * 100,
        "AznWachstumVielfachZehn": wachstum_bei_festem_vielfachen(
            azn["kurs"], azn["letztes"], T["Huerde"] / 100, 10) * 100,
        "AznWachstumVielfachMedZehn": wachstum_bei_festem_vielfachen(
            azn["kurs"], azn["median"], T["Huerde"] / 100, 10) * 100,
        "RdyWachstumVielfachZehn": wachstum_bei_festem_vielfachen(
            rdy["kurs"], rdy["letztes"], T["Huerde"] / 100, 10) * 100,
        "RdyWachstumVielfachMedZehn": wachstum_bei_festem_vielfachen(
            rdy["kurs"], rdy["median"], T["Huerde"] / 100, 10) * 100,
        # Einstiegspreis bei belegter Wachstumsrate und konstantem Vielfachen.
        # AZN: die eigene Umsatzambition bis 2030. RDY: die eigenen
        # historischen Raten, weil das Unternehmen kein Ziel nennt.
        "AznSchwelleAmbition": schwelle_bei_wachstum(
            azn["letztes"], T["Huerde"] / 100, T["AznAmbitionRate"] / 100, 10),
        "AznSchwelleAmbitionMed": schwelle_bei_wachstum(
            azn["median"], T["Huerde"] / 100, T["AznAmbitionRate"] / 100, 10),
        # Mit der Neunjahresrate (ueber der Huerde) gibt es keine Obergrenze;
        # der Wert bleibt deshalb bewusst leer und wird im Text benannt.
        "RdySchwelleHistorieSieben": schwelle_bei_wachstum(
            rdy["letztes"], T["Huerde"] / 100,
            cagr({j: w for j, w in rdy["jeAktie"].items() if j >= 2019}) / 100, 10),
        "AznCagr": cagr(azn["jeAktie"]),
        "RdyCagr": cagr(rdy["jeAktie"]),
        "RdyCagrSieben": cagr({j: w for j, w in rdy["jeAktie"].items() if j >= 2019}),
        "AznBuchwertJeAktieX": azn["buchwert"],
        "RdyBuchwertJeAktieX": rdy["buchwert"],
        "RdyKursRupienX": rdy["kurs"],
        # Abstand des Konsens-Kursziels zur eigenen Einstiegsschwelle. Die
        # Gegenueberstellung gehoert in Teil 7 und braucht beide Zahlen aus
        # derselben Rechnung, damit sie nicht aus zwei Quellen zusammengesetzt
        # werden muss.
        "AznKonsensUeberSchwelle": (T["AznKonsensZiel"]
                                    / ci.break_even_investment(cfg_azn, "VOLL", START + 10)
                                    - 1) * 100,
        "AznKonsensUeberVariante": (T["AznKonsensZiel"]
                                    / ci.break_even_investment(
                                        variante(azn, "USD je Aktie", quelle), "VOLL",
                                        START + 10) - 1) * 100,
        "RdyKonsensZielRupien": T["RdyKonsensZiel"] * T["Wechselkurs"],
        "RdyKonsensUeberSchwelle": (T["RdyKonsensZiel"] * T["Wechselkurs"]
                                    / ci.break_even_investment(cfg_rdy, "VOLL", START + 10)
                                    - 1) * 100,
        "RdyKonsensUeberVariante": (T["RdyKonsensZiel"] * T["Wechselkurs"]
                                    / ci.break_even_investment(
                                        variante(rdy, "INR je Aktie", quelle), "VOLL",
                                        START + 10) - 1) * 100,
    }
    with open(os.path.join(ROOT, "data", "flow_makros.tex"), "w") as f:
        f.write("% erzeugt von scripts/flow.py - NICHT von Hand aendern\n")
        for k, v in makros.items():
            if v is None:
                continue
            f.write(f"\\newcommand{{\\{k}}}{{{v:.6g}}}\n")
    print(f"[FLOW] AZN: FCF je Aktie LTM {azn['ltm']:.2f} / Median {azn['median']:.2f} "
          f"/ 2025 {azn['letztes']:.2f} USD, Buchwert {azn['buchwert']:.2f}, "
          f"Kurs {azn['kurs']:.2f}")
    print(f"[FLOW] RDY: FCF je Aktie LTM {rdy['ltm']:.2f} / Median {rdy['median']:.2f} "
          f"/ FY2026 {rdy['letztes']:.2f} INR, Buchwert {rdy['buchwert']:.2f}, "
          f"Kurs {rdy['kurs']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
