#!/usr/bin/env python3
"""
cashflow.py - Grobe Cash-Flow-Analyse der Optionen A, B und C (Aufgabe 6).

Erzeugt die Tabellenkoerper fuer Aufgabe 6.1 (je eine Tabelle pro Option),
die Vergleichstabelle 6.3 und die IRR-Sensitivitaet. Reine Standardbibliothek,
deterministische Ausgabe.

Sicht: inkrementell. Der ausgewiesene operative Cash-Flow ist der zusaetzliche
Cash-Flow der jeweiligen Option (zusaetzliches EBITDA abzueglich Aufbau der
Kapitalbindung), nicht der Konzern-Cash-Flow. So bleibt die Wirkung der
Entscheidung sichtbar, statt im Cash-Flow des Kerngeschaefts unterzugehen.

Die Investitionssummen, die Kapitalbindung und die EBITDA-Zuwaechse sind von
der Aufgabenstellung vorgegeben. Deren zeitliche Verteilung ist es nicht; sie
wird hier offengelegt und in Abschnitt 6.1 des Dokuments begruendet.
"""
import json
import os

# Nettofinanzliquiditaet zum 31.12.2024 laut Aufgabenstellung (Mio. EUR)
OPENING_LIQUIDITY = 138.2

# Horizont der Renditerechnung
HORIZON_START = 2026
HORIZON_END = 2035

ASSUMPTIONS = {
    "A": {
        "name": "Next Automation / Diversifikation",
        # 35 Mio. EUR gleichmaessig ueber drei Jahre
        "investment": {2026: 11.7, 2027: 11.7, 2028: 11.6},
        # Kapitalbindung 15 Mio. EUR, Aufbau vor dem Ergebnisbeitrag ab 2028
        "working_capital": {2027: 7.5, 2028: 7.5},
        "ebitda_delta": 16.0,
        "ebitda_start": 2028,
        "remarks": {
            2026: "Aufbau von Vertrieb und Applikationstechnik, erste Markterschlie{\\ss}ung",
            2027: "Markterschlie{\\ss}ung Clean Tech, Aerospace, Life Sciences; Aufbau Kapitalbindung",
            2028: "Erster voller EBITDA-Beitrag; Abschluss des Investitionsprogramms",
        },
    },
    "B": {
        "name": "Batterie- und Brennstoffzellen-Produktionstechnik",
        "investment": {2026: 10.0, 2027: 10.0, 2028: 10.0},
        # Ergebnisbeitrag ab 2027, Kapitalbindung entsprechend frueher
        "working_capital": {2026: 7.5, 2027: 7.5},
        "ebitda_delta": 14.0,
        "ebitda_start": 2027,
        "remarks": {
            2026: "Schwerpunkt F\\&E, Aufbau der Elektroden- und MEA-Fertigungskompetenz",
            2027: "Markteintritt; erster EBITDA-Beitrag",
            2028: "Skalierung der Linien; voller EBITDA-Beitrag",
        },
    },
    "C": {
        "name": "Anorganisches Wachstum / gezielte Akquisition",
        # Kaufpreis im Vollzugsjahr, danach Integrationsaufwand
        "investment": {2026: 45.0, 2027: 7.5, 2028: 7.5},
        # Kapitalbindung des erworbenen Geschaefts faellt mit dem Vollzug an
        "working_capital": {2026: 20.0},
        "ebitda_delta": 10.0,
        "ebitda_start": 2027,
        "remarks": {
            2026: "Vollzug der Akquisition; Kaufpreiszahlung und \\\"Ubernahme des Umlaufverm\\\"ogens",
            2027: "Integration; erster EBITDA-Beitrag aus dem erworbenen Gesch\\\"aft",
            2028: "Abschluss der Integration; Hebung der Synergien",
        },
    },
}

_ORDER = ("A", "B", "C")


def _ebitda(key: str, year: int) -> float:
    """Zusaetzliches EBITDA der Option im gegebenen Jahr (Mio. EUR)."""
    opt = ASSUMPTIONS[key]
    return opt["ebitda_delta"] if year >= opt["ebitda_start"] else 0.0


def _working_capital(key: str, year: int) -> float:
    """Aufbau der Kapitalbindung im gegebenen Jahr (Mio. EUR, Mittelabfluss)."""
    return ASSUMPTIONS[key]["working_capital"].get(year, 0.0)


def _investment(key: str, year: int) -> float:
    """Investitionsauszahlung im gegebenen Jahr (Mio. EUR)."""
    return ASSUMPTIONS[key]["investment"].get(year, 0.0)


def operating_cf(key: str, year: int) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Inkrementeller operativer Cash-Flow der Option: EBITDA-Zuwachs
        abzueglich des Aufbaus der Kapitalbindung.

    Inputs:
        key (str): "A", "B" oder "C".
        year (int): Geschaeftsjahr.

    Outputs:
        cf (float): Cash-Flow des Jahres in Mio. EUR.
    --------------------------------------------------------------------------
    """
    return _ebitda(key, year) - _working_capital(key, year)


def schedule(key: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Baut die Jahrestabelle 2026-2028 fuer eine Option (Aufgabe 6.1).

    Inputs:
        key (str): "A", "B" oder "C".

    Outputs:
        rows (list[dict]): je Jahr die Schluessel jahr, investition, op_cf,
                           liquiditaet, bemerkung.
    --------------------------------------------------------------------------
    """
    opt = ASSUMPTIONS[key]
    liq = OPENING_LIQUIDITY
    rows = []
    for year in (2026, 2027, 2028):
        inv = _investment(key, year)
        ocf = operating_cf(key, year)
        liq = liq - inv + ocf
        rows.append({
            "jahr": year,
            "investition": inv,
            "op_cf": ocf,
            "liquiditaet": liq,
            "bemerkung": opt["remarks"][year],
        })
    return rows


def full_cashflows(key: str, horizon_end: int = HORIZON_END) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Zahlungsreihe der Option ueber den Renditehorizont. Das zusaetzliche
        EBITDA laeuft ab seinem Startjahr konstant weiter; die Kapitalbindung
        wird im letzten Jahr freigesetzt.

    Inputs:
        key (str): "A", "B" oder "C".
        horizon_end (int): letztes Jahr des Horizonts.

    Outputs:
        flows (list[float]): ein Wert je Jahr von HORIZON_START bis horizon_end.
    --------------------------------------------------------------------------
    """
    total_wc = sum(ASSUMPTIONS[key]["working_capital"].values())
    flows = []
    for year in range(HORIZON_START, horizon_end + 1):
        flow = _ebitda(key, year) - _working_capital(key, year) - _investment(key, year)
        if year == horizon_end:
            flow += total_wc
        flows.append(flow)
    return flows


def npv(cashflows: list, rate: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Kapitalwert einer Zahlungsreihe; das erste Element liegt in t = 0.

    Inputs:
        cashflows (list[float]): Zahlungsreihe ab t = 0.
        rate (float): Kalkulationszins als Dezimalsatz.

    Outputs:
        value (float): Kapitalwert in Mio. EUR.
    --------------------------------------------------------------------------
    """
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows))


def irr(cashflows: list, lo: float = -0.95, hi: float = 5.0, tol: float = 1e-10) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Interner Zinsfuss der Zahlungsreihe per Bisektion.

    Inputs:
        cashflows (list[float]): Zahlungsreihe ab t = 0.
        lo, hi (float): Klammer der Nullstellensuche.
        tol (float): Abbruchbreite des Intervalls.

    Outputs:
        rate (float): Dezimalsatz, 0.12 entspricht 12 %.
    --------------------------------------------------------------------------
    """
    f_lo, f_hi = npv(cashflows, lo), npv(cashflows, hi)
    if f_lo * f_hi > 0:
        raise ValueError("Kein Vorzeichenwechsel im Suchintervall - IRR nicht bestimmbar.")
    for _ in range(500):
        mid = (lo + hi) / 2.0
        f_mid = npv(cashflows, mid)
        if abs(f_mid) < 1e-12 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def ebitda_marge_2028(key: str, umsatz: float, ebitda_basis: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        EBITDA-Marge Ende 2028 in Prozent. Die Umsatzbasis wird konstant
        fortgeschrieben; der EBITDA-Zuwachs der Option tritt hinzu.

    Inputs:
        key (str): "A", "B" oder "C".
        umsatz (float): Umsatzbasis in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.

    Outputs:
        marge (float): Marge in Prozent.
    --------------------------------------------------------------------------
    """
    return (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"]) / umsatz * 100.0


def ebitda_marge_2028_wachstum(key: str, umsatz: float, ebitda_basis: float,
                               wachstum: float = 0.03,
                               jahre: int = 3) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        EBITDA-Marge Ende 2028 bei wachsender Umsatzbasis. Der Basisfall
        schreibt den Umsatz 2025 konstant fort; da dieser Umsatz gegenueber
        2024 stark eingebrochen war, ist der Nenner klein und die
        ausgewiesene Marge entsprechend hoch. Diese Funktion zeigt, wie sich
        die Marge verschiebt, wenn der Umsatz wieder waechst.

    Inputs:
        key (str): "A", "B" oder "C".
        umsatz (float): Umsatzbasis des Geschaeftsjahres 2025 in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.
        wachstum (float): jaehrliche Umsatzwachstumsrate als Dezimalsatz.
        jahre (int): Anzahl der Wachstumsjahre bis 2028.

    Outputs:
        marge (float): Marge in Prozent.
    --------------------------------------------------------------------------
    """
    umsatz_2028 = umsatz * (1.0 + wachstum) ** jahre
    return (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"]) / umsatz_2028 * 100.0


def ebitda_marge_2028_referenz(key: str, umsatz_referenz: float,
                               ebitda_basis: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        EBITDA-Marge Ende 2028, bezogen auf ein frei gewaehltes Umsatzniveau
        im Nenner - im Dokument das Umsatzniveau des Geschaeftsjahres 2024.
        Die Groesse dient allein dem Vergleich; sie unterstellt nicht, dass
        dieses Niveau bis 2028 wieder erreicht wird.

    Inputs:
        key (str): "A", "B" oder "C".
        umsatz_referenz (float): Umsatz im Nenner in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.

    Outputs:
        marge (float): Marge in Prozent.
    --------------------------------------------------------------------------
    """
    return (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"]) / umsatz_referenz * 100.0


def net_debt_to_ebitda(key: str, ebitda_basis: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Net Debt / EBITDA Ende 2028. Eine positive Nettoliquiditaet ergibt
        einen negativen Verschuldungsgrad (Nettoguthaben).

    Inputs:
        key (str): "A", "B" oder "C".
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.

    Outputs:
        ratio (float): Verhaeltnis, negativ bei Nettoguthaben.
    --------------------------------------------------------------------------
    """
    liq = schedule(key)[-1]["liquiditaet"]
    return -liq / (ebitda_basis + ASSUMPTIONS[key]["ebitda_delta"])


# --- LaTeX-Ausgabe ---------------------------------------------------

def _n(value: float, digits: int = 1) -> str:
    """Zahl im englischen Format fuer siunitx (\\num formatiert spaeter)."""
    return f"{value:.{digits}f}"


def emit_schedule_table(key: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Tabellenkoerper fuer Aufgabe 6.1, ohne \\toprule / \\bottomrule.

    Inputs:
        key (str): "A", "B" oder "C".

    Outputs:
        body (str): drei LaTeX-Tabellenzeilen.
    --------------------------------------------------------------------------
    """
    lines = []
    for row in schedule(key):
        lines.append(
            f"{row['jahr']} & \\num{{{_n(row['investition'])}}} "
            f"& \\num{{{_n(row['op_cf'])}}} "
            f"& \\num{{{_n(row['liquiditaet'])}}} "
            f"& {row['bemerkung']} \\\\"
        )
    return "\n".join(lines) + "\n"


def emit_comparison_table(umsatz: float, ebitda_basis: float) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Tabellenkoerper fuer Aufgabe 6.3 (Vergleich der Optionen Ende 2028).

    Inputs:
        umsatz (float): Umsatzbasis in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.

    Outputs:
        body (str): sechs LaTeX-Tabellenzeilen.
    --------------------------------------------------------------------------
    """
    inv = {k: sum(ASSUMPTIONS[k]["investment"].values()) for k in _ORDER}
    rows = []
    rows.append("Kumulierte Investition (Mio.\\,\\euro) & "
                + " & ".join(f"\\num{{{_n(inv[k])}}}" for k in _ORDER) + " \\\\")
    rows.append("EBITDA-Zuwachs ab Jahr & "
                + " & ".join(
                    f"\\num{{{_n(ASSUMPTIONS[k]['ebitda_delta'])}}} ab {ASSUMPTIONS[k]['ebitda_start']}"
                    for k in _ORDER) + " \\\\")
    rows.append("EBITDA-Marge Ende 2028 & "
                + " & ".join(
                    f"\\num{{{_n(ebitda_marge_2028(k, umsatz, ebitda_basis))}}}\\,\\%"
                    for k in _ORDER) + " \\\\")
    rows.append("IRR (rd., Horizont 2026--2035) & "
                + " & ".join(
                    f"\\num{{{_n(irr(full_cashflows(k)) * 100.0)}}}\\,\\%" for k in _ORDER) + " \\\\")
    rows.append("Liquidit\\\"atsreserve Ende 2028 (Mio.\\,\\euro) & "
                + " & ".join(
                    f"\\num{{{_n(schedule(k)[-1]['liquiditaet'])}}}" for k in _ORDER) + " \\\\")
    rows.append("Net Debt/EBITDA & nicht relevant & nicht relevant & "
                + f"\\num{{{_n(net_debt_to_ebitda('C', ebitda_basis), 2)}}} \\\\")
    return "\n".join(rows) + "\n"


def emit_irr_sensitivity() -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Tabellenkoerper der IRR-Sensitivitaet nach Horizontlaenge. Ein IRR
        ohne erklaerten Horizont ist nicht aussagefaehig; die Tabelle zeigt,
        wie stark das Ergebnis von dieser Annahme abhaengt.

    Inputs:
        keine

    Outputs:
        body (str): drei LaTeX-Tabellenzeilen.
    --------------------------------------------------------------------------
    """
    rows = []
    for end in (2030, 2035, 2040):
        cells = " & ".join(
            f"\\num{{{_n(irr(full_cashflows(k, horizon_end=end)) * 100.0)}}}\\,\\%"
            for k in _ORDER)
        rows.append(f"{end - HORIZON_START + 1} Jahre (bis {end}) & {cells} \\\\")
    return "\n".join(rows) + "\n"


def emit_marge_sensitivity(umsatz: float, ebitda_basis: float,
                           umsatz_referenz: float,
                           wachstum: float = 0.03) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Tabellenkoerper der Margen-Sensitivitaet. Die EBITDA-Marge Ende 2028
        haengt vollstaendig davon ab, welcher Umsatz im Nenner steht. Der
        Basisfall schreibt den Umsatz 2025 konstant fort - ein Umsatz, der
        gegenueber 2024 stark eingebrochen war. Die Tabelle stellt diesem
        Fall zwei groessere Nenner gegenueber und macht damit sichtbar, dass
        die hohe Basismarge ein Effekt des kleinen Nenners ist.

    Inputs:
        umsatz (float): Umsatz des Geschaeftsjahres 2025 in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.
        umsatz_referenz (float): Umsatz des Geschaeftsjahres 2024 in Mio. EUR.
        wachstum (float): jaehrliche Wachstumsrate des mittleren Szenarios.

    Outputs:
        body (str): drei LaTeX-Tabellenzeilen.
    --------------------------------------------------------------------------
    """
    umsatz_gewachsen = umsatz * (1.0 + wachstum) ** 3
    prozent = _n(wachstum * 100.0, 0)
    szenarien = [
        (f"Umsatz 2025 konstant (\\num{{{_n(umsatz)}}} Mio.\\,\\euro, Basisfall)",
         lambda k: ebitda_marge_2028(k, umsatz, ebitda_basis)),
        (f"Umsatz +\\num{{{prozent}}}\\,\\% p.\\,a. bis 2028 "
         f"(\\num{{{_n(umsatz_gewachsen)}}} Mio.\\,\\euro)",
         lambda k: ebitda_marge_2028_wachstum(k, umsatz, ebitda_basis, wachstum)),
        (f"Umsatzniveau 2024 (\\num{{{_n(umsatz_referenz)}}} Mio.\\,\\euro)",
         lambda k: ebitda_marge_2028_referenz(k, umsatz_referenz, ebitda_basis)),
    ]
    rows = []
    for bezeichnung, fn in szenarien:
        cells = " & ".join(f"\\num{{{_n(fn(k))}}}\\,\\%" for k in _ORDER)
        rows.append(f"{bezeichnung} & {cells} \\\\")
    return "\n".join(rows) + "\n"


def emit_macros(umsatz: float, ebitda_basis: float) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Definiert die Ergebnisse der Rechnung als LaTeX-Makros. Die
        Wertungsabschnitte der Aufgabe 5 stuetzen sich auf dieselben Zahlen
        wie die Tabellen der Aufgabe 6; ohne diese Makros muessten sie
        hartkodiert werden und koennten auseinanderlaufen.

    Inputs:
        umsatz (float): Umsatzbasis in Mio. EUR.
        ebitda_basis (float): operatives EBITDA der Basis in Mio. EUR.

    Outputs:
        src (str): LaTeX-Quelltext mit \\newcommand-Definitionen.
    --------------------------------------------------------------------------
    """
    lines = []
    for key in _ORDER:
        opt = ASSUMPTIONS[key]
        invest = sum(opt["investment"].values())
        wc = sum(opt["working_capital"].values())
        werte = {
            "Investitionssumme": _n(invest),
            "Kapitalbindungssumme": _n(wc),
            "Kapitalbedarf": _n(invest + wc),
            "EbitdaZuwachs": _n(opt["ebitda_delta"]),
            "EbitdaStart": str(opt["ebitda_start"]),
            "Irr": _n(irr(full_cashflows(key)) * 100.0),
            "IrrFuenf": _n(irr(full_cashflows(key, horizon_end=2030)) * 100.0),
            "LiquiditaetEnde": _n(schedule(key)[-1]["liquiditaet"]),
            "MargeZweitausendachtundzwanzig":
                _n(ebitda_marge_2028(key, umsatz, ebitda_basis)),
            "InvestitionErstesJahr": _n(opt["investment"].get(2026, 0.0)),
            # Tiefster Bestand des Betrachtungszeitraums - bei C nicht das
            # Endjahr, sondern das Vollzugsjahr.
            "LiquiditaetTief": _n(min(r["liquiditaet"] for r in schedule(key))),
            # Gesamter Mittelabfluss des ersten Jahres, also Investitionsrate
            # zuzueglich des in diesem Jahr aufgebauten Umlaufvermoegens.
            "MittelabflussErstesJahr": _n(opt["investment"].get(2026, 0.0)
                                          + opt["working_capital"].get(2026, 0.0)),
        }
        for name, wert in werte.items():
            lines.append(f"\\newcommand{{\\{name}{key}}}{{{wert}}}")
        lines.append("")

    # Besonderheiten der Option C: Kaufpreis im Vollzugsjahr und
    # Verschuldungsgrad zum Ende des Betrachtungszeitraums.
    lines.append(r"\newcommand{\KaufpreisC}{" + _n(ASSUMPTIONS["C"]["investment"][2026]) + "}")
    lines.append(r"\newcommand{\NetDebtEbitdaC}{"
                 + _n(net_debt_to_ebitda("C", ebitda_basis), 2) + "}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die Basiswerte aus data/basis.json und schreibt die fuenf
        Tabellenkoerper nach data/.

    Inputs:
        keine (Kommandozeile ohne Argumente)

    Outputs:
        code (int): 0 bei Erfolg.
    --------------------------------------------------------------------------
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    with open(os.path.join(root, "data", "basis.json"), encoding="utf-8") as fh:
        basis = json.load(fh)
    umsatz = float(basis["umsatz_2025"])
    ebitda_basis = float(basis["ebitda_op_2025"])
    umsatz_referenz = float(basis["umsatz_2024"])

    outputs = {
        "cf_a.tex": emit_schedule_table("A"),
        "cf_b.tex": emit_schedule_table("B"),
        "cf_c.tex": emit_schedule_table("C"),
        "vergleich2028.tex": emit_comparison_table(umsatz, ebitda_basis),
        "irr_sensitivitaet.tex": emit_irr_sensitivity(),
        "marge_sensitivitaet.tex": emit_marge_sensitivity(
            umsatz, ebitda_basis, umsatz_referenz),
        "cf_makros.tex": emit_macros(umsatz, ebitda_basis),
    }
    header = "% Automatisch erzeugt von scripts/cashflow.py - nicht von Hand aendern.\n"
    for name, body in outputs.items():
        path = os.path.join(root, "data", name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + body)
        print(f"OK -> data/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
