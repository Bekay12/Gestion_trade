#!/usr/bin/env python3
"""
flow.py - Grobe Flow-Analyse auf Konzernebene fuer Brown & Brown, Inc.

Erzeugt die Tabellenkoerper der Teile 5.1 bis 5.4 und die Makros, mit denen
der Fliesstext dieselben Zahlen fuehrt wie die Tabellen:

    data/flow_pfad.tex         Entschuldungspfad 2026-2030, Basisszenario
    data/flow_szenarien.tex    Nettoverschuldung je EBITDAC in drei Szenarien
    data/flow_anleger.tex      Zahlungsreihe des Anlegers, Basisszenario
    data/flow_irr.tex          Interner Zinsfuss ueber drei Horizonte
    data/flow_irr_multiple.tex derselbe Zinsfuss ueber drei Ausstiegsvielfache
    data/flow_schwelle.tex     Einstiegskurs, bei dem der Zinsfuss die Huerde trifft
    data/flow_makros.tex       Einzelwerte fuer den Fliesstext

Alle Eingangsgroessen stammen aus scripts/kennzahlen.py und damit aus den
Primaerquellen; dieses Skript erfindet keinen Wert. Was es hinzufuegt, sind
die in Abschnitt 5.0 des Dokuments ausgeschriebenen Annahmen; sie stehen
unten in ANNAHMEN und nirgends sonst.

Der interne Zinsfuss wird durch Bisektion ueber ein festes Intervall
bestimmt, nicht von Hand geschaetzt.

Aufruf: python3 scripts/flow.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kennzahlen import ROH  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = {k: v[0] for k, v in ROH.items()}

JAHRE = [2026, 2027, 2028, 2029, 2030]

# --------------------------------------------------------------------------
# ANNAHMEN. Jede einzelne ist in Abschnitt 5.0 des Dokuments ausgeschrieben
# und begruendet. Keine Groesse steht hier, die aus den Quellen ableitbar
# waere - solche stehen in kennzahlen.py.
# --------------------------------------------------------------------------
# Organisches Wachstum: die drei Szenarien sind keine Schaetzungen, sondern
# drei vom Unternehmen selbst veroeffentlichte Raten, jeweils konstant
# fortgeschrieben. Damit muss keine Wachstumsannahme erfunden werden.
SZENARIEN = [
    ("Halbjahr 2026", W["HJOrganisch"] / 100),
    ('Gesch\\"aftsjahr 2025', W["OrganischWachstum"] / 100),
    ('Gesch\\"aftsjahr 2024', W["OrganischWachstumVJ"] / 100),
]
BASIS = 1  # das mittlere Szenario traegt die Haupttabellen

# Zinsaufwand aus dem veroeffentlichten Faelligkeitsprofil (10-K 2025, S. 46):
# 376 im ersten Jahr, 649 fuer die Jahre zwei und drei, 561 fuer vier und
# fuenf. Die Faelligkeiten werden also getilgt und nicht prolongiert - das
# ist die Annahme, die dem Profil des Unternehmens selbst zugrunde liegt.
ZINSEN = {
    2026: W["ZinsverpflichtungJahrEins"],
    2027: W["ZinsverpflichtungZweiDrei"] / 2,
    2028: W["ZinsverpflichtungZweiDrei"] / 2,
    2029: W["ZinsverpflichtungVierFuenf"] / 2,
    2030: W["ZinsverpflichtungVierFuenf"] / 2,
}
TILGUNG = {
    2026: W["TilgungJahrEins"],
    2027: W["TilgungJahrZweiDrei"] / 2,
    2028: W["TilgungJahrZweiDrei"] / 2,
    2029: W["TilgungJahrVierFuenf"] / 2,
    2030: W["TilgungJahrVierFuenf"] / 2,
}
# Erwerbspreisnachzahlungen aus demselben Profil: 405 im ersten Jahr,
# 437 verteilt auf die Jahre zwei und drei, danach keine.
EARNOUT = {2026: 405.0, 2027: 218.5, 2028: 218.5, 2029: 0.0, 2030: 0.0}

# Abschreibungen und Sachinvestitionen auf Jahresrate des ersten Halbjahres
# 2026 hochgerechnet: das Geschaeftsjahr 2025 enthaelt Accession nur fuenf
# Monate und untertreibt beide Groessen.
ABSCHREIBUNG = 2 * (W["HJAbschrImmat"] + W["HJAbschrSach"])
SACHINVEST = 2 * W["HJSachinvestitionen"]
# Dividende: Quartalsbeschluss vom 21.01.2026 auf vier Quartale und die zum
# 30.06.2026 umlaufenden Aktien. Keine weitere Erhoehung unterstellt.
DIVIDENDE_JE_AKTIE = 4 * W["DividendeQuartalNeu"]
DIVIDENDE = DIVIDENDE_JE_AKTIE * W["HJAktienUmlauf"]
# Aktienrueckkauf: der im ersten Halbjahr 2026 tatsaechlich getaetigte
# Betrag, danach keiner.
RUECKKAUF = {2026: W["HJRueckkauf"], 2027: 0.0, 2028: 0.0, 2029: 0.0, 2030: 0.0}
STEUERQUOTE = W["Ertragsteuern"] / W["ErgebnisVorSteuern"]
MARGE = W["EBITDACber"] / W["UmsatzGesamt"]
AKTIEN = W["HJAktienUmlauf"]
HORIZONTE = [5, 10, 15]
# LaTeX-Makronamen duerfen keine Ziffern enthalten.
HORIZONTNAME = {5: "Fuenf", 10: "Zehn", 15: "Fuenfzehn"}


def _kurs() -> float:
    """Liest den letzten Schlusskurs aus data/kurs_jahr.csv (Zeile 2026)."""
    pfad = os.path.join(ROOT, "data", "kurs_jahr.csv")
    with open(pfad, encoding="utf-8") as fh:
        for zeile in fh:
            if zeile.startswith("2026,"):
                return float(zeile.split(",")[1])
    raise ValueError("Kurszeile 2026 fehlt in data/kurs_jahr.csv")


def pfad(wachstum: float, jahre: list) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rechnet den Entschuldungspfad des Konzerns fort. Der nach Zins,
        Steuer, Sachinvestition, Dividende und Erwerbspreisnachzahlung
        verbleibende Betrag geht vollstaendig in die Schuldentilgung; die
        Liquiditaet bleibt auf dem Stand vom 31.12.2025. Das ist die
        Kapazitaetsrechnung "wie schnell kann der Konzern die Uebernahme
        abtragen" und nicht die Rechnung des Anlegers - dieselben Mittel
        koennen nur einmal verwendet werden. Die Anlegerrechnung steht in
        anlegerreihe() und bedient nur die planmaessige Tilgung.

        Der Zinsaufwand folgt dem veroeffentlichten Faelligkeitsprofil und
        sinkt nicht mit der vorzeitigen Tilgung. Die Rechnung untertreibt
        die Entschuldung dadurch, statt sie zu beschoenigen.

    Inputs:
        wachstum (float): organisches Wachstum je Jahr, z. B. 0.028
        jahre (list[int]): Jahre der Fortschreibung

    Outputs:
        zeilen (list[dict]): je Jahr Umsatz, EBITDAC, Zahlungsstroeme,
            Nettoverschuldung und Verschuldungsgrad
    --------------------------------------------------------------------------
    """
    umsatz = W["ProFormaUmsatz"]
    netto = W["Gesamtverschuldung"] - W["Zahlungsmittel"]
    zeilen = []
    for jahr in jahre:
        umsatz *= (1 + wachstum)
        ebitdac = umsatz * MARGE
        zinsen = ZINSEN.get(jahr, ZINSEN[2030])
        steuern = max(0.0, (ebitdac - ABSCHREIBUNG - zinsen) * STEUERQUOTE)
        earnout = EARNOUT.get(jahr, 0.0)
        rueckkauf = RUECKKAUF.get(jahr, 0.0)
        tilgung = TILGUNG.get(jahr, 0.0)
        frei = (ebitdac - zinsen - steuern - SACHINVEST - DIVIDENDE
                - earnout - rueckkauf)
        netto = max(0.0, netto - frei)
        zeilen.append({
            "jahr": jahr, "umsatz": umsatz, "ebitdac": ebitdac,
            "zinsen": zinsen, "steuern": steuern, "earnout": earnout,
            "tilgung": tilgung, "frei": frei, "netto": netto,
            "grad": netto / ebitdac,
        })
    return zeilen


def ergebnis_je_aktie(zeile: dict) -> float:
    """Ergebnis je Aktie eines Pfadjahres, nach Abschreibung, Zins und Steuer."""
    vor_steuern = zeile["ebitdac"] - ABSCHREIBUNG - zeile["zinsen"]
    return vor_steuern * (1 - STEUERQUOTE) / AKTIEN


def anlegerreihe(wachstum: float, horizont: int, vielfaches: float,
                 kurs: float) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Baut die Zahlungsreihe eines Anlegers, der heute eine Aktie kauft und
        am Ende des Horizonts verkauft. Zufluss je Jahr ist der an die
        Eigentuemer ausschuettbare Cash-Flow je Aktie: EBITDAC abzueglich
        Zins, Steuer, Sachinvestition, Erwerbspreisnachzahlung und
        planmaessiger Tilgung. Dividende und Aktienrueckkauf sind darin
        enthalten und werden nicht zusaetzlich abgezogen; wie sich der
        Betrag auf beide verteilt, aendert die Rendite des Anlegers nicht.

        Anders als pfad() wird hier nur die PLANMAESSIGE Tilgung bedient.
        Beide Rechnungen verwenden denselben Cash-Flow fuer verschiedene
        Zwecke und sind deshalb Alternativen, keine Ergaenzungen.

    Inputs:
        wachstum (float): organisches Wachstum je Jahr
        horizont (int): Haltedauer in Jahren
        vielfaches (float): Kurs-Gewinn-Verhaeltnis beim Verkauf
        kurs (float): Einstiegskurs in USD

    Outputs:
        reihe (list[float]): Zahlungen je Aktie und Jahr, Jahr 0 ist der Kauf
    --------------------------------------------------------------------------
    """
    zeilen = pfad(wachstum, list(range(2026, 2026 + horizont)))
    reihe = [-kurs]
    for i, z in enumerate(zeilen):
        ausschuettbar = (z["ebitdac"] - z["zinsen"] - z["steuern"]
                         - SACHINVEST - z["earnout"] - z["tilgung"])
        zahlung = ausschuettbar / AKTIEN
        if i == horizont - 1:
            zahlung += ergebnis_je_aktie(z) * vielfaches
        reihe.append(zahlung)
    return reihe


def irr(reihe: list, unten: float = -0.95, oben: float = 1.5) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Bestimmt den internen Zinsfuss durch Bisektion ueber ein festes
        Intervall. Kein Naeherungswert von Hand, kein Startwertproblem.

    Inputs:
        reihe (list[float]): Zahlungsreihe, Index 0 ist der Zeitpunkt null
        unten (float): untere Schranke des Suchintervalls
        oben (float): obere Schranke

    Outputs:
        zinsfuss (float): interner Zinsfuss als Dezimalzahl, oder float('nan'),
            wenn im Intervall kein Vorzeichenwechsel liegt
    --------------------------------------------------------------------------
    """
    def bar(r):
        return sum(z / (1 + r) ** i for i, z in enumerate(reihe))
    if bar(unten) * bar(oben) > 0:
        return float("nan")
    for _ in range(200):
        mitte = (unten + oben) / 2
        if bar(unten) * bar(mitte) <= 0:
            oben = mitte
        else:
            unten = mitte
    return (unten + oben) / 2


def huerde() -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt den Vergleichsmassstab des Anlageurteils zurueck: die
        Eigenkapitalrendite des Unternehmens auf das durchschnittliche
        Eigenkapital des Berichtsjahres. Bewusst KEIN Kapitalkostensatz nach
        dem Kapitalmarktmodell - Beta und Marktrisikopraemie stehen in keiner
        Primaerquelle, ein geschaetzter Satz waere eine Erfindung.

    Inputs:
        keine

    Outputs:
        rendite (float): Eigenkapitalrendite in Prozent
    --------------------------------------------------------------------------
    """
    mittel = (W["Eigenkapital"] + W["EigenkapitalVJ"]) / 2
    return W["Konzernergebnis"] / mittel * 100


def schwellenkurs(wachstum: float, horizont: int, vielfaches: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Bestimmt den Einstiegskurs, bei dem der interne Zinsfuss genau die
        Huerde trifft. Das ist die Zahl, die eine Anlageentscheidung
        tatsaechlich braucht: nicht "ist die Aktie gut", sondern "bis zu
        welchem Preis". Bisektion ueber den Kurs, weil der Zinsfuss mit
        steigendem Einstiegskurs streng faellt.

    Inputs:
        wachstum (float): organisches Wachstum je Jahr
        horizont (int): Haltedauer in Jahren
        vielfaches (float): Kurs-Gewinn-Verhaeltnis beim Verkauf

    Outputs:
        kurs (float): Einstiegskurs in USD
    --------------------------------------------------------------------------
    """
    ziel = huerde()
    unten, oben = 1.0, 1000.0
    for _ in range(120):
        mitte = (unten + oben) / 2
        r = irr(anlegerreihe(wachstum, horizont, vielfaches, mitte)) * 100
        if r != r:            # kein Vorzeichenwechsel: Kurs zu hoch
            oben = mitte
        elif r > ziel:
            unten = mitte
        else:
            oben = mitte
    return (unten + oben) / 2


def _z(x: float, stellen: int = 0) -> str:
    """Zahl im deutschen Format fuer siunitx (Punkt bleibt Dezimaltrenner)."""
    return f"{x:.{stellen}f}"


def main() -> None:
    kurs = _kurs()
    basis_w = SZENARIEN[BASIS][1]
    zeilen = pfad(basis_w, JAHRE)
    aus = {}

    # 5.1 Entschuldungspfad
    zeilen_tex = []
    for z in zeilen:
        zeilen_tex.append(
            f"{z['jahr']} & \\num{{{_z(z['ebitdac'])}}} & \\num{{{_z(z['zinsen'])}}} & "
            f"\\num{{{_z(z['steuern'])}}} & \\num{{{_z(z['earnout'])}}} & "
            f"\\num{{{_z(z['tilgung'])}}} & \\num{{{_z(z['frei'])}}} & "
            f"\\num{{{_z(z['netto'])}}} & \\num{{{_z(z['grad'], 2)}}}\\\\")
    aus["flow_pfad"] = "\n".join(zeilen_tex)

    # 5.2 Verschuldungsgrad je Szenario
    zeilen_tex = []
    for name, w in SZENARIEN:
        p = pfad(w, JAHRE)
        werte = " & ".join(f"\\num{{{_z(z['grad'], 2)}}}" for z in p)
        zeilen_tex.append(f"{name} & \\num{{{_z(w * 100, 1)}}} & {werte}\\\\")
    aus["flow_szenarien"] = "\n".join(zeilen_tex)

    # 5.3 Zahlungsreihe des Anlegers ueber den mittleren Horizont
    vielfaches = kurs / W["ProFormaEPSdil"]
    reihe = anlegerreihe(basis_w, HORIZONTE[0], vielfaches, kurs)
    zeilen_tex = [f"0 & \\num{{{_z(-kurs, 2)}}} & Kauf einer Aktie\\\\"]
    for i, z in enumerate(reihe[1:], start=1):
        jahr = 2025 + i
        bem = ('aussch\\"uttbarer Cash-Flow' if i < len(reihe) - 1
               else 'aussch\\"uttbarer Cash-Flow und Verkaufserl\\"os')
        zeilen_tex.append(f"{i} & \\num{{{_z(z, 2)}}} & {bem} ({jahr})\\\\")
    aus["flow_anleger"] = "\n".join(zeilen_tex)

    # 5.4 Zinsfuss je Horizont und Szenario
    zeilen_tex = []
    for h in HORIZONTE:
        felder = []
        for _, w in SZENARIEN:
            r = irr(anlegerreihe(w, h, vielfaches, kurs))
            felder.append(f"\\num{{{_z(r * 100, 1)}}}")
        zeilen_tex.append(f"{h} Jahre & " + " & ".join(felder) + "\\\\")
    aus["flow_irr"] = "\n".join(zeilen_tex)

    # 5.4 Zinsfuss je Horizont und Ausstiegsvielfachem
    vielfache = [15.0, vielfaches, 25.0]
    zeilen_tex = []
    for h in HORIZONTE:
        felder = []
        for v in vielfache:
            r = irr(anlegerreihe(basis_w, h, v, kurs))
            felder.append(f"\\num{{{_z(r * 100, 1)}}}")
        zeilen_tex.append(f"{h} Jahre & " + " & ".join(felder) + "\\\\")
    aus["flow_irr_multiple"] = "\n".join(zeilen_tex)

    # 7.3 Einstiegsschwelle: Kurs, bei dem der Zinsfuss die Huerde trifft
    zeilen_tex = []
    for name, w in SZENARIEN:
        felder = [f"\\num{{{_z(schwellenkurs(w, h, vielfaches), 2)}}}"
                  for h in HORIZONTE]
        zeilen_tex.append(f"{name} & \\num{{{_z(w * 100, 1)}}} & "
                          + " & ".join(felder) + "\\\\")
    aus["flow_schwelle"] = "\n".join(zeilen_tex)

    # Makros fuer den Fliesstext
    letzte = zeilen[-1]
    makros = {
        "FlowEinstiegskurs": _z(kurs, 2),
        "FlowVielfaches": _z(vielfaches, 1),
        "FlowMarge": _z(MARGE * 100, 2),
        "FlowSteuerquote": _z(STEUERQUOTE * 100, 1),
        "FlowAbschreibung": _z(ABSCHREIBUNG, 0),
        "FlowSachinvest": _z(SACHINVEST, 0),
        "FlowDividendeJeAktie": _z(DIVIDENDE_JE_AKTIE, 2),
        "FlowDividende": _z(DIVIDENDE, 0),
        "FlowEbitdacEnde": _z(letzte["ebitdac"], 0),
        "FlowNettoEnde": _z(letzte["netto"], 0),
        "FlowGradEnde": _z(letzte["grad"], 2),
        "FlowGradStart": _z(zeilen[0]["grad"], 2),
        "FlowFreiJahrEins": _z(zeilen[0]["frei"], 0),
        "FlowTilgungFuenfJahre": _z(sum(TILGUNG.values()), 0),
        "FlowZinsenFuenfJahre": _z(sum(ZINSEN.values()), 0),
        "FlowEPSEnde": _z(ergebnis_je_aktie(letzte), 2),
        "FlowVielfachesUnten": _z(vielfache[0], 0),
        "FlowVielfachesOben": _z(vielfache[2], 0),
        "FlowHuerde": _z(huerde(), 1),
    }
    # Schwellenkurse als Einzelmakros: der Verdikt-Abschnitt nennt sie im Text.
    for i, (_, w) in enumerate(SZENARIEN):
        for h in HORIZONTE:
            makros[f"FlowSchwelle{'ABC'[i]}{HORIZONTNAME[h]}"] = _z(
                schwellenkurs(w, h, vielfaches), 2)
    # Abstand des heutigen Kurses zur Schwelle des mittleren Szenarios auf
    # dem laengsten Horizont: die Kernzahl des Votums.
    _s = schwellenkurs(basis_w, HORIZONTE[-1], vielfaches)
    makros["FlowSchwelleBasis"] = _z(_s, 2)
    makros["FlowAbstandSchwelle"] = _z((kurs / _s - 1) * 100, 1)
    _su = schwellenkurs(SZENARIEN[0][1], HORIZONTE[-1], vielfaches)
    makros["FlowSchwelleUnten"] = _z(_su, 2)
    makros["FlowAbstandUnten"] = _z((kurs / _su - 1) * 100, 1)
    for i, (name, w) in enumerate(SZENARIEN):
        p = pfad(w, JAHRE)
        makros[f"FlowGradSzenario{'ABC'[i]}"] = _z(p[-1]["grad"], 2)
        for h in HORIZONTE:
            r = irr(anlegerreihe(w, h, vielfaches, kurs))
            makros[f"FlowIRR{'ABC'[i]}{HORIZONTNAME[h]}"] = _z(r * 100, 1)
    for h in HORIZONTE:
        makros[f"FlowIRRunten{HORIZONTNAME[h]}"] = _z(
            irr(anlegerreihe(basis_w, h, vielfache[0], kurs)) * 100, 1)
        makros[f"FlowIRRoben{HORIZONTNAME[h]}"] = _z(
            irr(anlegerreihe(basis_w, h, vielfache[2], kurs)) * 100, 1)

    zeilen_tex = ["% ERZEUGT von scripts/flow.py. Nicht haendisch aendern."]
    for name, wert in makros.items():
        zeilen_tex.append(f"\\newcommand{{\\{name}}}{{{wert}}}")
    aus["flow_makros"] = "\n".join(zeilen_tex)

    for name, inhalt in aus.items():
        with open(os.path.join(ROOT, "data", f"{name}.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write(inhalt + "\n")
    print(f"[FLOW] {len(aus)} Dateien geschrieben; "
          f"Verschuldungsgrad {makros['FlowGradStart']} -> "
          f"{makros['FlowGradEnde']}, "
          f"IRR 5/10/15 Jahre = {makros['FlowIRRBFuenf']} / "
          f"{makros['FlowIRRBZehn']} / {makros['FlowIRRBFuenfzehn']} Prozent")


if __name__ == "__main__":
    main()
