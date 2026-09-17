#!/usr/bin/env python3
"""
flow.py - Grobe Flow-Analyse fuer Alamos Gold Inc.

Erzeugt die Tabellenkoerper des Teils 5 und die Makros, mit denen der
Fliesstext dieselben Zahlen fuehrt wie die Tabellen:

    data/flow_kapazitaet.tex   Finanzierungskapazitaet des Wachstumsprogramms
    data/flow_anleger.tex      Zahlungsreihe des Anlegers, mittleres Szenario
    data/flow_irr.tex          Interner Zinsfuss ueber drei Horizonte
    data/flow_schwelle.tex     Einstiegskurs, bei dem der Zinsfuss die Huerde trifft
    data/flow_wachstum.tex     Wachstumsrate, die der heutige Kurs voraussetzt
    data/flow_endwert.tex      Endwert, den der heutige Kurs voraussetzt
    data/flow_sensitiv.tex     Zinsfuss bei abweichendem Wachstumskapital
    data/flow_makros.tex       Einzelwerte fuer den Fliesstext

Alle Eingangsgroessen stammen aus scripts/kennzahlen.py und damit aus den
Primaerquellen; dieses Skript erfindet keinen Wert. Was es hinzufuegt, sind
die in Abschnitt 5.0 des Dokuments ausgeschriebenen Annahmen; sie stehen
unten in einem einzigen Block und nirgends sonst.

ZWEI RECHNUNGEN, DIE EINANDER AUSSCHLIESSEN. Abschnitt 5.1 fragt, ob das
Wachstumsprogramm aus eigener Kraft finanzierbar ist; dort geht jeder
Ueberschuss in die Investition. Abschnitt 5.3 fragt, was ein Anleger beim
Einstieg heute verdient; dort geht der Ueberschuss an die Eigentuemer.
Dieselben Mittel koennen nur einmal verwendet werden - die beiden Tabellen
sind Alternativen und nicht Bestandteile derselben Zukunft.

Ein Entschuldungspfad, wie ihn die Schwesteranalyse ueber Brown & Brown
rechnet, entfaellt hier: Alamos haelt mehr Kasse als Finanzschulden. Die
Frage, die an seine Stelle tritt, ist die Finanzierungsfrage.

Die Umkehrungen (Schwellenkurs, erforderliches Wachstum, erforderlicher
Endwert) folgen der Methode der Skill startup-investment-analyzer; sie werden
durchweg durch Bisektion ueber ein erklaertes Intervall bestimmt und nie
geschaetzt.

Aufruf: python3 scripts/flow.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kennzahlen import ROH, EXTERN, ABGELEITET, median, _kurse, ROEMISCH, JAHRE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = {k: v[0] for k, v in ROH.items()}
W.update({k: v[0] for k, v in EXTERN.items()})
W.update({k: v for k, v in _kurse().items() if isinstance(v, float)})
for _name, _ausdruck in ABGELEITET:
    W[_name] = eval(_ausdruck, {"__builtins__": {}}, W)  # noqa: S307

# --------------------------------------------------------------------------
# ANNAHMEN. Jede einzelne ist in Abschnitt 5.0 des Dokuments ausgeschrieben
# und begruendet. Keine Groesse steht hier, die aus den Quellen ableitbar
# waere - solche stehen in kennzahlen.py.
# --------------------------------------------------------------------------
# Die Kapazitaetsrechnung endet mit dem veroeffentlichten Bauprogramm. Sie
# weiter fortzuschreiben liesse Kasse ohne unterstellte Verwendung auflaufen,
# und ein Bergbaukonzern, der derart Mittel erwirtschaftet, schuettet aus,
# kauft zu oder baut weiter - er hortet sie nicht. Die Frage dieser Tabelle
# ist, ob das Programm getragen wird, und das Programm endet 2029.
JAHRE_PFAD = [2026, 2027, 2028, 2029]

# Die drei Szenarien sind keine Schaetzungen, sondern drei vom Unternehmen
# selbst berichtete DURCHSCHNITTLICH ERZIELTE Goldpreise, jeweils konstant
# fortgeschrieben. Damit muss keine Preisannahme erfunden werden. Dass der
# Goldpreis die tragende Groesse ist und nicht das Mengenwachstum, ist der
# Unterschied zum Muster: Bei einem Makler traegt das organische Wachstum,
# bei einem Goldproduzenten der Preis.
SZENARIEN = [
    ("Erzielt 2024", W["GoldpreisVJ"]),
    ("Erzielt 2025", W["Goldpreis"]),
    ("Erzielt 1. Halbjahr 2026", W["HJGoldpreis"]),
]
BASIS = 1  # das mittlere Szenario traegt die Haupttabellen

# Menge und Stueckkosten folgen dem vom Unternehmen selbst
# veroeffentlichten Dreijahresausblick vom 04.02.2026 und werden danach
# konstant gehalten. Das ist KEINE eigene Prognose: Die Zahlen stehen im
# MD&A 2025 auf S. 7 und tragen dieselbe Belegpflicht wie jede andere.
#
# Warum sie noetig sind: Die Rechnung belastet jedes Jahr mit dem
# Wachstumskapital. Haelt man zugleich die Produktion auf dem Stand von 2026,
# bezahlt das Modell fuenfzehn Jahre lang eine Erweiterung und erhaelt sie
# nie. Das ist kein vorsichtiger Ansatz, sondern ein falscher.
#
# 2026-2027   Prognose 2026: 570-650 Tsd. Unzen, AISC (angehoben) 1775-1875
# ab 2028     Dreijahresausblick: 755-835 Tsd. Unzen, AISC 18 % unter 2025
MENGE_SECHS = (W["PrognoseProdUnten"] + W["PrognoseProdOben"]) / 2 * 1000
MENGE_ACHT = (W["ProgAchtProdUnten"] + W["ProgAchtProdOben"]) / 2 * 1000
AISC_SECHS = (W["PrognoseAiscUntenNeu"] + W["PrognoseAiscObenNeu"]) / 2
AISC_ACHT = W["AISCXXV"] * (1 - W["ProgAchtAISCRueckgang"] / 100)
# 2027 traegt keine eigene veroeffentlichte Zahl. Es wird auf dem Stand von
# 2026 gehalten und nicht zwischen 2026 und 2028 interpoliert: eine
# Interpolation waere eine erfundene Zwischenstufe, und die Zurueckhaltung
# faellt zulasten der Rechnung und nicht zu ihren Gunsten.
UMSTELLJAHR = 2028

# Wachstumskapital und aktivierte Exploration stehen ausserhalb der AISC und
# muessen deshalb eigens abgezogen werden; die AISC enthalten allein das
# Erhaltungskapital. Das veroeffentlichte Bauprogramm endet mit der ersten
# Produktion von Lynn Lake im ersten Halbjahr 2029 (MD&A 2025, S. 13); ab
# 2030 traegt die Rechnung deshalb kein Wachstumskapital mehr. Diese Annahme
# entscheidet das Ergebnis mit und wird in data/flow_sensitiv.tex in beide
# Richtungen durchgerechnet.
WACHSTUMSKAPITAL = (W["PrognoseWachstumsCapexUnten"] + W["PrognoseWachstumsCapexOben"]) / 2
BAUENDE = 2029
EXPLORATION = W["ExplorationAktiviert"]
STEUERQUOTE = W["Steuersatz"] / 100
HORIZONTE = [5, 10, 15]
HORIZONTNAME = {5: "Fuenf", 10: "Zehn", 15: "Fuenfzehn"}
SZENARIONAME = {0: "A", 1: "B", 2: "C"}
# Endwert der Beteiligung am Ende des Horizonts: der Buchwert des
# Eigenkapitals je Aktie. Belegt (Abschluss 2025, S. 6), unabhaengig vom
# Einstiegskurs und fuer eine ueber Buchwert notierende Aktie vorsichtig.
# Ein Bewertungsvielfaches waere genau der plausibel aussehende Platzhalter,
# den die Methode verbietet. Was der heutige Kurs statt dessen voraussetzt,
# rechnet erforderlicher_endwert() aus.
ENDWERT_JE_AKTIE = W["BuchwertJeAktie"]
AKTIEN_MIO = W["AktienMio"]
STARTJAHR = 2026


def menge(jahr: int) -> float:
    """Gefoerderte Unzen des Jahres nach dem veroeffentlichten Ausblick."""
    return MENGE_SECHS if jahr < UMSTELLJAHR else MENGE_ACHT


def aisc(jahr: int) -> float:
    """All-in Sustaining Costs je Unze nach dem veroeffentlichten Ausblick."""
    return AISC_SECHS if jahr < UMSTELLJAHR else AISC_ACHT


def wachstumskapital(jahr: int, faktor: float = 1.0, bauende: int = None) -> float:
    """Wachstumskapital des Jahres; nach dem Ende des Bauprogramms null."""
    ende = BAUENDE if bauende is None else bauende
    return WACHSTUMSKAPITAL * faktor if jahr <= ende else 0.0


def verteilbar(preis: float, jahr: int, faktor: float = 1.0,
               bauende: int = None) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt den verteilbaren Cash-Flow eines Jahres in Mio. USD zurueck.

        Der Aufbau folgt der Kostenkennzahl, die das Unternehmen selbst
        veroeffentlicht: Die All-in Sustaining Costs enthalten das
        Erhaltungskapital bereits, nicht aber das Wachstumskapital und nicht
        die aktivierte Exploration. Die Rechnung lautet deshalb

            Menge x (Goldpreis - AISC) - Wachstumskapital - Exploration - Steuer

        und nicht "operativer Cash-Flow minus Investitionen": die zweite Form
        liesse sich nicht nach dem Goldpreis aufloesen, und der Goldpreis ist
        die Groesse, die dieses Geschaeft bewegt.

    Inputs:
        preis (float): erzielter Goldpreis je Unze
        jahr (int): Kalenderjahr, bestimmt Menge, Kosten und Bauprogramm
        faktor (float): Vielfaches des Wachstumskapitals fuer die Sensitivitaet
        bauende (int): letztes Jahr mit Wachstumskapital, sonst BAUENDE

    Outputs:
        cashflow (float): verteilbarer Cash-Flow in Mio. USD
    --------------------------------------------------------------------------
    """
    marge = menge(jahr) * (preis - aisc(jahr)) / 1e6
    vor_steuer = marge - wachstumskapital(jahr, faktor, bauende) - EXPLORATION
    return vor_steuer * (1 - STEUERQUOTE) if vor_steuer > 0 else vor_steuer


def eichung() -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Wendet das Modell auf das Geschaeftsjahr 2025 an und stellt das
        Ergebnis dem tatsaechlichen freien Cash-Flow gegenueber. Ein Modell,
        das nicht gegen ein Ist-Jahr gehalten wird, ist eine Behauptung.

        Die Abweichung ist nicht wegzurechnen, sondern zu benennen: 2025 trug
        die Ausbuchung der von Argonaut uebernommenen Goldtermingeschaefte
        und einen Aufbau des Umlaufvermoegens, und beides steht in keiner
        Kostenkennzahl.

    Inputs:
        keine

    Outputs:
        befund (dict): Modellwert, Ist-Wert und die benannte Differenz
    --------------------------------------------------------------------------
    """
    marge = W["GoldVerkauft"] * (W["Goldpreis"] - W["AISCXXV"]) / 1e6
    wachstum = W["Sachinvestitionen"] - W["ErhaltungsKapital"]
    modell = marge - wachstum - W["SteuerLaufend"]
    ist = W["OpCashflow"] - W["Sachinvestitionen"]
    # Die vier Posten, die der Kostenkennzahl ihrem Wesen nach fehlen: eine
    # einmalige Ausbuchung uebernommener Termingeschaefte, eine einmalige
    # Vorauszahlung auf kuenftige Lieferungen, der Aufbau des
    # Umlaufvermoegens samt gezahlter Steuern und die aktivierten Bauzinsen.
    erklaert = (-W["HedgeAusbuchung"] + W["GoldVorauszahlung"]
                - W["UmlaufvermoegenSteuern"] - W["ZinsAktiviert"])
    return {"marge": marge, "wachstum": wachstum, "modell": modell, "ist": ist,
            "differenz": modell - ist, "erklaert": -erklaert,
            "rest": (modell - ist) + erklaert}


def huerde() -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt den Vergleichsmassstab des Anlageurteils zurueck: die
        Eigenkapitalrendite des Unternehmens auf das durchschnittliche
        Eigenkapital, gerechnet auf das BEREINIGTE Ergebnis.

        Die berichtete Rendite von rund 22 % traegt eine Wertaufholung von
        218,8 und einen Veraeusserungsgewinn von 231,0 Mio. USD. Beide sind
        einmalig; eine einmalige Bilanzkorrektur gibt keinen Massstab fuer
        eine Daueranlage ab. Bewusst KEIN Kapitalkostensatz nach dem
        Kapitalmarktmodell - Beta und Marktrisikopraemie stehen in keiner
        Primaerquelle, ein geschaetzter Satz waere eine Erfindung.

    Inputs:
        keine

    Outputs:
        rendite (float): Eigenkapitalrendite in Prozent
    --------------------------------------------------------------------------
    """
    return W["EigenkapitalrenditeBer"]


def anlegerreihe(preis: float, horizont: int, kurs: float = None,
                 faktor: float = 1.0, wachstum: float = 0.0,
                 bauende: int = None, endwert: float = None) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Zahlungsreihe je Aktie aus Sicht eines Anlegers, der heute kauft:
        der Kurs als Auszahlung, danach der verteilbare Cash-Flow je Aktie,
        im letzten Jahr zuzueglich des Endwerts der Beteiligung.

        Der Endwert entscheidet das Ergebnis und darf deshalb nicht fehlen.
        Ohne ihn beantwortete die Reihe eine andere Frage - wie lange die
        Ausschuettungen allein brauchen, um den Kaufpreis zur Huerde
        zurueckzuzahlen -, und deren Schwelle liegt bei jedem Unternehmen
        weit unter jedem Marktpreis.

    Inputs:
        preis (float): erzielter Goldpreis je Unze
        horizont (int): Zahl der Jahre
        kurs (float): Einstiegskurs, sonst der aktuelle Schlusskurs
        faktor (float): Vielfaches des Wachstumskapitals
        wachstum (float): jaehrliche Wachstumsrate des Cash-Flows, dezimal

    Outputs:
        reihe (list): Zahlungen je Aktie, Stelle 0 ist der Einstieg
    --------------------------------------------------------------------------
    """
    k = W["KursSchlussXXVI"] if kurs is None else kurs
    reihe = [-k]
    for i in range(1, horizont + 1):
        je_aktie = verteilbar(preis, STARTJAHR + i - 1, faktor, bauende) / AKTIEN_MIO
        zahlung = je_aktie * (1 + wachstum) ** (i - 1)
        if i == horizont:
            zahlung += ENDWERT_JE_AKTIE if endwert is None else endwert
        reihe.append(zahlung)
    return reihe


def barwert(reihe: list, zins: float) -> float:
    """Barwert einer Zahlungsreihe; Stelle 0 liegt im Zeitpunkt null."""
    return sum(z / (1 + zins) ** i for i, z in enumerate(reihe))


def irr(reihe: list, unten: float = -0.95, oben: float = 2.0) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Interner Zinsfuss durch Bisektion ueber ein erklaertes Intervall.
        Von Hand geschaetzt oder aus einer Tabellenkalkulation uebernommen
        waere er nicht nachvollziehbar; das Intervall steht hier und wird
        nicht so lange aufgeweitet, bis eine Zahl erscheint.

    Inputs:
        reihe (list): Zahlungsreihe
        unten, oben (float): Intervallgrenzen, dezimal

    Outputs:
        zins (float): interner Zinsfuss dezimal, oder None ohne Vorzeichenwechsel
    --------------------------------------------------------------------------
    """
    a, b = barwert(reihe, unten), barwert(reihe, oben)
    if a * b > 0:
        return None
    for _ in range(200):
        m = (unten + oben) / 2
        if barwert(reihe, m) * a > 0:
            unten = m
        else:
            oben = m
    return (unten + oben) / 2


def schwellenkurs(preis: float, horizont: int, faktor: float = 1.0,
                  endwert: float = None) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt den Einstiegskurs zurueck, bei dem der interne Zinsfuss die
        Huerde genau trifft. Das ist die Zahl, die eine Anlageentscheidung
        braucht: nicht ob der Fall gut ist, sondern bis zu welchem Preis.

    Inputs:
        preis (float): erzielter Goldpreis je Unze
        horizont (int): Zahl der Jahre
        faktor (float): Vielfaches des Wachstumskapitals

    Outputs:
        kurs (float): Schwellenkurs in USD, oder None ausserhalb des Intervalls
    --------------------------------------------------------------------------
    """
    ziel = huerde() / 100
    unten, oben = 0.01, 500.0

    def f(k):
        return barwert(anlegerreihe(preis, horizont, k, faktor, 0.0, None, endwert), ziel)

    if f(unten) * f(oben) > 0:
        return None
    for _ in range(200):
        m = (unten + oben) / 2
        if f(m) * f(unten) > 0:
            unten = m
        else:
            oben = m
    return (unten + oben) / 2


def erforderliches_wachstum(preis: float, horizont: int,
                            unten: float = -0.5, oben: float = 2.0) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die jaehrliche Wachstumsrate des verteilbaren Cash-Flows zurueck,
        die der HEUTIGE Kurs bereits voraussetzt.

        Das ist die Umkehrung, die die stehende Schwaeche eines
        Niveaumodells repariert. Den Cash-Flow konstant zu halten haelt die
        Rechnung ehrlich - sie prognostiziert nichts - und laesst zugleich
        jedes wachsende Unternehmen teuer aussehen, weil der ganze Fall in
        dem Glied steckt, das das Modell weggelassen hat. Die Umkehrung gibt
        dieses Glied zurueck, OHNE zu prognostizieren: Der Bericht nennt die
        Rate, die im Preis steckt, und der Leser haelt sie gegen die Raten,
        die dieses Unternehmen tatsaechlich geliefert hat.

    Inputs:
        preis (float): erzielter Goldpreis je Unze
        horizont (int): Zahl der Jahre
        unten, oben (float): Intervall der Bisektion, dezimal

    Outputs:
        rate (float): dezimale Wachstumsrate, oder None, wenn der Kurs auch
            am oberen Rand des Intervalls nicht zu rechtfertigen ist
    --------------------------------------------------------------------------
    """
    ziel = huerde() / 100

    def f(g):
        return barwert(anlegerreihe(preis, horizont, None, 1.0, g), ziel)

    if f(unten) * f(oben) > 0:
        return None
    for _ in range(200):
        m = (unten + oben) / 2
        if f(m) * f(unten) > 0:
            unten = m
        else:
            oben = m
    return (unten + oben) / 2


def erforderlicher_endwert(preis: float, horizont: int) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt den Endwert je Aktie zurueck, den der heutige Kurs voraussetzt,
        damit der interne Zinsfuss die Huerde trifft. Als Vielfaches des
        Buchwerts gelesen sagt er, wie viel des heutigen Preises auf dem
        Ausstieg ruht und wie wenig auf den Ausschuettungen - und macht damit
        eine Einstiegsschwelle weit unter dem Marktkurs verstaendlich statt
        bloss alarmierend.

    Inputs:
        preis (float): erzielter Goldpreis je Unze
        horizont (int): Zahl der Jahre

    Outputs:
        endwert (float): Endwert je Aktie in USD
    --------------------------------------------------------------------------
    """
    ziel = huerde() / 100
    kurs = W["KursSchlussXXVI"]
    # Barwert der laufenden Zahlungen ohne Endwert
    laufend = sum(verteilbar(preis, STARTJAHR + i - 1) / AKTIEN_MIO / (1 + ziel) ** i
                  for i in range(1, horizont + 1))
    return (kurs - laufend) * (1 + ziel) ** horizont


def kapazitaet(preis: float) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rechnet fort, ob das Wachstumsprogramm aus eigener Kraft finanzierbar
        ist: Anfangsbestand ist die Liquiditaet zum 30.06.2026, jedes Jahr
        kommt die AISC-Marge hinzu, ab gehen Wachstumskapital, Exploration,
        Steuer und Dividende. Das ist eine KAPAZITAETSRECHNUNG und keine
        Vorhersage; sie sagt, was das Programm an Mitteln verlangt, und nicht,
        welcher Goldpreis eintreten wird.

    Inputs:
        preis (float): erzielter Goldpreis je Unze

    Outputs:
        pfad (list): je Jahr ein dict mit den Zeilen der Tabelle
    --------------------------------------------------------------------------
    """
    bestand = W["HJLiquiditaet"]
    dividende = W["DividendeQuartalNeu"] * 4 * AKTIEN_MIO
    pfad = []
    for jahr in JAHRE_PFAD:
        marge = menge(jahr) * (preis - aisc(jahr)) / 1e6
        invest = wachstumskapital(jahr) + EXPLORATION
        steuer = max(0.0, marge - invest) * STEUERQUOTE
        veraenderung = marge - invest - steuer - dividende
        bestand += veraenderung
        pfad.append({"jahr": jahr, "marge": marge, "invest": invest,
                     "steuer": steuer, "dividende": dividende,
                     "veraenderung": veraenderung, "bestand": bestand})
    return pfad


def _z(x: float, stellen: int = 0) -> str:
    """Zahl im deutschen Satz: Komma als Dezimaltrennzeichen, kein Tausenderpunkt."""
    if x is None:
        return "--"
    return f"{x:,.{stellen}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def _schreib(name: str, zeilen: list) -> None:
    pfad = os.path.join(ROOT, "data", name)
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    with open(pfad, "w", encoding="utf-8") as fh:
        fh.write("\n".join(zeilen) + "\n")


def main() -> None:
    makros = []

    def makro(name: str, wert, stellen: int = None):
        makros.append(f"\\newcommand{{\\Flow{name}}}{{"
                      + (f"{wert:.10g}" if stellen is None else _z(wert, stellen)) + "}")

    preis_basis = SZENARIEN[BASIS][1]

    # --- 5.1 Finanzierungskapazitaet -------------------------------------
    zeilen = []
    for z in kapazitaet(preis_basis):
        zeilen.append(f"{z['jahr']} & {_z(z['marge'])} & {_z(z['invest'])} & "
                      f"{_z(z['steuer'])} & {_z(z['dividende'])} & "
                      f"{_z(z['veraenderung'])} & {_z(z['bestand'])}\\\\")
    _schreib("flow_kapazitaet.tex", zeilen)
    pfad_basis = kapazitaet(preis_basis)
    makro("BestandEnde", pfad_basis[-1]["bestand"])
    makro("BestandTief", min(z["bestand"] for z in pfad_basis))
    # Dieselbe Rechnung im unguenstigsten Szenario: die Frage, ob das
    # Bauprogramm auch dann getragen wird, wenn der Goldpreis auf den
    # Stand von 2024 zurueckfaellt.
    pfad_unten = kapazitaet(SZENARIEN[0][1])
    makro("BestandTiefUnten", min(z["bestand"] for z in pfad_unten))
    makro("BestandEndeUnten", pfad_unten[-1]["bestand"])

    # --- 5.2 Eichung des Modells -----------------------------------------
    e = eichung()
    for name, wert in (("EichMarge", e["marge"]), ("EichWachstum", e["wachstum"]),
                       ("EichModell", e["modell"]), ("EichIst", e["ist"]),
                       ("EichDifferenz", e["differenz"]), ("EichErklaert", e["erklaert"]),
                       ("EichRest", e["rest"])):
        makro(name, wert)

    # --- 5.3 Anlegerreihe, mittleres Szenario ----------------------------
    zeilen = []
    reihe = anlegerreihe(preis_basis, 15)
    for i, zahlung in enumerate(reihe):
        jahr = STARTJAHR - 1 + i
        if i == 0:
            zeilen.append(f"{jahr} & Einstieg & {_z(zahlung, 2)} \\\\")
        else:
            bem = ("Bauprogramm" if STARTJAHR + i - 1 <= BAUENDE else "nach Bauprogramm")
            if i == 15:
                bem = "Endwert: Buchwert je Aktie"
            zeilen.append(f"{jahr} & {bem} & {_z(zahlung, 2)} \\\\")
    _schreib("flow_anleger.tex", zeilen)

    # --- Interner Zinsfuss ------------------------------------------------
    zeilen = []
    for i, (name, preis) in enumerate(SZENARIEN):
        felder = []
        for h in HORIZONTE:
            z = irr(anlegerreihe(preis, h))
            felder.append(_z(z * 100, 1) if z is not None else "--")
            makro(f"IRR{SZENARIONAME[i]}{HORIZONTNAME[h]}",
                  z * 100 if z is not None else 0.0)
        zeilen.append(f"{name} & {_z(preis)} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_irr.tex", zeilen)

    # --- Einstiegsschwelle ------------------------------------------------
    zeilen = []
    for i, (name, preis) in enumerate(SZENARIEN):
        felder = []
        for h in HORIZONTE:
            s = schwellenkurs(preis, h)
            felder.append(_z(s, 2) if s is not None else "--")
            if s is not None:
                makro(f"Schwelle{SZENARIONAME[i]}{HORIZONTNAME[h]}", s)
        zeilen.append(f"{name} & {_z(preis)} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_schwelle.tex", zeilen)

    # --- Erforderliches Wachstum -----------------------------------------
    zeilen = []
    for i, (name, preis) in enumerate(SZENARIEN):
        felder = []
        for h in HORIZONTE:
            g = erforderliches_wachstum(preis, h)
            felder.append(_z(g * 100, 1) if g is not None else "nicht erreichbar")
            if g is not None:
                makro(f"Wachstum{SZENARIONAME[i]}{HORIZONTNAME[h]}", g * 100)
        zeilen.append(f"{name} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_wachstum.tex", zeilen)

    # --- Erforderlicher Endwert ------------------------------------------
    zeilen = []
    for i, (name, preis) in enumerate(SZENARIEN):
        felder = []
        for h in HORIZONTE:
            ew = erforderlicher_endwert(preis, h)
            felder.append(f"{_z(ew, 2)} & {_z(ew / ENDWERT_JE_AKTIE, 2)}")
            makro(f"Endwert{SZENARIONAME[i]}{HORIZONTNAME[h]}", ew)
            makro(f"EndwertVielfach{SZENARIONAME[i]}{HORIZONTNAME[h]}", ew / ENDWERT_JE_AKTIE)
        zeilen.append(f"{name} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_endwert.tex", zeilen)

    # --- Sensitivitaet gegen das Wachstumskapital -------------------------
    # Die Annahme, dass das Bauprogramm 2029 endet, entscheidet das Ergebnis
    # mit. Sie wird deshalb in beide Richtungen durchgerechnet: einmal ohne
    # Wachstumskapital ueberhaupt, einmal mit dauerhaft fortgefuehrtem.
    zeilen = []
    for beschriftung, faktor, ende in (
            ("kein Wachstumskapital", 0.0, BAUENDE),
            (f"bis {BAUENDE} (Basis)", 1.0, BAUENDE),
            ("dauerhaft fortgef\\\"uhrt", 1.0, 9999)):
        felder = []
        for h in HORIZONTE:
            z = irr(anlegerreihe(preis_basis, h, None, faktor, 0.0, ende))
            felder.append(_z(z * 100, 1) if z is not None else "--")
            if faktor == 0.0:
                makro(f"IRROhneBau{HORIZONTNAME[h]}", z * 100 if z is not None else 0.0)
            elif ende == 9999:
                makro(f"IRRDauerbau{HORIZONTNAME[h]}", z * 100 if z is not None else 0.0)
        zeilen.append(f"{beschriftung} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_sensitiv.tex", zeilen)

    # --- Sensitivitaet gegen den Endwert ----------------------------------
    # Der Endwert entscheidet das Ergebnis staerker als jede andere Annahme.
    # Er wird deshalb nicht nur erklaert, sondern gegen die einzige andere
    # BELEGTE Groesse gehalten, die als Endwert taugt: den vom Unternehmen
    # selbst veroeffentlichten Kapitalwert des Island-Gold-Distrikts je Aktie
    # (MD&A 2025, S. 7). Der ist eine Angabe des Unternehmens ueber sein
    # bestes Projekt bei einem Goldpreis von 4.500 $/oz und deshalb die
    # guenstigste belegbare Annahme - kein erfundenes Vielfaches.
    igd_je_aktie = W["IGDKapitalwert"] * 1000 / AKTIEN_MIO
    zeilen = []
    for i, (name, preis) in enumerate(SZENARIEN):
        felder = []
        for ew in (ENDWERT_JE_AKTIE, igd_je_aktie):
            s15 = schwellenkurs(preis, 15, 1.0, ew)
            felder.append(_z(s15, 2) if s15 is not None else "--")
            if ew != ENDWERT_JE_AKTIE:
                makro(f"SchwelleIGD{SZENARIONAME[i]}", s15)
        zeilen.append(f"{name} & " + " & ".join(felder) + "\\\\")
    _schreib("flow_endwertvariante.tex", zeilen)
    makro("SchwelleIGDBeste", schwellenkurs(SZENARIEN[2][1], 15, 1.0, igd_je_aktie))
    makro("AbstandIGDBeste",
          (W["KursSchlussXXVI"] / schwellenkurs(SZENARIEN[2][1], 15, 1.0, igd_je_aktie) - 1) * 100)

    # --- Einzelwerte fuer den Fliesstext ----------------------------------
    makro("Huerde", huerde())
    makro("HuerdeBerichtet", W["Eigenkapitalrendite"])
    makro("Einstiegskurs", W["KursSchlussXXVI"])
    makro("Endwert", ENDWERT_JE_AKTIE)
    makro("MengeSechs", MENGE_SECHS / 1000)
    makro("MengeAcht", MENGE_ACHT / 1000)
    makro("AiscSechs", AISC_SECHS)
    makro("AiscAcht", AISC_ACHT)
    makro("Wachstumskapital", WACHSTUMSKAPITAL)
    makro("Bauende", BAUENDE)
    makro("Steuerquote", STEUERQUOTE * 100)
    makro("Exploration", EXPLORATION)
    makro("VerteilbarSechs", verteilbar(preis_basis, 2026))
    makro("VerteilbarDreissig", verteilbar(preis_basis, 2030))
    # Abstand des heutigen Kurses zur guenstigsten Schwelle der Tabelle
    beste = schwellenkurs(SZENARIEN[2][1], 15)
    makro("SchwelleBeste", beste)
    makro("AbstandBeste", (W["KursSchlussXXVI"] / beste - 1) * 100)
    schlecht = schwellenkurs(SZENARIEN[BASIS][1], 15)
    makro("SchwelleBasis", schlecht)
    makro("AbstandBasis", (W["KursSchlussXXVI"] / schlecht - 1) * 100)
    # Der vom Unternehmen veroeffentlichte Kapitalwert des Island-Gold-
    # Distrikts je Aktie - der Massstab, an dem der erforderliche Endwert
    # gemessen wird, ohne dass ein Vielfaches erfunden werden muss.
    makro("IGDJeAktie", W["IGDKapitalwert"] * 1000 / AKTIEN_MIO)
    makro("Marktkapitalisierung", W["KursSchlussXXVI"] * AKTIEN_MIO / 1000)
    _schreib("flow_makros.tex", makros)

    print(f"[FLOW] {len(makros)} Makros, 7 Tabellen -> data/")


if __name__ == "__main__":
    main()
