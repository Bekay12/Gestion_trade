#!/usr/bin/env python3
"""
reihe.py - baut die Mehrjahresreihen je Emittent aus data/extrakt/*.json und prueft
sie ueber Kreuz (Kettenpruefung).

Jede Kapitalflussrechnung druckt zwei oder drei Jahre; dasselbe Jahr steht deshalb in
zwei Berichten. Weichen die Lesungen voneinander ab, ist eine Spalte verrutscht oder
der spaetere Bericht hat angepasst (restated) - beides wird gemeldet, nie gemittelt.
Massgeblich ist die JUENGSTE Lesung (sie traegt Anpassungen), belegt mit ihrer Seite.

Einheit je Seite aus dem Kopf der Abschlussseite ("in thousands", "in millions",
"EUR thousand"); alle Betraege werden auf Millionen umgerechnet, Aktienzahlen auf
Millionen Stueck, Ergebnis je Aktie bleibt je Aktie.

Investitionen: nur Sachanlagen, Schiffe, Oel- und Gasvermoegen, immaterielle Werte.
Finanzanlagen (Einlagen, Wertpapiere) und Unternehmenserwerbe gehoeren nicht in den
freien Mittelzufluss und werden ausgesondert.

Aufruf: python3 scripts/reihe.py
Ausgabe: data/reihen.json; Rueckgabewert 1 bei einem Kettenbruch.
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                      # noqa: E402
from zeile import aus_spalte, aus_zeile, entfalte, lies_werte  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRAKT = os.path.join(ROOT, "data", "extrakt")
# Berichte ohne lesbaren Abschluss (verweisen auf das Registration Document): ihre
# Lesungen stammen von Zusammenfassungsseiten und werden nicht verwendet.
AUSGESCHLOSSEN = {"tte-20-f-2016", "tte-20-f-2019"}
XBRL_QUELLE = "SEC XBRL companyfacts, abgerufen 19.09.2026"

KEIN_CAPEX = re.compile(r"(?i)deposit|securit|financial|short[- ]term|marketable|associat|joint venture|"
                        r"equity[- ]accounted|business|subsidiar|compan(y|ies)|acquisition of|investments? in "
                        r"(?!property|vessel)|proceeds|sale|disposal")
# Dividenden an Minderheiten und Vorzugsaktionaere sind kein Ausschuettungsstrom an die
# Stammaktionaere (TTE 2024/2025: "Dividends paid to non-controlling interests").
KEINE_STAMMDIVIDENDE = re.compile(r"(?i)non-controlling|minority|preferred|hybrid")
# Zweideutige Etiketten: dasselbe Etikett steht mehrfach auf der Seite. Gelesen wird die
# Fundstelle NACH dem Anker. TTE: "– Parent company shareholders" steht unter der
# Aktienausgabe (492) und unter "Dividends paid:"; nur die zweite ist die Dividende.
KONTEXT = {
    "tte": {
        "dividenden": (r"Dividends paid:", "Parent company shareholders"),
        "rueckkauf": (r"Issuance\s*/\s*\(repurchase\) of shares|repurchase\) of shares", "Treasury shares"),
    },
}
# Zusatzzeilen, die das Modell nicht nennt, die aber zum Posten gehoeren. TNK weist den
# Kauf gebrauchter Schiffe getrennt aus ("Vessel acquisitions"); ohne sie fehlte 2025
# der Grossteil der Flotteninvestition (XBRL-Abgleich: 190,3 statt 3,0 Mio. USD).
# Bevorzugtes Etikett je Emittent, wo zwei Zeilen dasselbe benennen: GEA druckt den
# operativen Mittelzufluss gesamt UND fortgefuehrt; 2020 las das Modell die Gesamtzeile
# (821,9), 2021 die fortgefuehrte (717,8) - eine Reihe aus zwei Abgrenzungen.
BEVORZUGT = {"gea": {"operativer_cf": "Cash flow from operating activities of continued operations"}}
ZUSATZ_ZEILEN = {"tnk": {"investitionen": ["Vessel acquisitions"]},
                 # GEA: Leasingtilgung steht in der rechten Spalte der Kapitalflussrechnung
                 "gea": {"leasing": ["Payments from lease liabilities", "Repayment of lease liabilities",
                                     "Payments for lease liabilities"]},
                 # OXY: fortgefuehrte Taetigkeit getrennt ausgewiesen (OxyChem 2026 verkauft)
                 "oxy": {"operativer_cf_fortgef": ["Operating cash flow from continuing operations"],
                         "invest_cf_aufgegeben": ["Investing cash flow from discontinued operations"]}}
# Erklaerte Abweichungen zwischen zwei Lesungen desselben Jahres: der spaetere Bericht hat
# angepasst. Massgeblich bleibt die juengste Lesung; die Liste haelt fest, WARUM ein Bruch
# kein Lesefehler ist. Jeder neue, nicht erklaerte Bruch laesst den Bau scheitern.
ERKLAERT = {
    ("oxy", "umsatz"): "10-K 2025 weist OxyChem als aufgegebene Taetigkeit aus (Verkauf 02.01.2026)",
    ("oxy", "investitionen"): "10-K 2025: Investitionen der fortgefuehrten Taetigkeit angepasst (OxyChem)",
    ("oxy", "eps_verwaessert"): "10-K 2025: Ergebnis je Aktie ohne aufgegebene Taetigkeit angepasst",
    ("oxy", "eigenkapital"): "10-K 2025 liest 'Total equity' (mit Minderheiten); ROH nutzt Stammkapital",
    ("tnk", "operativer_cf"): "20-F 2024 passt 2022 an (Darstellung Reisegeschaeft)",
    ("tnk", "umsatz"): "20-F 2024 passt 2022 an (Bruttoausweis Reiseerloese)",
    ("tnk", "ergebnis"): "20-F 2024 passt 2022 an",
    ("tte", "umsatz"): "20-F 2019 weist Umsatz 2015/2016 angepasst aus (ohne Verbrauchsteuern)",
    ("oxy", "operativer_cf_fortgef"): "10-K 2025 ordnet OxyChem 2023/2024 der aufgegebenen Taetigkeit zu",
    ("tnk", "investitionen"): "Zeile 'Vessel acquisitions' nur im 20-F 2025 gelesen; massgeblich die juengste Lesung",
    ("fro", "eigenkapital"): "Einheitenkopf im 20-F 2024 nicht erkannt; massgeblich 20-F 2025 S. F-6",
}
BETRAG = ("operativer_cf", "operativer_cf_fortgef", "invest_cf_aufgegeben", "investitionen", "dividenden", "rueckkauf", "leasing", "umsatz",
          "ergebnis", "vorzugsdividende", "eigenkapital", "vorzugsaktien")
STUECK = ("aktien_verwaessert", "aktien_ausstehend")


_NOTE = re.compile(r"(?i)^\(?\s*(see\s+)?notes?\s*[\d.,\sand&-]*\)?$")
_BETRAG = re.compile(r"(\(\s*)?(-?\d[\d,]*(?:\.\d+)?)\s*(\))?")
_ZEICHEN = str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-", "\u200b": "", "\u00a0": " "})


def _betraege(roh: str) -> list:
    return [(-1 if (a or c) else 1) * float(b.replace(",", "")) for a, b, c in _BETRAG.findall(roh)
            if not re.fullmatch(r"(19|20)\d\d", b)]


def hat_aenderungsspalte(text: str) -> bool:
    """Seitenkopf mit Veraenderungsspalte ("Change in %", GEA 2025): letzte Zahl je Zeile ist keine Jahresspalte."""
    return bool(re.search(r"(?i)change\s+in\s*%|ver[aä]nderung\s+in\s*%|in\s*%\s*$", text[:2500], re.M))


def lies_nach(text: str, etikett: str, n: int):
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die n Jahreswerte eines Etiketts, in beiden Satzformen der Quellen:
        PDF (-layout): Etikett und Werte in EINER Zeile -> die letzten n Betraege
        hinter dem Etikett. HTML (SEC): jede Zelle eine Zeile, die Klammer eines
        negativen Betrags auf eigenen Zeilen ("(", "8,121", ")") -> gesammelt bis
        zur naechsten Textzeile; Notenverweise ("(Note 9)") werden uebersprungen,
        Jahreszahlen nie als Betrag gelesen.

    Inputs:
        text (str): Seitentext; etikett (str): Zeilenbezeichnung; n (int): Jahre

    Outputs:
        werte (list | None): n Betraege, Klammer = negativ; None wenn nicht lesbar
    --------------------------------------------------------------------------
    """
    ziel = etikett.translate(_ZEICHEN).lower().strip(" -")
    zeilen = text.splitlines()
    for i, z in enumerate(zeilen):
        zn = z.translate(_ZEICHEN)
        pos = zn.lower().find(ziel)
        if pos < 0:
            continue
        # Zweispaltiger Satz (GEA 2020: rechts steht "Cash and cash equivalents ... 821,852"
        # in derselben Textzeile): nur bis zum naechsten Wort lesen, sonst greift die Zeile
        # in die Nachbarspalte.
        rest = zn[pos + len(ziel):]
        wort = re.search(r"[A-Za-z]{2,}", rest)
        gleiche = _betraege(rest[:wort.start()] if wort else rest)
        # Veraenderungsspalte: bei n+1 Zahlen hinter dem Etikett ist die letzte die
        # Veraenderung in % (GEA 2025: "Revenue 7.1 5,495,356 5,422,129 1.4").
        if len(gleiche) >= n + 1 and hat_aenderungsspalte(text):
            return gleiche[-n - 1:-1]
        if len(gleiche) >= n:
            return gleiche[-n:]
        teile = []
        for w in zeilen[i + 1:]:
            w = w.translate(_ZEICHEN).strip()
            if not w:
                continue
            if re.search(r"[A-Za-z]", w) and not _NOTE.match(w):
                break
            if not _NOTE.match(w):
                # ein Gedankenstrich als ganze Zelle ist eine Null ("—" bei TNK in Jahren ohne Kauf)
                teile.append("0" if w in ("—", "-", "–") else w)
        betraege = _betraege(" ".join(teile))
        if len(betraege) >= n:
            # die LETZTEN n: davor kann die Notenspalte stehen ("12" bei FRO), die ohne
            # Buchstaben nicht als Verweis erkennbar ist
            return betraege[-n:]
        # sonst war es eine Zwischenueberschrift ohne Werte ("CASH FLOW FROM OPERATING
        # ACTIVITIES" ueber der Summenzeile): naechste Fundstelle desselben Etiketts pruefen
    return None


def einheit(text: str) -> float:
    """
    Faktor auf Millionen aus der Einheitsangabe der Abschlussseite. Gesucht wird auf der
    ganzen Seite, weil BESI 2021 die Angabe erst unter dem Tabellenkopf setzt; die erste
    Fundstelle entscheidet, damit ein Fussnotentext "in millions" weiter unten nicht zaehlt.
    """
    low = text.lower()
    tausend = re.search(r"thousand|\(€ ?000\)|eur ?000|usd ?000|\$ ?000|in tausend|teur\b|k€", low)
    million = re.search(r"in millions|millions of|\(€? ?m(illion)?\)|eur m\b|usd m\b|mio\.|in million", low)
    if tausend and (not million or tausend.start() < million.start()):
        return 1e-3
    if re.search(r"in billions", low):
        return 1e3
    return 1.0


def aktien_faktor(text: str, werte: list) -> float:
    """Aktienzahlen auf Millionen: Kopf sagt Tausend/Millionen, sonst Groessenordnung."""
    kopf = text[:4000].lower()
    if re.search(r"shares in millions|in millions.{0,40}shares|millions of shares", kopf):
        return 1.0
    if re.search(r"shares in thousands|in thousands.{0,60}shares|thousands of shares", kopf):
        return 1e-3
    groesse = max(abs(w) for w in werte if w is not None)
    return 1e-6 if groesse > 1e6 else 1e-3 if groesse > 1e3 else 1.0


def lesungen() -> dict:
    """{kuerzel: {posten: {jahr: [(wert_mio, bericht, seite, etikett)]}}}."""
    aus = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for datei in sorted(os.listdir(EXTRAKT)):
        d = json.load(open(os.path.join(EXTRAKT, datei)))
        kuerzel = datei.split("-")[0]
        bericht = datei.replace(".json", "")
        if bericht in AUSGESCHLOSSEN:
            continue
        texte = seiten(os.path.join(ROOT, "refs", d["datei"]))
        for art, a in d["arten"].items():
            if art == "cf" and kuerzel in KONTEXT:
                cf_text = texte.get(a["seite"], "")
                for posten, (anker, etikett) in KONTEXT[kuerzel].items():
                    m = re.search(anker, cf_text)
                    if not m:
                        a["posten"].pop(posten, None)
                        continue
                    rest = cf_text[m.end():]
                    werte = (lies_nach(rest, etikett, len(a["jahre"]))
                             or aus_zeile(rest, etikett, len(a["jahre"]), 0, True)
                             or aus_spalte(rest, etikett, len(a["jahre"]), 0, True)
                             or aus_spalte(entfalte(rest), etikett, len(a["jahre"]), 0, True))
                    if werte and len([w for w in werte if w is not None]) == len(a["jahre"]):
                        a["posten"][posten] = [{"etikett": f"{anker.split('|')[0]} ... {etikett}",
                                                "werte": werte, "seite": a["seite"], "kontext": True}]
                    else:
                        a["posten"].pop(posten, None)
            if art == "cf":
                for posten, etiketten in ZUSATZ_ZEILEN.get(kuerzel, {}).items():
                    if posten == "investitionen" or posten in a["posten"]:
                        continue
                    for etikett in etiketten:
                        w = lies_nach(texte.get(a["seite"], ""), etikett, len(a["jahre"]))
                        if w:
                            a["posten"][posten] = [{"etikett": etikett, "werte": w, "seite": a["seite"],
                                                    "kontext": True}]
                            break
            for posten, etikett in BEVORZUGT.get(kuerzel, {}).items():
                if art == "cf" and posten in a["posten"]:
                    w = lies_nach(texte.get(a["seite"], ""), etikett, len(a["jahre"]))
                    if w:
                        a["posten"][posten] = [{"etikett": etikett, "werte": w, "seite": a["seite"],
                                                "kontext": True}]
            for posten, eintraege in a["posten"].items():
                # Werte von hinten neu lesen: die letzten n Zahlen der Zeile sind die Jahres-
                # spalten; Notennummern und Jahreszahlen stehen davor (TTE 2016: "2016" als Wert).
                pfad = os.path.join(ROOT, "refs", d["datei"])
                for e in eintraege:
                    if e.get("kontext"):
                        continue
                    neu = lies_nach(texte.get(e["seite"], ""), e["etikett"], len(a["jahre"])) or \
                        lies_werte(pfad, e["seite"], e["etikett"], n=len(a["jahre"]), von_hinten=True)
                    if neu and len([w for w in neu if w is not None]) == len(a["jahre"]):
                        e["werte"] = neu
                if posten == "dividenden":
                    # "on common and preferred stock" (OXY) bleibt: rechnung.py zieht die
                    # Vorzugsdividende der GuV ab. Nur reine Minderheiten-/Vorzugszeilen fallen weg.
                    eintraege = [e for e in eintraege if not KEINE_STAMMDIVIDENDE.search(e["etikett"])
                                 or re.search(r"(?i)common|ordinary", e["etikett"])]
                    if not eintraege:
                        continue
                if posten == "investitionen":
                    for zusatz in ZUSATZ_ZEILEN.get(kuerzel, {}).get(posten, []):
                        if any(zusatz.lower() == e["etikett"].lower() for e in eintraege):
                            continue
                        w = lies_nach(texte.get(a["seite"], ""), zusatz, len(a["jahre"]))
                        if w:
                            eintraege = eintraege + [{"etikett": zusatz, "werte": w, "seite": a["seite"]}]
                    eintraege = [e for e in eintraege if not KEIN_CAPEX.search(e["etikett"])]
                    if not eintraege:
                        continue
                    # mehrere Investitionszeilen je Jahr aufsummieren (Seite der ersten)
                    summe = [sum(e["werte"][i] for e in eintraege) for i in range(len(a["jahre"]))]
                    eintraege = [{"etikett": " + ".join(e["etikett"] for e in eintraege),
                                  "werte": summe, "seite": eintraege[0]["seite"]}]
                for e in eintraege[:1]:
                    text = texte.get(e["seite"], "")
                    if posten in BETRAG:
                        f = einheit(text)
                    elif posten in STUECK:
                        f = aktien_faktor(text, e["werte"])
                    else:
                        f = 1.0
                    for jahr, w in zip(a["jahre"], e["werte"]):
                        if w is not None:
                            aus[kuerzel][posten][int(jahr)].append(
                                (round(w * f, 6), bericht, e["seite"], e["etikett"], w))
    return aus


def main() -> int:
    alle = lesungen()
    reihen, brueche = {}, []
    for k, posten in alle.items():
        reihen[k] = {}
        for p, jahre in posten.items():
            reihen[k][p] = {}
            for jahr, les in sorted(jahre.items()):
                les = sorted(les, key=lambda x: x[1], reverse=True)   # juengster Bericht zuerst
                wert, bericht, seite, etikett, roh = les[0]
                abweichend = [x for x in les[1:] if abs(abs(x[0]) - abs(wert)) > max(0.5e-3 * abs(wert), 0.01)]
                if abweichend:
                    brueche.append((k, p, jahr, les))
                reihen[k][p][jahr] = {"wert": wert, "roh": roh, "bericht": bericht, "seite": seite,
                                      "etikett": etikett, "lesungen": len(les),
                                      "abweichend": [(x[0], x[1], x[2]) for x in abweichend],
                                      "summe": " + " in etikett}
    xbrl = json.load(open(os.path.join(ROOT, "data", "xbrl_reihen.json")))
    abgleich = []
    for k, posten in xbrl.items():
        for p, werte in posten.items():
            seite = reihen.setdefault(k, {}).setdefault(p, {})
            gemeinsam = [(j, seite[j]["wert"], w) for j, w in ((int(j), w) for j, w in werte.items()) if j in seite]
            abw = [(j, a, b) for j, a, b in gemeinsam if abs(a - b) > max(0.005 * abs(b), 0.5)]
            for j, a, b in gemeinsam:
                seite[j]["xbrl"] = b
            abgleich.append((k, p, len(gemeinsam), abw))
            # Fuellen nur bei belastbarem Abgleich: >= 3 gemeinsame Jahre, keine Abweichung
            if len(gemeinsam) >= 3 and not abw:
                for j, w in ((int(j), w) for j, w in werte.items()):
                    if j not in seite:
                        seite[j] = {"wert": w, "bericht": XBRL_QUELLE, "seite": None,
                                    "etikett": "XBRL", "lesungen": 1, "abweichend": []}
    for k, p, n, abw in abgleich:
        status = "ok" if not abw else "ABWEICHUNG " + "; ".join(f"{j}: Seite {a} / XBRL {b}" for j, a, b in abw)
        print(f"[XBRL-ABGLEICH] {k} {p}: {n} gemeinsame Jahre, {status}")
    json.dump(reihen, open(os.path.join(ROOT, "data", "reihen.json"), "w"), indent=1, ensure_ascii=False)
    for k in sorted(reihen):
        abdeckung = {p: f"{min(v)}-{max(v)}" for p, v in reihen[k].items() if v}
        doppelt = sum(1 for p in reihen[k].values() for v in p.values() if v["lesungen"] > 1)
        print(f"[REIHE] {k}: {abdeckung} | {doppelt} Jahre doppelt gelesen")
    offen = []
    for k, p, jahr, les in brueche:
        art = "ERKLAERT" if (k, p) in ERKLAERT else "BRUCH"
        if art == "BRUCH":
            offen.append((k, p, jahr))
        print(f"[{art}] {k} {p} {jahr}: " + "; ".join(f"{w} ({b} S.{s})" for w, b, s, _, _ in les))
    return 1 if offen else 0


if __name__ == "__main__":
    sys.exit(main())
