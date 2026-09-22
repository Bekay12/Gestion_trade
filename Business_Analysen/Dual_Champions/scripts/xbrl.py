#!/usr/bin/env python3
"""
xbrl.py - Jahreswerte aus den SEC-companyfacts (XBRL) der vier SEC-Emittenten.

Zweck: (1) zweite, unabhaengige Lesung jedes auf einer gedruckten Seite belegten Werts;
(2) Fuellung der Jahre, fuer die kein Bericht mit lesbarem Abschluss vorliegt (TTE vor
2020: die alten 20-F verweisen auf das Registration Document). XBRL traegt keine
gedruckte Seite; ein daraus gefuellter Wert ist Sekundaerquelle (URL + Abrufdatum) und
wird nur verwendet, wenn XBRL auf den Ueberlappungsjahren mit den Seitenwerten
uebereinstimmt (reihe.py prueft das).

Aufruf: python3 scripts/xbrl.py
Ausgabe: data/xbrl_reihen.json  {kuerzel: {posten: {jahr: wert_in_mio}}}
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Posten -> Liste von Begriffsgruppen; eine Gruppe wird summiert, die erste Gruppe mit
# Werten gewinnt je Jahr. Vorzeichen wie im Abschluss: Abfluesse negativ.
KARTE = {
    "tte": {
        "operativer_cf": [["ifrs-full:CashFlowsFromUsedInOperatingActivities"]],
        "investitionen": [["ifrs-full:AdditionsOtherThanThroughBusinessCombinationsPropertyPlantAndEquipment",
                           "ifrs-full:AdditionsOtherThanThroughBusinessCombinationsIntangibleAssetsOtherThanGoodwill"]],
        "dividenden": [["ifrs-full:DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities"]],
        "ergebnis": [["ifrs-full:ProfitLossAttributableToOwnersOfParent"]],
    },
    "oxy": {
        "operativer_cf": [["us-gaap:NetCashProvidedByUsedInOperatingActivities"]],
        "investitionen": [["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]],
        "ergebnis": [["us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"]],
    },
    "tnk": {
        "operativer_cf": [["us-gaap:NetCashProvidedByUsedInOperatingActivities"]],
        "investitionen": [["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]],
        "ergebnis": [["us-gaap:ProfitLoss"]],
    },
    "fro": {
        "operativer_cf": [["ifrs-full:CashFlowsFromUsedInOperatingActivities"],
                          ["us-gaap:NetCashProvidedByUsedInOperatingActivities"]],
        "investitionen": [["ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"]],
        "dividenden": [["ifrs-full:DividendsPaidClassifiedAsFinancingActivities"], ["us-gaap:PaymentsOfDividends"]],
        "ergebnis": [["ifrs-full:ProfitLoss"], ["us-gaap:NetIncomeLoss"]],
    },
}
# Konzepte, die als Abfluss gemeldet werden, aber positiv im XBRL stehen
ABFLUSS = {"investitionen", "dividenden"}


def jahreswerte(facts: dict, begriff: str) -> dict:
    ns, name = begriff.split(":")
    einheiten = facts.get(ns, {}).get(name, {}).get("units", {})
    aus = {}
    for einheit, xs in einheiten.items():
        for x in xs:
            if x.get("form") not in ("10-K", "20-F", "10-K/A", "20-F/A") or x.get("fp") != "FY":
                continue
            if "start" in x and x["start"][:4] != x["end"][:4] and x["end"][5:7] != "12":
                continue
            jahr = int(x["end"][:4])
            # juengste Einreichung gewinnt (angepasste Vorjahreswerte)
            if jahr not in aus or x["filed"] > aus[jahr][1]:
                aus[jahr] = (x["val"] / 1e6, x["filed"])
    return {j: v for j, (v, _) in aus.items()}


def main() -> None:
    aus = {}
    for k, karte in KARTE.items():
        facts = json.load(open(os.path.join(ROOT, "refs", f"{k}-companyfacts.json")))["facts"]
        aus[k] = {}
        for posten, gruppen in karte.items():
            werte = {}
            for gruppe in gruppen:
                teile = [jahreswerte(facts, b) for b in gruppe]
                for jahr in set().union(*teile):
                    if jahr in werte or not all(jahr in t for t in teile):
                        continue
                    v = sum(t[jahr] for t in teile)
                    werte[jahr] = -abs(v) if posten in ABFLUSS else v
            aus[k][posten] = {str(j): round(v, 3) for j, v in sorted(werte.items())}
        print(f"[XBRL] {k}: " + ", ".join(f"{p} {min(v)}-{max(v)}" for p, v in aus[k].items() if v))
    json.dump(aus, open(os.path.join(ROOT, "data", "xbrl_reihen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
