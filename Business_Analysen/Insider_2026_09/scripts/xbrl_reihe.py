#!/usr/bin/env python3
"""
xbrl_reihe.py - Jahresreihen aus den XBRL-Daten der 10-K (SEC companyfacts).

Je Kennzahl und Geschaeftsjahresende gilt die JUENGSTE 10-K-Einreichung; jede weitere
10-K, die dasselbe Jahr druckt, ist ein Kettenglied. Abweichungen ueber 0,5 % werden
gemeldet (Neudarstellung oder Lesefehler), nie still ueberschrieben.
Zeitraumgroessen: nur Eintraege mit etwa zwoelf Monaten Dauer (335-380 Tage).
Aufruf: python3 scripts/xbrl_reihe.py GME CRM     Ausgabe: data/xbrl_<T>.json
"""
import json, os, sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTEN = {
    "umsatz": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "ergebnis": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "sbc": ["ShareBasedCompensation"],
    "rueckkauf": ["PaymentsForRepurchaseOfCommonStock"],
    "dividenden": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "eigenkapital": ["StockholdersEquity"],
    "kasse": ["CashAndCashEquivalentsAtCarryingValue"],
    "wertpapiere_kurz": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "schulden": ["LongTermDebt", "LongTermDebtNoncurrent", "ConvertibleNotesPayable"],
    "zinsertrag": ["InvestmentIncomeInterest", "InvestmentIncomeInterestAndDividend"],
}
ZEITRAUM = {"umsatz", "ergebnis", "cfo", "capex", "sbc", "rueckkauf", "dividenden", "zinsertrag"}


def tage(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def main(tick):
    for t in tick:
        f = json.load(open(os.path.join(ROOT, "refs", f"facts_{t}.json")))["facts"]
        g = f["us-gaap"]
        aus, brueche = {}, []
        for p, tags in POSTEN.items():
            serie = {}
            for tag in tags:
                if tag not in g:
                    continue
                for e in g[tag]["units"].get("USD", []):
                    if e.get("form") != "10-K":
                        continue
                    if p in ZEITRAUM and not (e.get("start") and 335 <= tage(e["start"], e["end"]) <= 380):
                        continue
                    alt = serie.get(e["end"])
                    if alt and abs(alt["wert"] - e["val"]) > 0.005 * max(abs(e["val"]), 1):
                        brueche.append(f"{p} {e['end']}: {alt['wert']} ({alt['akte']}) vs {e['val']} ({e['accn']})")
                    if not alt or e["filed"] >= alt["eingereicht"]:
                        serie[e["end"]] = {"wert": e["val"], "akte": e["accn"], "eingereicht": e["filed"], "tag": tag}
                # Tags werden je Periodenende zusammengefuehrt (Umsatz wechselt den Tag mit ASC 606);
                # je Periode gilt die juengste Einreichung, der Tag steht im Eintrag.
            aus[p] = serie
        # Juengstes 10-Q: Bilanzwerte zum Quartalsende, operativer Zufluss laufendes Jahr (YTD)
        q = {}
        for p, tags in POSTEN.items():
            for tag in tags:
                for e in g.get(tag, {}).get("units", {}).get("USD", []):
                    if e.get("form") != "10-Q":
                        continue
                    schl = (p, e.get("start"), e["end"])
                    if schl not in q or e["filed"] > q[schl]["eingereicht"]:
                        q[schl] = {"wert": e["val"], "akte": e["accn"], "eingereicht": e["filed"], "tag": tag}
        letzte_q = max((k[2] for k in q), default=None)
        aus["q"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in q.items() if k[2] >= (letzte_q or "")[:4] + "-00"}
        aktien = {}
        for e in f.get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", []):
            aktien[e["end"]] = {"wert": e["val"], "akte": e["accn"], "form": e["form"]}
        aus["aktien_dei"] = aktien
        json.dump({"posten": aus, "brueche": brueche}, open(os.path.join(ROOT, "data", f"xbrl_{t}.json"), "w"), indent=1)
        enden = sorted({k for p in ZEITRAUM for k in aus.get(p, {})})[-9:]
        print(f"\n== {t}")
        for p in POSTEN:
            print(f"{p:16s}" + "".join(f"{aus[p][e]['wert']/1e6:>10.0f}" if e in aus[p] else f"{'.':>10s}" for e in enden))
        print("Periodenende   " + "".join(f"{e:>10s}" for e in enden))
        print("Brueche:", len(brueche)); [print("  ", b) for b in brueche[:12]]
        letzte = sorted(aktien)[-1]; print("Aktien (dei):", letzte, aktien[letzte])


if __name__ == "__main__":
    main(sys.argv[1:])
