#!/usr/bin/env python3
"""
form4.py - Directors' Dealings eines US-Emittenten aus den SEC-Formularen 4 der
letzten zwoelf Monate (report-template.md 1.2).

Gezaehlt werden nur Geschaefte am offenen Markt: Code P (Kauf) und S (Verkauf).
Optionsausuebung (M), Steuereinbehalt (F), Zuteilung (A) und Schenkung (G) sind
Verguetungs- oder Verwaltungsvorgaenge, keine Einschaetzung des Kurses, und
werden getrennt ausgewiesen statt mitgezaehlt. Eine Veraeusserung unter einem
vorab festgelegten Plan (Rule 10b5-1) wird markiert.

Beleg: je Formular die URL und das Abrufdatum (Sekundaerquelle, keine gedruckte Seite).

Aufruf: python3 scripts/form4.py <kuerzel> <CIK>
Ausgabe: data/form4_<kuerzel>.json
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Berkay Kaya berkam-research@gmail.com"}


def hole(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _t(node, pfad: str) -> str:
    x = node.find(pfad)
    return (x.text or "").strip() if x is not None and x.text else ""


def main(kuerzel: str, cik: int) -> None:
    heute = dt.date.today()
    ab = heute - dt.timedelta(days=365)
    rec = json.loads(hole(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]["recent"]
    geschaefte, sonstige = [], {}
    for i, form in enumerate(rec["form"]):
        if form != "4" or dt.date.fromisoformat(rec["filingDate"][i]) < ab:
            continue
        acc = rec["accessionNumber"][i].replace("-", "")
        doc = re.sub(r"^xslF345X\d+/", "", rec["primaryDocument"][i])  # rohes XML statt HTML-Ansicht
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
        try:
            x = ET.fromstring(hole(url))
        except Exception as exc:
            print(f"[FORM4] {url}: {exc}")
            continue
        time.sleep(0.15)
        person = _t(x, "reportingOwner/reportingOwnerId/rptOwnerName")
        rolle = _t(x, "reportingOwner/reportingOwnerRelationship/officerTitle") or (
            "Director" if _t(x, "reportingOwner/reportingOwnerRelationship/isDirector") in ("1", "true")
            else "10%-Eigner" if _t(x, "reportingOwner/reportingOwnerRelationship/isTenPercentOwner") in ("1", "true")
            else "")
        plan = "10b5-1" in ET.tostring(x, encoding="unicode")
        for tx in x.findall("nonDerivativeTable/nonDerivativeTransaction"):
            code = _t(tx, "transactionCoding/transactionCode")
            if code not in ("P", "S"):
                sonstige[code] = sonstige.get(code, 0) + 1
                continue
            stueck = float(_t(tx, "transactionAmounts/transactionShares/value") or 0)
            preis = float(_t(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
            geschaefte.append({"datum": _t(tx, "transactionDate/value"), "person": person, "rolle": rolle,
                               "art": "Kauf" if code == "P" else "Verkauf", "stueck": stueck,
                               "preis": preis, "volumen": round(stueck * preis, 2),
                               "plan_10b5_1": plan, "url": url})
    aus = {"emittent": kuerzel, "zeitraum": [ab.isoformat(), heute.isoformat()],
           "abgerufen": heute.isoformat(), "geschaefte": sorted(geschaefte, key=lambda g: g["datum"]),
           "sonstige_codes": sonstige}
    json.dump(aus, open(os.path.join(ROOT, "data", f"form4_{kuerzel}.json"), "w"), indent=1,
              ensure_ascii=False)
    kaeufe = [g for g in geschaefte if g["art"] == "Kauf"]
    verk = [g for g in geschaefte if g["art"] == "Verkauf"]
    print(f"[FORM4] {kuerzel}: {len(kaeufe)} Kaeufe von {len({g['person'] for g in kaeufe})} Personen, "
          f"{len(verk)} Verkaeufe von {len({g['person'] for g in verk})} Personen; sonstige {sonstige}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
