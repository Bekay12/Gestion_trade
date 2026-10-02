#!/usr/bin/env python3
"""
form4.py - liest alle Form-4-Meldungen der sieben Titel aus dem Boerse-Online-Beitrag
(28.09.2026) direkt von EDGAR und legt jede Transaktion mit Code, Stueckzahl, Kurs und
10b5-1-Kennzeichen ab. Primaerquelle statt Zeitschriftenbild: das Bild sagt nicht, ob ein
"Kauf" ein Kauf am Markt (Code P) oder eine Zuteilung (A) bzw. Optionsausuebung (M) ist.

Aufruf: python3 scripts/form4.py [--seit 2026-09-14]
Ausgabe: data/form4.json
"""
import argparse, json, os, re, subprocess, time
import xml.etree.ElementTree as ET

UA = "Gestion_trade research nomcripte@gmail.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKER = ["GME", "CRM", "NVDA", "ORCL", "CRWD", "PANW", "CRSP"]
import sys as _s
if "--nur" in _s.argv: TICKER = _s.argv[_s.argv.index("--nur") + 1].split(","); del _s.argv[_s.argv.index("--nur"):_s.argv.index("--nur") + 2]


def hole(url: str) -> str:
    time.sleep(0.15)                      # SEC: hoechstens 10 Anfragen je Sekunde
    return subprocess.run(["curl", "-s", "-A", UA, url], capture_output=True, text=True).stdout


def txt(el, pfad):
    x = el.find(pfad)
    return x.text.strip() if x is not None and x.text else None


def lies_form4(xml: str) -> dict:
    r = ET.fromstring(xml)
    owner = r.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship")
    rolle = [k for k in ("isDirector", "isOfficer", "isTenPercentOwner") if txt(rel, k) in ("1", "true")]
    titel = txt(rel, "officerTitle")
    fn = {f.get("id"): (f.text or "").strip() for f in r.iter("footnote")}
    plan = txt(r, "aff10b5One") in ("1", "true")
    aus = []
    for t in r.iter("nonDerivativeTransaction"):
        fids = [x.get("id") for x in t.iter() if x.tag == "footnoteId"]
        aus.append({
            "wertpapier": txt(t, "securityTitle/value"),
            "datum": txt(t, "transactionDate/value"),
            "code": txt(t, "transactionCoding/transactionCode"),
            "stueck": float(txt(t, "transactionAmounts/transactionShares/value") or 0),
            "preis": float(txt(t, "transactionAmounts/transactionPricePerShare/value") or 0) if txt(t, "transactionAmounts/transactionPricePerShare/value") else None,
            "richtung": txt(t, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "bestand_danach": float(txt(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value") or 0),
            "direkt": txt(t, "ownershipNature/directOrIndirectOwnership/value"),
            "fussnoten": [fn.get(i, "") for i in fids],
        })
    return {"person": txt(owner, "reportingOwnerId/rptOwnerName"), "rolle": rolle, "titel": titel,
            "plan_10b5_1": plan, "fussnoten_alle": fn, "transaktionen": aus}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seit", default="2026-09-14"); a = ap.parse_args()
    tick = json.loads(hole("https://www.sec.gov/files/company_tickers.json"))
    cik = {v["ticker"]: v["cik_str"] for v in tick.values()}
    alle = {}
    for t in TICKER:
        rec = json.loads(hole(f"https://data.sec.gov/submissions/CIK{cik[t]:010d}.json"))["filings"]["recent"]
        meldungen = []
        for form, dt, acc, doc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"]):
            if form != "4" or dt < a.seit:
                continue
            ordner = f"https://www.sec.gov/Archives/edgar/data/{cik[t]}/{acc.replace('-', '')}/"
            xmlname = re.sub(r"^xslF345X\d+/", "", doc)
            try:
                m = lies_form4(hole(ordner + xmlname))
            except Exception as e:
                m = {"fehler": str(e)}
            m.update({"eingereicht": dt, "url": ordner + doc})
            meldungen.append(m)
        alle[t] = {"cik": cik[t], "meldungen": meldungen}
        print(f"{t}: {len(meldungen)} Form 4 seit {a.seit}")
    json.dump(alle, open(os.path.join(ROOT, "data", f"form4_{a.seit}.json"), "w"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
