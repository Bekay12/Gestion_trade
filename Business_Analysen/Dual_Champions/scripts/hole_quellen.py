#!/usr/bin/env python3
"""
hole_quellen.py - beschafft die Primaerquellen der vier SEC-Emittenten.

Je Emittent: die zwei juengsten Jahresberichte (10-K bzw. 20-F) als HTML und
die XBRL-"companyfacts" (Mehrjahresreihen; Beleg per URL und Abrufdatum, keine
gedruckte Seite). GEA und BESI reichen nicht bei der SEC ein; ihre
Geschaeftsberichte kommen als PDF von der Investor-Relations-Seite
(hole_pdf.py).

Die SEC verlangt einen User-Agent mit Kontaktadresse; derselbe wie im Projekt
Pharma_AZN_RDY.

Aufruf: python3 scripts/hole_quellen.py
Ausgabe: refs/<kuerzel>-<form>-<jahr>.htm, refs/<kuerzel>-companyfacts.json,
         refs/_index.json (Abrufdatum, URL, Einreichungsdatum je Datei)
"""
import datetime as dt
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")
UA = {"User-Agent": "Berkay Kaya berkam-research@gmail.com"}

# Kuerzel -> (CIK, Formular des Jahresberichts)
EMITTENTEN = {
    "oxy": (797468, "10-K"),
    "tte": (879764, "20-F"),
    "tnk": (1419945, "20-F"),
    "fro": (913290, "20-F"),
}
# Berichtsjahre: jede Kapitalflussrechnung druckt drei Jahre. 2016, 2019, 2022 und
# 2025 decken 2014-2025 ab; 2024 liefert die zweite Lesung fuer die Kettenpruefung.
BERICHTSJAHRE = {"2016", "2019", "2022", "2024", "2025"}


def hole(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    os.makedirs(REFS, exist_ok=True)
    heute = dt.date.today().isoformat()
    index = {}
    for kuerzel, (cik, form) in EMITTENTEN.items():
        sub = json.loads(hole(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
        rec = {k: list(v) for k, v in sub["filings"]["recent"].items()}
        # Aeltere Einreichungen stehen in Nachlaufdateien ("files"), nicht in "recent".
        for datei in sub["filings"].get("files", []):
            alt = json.loads(hole(f"https://data.sec.gov/submissions/{datei['name']}"))
            for k in rec:
                rec[k] += alt.get(k, [])
            time.sleep(0.2)
        treffer, gesehen = [], set()
        for i, f in enumerate(rec["form"]):
            jahr = rec["reportDate"][i][:4]
            if f == form and jahr in BERICHTSJAHRE and jahr not in gesehen:
                treffer.append(i)
                gesehen.add(jahr)
        for i in treffer:
            acc = rec["accessionNumber"][i].replace("-", "")
            doc = rec["primaryDocument"][i]
            jahr = rec["reportDate"][i][:4]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
            ziel = os.path.join(REFS, f"{kuerzel}-{form.lower()}-{jahr}.htm")
            if not os.path.exists(ziel):
                open(ziel, "wb").write(hole(url))
                time.sleep(0.5)  # SEC: hoechstens 10 Anfragen je Sekunde
            index[os.path.basename(ziel)] = {"url": url, "eingereicht": rec["filingDate"][i],
                                             "berichtsdatum": rec["reportDate"][i], "abgerufen": heute}
            print(f"[SEC] {kuerzel} {form} {jahr} ({rec['filingDate'][i]})")
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        ziel = os.path.join(REFS, f"{kuerzel}-companyfacts.json")
        open(ziel, "wb").write(hole(url))
        index[os.path.basename(ziel)] = {"url": url, "abgerufen": heute}
        print(f"[XBRL] {kuerzel} companyfacts")
        time.sleep(0.5)
    json.dump(index, open(os.path.join(REFS, "_index.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
