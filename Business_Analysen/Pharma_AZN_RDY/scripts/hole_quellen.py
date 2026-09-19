#!/usr/bin/env python3
"""
hole_quellen.py - laedt die Einreichungen beider Emittenten von EDGAR nach refs/.

Die SEC verlangt einen echten User-Agent mit Kontaktadresse; ohne ihn wird
gedrosselt oder blockiert. Wir laden die Einreichungsliste, waehlen je Formtyp
die gewuenschte Zahl der juengsten Dokumente und legen sie unter refs/ ab.
"""
import json, os, re, sys, time, urllib.request

UA = {"User-Agent": "Berkay Kaya berkam-research@gmail.com"}
EMITTENTEN = {"AZN": "0000901832", "RDY": "0001135951"}


def _hole(url: str, ziel: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for versuch in range(4):
        try:
            roh = urllib.request.urlopen(req, timeout=120).read()
            break
        except Exception as fehler:                      # Netzfehler: erneut
            if versuch == 3:
                raise
            print(f"  ... erneuter Versuch ({fehler})", file=sys.stderr)
            time.sleep(3 * (versuch + 1))
    if ziel:
        with open(ziel, "wb") as f:
            f.write(roh)
    time.sleep(0.4)                                      # 10 Anfragen/s Limit
    return roh


def liste(cik: str) -> list[dict]:
    """Alle Einreichungen, auch die aelteren aus den Zusatzdateien."""
    d = json.loads(_hole(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    r = d["filings"]["recent"]
    eintraege = [dict(zip(r.keys(), w)) for w in zip(*r.values())]
    for zusatz in d["filings"].get("files", []):
        z = json.loads(_hole(f"https://data.sec.gov/submissions/{zusatz['name']}"))
        eintraege += [dict(zip(z.keys(), w)) for w in zip(*z.values())]
    return eintraege


if __name__ == "__main__":
    os.makedirs("refs", exist_ok=True)
    for kuerzel, cik in EMITTENTEN.items():
        eintraege = liste(cik)
        with open(f"refs/_index_{kuerzel}.json", "w") as f:
            json.dump(eintraege, f)
        formen = {}
        for e in eintraege:
            formen.setdefault(e["form"], 0)
            formen[e["form"]] += 1
        print(kuerzel, sorted(formen.items(), key=lambda x: -x[1])[:12])


def dokument(cik: str, accession: str, datei: str, ziel: str) -> None:
    """Laedt das Hauptdokument einer Einreichung nach refs/<ziel>."""
    nr = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nr}/{datei}"
    if os.path.exists(ziel) and os.path.getsize(ziel) > 0:
        print(f"  vorhanden: {ziel}")
        return
    _hole(url, ziel)
    print(f"  {ziel}  {os.path.getsize(ziel)/1e6:.1f} MB")
