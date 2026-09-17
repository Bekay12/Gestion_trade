#!/usr/bin/env python3
"""
hole_quellen.py - beschafft die Primaerquellen von Alamos Gold Inc. aus EDGAR.

Alamos ist ein auslaendischer Emittent (Foreign Private Issuer) und reicht
statt 10-K/10-Q die Form 40-F (Jahr) und Form 6-K (Quartal, Rundschreiben)
ein. Die inhaltlichen Dokumente liegen darin als Anhaenge EX-99.x:

    EX-99.1  Annual Information Form (AIF) - Organe, Reserven, Minenlaufzeit
    EX-99.2  Management's Discussion and Analysis (MD&A)
    EX-99.3  Konzernabschluss

Das Skript laedt die Anhaenge aller angeforderten Geschaeftsjahre nach
refs/ und legt dabei den EDGAR-Dateinamen offen, damit das
Quellenverzeichnis das Aktenzeichen nennen kann.

Aufruf:
    export SEC_USER_AGENT='Vorname Nachname mail@example.com'
    python3 scripts/hole_quellen.py --liste          # nur anzeigen
    python3 scripts/hole_quellen.py --jahre 2016-2025
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

CIK = "0001178819"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")
BASIS = "https://www.sec.gov/Archives/edgar/data/1178819"


def _ua() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit(
            "[EDGAR] SEC_USER_AGENT fehlt. Die SEC verlangt eine Kennung:\n"
            "  export SEC_USER_AGENT='Vorname Nachname mail@example.com'")
    return ua


def hole(url: str, ziel: str = None, pause: float = 0.34) -> bytes:
    """Laedt eine EDGAR-Adresse. pause haelt die Rate unter 10 Abrufen je Sekunde."""
    req = urllib.request.Request(url, headers={"User-Agent": _ua()})
    with urllib.request.urlopen(req, timeout=60) as fh:
        roh = fh.read()
    time.sleep(pause)
    if ziel:
        os.makedirs(os.path.dirname(ziel), exist_ok=True)
        with open(ziel, "wb") as fh:
            fh.write(roh)
    return roh


def einreichungen(formen: tuple) -> list:
    """Gibt [(form, datum, aktenzeichen)] der angeforderten Formen zurueck."""
    roh = hole(f"https://data.sec.gov/submissions/CIK{CIK}.json")
    r = json.loads(roh)["filings"]["recent"]
    return [(r["form"][i], r["filingDate"][i], r["accessionNumber"][i])
            for i in range(len(r["form"])) if r["form"][i] in formen]


def anhaenge(akte: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest das Inhaltsverzeichnis einer Einreichung und gibt je Anhang
        die Anhangsnummer und den Dateinamen zurueck. Der Dateiname wird
        NICHT geraten: die Namen wechseln von Jahr zu Jahr
        ("ex99212312025mda.htm", "a2021mdadecember31.htm").

    Inputs:
        akte (str): Aktenzeichen mit Bindestrichen, z. B. 0001178819-26-000048

    Outputs:
        paare (list): [(anhangsnummer, dateiname)], z. B. [("EX-99.2", "...htm")]
    --------------------------------------------------------------------------
    """
    nackt = akte.replace("-", "")
    roh = hole(f"{BASIS}/{nackt}/{akte}-index.htm").decode("utf-8", "replace")
    # Die Indexseite ist eine Tabelle; je Zeile stehen Dateiname und Typ. Der
    # Typ steht mal vor, mal hinter dem Dateinamen, deshalb wird die ganze
    # Zeile eingesammelt und beides daraus gelesen.
    paare = []
    for zeile in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", roh):
        text = re.sub(r"<[^>]+>", "|", zeile)
        datei = re.search(r"\|([A-Za-z0-9._-]+\.htm)\|", text)
        typ = re.search(r"\|(EX-99\.\d+|40-F|6-K)\|", text)
        if datei and typ:
            paare.append((typ.group(1), datei.group(1)))
    return paare


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jahre", default="2016-2025",
                   help="Geschaeftsjahre der 40-F-Einreichungen, z. B. 2016-2025")
    p.add_argument("--liste", action="store_true",
                   help="nur anzeigen, welche Anhaenge vorhanden sind")
    a = p.parse_args()
    von, bis = (int(x) for x in a.jahre.split("-"))

    for form, datum, akte in einreichungen(("40-F",)):
        # Die Einreichung eines Jahres betrifft das VORjahr.
        jahr = int(datum[:4]) - 1
        if not von <= jahr <= bis:
            continue
        print(f"\n40-F {jahr} (eingereicht {datum}, {akte})")
        for typ, datei in anhaenge(akte):
            print(f"   {typ:9s} {datei}")
            if a.liste:
                continue
            kurz = {"EX-99.1": "aif", "EX-99.2": "mda", "EX-99.3": "fs"}.get(typ)
            if not kurz:
                continue
            ziel = os.path.join(REFS, f"agi-{kurz}-{jahr}.htm")
            if os.path.exists(ziel):
                print(f"              -> bereits vorhanden")
                continue
            hole(f"{BASIS}/{akte.replace('-', '')}/{datei}", ziel)
            print(f"              -> refs/agi-{kurz}-{jahr}.htm "
                  f"({os.path.getsize(ziel) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
