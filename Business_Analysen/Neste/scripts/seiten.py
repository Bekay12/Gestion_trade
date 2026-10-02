#!/usr/bin/env python3
"""
seiten.py - zerlegt die Neste-Berichte unter refs/ in gedruckte Seiten.

Ein pdftotext-Lauf je Datei, getrennt am Seitenvorschub. Die gedruckte Seitenzahl
steht bei allen Jahrgaengen rechtsbuendig in einer der letzten drei Zeilen:

    GB 2019-2022   "Neste Annual Report 2021 | Innovation        10"
    GB 2023-2025   blanke Zahl als letzte Zeile ("10")
    FSR 2017/2018  "Neste Corporation - Financial Statements Release for 2017   10"
    HJ 2026        "Neste Corporation - Half-year financial report ... 2026   9"
                   (PDF-Seite = gedruckte + 1: das Deckblatt ist ungezaehlt)

Die Zeile wird NICHT von Leerzeichen befreit: im Halbjahresbericht verschmoelze
sonst "2026   9" zu "20269". Gelesen wird die Zahl hinter mindestens zwei
Leerzeichen am Zeilenende oder eine Zeile, die nur aus der Zahl besteht. Welche
Zahl die Seite ist, entscheidet die Folgekette (folgefilter), nicht die Stellung.

Ausgabe: refs/<name>.txt mit Trennzeilen "=== SEITE <n> (Block <i>) ===".
Aufruf: python3 scripts/seiten.py [refs/datei.pdf ...]
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgar_seiten import folgefilter, luecken                 # noqa: E402
from seiten_zerlegen import _bericht, _schreibe, fuelle_luecken  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENDE = re.compile(r"(?:^|\s{2,})(\d{1,3})\s*$")


def seitenzahl(zeilen: list) -> str:
    """Kandidat aus den letzten drei nichtleeren Zeilen, von unten gelesen."""
    for z in zeilen[-3:][::-1]:
        m = _ENDE.search(z)
        if m:
            return str(int(m.group(1)))
    return "?"


def zerlege_pdf(quelle: str, ziel: str):
    t = subprocess.run(["pdftotext", "-layout", quelle, "-"], capture_output=True,
                       text=True).stdout
    roh_kand, texte = [], []
    for block in t.split("\f"):
        zeilen = [z.rstrip() for z in block.splitlines() if z.strip()]
        roh_kand.append(seitenzahl(zeilen) if zeilen else "?")
        texte.append("\n".join(zeilen))
    if texte and not texte[-1].strip():          # Rest nach dem letzten Vorschub
        texte, roh_kand = texte[:-1], roh_kand[:-1]
    roh = folgefilter(roh_kand)
    seiten = fuelle_luecken(roh)
    _schreibe(ziel, seiten, texte)
    return seiten, roh


def main(argv: list) -> int:
    refs = os.path.join(ROOT, "refs")
    dateien = argv or sorted(os.path.join(refs, f) for f in os.listdir(refs)
                             if f.endswith(".pdf") and not f.startswith("_"))
    mangel = []
    for quelle in dateien:
        ziel = os.path.splitext(quelle)[0] + ".txt"
        alle, roh = zerlege_pdf(quelle, ziel)
        seiten = [s for s in alle if s != "?"]
        zahlen = sorted({int(s) for s in seiten if s.isdigit()})
        fehlend = luecken(seiten)
        if not zahlen:
            befund = "ohne Paginierung"
            mangel.append(os.path.basename(quelle))
        elif fehlend:
            befund = f"Seiten {zahlen[0]}-{zahlen[-1]}, {len(fehlend)} fehlend: {fehlend[:8]}"
            mangel.append(os.path.basename(quelle))
        else:
            versatz = next((i + 1 - int(s) for i, s in enumerate(roh) if s.isdigit()), None)
            befund = (f"Seiten {zahlen[0]}-{zahlen[-1]} lueckenlos, {_bericht(roh, alle)}, "
                      f"PDF-Seite = gedruckte + {versatz}")
        print(f"{os.path.basename(ziel):22s} {len(alle):4d} Bloecke, {befund}")
    if mangel:
        print(f"\n[SEITEN] zu pruefen: {', '.join(mangel)}")
    return 1 if mangel else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
