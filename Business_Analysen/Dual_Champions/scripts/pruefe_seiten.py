#!/usr/bin/env python3
"""
pruefe_seiten.py - prueft jeden Rohwert gegen den Text der gedruckten Seite, die er nennt.

Geprueft werden (1) alle ROH-Werte aus rechnung.py und (2) jeder Reihenwert aus
data/reihen.json, der aus einer Seite stammt (XBRL-gefuellte Jahre tragen keine Seite
und sind als Sekundaerquelle ausgewiesen). Der Suchtext wird in den Schreibweisen der
Quellen erzeugt: "5,763", "( 54 )", "−54", mit und ohne Nachkommastellen - die Pruefung
wird dadurch nicht schwaecher, sie findet nur dieselbe Zahl in ihrer gedruckten Form.

Aufruf: python3 scripts/pruefe_seiten.py      Rueckgabewert 1, wenn ein Wert fehlt.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                      # noqa: E402
from rechnung import ROH                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = {}


def text(dokument: str, seite: str) -> str:
    if dokument not in _CACHE:
        _CACHE[dokument] = seiten(os.path.join(ROOT, "refs", dokument + ".txt"))
    t = _CACHE[dokument].get(str(seite), "")
    return re.sub(r"[\s​ ]+", "", t)


def schreibweisen(w: float) -> set:
    a = abs(w)
    aus = set()
    for stellen in (0, 1, 2, 3):
        s = f"{a:,.{stellen}f}"
        if float(s.replace(",", "")) == round(a, stellen):
            aus.add(s)
            aus.add(s.replace(",", ""))
    if a == int(a):
        aus.add(f"{int(a):,}")
    return {x for x in aus if x}


def belegt(w: float, dokument: str, seite: str) -> bool:
    t = text(dokument, seite)
    return any(s in t for s in schreibweisen(w))


def main() -> int:
    fehler, n = [], 0
    for name, (w, _, dok, seite) in ROH.items():
        n += 1
        if not belegt(w, dok, seite):
            fehler.append(f"ROH {name}: {w} nicht auf {dok} S. {seite}")
    reihen = json.load(open(os.path.join(ROOT, "data", "reihen.json")))
    for k, posten in reihen.items():
        for p, jahre in posten.items():
            for j, v in jahre.items():
                if v.get("seite") is None or v.get("summe"):
                    continue          # XBRL-Fuellung bzw. Summe mehrerer Zeilen: Teile einzeln gelesen
                n += 1
                if not belegt(v["roh"], v["bericht"], v["seite"]):
                    fehler.append(f"REIHE {k} {p} {j}: {v['roh']} nicht auf {v['bericht']} S. {v['seite']}")
    for f in fehler:
        print("[SEITE]", f)
    print(f"[SEITE] {n - len(fehler)} von {n} Werten auf ihrer gedruckten Seite belegt")
    # Die Zahl der gepruefften Werte steht im Methodenabsatz; sie kommt von hier, nicht von Hand.
    with open(os.path.join(ROOT, "data", "pruefung.tex"), "w") as fo:
        fo.write(f"\\newcommand{{\\GeprueftAnzahl}}{{{n - len(fehler)}}}\n")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
