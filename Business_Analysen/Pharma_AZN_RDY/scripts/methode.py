#!/usr/bin/env python3
"""
methode.py - schreibt die Zahlen des Methodenabschnitts nach data/methode.tex.

Ein Dokument, das angibt, wie viele Werte es prueft, darf diese Zahl nicht
selbst tippen. Sie kommt aus den Wachhunden.

Aufruf: python3 scripts/methode.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pruefe_seiten                                          # noqa: E402
import reihe_azn                                              # noqa: E402
import reihe_rdy                                              # noqa: E402
from kennzahlen import EXTERN                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    puffer = io.StringIO()
    with redirect_stdout(puffer):
        rc = pruefe_seiten.main()
    zeilen = [z for z in puffer.getvalue().splitlines() if z.startswith("[SEITEN]")]
    geprueft = int(zeilen[0].split()[1])
    fehler = int(zeilen[0].split()[4])
    gleichungen = 0
    for modul in (reihe_azn, reihe_rdy):
        alle = modul.sammle()
        gleichungen += sum(len(v) - 1 for v in alle.values() if len(v) > 1)
    with open(os.path.join(ROOT, "data", "methode.tex"), "w") as f:
        f.write("% erzeugt von scripts/methode.py - NICHT von Hand aendern\n")
        f.write(f"\\newcommand{{\\MethodeWerte}}{{{geprueft}}}\n")
        f.write(f"\\newcommand{{\\MethodeFehler}}{{{fehler}}}\n")
        f.write(f"\\newcommand{{\\MethodeExtern}}{{{len(EXTERN)}}}\n")
        f.write(f"\\newcommand{{\\MethodeKette}}{{{gleichungen}}}\n")
    print(f"[METHODE] {geprueft} Werte, {fehler} nicht belegt, "
          f"{gleichungen} Kettengleichungen, {len(EXTERN)} Sekundaerwerte")
    return rc


if __name__ == "__main__":
    sys.exit(main())
