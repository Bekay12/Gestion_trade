#!/usr/bin/env python3
"""
lies.py - liest benannte Zahlenzeilen aus einer gedruckten Seite.

Die Einreichungen setzen jede Tabellenzeile als Etikett, dann die Werte der
Berichtsjahre, dazwischen "$", Klammern fuer negative Betraege und
Gedankenstriche fuer Null. Die Funktion liest genau dieses Muster und raet
nie: findet sie das Etikett nicht, gibt sie None zurueck und ueberlaesst dem
Beleg-Wachhund das Wort.

Aufruf:  python3 scripts/lies.py <refs/datei.txt> <Seite> "<Etikett>" [...]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ZAHL = re.compile(r"\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?")


def seiten(pfad: str) -> dict:
    """Gibt {gedruckte Seite: Text} zurueck; mehrfach belegte Seiten werden vereint."""
    voll = pfad if os.path.isabs(pfad) else os.path.join(ROOT, pfad)
    teile = re.split(r"(?m)^=== SEITE (\S+) \(Block (\d+)\) ===$",
                     open(voll, encoding="utf-8").read())
    d = {}
    for i in range(1, len(teile), 3):
        # Faellt eine gedruckte Seite auf zwei <hr>-Bloecke, darf der zweite
        # den ersten nicht verdraengen - sonst waere die halbe Seite
        # unbelegbar, ohne dass es auffiele.
        d[teile[i]] = d.get(teile[i], "") + "\n" + teile[i + 2]
    return d


def werte(text: str, etikett: str, n: int = 2) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die ersten n Zahlen zurueck, die im Text auf das Etikett folgen.

    Inputs:
        text (str): Text der gedruckten Seite
        etikett (str): Zeilenbeschriftung, Gross-/Kleinschreibung unerheblich
        n (int): Zahl der gesuchten Werte (Berichtsjahre)

    Outputs:
        werte (list) oder None, wenn das Etikett nicht vorkommt
    --------------------------------------------------------------------------
    """
    zeilen = [z.strip() for z in text.splitlines()]
    ziel = etikett.strip().lower()
    for i, z in enumerate(zeilen):
        if z.lower() != ziel:
            continue
        out, j, negativ = [], i + 1, False
        while j < len(zeilen) and len(out) < n:
            s = zeilen[j]
            if s == "(":
                negativ = True
            elif s in ("$", ")", ""):
                pass
            elif s in ("—", "–", "-"):
                out.append(0.0)
            elif _ZAHL.fullmatch(s):
                v = float(s.strip("()$ ").replace(",", ""))
                out.append(-v if negativ else v)
                negativ = False
            elif out:
                break
            j += 1
        return out
    return None


if __name__ == "__main__":
    text = seiten(sys.argv[1]).get(sys.argv[2], "")
    for etikett in sys.argv[3:]:
        print(f"  {etikett[:46]:48s} {werte(text, etikett, 3)}")
