#!/usr/bin/env python3
"""
lesen.py - liest benannte Zahlenzellen aus einer gedruckten Neste-Seite.

Die Kennzahlenseiten und die Kapitalflussrechnung setzen ZWEI Tabellen nebeneinander
in dieselbe Textzeile ("Revenue  EUR million  19,016 ...   Earnings per share (EPS)
EUR  0.19 ..."). Gelesen wird deshalb ab dem Etikett bis zum naechsten Wort, das eine
neue Zelle beginnt; Einheiten ("EUR million", "EUR", "%") und Fussnotenzeichen ("1)")
zaehlen nicht als Wort. Vor den Werten kann eine Notenspalte stehen ("4, 15"); darum
werden die LETZTEN n Zahlen des Abschnitts genommen, nie die ersten.

Minuszeichen: der Jahrgang 2019 setzt den Halbgeviertstrich ("–191"), die spaeteren
den Bindestrich. Beide werden vor dem Lesen vereinheitlicht.

Findet sich das Etikett nicht, kommt None zurueck; geraten wird nicht.
Aufruf: python3 scripts/lesen.py <refs/datei.txt> <Seite> <n> "<Regex-Etikett>" [...]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                      # noqa: E402

ZELLE = r"(?:^\s*|(?<=\s{2}))"
_ZAHL = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_EINHEIT = re.compile(r"^[\s,]*(?:EUR million|MEUR|EUR|USD/ton|%)(?![A-Za-z])")


def _vereinheitlicht(z: str) -> str:
    return z.replace("–", "-").replace("−", "-").replace("—", "-")


def _abschnitt(rest: str) -> str:
    """Text ab dem Etikett bis zur naechsten Zelle, die mit einem Wort beginnt."""
    rest = _EINHEIT.sub("", rest)
    rest = re.sub(r"(?<=\s)\d\)", " ", rest)            # Fussnote "1)" hinter einem Wert
    m = re.search(r"\s{2,}(?:[A-Za-z(]|-\s*[a-z])", rest)   # "- of weighted ..." beginnt auch eine Zelle
    return rest[:m.start()] if m else rest


def werte(text: str, etikett: str, n: int) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die letzten n Zahlen der Zelle zurueck, die mit dem Etikett beginnt.
        Steht das Etikett mehrfach auf der Seite, gilt die erste Fundstelle, die
        n Zahlen traegt.

    Inputs:
        text (str): Text der gedruckten Seite (pdftotext -layout)
        etikett (str): regulaerer Ausdruck fuer die Zeilenbeschriftung
        n (int): Zahl der Berichtsjahre in der Spalte

    Outputs:
        werte (list[float]) oder None
    --------------------------------------------------------------------------
    """
    for zeile in text.splitlines():
        z = _vereinheitlicht(zeile)
        for m in re.finditer(ZELLE + etikett, z):
            zahlen = [float(x.replace(",", "")) for x in _ZAHL.findall(_abschnitt(z[m.end():]))]
            if len(zahlen) >= n:
                return zahlen[-n:]
    return None


def werte_spalten(text: str, etikett: str, kopf: str, n: int, links: int = 10, rechts: int = 9) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die Werte nach Spaltenstellung statt nach Reihenfolge. Die
        Kapitalflussrechnung setzt vor die Werte eine Notenspalte ("13") und im
        Jahrgang 2025 hinter die rechte Tabelle eine Randspalte ("3", "18"); beide
        sind Zahlen und verschoeben jede Auswahl nach Position in der Zeile.
        Rechtsbuendige Werte enden in derselben Textspalte wie ihr Jahreskopf.

    Inputs:
        text (str): Text der gedruckten Seite
        etikett (str): regulaerer Ausdruck fuer die Zeilenbeschriftung
        kopf (str): regulaerer Ausdruck fuer einen Jahreskopf, z. B. r"Dec 20\\d\\d"
        n (int): Zahl der Jahresspalten rechts vom Etikett
        links, rechts (int): Fenster um die Kopfenden in Zeichen

    Outputs:
        werte (list[float]) in der Reihenfolge der Koepfe, oder None
    --------------------------------------------------------------------------
    """
    zeilen = [_vereinheitlicht(z) for z in text.splitlines()]
    koepfe = sorted({m.end() for z in zeilen for m in re.finditer(kopf, z)})
    for z in zeilen:
        for m in re.finditer(ZELLE + etikett, z):
            enden = [k for k in koepfe if k > m.end()][:n]
            if len(enden) < n:
                continue
            # Fenster um die Koepfe: davor steht die Notenspalte, dahinter im Jahrgang
            # 2025 eine Randspalte. Gemessen: Werte der rechten Tabelle enden bis zu
            # acht Zeichen hinter ihrem Kopf, die Randspalte elf und mehr.
            kand = [x for x in _ZAHL.finditer(z, m.end())
                    if enden[0] - links <= x.end() <= enden[-1] + rechts]
            if len(kand) < n:
                continue
            aus = []
            for k in enden:
                x = min(kand, key=lambda x: abs(x.end() - k))
                aus.append(float(x.group(0).replace(",", "")))
                kand.remove(x)
            return aus
    return None


def seite(dokument: str, s: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return seiten(os.path.join(root, "refs", dokument + ".txt")).get(str(s), "")


if __name__ == "__main__":
    t = seiten(sys.argv[1]).get(sys.argv[2], "")
    for e in sys.argv[4:]:
        print(f"  {e[:50]:52s} {werte(t, e, int(sys.argv[3]))}")
