#!/usr/bin/env python3
"""
zeile.py - liest benannte Zahlenzeilen aus einer gedruckten Seite.

Zwei Quellenarten, zwei Satzformen, und das ist der Grund fuer dieses Skript:

    AZN (PDF, pdftotext -layout)   Etikett und Werte stehen in EINER Zeile:
                                   "Total Revenue   58,739  54,073  45,811"
    RDY (HTML, Zellen je Zeile)    Etikett steht allein, die Werte folgen
                                   einzeln, dazwischen "Rs.", "U.S.$" und
                                   Notennummern.

Gelesen werden beide Formen; geraten wird in keiner. Findet sich das Etikett
nicht, kommt None zurueck, und der Wert wird nicht belegt.

Aufruf: python3 scripts/zeile.py <refs/datei.txt> <Seite> <n> "<Etikett>" [...]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                     # noqa: E402

_ZAHL = re.compile(r"\(?-?[\d][\d,]*(?:\.\d+)?\)?")
# Zeichen, die zwischen Etikett und Werten stehen duerfen, ohne dass die
# Zeile als Wert gilt: Waehrungszeichen, Notenverweise, Gedankenstriche.
_UEBERGEHEN = {"$", "Rs.", "U.S.$", "US$", "₹", "", "–", "—", "-", "(", ")", "%"}


def _zahl(s: str):
    s = s.strip()
    if s in ("–", "—", "-"):
        return 0.0
    neg = s.startswith("(") or s.endswith(")")
    s = s.strip("()$ ").replace(",", "").replace("Rs.", "").replace("U.S.$", "")
    if not s or not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    v = float(s)
    return -v if neg else v


def aus_zeile(text: str, etikett: str, n: int = 3, ueberspringen: int = 0,
               von_hinten: bool = False) -> list:
    """Werte aus der Zeile, die mit dem Etikett beginnt (AZN-Form)."""
    ziel = etikett.strip().lower()
    for z in text.splitlines():
        s = z.strip()
        if not s.lower().startswith(ziel):
            continue
        rest = s[len(ziel):]
        werte = []
        for m in _ZAHL.finditer(rest):
            v = _zahl(m.group(0))
            if v is not None:
                werte.append(v)
        if von_hinten:
            return werte[-n:] if len(werte) >= n else (werte or None)
        # Fussnotenverweis vor den Werten. Eine Zeile mit Notenverweis liefert
        # einen Wert zu viel, und der erste ist eine kleine ganze Zahl neben
        # Betraegen ganz anderer Groessenordnung: "Dividends paid  25  (4,971)
        # (4,629)". Ungeprueft wandert die Notennummer als Betrag in die
        # Zahlenschicht, und sie steht auf der zitierten Seite - der
        # Seitenwachhund kann sie nicht finden.
        if (len(werte) == n + 1 + ueberspringen and werte[0] == int(werte[0])
                and 0 < werte[0] < 40
                and all(abs(w) > 10 * abs(werte[0]) for w in werte[1:] if w)):
            werte = werte[1:]
        if len(werte) >= n + ueberspringen:
            return werte[ueberspringen:ueberspringen + n]
        if werte:
            return werte[ueberspringen:]
    return None


def entfalte(text: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Fuegt umbrochene Etiketten wieder zusammen. Die aelteren Einreichungen
        von Dr. Reddy's setzen "Expenditure on property," und "plant and
        equipment" in zwei Zeilen; ein Etikettenvergleich findet dort nichts,
        und die Reihe bricht still ab. Zusammengefuegt wird nur, was beides
        keine Zahl und keine Waehrungsmarke ist - die Wertespalten bleiben
        dadurch unberuehrt.

    Inputs:
        text (str): Text der gedruckten Seite

    Outputs:
        text (str): derselbe Text mit zusammengefuegten Etikettenzeilen
    --------------------------------------------------------------------------
    """
    aus = []
    for z in (z.replace("\u200b", "").strip() for z in text.splitlines()):
        ist_wert = z in _UEBERGEHEN or _zahl(z) is not None
        # Eine Zeile, die auf ":" endet, ist eine Zwischenueberschrift
        # ("Cash flows from investing activities:"). Wird das folgende
        # Etikett an sie angehaengt, verschwindet es aus der Suche.
        if (aus and not ist_wert and not aus[-1].endswith(":")
                and not (aus[-1] in _UEBERGEHEN or _zahl(aus[-1]) is not None)):
            aus[-1] = f"{aus[-1]} {z}"
        else:
            aus.append(z)
    return "\n".join(aus)


def aus_spalte(text: str, etikett: str, n: int = 3, ueberspringen: int = 0,
                von_hinten: bool = False) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die Werte aus den Folgezeilen des Etiketts (RDY-Form). Die
        Klammern negativer Betraege stehen dort als EIGENE Zeilen:

            Unrealized exchange (gain)/loss, net
            (
            529
            )

        Wer die Klammerzeilen ueberspringt, statt sie zu lesen, verkehrt jedes
        Vorzeichen ins Gegenteil - und der Betrag steht danach unveraendert
        auf der zitierten Seite, sodass der Seitenwachhund schweigt.

    Inputs:
        text (str): Text der gedruckten Seite
        etikett (str): Zeilenbeschriftung
        n (int): Zahl der gesuchten Werte

    Outputs:
        werte (list) oder None, wenn das Etikett nicht vorkommt
    --------------------------------------------------------------------------
    """
    # Null-Breite-Leerzeichen trennen bei AstraZeneca jede Tabellenzelle. Wer
    # sie stehen laesst, findet weder das Etikett noch den zweiten Jahreswert.
    zeilen = [z.replace("\u200b", "").strip() for z in text.splitlines()]
    ziel = etikett.strip().lower()
    for i, z in enumerate(zeilen):
        if z.lower() != ziel:
            continue
        out, j, negativ = [], i + 1, False
        n = n + ueberspringen + (6 if von_hinten else 0)
        # Notenverweis vor der Waehrungsmarke. Die Zeile "Revenues" wird von
        # der Notennummer "21" gefolgt, dann von "U.S.$" und erst dann von den
        # Betraegen. Wer die Nummer als Wert nimmt, verschiebt die ganze Zeile
        # um ein Jahr - und jeder Wert steht danach noch immer auf der
        # zitierten Seite. Deshalb: steht vor der ersten Zahl eine
        # Waehrungsmarke, wird erst dahinter gelesen.
        for k in range(i + 1, min(i + 6, len(zeilen))):
            if zeilen[k] in ("U.S.$", "US$", "Rs.", "$"):
                j = k + 1
                break
            # Eine blanke ein- bis zweistellige Zahl ist ein Notenverweis und
            # steht vor der Waehrungsmarke; alles andere ist bereits ein Wert,
            # und dann darf nicht weitergesucht werden. Ohne diese Grenze
            # laeuft die Suche in die Waehrungsmarke der NAECHSTEN Zeile und
            # liest deren Werte - bei "Total equity" das Bilanzsummenpaar.
            if re.fullmatch(r"\d{1,2}", zeilen[k]):
                continue
            if _zahl(zeilen[k]) is not None and zeilen[k] != "(":
                break
        while j < len(zeilen) and len(out) < n:
            s = zeilen[j]
            if s == "(":
                negativ = True
            elif s in _UEBERGEHEN:
                pass                        # Waehrungszeichen oder Klammer zu
            else:
                v = _zahl(s)
                if v is None:
                    if out:
                        break
                else:
                    out.append(-abs(v) if negativ else v)
                    negativ = False
            j += 1
        if von_hinten:
            return out[-(n - 6):] if len(out) >= n - 6 else (out or None)
        return out[ueberspringen:] or None
    return None


def lies_werte(datei: str, seite: str, etikett: str, n: int = 3, **kw) -> list:
    text = seiten(datei).get(str(seite), "")
    ueber, hinten = kw.get("ueberspringen", 0), kw.get("von_hinten", False)
    return (aus_zeile(text, etikett, n, ueber, hinten)
            or aus_spalte(text, etikett, n, ueber, hinten)
            or aus_spalte(entfalte(text), etikett, n, ueber, hinten))


if __name__ == "__main__":
    datei, seite, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    text = seiten(datei).get(seite, "")
    for etikett in sys.argv[4:]:
        a = aus_zeile(text, etikett, n)
        b = aus_spalte(text, etikett, n)
        print(f"  {etikett[:44]:46s} {a if a else b}")
