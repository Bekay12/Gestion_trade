#!/usr/bin/env python3
"""
pruefe_seiten.py - prueft jede Seitenangabe aus scripts/kennzahlen.py gegen
den Text der Seite, die sie nennt.

Regel des Projekts: Eine Zahl wird auf der gedruckten Seite belegt, auf der
sie tatsaechlich steht. Eine geschaetzte Seitenzahl faellt erst auf, wenn ein
Leser die Quelle oeffnet - und dann ist es der Leser, der den Fehler findet.
Deshalb ist der Beleg hier ein Skript und keine Gewohnheit.

Der Suchtext wird in mehreren Schreibweisen geprueft: die Einreichungen
setzen "5,763" fuer 5763, "( 54 )" fuer -54, Kurse mit zwei und Margen mit
einer Nachkommastelle. Die Schreibweisen werden erzeugt und die Pruefung
nicht abgeschwaecht: ein Wachhund, der bei jedem gruppierten Betrag falschen
Alarm schlaegt, wird abgeschaltet und schuetzt dann gar nichts mehr.

Aufruf: python3 scripts/pruefe_seiten.py
Rueckgabewert 1, wenn eine Angabe nicht belegt ist.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kennzahlen import ROH, EXTERN, reihen_roh                # noqa: E402
from lies import seiten                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATEI = {
    "Geschaeftsbericht 2021": "refs/azn-ar-2021.txt",
    "Geschaeftsbericht 2022": "refs/azn-ar-2022.txt",
    "Geschaeftsbericht 2023": "refs/azn-ar-2023.txt",
    "Geschaeftsbericht 2024": "refs/azn-ar-2024.txt",
    "Geschaeftsbericht 2025": "refs/azn-ar-2025.txt",
    "Halbjahresbericht 2026": "refs/azn-h1-2026.txt",
    "Form 20-F 2020": "refs/rdy-20f-2020.txt",
    "Form 20-F 2022": "refs/rdy-20f-2022.txt",
    "Form 20-F 2023": "refs/rdy-20f-2023.txt",
    "Form 20-F 2025": "refs/rdy-20f-2025.txt",
    "Form 20-F 2026": "refs/rdy-20f-2026.txt",
    "Quartalsbericht 2027": "refs/rdy-q1-fy2027.txt",
}
_QUELLE = re.compile(r"^(.+?),\s*S\.\s*(\S+)$")


def schreibweisen(wert) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die Schreibweisen zurueck, in denen der Wert auf der Seite stehen
        kann: mit und ohne Tausendertrennzeichen, negative zusaetzlich mit
        oeffnender Klammer, Bruchzahlen mit einer und zwei Nachkommastellen.

    Inputs:
        wert (int|float): der zu belegende Rohwert

    Outputs:
        formen (list): moegliche Zeichenfolgen
    --------------------------------------------------------------------------
    """
    formen = []
    if isinstance(wert, int) or (isinstance(wert, float) and wert == int(wert)):
        n = int(abs(wert))
        formen += [f"{n:,}", str(n)]
    else:
        a = abs(wert)
        formen += [f"{a:,.2f}", f"{a:.2f}", f"{a:,.1f}", f"{a:.1f}", f"{a:g}"]
    if wert < 0:
        formen = [f"({f}" for f in formen] + [f"( {f}" for f in formen] + formen
    return formen


def main() -> int:
    alle = dict(ROH)
    alle.update(reihen_roh())
    zwischen, fehler, geprueft, ohne = {}, [], 0, 0
    for name, (wert, quelle) in sorted(alle.items()):
        m = _QUELLE.match(quelle)
        if not m:
            ohne += 1
            continue
        dok, seite = m.group(1).strip(), m.group(2)
        if dok not in DATEI:
            fehler.append(f"{name}: unbekanntes Dokument {dok!r}")
            continue
        if dok not in zwischen:
            pfad = os.path.join(ROOT, DATEI[dok])
            if not os.path.exists(pfad):
                raise SystemExit(
                    f"[SEITEN] {DATEI[dok]} fehlt. Die Einreichungen liegen nicht im\n"
                    "Repository. Einmalig beschaffen mit:\n"
                    "  python3 scripts/hole_quellen.py && python3 scripts/seiten_zerlegen.py")
            zwischen[dok] = seiten(DATEI[dok])
        text = zwischen[dok].get(seite)
        geprueft += 1
        if text is None:
            fehler.append(f"{name}: Seite {seite} in {dok} nicht vorhanden")
            continue
        if not any(f in text for f in schreibweisen(wert)):
            fehler.append(f"{name} = {wert}: nicht auf {dok}, S. {seite}")
    for f in fehler:
        print(f"FEHLT  {f}")
    print(f"[SEITEN] {geprueft} Angaben geprueft, {len(fehler)} nicht belegt")
    print(f"[SEITEN] {len(EXTERN)} Sekundaerwerte stehen ausserhalb der Pruefung: "
          "sie tragen keine gedruckte Seite und sind im Dokument als "
          "Sekundaerquelle ausgewiesen.")
    if ohne:
        print(f"[SEITEN] WARNUNG: {ohne} Rohwerte ohne auswertbare Seitenangabe")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
