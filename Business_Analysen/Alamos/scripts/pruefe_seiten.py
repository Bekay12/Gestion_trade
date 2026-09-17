#!/usr/bin/env python3
"""
pruefe_seiten.py - prueft jede Seitenangabe aus scripts/kennzahlen.py gegen
die Quelle.

Regel des Projekts: Eine Zahl wird auf der gedruckten Seite belegt, auf der
sie tatsaechlich steht. Eine geschaetzte Seitenzahl faellt erst auf, wenn ein
Leser die Quelle oeffnet. Das Skript sucht jeden Rohwert im Text genau der
Seite, die sein Kommentar nennt, und meldet jede Angabe, die dort nicht
vorkommt.

Der Suchtext wird in mehreren Schreibweisen geprueft, weil die Einreichung
Betraege als "5,763" setzt, negative als "( 54 )" und Unzen als "545,400".
Die Schreibweisen werden erzeugt und die Pruefung nicht abgeschwaecht: Ein
Wachhund, der bei jedem gruppierten Betrag falschen Alarm schlaegt, wird
abgeschaltet und schuetzt dann gar nichts mehr.

Aufruf: python3 scripts/pruefe_seiten.py
Rueckgabewert 1, wenn eine Angabe nicht belegt ist.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kennzahlen import ROH, EXTERN  # noqa: E402
from lies import seiten  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATEI = {
    "MD&A 2016": "refs/agi-mda-2016.txt", "MD&A 2017": "refs/agi-mda-2017.txt",
    "MD&A 2018": "refs/agi-mda-2018.txt", "MD&A 2019": "refs/agi-mda-2019.txt",
    "MD&A 2020": "refs/agi-mda-2020.txt", "MD&A 2021": "refs/agi-mda-2021.txt",
    "MD&A 2022": "refs/agi-mda-2022.txt", "MD&A 2023": "refs/agi-mda-2023.txt",
    "MD&A 2024": "refs/agi-mda-2024.txt", "MD&A 2025": "refs/agi-mda-2025.txt",
    "MD&A Q2 2026": "refs/agi-mda-q2-2026.txt",
    "Abschluss 2016": "refs/agi-fs-2016.txt", "Abschluss 2017": "refs/agi-fs-2017.txt",
    "Abschluss 2018": "refs/agi-fs-2018.txt", "Abschluss 2019": "refs/agi-fs-2019.txt",
    "Abschluss 2020": "refs/agi-fs-2020.txt", "Abschluss 2021": "refs/agi-fs-2021.txt",
    "Abschluss 2022": "refs/agi-fs-2022.txt", "Abschluss 2023": "refs/agi-fs-2023.txt",
    "Abschluss 2024": "refs/agi-fs-2024.txt", "Abschluss 2025": "refs/agi-fs-2025.txt",
    "Abschluss Q2 2026": "refs/agi-fs-q2-2026.txt",
    "AIF 2025": "refs/agi-aif-2025.txt",
    "Circular 2026": "refs/agi-circular-2026.txt",
    "Reserven 2025": "refs/agi-reserven-2025.txt",
}
_QUELLE = re.compile(r"^(.+?),\s*S\.\s*(\S+)$")


def schreibweisen(wert) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die Schreibweisen zurueck, in denen der Wert im Seitentext stehen
        kann: mit und ohne Tausendertrennzeichen, bei negativen Werten
        zusaetzlich in Klammern, bei Bruchzahlen mit einer und mit zwei
        Nachkommastellen.

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
        formen += [f"{a:,.1f}", f"{a:.1f}", f"{a:,.2f}", f"{a:.2f}", f"{a:g}"]
    if wert < 0:
        formen = [f"({f}" for f in formen] + formen
    return formen


def main() -> int:
    zwischen, fehler, geprueft, ohne = {}, [], 0, 0
    for name, (wert, quelle) in ROH.items():
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
                    "  export SEC_USER_AGENT='Vorname Nachname mail@example.com'\n"
                    "  python3 scripts/hole_quellen.py && python3 scripts/edgar_seiten.py")
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
