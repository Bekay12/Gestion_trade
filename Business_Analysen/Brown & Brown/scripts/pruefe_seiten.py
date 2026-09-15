#!/usr/bin/env python3
"""
pruefe_seiten.py - prueft jede Seitenangabe aus scripts/kennzahlen.py gegen
die Quelle.

Regel des Projekts: eine Zahl wird auf der gedruckten Seite belegt, auf der
sie tatsaechlich steht. Eine geschaetzte Seitenzahl faellt erst auf, wenn ein
Leser die Quelle oeffnet. Das Skript sucht jeden Rohwert im Text genau der
Seite, die sein Kommentar nennt, und meldet jede Angabe, die dort nicht
vorkommt.

Der Suchtext wird in mehreren Schreibweisen geprueft (mit und ohne
Tausendertrennzeichen, mit Klammern fuer negative Betraege), weil die
EDGAR-Einreichung Betraege als "5,763" und Minuszeichen als "(54" setzt.

Aufruf: python3 scripts/pruefe_seiten.py
Rueckgabewert 1, wenn eine Angabe nicht belegt ist.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kennzahlen import ROH  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATEI = {
    "10-K 2025": "refs/10k-2025.txt",
    "10-K 2024": "refs/10k-2024.txt",
    "10-Q Q2 2026": "refs/10q-2026q2.txt",
    "Proxy Statement 2026": "refs/def14a-2026.txt",
}
_QUELLE = re.compile(r"^(.+?),\s*S\.\s*(\d+)")


def seiten(pfad: str) -> dict:
    """Gibt {gedruckte Seite: Text} der Einreichung zurueck."""
    voll = os.path.join(ROOT, pfad)
    if not os.path.exists(voll):
        raise SystemExit(
            f"[SEITEN] {pfad} fehlt. Die SEC-Einreichungen liegen nicht im\n"
            "Repository. Einmalig beschaffen mit:\n"
            "  export SEC_USER_AGENT='Vorname Nachname mail@example.com'\n"
            "  ./scripts/fetch_quellen.sh")
    txt = open(voll, encoding="utf-8").read()
    teile = re.split(r"(?m)^=== SEITE (\S+) \(Block (\d+)\) ===$", txt)
    return {teile[i]: teile[i + 2] for i in range(1, len(teile), 3)}


def schreibweisen(wert) -> list:
    """Gibt die Schreibweisen zurueck, in denen der Wert im Text stehen kann."""
    formen = []
    if isinstance(wert, int) or (isinstance(wert, float) and wert == int(wert)):
        n = int(abs(wert))
        formen += [f"{n:,}", str(n)]
        if wert < 0:
            formen = [f"({f}" for f in formen] + formen
    else:
        formen += [f"{abs(wert):g}", f"{abs(wert):.2f}", f"{abs(wert):.1f}"]
    return formen


def main() -> int:
    cache = {}
    fehler = []
    geprueft = 0
    for name, (wert, quelle) in ROH.items():
        m = _QUELLE.match(quelle)
        if not m:
            continue
        dok, seite = m.group(1).strip(), m.group(2)
        if dok not in DATEI:
            fehler.append(f"{name}: unbekanntes Dokument {dok!r}")
            continue
        if dok not in cache:
            cache[dok] = seiten(DATEI[dok])
        text = cache[dok].get(seite)
        geprueft += 1
        if text is None:
            fehler.append(f"{name}: Seite {seite} in {dok} nicht vorhanden")
            continue
        if not any(f in text for f in schreibweisen(wert)):
            fehler.append(f"{name} = {wert}: nicht auf {dok}, S. {seite}")
    for f in fehler:
        print(f"FEHLT  {f}")
    print(f"[SEITEN] {geprueft} Angaben geprueft, {len(fehler)} nicht belegt")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
