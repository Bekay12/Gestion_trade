#!/usr/bin/env python3
"""
seiten_zerlegen.py - zerlegt die Quellen beider Emittenten in gedruckte Seiten.

Die Regel des Verfahrens lautet: belegt wird die GEDRUCKTE Seite, und sie wird
gelesen, nie aus einer Blocknummer geschlossen. Die vier Quellenarten dieses
Projekts setzen ihre Seitenumbrueche verschieden, und genau daran scheitert
eine uebernommene Zerlegung:

    AZN Geschaeftsbericht   PDF; Fusszeile "AstraZeneca Annual Report &
                            Form 20-F Information <Jahr>   <Seite>".
                            Die PDF-Seite ist die gedruckte plus zwei.
    RDY Form 20-F           HTML ohne <hr>; der Umbruch steht als
                            "break-after:page" in einem <div>, die Seitenzahl
                            als letzte Textzeile des Blocks.
    AZN Halbjahresbericht   HTML; Umbruch als "page-break-after:always".
    RDY Quartalsbericht     HTML mit <hr>, wie die aelteren Einreichungen.

Ausgabe: refs/<name>.txt mit Trennzeilen "=== SEITE <n> (Block <i>) ===",
also im selben Format, das lies.py und pruefe_seiten.py erwarten.

Aufruf: python3 scripts/seiten_zerlegen.py [refs/datei ...]
Rueckgabewert 1, wenn eine Datei Luecken in der Seitenfolge hat.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgar_seiten import text_aus, folgefilter, luecken   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Trennmuster je Quellenart. Wird keins gefunden, bleibt die Datei ungeteilt
# und ihre Angaben sind nur ueber die Blocknummer zu belegen.
_TRENNER = [
    r'(?i)<div[^>]*break-after\s*:\s*page[^>]*>',
    r'(?i)<div[^>]*page-break-after\s*:\s*always[^>]*>',
    r'(?i)<div[^>]*page-break-before\s*:\s*always[^>]*>',
    r'(?i)<hr\b[^>]*>',
]
# Fusszeile des AZN-Geschaeftsberichts. Der Titel traegt das Jahr, deshalb
# steht es im Muster als Gruppe und nicht als fester Text.
_PDF_TITEL = r"AstraZeneca Annual Report (?:&|and) Form 20-F Information\s+\d{4}"
# Die Stellung der Seitenzahl wechselt zwischen den Jahrgaengen: 2025 setzt sie
# hinter den Titel, 2019 bis 2024 davor. Beide Formen werden gelesen; welche
# gilt, entscheidet nicht der Jahrgang, sondern die Zeile.
_PDF_FUSS = [re.compile(rf"^(\d{{1,3}})\s+{_PDF_TITEL}"),
             re.compile(rf"{_PDF_TITEL}\s+(\d{{1,3}})(?:\s|$)"),
             # Jahrgang 2024 setzt auf manchen Seiten nur die blanke Zahl; die
             # Jahrgaenge 2019 und 2020 verlieren den Titel beim Auslesen, weil
             # die Fusszeile in einer Schrift ohne Unicode-Zuordnung steht.
             re.compile(r"^(\d{1,3})$")]
_ZAHL = re.compile(r"^(\d{1,3})$")


def _fuss_html(zeilen: list) -> str:
    """Gedruckte Seitenzahl aus den letzten fuenf nichtleeren Zeilen eines Blocks.

    Fuenf statt drei, weil beide Emittenten hinter die Zahl noch
    Null-Breite-Zeichen und leere Tabellenzellen setzen; mit drei Zeilen
    blieben bei AstraZeneca die Seiten mit Fussnotenblock unbelegbar.
    """
    # TotalEnergies setzt Null-Breite-Zeichen auch ZWISCHEN Leerzeichen um die
    # Zahl ("\u200b \u200b18 \u200b"); daher alle Leer- und Null-Breite-Zeichen weg.
    for z in reversed([re.sub(r"[\s\u200b]+", "", z) for z in zeilen][-5:]):
        m = _ZAHL.match(z)
        if m:
            return m.group(1)
    return "?"


def zerlege_html(quelle: str, ziel: str) -> list:
    roh = open(quelle, encoding="utf-8", errors="replace").read()
    bloecke = max((re.split(m, roh) for m in _TRENNER), key=len)
    if len(bloecke) < 5:
        bloecke = [roh]
    texte = [text_aus(b) for b in bloecke]
    roh_seiten = folgefilter([_fuss_html(t.splitlines()) if t else "?" for t in texte])
    seiten = fuelle_luecken(roh_seiten)
    _schreibe(ziel, seiten, texte)
    return seiten, roh_seiten


def zerlege_pdf(quelle: str, ziel: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Zerlegt einen AZN-Geschaeftsbericht seitenweise und liest die gedruckte
        Seitenzahl aus der Fusszeile. Die gedruckte Seite und die PDF-Seite
        gehen um den Vorspann auseinander; geschlossen wird sie nie.

    Inputs:
        quelle (str): Pfad zur PDF-Datei
        ziel (str): Pfad zur zu schreibenden Textdatei

    Outputs:
        seiten (list): gefundene gedruckte Seitenzahlen
    --------------------------------------------------------------------------
    """
    info = subprocess.run(["pdfinfo", quelle], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    seiten, texte = [], []
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), quelle, "-"],
                           capture_output=True, text=True).stdout
        zeilen = [z.rstrip() for z in t.splitlines() if z.strip()]
        treffer = "?"
        for z in reversed(zeilen[-4:]):
            for muster in _PDF_FUSS:
                m = muster.search(z.strip())
                if m:
                    treffer = m.group(1)
                    break
            if treffer != "?":
                break
        seiten.append(treffer)
        texte.append("\n".join(zeilen))
    roh_seiten = folgefilter(seiten)
    seiten = fuelle_luecken(roh_seiten)
    _schreibe(ziel, seiten, texte)
    return seiten, roh_seiten


def fuelle_luecken(seiten: list) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Schliesst Luecken in der Seitenfolge, wo der Versatz zwischen Block-
        und Seitenzahl links und rechts der Luecke derselbe ist. Mehr als ein
        Dutzend Fusszeilenformen liegen ueber acht Jahrgaenge verteilt, und
        jede unerkannte macht ihre Seiten unbelegbar. Geschlossen wird nur,
        was die Paginierung selbst erzwingt: liegt Block i auf Seite n und
        Block i+k auf Seite n+k, dann traegt jeder Block dazwischen genau die
        Seite, die die Zaehlung verlangt. Wo die beiden Versaetze auseinander-
        gehen, bleibt die Luecke stehen - dort ist die Seite unbekannt und
        wird nicht geraten.

    Inputs:
        seiten (list): Seitenbezeichnungen je Block, "?" wo keine gefunden

    Outputs:
        gefuellt (list): dieselbe Liste, belegbare Luecken geschlossen
    --------------------------------------------------------------------------
    """
    aus = list(seiten)
    bekannt = [(i, int(s)) for i, s in enumerate(aus) if s.isdigit()]
    for (i, a), (j, b) in zip(bekannt, bekannt[1:]):
        if j - i == b - a and j - i > 1:
            for k in range(i + 1, j):
                aus[k] = str(a + (k - i))
    # Raender: vor dem ersten und nach dem letzten Treffer mit demselben Versatz
    if bekannt:
        i, a = bekannt[0]
        for k in range(i - 1, -1, -1):
            if a - (i - k) >= 1:
                aus[k] = str(a - (i - k))
        j, b = bekannt[-1]
        for k in range(j + 1, len(aus)):
            aus[k] = str(b + (k - j))
    return aus


def _schreibe(ziel: str, seiten: list, texte: list) -> None:
    aus = [f"=== SEITE {s} (Block {i}) ===\n{t}\n"
           for i, (s, t) in enumerate(zip(seiten, texte), start=1)]
    open(ziel, "w", encoding="utf-8").write("\n".join(aus))


def _bericht(seiten_roh: list, seiten: list) -> str:
    """Wie viele Seitenzahlen gelesen und wie viele aus der Zaehlung ergaenzt."""
    gelesen = sum(1 for s in seiten_roh if s.isdigit())
    ergaenzt = sum(1 for a, b in zip(seiten_roh, seiten) if not a.isdigit() and b.isdigit())
    return f"{gelesen} gelesen, {ergaenzt} ergaenzt"


def main(argv: list) -> int:
    refs = os.path.join(ROOT, "refs")
    dateien = argv or sorted(
        os.path.join(refs, f) for f in os.listdir(refs)
        if f.endswith((".htm", ".pdf")) and not f.startswith("_"))
    mangel = []
    for quelle in dateien:
        ziel = os.path.splitext(quelle)[0] + ".txt"
        alle, roh_seiten = (zerlege_pdf if quelle.endswith(".pdf") else zerlege_html)(quelle, ziel)
        seiten = [s for s in alle if s != "?"]
        zahlen = sorted({int(s) for s in seiten if s.isdigit()})
        fehlend = luecken(seiten)
        if not zahlen:
            befund = "ohne Paginierung, Beleg ueber Blocknummer"
        elif fehlend:
            befund = f"Seiten {zahlen[0]}–{zahlen[-1]}, {len(fehlend)} fehlend: {fehlend[:8]}"
            mangel.append(os.path.basename(quelle))
        else:
            befund = (f"Seiten {zahlen[0]}–{zahlen[-1]} lueckenlos, "
                      f"{_bericht(roh_seiten, alle)}")
        print(f"{os.path.basename(ziel):26s} {len(seiten):4d} Seiten, {befund}")
    if mangel:
        print(f"\n[SEITEN] Luecken in: {', '.join(mangel)}")
    return 1 if mangel else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
