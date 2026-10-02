#!/usr/bin/env python3
"""
edgar_seiten.py - zerlegt die EDGAR-Einreichungen von Alamos in gedruckte Seiten.

Die SEC liefert jeden Anhang als eine einzige HTML-Datei. Die Seitenumbrueche
des gedruckten Dokuments stehen als <hr>-Elemente darin; die gedruckte
Seitenzahl steht in der Fusszeile davor. Sie wird GELESEN und nie aus der
Blocknummer geschlossen: eine aus dem Zusammenhang geratene Seitenzahl faellt
erst auf, wenn ein Leser die Quelle oeffnet.

Alamos setzt die Fusszeile je Dokumentart verschieden:

    MD&A                "12"                        blanke Zahl
    Abschluss           "12" / "Alamos Gold Inc."   Zahl, dann Firmenname
    AIF                 "12 | Alamos Gold Inc."     Zahl, Strich, Firmenname
    Circular            "12 | 2026 Management Information Circular"
    Circular, Vorspann  "IV"                        roemische Zahl

Ausgabe: refs/<name>.txt mit Trennzeilen "=== SEITE <gedruckt> (Block <i>) ===".
Der Deckungsgrad wird je Datei gemeldet; er ist die Warnung davor, dass eine
unbekannte Fusszeilenform jede Angabe daraus unbelegbar macht.

Aufruf: python3 scripts/edgar_seiten.py [refs/name.htm ...]
"""
import html
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")

# Formen der Fusszeile. Die Reihenfolge ist unerheblich; jede Zeile der
# letzten drei wird gegen alle geprueft.
# Das Trennzeichen zwischen Zahl und Firmenname wechselt zwischen "|",
# Halbgeviertstrich und Bindestrich; ab 2018 steht es gar nicht mehr da.
_TRENN = r"[|–—-]"
_NAME = r"(?:ALAMOS GOLD INC\.?|\d{4} Management Information Circular)"
# Roemische Seitenzahlen tragen der Vorspann des Circular und die AIF
# 2016/2017. Gross- und Kleinschreibung wechseln zwischen beiden.
_ROM = r"(?:x{0,3}(?:ix|iv|v?i{1,3}|v))"
_FUSS = [
    re.compile(rf"^(\d{{1,3}}|{_ROM})$", re.I),
    # "1 |" als eigene Zeile, der Firmenname folgt erst in der naechsten -
    # so setzen die AIF 2016 und 2017 ihre Fusszeile. Ohne diesen Fall
    # blieben dort 135 von 137 Bloecken ohne Seitenzahl.
    re.compile(rf"^(\d{{1,3}}|{_ROM})\s*{_TRENN}\s*$", re.I),
    re.compile(rf"^(\d{{1,3}}|{_ROM})\s*{_TRENN}?\s*{_NAME}$", re.I),
    re.compile(rf"^{_NAME}\s*{_TRENN}?\s*(\d{{1,3}}|{_ROM})$", re.I),
]


def text_aus(block: str) -> str:
    """Entfernt Tags, behaelt Zeilen- und Zellentrennung."""
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", block)
    t = re.sub(r"(?i)</t[dh]>", " \t", t)
    t = re.sub(r"(?i)</(tr|p|div|li|h[1-6])>", "\n", t)
    t = re.sub(r"(?i)<br[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = html.unescape(t).replace(" ", " ").replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(l.strip() for l in t.splitlines() if l.strip())


def fusszeile(zeilen: list) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die gedruckte Seitenzahl aus den letzten drei nichtleeren Zeilen
        eines Blocks. Drei statt einer, weil der Abschluss hinter die Zahl
        noch den Firmennamen setzt und die Zahl dann nicht mehr die letzte
        Zeile ist; nur die letzte zu pruefen liess bei Honeywell in der
        Schwesteranalyse 0 % der Seiten zitierbar.

    Inputs:
        zeilen (list): nichtleere Zeilen des Blocks

    Outputs:
        seite (str): gedruckte Seitenzahl, roemische Zahl oder "?"
    --------------------------------------------------------------------------
    """
    letzte = [z.strip() for z in zeilen][-3:]
    for zeile in reversed(letzte):
        for muster in _FUSS:
            m = muster.match(zeile)
            if m:
                return m.group(1)
    return "?"


def folgefilter(kandidaten: list) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Verwirft Fusszeilentreffer, die nicht in die Seitenfolge des Dokuments
        passen. Eine Fusszeilenform allein genuegt nicht: In der AIF 2017 steht
        in den letzten drei Zeilen eines Blocks die Reservenzahl "452", und sie
        erfuellt jedes Muster fuer eine blanke Seitenzahl. Als Seite gelesen
        machte sie aus "Seiten 1 bis 68" ein "Seiten 1 bis 452" und haette
        jeden Beleg aus diesem Block an die falsche Seite gehaengt.

        Eine gedruckte Seitenzahl ist daran erkennbar, dass sie die vorige um
        genau eins erhoeht (oder sie wiederholt, wenn eine Seite auf zwei
        Bloecke faellt). Gesucht wird deshalb die laengste solche Kette; was
        ausserhalb liegt, wird verworfen und nicht belegt.

    Inputs:
        kandidaten (list): Seitenbezeichnungen je Block, "?" wo keine gefunden

    Outputs:
        gefiltert (list): dieselbe Liste, Ausreisser durch "?" ersetzt
    --------------------------------------------------------------------------
    """
    stellen = [(i, int(s)) for i, s in enumerate(kandidaten) if s.isdigit()]
    if not stellen:
        return list(kandidaten)
    # laengste Kette dynamisch: beste[j] = Laenge der Kette, die bei j endet
    beste = [1] * len(stellen)
    vor = [-1] * len(stellen)
    for j in range(len(stellen)):
        for k in range(j):
            if stellen[j][1] - stellen[k][1] in (0, 1) and beste[k] + 1 > beste[j]:
                beste[j], vor[j] = beste[k] + 1, k
    j = max(range(len(stellen)), key=lambda x: beste[x])
    kette = set()
    while j != -1:
        kette.add(stellen[j][0])
        j = vor[j]
    return [s if (i in kette or not s.isdigit()) else "?"
            for i, s in enumerate(kandidaten)]


def zerlege(quelle: str, ziel: str) -> list:
    """Schreibt <ziel> und gibt die gefundenen gedruckten Seiten zurueck."""
    roh = open(quelle, encoding="utf-8", errors="replace").read()
    bloecke = re.split(r"(?i)<hr\b[^>]*>", roh)
    texte = [text_aus(b) for b in bloecke]
    roh_seiten = [fusszeile(t.splitlines()) if t else "?" for t in texte]
    seiten = folgefilter(roh_seiten)
    aus = [f"=== SEITE {s} (Block {i}) ===\n{t}\n"
           for i, (s, t) in enumerate(zip(seiten, texte), start=1)]
    open(ziel, "w", encoding="utf-8").write("\n".join(aus))
    return [s for s in seiten if s != "?"]


def luecken(seiten: list) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt die arabischen Seitenzahlen zurueck, die zwischen der ersten und
        der letzten gefundenen fehlen. Das ist das richtige Mass fuer die
        Deckung, und der Anteil der Bloecke mit Seitenzahl ist es nicht: die
        MD&A bis 2019 setzen je gedruckter Seite ZWEI durch <hr> getrennte
        Bloecke, einen fuer die Kopfzeile und einen fuer den Inhalt. Nur der
        zweite traegt die Fusszeile, weshalb die Blockquote dort bei 58 %
        stand, obwohl die Seiten 3 bis 49 lueckenlos vorlagen. Gemessen wird,
        was gebraucht wird: ob jede Seite zitierbar ist.

    Inputs:
        seiten (list): gefundene Seitenbezeichnungen, arabisch und roemisch

    Outputs:
        fehlend (list): fehlende arabische Seitenzahlen des Intervalls
    --------------------------------------------------------------------------
    """
    zahlen = sorted({int(s) for s in seiten if s.isdigit()})
    if not zahlen:
        return []
    return [n for n in range(zahlen[0], zahlen[-1] + 1) if n not in zahlen]


def main(argv: list) -> int:
    dateien = argv or sorted(
        os.path.join(REFS, f) for f in os.listdir(REFS) if f.endswith(".htm"))
    mangel = []
    for quelle in dateien:
        ziel = quelle[:-4] + ".txt"
        seiten = zerlege(quelle, ziel)
        zahlen = sorted({int(s) for s in seiten if s.isdigit()})
        fehlend = luecken(seiten)
        if not zahlen:
            # Pressemitteilungen tragen keine Paginierung. Das ist kein
            # Mangel des Zerlegers; solche Quellen werden ueber die
            # Blocknummer belegt, nicht ueber eine Seite.
            befund = "ohne Paginierung, Beleg ueber Blocknummer"
        elif fehlend:
            befund = (f"Seiten {zahlen[0]}\u2013{zahlen[-1]}, "
                      f"{len(fehlend)} fehlend: {fehlend[:8]}")
            mangel.append(os.path.basename(quelle))
        else:
            befund = f"Seiten {zahlen[0]}\u2013{zahlen[-1]} lueckenlos"
        print(f"{os.path.basename(ziel):28s} {len(seiten):3d} Seiten, {befund}")
    if mangel:
        print(f"\n[SEITEN] Luecken in: {', '.join(mangel)}")
    return 1 if mangel else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
