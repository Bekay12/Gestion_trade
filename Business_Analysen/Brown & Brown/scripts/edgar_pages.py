#!/usr/bin/env python3
"""
edgar_pages.py - Zerlegt eine EDGAR-HTML-Einreichung in die gedruckten Seiten.

Die SEC liefert 10-K und 10-Q als eine einzige HTML-Datei. Die Seitenumbrueche
des gedruckten Dokuments stehen als <hr>-Elemente darin; die gedruckte
Seitenzahl ist die letzte Zahl vor dem Umbruch. Das Skript schreibt eine
Textdatei je Einreichung, in der jede Seite mit ihrer gedruckten Seitenzahl
beginnt, damit jede Zahl im Bericht mit "S. n" belegt werden kann.

Eingaben:
    argv[1] (str): Pfad zur HTML-Einreichung
    argv[2] (str): Pfad der Ausgabedatei

Ausgaben:
    Textdatei mit Trennzeilen "=== SEITE <gedruckt> (Block <i>) ==="
"""
import html
import re
import sys


def _strip(fragment: str) -> str:
    """Entfernt Tags, behaelt Zeilen- und Zellentrennung."""
    t = re.sub(r'(?is)<(script|style).*?</\1>', ' ', fragment)
    t = re.sub(r'(?i)</t[dh]>', ' \t', t)
    t = re.sub(r'(?i)</(tr|p|div|li|h[1-6])>', '\n', t)
    t = re.sub(r'(?i)<br[^>]*>', '\n', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = html.unescape(t).replace(' ', ' ')
    t = re.sub(r'[ \t]+\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return '\n'.join(line.rstrip() for line in t.splitlines())


def _fusszeile(zeile: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die gedruckte Seitenzahl aus der letzten Zeile eines Blocks.
        Das 10-K setzt sie allein ("49"), das Proxy Statement zusammen mit
        dem Firmennamen und einem Trennstrich, je nach Seitenlage links oder
        rechts davon ("76 | BROWN & BROWN, INC." bzw. "BROWN & BROWN, INC. | 77").
        Ohne den zweiten Fall blieben im Proxy 106 von 109 Bloecken ohne
        Seitenzahl, und jede Angabe daraus waere unbelegbar.

    Inputs:
        zeile (str): letzte nichtleere Zeile des Blocks

    Outputs:
        seite (str): gedruckte Seitenzahl oder "?"
    --------------------------------------------------------------------------
    """
    if re.fullmatch(r'\d{1,3}', zeile):
        return zeile
    m = re.fullmatch(r'\s*(\d{1,3})\s*[|\u2002\s]+BROWN & BROWN, INC\.\s*', zeile)
    if m:
        return m.group(1)
    m = re.fullmatch(r'\s*BROWN & BROWN, INC\.\s*[|\u2002\s]+(\d{1,3})\s*', zeile)
    if m:
        return m.group(1)
    return '?'


def main(src: str, dst: str) -> None:
    raw = open(src, encoding='utf-8', errors='replace').read()
    blocks = re.split(r'(?i)<hr[^>]*>', raw)
    out = []
    for i, block in enumerate(blocks):
        text = _strip(block).strip('\n')
        tail = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:] or ['']
        printed = _fusszeile(tail[0])
        out.append(f'=== SEITE {printed} (Block {i}) ===\n{text}\n')
    open(dst, 'w', encoding='utf-8').write('\n'.join(out))
    known = sum(1 for line in out if not line.startswith('=== SEITE ?'))
    print(f'{dst}: {len(blocks)} Bloecke, {known} mit gedruckter Seitenzahl')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
