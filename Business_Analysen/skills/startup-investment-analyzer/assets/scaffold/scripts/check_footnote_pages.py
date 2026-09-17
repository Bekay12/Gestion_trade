#!/usr/bin/env python3
"""
check_footnote_pages.py - findet Quelle+Seite, die mehrfach auf derselben
gedruckten PDF-Seite zitiert wird, ohne die Fussnotennummer wiederzu-
verwenden (\\quelleMerken/\\quelleErneut in preamble.tex).

Zwei \\quelleGB{2024}{10}-Aufrufe auf derselben Seite erzeugen sonst zwei
verschiedene, aber wortgleiche Fussnoten. Die tatsaechliche PDF-Seite eines
Zitationsaufrufs haengt vom Seitenumbruch ab und ist daher nicht aus dem
Quelltext ablesbar; sie wird ueber SyncTeX abgefragt (Ground Truth, nicht
geschaetzt).

Aufruf: python3 scripts/check_footnote_pages.py
Voraussetzung: ./build.sh mit -synctex=1 (siehe build.sh) wurde bereits
ausgefuehrt, out/analyse.synctex.gz existiert.
Rueckgabewert 1, wenn ein unbehandeltes Duplikat gefunden wurde.
"""
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projekt import (ZITATE_ZWEISTELLIG, ZITATE_EINSTELLIG,  # noqa: E402
                     ZITATE_DREISTELLIG)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "out", "analyse.pdf")

# Die Zitiermakros stehen in scripts/projekt.py, weil sie von Emittent zu
# Emittent wechseln: Wer Form 10-K einreicht, braucht andere als wer Form 40-F
# einreicht. Ohne einen passenden Ausdruck findet der Pruefer null Zitationen
# und meldet schweigend "alles in Ordnung" - ein Wachhund, der nichts sieht,
# ist gefaehrlicher als keiner. Deshalb bricht main() ab, wenn gar keine
# Zitation gefunden wird.
_CALL = re.compile(
    r"\\quelle(" + "|".join(ZITATE_ZWEISTELLIG + ZITATE_DREISTELLIG)
    + r")\{([^}]*)\}\{([^}]*)\}"
    r"|\\quelle(" + "|".join(ZITATE_EINSTELLIG) + r")\{([^}]*)\}")


def parse_citations(sections_dir: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Findet jeden \\quelleGB/\\quelleQM/\\quelleWeb-Aufruf in den Section-
        Dateien. Reine Textanalyse, kein LaTeX-Lauf noetig.

    Inputs:
        sections_dir (str): Pfad zu sections/.

    Outputs:
        entries (list[tuple[str, str, int]]): (Schluessel, Datei relativ zu
            sections_dir, Zeilennummer 1-basiert) je Aufruf.
    --------------------------------------------------------------------------
    """
    entries = []
    for path in sorted(glob.glob(os.path.join(sections_dir, "*.tex"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                code = raw.split("%")[0]
                for m in _CALL.finditer(code):
                    art, a, b, art1, seite = m.groups()
                    if art1:
                        # Einstellige Makros: Dokument steht fest, nur die
                        # Seite unterscheidet.
                        entries.append((f"{art1} S.{seite}", name, lineno))
                        continue
                    if art in ("MDA", "FS"):
                        key = f"{art} {a} S.{b}"
                    else:
                        # Der Schluessel muss die Webquelle identifizieren,
                        # nicht nur ihre Art: Zwei \quelleWeb-Aufrufe auf
                        # dieselbe Seite sind nur dann ein Duplikat, wenn sie
                        # denselben Fussnotentext drucken - also gleicher
                        # Titel UND gleiche Adresse. Frueher stand hier ein
                        # fester Schluessel; solange das Dokument genau eine
                        # Webquelle hatte, fiel das nicht auf.
                        key = f"Web {a} <{b}>"
                    entries.append((key, name, lineno))
    return entries


def group_duplicates(dated_entries: list) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gruppiert Zitationen nach (PDF-Seite, Schluessel) und behaelt nur
        Gruppen mit mehr als einem Vorkommen. Reine Funktion, kein I/O -
        die einzige gut isoliert testbare Stelle in diesem Skript.

    Inputs:
        dated_entries (list[tuple[int, str, str, int]]): (PDF-Seite,
            Schluessel, Datei, Zeile) je Aufruf.

    Outputs:
        duplicates (dict[int, dict[str, list[tuple[str, int]]]]): je
            PDF-Seite die Schluessel mit mehr als einem Fundort.
    --------------------------------------------------------------------------
    """
    by_page = defaultdict(lambda: defaultdict(list))
    for page, key, name, lineno in dated_entries:
        by_page[page][key].append((name, lineno))
    return {
        page: {k: v for k, v in keys.items() if len(v) > 1}
        for page, keys in by_page.items()
        if any(len(v) > 1 for v in keys.values())
    }


def resolve_page(sections_dir: str, name: str, lineno: int) -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        Fragt per SyncTeX die tatsaechlich gerenderte PDF-Seite einer
        Quelltextzeile ab. Erfordert eine mit -synctex=1 gebaute PDF.

    Inputs:
        sections_dir (str): Pfad zu sections/ (fuer den relativen Dateinamen).
        name (str): Dateiname innerhalb sections/.
        lineno (int): Zeilennummer 1-basiert.

    Outputs:
        page (int): PDF-Seitenzahl, oder -1 wenn nicht auffindbar.
    --------------------------------------------------------------------------
    """
    rel = os.path.relpath(os.path.join(sections_dir, name), ROOT)
    out = subprocess.run(
        ["synctex", "view", "-i", f"{lineno}:1:{rel}", "-o", PDF],
        capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"^Page:(\d+)", out, re.M)
    return int(m.group(1)) if m else -1


def main() -> int:
    if not os.path.exists(PDF):
        print(f"FEHLER: {PDF} fehlt - zuerst ./build.sh ausfuehren.", file=sys.stderr)
        return 1
    sections_dir = os.path.join(ROOT, "sections")
    citations = parse_citations(sections_dir)
    dated = [
        (resolve_page(sections_dir, name, lineno), key, name, lineno)
        for key, name, lineno in citations
    ]
    duplicates = group_duplicates(dated)
    if not duplicates:
        print(f"OK - keine unbehandelten Fussnoten-Duplikate "
              f"({len(citations)} Zitationen geprueft).")
        return 0
    for page in sorted(duplicates):
        print(f"PDF-Seite {page}:")
        for key, locs in duplicates[page].items():
            orte = ", ".join(f"{n}:{l}" for n, l in locs)
            print(f"  {key} x{len(locs)} -> {orte}")
    print("\nDiese Stellen zitieren dieselbe Quelle+Seite mehrfach auf "
          "derselben gedruckten Seite. Fuer alle bis auf die erste "
          "\\quelleErneut{<key>} statt \\quelleGB/QM/Web einsetzen; an der "
          "ersten \\quelleMerken{<key>} anhaengen (siehe preamble.tex).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
