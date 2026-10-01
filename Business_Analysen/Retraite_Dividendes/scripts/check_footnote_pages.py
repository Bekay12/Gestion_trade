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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "out", "analyse.pdf")

# Die Zitiermakros dieses Projekts (tache 10, preamble.tex): \Q{Dokument}{Seite}
# und \QL{Schluessel}{Dokument}{Seite} fuer Berichte/Dokumente,
# \QW{Titel}{Schluessel}{Datum} fuer eine Webquelle ohne quellen.json-Eintrag,
# \QH{cle} fuer eine Quelle aus data/quellen.json (Schluessel = cle selbst, die
# Fussnote ist bei gleicher cle IMMER wortgleich) und \QP{id} fuer eine
# Pressequelle aus data/presse.json (Schluessel = id).
# Ohne \QH/\QP in diesem Ausdruck fand der Pruefer nach Aufgabe 10 dauerhaft
# null Zitationen (die Kapitel zitieren fast ausschliesslich ueber \QH/\QP) und
# meldete schweigend "alles in Ordnung" - ein Wachhund, der nichts sieht, ist
# gefaehrlicher als keiner.
_CALL = re.compile(
    r"\\QL?(?:\{[^}]*\}(?=\{[^}]*\}\{))?\{([^}]*)\}\{([^}]*)\}"
    r"|\\QW\{[^}]*\}\{([^}]*)\}\{[^}]*\}"
    r"|\\QH\{([^}]*)\}"
    r"|\\QP\{([^}]*)\}"
)


# "%" non precede d'un backslash: debut de commentaire LaTeX.
_COMMENTAIRE = re.compile(r"(?<!\\)%")

def parse_citations(sections_dir: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Findet jeden \\quelleGB/\\quelleZF/\\quelleWeb-Aufruf in den Section-
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
                # Commentaire = "%" non echappe: un simple split("%") coupait aussi a
                # chaque "\\%" (pourcentage) et rendait invisibles les citations
                # placees apres un pourcentage sur la meme ligne (tache 10).
                code = _COMMENTAIRE.split(raw, maxsplit=1)[0]
                for m in _CALL.finditer(code):
                    dok, seite, web, quellen_cle, presse_id = m.groups()
                    if dok is not None:
                        key = f"{dok} S.{seite}"
                    elif web is not None:
                        key = f"Web {web}"
                    elif quellen_cle is not None:
                        key = f"QH {quellen_cle}"
                    else:
                        key = f"QP {presse_id}"
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
    if not citations:
        # Ein Wachhund, der nichts findet, meldet sonst schweigend "OK".
        print("FEHLER: keine Zitation gefunden - Makromuster pruefen.", file=sys.stderr)
        return 1
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
          "derselben gedruckten Seite. Fuer \\Q/\\QW: fuer alle bis auf die "
          "erste \\QR{<key>} statt \\Q/\\QL/\\QW einsetzen, an der ersten "
          "\\QL{<key>}{...}{...} anhaengen. Fuer \\QH/\\QP: entweder die "
          "Wiederholung streichen (die Fussnote ist ohnehin wortgleich) oder "
          "manuell ankern - \\footnote{\\label{fn:<key>}\\QHtexte{<cle>}} "
          "bzw. \\QPtexte{<id>}, dann an der Wiederholung \\QR{<key>} "
          "(siehe preamble.tex, Kommentar bei \\QH/\\QP).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
