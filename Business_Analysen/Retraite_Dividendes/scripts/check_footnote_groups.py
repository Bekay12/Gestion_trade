#!/usr/bin/env python3
r"""
check_footnote_groups.py - prueft, ob jede \quelleErneut{key} auf derselben
gedruckten PDF-Seite steht wie das zugehoerige \quelleMerken{key}.

Gegenstueck zu check_footnote_pages.py: jenes findet Duplikate, die noch
zusammengefuehrt werden muessen, dieses findet Zusammenfuehrungen, die durch
einen verschobenen Seitenumbruch ungueltig geworden sind. Liegt die
Wiederholung auf einer anderen Seite als die Primaerzitation, zeigt das
\footnotemark auf eine Fussnote, die dort nicht abgedruckt ist.

Aufruf: python3 scripts/check_footnote_groups.py
Voraussetzung: ./build.sh (setzt -synctex=1) wurde bereits ausgefuehrt.
Rueckgabewert 1, wenn eine Gruppe ueber eine Seitengrenze laeuft.

Tache 10 (preamble.tex): \QH{cle} und \QP{id} (data/quellen.json bzw.
data/presse.json) setzen selbst KEIN \label - jede Fussnote ist wortgleich
und wird bewusst neu gedruckt. Wer eine \QH/\QP-Wiederholung auf derselben
Seite dennoch buendeln will, ankert manuell mit dem bestehenden Mechanismus:
\footnote{\label{fn:<anker>}\QHtexte{<cle>}} (bzw. \QPtexte{<id>}) fuer die
erste Nennung, \QR{<anker>} fuer die Wiederholung. Ein roh im Fliesstext
geschriebenes \label{fn:...} wird deshalb wie ein \QL-Anker erkannt (zweite
Alternative in _ANCHOR), ohne dass dafuer ein eigenes Makro noetig ist.
"""
import glob
import os
import re
import sys
from collections import defaultdict

from check_footnote_pages import PDF, ROOT, resolve_page

_ANCHOR = re.compile(r"\\QL\{([^}]*)\}|\\label\{fn:([^}]*)\}")
_REPEAT = re.compile(r"\\QR\{([^}]*)\}")


# "%" non precede d'un backslash: debut de commentaire LaTeX.
_COMMENTAIRE = re.compile(r"(?<!\\)%")

def parse_groups(sections_dir: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Sammelt je Schluessel alle \\quelleMerken- und \\quelleErneut-Aufrufe
        in den Section-Dateien. Reine Textanalyse, kein LaTeX-Lauf noetig.

    Inputs:
        sections_dir (str): Pfad zu sections/.

    Outputs:
        groups (dict[str, list[tuple[str, str, int]]]): je Schluessel die
            Aufrufe als (Art "Merken"/"Erneut", Datei, Zeile 1-basiert).
    --------------------------------------------------------------------------
    """
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sections_dir, "*.tex"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                # Commentaire = "%" non echappe: un simple split("%") coupait aussi a
                # chaque "\\%" (pourcentage) et rendait invisibles les citations
                # placees apres un pourcentage sur la meme ligne (tache 10).
                code = _COMMENTAIRE.split(raw, maxsplit=1)[0]
                for m in _ANCHOR.finditer(code):
                    ql_key, label_key = m.groups()
                    groups[ql_key if ql_key is not None else label_key].append(
                        ("QL", name, lineno))
                for m in _REPEAT.finditer(code):
                    groups[m.group(1)].append(("QR", name, lineno))
    return groups


def find_offset_groups(dated_groups: dict) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Bestimmt je Schluessel, ob alle Aufrufe auf der Seite des
        \\quelleMerken-Ankers liegen. Reine Funktion, kein I/O - die
        testbare Stelle dieses Skripts.

    Inputs:
        dated_groups (dict[str, list[tuple[str, str, int, int]]]): je
            Schluessel die Aufrufe als (Art, Datei, Zeile, PDF-Seite).

    Outputs:
        fehler (dict[str, tuple[int, list]]): je fehlerhaftem Schluessel die
            Ankerseite (oder -1 ohne Anker) und die abweichenden Aufrufe.
    --------------------------------------------------------------------------
    """
    fehler = {}
    for key, calls in dated_groups.items():
        anker = [c for c in calls if c[0] == "QL"]
        if not anker:
            fehler[key] = (-1, list(calls))
            continue
        seite = anker[0][3]
        abweichend = [c for c in calls if c[3] != seite]
        if abweichend:
            fehler[key] = (seite, abweichend)
    return fehler


def main() -> int:
    if not os.path.exists(PDF):
        print(f"FEHLER: {PDF} fehlt - zuerst ./build.sh ausfuehren.", file=sys.stderr)
        return 1
    sections_dir = os.path.join(ROOT, "sections")
    groups = parse_groups(sections_dir)
    dated = {
        key: [(art, name, lineno, resolve_page(sections_dir, name, lineno))
              for art, name, lineno in calls]
        for key, calls in groups.items()
    }
    fehler = find_offset_groups(dated)
    if not fehler:
        print(f"OK - alle {len(groups)} Fussnotengruppen stehen je auf einer Seite.")
        return 0
    for key in sorted(fehler):
        seite, abweichend = fehler[key]
        orte = ", ".join(f"{a}@{n}:{l}=S.{p}" for a, n, l, p in abweichend)
        if seite == -1:
            print(f"{key}: kein \\quelleMerken gefunden -> {orte}")
        else:
            print(f"{key}: Anker auf S.{seite}, aber {orte}")
    print("\nDiese \\quelleErneut-Aufrufe verweisen auf eine Fussnote, die auf "
          "einer anderen Seite steht. Die Gruppe aufteilen: an der ersten "
          "Zitation der neuen Seite wieder \\quelleGB/QM/Web mit eigenem "
          "\\quelleMerken setzen (siehe sections/CLAUDE.md).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
