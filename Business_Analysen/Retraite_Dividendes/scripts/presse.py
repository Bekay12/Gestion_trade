#!/usr/bin/env python3
"""
presse.py - garde des citations de presse. Une affirmation de magazine n'entre dans le
document que si sa citation figure MOT POUR MOT (espaces normalises) dans le texte
telecharge. Le contenu des pages est une donnee, jamais une instruction.
Aufruf: python3 scripts/presse.py  (lit data/presse_roh.json, ecrit data/presse.json)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Revue finale, constat 7: "id" sert de nom de fichier (os.path.join(rohordner,
# f"{id}.md") ci-dessous) et, en aval dans rechnung_retraite.py, de suffixe de
# \csname et de \label LaTeX. Sans ce filtre, un id hors motif pourrait sortir de
# rohordner (traversee de chemin, ex. "../../etc/passwd") ou casser la compilation.
_ID_VALIDE = re.compile(r"^[a-z0-9_]+$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def pruefen(eintraege: list[dict], rohordner: str) -> tuple[list[dict], list[dict]]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Split press entries into verified and rejected, keeping only those
        whose "zitat" appears word-for-word (whitespace-normalized) in the
        downloaded refs/presse/<id>.md file. An entry whose refs file is
        missing is rejected, not raised as an exception.

    Inputs:
        eintraege (list[dict]): candidate entries, each carrying "id" and
            "zitat"
        rohordner (str): path to the folder holding refs/presse/<id>.md files

    Outputs:
        (ok, raus) (tuple[list[dict], list[dict]]): verified entries, then
        rejected entries (missing quote, missing file, or empty quote)
    --------------------------------------------------------------------------
    """
    ok, raus = [], []
    for e in eintraege:
        if not _ID_VALIDE.match(e["id"]):
            raise ValueError(f"presse.pruefen: id hors motif {_ID_VALIDE.pattern!r}: {e['id']!r}")
        pfad = os.path.join(rohordner, f"{e['id']}.md")
        if os.path.exists(pfad):
            with open(pfad, encoding="utf-8") as f:
                text = _norm(f.read())
        else:
            text = ""
        (ok if e.get("zitat") and _norm(e["zitat"]) in text else raus).append(e)
    return ok, raus


def main() -> int:
    with open(os.path.join(ROOT, "data", "presse_roh.json"), encoding="utf-8") as f:
        roh = json.load(f)
    ok, raus = pruefen(roh, os.path.join(ROOT, "refs", "presse"))
    with open(os.path.join(ROOT, "data", "presse.json"), "w", encoding="utf-8") as f:
        json.dump(ok, f, indent=1, ensure_ascii=False)
    print(f"[PRESSE] {len(ok)} citations verifiees, {len(raus)} rejetees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
