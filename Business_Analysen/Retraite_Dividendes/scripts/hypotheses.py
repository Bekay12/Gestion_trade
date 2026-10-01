#!/usr/bin/env python3
"""
hypotheses.py - charge data/quellen.json, la source unique de toutes les hypotheses.

Chaque entree: {"wert": ..., "einheit": ..., "quelle": ..., "url": ..., "abgerufen":
"JJJJ-MM-TT", "seite": ... (optionnel), "primaer": true|false}. Aucune hypothese n'est
ecrite ailleurs dans le code: un module qui en a besoin appelle wert(schluessel).

Aufruf: python3 scripts/hypotheses.py --pruefen   (code 1 si une entree manque)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFAD = os.path.join(ROOT, "data", "quellen.json")


def alle() -> dict:
    return json.load(open(PFAD, encoding="utf-8"))


def wert(schluessel: str):
    e = alle().get(schluessel)
    return None if e is None else e.get("wert")


def pruefen() -> list:
    """Defauts: valeur absente, source absente, date absente."""
    fehler = []
    for k, e in alle().items():
        if e.get("wert") is None:
            fehler.append(f"{k}: wert fehlt")
        if not e.get("quelle") or not e.get("abgerufen"):
            fehler.append(f"{k}: quelle oder abgerufen fehlt")
    return fehler


if __name__ == "__main__":
    if "--pruefen" in sys.argv:
        f = pruefen()
        for x in f:
            print("[QUELLEN]", x)
        print(f"[QUELLEN] {len(alle()) - len(f)} Eintraege ok, {len(f)} Defekte")
        sys.exit(1 if f else 0)
