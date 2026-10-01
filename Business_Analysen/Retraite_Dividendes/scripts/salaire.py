#!/usr/bin/env python3
"""
salaire.py - impot sur le revenu (§32a EStG), salaire net approche, trajectoire de carriere.

Approximation declaree (Annahme dans le chapitre 5): revenu imposable = brut - part
salariale des cotisations sociales - forfait de frais professionnels 1 230 EUR; le Soli
est nul sous sa Freigrenze. L'ecart au calculateur officiel est mesure une fois et publie
dans docs/salaire_controle.md.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WERBUNGSKOSTEN = 1230.0


def est_32a(zve: float, tarif: dict) -> int:
    """Impot sur le revenu selon §32a EStG.

    tarif["zonen"] est une liste de lignes [bis, typ, a, b, c]: bis est la borne
    superieure de la zone (None pour la derniere), typ in {"null", "y", "z", "lin"}.
    Zones "y"/"z": avec t = (x - borne_inferieure)/10000, ESt = floor((a*t + b)*t + c).
    Zone "lin": ESt = floor(a*x + b), b deja negatif (convention data/quellen.json).
    """
    x = math.floor(zve)
    unten = None
    for bis, typ, a, b, c in tarif["zonen"]:
        if bis is None or x <= bis:
            if typ == "null":
                return 0
            if typ in ("y", "z"):
                basis = unten
                t = (x - basis) / 10000
                return math.floor((a * t + b) * t + c)
            return math.floor(a * x + b)
        unten = bis
    raise ValueError("tarif incomplet")


def sv_anteil(brutto: float, sv: dict) -> float:
    """Part salariale des cotisations sociales (RV+AV, KV, PV) sur un brut annuel."""
    rv = min(brutto, sv["bbg_rv_jahr"]) * (sv["rv"] + sv["av"])
    kv_basis = min(brutto, sv["bbg_kv_jahr"])
    kv = kv_basis * (sv["kv_allgemein"] + sv["kv_zusatz"]) / 2
    pv = kv_basis * (sv["pv"] / 2 + sv["pv_kinderlos_zuschlag"])
    return rv + kv + pv


def netto_jahr(brutto: float, tarif: dict, sv: dict, soli_freigrenze: float) -> float:
    """Salaire net annuel approche: brut moins cotisations sociales, impot et Soli."""
    abgaben = sv_anteil(brutto, sv)
    zve = max(0.0, brutto - abgaben - WERBUNGSKOSTEN)
    est = est_32a(zve, tarif)
    soli = 0.055 * est if est > soli_freigrenze else 0.0
    return brutto - abgaben - est - soli


def trajektorie(start_jahr: int = 2027, jahre: int = 45) -> list[dict]:
    """Trajectoire de carriere en euros de 2026 (reel): brut, net, net mensuel par annee."""
    import hypotheses as h
    tarif, sv = h.wert("est_tarif_2026"), h.wert("sv_arbeitnehmer")
    g0, wachstum = h.wert("einstiegsgehalt_brutto"), h.wert("gehaltssteigerung_real")
    sfg = h.wert("soli_freigrenze_2026")
    aus: list[dict] = []
    for i in range(jahre):
        brutto = g0 * (1 + wachstum) ** i          # euros de 2026, bareme 2026 fige
        netto = netto_jahr(brutto, tarif, sv, sfg)
        aus.append({"jahr": start_jahr + i, "brutto": brutto, "netto": netto,
                    "netto_monat": netto / 12})
    return aus
