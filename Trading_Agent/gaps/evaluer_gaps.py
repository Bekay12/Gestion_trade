#!/usr/bin/env python3
"""
evaluer_gaps - confronte une detection de gap au cours reel, apres coup.

Une detection qui n'est jamais notee ne sert a rien : elle devient une opinion
qu'on se rappelle quand elle avait raison. Ce script relit un fichier de
detection, va chercher ce que le titre a fait depuis, et note chaque verdict
contre ce que ce verdict PROMETTAIT.

Les quatre verdicts ne se jugent pas sur le meme critere :

    FADE          promet un comblement -> le gap s'est-il comble ?
    CONTINUATION  promet une tenue     -> le cours a-t-il tenu au-dessus ?
    SQUEEZE       promet plusieurs jours -> ou en est-on a J+3 ?
    PUMP_RISK     promet une chute     -> le titre est-il retombe ?

Un verdict sans donnee de cours n'est pas compte comme juste : il ressort
"incomplet". La regle du depot vaut ici aussi - une donnee absente ne vaut pas
feu vert, et elle ne vaut pas non plus succes.

Usage:
    python gaps/evaluer_gaps.py                          # la detection du jour
    python gaps/evaluer_gaps.py --fichier detections/2026-09-22_premarket.json
    python gaps/evaluer_gaps.py --jours 3                # fenetre d'evaluation
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

RACINE = os.path.dirname(os.path.abspath(__file__))
DETECTIONS = os.path.join(RACINE, "detections")

# Ce que chaque verdict promet, et donc ce qui le valide ou l'invalide.
PROMESSES = {
    "FADE":         "le gap se comble (retour au cours de cloture precedent)",
    "CONTINUATION": "le cours tient au-dessus du VWAP / du plus bas des 30 min",
    "SQUEEZE":      "le mouvement se poursuit sur plusieurs seances",
    "PUMP_RISK":    "le titre retombe, souvent sous son niveau d'avant-gap",
    "INSUFFISANT":  "aucune promesse : titre ecarte pour liquidite",
}


def dernier_fichier() -> str | None:
    fichiers = sorted(glob.glob(os.path.join(DETECTIONS, "*.json")))
    return fichiers[-1] if fichiers else None


def cours_depuis(tickers: list, depart: str, jours: int) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Cours quotidiens depuis la date de detection, en UN SEUL appel groupe.
        Le budget de requetes yfinance est une contrainte dure du depot.

    Inputs:
        tickers (list): symboles de la detection
        depart (str): date de detection, AAAA-MM-JJ
        jours (int): fenetre d'evaluation en seances

    Outputs:
        series (dict): {ticker: {"cloture": [...], "haut": [...], "bas": [...]}}
    --------------------------------------------------------------------------
    """
    import pandas as pd
    import yfinance as yf

    if not tickers:
        return {}
    lot = yf.download(tickers, start=depart, period=None, interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False,
                      threads=True)
    series = {}
    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            if d.empty:
                continue
            series[t] = {
                "dates":   [i.strftime("%Y-%m-%d") for i in d.index[:jours + 1]],
                "cloture": [float(x) for x in d["Close"].iloc[:jours + 1]],
                "haut":    [float(x) for x in d["High"].iloc[:jours + 1]],
                "bas":     [float(x) for x in d["Low"].iloc[:jours + 1]],
                "ouverture": [float(x) for x in d["Open"].iloc[:jours + 1]],
            }
        except Exception:
            continue
    return series


def noter(verdict: dict, serie: dict | None) -> dict:
    """Note un verdict contre sa propre promesse. Sans cours : incomplet."""
    classe = verdict.get("classe", "?")
    ticker = verdict.get("ticker", "?")
    if not serie or len(serie.get("cloture", [])) < 1:
        return {"ticker": ticker, "classe": classe, "resultat": "incomplet",
                "motif": "cours indisponible sur la fenetre",
                "variation_pct": None}

    ouverture = serie["ouverture"][0]
    cloture_j0 = serie["cloture"][0]
    dernier = serie["cloture"][-1]
    var_j0 = (cloture_j0 / ouverture - 1) * 100 if ouverture else None
    var_totale = (dernier / ouverture - 1) * 100 if ouverture else None

    if classe == "INSUFFISANT":
        return {"ticker": ticker, "classe": classe, "resultat": "non juge",
                "motif": "ecarte a la detection, aucune promesse",
                "variation_pct": round(var_totale, 2) if var_totale else None}

    if classe == "FADE":
        # Promesse : comblement. Le bas du jour revient-il sous l'ouverture ?
        comble = serie["bas"][0] <= ouverture * 0.995
        return {"ticker": ticker, "classe": classe,
                "resultat": "juste" if comble else "faux",
                "motif": ("gap comble dans la seance" if comble
                          else f"pas de comblement, cloture {var_j0:+.1f} % vs ouverture"),
                "variation_pct": round(var_j0, 2) if var_j0 is not None else None}

    if classe == "CONTINUATION":
        tenu = cloture_j0 >= ouverture
        return {"ticker": ticker, "classe": classe,
                "resultat": "juste" if tenu else "faux",
                "motif": (f"cloture {var_j0:+.1f} % au-dessus de l'ouverture" if tenu
                          else f"cloture {var_j0:+.1f} %, le gap n'a pas tenu"),
                "variation_pct": round(var_j0, 2) if var_j0 is not None else None}

    if classe == "SQUEEZE":
        # Promesse : plusieurs seances. Juge sur la fenetre entiere, pas sur J0.
        if len(serie["cloture"]) < 2:
            return {"ticker": ticker, "classe": classe, "resultat": "incomplet",
                    "motif": "moins de deux seances ecoulees",
                    "variation_pct": round(var_j0, 2) if var_j0 is not None else None}
        poursuivi = dernier >= ouverture
        return {"ticker": ticker, "classe": classe,
                "resultat": "juste" if poursuivi else "faux",
                "motif": (f"mouvement poursuivi, {var_totale:+.1f} % sur la fenetre"
                          if poursuivi else
                          f"retombe a {var_totale:+.1f} %, pas un squeeze"),
                "variation_pct": round(var_totale, 2) if var_totale else None}

    if classe == "PUMP_RISK":
        retombe = dernier < ouverture
        return {"ticker": ticker, "classe": classe,
                "resultat": "juste" if retombe else "faux",
                "motif": (f"retombe de {var_totale:+.1f} %, l'alerte etait fondee"
                          if retombe else
                          f"a tenu ({var_totale:+.1f} %) : alerte trop severe"),
                "variation_pct": round(var_totale, 2) if var_totale else None}

    return {"ticker": ticker, "classe": classe, "resultat": "non juge",
            "motif": f"classe inconnue: {classe}", "variation_pct": None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluation d'une detection de gaps")
    p.add_argument("--fichier", default=None, help="detection a evaluer")
    p.add_argument("--jours", type=int, default=3, help="fenetre en seances")
    args = p.parse_args(argv)

    chemin = args.fichier or dernier_fichier()
    if not chemin or not os.path.exists(chemin):
        print("Aucune detection a evaluer. Lancer d'abord une detection.")
        return 1

    brut = json.load(open(chemin))
    # Deux formes tolerees: l'enveloppe datee (forme courante) et la liste nue
    # des verdicts (premiers fichiers produits avant l'ajout de l'enveloppe).
    if isinstance(brut, list):
        detection = {"verdicts": brut, "mode": "?", "date": ""}
        # Sans enveloppe, la date vient du nom du fichier: AAAA-MM-JJ_mode.json
        base = os.path.basename(chemin)
        detection["date"] = base[:10] if base[:4].isdigit() else ""
    else:
        detection = brut
    verdicts = detection.get("verdicts", [])
    date_detection = detection.get("date", "")[:10]
    if not date_detection:
        print("Date de detection introuvable: impossible de borner la fenetre.")
        return 1
    tickers = [v.get("ticker") for v in verdicts if v.get("ticker")]

    print(f"Detection : {os.path.basename(chemin)}")
    print(f"Date      : {date_detection} ({detection.get('mode', '?')})")
    print(f"Titres    : {len(tickers)}")
    print(f"Fenetre   : {args.jours} seance(s)\n")

    series = cours_depuis(tickers, date_detection, args.jours)
    notes = [noter(v, series.get(v.get("ticker"))) for v in verdicts]

    largeur = max((len(n["ticker"]) for n in notes), default=6)
    for n in sorted(notes, key=lambda x: (x["resultat"], x["classe"])):
        var = f"{n['variation_pct']:+7.2f} %" if n["variation_pct"] is not None else "      --"
        print(f"  {n['ticker']:<{largeur}}  {n['classe']:<13} {n['resultat']:<10} "
              f"{var}   {n['motif']}")

    juges = [n for n in notes if n["resultat"] in ("juste", "faux")]
    justes = [n for n in juges if n["resultat"] == "juste"]
    incomplets = [n for n in notes if n["resultat"] == "incomplet"]
    print()
    if juges:
        print(f"Taux de justesse : {len(justes)}/{len(juges)} "
              f"({100*len(justes)/len(juges):.0f} %)")
    else:
        print("Aucun verdict jugeable sur cette fenetre.")
    if incomplets:
        print(f"Incomplets (cours manquant) : {len(incomplets)} - non comptes")

    sortie = chemin.replace(".json", f"_evalue_J{args.jours}.json")
    json.dump({"detection": os.path.basename(chemin), "evalue_le": datetime.now().isoformat(),
               "fenetre_jours": args.jours, "notes": notes},
              open(sortie, "w"), indent=1, ensure_ascii=False)
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
