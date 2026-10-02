#!/usr/bin/env python3
"""
evaluer_gaps - confronte une detection de gap au cours reel, apres coup.

Une detection qui n'est jamais notee ne sert a rien : elle devient une opinion
qu'on se rappelle quand elle avait raison. Ce script relit un fichier de
detection, va chercher ce que le titre a fait depuis, et note chaque verdict
contre ce que ce verdict PROMETTAIT.

Les verdicts ne se jugent pas sur le meme critere :

    FADE          promet un comblement   -> le gap s'est-il comble ?
    CONTINUATION  promet une tenue       -> le cours a-t-il tenu au-dessus ?
    SQUEEZE       promet plusieurs jours -> ou en est-on a J+3 ?
    PUMP_RISK     promet une chute       -> le titre est-il retombe ?
    A_CONFIRMER   ne promet rien         -> le gap a-t-il tenu, ou l'attente
                                            a-t-elle evite une entree ?

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from executabilite import CLASSES_VENDEUSES, barres_ouverture, bilan_shorts, juger  # noqa: E402

RACINE = os.path.dirname(os.path.abspath(__file__))
DETECTIONS = os.path.join(RACINE, "detections")

# Ce que chaque verdict promet, et donc ce qui le valide ou l'invalide.
PROMESSES = {
    "FADE":         "le gap se comble (retour au cours de cloture precedent)",
    "CONTINUATION": "le cours tient au-dessus du VWAP / du plus bas des 30 min",
    "SQUEEZE":      "le mouvement se poursuit sur plusieurs seances",
    "PUMP_RISK":    "le titre retombe, souvent sous son niveau d'avant-gap",
    "INSUFFISANT":  "aucune promesse : titre ecarte pour liquidite",
    "A_CONFIRMER":  "rien n'est promis : le verdict attend la regle des 30 min",
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

    from datetime import date, timedelta

    if not tickers:
        return {}
    # Dix jours calendaires en arriere: la cloture de la veille est la reference de
    # la promesse FADE, et un week-end ou un jour ferie la repousse de plusieurs jours.
    avant = (date.fromisoformat(depart) - timedelta(days=10)).isoformat()
    lot = yf.download(tickers, start=avant, period=None, interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False,
                      threads=True)
    series = {}
    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            anterieur = d[d.index.strftime("%Y-%m-%d") < depart]
            d = d[d.index.strftime("%Y-%m-%d") >= depart]
            if d.empty:
                continue
            series[t] = {
                "veille":  float(anterieur["Close"].iloc[-1]) if len(anterieur) else None,
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
        # Promesse: le gap se comble, c'est-a-dire retour a la CLOTURE DE LA VEILLE.
        # Jusqu'au 28.09.2026 le test etait "bas <= ouverture x 0,995": un repli de
        # 0,5 % sous l'ouverture suffisait, et ARAY (clos +36,3 % sur la veille) comme
        # SRFM (clos +5,3 % sur l'ouverture) ressortaient "juste". Tolerance de 0,5 %
        # conservee, mais sur la bonne reference. La variation rapportee est celle
        # d'une vente a l'ouverture rachetee a la cloture, signe inverse.
        veille = serie.get("veille")
        if not veille:
            return {"ticker": ticker, "classe": classe, "resultat": "incomplet",
                    "motif": "cloture de la veille indisponible",
                    "variation_pct": round(var_j0, 2) if var_j0 is not None else None}
        comble = serie["bas"][0] <= veille * 1.005
        vente = -var_j0 if var_j0 is not None else None
        return {"ticker": ticker, "classe": classe,
                "resultat": "juste" if comble else "faux",
                "motif": (f"gap comble (bas {serie['bas'][0]:.3g} <= veille {veille:.3g}); "
                          f"vente ouverture->cloture {vente:+.1f} %" if comble else
                          f"gap non comble (bas {serie['bas'][0]:.3g} > veille {veille:.3g}); "
                          f"vente ouverture->cloture {vente:+.1f} %"),
                "variation_pct": round(var_j0, 2) if var_j0 is not None else None,
                "vente_ouverture_cloture_pct": round(vente, 2) if vente is not None else None}

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

    if classe == "A_CONFIRMER":
        # Ce verdict ne promet pas une direction, il promet qu'une REQUALIFICATION
        # etait necessaire. On le note donc sur ce que la seance a revele: le gap
        # a-t-il tenu (auquel cas la requalification etait justifiee) ou s'est-il
        # comble d'emblee (auquel cas l'attente a evite une entree perdante) ?
        tenu = cloture_j0 >= ouverture
        return {"ticker": ticker, "classe": classe,
                "resultat": "tenu" if tenu else "comble",
                "motif": (f"gap tenu, cloture {var_j0:+.1f} % : requalification justifiee"
                          if tenu else
                          f"gap comble, cloture {var_j0:+.1f} % : l'attente a evite l'entree"),
                "variation_pct": round(var_j0, 2) if var_j0 is not None else None}

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
    # Un short juste n'est gagnant que s'il etait executable (29.09.2026, EGG).
    vendeurs = [n["ticker"] for n in notes if n["classe"] in CLASSES_VENDEUSES]
    barres = barres_ouverture(vendeurs, date_detection) if vendeurs else {}
    for n in notes:
        if n["classe"] in CLASSES_VENDEUSES:
            n["executabilite"] = juger(barres.get(n["ticker"]))

    largeur = max((len(n["ticker"]) for n in notes), default=6)
    for n in sorted(notes, key=lambda x: (x["resultat"], x["classe"])):
        var = f"{n['variation_pct']:+7.2f} %" if n["variation_pct"] is not None else "      --"
        print(f"  {n['ticker']:<{largeur}}  {n['classe']:<13} {n['resultat']:<10} "
              f"{var}   {n['motif']}")
        if "executabilite" in n:
            e = n["executabilite"]
            print(f"  {'':<{largeur}}  executable: {e['executable']:<8} {e['motif']}")

    juges = [n for n in notes if n["resultat"] in ("juste", "faux")]
    justes = [n for n in juges if n["resultat"] == "juste"]
    incomplets = [n for n in notes if n["resultat"] == "incomplet"]
    print()
    if juges:
        print(f"Taux de justesse : {len(justes)}/{len(juges)} "
              f"({100*len(justes)/len(juges):.0f} %)")
    else:
        print("Aucun verdict jugeable sur cette fenetre.")
    shorts = bilan_shorts(notes)
    if shorts["justes"]:
        print(f"Shorts justes ET executables : {shorts['justes_executables']}/{shorts['justes']} "
              "(spread et borrow non mesures)")
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
