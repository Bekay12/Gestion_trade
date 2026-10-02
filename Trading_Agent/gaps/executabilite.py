#!/usr/bin/env python3
"""
executabilite - un short juste n'est gagnant que s'il etait executable.

evaluer_gaps note la PROMESSE d'un verdict (le gap s'est-il comble, le titre
est-il retombe). Ce module repond a une autre question: aurait-on pu vendre a
l'ouverture, avec une taille reelle, sans que le carnet decide du prix ?
Motif, 29.09.2026: la revue Germain sur EGG (pic a 13,11 $ dans un carnet vide
a 04:00 ET, sans prix negociable) et nos shorts SLXN / CYAB notes justes sur
des titres a quelques centimes.

Mesure, sur les barres de 5 min de 09:35 a 10:00 ET:
    - volume en dollars >= SEUIL_VOLUME_POST_OUVERTURE_USD
    - moins de BARRES_VIDES_MAX barres sans aucun echange
La barre de 09:30 est EXCLUE: yfinance y range l'enchere d'ouverture et le
volume pre-marche (CYAB 29.09: 63 M titres sur 71 M de la journee). Avec elle,
EGG passait pour liquide (5,9 M$) alors que le carnet etait vide ensuite (0,2 M$).

Ce que la mesure NE couvre PAS, faute de donnee:
    - le spread: yfinance ne publie pas de bid/ask historique;
    - le borrow (disponibilite, cout): se releve chez le courtier.
Une donnee absente vaut refus: sans barres, le verdict est "inconnu", et
"inconnu" ne compte jamais comme executable.
"""
from __future__ import annotations

from datetime import date, timedelta

# Une position de 10 000 $ (capital du portefeuille papier, agent/live.py) ne
# doit pas depasser 5 % du flux de 09:35 a 10:00: 10 000 / 0,05.
SEUIL_VOLUME_POST_OUVERTURE_USD = 200_000
# Sur cinq barres de 5 min, deux barres sans echange signalent un carnet troue.
BARRES_VIDES_MAX = 2
# yfinance ne sert les barres de 5 min que sur les 60 derniers jours.
PROFONDEUR_INTRADAY_JOURS = 59

CLASSES_VENDEUSES = ("FADE", "PUMP_RISK")


def juger(barres: list[dict] | None) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Dire si un titre etait negociable apres le premier prix d'ouverture.

    Inputs:
        barres (list[dict] | None): barres de 5 min 09:30-09:55 ET,
            chacune {"cloture": float, "volume": float}; la premiere (09:30)
            est ignoree

    Outputs:
        verdict (dict): executable ("oui" / "non" / "inconnu"), motif,
            volume_post_ouverture_usd, barres_vides
    --------------------------------------------------------------------------
    """
    flux = (barres or [])[1:]
    if not flux:
        return {"executable": "inconnu", "volume_post_ouverture_usd": None, "barres_vides": None,
                "motif": "barres intraday indisponibles (au-dela de 60 jours ou absentes)"}
    dollars = sum(b["cloture"] * b["volume"] for b in flux)
    vides = sum(1 for b in flux if not b["volume"])
    motifs = []
    if dollars < SEUIL_VOLUME_POST_OUVERTURE_USD:
        motifs.append(f"volume 09:35-10:00 {dollars / 1e3:.0f} k$ "
                      f"< {SEUIL_VOLUME_POST_OUVERTURE_USD / 1e3:.0f} k$")
    if vides >= BARRES_VIDES_MAX:
        motifs.append(f"{vides} barres de 5 min sans echange")
    return {"executable": "non" if motifs else "oui",
            "volume_post_ouverture_usd": round(dollars),
            "barres_vides": vides,
            "motif": "; ".join(motifs) if motifs else
                     f"volume 09:35-10:00 {dollars / 1e3:.0f} k$, carnet continu "
                     "(spread et borrow non mesures)"}


def barres_ouverture(tickers: list, jour: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Barres de 5 min 09:30-10:00 ET du jour de detection, en UN SEUL appel
        groupe (budget yfinance du depot).

    Inputs:
        tickers (list): symboles a mesurer
        jour (str): date de detection, AAAA-MM-JJ

    Outputs:
        barres (dict): {ticker: [{"cloture", "volume"}, ...]}; un ticker absent
            du dict n'a pas de mesure, et sera juge "inconnu"
    --------------------------------------------------------------------------
    """
    import logging

    import pandas as pd
    import yfinance as yf

    j = date.fromisoformat(jour)
    if not tickers or (date.today() - j).days > PROFONDEUR_INTRADAY_JOURS:
        return {}
    try:
        lot = yf.download(tickers, start=j.isoformat(), end=(j + timedelta(days=1)).isoformat(),
                          interval="5m", group_by="ticker", auto_adjust=False,
                          prepost=False, progress=False, threads=True)
    except Exception as exc:   # reseau, quota: pas de mesure, donc "inconnu"
        logging.getLogger(__name__).warning("[EXECUTABILITE] telechargement 5 min: %s", exc)
        return {}
    return _extraire(lot, tickers)


def _extraire(lot, tickers: list) -> dict:
    """
    Six barres 09:30-09:55 ET par ticker. yfinance OMET une barre sans echange au
    lieu de la mettre a zero: chaque creneau manquant est donc reconstitue avec un
    volume nul, sinon un carnet troue passerait pour continu.
    """
    import pandas as pd

    creneaux = [f"09:{m:02d}" for m in range(30, 60, 5)]
    resultat = {}
    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna(subset=["Close"])
            d.index = d.index.tz_convert("America/New_York")
            d = d.between_time("09:30", "09:55")
            if d.empty:
                continue
            par_heure = {i.strftime("%H:%M"): (float(c), float(v))
                         for i, c, v in zip(d.index, d["Close"], d["Volume"])}
            barres, dernier = [], par_heure[min(par_heure)][0]
            for h in creneaux:
                c, v = par_heure.get(h, (dernier, 0.0))
                barres.append({"cloture": c, "volume": v})
                dernier = c
            resultat[t] = barres
        except Exception:
            continue
    return resultat


def bilan_shorts(notes: list[dict]) -> dict:
    """Shorts justes, et parmi eux ceux qui etaient executables ("inconnu" = non)."""
    justes = [n for n in notes
              if n.get("classe") in CLASSES_VENDEUSES and n.get("resultat") == "juste"]
    executables = [n for n in justes
                   if n.get("executabilite", {}).get("executable") == "oui"]
    return {"justes": len(justes), "justes_executables": len(executables)}
