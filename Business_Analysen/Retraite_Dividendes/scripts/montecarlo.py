#!/usr/bin/env python3
"""
montecarlo.py - tirage par blocs de 5 ans dans l'historique Shiller (garde les series de
crises), graine fixe pour que le document soit reproductible.
Reussite = objectif atteint ET, pendant rentenjahre, revenu en dividendes >= 80 % de
l'objectif chaque annee; une reserve de puffer_jahre annees d'objectif comble les trous.
"""
import warnings

import numpy as np
import pandas as pd


def pfade(jahres: pd.DataFrame, n: int, jahre: int, block: int = 5,
          seed: int = 20260929) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Purpose:
        Bootstrap par blocs de `block` annees contigues dans l'historique reel
        (rendite_real, div_wachstum_real) pour produire n trajectoires de
        longueur `jahre`. Le tirage par blocs (plutot qu'annee par annee)
        preserve l'autocorrelation des crises et des reprises presentes dans
        la serie source, contrairement a un tirage i.i.d. annee par annee.

        La premiere ligne de `jahres` (telle que produite par
        histoire.jahresreihe) est un artefact de bord, pas une observation
        reelle : div_wachstum_real y est force a 0.0 (rien a comparer avant
        la premiere annee de la serie) et rendite_real y provient d'un
        fillna(0.0) faute de p_ende/cpi_ende de l'annee precedente. La
        laisser dans le pool de tirage introduirait un pseudo-scenario
        "annee sans rendement ni croissance" qui n'a jamais ete observe
        historiquement ; elle est donc retiree avant l'echantillonnage
        (decision du controleur, 2026-09-29).

    Inputs:
        jahres (pandas.DataFrame): sortie de histoire.jahresreihe() (colonnes
            "rendite_real", "div_wachstum_real" au minimum) ; la premiere
            ligne est ecartee, voir ci-dessus.
        n (int): nombre de trajectoires simulees.
        jahre (int): longueur en annees de chaque trajectoire.
        block (int): longueur des blocs contigus tires, defaut 5 ans.
        seed (int): graine du generateur, fixee par defaut pour un document
            reproductible.

    Outputs:
        result (numpy.ndarray): tableau (n, jahre, 2) ; [..., 0] =
            rendite_real, [..., 1] = div_wachstum_real.
    --------------------------------------------------------------------------
    """
    daten = jahres[["rendite_real", "div_wachstum_real"]].to_numpy()[1:]
    rng = np.random.default_rng(seed)
    aus = np.empty((n, jahre, 2))
    for i in range(n):
        reihe: list[np.ndarray] = []
        while len(reihe) < jahre:
            # rng.integers a une borne superieure exclusive : sans le "+ 1", le dernier
            # bloc possible (depart len(daten) - block) n'est jamais tire (decision du
            # controleur, 2026-09-29).
            s = rng.integers(0, len(daten) - block + 1)
            reihe.extend(daten[s:s + block])
        aus[i] = np.array(reihe[:jahre])
    return aus


def erfolg(pfade: np.ndarray, sparplan: list[float], ziel_real_jahr: float,
           rendite_div: float = 0.035, rentenjahre: int = 40,
           puffer_jahre: int = 0, div_schock_erstes_rentenjahr: float = 0.0) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Evaluer, sur un ensemble de trajectoires simulees par pfade(), le taux
        de reussite d'un plan d'epargne : accumulation jusqu'a ce que le
        revenu de dividendes projete atteigne ziel_real_jahr, puis verification
        que ce revenu reste au moins a 80 % de l'objectif pendant rentenjahre
        annees de retraite (une reserve de puffer_jahre annees d'objectif
        comble les trous ponctuels).

        Contrat d'appel : `jahre` (2e dimension de `pfade`, l'argument) doit
        couvrir au moins l'horizon d'accumulation attendu + rentenjahre.
        Une trajectoire qui atteint l'objectif trop tard pour derouler tout
        rentenjahre dans la fenetre simulee (erreicht + rentenjahre >= jahre)
        ne peut pas etre jugee : elle est comptee comme un echec (le
        denominateur reste n, ne pas atteindre l'objectif a temps EST un
        echec) mais, plutot que d'etre ignoree en silence, son compte figure
        dans la cle "abgeschnitten" (decision du controleur, 2026-09-29) :
        un "abgeschnitten" > 0 signale a l'appelant (Tache 9, chapitre 7)
        qu'il faut allonger `jahre` pour dimensionner correctement la
        simulation, pas que la strategie a reellement echoue dans ces cas-la.

    Inputs:
        pfade (numpy.ndarray): tableau (n, jahre, 2) produit par pfade().
        sparplan (list[float]): versements annuels reels a partir de l'annee
            0, tant que l'objectif n'est pas encore atteint.
        ziel_real_jahr (float): revenu de dividendes reel annuel vise.
        rendite_div (float): rendement de dividende suppose applique au
            capital accumule pour estimer le revenu, defaut 3,5 %.
        rentenjahre (int): duree de la phase de retraite testee, defaut 40.
        puffer_jahre (int): nombre d'annees d'objectif tenues en reserve pour
            combler un manque ponctuel de revenu, defaut 0.
        div_schock_erstes_rentenjahr (float): choc ADDITIF applique a la
            croissance reelle du dividende (div_wachstum_real) de la PREMIERE
            annee de retraite de chaque trajectoire (l'annee erreicht + 1, qui
            varie d'une trajectoire a l'autre selon la date d'atteinte de
            l'objectif) - stress-test de sequence des rendements a l'entree en
            retraite (Tache 9, chapitre 7, revue de code du 2026-09-30: cette
            option remplace une reimplementation dupliquee qui vivait dans
            rechnung_retraite.py). Defaut 0.0: additif avec un choc nul ne
            change rien, le comportement par defaut de erfolg() est donc
            identique bit a bit a la version sans ce parametre (teste dans
            test_montecarlo.py). Additif (pas un remplacement de la valeur
            tiree) pour que le defaut soit neutre par construction, plutot
            qu'un sentinel a exclure explicitement du calcul.

    Outputs:
        result (dict): {"ziel_jahr_perzentile": {10: float, 50: float,
            90: float}, "erfolgsquote": float, "wert_perzentile":
            ndarray (jahre, 3), "abgeschnitten": int}.
    --------------------------------------------------------------------------
    """
    n, jahre, _ = pfade.shape
    ziel_jahre, erfolge, abgeschnitten = [], 0, 0
    werte = np.zeros((n, jahre))
    for i in range(n):
        wert, erreicht = 0.0, None
        for t in range(jahre):
            beitrag = sparplan[t] if erreicht is None and t < len(sparplan) else 0.0
            wert = wert * (1 + pfade[i, t, 0]) + beitrag
            werte[i, t] = wert
            if erreicht is None and wert * rendite_div >= ziel_real_jahr:
                erreicht = t
        if erreicht is None:
            continue
        if erreicht + rentenjahre >= jahre:
            abgeschnitten += 1
            continue
        ziel_jahre.append(erreicht)
        einkommen, puffer, ok = ziel_real_jahr, puffer_jahre * ziel_real_jahr, True
        for t in range(erreicht + 1, erreicht + 1 + rentenjahre):
            wachstum = pfade[i, t, 1]
            if t == erreicht + 1:
                wachstum += div_schock_erstes_rentenjahr
            einkommen *= 1 + wachstum
            fehlt = max(0.0, 0.8 * ziel_real_jahr - einkommen)
            if fehlt > puffer:
                ok = False
                break
            puffer -= fehlt
        erfolge += ok
    zj = np.array(ziel_jahre) if ziel_jahre else np.array([np.nan])
    # np.nanpercentile emet un RuntimeWarning benin quand zj est entierement NaN (aucune
    # trajectoire n'a atteint l'objectif a temps, ou toutes ont ete tronquees) ; le
    # resultat nan est le comportement voulu dans ce cas, donc l'avertissement est cible
    # (ce bloc seulement) plutot que masque globalement pour le module.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                 category=RuntimeWarning)
        ziel_perzentile = {p: float(np.nanpercentile(zj, p)) for p in (10, 50, 90)}
    return {"ziel_jahr_perzentile": ziel_perzentile,
            "erfolgsquote": erfolge / n,
            "wert_perzentile": np.percentile(werte, [10, 50, 90], axis=0).T,
            "abgeschnitten": abgeschnitten}
