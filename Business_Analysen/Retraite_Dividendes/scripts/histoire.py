#!/usr/bin/env python3
"""
histoire.py - S&P 500 depuis 1871 (Shiller, ie_data.xls): rendement reel, dividende reel,
baisses de dividende, rejeu de la strategie par annee de depart.

Limite declaree: marche americain seul (la seule serie de dividendes aussi longue et
publique); le chapitre 7 le dit, et la sensibilite europeenne passe par le Monte Carlo.

Colonne D de la feuille Shiller: taux de dividende annualise, publie chaque mois (pas un
montant mensuel cumulable). La moyenne mensuelle sur une annee est donc deja le dividende
annuel; c'est pourquoi jahresreihe() nomme cette colonne agregee ``d_jahr`` (et non
``d_summe``, qui suggererait a tort une somme de montants mensuels).
"""
import os

import pandas as pd


def laden(pfad: str) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------
    Purpose:
        Charger la feuille "Data" du classeur Shiller ie_data.xls et ne garder
        que les lignes avec un dividende renseigne.

    Inputs:
        pfad (str): chemin vers ie_data.xls (feuille "Data", en-tetes lignes
            4-7, donnees a partir de la ligne 8 ; verifie le 30.09.2026 sur le
            fichier telecharge, colonnes 0=Date, 1=P, 2=D, 3=E, 4=CPI).

    Outputs:
        result (pandas.DataFrame): colonnes "datum" (float annee.mois), "p",
            "d", "cpi" ; lignes non numeriques ou sans dividende retirees.
    --------------------------------------------------------------------------
    """
    x = pd.read_excel(pfad, sheet_name="Data", header=None, skiprows=8)
    df = x.iloc[:, [0, 1, 2, 4]].copy()
    df.columns = ["datum", "p", "d", "cpi"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def jahresreihe(df: pd.DataFrame) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------
    Purpose:
        Agreger la serie mensuelle en serie annuelle: rendement total reel,
        dividende reel annuel et sa croissance reelle, inflation.

    Inputs:
        df (pandas.DataFrame): sortie de laden() ou serie de test equivalente
            (colonnes "datum", "p", "d", "cpi").

    Outputs:
        result (pandas.DataFrame): colonnes "jahr", "rendite_real",
            "div_real", "div_wachstum_real", "inflation". La derniere annee
            est retiree si elle compte moins de 12 lignes mensuelles (annee
            en cours, non terminee dans la source) : "p_ende"/"cpi_ende" y
            porteraient sur un mois intermediaire (ex. septembre) compare a
            un decembre l'annee precedente, ce qui fausserait rendite_real
            et div_wachstum_real de cette derniere ligne.
    --------------------------------------------------------------------------
    """
    d = df.copy()
    d["jahr"] = d["datum"].astype(int)
    # Ecarter une derniere annee incomplete (donnees Shiller arretees en cours d'annee,
    # ex. 2023.09) avant l'agregation : sinon p_ende/cpi_ende de cette ligne comparent un
    # mois intermediaire au decembre precedent, ce qui fausse rendite_real et
    # div_wachstum_real de la derniere annee.
    compte = d.groupby("jahr").size()
    derniere_annee = compte.index[-1]
    if compte.loc[derniere_annee] < 12:
        d = d[d["jahr"] != derniere_annee]
    # colonne D de Shiller = taux de dividende annualise publie chaque mois ;
    # la moyenne mensuelle sur l'annee est donc le dividende annuel (pas une somme).
    g = d.groupby("jahr").agg(p_ende=("p", "last"), d_jahr=("d", "mean"), cpi_ende=("cpi", "last"))
    g = g.reset_index()
    g["inflation"] = g["cpi_ende"].pct_change()
    g["div_real"] = g["d_jahr"] / g["cpi_ende"] * g["cpi_ende"].iloc[-1]
    g["div_wachstum_real"] = g["div_real"].pct_change()
    nominal = (g["p_ende"] + g["d_jahr"]) / g["p_ende"].shift(1) - 1
    g["rendite_real"] = (1 + nominal) / (1 + g["inflation"]) - 1
    g.loc[g.index[0], ["div_wachstum_real"]] = 0.0
    return g[["jahr", "rendite_real", "div_real", "div_wachstum_real", "inflation"]].fillna(0.0)


def div_einbrueche(jahres: pd.DataFrame, schwelle: float = -0.10) -> list[dict]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Detecter les episodes ou le dividende reel baisse d'au moins |schwelle|
        depuis le sommet precedent (utilise par la Tache 9 pour les figures).

    Inputs:
        jahres (pandas.DataFrame): sortie de jahresreihe().
        schwelle (float): seuil de baisse relative (negatif), defaut -0.10.

    Outputs:
        result (list[dict]): chaque element {"von": int, "bis": int,
            "rueckgang": float} (annee de sommet, annee de creux, baisse
            relative du sommet au creux).
    --------------------------------------------------------------------------
    """
    aus, spitze, spitze_jahr = [], None, None
    for _, r in jahres.iterrows():
        if spitze is None or r["div_real"] >= spitze:
            spitze, spitze_jahr = r["div_real"], int(r["jahr"])
            continue
        rueck = r["div_real"] / spitze - 1
        if rueck <= schwelle:
            if aus and aus[-1]["von"] == spitze_jahr:
                if rueck < aus[-1]["rueckgang"]:
                    aus[-1].update(bis=int(r["jahr"]), rueckgang=rueck)
            else:
                aus.append({"von": spitze_jahr, "bis": int(r["jahr"]), "rueckgang": rueck})
    return aus


def rueckspiel(jahres: pd.DataFrame, sparplan: list[float], ziel_real_jahr: float,
               rendite_div: float = 0.035, rentenjahre: int = 40) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rejouer la strategie d'epargne pour chaque annee historique de depart:
        accumulation avec les rendements reels historiques jusqu'a ce que le
        revenu de dividendes projete atteigne l'objectif, puis verifier la
        survie du revenu sur rentenjahre annees de retraite.

    Inputs:
        jahres (pandas.DataFrame): sortie de jahresreihe() (indexee par annee
            via "jahr", colonnes "rendite_real", "div_wachstum_real").
        sparplan (list[float]): versements annuels reels a partir de l'annee
            de depart (index 0 = premiere annee).
        ziel_real_jahr (float): revenu de dividendes reel annuel vise.
        rendite_div (float): rendement de dividende suppose applique au
            capital accumule pour estimer le revenu, defaut 3,5 %.
        rentenjahre (int): duree de la phase de retraite testee, defaut 40.

    Outputs:
        result (pandas.DataFrame): colonnes "start", "ziel_jahr",
            "jahre_bis_ziel", "ueberlebt" (None si l'objectif n'est jamais
            atteint ou si l'historique s'arrete avant la fin des
            rentenjahre annees).
    --------------------------------------------------------------------------
    """
    serie = jahres.set_index("jahr")
    zeilen = []
    for start in serie.index:
        wert, erreicht = 0.0, None
        for i, beitrag in enumerate(sparplan):
            j = start + i
            if j not in serie.index:
                break
            wert = wert * (1 + serie.at[j, "rendite_real"]) + beitrag
            if wert * rendite_div >= ziel_real_jahr:
                erreicht = j
                break
        if erreicht is None:
            zeilen.append({"start": start, "ziel_jahr": None, "jahre_bis_ziel": None, "ueberlebt": None})
            continue
        einkommen, ok, vollstaendig = ziel_real_jahr, True, True
        for k in range(1, rentenjahre + 1):
            j = erreicht + k
            if j not in serie.index:
                vollstaendig = False
                break
            einkommen *= 1 + serie.at[j, "div_wachstum_real"]
            if einkommen < 0.8 * ziel_real_jahr:
                ok = False
        zeilen.append({"start": start, "ziel_jahr": erreicht, "jahre_bis_ziel": erreicht - start + 1,
                       "ueberlebt": ok if vollstaendig else None})
    return pd.DataFrame(zeilen)


if __name__ == "__main__":
    ici = os.path.dirname(os.path.abspath(__file__))
    pfad_source = os.path.join(ici, "..", "refs", "ie_data.xls")
    pfad_sortie = os.path.join(ici, "..", "data", "shiller_jahr.csv")

    jahres = jahresreihe(laden(pfad_source))
    jahres.to_csv(pfad_sortie, index=False)
    print(f"annees: {len(jahres)}, premiere: {int(jahres['jahr'].min())}, "
          f"derniere: {int(jahres['jahr'].max())}")
