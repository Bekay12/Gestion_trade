"""
Contrat unique du vecteur de parametres de l'optimisateur hybride.

Decrit UNE seule fois, pour chaque parametre : son nom logique, ses bornes de
recherche et la colonne de `optimization_runs` qui le porte. L'optimisateur, la
sauvegarde et la relecture de l'historique en derivent tout.

Motif : les bornes etaient ecrites trois fois et avaient diverge sur quatre
parametres, donc le vecteur ecrit en base n'etait pas celui qui avait produit le
score. Voir docs/superpowers/specs/2026-08-05-optimisateur-hybride-lot1-design.md.

L'arrondi a la precision de recherche reste la responsabilite de l'optimiseur ;
ce module ne fait que borner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpecParam:
    """
    --------------------------------------------------------------------------
    Objectif:
        Decrire un parametre du vecteur d'optimisation.

    Inputs:
        cle (str): nom logique, par exemple 'a1', 'th_rsi', 'seuil_achat'
        bornes (tuple | None): (min, max) de recherche, None si le parametre
            est gele donc absent du vecteur
        colonne_db (str): colonne de optimization_runs qui porte la valeur
        gele (float | None): valeur figee des parametres non optimises

    Outputs:
        (dataclass immuable)
    --------------------------------------------------------------------------
    """

    cle: str
    bornes: tuple[float, float] | None
    colonne_db: str
    gele: float | None = None


# L'ordre EST l'ordre du vecteur. Ne pas reordonner : les optimiseurs
# travaillent par index.
VECTEUR_BASE: tuple[SpecParam, ...] = (
    SpecParam('a1', (-1.5, 3.0), 'a1'),
    SpecParam('a2', (-1.5, 3.0), 'a2'),
    SpecParam('a3', (-1.5, 3.0), 'a3'),
    SpecParam('a4', (-1.5, 3.0), 'a4'),
    SpecParam('a5', (-1.5, 3.0), 'a5'),
    SpecParam('a6', (-1.5, 3.0), 'a6'),
    SpecParam('a7', (-1.5, 3.0), 'a7'),
    SpecParam('a8', (-1.5, 3.0), 'a8'),
    SpecParam('th_rsi', (30.0, 70.0), 'th1'),
    SpecParam('th_vol', (0.5, 2.5), 'th4'),
    SpecParam('th_adx', (15.0, 35.0), 'th5'),
    SpecParam('th_score', (2.0, 6.0), 'th8'),
    SpecParam('seuil_achat', (1.0, 6.0), 'seuil_achat'),
    SpecParam('seuil_vente', (-6.0, -1.0), 'seuil_vente'),
)

# Seuils que le projet a figes : hors du vecteur de recherche, mais ecrits en
# base pour que la ligne soit complete et relisible.
SEUILS_GELES: tuple[SpecParam, ...] = (
    SpecParam('th_macd', None, 'th2', gele=0.0),
    SpecParam('th_ema', None, 'th3', gele=0.0),
    SpecParam('th_ichimoku', None, 'th6', gele=0.0),
    SpecParam('th_boll', None, 'th7', gele=0.5),
)

EXTRAS_PRIX: tuple[SpecParam, ...] = (
    SpecParam('use_price_extras', (0.0, 1.0), 'use_price_extras'),
    SpecParam('a_price_slope', (-1.5, 3.0), 'a9'),
    SpecParam('a_price_acc', (-1.5, 3.0), 'a10'),
    SpecParam('th_price_slope', (-0.25, 0.25), 'th9'),
    SpecParam('th_price_acc', (-0.25, 0.25), 'th10'),
    SpecParam('a_price_rsi_slope', (-1.5, 3.0), 'a16'),
    SpecParam('a_price_vol_slope', (-1.5, 3.0), 'a17'),
    SpecParam('a_price_var5j', (-1.5, 3.0), 'a18'),
    SpecParam('th_price_rsi_slope', (-0.25, 0.25), 'th16'),
    SpecParam('th_price_vol_slope', (-0.25, 0.25), 'th17'),
    SpecParam('th_price_var5j', (-20.0, 20.0), 'th18'),
)

EXTRAS_FONDAMENTAUX: tuple[SpecParam, ...] = (
    SpecParam('use_fundamentals', (0.0, 1.0), 'use_fundamentals'),
    SpecParam('a_rev_growth', (-1.5, 3.0), 'a11'),
    SpecParam('a_eps_growth', (-1.5, 3.0), 'a12'),
    SpecParam('a_roe', (-1.5, 3.0), 'a13'),
    SpecParam('a_fcf_yield', (-1.5, 3.0), 'a14'),
    SpecParam('a_de_ratio', (-1.5, 3.0), 'a15'),
    SpecParam('th_rev_growth', (-30.0, 30.0), 'th11'),
    SpecParam('th_eps_growth', (-30.0, 30.0), 'th12'),
    SpecParam('th_roe', (-30.0, 30.0), 'th13'),
    SpecParam('th_fcf_yield', (-20.0, 20.0), 'th14'),
    SpecParam('th_de_ratio', (-20.0, 20.0), 'th15'),
)

# Ordre des 8 seuils attendu par `domain_thresholds` du moteur de backtest.
ORDRE_SEUILS_MOTEUR: tuple[str, ...] = (
    'th_rsi', 'th_macd', 'th_ema', 'th_vol',
    'th_adx', 'th_ichimoku', 'th_boll', 'th_score',
)

# Parametres binaires : ramenes a 0 ou 1 par contraindre().
DRAPEAUX = frozenset({'use_price_extras', 'use_fundamentals'})


def specs(prix: bool = False, fond: bool = False) -> tuple[SpecParam, ...]:
    """Specs du vecteur de recherche, dans l'ordre des index."""
    sortie = VECTEUR_BASE
    if prix:
        sortie = sortie + EXTRAS_PRIX
    if fond:
        sortie = sortie + EXTRAS_FONDAMENTAUX
    return sortie


def bornes(prix: bool = False, fond: bool = False) -> list[tuple[float, float]]:
    """Bornes a passer aux optimiseurs, dans l'ordre du vecteur."""
    return [spec.bornes for spec in specs(prix, fond)]


def indices(prix: bool = False, fond: bool = False) -> dict[str, int]:
    """{cle: index} du vecteur. Remplace tout calcul d'offset."""
    return {spec.cle: index for index, spec in enumerate(specs(prix, fond))}


def contraindre(vecteur, prix: bool = False, fond: bool = False) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Objectif:
        Ramener chaque valeur dans ses bornes. SEUL bridage du projet : appele
        a l'entree de l'evaluation, en sortie du croisement genetique et avant
        la sauvegarde, pour que les trois voient le meme vecteur.

    Inputs:
        vecteur (Sequence[float]): vecteur brut
        prix (bool), fond (bool): drapeaux de features actives

    Outputs:
        contraint (np.ndarray): copie bornee, drapeaux ramenes a 0 ou 1
    --------------------------------------------------------------------------
    """
    liste = specs(prix, fond)
    valeurs_np = np.asarray(vecteur, dtype=float).ravel()
    if valeurs_np.shape[0] != len(liste):
        raise ValueError(
            f"vecteur de taille {valeurs_np.shape[0]}, attendu {len(liste)} "
            f"(prix={prix}, fond={fond})"
        )
    sortie = valeurs_np.copy()
    for index, spec in enumerate(liste):
        bas, haut = spec.bornes
        valeur = min(max(float(sortie[index]), bas), haut)
        if spec.cle in DRAPEAUX:
            valeur = float(int(valeur + 0.5))
        sortie[index] = valeur
    return sortie


def valeurs(vecteur, prix: bool = False, fond: bool = False) -> dict[str, float]:
    """{cle: valeur} du vecteur contraint, seuils geles inclus."""
    contraint = contraindre(vecteur, prix, fond)
    sortie = {spec.cle: float(contraint[index])
              for index, spec in enumerate(specs(prix, fond))}
    for spec in SEUILS_GELES:
        sortie[spec.cle] = float(spec.gele)
    return sortie


def coefficients(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, ...]:
    """Les 8 coefficients a1 a a8."""
    lues = valeurs(vecteur, prix, fond)
    return tuple(lues[f'a{numero}'] for numero in range(1, 9))


def globaux(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, float]:
    """(seuil_achat, seuil_vente)."""
    lues = valeurs(vecteur, prix, fond)
    return lues['seuil_achat'], lues['seuil_vente']


def seuils_features(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, ...]:
    """Les 8 seuils dans l'ordre attendu par `domain_thresholds`."""
    lues = valeurs(vecteur, prix, fond)
    return tuple(lues[cle] for cle in ORDRE_SEUILS_MOTEUR)


def dict_extras_prix(vecteur, prix: bool = False, fond: bool = False) -> dict | None:
    """Dict des extras prix attendu par le moteur, ou None si desactives."""
    if not prix:
        return None
    lues = valeurs(vecteur, prix, fond)
    sortie = {spec.cle: lues[spec.cle] for spec in EXTRAS_PRIX}
    sortie['use_price_extras'] = int(sortie['use_price_extras'])
    return sortie


def dict_extras_fondamentaux(vecteur, prix: bool = False, fond: bool = False) -> dict | None:
    """Dict des extras fondamentaux attendu par le moteur, ou None."""
    if not fond:
        return None
    lues = valeurs(vecteur, prix, fond)
    sortie = {spec.cle: lues[spec.cle] for spec in EXTRAS_FONDAMENTAUX}
    sortie['use_fundamentals'] = int(sortie['use_fundamentals'])
    return sortie


def vers_colonnes(vecteur, prix: bool = False, fond: bool = False) -> dict[str, float]:
    """{colonne SQLite: valeur} pour l'INSERT, seuils geles inclus."""
    lues = valeurs(vecteur, prix, fond)
    sortie: dict[str, float] = {}
    for spec in specs(prix, fond) + SEUILS_GELES:
        valeur = lues[spec.cle]
        sortie[spec.colonne_db] = int(valeur) if spec.cle in DRAPEAUX else float(valeur)
    return sortie


def depuis_colonnes(row, prix: bool = False, fond: bool = False) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Objectif:
        Reconstruire un vecteur depuis une ligne de `optimization_runs`, pour
        rejouer un run historique.

    Inputs:
        row (Mapping | sqlite3.Row): indexable par nom de colonne
        prix (bool), fond (bool): drapeaux de features actives

    Outputs:
        vecteur (np.ndarray): contraint, donc toujours dans le domaine valide.
        Une colonne absente ou nulle prend le milieu de ses bornes. Le defaut
        historique seuil_vente = -0.5 etait d'ailleurs hors des bornes
        declarees (-6.0, -1.0), ce que le milieu evite.
        Exception, les DRAPEAUX : leur milieu vaut 0.5, que contraindre()
        arrondit a 1, donc ACTIVE. Une ligne ecrite avant l'introduction de la
        colonne se verrait alors attribuer des features qu'elle n'a jamais
        eues, avec des coefficients pris au milieu des bornes. La convention du
        depot, celle de l'ancien save_optimization_results, est l'inverse :
        drapeau absent vaut desactive. Ils retombent donc sur 0.
    --------------------------------------------------------------------------
    """
    brut = []
    for spec in specs(prix, fond):
        valeur = None
        try:
            valeur = row[spec.colonne_db]
        except (KeyError, IndexError, TypeError):
            valeur = None
        if valeur is None:
            if spec.cle in DRAPEAUX:
                valeur = 0.0
            else:
                bas, haut = spec.bornes
                valeur = (bas + haut) / 2.0
        brut.append(float(valeur))
    return contraindre(np.asarray(brut, dtype=float), prix, fond)
