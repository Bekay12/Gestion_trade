# Optimisateur hybride, lot 1 : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre l'optimisateur hybride démarrable et faire en sorte que ce qu'il écrit en base soit exactement ce qu'il a mesuré.

**Architecture:** Un module de contrat (`core/optim_params.py`) devient la seule description du vecteur de paramètres, avec ses bornes et ses colonnes SQLite ; l'optimisateur, la sauvegarde et la relecture de l'historique en dérivent tout. L'objectif est épinglé sur le chemin de backtest qui honore les seuils. Un garde-fou refuse une bibliothèque C instrumentée avant son chargement, ce qui débloque le point d'entrée CLI.

**Tech Stack:** Python 3.10, numpy, scipy (`differential_evolution`, `qmc`), pandas, SQLite, pytest.

## Global Constraints

- Environnement : `.venv_new` à la racine du dépôt. Toute commande Python utilise `../.venv_new/bin/python` depuis `stock-analysis-ui/`.
- Les tests tournent depuis `stock-analysis-ui/`. Le sous-ensemble par défaut est `pytest -m "not integration"` et doit rester **hors réseau et hors base réelle**.
- Le marqueur `integration` signifie accès réseau ou base réelle. Aucun nouveau test de ce lot ne le porte.
- `src/tests/conftest.py` pose déjà `QSI_DISABLE_C_ACCELERATION=1`, `QSI_CONSENSUS_OFFLINE=1` et `QSI_DISABLE_PROFILE_FETCH=1`.
- Conventions : docstrings et messages en français, `snake_case`, types dans les signatures, `logging.getLogger(__name__)` avec préfixe `[TAG]` si journalisation.
- Interdiction du tiret d'incise en prose et en commentaires (`--`, `—`, `–`). Utiliser virgule, deux-points ou parenthèses.
- Ne jamais ajouter de boucle yfinance par symbole. Ce lot n'ajoute aucun appel réseau.
- Bornes de référence : celles de la recherche, `optimisateur_hybride.py` lignes 527 à 572. Les bridages plus étroits des lignes 644 à 701 et 1564 à 1617 sont abandonnés.
- Spec de référence : `docs/superpowers/specs/2026-08-05-optimisateur-hybride-lot1-design.md`.

---

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `stock-analysis-ui/src/core/optim_params.py` (créer) | Contrat unique : specs, bornes, bridage, index, correspondance SQLite, aller-retour |
| `stock-analysis-ui/src/core/optim_budget.py` (créer) | Arithmétique du budget d'évaluations, partagée menu et lancement |
| `stock-analysis-ui/src/optimisateur_hybride.py` (modifier) | Consomme les deux modules, plus B2, parallélisme, graine, hygiène |
| `stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py` (modifier) | Garde-fou ASan avant `dlopen`, docstring du coût de transaction |
| `stock-analysis-ui/src/tests/test_optim_params.py` (créer) | Contrat |
| `stock-analysis-ui/src/tests/test_optim_budget.py` (créer) | Budget |
| `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (créer) | B2 et run de fumée, backtest simulé, base temporaire |
| `stock-analysis-ui/src/tests/test_c_module_guard.py` (créer) | Détection d'un binaire instrumenté |
| `stock-analysis-ui/src/tests/test_fundamentals_integration.py` (modifier) | Assertions de dimensions périmées |
| `stock-analysis-ui/CHANGELOG.md` (modifier) | Entrée de version |

---

### Task 1: Contrat unique du vecteur de paramètres

**Files:**
- Create: `stock-analysis-ui/src/core/optim_params.py`
- Test: `stock-analysis-ui/src/tests/test_optim_params.py`

**Interfaces:**
- Consumes: rien.
- Produces:
  - `SpecParam(cle: str, bornes: tuple[float, float] | None, colonne_db: str, gele: float | None = None)`
  - `specs(prix: bool = False, fond: bool = False) -> tuple[SpecParam, ...]`
  - `bornes(prix: bool = False, fond: bool = False) -> list[tuple[float, float]]`
  - `indices(prix: bool = False, fond: bool = False) -> dict[str, int]`
  - `contraindre(vecteur, prix: bool = False, fond: bool = False) -> np.ndarray`
  - `valeurs(vecteur, prix: bool = False, fond: bool = False) -> dict[str, float]`
  - `coefficients(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, ...]` (8 valeurs)
  - `seuils_features(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, ...]` (8 valeurs, ordre moteur)
  - `globaux(vecteur, prix: bool = False, fond: bool = False) -> tuple[float, float]` (achat, vente)
  - `dict_extras_prix(vecteur, prix: bool = False, fond: bool = False) -> dict | None`
  - `dict_extras_fondamentaux(vecteur, prix: bool = False, fond: bool = False) -> dict | None`
  - `vers_colonnes(vecteur, prix: bool = False, fond: bool = False) -> dict[str, float]`
  - `depuis_colonnes(row, prix: bool = False, fond: bool = False) -> np.ndarray`
  - Constantes : `VECTEUR_BASE`, `SEUILS_GELES`, `EXTRAS_PRIX`, `EXTRAS_FONDAMENTAUX`, `ORDRE_SEUILS_MOTEUR`, `DRAPEAUX`

- [ ] **Step 1: Écrire les tests, qui échouent**

Créer `stock-analysis-ui/src/tests/test_optim_params.py` :

```python
"""
Verrouillage du contrat de parametres de l'optimisateur hybride.

Les bornes etaient decrites trois fois (recherche, bridage a l'evaluation,
bridage a la sauvegarde) et avaient diverge sur quatre parametres : le vecteur
ecrit en base n'etait donc pas celui qui avait produit le score. Ce module est
desormais la seule description, et ces tests la verrouillent.
"""
import numpy as np
import pytest

from core import optim_params as params


def test_dimensions_selon_les_drapeaux() -> None:
    """Verrouille les tailles de vecteur : 14, 25, 25 et 36."""
    assert len(params.bornes()) == 14
    assert len(params.bornes(prix=True)) == 25
    assert len(params.bornes(fond=True)) == 25
    assert len(params.bornes(prix=True, fond=True)) == 36


def test_toute_spec_du_vecteur_a_des_bornes() -> None:
    """Un parametre cherche sans bornes rendrait contraindre() incoherent."""
    for spec in params.specs(prix=True, fond=True):
        assert spec.bornes is not None, spec.cle
        bas, haut = spec.bornes
        assert bas < haut, spec.cle


def test_colonnes_db_uniques() -> None:
    """Deux parametres sur la meme colonne s'ecraseraient a la sauvegarde."""
    colonnes = [s.colonne_db for s in
                params.specs(prix=True, fond=True) + params.SEUILS_GELES]
    assert len(colonnes) == len(set(colonnes))


def test_contraindre_ramene_dans_les_bornes() -> None:
    """Une valeur hors bornes est ramenee au bord, jamais laissee dehors."""
    hautes = np.full(36, 1e6)
    basses = np.full(36, -1e6)
    for vecteur in (hautes, basses):
        sortie = params.contraindre(vecteur, prix=True, fond=True)
        for valeur, spec in zip(sortie, params.specs(prix=True, fond=True)):
            bas, haut = spec.bornes
            assert bas <= valeur <= haut, spec.cle


def test_contraindre_est_idempotente() -> None:
    """contraindre(contraindre(v)) == contraindre(v)."""
    brut = np.linspace(-50, 50, 36)
    une = params.contraindre(brut, prix=True, fond=True)
    deux = params.contraindre(une, prix=True, fond=True)
    assert np.array_equal(une, deux)


def test_contraindre_arrondit_les_drapeaux() -> None:
    """Les drapeaux valent 0 ou 1, jamais une valeur intermediaire."""
    idx = params.indices(prix=True, fond=True)
    brut = np.zeros(36)
    brut[idx['use_price_extras']] = 0.7
    brut[idx['use_fundamentals']] = 0.2
    sortie = params.contraindre(brut, prix=True, fond=True)
    assert sortie[idx['use_price_extras']] == 1.0
    assert sortie[idx['use_fundamentals']] == 0.0


def test_contraindre_refuse_une_taille_incoherente() -> None:
    """Une taille fausse doit lever, pas produire un vecteur tronque."""
    with pytest.raises(ValueError):
        params.contraindre(np.zeros(14), prix=True, fond=True)


def test_borne_basse_des_poids_prix_preservee() -> None:
    """Regression B3 : l'ancien bridage a l'evaluation ramenait -1.5 a -0.5.

    Le poids a_price_slope a pour bornes (-1.5, 3.0). L'ancien code le bridait
    a (-0.5, 3.0) a l'evaluation puis a (0.0, 3.0) a la sauvegarde, donc le
    vecteur sauvegarde n'etait pas celui qui avait produit le score.
    """
    idx = params.indices(prix=True)
    brut = np.zeros(25)
    brut[idx['a_price_slope']] = -1.5
    sortie = params.contraindre(brut, prix=True)
    assert sortie[idx['a_price_slope']] == -1.5


def test_th_score_garde_ses_bornes_de_recherche() -> None:
    """Regression B3, quatrieme instance : bornes (2.0, 6.0), pas (1.0, 6.0)."""
    spec = next(s for s in params.specs() if s.cle == 'th_score')
    assert spec.bornes == (2.0, 6.0)


def test_seuils_features_dans_l_ordre_du_moteur() -> None:
    """Les 8 seuils sortent dans l'ordre attendu par domain_thresholds,
    avec les geles a leur place : MACD, EMA, Ichimoku a 0, Bollinger a 0.5."""
    idx = params.indices()
    vecteur = np.zeros(14)
    vecteur[idx['th_rsi']] = 55.0
    vecteur[idx['th_vol']] = 1.4
    vecteur[idx['th_adx']] = 22.0
    vecteur[idx['th_score']] = 4.5
    seuils = params.seuils_features(params.contraindre(vecteur))
    assert len(seuils) == 8
    assert seuils[0] == 55.0    # RSI
    assert seuils[1] == 0.0     # MACD gele
    assert seuils[2] == 0.0     # EMA gele
    assert seuils[3] == 1.4     # Volume
    assert seuils[4] == 22.0    # ADX
    assert seuils[5] == 0.0     # Ichimoku gele
    assert seuils[6] == 0.5     # Bollinger gele
    assert seuils[7] == 4.5     # Score


def test_aller_retour_colonnes_sans_perte() -> None:
    """vers_colonnes puis depuis_colonnes rend le meme vecteur."""
    brut = params.contraindre(np.linspace(-2, 2, 36), prix=True, fond=True)
    colonnes = params.vers_colonnes(brut, prix=True, fond=True)
    retour = params.depuis_colonnes(colonnes, prix=True, fond=True)
    assert np.allclose(retour, brut)


def test_vers_colonnes_ecrit_les_seuils_geles() -> None:
    """Les seuils geles doivent partir en base, pas rester absents."""
    colonnes = params.vers_colonnes(np.zeros(14))
    assert colonnes['th2'] == 0.0
    assert colonnes['th3'] == 0.0
    assert colonnes['th6'] == 0.0
    assert colonnes['th7'] == 0.5


def test_depuis_colonnes_tolere_les_colonnes_absentes() -> None:
    """Un run historique ancien n'a pas toutes les colonnes ; le vecteur
    reconstruit doit rester dans le domaine valide."""
    retour = params.depuis_colonnes({'a1': 1.0}, prix=True, fond=True)
    assert len(retour) == 36
    for valeur, spec in zip(retour, params.specs(prix=True, fond=True)):
        bas, haut = spec.bornes
        assert bas <= valeur <= haut, spec.cle


def test_indices_decalent_les_fondamentaux_selon_le_prix() -> None:
    """Remplace l'offset magique `25 if use_price_features else 14`."""
    assert params.indices(fond=True)['a_rev_growth'] == 15
    assert params.indices(prix=True, fond=True)['a_rev_growth'] == 26


def test_coefficients_et_globaux() -> None:
    """Extraction des 8 coefficients et des 2 seuils globaux."""
    idx = params.indices()
    vecteur = np.zeros(14)
    for i in range(1, 9):
        vecteur[idx[f'a{i}']] = float(i) / 10.0
    vecteur[idx['seuil_achat']] = 4.2
    vecteur[idx['seuil_vente']] = -2.0
    contraint = params.contraindre(vecteur)
    assert params.coefficients(contraint) == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    assert params.globaux(contraint) == (4.2, -2.0)


def test_dicts_extras_absents_si_drapeau_desactive() -> None:
    """Sans features prix, aucun dict prix ne doit etre fabrique."""
    assert params.dict_extras_prix(np.zeros(14)) is None
    assert params.dict_extras_fondamentaux(np.zeros(14)) is None


def test_dict_extras_prix_porte_les_cles_du_moteur() -> None:
    """Les cles doivent etre exactement celles que lit le backtest."""
    attendu = {
        'use_price_extras', 'a_price_slope', 'a_price_acc', 'th_price_slope',
        'th_price_acc', 'a_price_rsi_slope', 'a_price_vol_slope',
        'a_price_var5j', 'th_price_rsi_slope', 'th_price_vol_slope',
        'th_price_var5j',
    }
    dico = params.dict_extras_prix(np.zeros(25), prix=True)
    assert set(dico) == attendu
    assert isinstance(dico['use_price_extras'], int)


def test_dict_extras_fondamentaux_porte_les_cles_du_moteur() -> None:
    attendu = {
        'use_fundamentals', 'a_rev_growth', 'a_eps_growth', 'a_roe',
        'a_fcf_yield', 'a_de_ratio', 'th_rev_growth', 'th_eps_growth',
        'th_roe', 'th_fcf_yield', 'th_de_ratio',
    }
    dico = params.dict_extras_fondamentaux(np.zeros(25), fond=True)
    assert set(dico) == attendu
    assert isinstance(dico['use_fundamentals'], int)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'core.optim_params'`

- [ ] **Step 3: Écrire le module**

Créer `stock-analysis-ui/src/core/optim_params.py` :

```python
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
            bas, haut = spec.bornes
            valeur = (bas + haut) / 2.0
        brut.append(float(valeur))
    return contraindre(np.asarray(brut, dtype=float), prix, fond)
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py -q`
Expected: PASS, 18 tests

- [ ] **Step 5: Commit**

```bash
git add stock-analysis-ui/src/core/optim_params.py stock-analysis-ui/src/tests/test_optim_params.py
git commit -m "feat(optim): contrat unique du vecteur de parametres

Les bornes etaient decrites trois fois et avaient diverge sur quatre
parametres, dont th_score, non releve par l'audit. Le vecteur ecrit en base
n'etait donc pas celui qui avait produit le score."
```

---

### Task 2: Garde-fou contre une bibliothèque C instrumentée

**Files:**
- Modify: `stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py:26-88`
- Test: `stock-analysis-ui/src/tests/test_c_module_guard.py`

**Interfaces:**
- Consumes: rien.
- Produces, dans `qsi_optimized` :
  - `_so_instrumente(chemin: str) -> bool`
  - `_binaires_candidats(module_name: str) -> list[str]`

- [ ] **Step 1: Écrire les tests, qui échouent**

Créer `stock-analysis-ui/src/tests/test_c_module_guard.py` :

```python
"""
Verrouillage du garde-fou de chargement du module C.

Le .so du 2026-04-18 avait ete compile avec AddressSanitizer. Son chargement ne
leve pas une exception : il fait AVORTER le processus (« ASan runtime does not
come first in initial library list »), donc le try/except de _diagnose_import
est impuissant. Consequence mesuree : `python optimisateur_hybride.py` mourait a
l'import, sans message Python. Le refus doit donc avoir lieu AVANT le dlopen.
"""
from trading_c_acceleration import qsi_optimized


def test_binaire_instrumente_detecte(tmp_path) -> None:
    """Un binaire portant les symboles ASan est reconnu."""
    faux = tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so"
    faux.write_bytes(b"\x7fELF" + b"\x00" * 64 + b"__asan_init" + b"\x00" * 16)

    assert qsi_optimized._so_instrumente(str(faux)) is True


def test_binaire_propre_accepte(tmp_path) -> None:
    """Un binaire sans symbole ASan passe."""
    propre = tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so"
    propre.write_bytes(b"\x7fELF" + b"\x00" * 512)

    assert qsi_optimized._so_instrumente(str(propre)) is False


def test_fichier_absent_ne_leve_pas(tmp_path) -> None:
    """Un chemin invalide ne doit pas casser le chargement."""
    assert qsi_optimized._so_instrumente(str(tmp_path / "absent.so")) is False


def test_candidats_filtres_par_nom_et_extension(monkeypatch, tmp_path) -> None:
    """Seules les bibliotheques compilees du bon module sont candidates."""
    (tmp_path / "trading_c.cpython-310-x86_64-linux-gnu.so").write_bytes(b"x")
    (tmp_path / "trading_c.pyd").write_bytes(b"x")
    (tmp_path / "autre_module.so").write_bytes(b"x")
    (tmp_path / "trading_c.py").write_text("# pas un binaire")
    monkeypatch.setattr(qsi_optimized, "__file__", str(tmp_path / "qsi_optimized.py"))

    trouves = {p.rsplit("/", 1)[-1] for p in qsi_optimized._binaires_candidats("trading_c")}

    assert trouves == {"trading_c.cpython-310-x86_64-linux-gnu.so", "trading_c.pyd"}
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_c_module_guard.py -q`
Expected: FAIL, `AttributeError: module 'trading_c_acceleration.qsi_optimized' has no attribute '_so_instrumente'`

- [ ] **Step 3: Ajouter les deux helpers**

Dans `stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py`, juste après le bloc `_is_c_acceleration_disabled` (ligne 33), insérer :

```python
def _so_instrumente(chemin: str) -> bool:
    """
    --------------------------------------------------------------------------
    Objectif:
        Dire si une bibliotheque compilee embarque AddressSanitizer, sans la
        charger.

        Un binaire construit avec -fsanitize=address ne leve pas : il fait
        avorter le processus au dlopen. Aucun try/except ne l'attrape, il faut
        donc l'ecarter avant. Constate le 2026-08-05 : le .so du 2026-04-18
        tuait l'import de optimisateur_hybride.py.

    Inputs:
        chemin (str): chemin de la bibliotheque

    Outputs:
        instrumente (bool): False si le fichier est illisible
    --------------------------------------------------------------------------
    """
    try:
        with open(chemin, 'rb') as binaire:
            return b'__asan_' in binaire.read()
    except OSError:
        return False


def _binaires_candidats(module_name: str) -> list[str]:
    """Bibliotheques compilees du dossier de ce module portant ce nom."""
    dossier = os.path.dirname(os.path.abspath(__file__))
    try:
        noms = os.listdir(dossier)
    except OSError:
        return []
    return [
        os.path.join(dossier, nom) for nom in noms
        if nom.startswith(module_name) and nom.endswith(('.so', '.pyd', '.dll'))
    ]
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_c_module_guard.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Brancher le garde-fou sur le chemin d'import**

Dans le même fichier, remplacer le bloc d'import (lignes 76 à 84 aujourd'hui) :

```python
# Try import with diagnostics (unless explicitly disabled)
if _is_c_acceleration_disabled():
    trading_c, C_ACCELERATION = None, False
    print("⚠️ Accélération C désactivée via QSI_DISABLE_C_ACCELERATION=1")
else:
    trading_c, C_ACCELERATION = _diagnose_import('trading_c')
    if not C_ACCELERATION:
        print("⚠️ Module C non disponible - Mode Python standard")
        print("   Compilez avec: python setup.py build_ext --inplace")
```

par :

```python
# Try import with diagnostics (unless explicitly disabled)
if _is_c_acceleration_disabled():
    trading_c, C_ACCELERATION = None, False
    print("⚠️ Accélération C désactivée via QSI_DISABLE_C_ACCELERATION=1")
else:
    _instrumentes = [c for c in _binaires_candidats('trading_c') if _so_instrumente(c)]
    if _instrumentes:
        # Refus AVANT le dlopen : un binaire ASan abort le processus.
        trading_c, C_ACCELERATION = None, False
        print("⚠️ Module C ignoré : binaire compilé avec AddressSanitizer")
        for _binaire in _instrumentes:
            print(f"   {_binaire}")
        print("   Recompilez sans QSI_DEBUG_C_MODE ni QSI_USE_ASAN :")
        print("   python setup.py build_ext --inplace")
    else:
        trading_c, C_ACCELERATION = _diagnose_import('trading_c')
        if not C_ACCELERATION:
            print("⚠️ Module C non disponible - Mode Python standard")
            print("   Compilez avec: python setup.py build_ext --inplace")
```

- [ ] **Step 6: Vérifier que le CLI ne meurt plus à l'import**

Run: `cd stock-analysis-ui/src && ../../.venv_new/bin/python -c "import optimisateur_hybride; print('import OK')"`
Expected: affiche le refus du binaire instrumenté puis `import OK`. Avant ce correctif, la commande n'affichait rien et le processus avortait.

- [ ] **Step 7: Recompiler proprement**

Run: `cd /home/berkam/Projets/Gestion_trade && .venv_new/bin/python setup.py`
Expected: compilation avec `-O3 -march=native -ffast-math -funroll-loops`, sans mention de DEBUG MODE.

Puis vérifier :
Run: `cd stock-analysis-ui/src && ../../.venv_new/bin/python -c "from trading_c_acceleration.qsi_optimized import C_ACCELERATION; print('C =', C_ACCELERATION)"`
Expected: `C = True`

Si la compilation échoue faute de compilateur, laisser le `.so` instrumenté en place : le garde-fou le neutralise et le chemin Python prend le relais. Noter l'échec dans le commit.

- [ ] **Step 8: Vérifier que la suite reste verte**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS, aucun test cassé par la recompilation, le conftest gardant le chemin Python.

- [ ] **Step 9: Commit**

```bash
git add stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py \
        stock-analysis-ui/src/tests/test_c_module_guard.py
git commit -m "fix(c): refuser un binaire ASan avant le dlopen

Le .so du 2026-04-18 etait compile avec -fsanitize=address. Son chargement
avorte le processus au lieu de lever, donc le try/except de _diagnose_import
etait impuissant et python optimisateur_hybride.py mourait a l'import."
```

---

### Task 3: Brancher le contrat dans `HybridOptimizer`

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:15-27` (imports), `:522-585` (bounds), `:594-701` (extraction), `:852-869` (croisement)
- Test: `stock-analysis-ui/src/tests/test_optim_params.py` (ajout)

**Interfaces:**
- Consumes: `core.optim_params` (Task 1), toutes les fonctions listées en Task 1.
- Produces: `HybridOptimizer.bounds` dérivé de `params.bornes()`, `HybridOptimizer.evaluate_config` bridant via `params.contraindre`.

- [ ] **Step 1: Écrire le test de non-régression des dimensions**

Ajouter à la fin de `stock-analysis-ui/src/tests/test_optim_params.py` :

```python
def test_optimiseur_derive_ses_bornes_du_contrat() -> None:
    """Verrouille : HybridOptimizer.bounds vient du contrat, pas d'une copie.

    Les assertions historiques de test_fundamentals_integration.py attendaient
    18, 24, 28 et 34 bornes alors que le code en produisait 14, 25, 25 et 36.
    """
    import numpy as np
    import pandas as pd

    from optimisateur_hybride import HybridOptimizer

    donnees = {
        'AAA': {
            'Close': pd.Series(np.linspace(100.0, 120.0, 120)),
            'Volume': pd.Series(np.full(120, 1_000_000.0)),
        }
    }
    for prix, fond, attendu in (
        (False, False, 14), (True, False, 25), (False, True, 25), (True, True, 36),
    ):
        optimiseur = HybridOptimizer(
            donnees, 'Technology_Large',
            use_price_features=prix, use_fundamentals_features=fond,
        )
        assert len(optimiseur.bounds) == attendu
        assert optimiseur.bounds == params.bornes(prix=prix, fond=fond)
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py::test_optimiseur_derive_ses_bornes_du_contrat -q`
Expected: FAIL, l'égalité `optimiseur.bounds == params.bornes(...)` est fausse, `bounds` étant encore une liste construite à la main.

- [ ] **Step 3: Importer le contrat**

Dans `stock-analysis-ui/src/optimisateur_hybride.py`, après la ligne 16, ajouter :

```python
from core import optim_params as params
```

- [ ] **Step 4: Remplacer la construction des bornes**

Remplacer tout le bloc des lignes 522 à 572, du commentaire `# 🔧 Définir les bounds avec dimension réduite` jusqu'à `self.bounds += fundamentals_bounds`, par :

```python
        # Bornes derivees du contrat unique (core/optim_params.py). Elles
        # etaient auparavant decrites ici, puis re-decrites plus etroitement a
        # l'evaluation et a la sauvegarde, ce qui faisait diverger le vecteur
        # sauvegarde du vecteur evalue.
        self.bounds = params.bornes(prix=self.use_price_features,
                                    fond=self.use_fundamentals_features)
```

- [ ] **Step 5: Remplacer l'extraction et le bridage dans `evaluate_config`**

Remplacer le bloc des lignes 594 à 701, de `def evaluate_config` jusqu'à la fin de la construction de `fundamentals_extras`, par :

```python
    def evaluate_config(self, params_vecteur):
        """Évalue une configuration. Le bridage passe par le contrat unique."""
        prix = self.use_price_features
        fond = self.use_fundamentals_features

        vecteur = params.contraindre(
            self.round_params(params_vecteur), prix=prix, fond=fond)
        param_key = tuple(vecteur)
        if param_key in self.best_cache:
            return self.best_cache[param_key]

        coeffs = params.coefficients(vecteur, prix=prix, fond=fond)
        feature_thresholds = params.seuils_features(vecteur, prix=prix, fond=fond)
        seuil_achat, seuil_vente = params.globaux(vecteur, prix=prix, fond=fond)
        price_extras = params.dict_extras_prix(vecteur, prix=prix, fond=fond)
        fundamentals_extras = params.dict_extras_fondamentaux(
            vecteur, prix=prix, fond=fond)
```

Le reste de la méthode, du `total_gain = 0.0` jusqu'au `return`, est conservé tel quel à ce stade. Task 4 y branchera les seuils.

- [ ] **Step 6: Borner les enfants du croisement génétique**

Remplacer la fin de `_crossover`, ligne 869, `return child1, child2`, par :

```python
        # BLX-alpha etend l'intervalle parental, donc les enfants peuvent sortir
        # des bornes. Sans ce bridage, le meilleur individu retourne par le GA
        # pouvait etre hors domaine et partir en base tel quel.
        prix = self.use_price_features
        fond = self.use_fundamentals_features
        return (params.contraindre(child1, prix=prix, fond=fond),
                params.contraindre(child2, prix=prix, fond=fond))
```

- [ ] **Step 7: Lancer les tests**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py -q`
Expected: PASS, 19 tests

- [ ] **Step 8: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py stock-analysis-ui/src/tests/test_optim_params.py
git commit -m "refactor(optim): HybridOptimizer derive tout du contrat

Supprime les trois descriptions concurrentes des bornes et l'offset magique
des fondamentaux. Borne aussi les enfants du croisement genetique, qui
pouvaient sortir du domaine et etre sauvegardes ainsi."
```

---

### Task 4: Rendre les 4 seuils effectifs

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:708-726` (`evaluate_symbol`)
- Test: `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (créer)

**Interfaces:**
- Consumes: `params.seuils_features` (Task 1), `evaluate_config` remanié (Task 3).
- Produces: `HybridOptimizer.evaluate_config` appelle `backtest_signals_with_events` avec `domain_thresholds`, jamais `backtest_signals_c_extended`.

- [ ] **Step 1: Écrire le test, qui échoue**

Créer `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` :

```python
"""
Verrouillage de l'objectif de l'optimisateur et de sa sauvegarde.

Deux defauts sont verrouilles ici :
  B1, les 4 seuils optimises (RSI, Volume, ADX, Score) etaient calcules puis
  jamais transmis au backtest, car backtest_signals_c_extended n'a pas de
  parametre de seuils ; ils partaient pourtant en base et pilotaient les
  signaux reels.
  B2, les `trades` et le `success_rate` sauvegardes venaient de la meilleure
  configuration jamais vue, pas du vecteur reellement sauvegarde.

Aucun acces reseau : le backtest est simule.
"""
import numpy as np
import pandas as pd
import pytest

import optimisateur_hybride as oh
from core import optim_params as params


def _donnees(nb_symboles: int = 2, longueur: int = 120) -> dict:
    serie = pd.Series(np.linspace(100.0, 130.0, longueur))
    volume = pd.Series(np.full(longueur, 1_000_000.0))
    return {f"SYM{i}": {'Close': serie, 'Volume': volume}
            for i in range(nb_symboles)}


def test_les_seuils_sont_transmis_au_backtest(monkeypatch) -> None:
    """Verrouille B1 : domain_thresholds recoit les 8 seuils du vecteur."""
    recus = []

    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                      domain_coeffs=None, domain_thresholds=None, **kwargs):
        recus.append(domain_thresholds)
        return {'gain_total': 10.0, 'trades': 2, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events', faux_backtest)

    optimiseur = oh.HybridOptimizer(_donnees(1), 'Technology_Large')
    idx = params.indices()
    vecteur = np.zeros(14)
    vecteur[idx['th_rsi']] = 55.0
    vecteur[idx['th_vol']] = 1.4
    vecteur[idx['th_adx']] = 22.0
    vecteur[idx['th_score']] = 4.5
    vecteur[idx['seuil_achat']] = 4.2
    vecteur[idx['seuil_vente']] = -2.0

    optimiseur.evaluate_config(vecteur)

    assert recus, "le backtest n'a pas ete appele"
    seuils = list(recus[0].values())[0]
    assert seuils[0] == 55.0    # RSI
    assert seuils[3] == 1.4     # Volume
    assert seuils[4] == 22.0    # ADX
    assert seuils[7] == 4.5     # Score
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: FAIL, `AttributeError: module 'optimisateur_hybride' has no attribute 'backtest_signals_with_events'`, le module n'important aujourd'hui que `backtest_signals` et `backtest_signals_with_events`. Si l'import existe déjà, l'échec est `assert recus`, le backtest appelé étant `backtest_signals_c_extended`.

- [ ] **Step 3: Épingler l'objectif sur le chemin qui honore les seuils**

Dans `evaluate_symbol`, remplacer l'appel des lignes 714 à 722 :

```python
                result = backtest_signals_c_extended(
                    ser_data['Close'], ser_data['Volume'],
                    coeffs=coeffs,
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    montant=self.montant, transaction_cost=self.transaction_cost,
                    price_extras=price_extras if self.use_price_features else None,
                    fundamentals_extras=fundamentals_extras if self.use_fundamentals_features else None,
                    symbol_name=symbol
                )
                return result['gain_total'], result['trades'], result.get('gagnants', 0)
```

par :

```python
                # backtest_signals_c_extended n'a AUCUN parametre de seuils :
                # py_backtest_symbol ne prend que (prices, volumes, coeffs,
                # montant, cost). Les 4 seuils optimises y etaient donc perdus.
                # with_events les honore via domain_thresholds, pour +1 % de
                # cout mesure le 2026-08-05.
                result, _evenements = backtest_signals_with_events(
                    ser_data['Close'], ser_data['Volume'], "default",
                    self.montant, self.transaction_cost,
                    domain_coeffs={"default": coeffs},
                    domain_thresholds={"default": feature_thresholds},
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    extra_params=price_extras,
                    fundamentals_extras=fundamentals_extras,
                    symbol_name=symbol,
                )
                return result['gain_total'], result['trades'], result.get('gagnants', 0)
```

Note pour l'implémenteur : le paramètre des extras prix s'appelle `extra_params` dans `backtest_signals_with_events` et `price_extras` dans `backtest_signals_c_extended`. Ne pas les confondre.

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: PASS, 1 test

- [ ] **Step 5: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py stock-analysis-ui/src/tests/test_optim_sauvegarde.py
git commit -m "fix(optim): rendre effectifs les 4 seuils optimises

Ils etaient calcules puis jetes, le moteur C n'ayant pas de parametre de
seuils, et partaient pourtant en base pour piloter les signaux reels."
```

---

### Task 5: Rattacher les métriques à leur vecteur

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:496-520` (`__init__`), `:594-774` (`evaluate_config`), `:1619-1622` (lecture des métriques)
- Test: `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (ajout)

**Interfaces:**
- Consumes: `params.contraindre` (Task 1), objectif épinglé (Task 4).
- Produces:
  - `Mesure = namedtuple('Mesure', 'score gain_moyen trades gagnants')`, au niveau module
  - `HybridOptimizer.mesures` : cache borné, clé `tuple(vecteur contraint)`
  - `HybridOptimizer.mesure_de(vecteur) -> Mesure`

- [ ] **Step 1: Écrire le test, qui échoue**

Ajouter à `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` :

```python
def test_les_metriques_suivent_leur_vecteur(monkeypatch) -> None:
    """Verrouille B2 : mesure_de() rend les metriques DU vecteur demande.

    Le faux backtest renvoie un gain proportionnel au premier coefficient, donc
    deux vecteurs ont des metriques distinctes et une confusion se voit.
    """
    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                      domain_coeffs=None, domain_thresholds=None, **kwargs):
        premier = list(domain_coeffs.values())[0][0]
        trades = 2 if premier > 0 else 8
        return {'gain_total': premier * 100.0, 'trades': trades, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events', faux_backtest)

    optimiseur = oh.HybridOptimizer(_donnees(2), 'Technology_Large')
    idx = params.indices()
    bon = np.zeros(14); bon[idx['a1']] = 2.0
    mauvais = np.zeros(14); mauvais[idx['a1']] = -1.0
    optimiseur.evaluate_config(bon)
    optimiseur.evaluate_config(mauvais)

    assert optimiseur.mesure_de(bon).trades == 4      # 2 symboles x 2 trades
    assert optimiseur.mesure_de(mauvais).trades == 16  # 2 symboles x 8 trades


def test_mesure_de_evalue_si_absente(monkeypatch) -> None:
    """Verrouille le cas du DE en sous-processus : la mesure manque au parent,
    mesure_de() doit alors evaluer au lieu de rendre zero."""
    appels = []

    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                      domain_coeffs=None, domain_thresholds=None, **kwargs):
        appels.append(1)
        return {'gain_total': 5.0, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events', faux_backtest)

    optimiseur = oh.HybridOptimizer(_donnees(1), 'Technology_Large')
    mesure = optimiseur.mesure_de(np.zeros(14))   # jamais evalue avant

    assert appels, "mesure_de aurait du declencher une evaluation"
    assert mesure.trades == 3
    assert mesure.gagnants == 2
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: FAIL, `AttributeError: 'HybridOptimizer' object has no attribute 'mesure_de'`

- [ ] **Step 3: Déclarer `Mesure` et le cache borné**

Dans `stock-analysis-ui/src/optimisateur_hybride.py`, après les imports, ajouter :

```python
from collections import namedtuple

from core.cache import _BoundedCache

# Metriques d'UNE configuration. Elles doivent voyager avec leur vecteur :
# la sauvegarde lisait auparavant `meilleur_trades`, qui appartenait a la
# meilleure configuration jamais vue, pas a celle qui etait ecrite en base.
Mesure = namedtuple('Mesure', 'score gain_moyen trades gagnants')
```

Dans `HybridOptimizer.__init__`, remplacer `self.best_cache = {}` par :

```python
        # Borne pour ne pas croitre indefiniment sur des dizaines de milliers
        # d'evaluations. Cle : tuple du vecteur contraint.
        self.mesures = _BoundedCache(maxsize=4096)
```

- [ ] **Step 4: Enregistrer la mesure dans `evaluate_config`**

Dans `evaluate_config`, remplacer tous les usages restants de `self.best_cache` et le calcul final. Le bloc de sortie devient :

```python
            if total_trades == 0:
                mesure = Mesure(score=-1e6, gain_moyen=avg_gain, trades=0, gagnants=0)
                self.mesures[param_key] = mesure
                return mesure.score

            n_symbols = len(self.stock_data)
            trades_per_symbol = total_trades / n_symbols if n_symbols > 0 else 0
            score = avg_gain - self.trade_efficiency_penalty * trades_per_symbol

            if self.optimization_mode == 'taux_reussite':
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
                if success_rate < 50.0 or avg_gain <= 0:
                    mesure = Mesure(score=-1e6, gain_moyen=avg_gain,
                                    trades=total_trades, gagnants=total_success)
                    self.mesures[param_key] = mesure
                    return mesure.score

            mesure = Mesure(score=score, gain_moyen=avg_gain,
                            trades=total_trades, gagnants=total_success)
            self.mesures[param_key] = mesure

            # meilleur_* ne sert plus qu'a l'affichage tqdm. Ne rien en deduire
            # pour la sauvegarde : sous workers=-1 ces compteurs restent dans
            # les sous-processus.
            if score > self.meilleur_score:
                self.meilleur_score = score
                self.meilleur_trades = total_trades
                self.meilleur_success = total_success

            return score
```

et la lecture du cache en tête de méthode devient :

```python
        param_key = tuple(vecteur)
        connue = self.mesures.get(param_key)
        if connue is not None:
            return connue.score
```

- [ ] **Step 5: Ajouter `mesure_de`**

Après `evaluate_config`, ajouter :

```python
    def mesure_de(self, vecteur) -> Mesure:
        """
        --------------------------------------------------------------------------
        Objectif:
            Rendre les metriques DU vecteur demande, en evaluant si elles
            manquent. C'est ce qui garantit que les `trades` ecrits en base
            appartiennent aux coefficients ecrits sur la meme ligne.

        Inputs:
            vecteur (Sequence[float]): vecteur de parametres

        Outputs:
            mesure (Mesure): score, gain moyen, trades, gagnants
        --------------------------------------------------------------------------
        """
        contraint = params.contraindre(
            self.round_params(vecteur),
            prix=self.use_price_features, fond=self.use_fundamentals_features)
        cle = tuple(contraint)
        connue = self.mesures.get(cle)
        if connue is None:
            self.evaluate_config(contraint)
            connue = self.mesures[cle]
        return connue
```

- [ ] **Step 6: Lire les métriques du bon vecteur à la sauvegarde**

Dans `optimize_sector_coefficients_hybrid`, remplacer les lignes 1619 à 1622 :

```python
    # Réutiliser les statistiques déjà calculées pendant l'optimisation (évite un recalcul lent)
    total_trades = optimizer.meilleur_trades
    total_success = optimizer.meilleur_success
    success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
```

par :

```python
    # Metriques DU vecteur retenu, jamais celles de la meilleure configuration
    # jamais vue. Avec workers=-1 les compteurs de l'optimiseur restaient a zero
    # et strategy='differential' ne sauvegardait donc jamais rien.
    mesure_retenue = optimizer.mesure_de(best_params)
    total_trades = mesure_retenue.trades
    total_success = mesure_retenue.gagnants
    best_score = mesure_retenue.score
    success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
```

- [ ] **Step 7: Lancer les tests**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: PASS, 3 tests

- [ ] **Step 8: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py stock-analysis-ui/src/tests/test_optim_sauvegarde.py
git commit -m "fix(optim): rattacher trades et success_rate a leur vecteur

Ils venaient de la meilleure configuration jamais vue. Sous workers=-1 les
compteurs restaient dans les sous-processus, donc strategy='differential' sur
un groupe sans historique ne sauvegardait jamais rien."
```

---

### Task 6: Sauvegarde et relecture par le contrat

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:1046-1122` (`replay_all_historical`), `:1537-1617` (extraction), `:1690-1834` (`save_optimization_results`)
- Test: `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (ajout)

**Interfaces:**
- Consumes: `params.vers_colonnes`, `params.depuis_colonnes` (Task 1), `mesure_de` (Task 5).
- Produces: `save_optimization_results(domain, vecteur, mesure, cap_range, prix, fond, transaction_cost, seed)`, signature nouvelle.

- [ ] **Step 1: Écrire le test, qui échoue**

Ajouter à `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` :

```python
def _base_vide(tmp_path):
    """Cree une base optimization_runs minimale et retourne son chemin."""
    import sqlite3
    chemin = tmp_path / "optimization_hist.db"
    conn = sqlite3.connect(chemin)
    conn.execute("""
        CREATE TABLE optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            sector TEXT NOT NULL,
            gain_moy REAL, success_rate REAL, trades INTEGER,
            seuil_achat REAL, seuil_vente REAL,
            a1 REAL, a2 REAL, a3 REAL, a4 REAL, a5 REAL, a6 REAL, a7 REAL, a8 REAL,
            th1 REAL, th2 REAL, th3 REAL, th4 REAL, th5 REAL, th6 REAL, th7 REAL, th8 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            market_cap_range TEXT,
            UNIQUE(sector, timestamp)
        )
    """)
    conn.commit()
    conn.close()
    return str(chemin)


def test_sauvegarde_ecrit_le_vecteur_et_ses_metriques(monkeypatch, tmp_path) -> None:
    """Verrouille l'aller-retour : ce qui est ecrit se relit a l'identique."""
    import sqlite3

    import config

    chemin = _base_vide(tmp_path)
    # save_optimization_results fait `from config import OPTIMIZATION_DB_PATH`
    # DANS son corps, donc patcher le module config suffit.
    monkeypatch.setattr(config, 'OPTIMIZATION_DB_PATH', chemin)

    idx = params.indices()
    vecteur = params.contraindre(np.zeros(14))
    vecteur[idx['a1']] = 1.25
    vecteur[idx['th_rsi']] = 62.0
    vecteur = params.contraindre(vecteur)
    mesure = oh.Mesure(score=12.5, gain_moyen=13.0, trades=7, gagnants=5)

    oh.save_optimization_results(
        'Technology_Large', vecteur, mesure, cap_range='Large',
        prix=False, fond=False, transaction_cost=1.0, seed=1234)

    conn = sqlite3.connect(chemin)
    conn.row_factory = sqlite3.Row
    ligne = conn.execute("SELECT * FROM optimization_runs").fetchone()
    conn.close()

    assert ligne['trades'] == 7
    assert ligne['a1'] == pytest.approx(1.25)
    assert ligne['th1'] == pytest.approx(62.0)
    assert ligne['th7'] == pytest.approx(0.5)     # Bollinger gele
    assert ligne['transaction_cost'] == pytest.approx(1.0)
    assert ligne['seed'] == 1234
    relu = params.depuis_colonnes(ligne)
    assert np.allclose(relu, vecteur)
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py::test_sauvegarde_ecrit_le_vecteur_et_ses_metriques -q`
Expected: FAIL, `TypeError`, la signature actuelle attendant `(domain, coeffs, gain_total, success_rate, total_trades, thresholds, ...)`

- [ ] **Step 3: Réécrire `save_optimization_results`**

Remplacer entièrement la fonction, lignes 1690 à 1834, par :

```python
def save_optimization_results(domain, vecteur, mesure, cap_range=None,
                              prix=False, fond=False,
                              transaction_cost=COUT_TRANSACTION_PAR_TRADE,
                              seed=None):
    """
    --------------------------------------------------------------------------
    Objectif:
        Ecrire une ligne de optimization_runs decrivant UNE configuration et
        les metriques de CETTE configuration.

    Inputs:
        domain (str): secteur ou cle composite secteur_capRange
        vecteur (Sequence[float]): vecteur de parametres, deja contraint
        mesure (Mesure): metriques du meme vecteur
        cap_range (str | None): segment de capitalisation
        prix (bool), fond (bool): drapeaux de features actives
        transaction_cost (float): cout absolu par trade utilise pour la mesure
        seed (int | None): graine du run, pour rejouabilite

    Outputs:
        None. Journalise l'echec sans le propager a l'appelant.
    --------------------------------------------------------------------------
    """
    from datetime import datetime
    import sqlite3
    from config import OPTIMIZATION_DB_PATH

    def _ensure_opt_runs_schema(conn):
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(optimization_runs)")
            cols = {row[1] for row in cur.fetchall()}
            nouvelles = [('market_cap_range', 'TEXT')]
            nouvelles += [(spec.colonne_db, 'REAL')
                          for spec in EXTRAS_COLONNES_MIGRATION]
            nouvelles += [('use_price_extras', 'INTEGER DEFAULT 0'),
                          ('use_fundamentals', 'INTEGER DEFAULT 0'),
                          ('transaction_cost', 'REAL'),
                          ('seed', 'INTEGER')]
            for nom, decl in nouvelles:
                if nom not in cols:
                    try:
                        cur.execute(
                            f"ALTER TABLE optimization_runs ADD COLUMN {nom} {decl}")
                    except Exception:
                        pass
            conn.commit()
        except Exception:
            # Non fatal : la retrocompatibilite prime sur la migration.
            pass

    normalized_cap = cap_range or 'Unknown'
    normalized_sector = domain
    allowed_caps = {'Small', 'Mid', 'Large', 'Mega', 'Unknown'}
    if '_' in domain:
        maybe_sector, maybe_cap = domain.rsplit('_', 1)
        if maybe_cap in allowed_caps:
            normalized_sector = maybe_sector
            if normalized_cap == 'Unknown':
                normalized_cap = maybe_cap

    colonnes = params.vers_colonnes(vecteur, prix=prix, fond=fond)
    colonnes.update({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sector': normalized_sector,
        'market_cap_range': normalized_cap,
        'gain_moy': float(mesure.score),
        'success_rate': (mesure.gagnants / mesure.trades * 100) if mesure.trades else 0.0,
        'trades': int(mesure.trades),
        'transaction_cost': float(transaction_cost),
        'seed': int(seed) if seed is not None else None,
    })

    conn = None
    try:
        conn = sqlite3.connect(OPTIMIZATION_DB_PATH)
        _ensure_opt_runs_schema(conn)
        noms = list(colonnes)
        marqueurs = ', '.join('?' for _ in noms)
        conn.execute(
            f"INSERT OR REPLACE INTO optimization_runs ({', '.join(noms)}) "
            f"VALUES ({marqueurs})",
            [colonnes[nom] for nom in noms],
        )
        conn.commit()
        print(f"📝 Résultats sauvegardés pour {normalized_sector} ({normalized_cap})")
    except Exception as exc:
        print(f"⚠️ Erreur lors de la sauvegarde: {exc}")
    finally:
        if conn is not None:
            conn.close()
```

Ajouter, près des constantes de module. La constante de coût est déclarée ici
parce que la signature ci-dessus s'en sert comme valeur par défaut ; la tâche 8
ne fera qu'unifier les appelants et corriger le docstring du moteur.

```python
# Cout par trade, en MONTANT absolu et non en pourcentage : le moteur calcule
# profit = (close - entry) / entry * montant - transaction_cost. Sur une
# position de 50 $, 1.0 vaut 2 %, ce qui est realiste pour un petit ordre au
# detail ; les 0.02 que passait le CLI valaient deux centimes et flattaient
# mecaniquement les configurations qui multiplient les trades.
COUT_TRANSACTION_PAR_TRADE = 1.0

# Colonnes d'extras a creer si la base precede leur introduction.
EXTRAS_COLONNES_MIGRATION = tuple(
    spec for spec in (params.EXTRAS_PRIX + params.EXTRAS_FONDAMENTAUX)
    if spec.cle not in params.DRAPEAUX
)
```

- [ ] **Step 4: Adapter l'appelant**

Dans `optimize_sector_coefficients_hybrid`, supprimer le bloc d'extraction des lignes 1537 à 1617, qui reconstruisait coefficients, seuils et dicts d'extras à la main, et remplacer l'appel de sauvegarde ligne 1669 par :

```python
    if should_save:
        save_optimization_results(
            domain, best_params, mesure_retenue, cap_range=cap_range,
            prix=use_price_features, fond=use_fundamentals_features,
            transaction_cost=transaction_cost, seed=seed)
```

Les affichages qui utilisaient `best_coeffs`, `best_feature_thresholds`, `extra_params` et `fundamentals_extras` prennent désormais leurs valeurs du contrat :

```python
    best_coeffs = params.coefficients(best_params, prix=use_price_features,
                                      fond=use_fundamentals_features)
    best_feature_thresholds = params.seuils_features(
        best_params, prix=use_price_features, fond=use_fundamentals_features)
    best_seuil_achat, best_seuil_vente = params.globaux(
        best_params, prix=use_price_features, fond=use_fundamentals_features)
    extra_params = params.dict_extras_prix(
        best_params, prix=use_price_features, fond=use_fundamentals_features)
    fundamentals_extras = params.dict_extras_fondamentaux(
        best_params, prix=use_price_features, fond=use_fundamentals_features)
    all_thresholds = best_feature_thresholds + (best_seuil_achat, best_seuil_vente)
```

- [ ] **Step 5: Simplifier `replay_all_historical`**

Remplacer le corps de la boucle `for row in rows:`, lignes 1053 à 1122, par :

```python
        for row in rows:
            try:
                vecteur = params.depuis_colonnes(
                    row, prix=use_price_features, fond=use_fundamentals_features)
                score = self.evaluate_config(vecteur)
                if score > best_score:
                    best_score = score
                    best_params = vecteur.copy()
                    best_label = f"Historical ({row['timestamp']})"
            except Exception:
                continue
```

La fonction locale `_read` et les listes de colonnes écrites à la main deviennent inutiles ; les supprimer.

- [ ] **Step 6: Lancer les tests**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: PASS, 4 tests

- [ ] **Step 7: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py stock-analysis-ui/src/tests/test_optim_sauvegarde.py
git commit -m "refactor(optim): sauvegarde et relecture par le contrat

La correspondance vers les colonnes SQL etait ecrite a la main deux fois, dans
save_optimization_results et a l'envers dans replay_all_historical. Ajoute les
colonnes transaction_cost et seed, et ferme les connexions en finally."
```

---

### Task 7: Budget d'évaluations respecté

**Files:**
- Create: `stock-analysis-ui/src/core/optim_budget.py`
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:882-908` (DE), `:1463-1505` (répartition), `:1958-2059` (menu et budget)
- Test: `stock-analysis-ui/src/tests/test_optim_budget.py`

**Interfaces:**
- Consumes: rien.
- Produces:
  - `budget_effectif(budget_base: int, precision: int) -> int`
  - `plan_differential(budget: int, dimension: int) -> tuple[int, int]` (multiplicateur popsize, maxiter)
  - `evaluations_differential(multiplicateur: int, maxiter: int, dimension: int) -> int`
  - `plan_genetique(budget: int) -> tuple[int, int]` (population, generations)
  - `plan_pso(budget: int) -> tuple[int, int]` (particules, iterations)
  - `plan_lhs(budget: int) -> int` (echantillons)
  - `REPARTITION_HYBRIDE: dict[str, float]`

- [ ] **Step 1: Écrire les tests, qui échouent**

Créer `stock-analysis-ui/src/tests/test_optim_budget.py` :

```python
"""
Verrouillage de l'arithmetique du budget d'evaluations.

`popsize` de SciPy est un MULTIPLICATEUR : la population vaut
popsize * dimension. En passant 200 avec 36 dimensions, l'optimisateur lancait
7 200 individus par generation, soit environ 1,45 million d'evaluations par
groupe au lieu des 75 000 du budget. Le menu affichait par ailleurs une
estimation calculee sur 3 500 quand le lancement utilisait 30 000.
"""
import pytest

from core import optim_budget as budget


@pytest.mark.parametrize("precision, attendu", [(1, 15000), (2, 30000), (3, 60000)])
def test_budget_effectif_par_precision(precision, attendu) -> None:
    """Une seule regle d'echelle, partagee par le menu et le lancement."""
    assert budget.budget_effectif(30000, precision) == attendu


def test_budget_effectif_refuse_une_precision_inconnue() -> None:
    with pytest.raises(ValueError):
        budget.budget_effectif(30000, 7)


@pytest.mark.parametrize("enveloppe", [200, 2000, 30000, 75000])
@pytest.mark.parametrize("dimension", [14, 25, 36])
def test_differential_ne_depasse_pas_le_budget(enveloppe, dimension) -> None:
    """population * (maxiter + 1) reste sous l'enveloppe."""
    multiplicateur, maxiter = budget.plan_differential(enveloppe, dimension)

    assert multiplicateur >= 1
    assert maxiter >= 1
    prevu = budget.evaluations_differential(multiplicateur, maxiter, dimension)
    assert prevu <= enveloppe, f"{prevu} > {enveloppe}"


def test_differential_utilise_une_part_utile_du_budget() -> None:
    """Un plan qui n'utiliserait que 1 % du budget serait inutile."""
    multiplicateur, maxiter = budget.plan_differential(30000, 36)
    prevu = budget.evaluations_differential(multiplicateur, maxiter, 36)

    assert prevu >= 30000 * 0.5


def test_regression_popsize_multiplicateur() -> None:
    """Regression B4 : l'ancien code passait popsize=200 avec 36 dimensions,
    soit 7 200 individus par generation."""
    multiplicateur, maxiter = budget.plan_differential(75000, 36)
    population = multiplicateur * 36

    assert population <= 75000
    assert population < 7200


@pytest.mark.parametrize("enveloppe", [200, 2000, 30000])
def test_plans_des_autres_strategies(enveloppe) -> None:
    """Chaque strategie annonce un nombre d'evaluations sous l'enveloppe."""
    population, generations = budget.plan_genetique(enveloppe)
    assert population * generations <= enveloppe
    assert population >= 4 and generations >= 1

    particules, iterations = budget.plan_pso(enveloppe)
    assert particules * (iterations + 1) <= enveloppe
    assert particules >= 4 and iterations >= 1

    echantillons = budget.plan_lhs(enveloppe)
    assert 0 < echantillons <= enveloppe


def test_repartition_hybride_somme_a_un() -> None:
    """En mode hybrid, le budget est reparti, pas pris en entier par chacune."""
    assert set(budget.REPARTITION_HYBRIDE) == {'differential', 'pso', 'lhs'}
    assert sum(budget.REPARTITION_HYBRIDE.values()) == pytest.approx(1.0)


def test_hybride_total_sous_le_budget() -> None:
    """La somme des trois plans reste sous le budget global."""
    enveloppe, dimension = 30000, 36
    part = budget.REPARTITION_HYBRIDE
    mult, maxiter = budget.plan_differential(int(enveloppe * part['differential']), dimension)
    particules, iterations = budget.plan_pso(int(enveloppe * part['pso']))
    echantillons = budget.plan_lhs(int(enveloppe * part['lhs']))

    total = (budget.evaluations_differential(mult, maxiter, dimension)
             + particules * (iterations + 1)
             + echantillons)
    assert total <= enveloppe
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_budget.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'core.optim_budget'`

- [ ] **Step 3: Écrire le module**

Créer `stock-analysis-ui/src/core/optim_budget.py` :

```python
"""
Arithmetique du budget d'evaluations de l'optimisateur hybride.

Une seule regle d'echelle selon la precision, et un plan par strategie qui
annonce son nombre d'evaluations. Le menu et le lancement lisent les memes
fonctions, ce qui supprime l'ecart entre le budget affiche et le budget utilise.

`popsize` de SciPy est un multiplicateur : la population vaut
popsize * dimension. L'ignorer faisait passer un groupe de 75 000 evaluations
prevues a environ 1,45 million.
"""
from __future__ import annotations

# Facteur d'echelle du budget selon le nombre de decimales cherchees.
FACTEUR_PRECISION: dict[int, float] = {1: 0.5, 2: 1.0, 3: 2.0}

# Repartition du budget entre strategies en mode hybrid.
REPARTITION_HYBRIDE: dict[str, float] = {
    'differential': 0.4,
    'pso': 0.3,
    'lhs': 0.3,
}


def budget_effectif(budget_base: int, precision: int) -> int:
    """
    --------------------------------------------------------------------------
    Objectif:
        Budget d'evaluations pour une precision donnee. Seule regle d'echelle
        du projet : l'ancien code la comptait deux fois, une fois dans le CLI
        et une fois dans optimize_sector_coefficients_hybrid.

    Inputs:
        budget_base (int): budget de reference
        precision (int): 1, 2 ou 3 decimales

    Outputs:
        budget (int)
    --------------------------------------------------------------------------
    """
    if precision not in FACTEUR_PRECISION:
        raise ValueError(
            f"precision {precision} inconnue, attendu {sorted(FACTEUR_PRECISION)}")
    return int(budget_base * FACTEUR_PRECISION[precision])


def plan_differential(enveloppe: int, dimension: int) -> tuple[int, int]:
    """
    --------------------------------------------------------------------------
    Objectif:
        Choisir (multiplicateur popsize, maxiter) tels que le nombre
        d'evaluations reste sous l'enveloppe.

    Inputs:
        enveloppe (int): evaluations autorisees
        dimension (int): taille du vecteur

    Outputs:
        (multiplicateur, maxiter) (tuple[int, int]): chacun au moins 1
    --------------------------------------------------------------------------
    """
    dimension = max(1, int(dimension))
    enveloppe = max(4, int(enveloppe))
    # Cible : une population large mais qui laisse au moins 10 generations.
    multiplicateur = max(1, min(15, enveloppe // (dimension * 10)))
    population = multiplicateur * dimension
    maxiter = max(1, enveloppe // population - 1)
    return multiplicateur, maxiter


def evaluations_differential(multiplicateur: int, maxiter: int, dimension: int) -> int:
    """Evaluations qu'un plan DE consommera : population initiale plus iterations."""
    return multiplicateur * dimension * (maxiter + 1)


def plan_genetique(enveloppe: int) -> tuple[int, int]:
    """(population, generations) sous l'enveloppe."""
    enveloppe = max(4, int(enveloppe))
    population = max(4, min(300, int(enveloppe ** 0.5)))
    generations = max(1, min(100, enveloppe // population))
    return population, generations


def plan_pso(enveloppe: int) -> tuple[int, int]:
    """(particules, iterations) sous l'enveloppe, l'initialisation comptant."""
    enveloppe = max(4, int(enveloppe))
    particules = max(4, min(100, int(enveloppe ** 0.5)))
    iterations = max(1, enveloppe // particules - 1)
    return particules, iterations


def plan_lhs(enveloppe: int) -> int:
    """Nombre d'echantillons, un par evaluation."""
    return max(1, min(1200, int(enveloppe)))
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_budget.py -q`
Expected: PASS, 20 tests

- [ ] **Step 5: Brancher le module dans les stratégies**

Dans `optimisateur_hybride.py`, ajouter l'import :

```python
from core import optim_budget
```

Remplacer `differential_evolution_opt`, lignes 882 à 908, par :

```python
    def differential_evolution_opt(self, enveloppe: int = 30000):
        """Évolution différentielle. L'enveloppe est un nombre d'évaluations."""
        bounds = self.bounds
        dimension = len(bounds)
        multiplicateur, max_iterations = optim_budget.plan_differential(enveloppe, dimension)
        prevu = optim_budget.evaluations_differential(multiplicateur, max_iterations, dimension)
        print(f"🔄 Évolution différentielle : population={multiplicateur * dimension}, "
              f"iterations={max_iterations}, évaluations prévues={prevu} "
              f"(enveloppe={enveloppe}, précision={self.precision})")

        with tqdm(total=max_iterations, desc="🔄 Évolution différentielle", unit="iter") as pbar:
            def callback(xk, convergence):
                pbar.set_postfix({
                    'Convergence': f"{convergence:.6f}",
                    'Score': f"{self.meilleur_score:.3f} ({self.meilleur_success_rate:.1f}%)",
                    'Trades': self.meilleur_trades,
                })
                pbar.update(1)

            result = differential_evolution(
                _de_objective,
                bounds,
                args=(self,),
                maxiter=max_iterations,
                popsize=multiplicateur,   # MULTIPLICATEUR, population = popsize * dimension
                mutation=(0.5, 1.5),
                recombination=0.7,
                callback=callback,
                polish=False,
                seed=self.seed,
                # workers=1 : sous multiprocessing, les mesures enregistrees par
                # evaluate_config restent dans les sous-processus, ce dont la
                # sauvegarde depend. La vitesse viendra du lot 2.
                workers=1,
            )

        return params.contraindre(
            result.x, prix=self.use_price_features,
            fond=self.use_fundamentals_features), -result.fun
```

Remplacer le bloc de choix des stratégies, lignes 1463 à 1505, par :

```python
    # Une seule regle d'echelle : le budget recu est deja final.
    print(f"🚀 Optimisation hybride pour {domain}, stratégie '{strategy}', "
          f"précision {precision}, budget {budget_evaluations} évaluations")

    results = []
    if historical_candidate:
        results.append(historical_candidate)

    dimension = len(optimizer.bounds)
    part = optim_budget.REPARTITION_HYBRIDE if strategy == 'hybrid' else None

    def _enveloppe(nom: str) -> int:
        return int(budget_evaluations * part[nom]) if part else budget_evaluations

    if strategy == 'genetic':
        population, generations = optim_budget.plan_genetique(budget_evaluations)
        params_ga, score_ga = optimizer.genetic_algorithm(population, generations)
        results.append(('Genetic Algorithm', params_ga, score_ga))

    if strategy in ('hybrid', 'differential'):
        params_de, score_de = optimizer.differential_evolution_opt(_enveloppe('differential'))
        results.append(('Differential Evolution', params_de, score_de))

    if strategy in ('hybrid', 'pso'):
        particules, iterations = optim_budget.plan_pso(_enveloppe('pso'))
        params_pso, score_pso = optimizer.particle_swarm_optimization(particules, iterations)
        results.append(('PSO', params_pso, score_pso))

    if strategy in ('hybrid', 'lhs'):
        params_lhs, score_lhs = optimizer.latin_hypercube_sampling(
            optim_budget.plan_lhs(_enveloppe('lhs')))
        results.append(('Latin Hypercube', params_lhs, score_lhs))
```

Supprimer les lignes 1463 à 1469 qui calculaient `precision_factor` et `adjusted_budget`, cette échelle étant désormais appliquée une seule fois, par `budget_effectif`, du côté de l'appelant.

- [ ] **Step 6: Aligner le menu sur le budget réel**

Dans `_show_config`, remplacer le calcul local du budget :

```python
        budget_base = 3500
        budget = int(budget_base * (0.5 if precision == 1 else (2.0 if precision == 3 else 1.0)))
```

par :

```python
        budget_affiche = optim_budget.budget_effectif(BUDGET_BASE, precision)
```

et l'affichage correspondant par :

```python
        print(f"  📊 {total_symbols} symboles · {total_to_optimize} groupes · "
              f"{param_count} params · {budget_affiche} éval/groupe")
```

Déclarer près des constantes de module :

```python
# Budget de reference, unique. Le menu et le lancement en derivent tous deux via
# optim_budget.budget_effectif(), qui affichaient auparavant 3 500 contre 30 000.
BUDGET_BASE = 30000
```

Et remplacer le calcul du budget final, lignes 2050 à 2056, par :

```python
    budget_evaluations = optim_budget.budget_effectif(BUDGET_BASE, precision)
```

- [ ] **Step 7: Lancer la suite complète**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add stock-analysis-ui/src/core/optim_budget.py stock-analysis-ui/src/optimisateur_hybride.py \
        stock-analysis-ui/src/tests/test_optim_budget.py
git commit -m "fix(optim): respecter le budget d'evaluations

popsize de SciPy est un multiplicateur : 200 avec 36 dimensions donnait 7 200
individus par generation, environ 1,45 million d'evaluations par groupe au lieu
de 75 000. Le budget est aussi reparti entre strategies en mode hybrid, et le
menu affiche desormais le budget reellement utilise."
```

---

### Task 8: Coût de transaction et graine

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:496-520`, `:1132-1144`, `:2074-2087`
- Modify: `stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py:330` (docstring)
- Test: `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (ajout)

**Interfaces:**
- Consumes: `save_optimization_results` (Task 6).
- Produces:
  - `COUT_TRANSACTION_PAR_TRADE: float = 1.0`, constante de module
  - `HybridOptimizer.seed: int | None`
  - `optimize_sector_coefficients_hybrid(..., seed: int | None = None)`

- [ ] **Step 1: Écrire le test, qui échoue**

Ajouter à `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` :

```python
def test_cout_de_transaction_unique_et_absolu() -> None:
    """Verrouille S5 : une seule valeur par defaut, exprimee en montant.

    Le moteur fait profit = (close - entry)/entry * montant - transaction_cost,
    donc c'est un montant absolu par trade et non un pourcentage, contrairement
    a ce qu'annoncait le docstring. Les deux defauts contradictoires etaient
    1.0 dans la fonction et 0.02 passe par le CLI.
    """
    import inspect

    assert oh.COUT_TRANSACTION_PAR_TRADE == 1.0

    signature = inspect.signature(oh.optimize_sector_coefficients_hybrid)
    assert signature.parameters['transaction_cost'].default == oh.COUT_TRANSACTION_PAR_TRADE

    signature_optim = inspect.signature(oh.HybridOptimizer.__init__)
    assert signature_optim.parameters['transaction_cost'].default == oh.COUT_TRANSACTION_PAR_TRADE


def test_graine_rend_le_tirage_reproductible(monkeypatch) -> None:
    """Verrouille M2 : deux optimiseurs a graine egale tirent la meme
    population initiale. La graine etait auparavant np.random.randint()."""
    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                      domain_coeffs=None, domain_thresholds=None, **kwargs):
        return {'gain_total': 1.0, 'trades': 1, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events', faux_backtest)

    premier = oh.HybridOptimizer(_donnees(1), 'X', seed=4321)
    second = oh.HybridOptimizer(_donnees(1), 'X', seed=4321)

    assert premier.seed == 4321
    a = premier.latin_hypercube_sampling(3)[0]
    b = second.latin_hypercube_sampling(3)[0]
    assert np.allclose(a, b)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: FAIL, `AttributeError: module 'optimisateur_hybride' has no attribute 'COUT_TRANSACTION_PAR_TRADE'`

- [ ] **Step 3: Corriger le docstring du moteur**

`COUT_TRANSACTION_PAR_TRADE` a été déclarée en tâche 6, la signature de
`save_optimization_results` s'en servant comme valeur par défaut. Il reste à
corriger la documentation du moteur, qui annonce l'inverse de ce qu'il calcule.

Dans `qsi_optimized.py` ligne 330, remplacer :

```python
        transaction_cost: Coût de transaction en %
```

par :

```python
        transaction_cost: Coût par trade, en MONTANT absolu et non en pourcentage
            (profit = (close - entry) / entry * montant - transaction_cost)
```

- [ ] **Step 4: Unifier les défauts et ajouter la graine**

Dans `HybridOptimizer.__init__`, remplacer la signature par :

```python
    def __init__(self, stock_data, domain, montant=50,
                 transaction_cost=COUT_TRANSACTION_PAR_TRADE, precision=2,
                 use_price_features: bool = False,
                 use_fundamentals_features: bool = False,
                 optimization_mode: str = 'gain_moyen',
                 compute_backend: Optional[ComputeBackend] = None,
                 seed: Optional[int] = None):
```

et ajouter dans le corps, avant la construction des bornes :

```python
        # Graine explicite : elle amorce numpy ET random, utilises par le GA,
        # le PSO et clean_sector_cap_groups. Un run peut ainsi etre rejoue.
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
```

Dans `latin_hypercube_sampling`, semer explicitement l'échantillonneur. C'est
indispensable : `scipy.stats.qmc.LatinHypercube` possède son propre générateur et
**n'obéit pas** à `np.random.seed`, donc sans ce paramètre le LHS resterait
irreproductible même avec une graine fixée.

```python
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=self.seed)
```

Dans `optimize_sector_coefficients_hybrid`, remplacer la signature `transaction_cost=1.0` par `transaction_cost=COUT_TRANSACTION_PAR_TRADE` et ajouter `seed: Optional[int] = None`, puis passer `seed=seed` à la construction de `HybridOptimizer`.

Dans le `__main__`, ligne 2079, remplacer `transaction_cost=0.02` par `transaction_cost=COUT_TRANSACTION_PAR_TRADE`, et ajouter `seed=graine` avec, avant la boucle :

```python
    graine = int(datetime.now().timestamp()) % 100000
    print(f"   Graine du run : {graine} (rejouable, stockée en base)")
```

- [ ] **Step 5: Lancer les tests**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py -q`
Expected: PASS, 6 tests

- [ ] **Step 6: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py \
        stock-analysis-ui/src/trading_c_acceleration/qsi_optimized.py \
        stock-analysis-ui/src/tests/test_optim_sauvegarde.py
git commit -m "fix(optim): cout de transaction unique et runs rejouables

Le cout est un montant absolu par trade, pas un pourcentage : le docstring
disait l'inverse et deux defauts contradictoires coexistaient, 1.0 dans la
fonction et 0.02 en CLI, soit un facteur 50. La graine devient explicite et
part en base."
```

---

### Task 9: Hygiène du module

**Files:**
- Modify: `stock-analysis-ui/src/optimisateur_hybride.py:1-30`, `:157-176`, `:211-221`, `:558`, `:727-737`, `:1000-1034`
- Test: `stock-analysis-ui/src/tests/test_optim_params.py` (ajout)

**Interfaces:**
- Consumes: `config.CACHE_DIR`.
- Produces: aucune interface nouvelle.

- [ ] **Step 1: Écrire le test d'absence d'effet de bord**

Ajouter à `stock-analysis-ui/src/tests/test_optim_params.py` :

```python
def test_importer_l_optimisateur_ne_cree_aucun_dossier(tmp_path, monkeypatch) -> None:
    """Verrouille : plus de mkdir a l'import.

    `Path("cache_data/sector_cache.json")` etait relatif et son dossier etait
    cree au chargement du module. Trois dossiers cache_data/ existaient donc
    dans le depot, dont un cree par une simple execution de pytest.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    racine = Path(__file__).resolve().parent.parent
    # On herite de l'environnement : un env minimal priverait matplotlib de HOME
    # et l'import echouerait pour une raison etrangere au test.
    environnement = dict(os.environ)
    environnement.update({
        "PYTHONPATH": str(racine),
        "MPLCONFIGDIR": str(tmp_path / ".mpl"),
        "QSI_DISABLE_C_ACCELERATION": "1",
        "QSI_CONSENSUS_OFFLINE": "1",
        "QSI_DISABLE_PROFILE_FETCH": "1",
    })
    resultat = subprocess.run(
        [sys.executable, "-c", "import optimisateur_hybride"],
        cwd=str(tmp_path), env=environnement,
        capture_output=True, text=True, timeout=300)

    assert resultat.returncode == 0, resultat.stderr[-2000:]
    assert not (tmp_path / "cache_data").exists()
    assert not (tmp_path / "cache_logs").exists()
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py::test_importer_l_optimisateur_ne_cree_aucun_dossier -q`
Expected: FAIL, le dossier `cache_data` est créé dans `tmp_path`.

- [ ] **Step 3: Supprimer l'effet de bord et l'ignorance globale des avertissements**

Remplacer les lignes 156 à 160 :

```python
SECTOR_CACHE_FILE = Path("cache_data/sector_cache.json")
SECTOR_CACHE_FILE.parent.mkdir(exist_ok=True)
```

par :

```python
# Chemin absolu venant de config, et creation du dossier a l'ECRITURE. Le
# chemin relatif et le mkdir a l'import fabriquaient un dossier cache_data/ la
# ou se trouvait le repertoire courant.
from config import CACHE_DIR

SECTOR_CACHE_FILE = Path(CACHE_DIR) / "sector_cache.json"
```

Dans `_save_sector_cache`, ajouter en première ligne du corps :

```python
    SECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
```

Supprimer la ligne 24 :

```python
warnings.filterwarnings("ignore")
```

et l'import `warnings` s'il devient inutilisé.

- [ ] **Step 4: Supprimer le code mort et les doublons**

- Supprimer la fonction `get_best_gain_csv`, lignes 211 à 221, jamais appelée.
- Supprimer les deux `from pathlib import Path` en trop, lignes 11 et 17, en gardant celui de la ligne 10.
- Rendre le `sys.path.insert` idempotent, lignes 12 à 14 :

```python
_trading_accel_path = Path(__file__).parent / "trading_c_acceleration"
if _trading_accel_path.exists():
    _racine_src = str(_trading_accel_path.parent)
    if _racine_src not in sys.path:
        sys.path.insert(0, _racine_src)
```

- Corriger le commentaire de la ligne 558, `Extra 13 params`, qui en décrit onze :

```python
            # 11 parametres fondamentaux : 1 drapeau, 5 poids, 5 seuils
```

- [ ] **Step 5: Hisser le pool de threads et fermer les connexions**

Dans `HybridOptimizer.__init__`, ajouter :

```python
        # Un seul pool pour toute la duree de vie de l'optimiseur. Il etait
        # cree et detruit a CHAQUE appel d'objectif, soit des dizaines de
        # milliers de fois par groupe.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, min(MAX_WORKERS, len(self.stock_data))))
```

Ajouter la méthode de fermeture :

```python
    def fermer(self) -> None:
        """Libere le pool de threads. Idempotent."""
        pool = getattr(self, '_pool', None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._pool = None
```

Dans `evaluate_config`, remplacer le bloc `with ThreadPoolExecutor(...) as executor:` par un usage de `self._pool` :

```python
            futures = {self._pool.submit(evaluate_symbol, symbol): symbol
                       for symbol in self.stock_data.keys()}
            for future in as_completed(futures):
                gain, trades, success = future.result()
                total_gain += gain
                total_trades += trades
                total_success += success
```

Dans `optimize_sector_coefficients_hybrid`, appeler `optimizer.fermer()` dans un `finally` couvrant les stratégies.

Dans `replay_all_historical`, entourer la connexion SQLite d'un `try/finally` fermant `conn`.

- [ ] **Step 6: Lancer la suite complète**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add stock-analysis-ui/src/optimisateur_hybride.py stock-analysis-ui/src/tests/test_optim_params.py
git commit -m "chore(optim): hygiene du module

Supprime le filterwarnings global qui rendait toute l'application muette, le
mkdir a l'import qui fabriquait des dossiers cache_data/ parasites, la fonction
morte get_best_gain_csv et les imports dupliques. Hisse le pool de threads hors
de l'objectif et ferme les connexions SQLite en finally."
```

---

### Task 10: Run de fumée, assertions périmées, journal

**Files:**
- Modify: `stock-analysis-ui/src/tests/test_fundamentals_integration.py:176-191`
- Modify: `stock-analysis-ui/CHANGELOG.md`
- Test: `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` (ajout)

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces: aucune interface nouvelle.

- [ ] **Step 1: Écrire le run de fumée, qui échoue**

Ajouter à `stock-analysis-ui/src/tests/test_optim_sauvegarde.py` :

```python
def test_run_de_fumee_ecrit_une_ligne_coherente(monkeypatch, tmp_path) -> None:
    """Critere d'acceptation 2 : un groupe, deux symboles, budget 40, LHS.

    Verifie qu'une ligne est ecrite et que ses metriques correspondent a un
    recalcul sur ses propres coefficients, ce qui est exactement ce que B2
    cassait.
    """
    import sqlite3

    import config

    chemin = _base_vide(tmp_path)
    monkeypatch.setattr(config, 'OPTIMIZATION_DB_PATH', chemin)

    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                      domain_coeffs=None, domain_thresholds=None, **kwargs):
        premier = list(domain_coeffs.values())[0][0]
        return {'gain_total': 20.0 + premier, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events', faux_backtest)
    monkeypatch.setattr(oh, 'download_stock_data', lambda symbols, period=None: _donnees(2))
    monkeypatch.setattr(oh, 'extract_best_parameters', lambda db_path=None: {})

    coeffs, score, success_rate, seuils, resume = oh.optimize_sector_coefficients_hybrid(
        ['SYM0', 'SYM1'], 'Technology_Large', strategy='lhs',
        budget_evaluations=40, precision=2, cap_range='Large', seed=99)

    conn = sqlite3.connect(chemin)
    conn.row_factory = sqlite3.Row
    lignes = conn.execute("SELECT * FROM optimization_runs").fetchall()
    conn.close()

    assert len(lignes) == 1
    ligne = lignes[0]
    assert ligne['trades'] == 6            # 2 symboles x 3 trades
    assert ligne['success_rate'] == pytest.approx(2 / 3 * 100)
    assert ligne['seed'] == 99
    assert len(coeffs) == 8
    assert len(seuils) == 10               # 8 seuils features + achat + vente
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue ou passe**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_sauvegarde.py::test_run_de_fumee_ecrit_une_ligne_coherente -q`
Expected: si les tâches 1 à 9 sont faites, PASS. En cas d'échec, corriger le chaînage réel, ce test étant le seul à parcourir la fonction de bout en bout.

- [ ] **Step 3: Corriger les assertions périmées**

Dans `stock-analysis-ui/src/tests/test_fundamentals_integration.py`, remplacer les quatre assertions des lignes 176 à 191 :

```python
        assert len(opt_base.bounds) == 18, f"Expected 18 bounds, got {len(opt_base.bounds)}"
        ...
        assert len(opt_price.bounds) == 24, f"Expected 24 bounds, got {len(opt_price.bounds)}"
        ...
        assert len(opt_fund.bounds) == 28, f"Expected 28 bounds, got {len(opt_fund.bounds)}"
        ...
        assert len(opt_both.bounds) == 34, f"Expected 34 bounds, got {len(opt_both.bounds)}"
```

par les valeurs réelles, dérivées du contrat pour qu'elles ne puissent plus se périmer :

```python
        from core import optim_params as contrat

        assert len(opt_base.bounds) == len(contrat.bornes())
        ...
        assert len(opt_price.bounds) == len(contrat.bornes(prix=True))
        ...
        assert len(opt_fund.bounds) == len(contrat.bornes(fond=True))
        ...
        assert len(opt_both.bounds) == len(contrat.bornes(prix=True, fond=True))
```

- [ ] **Step 4: Vérifier les cinq critères d'acceptation**

```bash
# 1. Le CLI atteint le menu sans variable d'environnement
cd stock-analysis-ui/src && timeout 120 ../../.venv_new/bin/python -c "import optimisateur_hybride; print('import OK')"

# 2 et 3. Suite complete verte, run de fumee inclus
cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q

# 4. Evaluations prevues egales aux evaluations observees
cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_budget.py -q

# 5. Aucun dossier parasite
cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_params.py::test_importer_l_optimisateur_ne_cree_aucun_dossier -q
find /home/berkam/Projets/Gestion_trade -maxdepth 3 -type d -name cache_data
```

Expected: `import OK` ; suite verte ; seul `stock-analysis-ui/src/cache_data` subsiste, les deux autres dossiers étant à supprimer à la main une fois le test vert.

- [ ] **Step 5: Journaliser dans le CHANGELOG**

Ajouter en tête de `stock-analysis-ui/CHANGELOG.md`, avant la section `## Version 1.6.0` :

```markdown
## Version 1.7.0 - Optimisateur hybride : lot 1, démarrage et cohérence (2026-08-05)

### 🐛 Corrections

- **Le CLI de l'optimisateur mourait à l'import**
  - `trading_c.cpython-310-x86_64-linux-gnu.so` avait été compilé le 2026-04-18 avec AddressSanitizer (`QSI_DEBUG_C_MODE=1 QSI_USE_ASAN=1`). Son chargement ne lève pas, il **avorte le processus**, donc le `try/except` de `_diagnose_import` était impuissant et `python optimisateur_hybride.py` mourait sans message. Un garde-fou inspecte désormais le binaire avant tout `dlopen` et se replie sur le chemin Python, et le module a été recompilé avec les drapeaux de production.

- **Les 4 seuils optimisés n'avaient aucun effet mais partaient en production**
  - `evaluate_config` calculait les seuils RSI, Volume, ADX et Score puis appelait `backtest_signals_c_extended`, qui n'a aucun paramètre de seuils : `py_backtest_symbol` ne prend que `(prices, volumes, coeffs, montant, cost)`. Quatre des quatorze dimensions étaient donc du bruit, et ces valeurs jamais évaluées étaient sauvegardées en `th1`, `th4`, `th5`, `th8` puis appliquées aux signaux réels. L'objectif passe par `backtest_signals_with_events` avec `domain_thresholds`, pour un surcoût mesuré de 1 %.

- **Les `trades` et le `success_rate` sauvegardés venaient d'une autre configuration**
  - La sauvegarde lisait `optimizer.meilleur_trades`, qui suit la meilleure configuration jamais vue, pas le vecteur écrit sur la même ligne. Sous `workers=-1` ces compteurs restaient dans les sous-processus, si bien que `strategy='differential'` sur un groupe sans historique n'écrivait **jamais** rien. Les métriques voyagent désormais avec leur vecteur.

- **Trois plages concurrentes par paramètre, quatre divergences**
  - `core/optim_params.py` devient la seule description du vecteur. Exemple : `a_price_slope` était cherché sur (-1.5, 3.0), bridé à (-0.5, 3.0) à l'évaluation et à (0.0, 3.0) à la sauvegarde. `th_score`, non relevé par l'audit, était cherché sur (2.0, 6.0) et bridé à (1.0, 6.0).

- **Le budget d'évaluations était ignoré d'un facteur 19**
  - `popsize` de SciPy est un multiplicateur : la population vaut `popsize * dimension`. Passer 200 avec 36 dimensions donnait 7 200 individus par génération, soit environ 1,45 million d'évaluations par groupe au lieu des 75 000 visés. Le budget est désormais calculé, réparti entre stratégies en mode `hybrid`, et le menu affiche la valeur réellement utilisée, contre 3 500 annoncés pour 30 000 utilisés.

- **Le croisement génétique produisait des individus hors bornes**
  - BLX-α étend l'intervalle parental sans borner ; seule la mutation bornait, et avec 10 % de probabilité par gène. Le meilleur individu retourné pouvait donc sortir du domaine et être sauvegardé tel quel.

- **Coût de transaction incohérent d'un facteur 50**
  - C'est un montant absolu par trade et non un pourcentage, contrairement au docstring. Les deux défauts contradictoires, `1.0` dans la fonction et `0.02` passé par le CLI, sont remplacés par une constante unique à `1.0`, et une colonne `transaction_cost` dit désormais dans quel monde chaque ligne a été mesurée.

### ✨ Nouveautés

- **Runs rejouables** : la graine devient explicite, amorce `numpy` et `random`, et est stockée dans une colonne `seed`.
- **Contrat de paramètres testable** : `core/optim_params.py` et `core/optim_budget.py`, verrouillés par 45 tests hors réseau. Le module n'en avait aucun.

### ♻️ Interne

- Suppression du `warnings.filterwarnings("ignore")` de niveau module, qui rendait toute l'application muette dès l'ouverture de la fenêtre d'optimisation.
- Le cache secteur prend `config.CACHE_DIR`, absolu, et crée son dossier à l'écriture. Trois dossiers `cache_data/` parasites existaient, dont un créé par une simple exécution de pytest.
- Le pool de threads est créé une fois par optimiseur au lieu d'être reconstruit à chaque appel d'objectif, et `workers=-1` passe à `workers=1`, ce qui supprime la sur-souscription et la sérialisation des séries à chaque génération.
- Suppression de `get_best_gain_csv`, morte, et des imports dupliqués.

### ⚠️ Connu, non traité dans ce lot

- Un run complet reste hors de portée : l'objectif coûte 7,6 s par backtest, `get_trading_signal` étant appelé une fois par barre et recalculant tous les indicateurs, avec une requête SQLite par barre. C'est l'objet du lot 2.
- `get_sector` et `classify_cap_range` consomment toujours une requête yfinance par symbole. Lot 3.
```

- [ ] **Step 6: Commit**

```bash
git add stock-analysis-ui/src/tests/test_optim_sauvegarde.py \
        stock-analysis-ui/src/tests/test_fundamentals_integration.py \
        stock-analysis-ui/CHANGELOG.md
git commit -m "test(optim): run de fumee de bout en bout et journal du lot 1

Remplace aussi les quatre assertions de dimensions perimees, qui attendaient
18, 24, 28 et 34 bornes quand le code en produisait 14, 25, 25 et 36, et qui ne
tournaient jamais faute d'etre dans le sous-ensemble par defaut."
```

---

## Traçabilité vers la spec

| Réf spec | Défaut | Tâche |
|---|---|---|
| N1 | `.so` ASan, abort à l'import | 2, étapes 3 à 7 |
| N2 | Objectif en O(n²) à 7,6 s | hors lot, voir lot 2 |
| B1 | Les 4 seuils optimisés sans effet | 4 |
| B2 | Métriques d'un autre vecteur | 5 |
| B3 | Trois plages par paramètre | 1, puis 3 étapes 4 et 5 |
| B4 | `popsize` et budget ignoré | 7 |
| S1 | Croisement GA hors bornes | 3, étape 6 |
| S2 | Parallélisme imbriqué | 7 étape 5 (`workers=1`) et 9 étape 5 (pool hissé) |
| S3 | yfinance par symbole | hors lot, voir lot 3 |
| S4 | `mkdir` à l'import, chemin relatif | 9, étape 3 (moitié cheap) ; reste au lot 3 |
| S5 | Coût de transaction incohérent | 6 (constante) et 8 (appelants, docstring) |
| M1 | `filterwarnings` global | 9, étape 3 |
| M2 | Runs non reproductibles | 8, étape 4 |
| M3 | Budget affiché ≠ utilisé | 7, étape 6 |
| M4 | Assertions de test périmées | 10, étape 3 |
| M5 | Dette diverse | 9, étapes 4 et 5 |

## Notes pour l'implémenteur

**Ce que ce lot ne corrige volontairement pas.** `get_trading_signal` appelle `extract_best_parameters()` à chaque barre, soit 1 210 requêtes SQLite par backtest, pour 2,2 s des 7,6 s. Mémoriser ce résultat serait un gain immédiat de 29 %, mais cela change le comportement de la production en cours de run et appartient au lot 2, qui traite le coût de l'objectif dans son ensemble.

**Ordre des tâches.** Les tâches 1 et 2 sont indépendantes et peuvent être faites dans n'importe quel ordre. Les tâches 3 à 6 se suivent strictement. Les tâches 7, 8 et 9 dépendent de 3 mais sont indépendantes entre elles. La tâche 10 vient en dernier.

**Piège de nommage.** Le paramètre des extras prix s'appelle `extra_params` dans `backtest_signals_with_events` et `price_extras` dans `backtest_signals_c_extended`. Le premier est celui à utiliser désormais.

**Piège de signature.** `backtest_signals_with_events` renvoie un tuple `(dict, evenements)`, alors que `backtest_signals_c_extended` renvoie un dict. Dépaqueter.
