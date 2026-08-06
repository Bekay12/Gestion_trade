"""
Verrouillage du cout de l'objectif d'optimisation.

TA_CACHE memoise les instantanes techniques par barre. Il etait dimensionne a
500 entrees alors qu'un backtest de 5 ans en produit environ 1160 pour un seul
symbole : il evincait donc les premieres barres avant de pouvoir les reutiliser
et son taux de reussite etait nul. Mesure du 2026-08-06, meme serie, resultats
identiques au bit pres :
    maxsize=500    eval1 7,23 s   eval2 7,18 s   eval3 7,13 s
    maxsize=5000   eval1 7,09 s   eval2 2,36 s   eval3 2,35 s

Aucun acces reseau, aucune base reelle.
"""
import numpy as np
import pandas as pd
import pytest

from core import cache as cache_module
from trading_c_acceleration.qsi_optimized import backtest_signals_with_events

# Une ligne reelle de optimization_runs (Technology, 149 trades), pour que le
# backtest produise de vrais signaux plutot qu'un score constamment nul.
COEFFS = (1.28, 3.0, 0.94, 2.19, 1.57, 0.5, 0.77, 3.0)
SEUILS = (49.9, 0.0, 0.0, 1.48, 24.8, 0.0, 0.5, 3.84)
ACHAT, VENTE = 2.0, -2.62
NB_BARRES = 1210
PREMIERE_BARRE = 50   # backtest_signals_with_events demarre a l'indice 50


def _serie(graine: int = 7) -> tuple[pd.Series, pd.Series]:
    """Serie de prix et de volumes deterministe, sans reseau."""
    rng = np.random.default_rng(graine)
    index = pd.date_range("2021-01-01", periods=NB_BARRES, freq="D")
    prix = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.018, NB_BARRES)), index=index)
    volumes = pd.Series(rng.lognormal(13.5, 0.6, NB_BARRES), index=index)
    return prix, volumes


def _backtest(prix, volumes, symbole: str) -> tuple[float, int]:
    resultat, _evenements = backtest_signals_with_events(
        prix, volumes, "default", 50, 1.0,
        domain_coeffs={"default": COEFFS},
        domain_thresholds={"default": SEUILS},
        seuil_achat=ACHAT, seuil_vente=VENTE,
        symbol_name=symbole)
    return resultat["gain_total"], resultat["trades"]


def test_ta_cache_dimensionne_pour_un_groupe_entier() -> None:
    """Le plafond doit couvrir plusieurs symboles, pas une fraction d'un seul.

    Un backtest produit environ 1160 instantanes pour UN symbole, et un groupe
    en compte des dizaines.
    """
    assert cache_module.TA_CACHE_MAXSIZE >= 50_000
    assert cache_module.TA_CACHE._maxsize == cache_module.TA_CACHE_MAXSIZE


def test_ta_cache_retient_un_backtest_complet() -> None:
    """Verrouille la regression : a 500, le cache saturait et n'aidait jamais."""
    cache_module.TA_CACHE.clear()
    prix, volumes = _serie()

    _backtest(prix, volumes, "TEST_RETENTION")

    attendu = NB_BARRES - PREMIERE_BARRE
    assert len(cache_module.TA_CACHE) >= attendu, (
        f"{len(cache_module.TA_CACHE)} instantanes retenus, {attendu} attendus")


def test_le_cache_ne_change_aucun_resultat() -> None:
    """Deuxieme evaluation servie par le cache : resultat identique au bit pres."""
    cache_module.TA_CACHE.clear()
    prix, volumes = _serie()

    froid = _backtest(prix, volumes, "TEST_IDENTITE")
    chaud = _backtest(prix, volumes, "TEST_IDENTITE")

    assert froid == chaud
