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
import os

import numpy as np
import pandas as pd
import pytest

import qsi
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
    """Cache neutralise (froid) contre cache chaud : resultat identique au bit pres.

    qsi.get_trading_signal lit et ecrit le nom TA_CACHE tel que lie dans le namespace
    de qsi.py (qsi.py:20 : `from core.cache import ... TA_CACHE`), un import qui capture
    l'objet au moment ou le module est charge. Remplacer core.cache.TA_CACHE par un objet
    different ne changerait donc rien au comportement reel : c'est qsi.TA_CACHE qu'il faut
    neutraliser pour que la version « froide » n'utilise vraiment aucun cache.

    Neutralisation : un _BoundedCache(maxsize=0) s'auto-vide a chaque ecriture (le
    `__setitem__` de _BoundedCache evince des que len(self) > maxsize, donc l'entree qui
    vient d'etre inseree est retiree immediatement), si bien qu'aucun `get()` ne peut jamais
    trouver quoi que ce soit : chaque barre est recalculee depuis zero, comme sans cache.
    """
    prix, volumes = _serie()

    original_ta_cache = qsi.TA_CACHE
    try:
        qsi.TA_CACHE = cache_module._BoundedCache(maxsize=0)
        froid = _backtest(prix, volumes, "TEST_IDENTITE")
    finally:
        qsi.TA_CACHE = original_ta_cache

    cache_module.TA_CACHE.clear()
    _backtest(prix, volumes, "TEST_IDENTITE")  # echauffement : remplit le cache
    chaud = _backtest(prix, volumes, "TEST_IDENTITE")

    assert froid == chaud


def _base_avec_une_ligne(chemin, secteur: str = "Technology", a1: float = 1.28):
    """
    Cree une base optimization_runs portant une ligne.

    ATTENTION, piege verifie le 2026-08-06 : la requete de
    _extract_best_parameters_sans_cache exige SANS REPLI les colonnes
    a9, a10, th9, th10, use_price_slope, use_price_acc, a11 a a15,
    th11 a th15 et use_fundamentals (qsi.py:126-138). Seules a16 a a18,
    th16 a th18 et use_price_extras recoivent un `NULL AS`. Une table plus
    courte ferait echouer la requete et rendre {} silencieusement, donnant un
    echec de test sans rapport avec le sujet.
    """
    import sqlite3

    conn = sqlite3.connect(chemin)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            sector TEXT NOT NULL,
            market_cap_range TEXT,
            gain_moy REAL, success_rate REAL, trades INTEGER,
            seuil_achat REAL, seuil_vente REAL,
            a1 REAL, a2 REAL, a3 REAL, a4 REAL, a5 REAL, a6 REAL, a7 REAL, a8 REAL,
            th1 REAL, th2 REAL, th3 REAL, th4 REAL, th5 REAL, th6 REAL, th7 REAL, th8 REAL,
            a9 REAL, a10 REAL, th9 REAL, th10 REAL,
            use_price_slope INTEGER DEFAULT 0, use_price_acc INTEGER DEFAULT 0,
            a11 REAL, a12 REAL, a13 REAL, a14 REAL, a15 REAL,
            th11 REAL, th12 REAL, th13 REAL, th14 REAL, th15 REAL,
            use_fundamentals INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "INSERT INTO optimization_runs ("
        " timestamp, sector, market_cap_range, gain_moy, trades,"
        " seuil_achat, seuil_vente,"
        " a1, a2, a3, a4, a5, a6, a7, a8,"
        " th1, th2, th3, th4, th5, th6, th7, th8,"
        " a9, a10, th9, th10, use_price_slope, use_price_acc,"
        " a11, a12, a13, a14, a15, th11, th12, th13, th14, th15, use_fundamentals"
        ") VALUES ("
        " '2026-01-01 00:00:00', ?, 'Large', 10.0, 5,"
        " 4.2, -2.0,"
        " ?, 1, 1, 1, 1, 1, 1, 1,"
        " 50, 0, 0, 1.5, 25, 0, 0.5, 4,"
        " 0, 0, 0, 0, 0, 0,"
        " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
        (secteur, a1))
    conn.commit()
    conn.close()


def test_extract_best_parameters_ne_lit_la_base_qu_une_fois(monkeypatch, tmp_path) -> None:
    """Verrouille : 1160 requetes SQLite par backtest devenaient une seule."""
    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin)

    appels = []
    vrai = qsi._extract_best_parameters_sans_cache

    def compte(db_path):
        appels.append(db_path)
        return vrai(db_path)

    monkeypatch.setattr(qsi, "_extract_best_parameters_sans_cache", compte)
    qsi._BEST_PARAMS_CACHE.clear()

    for _ in range(50):
        qsi.extract_best_parameters(chemin)

    assert len(appels) == 1, f"{len(appels)} lectures de base, 1 attendue"
    # Invariant central du lot : la reponse memoisee doit egaler une lecture
    # non memoisee, au bit pres (contrainte globale du plan).
    assert qsi.extract_best_parameters(chemin) == vrai(chemin)


def test_une_ecriture_en_base_invalide_le_cache(tmp_path) -> None:
    """La cle derivee de l'etat du fichier doit rendre l'invalidation automatique,
    y compris sans delai artificiel : c'est le cas le plus realiste et le plus
    difficile. Mesure du 2026-08-06 : inserer une deuxieme ligne dans une base
    SQLite neuve laisse st_size a 8192 dans les deux cas (la ligne tient dans la
    page deja allouee) ; seul st_mtime_ns bouge, de quelques millisecondes, et
    cela suffit sans qu'aucun sleep soit necessaire.
    """
    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin, a1=1.0)
    qsi._BEST_PARAMS_CACHE.clear()

    premier = qsi.extract_best_parameters(chemin)
    assert premier, "la base de test devrait produire au moins un secteur"

    _base_avec_une_ligne(chemin, secteur="Healthcare", a1=2.0)

    second = qsi.extract_best_parameters(chemin)

    assert set(second) != set(premier), "le cache n'a pas ete invalide"


def test_granularite_grossiere_ne_detecte_pas_lecriture(monkeypatch, tmp_path) -> None:
    """Limite connue, documentee plutot que masquee : si mtime_ns ET la taille sont
    identiques entre deux etats (systeme de fichiers a granularite grossiere, ou
    ecriture qui ne change ni l'un ni l'autre), aucune des deux composantes de la
    cle ne detecte l'ecriture, et le cache sert alors une reponse perimee jusqu'a
    la prochaine ecriture qui change reellement mtime ou taille.
    """
    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin, a1=1.0)
    qsi._BEST_PARAMS_CACHE.clear()

    etat_gele = os.stat(chemin)
    vrai_stat = os.stat

    def stat_gele(chemin_demande, *args, **kwargs):
        if str(chemin_demande) == chemin:
            return etat_gele
        return vrai_stat(chemin_demande, *args, **kwargs)

    monkeypatch.setattr(qsi.os, "stat", stat_gele)

    premier = qsi.extract_best_parameters(chemin)
    _base_avec_une_ligne(chemin, secteur="Healthcare", a1=2.0)
    second = qsi.extract_best_parameters(chemin)

    assert set(second) == set(premier), (
        "limite connue : mtime_ns et taille geles ne detectent pas l'ecriture"
    )


def test_un_echec_de_lecture_n_est_pas_memoise_et_reessaie(monkeypatch, tmp_path) -> None:
    """Un echec reel de lecture (verrou SQLite, corruption, permission) ne doit
    pas figer un {} en cache sous la cle de l'etat de fichier courant : l'appel
    suivant doit reessayer, pas servir la reponse perimee du premier echec.
    """
    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin)
    qsi._BEST_PARAMS_CACHE.clear()

    vrai = qsi._extract_best_parameters_sans_cache
    appels = []

    def echoue_puis_reussit(db_path):
        appels.append(db_path)
        if len(appels) == 1:
            raise qsi._LectureParametresEchouee("verrou simule")
        return vrai(db_path)

    monkeypatch.setattr(qsi, "_extract_best_parameters_sans_cache", echoue_puis_reussit)

    premier = qsi.extract_best_parameters(chemin)
    assert premier == {}, "un echec de lecture doit rendre un dict vide, pas lever"

    second = qsi.extract_best_parameters(chemin)
    assert second, "l'appel suivant doit reessayer au lieu de servir un {} memorise"
    assert len(appels) == 2, f"{len(appels)} tentatives, 2 attendues (echec puis succes)"


def test_une_base_absente_ne_leve_pas(tmp_path) -> None:
    """Un chemin invalide doit rendre un dict vide, pas une exception."""
    import qsi

    qsi._BEST_PARAMS_CACHE.clear()
    assert qsi.extract_best_parameters(str(tmp_path / "absente.db")) == {}
