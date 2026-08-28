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


def test_depuis_colonnes_desactive_les_drapeaux_absents() -> None:
    """Un drapeau absent vaut desactive, jamais active.

    Le milieu des bornes (0.0, 1.0) vaut 0.5, que contraindre() arrondit a 1 :
    une ligne ecrite avant l'introduction de la colonne se verrait attribuer
    des features qu'elle n'avait pas. La convention du depot, celle de l'ancien
    save_optimization_results, est l'inverse.
    """
    retour = params.depuis_colonnes({'a1': 1.0}, prix=True, fond=True)
    idx = params.indices(prix=True, fond=True)
    assert retour[idx['use_price_extras']] == 0.0
    assert retour[idx['use_fundamentals']] == 0.0
    # Une colonne presente reste lue telle quelle.
    active = params.depuis_colonnes(
        {'use_price_extras': 1, 'use_fundamentals': 1}, prix=True, fond=True)
    assert active[idx['use_price_extras']] == 1.0
    assert active[idx['use_fundamentals']] == 1.0


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


def test_crossover_borne_les_enfants_hors_domaine() -> None:
    """Verrouille le bridage ajoute a la fin de `_crossover`.

    BLX-alpha etend l'intervalle parental au-dela des bornes ; sans l'appel a
    `params.contraindre` sur chaque enfant, le meilleur individu retourne par
    le GA pouvait sortir du domaine et etre sauvegarde tel quel. Parents places
    aux deux bords de chaque borne pour maximiser la probabilite de depassement.
    Retirer le bridage (`optimisateur_hybride.py`, fin de `_crossover`) fait
    echouer ce test : verifie manuellement en le retirant temporairement.
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
    optimiseur = HybridOptimizer(
        donnees, 'Technology_Large',
        use_price_features=True, use_fundamentals_features=True,
    )
    bornes = params.bornes(prix=True, fond=True)
    parent1 = np.array([bas for bas, _ in bornes])
    parent2 = np.array([haut for _, haut in bornes])

    np.random.seed(0)
    child1, child2 = optimiseur._crossover(parent1, parent2, alpha=0.3)

    idx = params.indices(prix=True, fond=True)
    indices_drapeaux = {idx['use_price_extras'], idx['use_fundamentals']}
    for enfant in (child1, child2):
        for i, (bas, haut) in enumerate(bornes):
            assert bas <= enfant[i] <= haut, (i, enfant[i])
            if i in indices_drapeaux:
                assert enfant[i] in (0.0, 1.0), (i, enfant[i])


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
