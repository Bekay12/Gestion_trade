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


def _fabrique_faux_backtest(logique):
    """
    --------------------------------------------------------------------------
    Objectif:
        Fabriquer un faux backtest a signature durcie, partage par les tests
        qui remplacent `backtest_signals_with_events`. La signature declare
        explicitement chaque parametre reellement passe par la production
        (aucun **kwargs final) : un mauvais nom cote production, par exemple
        `price_extras=` au lieu de `extra_params=`, leve alors un TypeError a
        l'appel plutot que d'etre absorbe silencieusement.

    Inputs:
        logique (Callable): recoit les memes arguments nommes que le vrai
            backtest (les inutilises peuvent etre ignores via **_) et renvoie
            (dict_resultat, evenements)

    Outputs:
        faux_backtest (Callable): a brancher via monkeypatch sur
            oh.backtest_signals_with_events
    --------------------------------------------------------------------------
    """
    def faux_backtest(prices, volumes, domaine, montant=50, transaction_cost=0.02,
                       domain_coeffs=None, domain_thresholds=None,
                       seuil_achat=None, seuil_vente=None,
                       extra_params=None, fundamentals_extras=None,
                       symbol_name=None):
        return logique(
            prices=prices, volumes=volumes, domaine=domaine,
            montant=montant, transaction_cost=transaction_cost,
            domain_coeffs=domain_coeffs, domain_thresholds=domain_thresholds,
            seuil_achat=seuil_achat, seuil_vente=seuil_vente,
            extra_params=extra_params, fundamentals_extras=fundamentals_extras,
            symbol_name=symbol_name,
        )
    return faux_backtest


def test_les_seuils_sont_transmis_au_backtest(monkeypatch) -> None:
    """Verrouille B1 : domain_thresholds recoit les 8 seuils du vecteur.

    Le faux backtest partage `_fabrique_faux_backtest` (signature durcie,
    aucun **kwargs final) : un mauvais nom cote production leverait un
    TypeError a l'appel plutot que d'etre absorbe silencieusement.
    """
    recus = []

    def logique(domain_thresholds=None, **_ignores):
        recus.append(domain_thresholds)
        return {'gain_total': 10.0, 'trades': 2, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))

    optimiseur = oh.HybridOptimizer(_donnees(1), 'Technology_Large')
    idx = params.indices()
    vecteur = np.zeros(14)
    vecteur[idx['th_rsi']] = 55.0
    vecteur[idx['th_vol']] = 1.4
    vecteur[idx['th_adx']] = 22.0
    vecteur[idx['th_score']] = 4.5
    vecteur[idx['seuil_achat']] = 4.2
    vecteur[idx['seuil_vente']] = -2.0

    score = optimiseur.evaluate_config(vecteur)

    assert recus, "le backtest n'a pas ete appele"
    seuils = list(recus[0].values())[0]
    assert seuils[0] == 55.0    # RSI
    assert seuils[3] == 1.4     # Volume
    assert seuils[4] == 22.0    # ADX
    assert seuils[7] == 4.5     # Score

    # evaluate_symbol encadre l'appel d'un `except Exception` large : un
    # TypeError (mauvais nom de parametre, depaquetage du tuple casse) y est
    # avale et renvoie (0.0, 0, 0), ce qui fait retomber evaluate_config sur
    # sa penalite degradee -1e6. Verifier seulement qu'un nombre est revenu
    # laisserait passer ce cas degrade : on verrouille la vraie valeur
    # calculee a partir du faux backtest (gain_total=10.0, trades=2, penalite
    # d'efficacite 0.02 par trade/symbole => 10.0 - 0.02*2 = 9.96).
    assert score == pytest.approx(9.96), (
        f"score degrade ({score}) : le faux backtest n'a pas ete appele "
        "avec les bons parametres, ou son retour n'a pas ete correctement "
        "depaquete"
    )


def test_les_metriques_suivent_leur_vecteur(monkeypatch) -> None:
    """Verrouille B2 : mesure_de() rend les metriques DU vecteur demande.

    Le faux backtest renvoie un gain proportionnel au premier coefficient, donc
    deux vecteurs ont des metriques distinctes et une confusion se voit. Le
    faux partage `_fabrique_faux_backtest` (signature durcie, aucun **kwargs
    final).
    """
    def logique(domain_coeffs=None, **_ignores):
        premier = list(domain_coeffs.values())[0][0]
        trades = 2 if premier > 0 else 8
        return {'gain_total': premier * 100.0, 'trades': trades, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))

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
    mesure_de() doit alors evaluer au lieu de rendre zero. Le faux partage
    `_fabrique_faux_backtest` (signature durcie, aucun **kwargs final).
    """
    appels = []

    def logique(**_ignores):
        appels.append(1)
        return {'gain_total': 5.0, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))

    optimiseur = oh.HybridOptimizer(_donnees(1), 'Technology_Large')
    mesure = optimiseur.mesure_de(np.zeros(14))   # jamais evalue avant

    assert appels, "mesure_de aurait du declencher une evaluation"
    assert mesure.trades == 3
    assert mesure.gagnants == 2


def test_mesure_de_ne_leve_pas_si_la_cle_diverge(monkeypatch) -> None:
    """Verrouille la divergence de cle mesuree en revue : a precision=1 avec
    prix et fondamentaux actifs, un vecteur hors bornes peut se re-arrondir
    sur une valeur differente lors de la deuxieme passe round+contraindre
    faite par evaluate_config(). La cle calculee par mesure_de() ne correspond
    alors plus a celle ecrite en cache, et `self.mesures[cle]` leverait un
    KeyError juste apres tout le budget d'evaluations depense.

    Vecteur reproducteur (verifie a la main) : th_price_slope=0.4, bornes
    (-0.25, 0.25). round(0.4, 1) = 0.4, contraindre() ram a 0.25 => cle de
    mesure_de(). Reinjecte dans evaluate_config(), round(0.25, 1) = 0.2 (deja
    dans les bornes) => cle differente ecrite en cache.
    """
    def logique(**_ignores):
        return {'gain_total': 5.0, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))

    optimiseur = oh.HybridOptimizer(
        _donnees(1), 'Technology_Large', precision=1,
        use_price_features=True, use_fundamentals_features=True)
    idx = params.indices(prix=True, fond=True)
    vecteur = np.zeros(len(params.specs(prix=True, fond=True)))
    vecteur[idx['th_price_slope']] = 0.4  # hors bornes (-0.25, 0.25)

    mesure = optimiseur.mesure_de(vecteur)  # ne doit pas lever KeyError

    assert isinstance(mesure, oh.Mesure)


def test_replay_restaure_les_drapeaux_de_prix_herites() -> None:
    """Verrouille la compatibilite des lignes historiques d'avant
    `use_price_extras`.

    La base reelle ne porte pas cette colonne : elle decrit les features de prix
    par les drapeaux `use_price_slope` et `use_price_acc` d'un ancien vecteur.
    Relue par le seul contrat, une telle ligne rejouerait ses 14 parametres de
    base et perdrait ses deux features, ce qui change son score et donc la
    decision de sauvegarde, silencieusement. Les extras introduits APRES elle
    valent 0.0, le poids neutre d'un coefficient, jamais le milieu des bornes.
    """
    ligne = {
        'a1': 1.0, 'th1': 55.0, 'th4': 1.2, 'th5': 25.0, 'th8': 4.0,
        'seuil_achat': 4.2, 'seuil_vente': -2.0,
        'use_price_slope': 1, 'use_price_acc': 0,
        'a9': 1.4, 'a10': 0.8, 'th9': 0.02, 'th10': -0.03,
    }
    idx = params.indices(prix=True)

    vecteur = oh._vecteur_depuis_ligne_historique(ligne, prix=True)

    assert vecteur[idx['use_price_extras']] == 1.0
    assert vecteur[idx['a_price_slope']] == pytest.approx(1.4)
    assert vecteur[idx['a_price_acc']] == pytest.approx(0.8)
    for cle in ('a_price_rsi_slope', 'a_price_vol_slope', 'a_price_var5j',
                'th_price_rsi_slope', 'th_price_vol_slope', 'th_price_var5j'):
        assert vecteur[idx[cle]] == 0.0, cle

    # Sans features de prix, le vecteur n'a aucun emplacement ou poser le
    # repli : la ligne se relit sur ses 14 parametres de base.
    assert len(oh._vecteur_depuis_ligne_historique(ligne, prix=False)) == 14


def test_replay_respecte_un_drapeau_de_prix_explicite() -> None:
    """Contre-epreuve : une ligne moderne passe inchangee.

    Elle porte une vraie colonne `use_price_extras`, meme a 0. Le repli ne doit
    pas la reactiver au pretexte d'un drapeau herite reste a 1.
    """
    ligne = {'use_price_extras': 0, 'use_price_slope': 1, 'a9': 1.4}
    idx = params.indices(prix=True)

    vecteur = oh._vecteur_depuis_ligne_historique(ligne, prix=True)

    assert vecteur[idx['use_price_extras']] == 0.0
    assert vecteur[idx['a_price_slope']] == pytest.approx(1.4)


def test_le_vecteur_historique_suit_les_index_du_contrat() -> None:
    """Verrouille la pose du vecteur historique sur `params.indices()`.

    Le bloc de relecture decrivait le vecteur une quatrieme fois, a la main :
    8 coefficients, 4 seuils, 2 globaux, 11 extras de prix puis 11 extras
    fondamentaux, dans un ordre litteral. Rien ne le testait, alors qu'un
    reordonnancement du contrat y aurait mis des coefficients sur des seuils
    sans lever la moindre erreur. Chaque categorie recoit ici une plage de
    valeurs distincte pour qu'une transposition soit visible.
    """
    coeffs = tuple(100.0 + numero for numero in range(1, 9))
    seuils_moteur = (55.0, 0.0, 0.0, 1.7, 28.0, 0.0, 0.5, 3.5)
    extras_prix = {spec.cle: 0.0 for spec in params.EXTRAS_PRIX}
    extras_prix.update({'use_price_extras': 1.0, 'a_price_slope': 201.0,
                        'th_price_var5j': 202.0})
    extras_fond = {spec.cle: 0.0 for spec in params.EXTRAS_FONDAMENTAUX}
    extras_fond.update({'use_fundamentals': 1.0, 'a_roe': 301.0,
                        'th_de_ratio': 302.0})

    vecteur = oh._vecteur_historique_depuis_champs(
        coeffs, seuils_moteur, 4.2, -2.0,
        extras_prix=extras_prix, extras_fondamentaux=extras_fond,
        prix=True, fond=True)

    idx = params.indices(prix=True, fond=True)
    assert len(vecteur) == len(idx)
    for numero in range(1, 9):
        assert vecteur[idx[f'a{numero}']] == pytest.approx(100.0 + numero)
    assert vecteur[idx['th_rsi']] == pytest.approx(55.0)
    assert vecteur[idx['th_vol']] == pytest.approx(1.7)
    assert vecteur[idx['th_adx']] == pytest.approx(28.0)
    assert vecteur[idx['th_score']] == pytest.approx(3.5)
    assert vecteur[idx['seuil_achat']] == pytest.approx(4.2)
    assert vecteur[idx['seuil_vente']] == pytest.approx(-2.0)
    assert vecteur[idx['use_price_extras']] == pytest.approx(1.0)
    assert vecteur[idx['a_price_slope']] == pytest.approx(201.0)
    assert vecteur[idx['th_price_var5j']] == pytest.approx(202.0)
    assert vecteur[idx['use_fundamentals']] == pytest.approx(1.0)
    assert vecteur[idx['a_roe']] == pytest.approx(301.0)
    assert vecteur[idx['th_de_ratio']] == pytest.approx(302.0)

    # Les 4 seuils geles n'occupent aucun emplacement du vecteur de recherche :
    # ils sont lus dans `seuils_moteur` mais ne peuvent pas decaler les globaux.
    for spec in params.SEUILS_GELES:
        assert spec.cle not in idx


def test_le_vecteur_historique_refuse_un_emplacement_sans_valeur() -> None:
    """Contre-epreuve : un emplacement non alimente leve, il ne vaut pas zero.

    Zero est le poids neutre d'un coefficient mais une valeur parfaitement
    active pour un drapeau ou un seuil : un trou silencieux fabriquerait une
    configuration qui n'a jamais tourne.
    """
    with pytest.raises(ValueError):
        oh._vecteur_historique_depuis_champs(
            tuple(1.0 for _ in range(8)),
            (55.0, 0.0, 0.0, 1.7, 28.0, 0.0, 0.5, 3.5),
            4.2, -2.0, extras_prix=None, extras_fondamentaux=None,
            prix=True, fond=False)


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
    population initiale. La graine etait auparavant np.random.randint().

    Le faux backtest partage `_fabrique_faux_backtest` (signature durcie,
    aucun **kwargs final) pour rester coherent avec les autres tests de ce
    fichier, contrairement au faux backtest a **kwargs propose par le brief.
    """
    def logique(**_ignores):
        return {'gain_total': 1.0, 'trades': 1, 'gagnants': 1}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))

    premier = oh.HybridOptimizer(_donnees(1), 'X', seed=4321)
    second = oh.HybridOptimizer(_donnees(1), 'X', seed=4321)

    assert premier.seed == 4321
    a = premier.latin_hypercube_sampling(3)[0]
    b = second.latin_hypercube_sampling(3)[0]
    assert np.allclose(a, b)


def test_run_de_fumee_ecrit_une_ligne_coherente(monkeypatch, tmp_path) -> None:
    """Critere d'acceptation 2 : un groupe, deux symboles, budget 40, LHS.

    Verifie qu'une ligne est ecrite et que ses metriques correspondent a un
    recalcul sur ses propres coefficients, ce qui est exactement ce que B2
    cassait. Traverse toute la chaine reelle : c'est le seul test du lot qui
    passe par `optimize_sector_coefficients_hybrid` de bout en bout. Le faux
    backtest partage `_fabrique_faux_backtest` (signature durcie, aucun
    **kwargs final), contrairement au faux backtest a **kwargs propose par le
    brief.
    """
    import sqlite3

    import config

    chemin = _base_vide(tmp_path)
    monkeypatch.setattr(config, 'OPTIMIZATION_DB_PATH', chemin)

    def logique(domain_coeffs=None, **_ignores):
        premier = list(domain_coeffs.values())[0][0]
        return {'gain_total': 20.0 + premier, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))
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


def test_le_baseline_historique_passe_par_le_meme_moteur(monkeypatch, tmp_path) -> None:
    """Verrouille B3 : l'historique et le nouveau score sortent du meme moteur.

    `hist_avg_gain`, `hist_total_trades` et `hist_success_rate` venaient d'une
    boucle `backtest_signals_c_extended`, un moteur sans aucun parametre de
    seuils : la ligne de fin de groupe opposait un backtest entierement
    parametre a un backtest de coefficients seuls. Le faux backtest ci-dessous
    ne repond QUE sur `domain_thresholds`, donc le resume ne peut porter la
    bonne valeur que si le baseline est passe par `mesure_de`.
    """
    import config

    monkeypatch.setattr(config, 'OPTIMIZATION_DB_PATH', _base_vide(tmp_path))

    def logique(domain_thresholds=None, **_ignores):
        th_rsi = list(domain_thresholds.values())[0][0]
        return {'gain_total': 10.0 + th_rsi, 'trades': 3, 'gagnants': 2}, []

    monkeypatch.setattr(oh, 'backtest_signals_with_events',
                         _fabrique_faux_backtest(logique))
    monkeypatch.setattr(oh, 'download_stock_data', lambda symbols, period=None: _donnees(2))
    monkeypatch.setattr(oh, 'extract_best_parameters', lambda db_path=None: {
        'Technology_Large': (
            tuple(1.0 for _ in range(8)),                       # coeffs
            (55.0, 0.0, 0.0, 1.7, 28.0, 0.0, 0.5, 3.5),        # seuils moteur
            (4.2, -2.0),                                        # globaux
            7.0,                                                # gain historique
            {'timestamp': '2026-01-01 00:00:00'},
        )
    })

    _coeffs, _score, _taux, _seuils, resume = oh.optimize_sector_coefficients_hybrid(
        ['SYM0', 'SYM1'], 'Technology_Large', strategy='lhs',
        budget_evaluations=20, precision=2, cap_range='Large', seed=7)

    # th_rsi = 55.0 traverse domain_thresholds, donc gain_total = 65.0 par
    # symbole ; 2 symboles pour une moyenne de 65.0. Le moteur de coefficients
    # seuls ne pouvait pas produire ce nombre.
    assert resume['gain_old'] == pytest.approx(65.0)
    assert resume['trades_old'] == 6                 # 2 symboles x 3 trades
    assert resume['success_old'] == pytest.approx(2 / 3 * 100)
