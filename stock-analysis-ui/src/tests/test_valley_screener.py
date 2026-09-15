"""Verrouille le contrat du screener de creux pour l'interface.

Les autres screeners rendent {title, headers, rows} avec le symbole en colonne 0,
et ScreenerResultsDialog en dépend. Ce test fige ce contrat pour le nouveau
moteur, sans aucun appel réseau : `analyser` est remplacé par un faux.
"""
import pandas as pd
import pytest

from core import valley_screener as vsc


def _faux_resultat():
    return pd.DataFrame([
        {'ticker': 'G1A.DE', 'signal': '🕳️ DIVERGENCE', 'note': 4.3, 'prix': 63.85,
         'baisse_%': -33.6, 'au_dessus_du_bas_%': 12.0, 'mouv_3m_%': -15.6,
         'indice_3m_%': -6.8, 'fraction_propre_3m_%': 67.6, 'ca_var_a1_%': 3.4,
         'indice': '^GDAXI', 'motif': 'recul partagé'},
        {'ticker': 'NESTE.HE', 'signal': '👀 À SUIVRE', 'note': 2.0, 'prix': 32.89,
         'baisse_%': -83.4, 'au_dessus_du_bas_%': 380.0, 'mouv_3m_%': -39.8,
         'indice_3m_%': 8.6, 'fraction_propre_3m_%': 128.6, 'ca_var_a1_%': -12.0,
         'indice': '^OMXH25', 'motif': 'baisse propre'},
        {'ticker': 'SAND.ST', 'signal': '—', 'note': 0.0, 'prix': 367.6,
         'baisse_%': -4.0, 'au_dessus_du_bas_%': 90.0, 'mouv_3m_%': 1.0,
         'indice_3m_%': 3.4, 'fraction_propre_3m_%': 50.0, 'ca_var_a1_%': 5.0,
         'indice': '^OMX', 'motif': 'pas de baisse'},
    ])


@pytest.fixture
def faux_analyser(monkeypatch):
    appels = {}

    def _faux(symboles, seuil_baisse, seuil_bas, quiet=False, asof=None,
              max_part_propre=75.0, progress=None):
        appels['symboles'] = symboles
        appels['seuil_baisse'] = seuil_baisse
        if progress:                      # l'interface doit pouvoir suivre l'avancement
            progress(1, len(symboles), symboles[0])
        return _faux_resultat()

    monkeypatch.setattr(vsc, 'analyser', _faux)
    return appels


def test_contrat_de_sortie(faux_analyser):
    res = vsc.run_valley(universe=['G1A.DE', 'NESTE.HE', 'SAND.ST'])
    assert set(res) >= {'title', 'headers', 'rows'}
    assert res['headers'][0].lower().startswith('sym'), 'le symbole doit être en colonne 0'
    assert all(len(r) == len(res['headers']) for r in res['rows']), \
        'chaque ligne doit avoir autant de cellules que d’en-têtes'


def test_les_titres_sans_signal_sont_exclus(faux_analyser):
    res = vsc.run_valley(universe=['G1A.DE', 'NESTE.HE', 'SAND.ST'])
    symboles = [r[0] for r in res['rows']]
    assert 'SAND.ST' not in symboles, "un titre sans baisse n'a rien à faire dans la liste"
    assert 'G1A.DE' in symboles


def test_filtrage_par_signal(faux_analyser):
    res = vsc.run_valley(universe=['G1A.DE', 'NESTE.HE'], signal='divergence')
    assert [r[0] for r in res['rows']] == ['G1A.DE']
    assert 'DIVERGENCE' in res['title'].upper()


def test_le_plus_fort_signal_arrive_en_premier(faux_analyser):
    res = vsc.run_valley(universe=['G1A.DE', 'NESTE.HE'])
    assert res['rows'][0][0] == 'G1A.DE', 'tri par note décroissante attendu'


def test_rappel_de_progression_transmis(faux_analyser):
    vus = []
    vsc.run_valley(universe=['G1A.DE'], progress=lambda i, n, s: vus.append((i, n, s)))
    assert vus, "le rappel de progression doit être transmis jusqu'au moteur"


def test_resultat_vide_ne_casse_pas(monkeypatch):
    monkeypatch.setattr(vsc, 'analyser',
                        lambda *a, **k: pd.DataFrame(columns=['ticker', 'signal', 'note']))
    res = vsc.run_valley(universe=['X'])
    assert res['rows'] == []


def test_la_vraie_signature_accepte_les_parametres_utilises():
    """Garde contre le faux commode.

    Les tests ci-dessus remplacent `analyser` : ils prouvent que le module
    transmet bien ses arguments, pas que le moteur les accepte. Sans cette
    vérification, un appel réel échouerait avec un TypeError que toute la suite
    aurait laissé passer.
    """
    import inspect

    from Valley_scan import analyser as vrai
    params = inspect.signature(vrai).parameters
    for attendu in ('symboles', 'seuil_baisse', 'seuil_bas', 'quiet',
                    'max_part_propre', 'progress'):
        assert attendu in params, f"le moteur n'accepte pas '{attendu}'"
