"""Verrouille deux defauts mesures des scanners autonomes (*_scan.py).

D1 — Conversion de devise. Les deux scanners ne convertissaient que l'USD :
     `mc_eur = mc / EUR_USD_RATE if currency == "USD" else mc`, ce qui traite
     la couronne suedoise, la danoise, la norvegienne, le franc suisse et le
     penny britannique comme s'ils etaient deja des euros. Sur 43 valeurs
     europeennes, 13 avaient une capitalisation fausse d'un facteur 7 a 11, et
     Tate & Lyle d'un facteur 85 (cotation en pence). Le critere « > 10 Mrd € »
     s'en trouvait fausse.

D2 — Momentum et derniere barre NaN. `g4_momentum` lisait `close.iloc[-1]`
     sans nettoyage. Sur les bourses europeennes la derniere ligne est la
     seance en cours et porte NaN, donc `cur`, `r3`, `r6` et `sma50` devenaient
     tous NaN et le critere tombait en silence. Mesure du 14.09.2026 : le
     critere etait calculable sur 2 valeurs sur 43, et aucune ne le remplissait ;
     apres nettoyage il est calculable sur 43 et 12 le remplissent.
"""
import numpy as np
import pandas as pd
import pytest

import Combined_scan as cs
import Sichere_Unternehmen_scan as ss


# ── D1 : conversion de devise ──────────────────────────────────────────────
@pytest.mark.parametrize('module', [ss, cs], ids=['sichere', 'combined'])
def test_capitalisation_convertie_depuis_toute_devise(module, monkeypatch):
    """Une capitalisation en SEK ne vaut pas le meme nombre d'euros."""
    monkeypatch.setattr(module, 'EUR_RATES',
                        {'EUR': 1.0, 'USD': 1.15, 'SEK': 11.28, 'GBP': 0.855, 'GBP_PENCE': 85.5})
    # 88,2 milliards de couronnes suedoises font 7,8 milliards d'euros, pas 88,2.
    assert module._mcap_en_mrd_eur(88.2e9, 'SEK') == pytest.approx(7.82, abs=0.02)
    assert module._mcap_en_mrd_eur(10.0e9, 'EUR') == pytest.approx(10.0, abs=0.01)
    assert module._mcap_en_mrd_eur(11.5e9, 'USD') == pytest.approx(10.0, abs=0.02)


@pytest.mark.parametrize('module', [ss, cs], ids=['sichere', 'combined'])
def test_cotation_en_pence_divisee_par_cent(module, monkeypatch):
    """GBp est un sous-multiple : l'ignorer gonfle la capitalisation de x100."""
    monkeypatch.setattr(module, 'EUR_RATES',
                        {'EUR': 1.0, 'USD': 1.15, 'GBP': 0.855, 'GBP_PENCE': 85.5})
    # 247,4 milliards de pence font 2,9 milliards d'euros.
    assert module._mcap_en_mrd_eur(247.4e9, 'GBp') == pytest.approx(2.89, abs=0.02)


@pytest.mark.parametrize('module', [ss, cs], ids=['sichere', 'combined'])
def test_devise_inconnue_ne_vaut_pas_un_critere_rempli(module, monkeypatch):
    """Sans taux, le critere n'est pas evaluable — et ne doit donc pas passer.

    Se rabattre sur 1.0 serait exactement le defaut corrige ici, en plus
    silencieux.
    """
    monkeypatch.setattr(module, 'EUR_RATES', {'EUR': 1.0})
    assert module._mcap_en_mrd_eur(50e9, 'JPY') is None
    ok, valeur, *_ = (module.c1_market_cap if module is ss else module.s1_market_cap)(
        {'marketCap': 50e9, 'currency': 'JPY'})
    assert ok is False


# ── D2 : momentum et derniere barre NaN ────────────────────────────────────
def _historique(n=200, dernier_nan=False):
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    # Serie croissante puis stable : +12 % sur 3 mois, au-dessus de la SMA50.
    close = pd.Series(np.linspace(100.0, 118.0, n), index=idx)
    if dernier_nan:
        close.iloc[-1] = np.nan
    return pd.DataFrame({'Close': close, 'Volume': pd.Series(1e6, index=idx)})


def test_momentum_calculable_malgre_la_seance_en_cours():
    """La derniere ligne NaN ne doit pas aneantir le critere."""
    sans = cs.g4_momentum(_historique())
    avec = cs.g4_momentum(_historique(dernier_nan=True))
    assert sans[1] is not None, 'temoin : le cas propre doit donner une valeur'
    assert avec[1] is not None, 'le cas avec NaN final doit aussi donner une valeur'
    assert avec[1] == pytest.approx(sans[1], abs=0.3)


def test_momentum_reste_indisponible_si_lhistorique_est_trop_court():
    """Le garde-fou de longueur doit survivre au nettoyage."""
    ok, valeur = cs.g4_momentum(_historique(n=100))
    assert ok is False and valeur is None


def test_big_growth_momentum_survit_a_la_seance_en_cours():
    """Même défaut que dans Combined_scan, même correctif.

    `c4_nascent_momentum` lisait `close.iloc[-1]` sans nettoyage. Sur les
    bourses européennes la dernière ligne est la séance en cours et porte NaN,
    ce qui annulait silencieusement le critère de momentum — un cinquième du
    score de croissance.
    """
    import Big_Growth_scan as bg

    sans = bg.c4_nascent_momentum(_historique())
    avec = bg.c4_nascent_momentum(_historique(dernier_nan=True))
    assert sans[1] is not None, 'témoin : le cas propre doit donner une valeur'
    assert avec[1] is not None, 'le cas avec NaN final doit aussi donner une valeur'
    assert avec[1] == pytest.approx(sans[1], abs=0.3)
