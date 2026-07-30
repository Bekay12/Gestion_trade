"""
Verrouillage du biais de look-ahead dans compute_pit_fundamentals.
"""

from typing import List, Optional, Dict

import pytest

# conftest.py met deja src/ sur sys.path et pose les garde-fous hors ligne.
from fundamentals_cache import compute_pit_fundamentals


def _quarter(date: str,
             revenue: Optional[float] = None,
             diluted_eps: Optional[float] = None,
             gross_profit: Optional[float] = None,
             net_income: Optional[float] = None,
             stockholders_equity: Optional[float] = None,
             total_debt: Optional[float] = None,
             total_assets: Optional[float] = None,
             free_cash_flow: Optional[float] = None,
             operating_cash_flow: Optional[float] = None) -> dict:
    """
    Fabrication de dictionnaire trimestre pour les tests.
    """
    return {
        'quarter_date': date,
        'total_revenue': revenue,
        'diluted_eps': diluted_eps,
        'gross_profit': gross_profit,
        'net_income': net_income,
        'stockholders_equity': stockholders_equity,
        'total_debt': total_debt,
        'total_assets': total_assets,
        'free_cash_flow': free_cash_flow,
        'operating_cash_flow': operating_cash_flow
    }


def test_returns_none_when_no_quarter_published_yet() -> None:
    """Verrouille que None est retourné si aucun trimestre n'est publié à la date cible."""
    quarters: List[dict] = [_quarter('2026-01-01', revenue=999999)]
    as_of_date: str = '2018-01-01'
    result: Optional[Dict[str, Optional[float]]] = compute_pit_fundamentals(
        quarters, as_of_date, publication_delay_days=45
    )
    # Verrou principal du correctif : une branche de repli « last resort » a existé
    # dans compute_pit_fundamentals et retournait le trimestre le plus récent du
    # cache quand aucun n'était publié à as_of_date — soit, ici, les fondamentaux
    # de 2026 pour une barre de 2018. Réintroduire ce repli fait échouer ce test.
    assert result is None


def test_ignores_quarters_published_after_as_of_date() -> None:
    """Verrouille qu'un trimestre postérieur à as_of_date n'entre pas dans le calcul."""
    # Cinq trimestres publiés à 2017-06-01, plus un trimestre de 2026 qui ne l'est
    # jamais. Un simple `is not None` ne prouverait rien ici : il faut une valeur
    # attendue que la fuite du trimestre futur rendrait fausse.
    quarters: List[dict] = [
        _quarter('2016-01-01', revenue=100),
        _quarter('2016-04-01', revenue=200),
        _quarter('2016-07-01', revenue=300),
        _quarter('2016-10-01', revenue=400),
        _quarter('2017-01-01', revenue=500),
        _quarter('2026-01-01', revenue=999999),
    ]
    as_of_date: str = '2017-06-01'
    result: Optional[Dict[str, Optional[float]]] = compute_pit_fundamentals(
        quarters, as_of_date, publication_delay_days=45, lookback_quarters=4
    )

    assert result is not None
    # Sur les cinq trimestres publiés : (500 - 100) / 100 * 100 = 400 %.
    # Si le trimestre de 2026 fuitait, la croissance depasserait 499 000 %.
    assert result['rev_growth'] == pytest.approx(400.0, abs=0.01)


def test_respects_publication_delay_boundary() -> None:
    """Verrouille que la borne du délai de publication est inclusive à +45j."""
    quarter: dict = _quarter('2020-01-01', revenue=1000)
    publication_delay_days: int = 45

    # 2020-01-01 + 45j = 2020-02-15
    # 2020-02-14 (44j après) : pas publié
    # 2020-02-15 (45j après) : publié
    as_of_date_44_days: str = '2020-02-14'
    as_of_date_45_days: str = '2020-02-15'

    result_44_days: Optional[Dict] = compute_pit_fundamentals(
        [quarter], as_of_date_44_days, publication_delay_days=publication_delay_days
    )
    result_45_days: Optional[Dict] = compute_pit_fundamentals(
        [quarter], as_of_date_45_days, publication_delay_days=publication_delay_days
    )

    # À 44 jours, la donnée n'est pas encore publiée
    assert result_44_days is None, "À 44j, la donnée ne doit pas être disponible"
    # À 45 jours (inclus), la donnée est publiée
    assert result_45_days is not None, "À 45j inclus, la donnée doit être disponible"


def test_annual_fallback_uses_90_day_delay() -> None:
    """Verrouille que le fallback annuel utilise un délai de 90j, pas publication_delay_days."""
    annual: dict = {
        'fiscal_date': '2020-01-01',
        'total_revenue': 1000
    }
    # as_of_date = 2020-03-30 est à 89 jours de 2020-01-01
    # Le fallback annuel nécessite 90 jours minimum
    as_of_date: str = '2020-03-30'

    result: Optional[Dict] = compute_pit_fundamentals(
        [], as_of_date, publication_delay_days=45, annuals_sorted=[annual]
    )

    # À 89 jours, le fallback annuel ne doit pas être publié (90j requis)
    assert result is None, "Le fallback annuel ne doit pas être publié avant 90 jours"


def test_yoy_growth_uses_fourth_previous_quarter() -> None:
    """Verrouille que la croissance YoY compare le 5e trimestre au 1er."""
    # 5 trimestres avec revenus croissants
    quarters: List[dict] = [
        _quarter('2020-01-01', revenue=100),
        _quarter('2020-04-01', revenue=200),
        _quarter('2020-07-01', revenue=300),
        _quarter('2020-10-01', revenue=400),
        _quarter('2021-01-01', revenue=500)
    ]
    # À 2021-06-01, tous les trimestres sont publiés (45j de délai)
    as_of_date: str = '2021-06-01'

    result: Optional[Dict] = compute_pit_fundamentals(
        quarters, as_of_date, publication_delay_days=45, lookback_quarters=4
    )

    # Croissance YoY = (500 - 100) / 100 * 100 = 400%
    assert result is not None, "Résultat ne doit pas être None"
    expected_rev_growth: float = 400.0
    assert result.get('rev_growth') is not None, "rev_growth doit être calculé"
    assert result['rev_growth'] == pytest.approx(expected_rev_growth, abs=0.01)


def test_returns_none_on_empty_input() -> None:
    """Verrouille que None est retourné sur entrées vides."""
    result: Optional[Dict] = compute_pit_fundamentals(
        [], '2020-01-01', annuals_sorted=None
    )
    assert result is None, "Résultat doit être None sur entrées vides"


def test_malformed_dates_are_skipped_not_crashing() -> None:
    """Verrouille que les dates malformées sont skippées sans crash."""
    quarters: List[dict] = [
        _quarter('2016-01-01', revenue=100),
        _quarter('pas-une-date', revenue=999999),
        _quarter('2016-04-01', revenue=200),
        _quarter('2016-07-01', revenue=300),
        _quarter('2016-10-01', revenue=400),
        _quarter('2017-01-01', revenue=500),
    ]
    as_of_date: str = '2017-06-01'

    # La date illisible ne doit ni lever d'exception ni entrer dans le calcul.
    result: Optional[Dict] = compute_pit_fundamentals(
        quarters, as_of_date, publication_delay_days=45, lookback_quarters=4
    )

    assert result is not None
    # Même attendu que sans la ligne corrompue : elle a bien été écartée.
    assert result['rev_growth'] == pytest.approx(400.0, abs=0.01)
