"""Indicateur avance point-in-time du backtest Valley (hors reseau)."""
import pandas as pd

import Valley_backtest as vb


def _q(dates, rev):
    return pd.DataFrame({"d": pd.to_datetime(dates), "total_revenue": rev,
                         "free_cash_flow": None, "total_debt": None})


def test_trimestre_non_publie_ignore():
    q = _q(["2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"], [100, 90, 95, 120])
    # Au 01.05.2025, le T1 2025 (publie le 15.05) n'est pas encore connu : 100 > 90 < 95, pas de reprise.
    f = vb.indicateur_pit(q, None, pd.Timestamp("2025-05-01"))
    assert f["indicateur_retourne"] is False
    # Au 20.05.2025 il l'est : 90 < 95 < 120, reprise sequentielle.
    assert vb.indicateur_pit(q, None, pd.Timestamp("2025-05-20"))["indicateur_retourne"] is True


def test_casse_sur_un_an_avec_cinq_trimestres():
    q = _q(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"], [100, 95, 92, 91, 85])
    f = vb.indicateur_pit(q, None, pd.Timestamp("2025-06-01"))
    assert f["indicateur_casse"] is True and round(f["ca_var_a1_%"]) == -15
