"""Briques du backtest point-in-time (Combined_backtest.py), hors reseau.

Ce qui doit tenir pour que le backtest ne mente pas : aucun compte publie
apres T, des dividendes retrouves dans les cours, un PER dans la bonne
devise, et un rendement qui refuse une echeance non atteinte.
"""
import numpy as np
import pandas as pd
import pytest

import Combined_backtest as bt


def _comptes(dates):
    return pd.DataFrame({"d": pd.to_datetime(dates), "total_revenue": range(1, len(dates) + 1)})


def test_aucun_exercice_publie_apres_T():
    a = _comptes(["2022-12-31", "2023-12-31", "2024-12-31"])
    # Exercice 2023 publie le 30.03.2024 (cloture + 90 j) : visible au 01.04.2024 ...
    assert list(bt.publies(a, pd.Timestamp("2024-04-01"), 90)["d"].dt.year) == [2022, 2023]
    # ... mais pas la veille de sa publication.
    assert list(bt.publies(a, pd.Timestamp("2024-03-29"), 90)["d"].dt.year) == [2022]


def test_dividende_retrouve_dans_le_cours_ajuste():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 100.0, 97.0, 97.0, 97.0], index=idx)
    # Detachement de 3 le 3e jour : les jours precedents sont ajustes par f = 1 - 3/100.
    adj = close * pd.Series([0.97, 0.97, 1.0, 1.0, 1.0], index=idx)
    y = bt.rendement_dividende(pd.DataFrame({"close": close, "adj_close": adj}))
    assert y == pytest.approx(3 / 97, rel=1e-6)


def test_pas_de_dividende_sans_ecart_ajuste_brut():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    close = pd.Series(np.linspace(100, 110, 5), index=idx)
    assert bt.rendement_dividende(pd.DataFrame({"close": close, "adj_close": close})) == 0.0


def test_facteur_devise():
    # Meme devise : cours / PER = BPA des comptes.
    assert bt.facteur_devise(pd.Series({"currency": "EUR", "trailing_pe": 10, "current_price": 50}), 5.0) == 1.0
    # Pence : cours / PER = 100 x BPA en livres.
    assert bt.facteur_devise(pd.Series({"currency": "GBp", "trailing_pe": 10, "current_price": 5000}), 5.0) == 100.0
    # ADR aux comptes en yuans : rapport ~7, PER non calculable.
    assert bt.facteur_devise(pd.Series({"currency": "USD", "trailing_pe": 10, "current_price": 50}), 35.0) is None


def test_rendement_refuse_une_echeance_non_atteinte():
    idx = pd.bdate_range("2025-01-01", "2025-09-30")
    px = pd.DataFrame({"adj_close": np.linspace(100, 120, len(idx))}, index=idx)
    T = pd.Timestamp("2025-01-01")
    assert bt.rendement(px, T, 6, pd.Timestamp("2025-09-30")) == pytest.approx(
        px.loc[:"2025-07-01", "adj_close"].iloc[-1] / 100 - 1)
    assert bt.rendement(px, T, 12, pd.Timestamp("2025-09-30")) is None


def test_benchmark_par_suffixe():
    assert bt.benchmark_for("ALV.DE") == "^GDAXI"
    assert bt.benchmark_for("AAPL") == "^GSPC"
    assert bt.benchmark_for("XYZ.ZZ") == "^GSPC"


def test_saut_de_regroupement_non_ajuste_est_signale():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    adj = pd.Series(0.01, index=idx)
    adj.loc["2025-01-29":] = 625.0          # PPCB : 0,01 -> 625 en une seance
    px = pd.DataFrame({"adj_close": adj})
    assert bt.saut_suspect(px, pd.Timestamp("2025-01-01"), 6) is True
    calme = pd.DataFrame({"adj_close": pd.Series(np.linspace(10, 14, len(idx)), index=idx)})
    assert bt.saut_suspect(calme, pd.Timestamp("2025-01-01"), 6) is False
