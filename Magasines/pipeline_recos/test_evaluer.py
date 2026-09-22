"""
Tests hors ligne de evaluer_recommandations (aucun réseau, aucun store).
Lancer : .venv_new/bin/python -m pytest Magasines/pipeline_recos/test_evaluer.py
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluer_recommandations import benchmark_for, evaluate_row, parse_price, ratios  # noqa: E402

D0 = dt.date(2026, 1, 5)


def _px(closes, highs=None, lows=None, pad=True):
    """Série synthétique ; pad=True la prolonge (dernier cours répété) au-delà de l'échéance."""
    highs, lows = highs or closes, lows or closes
    if pad:
        n = 150 - len(closes)
        closes, highs, lows = closes + [closes[-1]] * n, highs + [closes[-1]] * n, lows + [closes[-1]] * n
    idx = pd.bdate_range(D0, periods=len(closes))
    return pd.DataFrame({"close": closes, "high": highs, "low": lows}, index=idx)


def test_parse_price():
    assert parse_price("55,75 €") == (55.75, "€")
    assert parse_price("1.190,00 €") == (1190.0, "€")
    assert parse_price("15,05 $") == (15.05, "$")
    assert parse_price("190 Euro") == (190.0, "€")
    assert parse_price("95,87 %") == (None, None)
    assert parse_price("?") == (None, None)


def test_ratios_same_currency_only():
    tr, sr = ratios({"cours": "100,00 €", "objectif": "120,00 €", "stop": "90,00 €"})
    assert tr == pytest.approx(1.2) and sr == pytest.approx(0.9)
    assert ratios({"cours": "100,00 €", "objectif": "120,00 $", "stop": ""}) == (None, None)


def test_target_first():
    px = _px([100, 105, 121, 80])
    r = evaluate_row(px, None, D0, dt.date(2026, 7, 1), 182, 1.2, 0.9)
    assert r["statut"] == "objectif atteint"


def test_stop_first_and_same_day_counts_stop():
    px = _px([100, 89, 130])
    assert evaluate_row(px, None, D0, dt.date(2026, 7, 1), 182, 1.2, 0.9)["statut"] == "stop touché"
    px = _px([100, 100], highs=[100, 125], lows=[100, 85])
    assert evaluate_row(px, None, D0, dt.date(2026, 7, 1), 182, 1.2, 0.9)["statut"] == "stop touché"


def test_truncated_history_is_flagged():
    px = _px([100, 103, 104], pad=False)  # s'arrête le 7 janvier
    assert evaluate_row(px, None, D0, D0 + dt.timedelta(days=10), 182, 1.2, 0.9)["statut"] == "cours incomplets"


def test_pending_and_matured():
    px = _px([100, 103, 104])
    assert evaluate_row(px, None, D0, D0 + dt.timedelta(days=2), 182, 1.2, 0.9)["statut"] == "en cours"
    r = evaluate_row(px, None, D0, D0 + dt.timedelta(days=200), 182, None, None)
    assert r["statut"] == "échu gagnant" and r["perf"] == pytest.approx(0.04)


def test_excess_vs_benchmark():
    r = evaluate_row(_px([100, 110]), _px([50, 52]), D0, D0 + dt.timedelta(days=200), 182, None, None)
    assert r["exces"] == pytest.approx(0.10 - 0.04)


def test_benchmark_mapping():
    assert benchmark_for("SIE.DE") == "^GDAXI"
    assert benchmark_for("7751.T") == "^N225"
    assert benchmark_for("AAPL") == "^GSPC"
