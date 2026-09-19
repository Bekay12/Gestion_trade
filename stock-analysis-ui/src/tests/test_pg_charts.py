"""
Tests du prototype pyqtgraph (ui/pg_charts.py) : hors ligne, rendu offscreen.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyqtgraph")
from PyQt5.QtWidgets import QApplication

from ui.pg_charts import _to_epoch_seconds, build_symbol_chart, chart_backend


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_backend_defaults_to_matplotlib(monkeypatch):
    monkeypatch.delenv("QSI_CHART_BACKEND", raising=False)
    assert chart_backend() == "matplotlib"
    monkeypatch.setenv("QSI_CHART_BACKEND", " PyQtGraph ")
    assert chart_backend() == "pyqtgraph"
    monkeypatch.setenv("QSI_CHART_BACKEND", "autre")
    assert chart_backend() == "matplotlib"


def test_epoch_seconds_handles_tz_and_bad_dates():
    naive = _to_epoch_seconds(["1970-01-02"])
    aware = _to_epoch_seconds(pd.DatetimeIndex(["1970-01-02"], tz="America/New_York"))
    bad = _to_epoch_seconds(["pas une date"])
    assert naive[0] == pytest.approx(86400.0)
    assert aware[0] == pytest.approx(86400.0)
    assert np.isnan(bad[0])


def test_build_chart_with_events_and_score(qapp):
    idx = pd.bdate_range("2024-01-01", periods=120)
    prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    events = [
        {"type": "BUY", "date": idx[10], "price": prices.iloc[10]},
        {"type": "SELL", "date": str(idx[80].date()), "price": prices.iloc[80]},
        {"type": "BUY", "date": "illisible", "price": 1.0},
    ]
    w = build_symbol_chart("TEST", prices, events=events, score_dates=list(idx),
                           score_values=list(np.linspace(-3, 3, len(idx))), buy_thr=2.0, sell_thr=-2.0)
    plots = [w.ci.getItem(r, 0) for r in range(3)]
    assert all(p is not None for p in plots)
    # Les trois panneaux partagent l'axe du temps.
    assert plots[1].vb.linkedView(0) is plots[0].vb
    assert plots[2].vb.linkedView(0) is plots[0].vb


def test_build_chart_without_score(qapp):
    idx = pd.bdate_range("2024-01-01", periods=5)
    w = build_symbol_chart("X", pd.Series([1.0, 2, 3, 2, 1], index=idx))
    assert w.minimumHeight() == 520


def test_rsi_zones_matches_per_day_loop():
    """Les bandes regroupées couvrent exactement ce que traçait l'ancien axvspan jour par jour."""
    from core.charts import rsi_zones

    idx = pd.bdate_range("2020-01-01", periods=400)
    rsi = pd.Series(np.random.default_rng(3).uniform(10, 90, len(idx)), index=idx)
    rsi.iloc[:17] = np.nan  # début de fenêtre RSI : zone neutre, comme avant

    def zone(v):
        return "surachat" if v > 70 else "survente" if v < 30 else "neutre"

    expected = [(idx[i - 1], idx[i], zone(rsi.iloc[i - 1])) for i in range(1, len(idx))]
    zones = rsi_zones(idx, rsi)
    rebuilt = []
    for start, end, z in zones:
        a, b = idx.get_loc(start), idx.get_loc(end)
        rebuilt += [(idx[i], idx[i + 1], z) for i in range(a, b)]
    assert rebuilt == expected
    assert all(zones[k][2] != zones[k + 1][2] for k in range(len(zones) - 1))
    assert rsi_zones(idx[:1], rsi[:1]) == []
