"""
pg_charts - rendu interactif (pyqtgraph) du graphique symbole + score.

Prototype : pendant du matplotlib `MainWindow._build_symbol_figure_with_score`,
activé par `QSI_CHART_BACKEND=pyqtgraph`. Trois panneaux liés sur l'axe du temps
(prix + moyennes mobiles, MACD, score avec seuils), zoom/pan à la souris et
réticule avec lecture des valeurs.
"""
import logging
import os

import numpy as np
import pandas as pd
import pyqtgraph as pg
import ta

from core.charts import RSI_ZONE_COLORS, rsi_zones
from core.indicators import calculate_macd

logger = logging.getLogger(__name__)

_EPOCH = pd.Timestamp("1970-01-01")


def chart_backend() -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Nom du moteur de graphiques choisi par l'environnement.

    Outputs:
        backend (str): "pyqtgraph" ou "matplotlib" (défaut).
    --------------------------------------------------------------------------
    """
    value = os.environ.get("QSI_CHART_BACKEND", "matplotlib").strip().lower()
    return "pyqtgraph" if value == "pyqtgraph" else "matplotlib"


def _to_epoch_seconds(dates) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Purpose:
        Convertit des dates quelconques en secondes Unix (axe x de pyqtgraph).

    Inputs:
        dates (iterable): DatetimeIndex, Timestamps ou chaînes de dates.

    Outputs:
        seconds (np.ndarray): float64, NaN pour une date illisible.
    --------------------------------------------------------------------------
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return ((idx - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64", na_value=np.nan)


def _event_points(events, xs: np.ndarray, ys: np.ndarray, kind: str):
    """Positions (x, y) des événements `kind` recalées sur la date la plus proche de la série."""
    out_x, out_y = [], []
    if len(xs) == 0:
        return out_x, out_y
    for ev in events or []:
        if str(ev.get("type", "")).upper() != kind:
            continue
        ts = _to_epoch_seconds([ev.get("date")])[0]
        if np.isnan(ts):
            continue
        pos = int(np.nanargmin(np.abs(xs - ts)))
        out_x.append(xs[pos])
        out_y.append(float(ys[pos]))
    return out_x, out_y


def _add_rsi_background(plot, prices: pd.Series) -> None:
    """Fond du panneau prix coloré par zone RSI (fenêtre 17, comme plot_unified_chart)."""
    if len(prices) < 2:
        return
    try:
        rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    except Exception as exc:
        logger.warning("[CHART] RSI indisponible pour le fond : %s", exc)
        return
    # Les bornes sont prises directement en secondes : rsi_zones n'indexe que par position.
    zones = rsi_zones(_to_epoch_seconds(prices.index), rsi)
    if not zones:
        return
    brushes = {}
    for zone, hex_color in RSI_ZONE_COLORS.items():
        color = pg.mkColor(hex_color)
        color.setAlphaF(0.1)
        brushes[zone] = pg.mkBrush(color)
    # Un seul BarGraphItem pour toutes les bandes (un LinearRegionItem par bande doublait
    # le temps de construction). Hauteur très large, exclue de l'autoscale : les bandes
    # remplissent le panneau sans modifier l'échelle des prix.
    lo, hi = float(np.nanmin(prices)), float(np.nanmax(prices))
    span = max(hi - lo, abs(hi), 1.0)
    x0 = np.array([z[0] for z in zones])
    x1 = np.array([z[1] for z in zones])
    bands = pg.BarGraphItem(x0=x0, x1=x1, y0=lo - 100 * span, y1=hi + 100 * span,
                            brushes=[brushes[z[2]] for z in zones], pen=pg.mkPen(None))
    bands.setZValue(-10)
    plot.addItem(bands, ignoreBounds=True)


def build_symbol_chart(sym: str, prices, events=None, score_dates=None, score_values=None,
                       buy_thr=None, sell_thr=None, title: str = None) -> pg.GraphicsLayoutWidget:
    """
    --------------------------------------------------------------------------
    Purpose:
        Construit le widget interactif prix / MACD / score d'un symbole.

    Inputs:
        sym (str): symbole affiché.
        prices (pd.Series): clôtures indexées par date.
        events (list[dict]): événements backtest {'type': BUY|SELL, 'date', 'price'}.
        score_dates (list): dates de la série de score.
        score_values (list): valeurs de la série de score.
        buy_thr (float | None): seuil d'achat appliqué au score.
        sell_thr (float | None): seuil de vente appliqué au score.
        title (str | None): titre du panneau prix (défaut : le symbole).

    Outputs:
        widget (pg.GraphicsLayoutWidget): widget Qt prêt à insérer dans un layout.
    --------------------------------------------------------------------------
    """
    events = events or []
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    prices = pd.Series(prices).astype("float64")

    widget = pg.GraphicsLayoutWidget()
    widget.setBackground("w")
    widget.setMinimumHeight(520)

    x = _to_epoch_seconds(prices.index)
    y = prices.to_numpy()

    # Panneau 1 : prix et moyennes mobiles (mêmes réglages que plot_unified_chart).
    p_price = widget.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
    p_price.setTitle(title or sym, color="k", size="10pt")
    p_price.addLegend(offset=(10, 10), brush=pg.mkBrush(255, 255, 255, 190))
    p_price.showGrid(x=True, y=True, alpha=0.25)
    _add_rsi_background(p_price, prices)
    p_price.plot(x, y, pen=pg.mkPen("#1f77b4", width=1.8), name="Prix")
    p_price.plot(x, prices.ewm(span=20, adjust=False).mean().to_numpy(),
                 pen=pg.mkPen("orange", width=1.2, style=pg.QtCore.Qt.DashLine), name="EMA20")
    p_price.plot(x, prices.ewm(span=50, adjust=False).mean().to_numpy(),
                 pen=pg.mkPen("purple", width=1.2, style=pg.QtCore.Qt.DashDotLine), name="EMA50")
    if len(prices) >= 50:
        p_price.plot(x, prices.rolling(window=50).mean().to_numpy(),
                     pen=pg.mkPen("green", width=1.2, style=pg.QtCore.Qt.DotLine), name="SMA50")

    for kind, symbol, brush in (("BUY", "t1", "g"), ("SELL", "t", "r")):
        ex, ey = [], []
        for ev in events:
            if str(ev.get("type", "")).upper() != kind:
                continue
            ts = _to_epoch_seconds([ev.get("date")])[0]
            try:
                ey_val = float(ev.get("price"))
            except (TypeError, ValueError):
                continue
            if not np.isnan(ts):
                ex.append(ts)
                ey.append(ey_val)
        if ex:
            p_price.plot(ex, ey, pen=None, symbol=symbol, symbolSize=12,
                         symbolBrush=brush, symbolPen="k", name=kind)

    # Panneau 2 : MACD et histogramme.
    p_macd = widget.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem()})
    p_macd.setMaximumHeight(130)
    p_macd.showGrid(x=True, y=True, alpha=0.25)
    p_macd.setLabel("left", "MACD")
    macd, signal_line = calculate_macd(prices)
    hist = (macd - signal_line).to_numpy()
    if len(x) > 1:
        bar_w = float(np.nanmedian(np.diff(x))) * 0.8
        brushes = [pg.mkBrush(0, 160, 0, 110) if h >= 0 else pg.mkBrush(200, 0, 0, 110) for h in hist]
        p_macd.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(hist), width=bar_w, brushes=brushes, pen=None))
    p_macd.plot(x, macd.to_numpy(), pen=pg.mkPen("#1f77b4", width=1.2))
    p_macd.plot(x, signal_line.to_numpy(), pen=pg.mkPen("orange", width=1.2))
    p_macd.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("gray", style=pg.QtCore.Qt.DashLine)))
    p_macd.setXLink(p_price)

    # Panneau 3 : score au fil du temps, seuils et positions du backtest.
    p_score = widget.addPlot(row=2, col=0, axisItems={"bottom": pg.DateAxisItem()})
    p_score.setMaximumHeight(160)
    p_score.showGrid(x=True, y=True, alpha=0.25)
    p_score.setLabel("left", "Score")
    p_score.setXLink(p_price)
    if score_dates and score_values:
        sx = _to_epoch_seconds(score_dates)
        sy = np.asarray(score_values, dtype="float64")
        p_score.addLegend(offset=(10, 5), brush=pg.mkBrush(255, 255, 255, 190))
        p_score.plot(sx, sy, pen=pg.mkPen("#1565C0", width=1.6), name="Score")
        for thr, color, label in ((buy_thr, "g", "Seuil Achat"), (sell_thr, "r", "Seuil Vente")):
            if thr is None:
                continue
            # Tracé comme une courbe nommée pour que le seuil figure dans la légende.
            p_score.plot([np.nanmin(sx), np.nanmax(sx)], [float(thr)] * 2,
                         pen=pg.mkPen(color, width=1, style=pg.QtCore.Qt.DashLine),
                         name=f"{label} ({float(thr):.2f})")
        for kind, symbol, brush in (("BUY", "t1", "g"), ("SELL", "t", "r")):
            ex, ey = _event_points(events, sx, sy, kind)
            if ex:
                p_score.plot(ex, ey, pen=None, symbol=symbol, symbolSize=9, symbolBrush=brush, symbolPen="k")
    else:
        p_score.addItem(pg.TextItem("Score indisponible", color="k", anchor=(0.5, 0.5)))

    # Dates uniquement sous le dernier panneau, comme la version matplotlib.
    for p in (p_price, p_macd):
        p.getAxis("bottom").setStyle(showValues=False)
    for p in (p_price, p_macd, p_score):
        p.getAxis("left").setWidth(60)

    _add_crosshair(widget, (p_price, p_macd, p_score), x, y)
    return widget


def _add_crosshair(widget, plots, x: np.ndarray, y: np.ndarray) -> None:
    """Réticule vertical partagé ; une étiquette en haut du panneau prix affiche date et prix."""
    if len(x) == 0:
        return
    lines = []
    for p in plots:
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888", width=0.8))
        p.addItem(line, ignoreBounds=True)
        lines.append(line)
    readout = pg.TextItem(color="k", anchor=(0, 0))
    plots[0].addItem(readout, ignoreBounds=True)
    vb = plots[0].vb

    def _on_move(pos):
        if not plots[0].sceneBoundingRect().contains(pos) and not any(
                p.sceneBoundingRect().contains(pos) for p in plots[1:]):
            return
        mx = vb.mapSceneToView(pos).x()
        i = int(np.clip(np.searchsorted(x, mx), 0, len(x) - 1))
        for line in lines:
            line.setPos(x[i])
        date = pd.Timestamp(x[i], unit="s").strftime("%Y-%m-%d")
        readout.setText(f"{date}  {y[i]:.2f}")
        (x0, x1), (y0, y1) = vb.viewRange()
        readout.setPos(x0, y1)

    # Garder une référence : sans elle le SignalProxy est détruit et le réticule reste figé.
    widget._crosshair_proxy = pg.SignalProxy(widget.scene().sigMouseMoved, rateLimit=60,
                                             slot=lambda evt: _on_move(evt[0]))
