"""
Screeners Finviz *market-wide*.

Contrairement aux screeners store-based (limités au catalogue local), ceux-ci
interrogent l'UNIVERS entier du marché US en 1 requête Finviz et renvoient les
tickers correspondants — pour DÉCOUVRIR de nouveaux titres, pas seulement filtrer
les siens. Coût : 1 requête Finviz par screener (0 requête yfinance).

Chaque preset mappe une stratégie vers un dict de filtres Finviz (clés/options
exactes issues de finvizfinance.constants.filter_dict).
"""
import pandas as pd

# Presets : nom interne → {title, filters, order}
PRESETS = {
    "big_growth": {
        "title": "Big Growth (marché) — croissance forte + accumulation volume",
        # Fidèle aux critères G1/G4/G5 : croissance CA & EPS courante (YoY) + volume
        # relatif élevé (accumulation). Filtres AND ⇒ on garde les 4 plus discriminants.
        "filters": {
            "Sales growthqtr over qtr": "Over 20%",
            "EPS growththis year": "Over 20%",
            "P/E": "Profitable (>0)",
            "Relative Volume": "Over 1.5",
        },
    },
    "garp": {
        "title": "GARP — croissance à prix raisonnable (PEG<1, EPS>15%)",
        "filters": {
            "PEG": "Under 1",
            "EPS growthpast 5 years": "Over 15%",
            "P/E": "Under 25",
        },
    },
    "secure_growth": {
        "title": "Secure / Quality Defensive (large, peu endetté, dividende)",
        "filters": {
            "Market Cap.": "+Large (over $10bln)",
            "Debt/Equity": "Under 1",
            "Beta": "Under 1",
            "Dividend Yield": "Positive (>0%)",
            "Net Profit Margin": "Positive (>0%)",
        },
    },
    "minervini": {
        "title": "Minervini Trend Template — uptrend + proche du +haut 52s",
        # SEPA fidèle : prix au-dessus des SMA50/200 ET à ≤ 5% du plus-haut 52s
        # (leaders en tendance forte), avec liquidité minimale.
        "filters": {
            "50-Day Simple Moving Average": "Price above SMA50",
            "200-Day Simple Moving Average": "Price above SMA200",
            "52-Week High/Low": "0-5% below High",
            "Average Volume": "Over 500K",
        },
    },
    "magic_formula": {
        "title": "Magic Formula-like — qualité (ROA/ROE élevés) + valorisation basse",
        # Approxime Greenblatt (rendement du capital élevé + earnings yield élevé)
        # via ROA>15%, ROE>30% et P/E bas.
        "filters": {
            "Return on Assets": "Over +15%",
            "Return on Equity": "Very Positive (>30%)",
            "P/E": "Under 15",
        },
    },
    "rs_leaders": {
        "title": "RS Leaders — perf 6 mois ≥ +30%, au-dessus SMA200",
        "filters": {
            "Performance": "Half +30%",
            "Average Volume": "Over 500K",
            "200-Day Simple Moving Average": "Price above SMA200",
        },
    },
    "new_high": {
        "title": "Nouveaux plus-hauts 52 semaines (breakout)",
        "filters": {
            "52-Week High/Low": "New High",
            "Average Volume": "Over 500K",
        },
    },
    "oversold_quality": {
        "title": "Oversold Quality — RSI<30 sur large cap à dividende",
        "filters": {
            "RSI (14)": "Oversold (30)",
            "Market Cap.": "+Large (over $10bln)",
            "Dividend Yield": "Positive (>0%)",
        },
    },
    "low_vol_def": {
        "title": "Low-Vol Défensif — beta < 0.5, large cap, dividende > 2%",
        "filters": {
            "Market Cap.": "+Large (over $10bln)",
            "Beta": "Under 0.5",
            "Dividend Yield": "Over 2%",
        },
    },
    "gap_up": {
        "title": "Gap Up ≥ 5% (marché) — cassures haussières du jour",
        # Remplace le gap up sur symboles populaires : tout le marché, pas le catalogue.
        "filters": {
            "Gap": "Up 5%",
            "Average Volume": "Over 500K",
        },
        "order": "Change",
    },
    "gap_down": {
        "title": "Gap Down ≥ 5% (marché) — décrochages du jour",
        "filters": {
            "Gap": "Down 5%",
            "Average Volume": "Over 500K",
        },
        "order": "Change",
        "ascend": True,   # gaps les plus négatifs en premier
    },
}


def _num(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _run_finviz(filters: dict, order: str = "Change", limit: int = 100, ascend: bool = False) -> pd.DataFrame:
    """Exécute un screen Finviz Overview avec contournement du bot-blocking
    (session curl_cffi impersonate). Retourne le DataFrame brut de finvizfinance.
    Lève une exception explicite si la dépendance manque."""
    from finvizfinance.screener.overview import Overview
    import finvizfinance.util as _fv_util
    from curl_cffi.requests import Session as CurlSession

    _orig = _fv_util.session
    _fv_util.session = CurlSession(impersonate="chrome")
    try:
        fov = Overview()
        fov.set_filter(filters_dict=filters)
        return fov.screener_view(order=order, limit=limit, ascend=ascend)
    finally:
        _fv_util.session = _orig


def run_preset(key: str, limit: int = 100) -> dict:
    """Exécute un preset market-wide. Retourne {title, headers, rows} pour le dialog."""
    preset = PRESETS.get(key)
    if preset is None:
        raise ValueError(f"Preset Finviz inconnu : {key}")

    df = _run_finviz(
        preset["filters"], order=preset.get("order", "Change"),
        limit=limit, ascend=preset.get("ascend", False),
    )
    headers = ["Symbole", "Secteur", "Cap", "P/E", "Prix", "Var %"]
    rows = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            sym = str(r.get("Ticker") or "").strip().upper()
            if not sym:
                continue
            change = _num(r.get("Change"))
            rows.append([
                sym,
                str(r.get("Sector") or "N/A"),
                str(r.get("Market Cap") or "N/A"),
                _num(r.get("P/E")),
                _num(r.get("Price")),
                round(change * 100, 2) if change is not None else None,
            ])
    return {
        "title": f"{preset['title']} — {len(rows)} résultat(s)",
        "headers": headers,
        "rows": rows,
    }
