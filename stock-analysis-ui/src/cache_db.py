"""
SQLite-backed market data warehouse — shim de compatibilité.

Ce module délègue désormais toutes les opérations vers market_store.py
(backend Parquet + DuckDB, conçu pour 5 000+ stocks / 30 ans d'historique).

Les imports existants de la forme `from cache_db import ...` continuent de
fonctionner sans modification.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Délégation vers market_store (nouveau backend Parquet + DuckDB)
# ---------------------------------------------------------------------------
from market_store import (  # noqa: F401  (ré-export public)
    DEFAULT_FEATURE_START_DATE,
    DEFAULT_BOOTSTRAP_SYMBOLS,
    PARQUET_DIR,
    ensure_market_data_schema,
    upsert_instrument,
    store_price_history,
    store_fundamental_snapshot,
    store_daily_feature_series,
    fetch_and_store_symbol_series,
    bootstrap_market_database,
    get_symbol_storage_summary,
    get_latest_feature_row,
    get_symbol_last_trade_date,
    refresh_symbol_incremental,
    query_features,
    _build_daily_feature_frame,
    _get_rate_to_usd,
    _normalize_symbol,
    _safe_float,
    _safe_int,
    _safe_text,
    _compute_rsi,
    _cap_range_from_market_cap_b,
    _safe_score_signal,
    _prepare_history_frame,
    _build_quarter_feature_points,
    # Nouveau : financial cache (remplace pickle)
    get_financial_cache,
    save_financial_cache,
    # Nouveau : timeline (remplace SQLite)
    store_timeline_earnings,
    store_timeline_recommendations,
    store_timeline_insider,
    get_timeline_pit_data,
    update_timeline_data,
)

# ---------------------------------------------------------------------------
# Garde l'ancien chemin SQLite accessible pour les scripts de migration
# ---------------------------------------------------------------------------
import sqlite3
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from config import DB_PATH
from fundamentals_cache import get_all_quarters_sorted, get_fundamental_metrics

# ---------------------------------------------------------------------------
# Tout le reste du module (helpers internes SQLite) est conservé ci-dessous
# pour rétrocompatibilité avec les scripts d'analyse qui lisent directement
# la base SQLite ou appellent _connect().  Aucune rupture de contrat.
# ---------------------------------------------------------------------------


DEFAULT_FEATURE_START_DATE = "2016-01-01"
DEFAULT_BOOTSTRAP_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "TSM",
    "02020.HK",
    "ENR.DE",
)
QUARTER_PUBLICATION_DELAY_DAYS = 45
_FX_RATE_CACHE: dict[str, float] = {}


def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _flag(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(int)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _cap_range_from_market_cap_b(market_cap_b: Optional[float]) -> str:
    mc = _safe_float(market_cap_b)
    if mc is None or mc <= 0:
        return "Unknown"
    if mc < 2.0:
        return "Small"
    if mc < 10.0:
        return "Mid"
    return "Large"


def _safe_score_signal(score: float) -> str:
    if score >= 3.0:
        return "ACHAT"
    if score <= 1.0:
        return "VENTE"
    return "NEUTRE"


def _get_rate_to_usd(currency: str) -> float:
    cur = str(currency or "USD").strip().upper()
    if not cur or cur == "USD":
        return 1.0
    if cur in _FX_RATE_CACHE:
        return _FX_RATE_CACHE[cur]

    try:
        pair = f"{cur}USD=X"
        hist = yf.download(pair, period="7d", interval="1d", progress=False, auto_adjust=False)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            rate = _safe_float(hist["Close"].dropna().iloc[-1])
            if rate and rate > 0:
                _FX_RATE_CACHE[cur] = rate
                return rate
    except Exception:
        pass

    _FX_RATE_CACHE[cur] = 1.0
    return 1.0


def _ensure_table_columns(conn: sqlite3.Connection, table: str, column_defs: dict[str, str]) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for col_name, col_def in column_defs.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_market_data_schema() -> None:
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_instruments (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            short_name TEXT,
            long_name TEXT,
            sector TEXT,
            industry TEXT,
            country TEXT,
            exchange TEXT,
            currency TEXT,
            quote_type TEXT,
            shares_outstanding REAL,
            market_cap REAL,
            enterprise_value REAL,
            first_seen_at TEXT NOT NULL,
            last_profile_refresh TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'yfinance'
        );

        CREATE TABLE IF NOT EXISTS market_price_history (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL,
            dividends REAL,
            stock_splits REAL,
            currency TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date),
            FOREIGN KEY (symbol) REFERENCES market_instruments(symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_market_price_symbol_date
            ON market_price_history(symbol, trade_date DESC);

        CREATE TABLE IF NOT EXISTS market_fundamental_snapshots (
            symbol TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            name TEXT,
            sector TEXT,
            industry TEXT,
            currency TEXT,
            current_price REAL,
            market_cap REAL,
            enterprise_value REAL,
            shares_outstanding REAL,
            trailing_pe REAL,
            forward_pe REAL,
            peg_ratio REAL,
            price_to_book REAL,
            price_to_sales REAL,
            enterprise_to_revenue REAL,
            enterprise_to_ebitda REAL,
            gross_margins REAL,
            operating_margins REAL,
            profit_margins REAL,
            revenue_growth REAL,
            earnings_growth REAL,
            return_on_equity REAL,
            debt_to_equity REAL,
            total_revenue REAL,
            gross_profits REAL,
            ebitda REAL,
            free_cashflow REAL,
            operating_cashflow REAL,
            trailing_eps REAL,
            dividend_rate REAL,
            dividend_yield REAL,
            payout_ratio REAL,
            beta REAL,
            fifty_two_week_high REAL,
            fifty_two_week_low REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, as_of_date),
            FOREIGN KEY (symbol) REFERENCES market_instruments(symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_market_snapshot_symbol_date
            ON market_fundamental_snapshots(symbol, as_of_date DESC);

        CREATE TABLE IF NOT EXISTS market_feature_series (
            symbol TEXT NOT NULL,
            feature_date TEXT NOT NULL,
            name TEXT,
            sector TEXT,
            price REAL,
            currency TEXT,
            fx_rate_to_usd REAL,
            values_in_usd INTEGER,
            market_cap_b REAL,
            score INTEGER,
            signal TEXT,
            trend TEXT,
            rsi REAL,
            volume_mean_usd REAL,
            domaine TEXT,
            cap_range TEXT,
            param_key TEXT,
            score_over_threshold REAL,
            consensus TEXT,
            consensus_mean REAL,
            seuil_achat REAL,
            seuil_vente REAL,
            selected_param_key TEXT,
            dprice REAL,
            var5j_pct REAL,
            drsi REAL,
            dvolrel REAL,
            rev_growth_pct REAL,
            ebitda_yield_pct REAL,
            fcf_yield_pct REAL,
            de_ratio REAL,
            market_cap_b_usd REAL,
            roe_pct REAL,
            fiabilite REAL,
            nb_trades INTEGER,
            gagnants INTEGER,
            gain_total_usd REAL,
            gain_moyen_usd REAL,
            detection_time TEXT,
            c1_revenue_growth_pct REAL,
            c1_ok INTEGER,
            c2_gross_margin_pct REAL,
            c2_ok INTEGER,
            c3_undervaluation_hits INTEGER,
            c3_pe_ok INTEGER,
            c3_peg_ok INTEGER,
            c3_below_75pct_52wh INTEGER,
            c3_ok INTEGER,
            c4_momentum_3m_pct REAL,
            c4_momentum_6m_pct REAL,
            c4_above_sma50 INTEGER,
            c4_ok INTEGER,
            c5_volume_ratio REAL,
            c5_ok INTEGER,
            close_vs_52wh_ratio REAL,
            sma50 REAL,
            avg_volume_30d REAL,
            avg_volume_90d REAL,
            trailing_pe REAL,
            peg_ratio REAL,
            ttm_eps REAL,
            eps_growth_pct REAL,
            dividend_yield REAL,
            payout_ratio REAL,
            margin_stability_4q REAL,
            dividend_stability_1y REAL,
            event_count INTEGER,
            event_flags TEXT,
            source_publication_date TEXT,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (symbol, feature_date),
            FOREIGN KEY (symbol) REFERENCES market_instruments(symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_market_feature_symbol_date
            ON market_feature_series(symbol, feature_date DESC);

        CREATE TABLE IF NOT EXISTS market_events (
            symbol TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_label TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (symbol, event_date, event_type, event_label),
            FOREIGN KEY (symbol) REFERENCES market_instruments(symbol)
        );

        CREATE TABLE IF NOT EXISTS market_ingestion_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            symbols TEXT NOT NULL,
            start_date TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT
        );
        """
    )

    _ensure_table_columns(
        conn,
        "market_feature_series",
        {
            "currency": "TEXT",
            "fx_rate_to_usd": "REAL",
            "values_in_usd": "INTEGER",
            "signal": "TEXT",
            "trend": "TEXT",
            "rsi": "REAL",
            "volume_mean_usd": "REAL",
            "domaine": "TEXT",
            "cap_range": "TEXT",
            "param_key": "TEXT",
            "score_over_threshold": "REAL",
            "consensus": "TEXT",
            "consensus_mean": "REAL",
            "seuil_achat": "REAL",
            "seuil_vente": "REAL",
            "selected_param_key": "TEXT",
            "dprice": "REAL",
            "var5j_pct": "REAL",
            "drsi": "REAL",
            "dvolrel": "REAL",
            "rev_growth_pct": "REAL",
            "ebitda_yield_pct": "REAL",
            "fcf_yield_pct": "REAL",
            "de_ratio": "REAL",
            "market_cap_b_usd": "REAL",
            "roe_pct": "REAL",
            "fiabilite": "REAL",
            "nb_trades": "INTEGER",
            "gagnants": "INTEGER",
            "gain_total_usd": "REAL",
            "gain_moyen_usd": "REAL",
            "detection_time": "TEXT",
        },
    )

    conn.commit()
    conn.close()


def _prepare_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy().reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True).dt.tz_localize(None)
    frame = frame.rename(columns={date_col: "trade_date"})
    return frame.sort_values("trade_date").reset_index(drop=True)


def _build_quarter_feature_points(symbol: str) -> pd.DataFrame:
    quarters = get_all_quarters_sorted(symbol)
    if not quarters:
        return pd.DataFrame(
            columns=[
                "source_publication_date",
                "c1_revenue_growth_pct",
                "c2_gross_margin_pct",
                "ttm_eps",
                "eps_growth_pct",
                "margin_stability_4q",
            ]
        )

    qdf = pd.DataFrame(quarters)
    qdf["quarter_date"] = pd.to_datetime(qdf["quarter_date"], errors="coerce")
    qdf = qdf.dropna(subset=["quarter_date"]).sort_values("quarter_date").drop_duplicates("quarter_date")
    if qdf.empty:
        return pd.DataFrame()

    qdf["source_publication_date"] = qdf["quarter_date"] + timedelta(days=QUARTER_PUBLICATION_DELAY_DAYS)

    revenue = qdf["total_revenue"].astype(float)
    gross_profit = qdf["gross_profit"].astype(float)
    diluted_eps = qdf["diluted_eps"].astype(float)

    qdf["c1_revenue_growth_pct"] = ((revenue - revenue.shift(4)) / revenue.shift(4).abs()) * 100.0
    qdf["c2_gross_margin_pct"] = (gross_profit / revenue.abs()) * 100.0
    qdf["ttm_eps"] = diluted_eps.rolling(4, min_periods=1).sum()
    qdf["eps_growth_pct"] = ((diluted_eps - diluted_eps.shift(4)) / diluted_eps.shift(4).abs()) * 100.0
    qdf["margin_stability_4q"] = qdf["c2_gross_margin_pct"].rolling(4, min_periods=2).std()

    return qdf[
        [
            "source_publication_date",
            "c1_revenue_growth_pct",
            "c2_gross_margin_pct",
            "ttm_eps",
            "eps_growth_pct",
            "margin_stability_4q",
        ]
    ].sort_values("source_publication_date")


def _build_daily_feature_frame(symbol: str, history: pd.DataFrame, info: dict) -> pd.DataFrame:
    frame = _prepare_history_frame(history)
    if frame.empty:
        return frame

    currency = str(info.get("currency") or "USD").upper()
    fx_rate_to_usd = _get_rate_to_usd(currency)

    frame["price"] = frame["Close"].astype(float) * fx_rate_to_usd
    frame["currency"] = currency
    frame["fx_rate_to_usd"] = fx_rate_to_usd
    frame["values_in_usd"] = 1
    frame["avg_volume_30d"] = frame["Volume"].rolling(30, min_periods=10).mean()
    frame["avg_volume_90d"] = frame["Volume"].rolling(90, min_periods=30).mean()
    frame["c5_volume_ratio"] = frame["avg_volume_30d"] / frame["avg_volume_90d"]
    frame["volume_mean_usd"] = frame["avg_volume_30d"] * frame["price"]
    frame["sma50"] = frame["price"].rolling(50, min_periods=20).mean()
    frame["c4_momentum_3m_pct"] = frame["price"].pct_change(63) * 100.0
    frame["c4_momentum_6m_pct"] = frame["price"].pct_change(126) * 100.0
    frame["rolling_52w_high"] = frame["price"].rolling(252, min_periods=20).max()
    frame["close_vs_52wh_ratio"] = frame["price"] / frame["rolling_52w_high"]
    frame["c4_above_sma50"] = frame["price"] > frame["sma50"]
    frame["c4_ok"] = (
        frame["c4_momentum_3m_pct"].between(8.0, 60.0, inclusive="neither")
        & frame["c4_above_sma50"]
        & (frame["c4_momentum_6m_pct"] < 150.0)
    )
    frame["c5_ok"] = frame["c5_volume_ratio"] > 1.20

    if "Dividends" in frame.columns:
        trailing_dividend = frame["Dividends"].rolling(252, min_periods=20).sum()
        frame["dividend_stability_1y"] = trailing_dividend.rolling(126, min_periods=20).std()
    else:
        frame["dividend_stability_1y"] = pd.NA

    quarter_points = _build_quarter_feature_points(symbol)
    if not quarter_points.empty:
        frame = pd.merge_asof(
            frame.sort_values("trade_date"),
            quarter_points.sort_values("source_publication_date"),
            left_on="trade_date",
            right_on="source_publication_date",
            direction="backward",
        )
    else:
        frame["source_publication_date"] = pd.NaT
        frame["c1_revenue_growth_pct"] = pd.NA
        frame["c2_gross_margin_pct"] = pd.NA
        frame["ttm_eps"] = pd.NA
        frame["eps_growth_pct"] = pd.NA
        frame["margin_stability_4q"] = pd.NA

    frame["trailing_pe"] = frame["price"] / frame["ttm_eps"]
    frame.loc[frame["ttm_eps"].isna() | (frame["ttm_eps"] <= 0), "trailing_pe"] = pd.NA
    frame["peg_ratio"] = frame["trailing_pe"] / frame["eps_growth_pct"]
    frame.loc[frame["eps_growth_pct"].isna() | (frame["eps_growth_pct"] <= 0), "peg_ratio"] = pd.NA

    frame["c1_ok"] = frame["c1_revenue_growth_pct"] > 20.0
    frame["c2_ok"] = frame["c2_gross_margin_pct"] > 30.0
    frame["c3_pe_ok"] = frame["trailing_pe"].between(0.0, 25.0, inclusive="neither")
    frame["c3_peg_ok"] = frame["peg_ratio"].between(0.0, 1.5, inclusive="neither")
    frame["c3_below_75pct_52wh"] = frame["close_vs_52wh_ratio"] < 0.75
    frame["c3_undervaluation_hits"] = (
        _flag(frame["c3_pe_ok"])
        + _flag(frame["c3_peg_ok"])
        + _flag(frame["c3_below_75pct_52wh"])
    )
    frame["c3_ok"] = frame["c3_undervaluation_hits"] >= 2

    shares_outstanding = _safe_float(info.get("sharesOutstanding"))
    frame["market_cap_b"] = pd.NA
    if shares_outstanding and shares_outstanding > 0:
        frame["market_cap_b"] = (frame["price"] * shares_outstanding) / 1_000_000_000.0

    name = info.get("shortName") or info.get("longName") or symbol
    sector = info.get("sector") or "N/A"
    frame["name"] = name
    frame["sector"] = sector
    frame["domaine"] = sector
    frame["dividend_yield"] = _safe_float(info.get("dividendYield"))
    frame["payout_ratio"] = _safe_float(info.get("payoutRatio"))
    frame["event_count"] = 0
    frame["event_flags"] = None
    frame["score"] = (
        _flag(frame["c1_ok"])
        + _flag(frame["c2_ok"])
        + _flag(frame["c3_ok"])
        + _flag(frame["c4_ok"])
        + _flag(frame["c5_ok"])
    )

    frame["rsi"] = _compute_rsi(frame["price"])
    frame["trend"] = frame.apply(
        lambda r: "Hausse" if (_safe_float(r.get("price")) or 0.0) >= (_safe_float(r.get("sma50")) or 0.0) else "Baisse",
        axis=1,
    )
    frame["signal"] = frame["score"].apply(_safe_score_signal)

    default_seuil_achat = 4.2
    default_seuil_vente = -0.5
    frame["seuil_achat"] = default_seuil_achat
    frame["seuil_vente"] = default_seuil_vente
    frame["score_over_threshold"] = frame["score"] / default_seuil_achat
    frame["consensus"] = "Neutre"
    frame["consensus_mean"] = pd.NA

    frame["dprice"] = frame["price"].pct_change(1) * 100.0
    frame["var5j_pct"] = frame["price"].pct_change(5) * 100.0
    frame["drsi"] = frame["rsi"].diff(1)
    frame["dvolrel"] = (frame["c5_volume_ratio"] - 1.0) * 100.0

    frame["rev_growth_pct"] = frame["c1_revenue_growth_pct"]
    market_cap_usd = _safe_float(info.get("marketCap"))
    if market_cap_usd:
        market_cap_usd = market_cap_usd * fx_rate_to_usd
    ev_usd = _safe_float(info.get("enterpriseValue"))
    if ev_usd:
        ev_usd = ev_usd * fx_rate_to_usd
    ebitda_usd = _safe_float(info.get("ebitda"))
    if ebitda_usd:
        ebitda_usd = ebitda_usd * fx_rate_to_usd
    fcf_usd = _safe_float(info.get("freeCashflow"))
    if fcf_usd:
        fcf_usd = fcf_usd * fx_rate_to_usd

    denom_ebitda = ev_usd if ev_usd and ev_usd > 0 else market_cap_usd
    ebitda_yield_pct = (ebitda_usd / denom_ebitda) * 100.0 if (ebitda_usd and denom_ebitda and denom_ebitda > 0) else pd.NA
    fcf_yield_pct = (fcf_usd / market_cap_usd) * 100.0 if (fcf_usd and market_cap_usd and market_cap_usd > 0) else pd.NA

    frame["ebitda_yield_pct"] = ebitda_yield_pct
    frame["fcf_yield_pct"] = fcf_yield_pct
    frame["de_ratio"] = _safe_float(info.get("debtToEquity"))
    roe = _safe_float(info.get("returnOnEquity"))
    frame["roe_pct"] = (roe * 100.0) if roe is not None else pd.NA
    frame["market_cap_b_usd"] = (market_cap_usd / 1e9) if market_cap_usd else frame["market_cap_b"]

    frame["cap_range"] = frame["market_cap_b"].apply(_cap_range_from_market_cap_b)
    frame["param_key"] = frame["domaine"].fillna("Inconnu").astype(str) + "_" + frame["cap_range"].fillna("Unknown").astype(str)
    frame["selected_param_key"] = frame["param_key"]
    frame["fiabilite"] = pd.NA
    frame["nb_trades"] = 0
    frame["gagnants"] = 0
    frame["gain_total_usd"] = 0.0
    frame["gain_moyen_usd"] = 0.0

    frame["computed_at"] = _utcnow()
    frame["feature_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["detection_time"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["source_publication_date"] = pd.to_datetime(
        frame["source_publication_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    return frame[
        [
            "feature_date",
            "name",
            "sector",
            "price",
            "currency",
            "fx_rate_to_usd",
            "values_in_usd",
            "market_cap_b",
            "score",
            "signal",
            "trend",
            "rsi",
            "volume_mean_usd",
            "domaine",
            "cap_range",
            "param_key",
            "score_over_threshold",
            "consensus",
            "consensus_mean",
            "seuil_achat",
            "seuil_vente",
            "selected_param_key",
            "dprice",
            "var5j_pct",
            "drsi",
            "dvolrel",
            "rev_growth_pct",
            "ebitda_yield_pct",
            "fcf_yield_pct",
            "de_ratio",
            "market_cap_b_usd",
            "roe_pct",
            "fiabilite",
            "nb_trades",
            "gagnants",
            "gain_total_usd",
            "gain_moyen_usd",
            "detection_time",
            "c1_revenue_growth_pct",
            "c1_ok",
            "c2_gross_margin_pct",
            "c2_ok",
            "c3_undervaluation_hits",
            "c3_pe_ok",
            "c3_peg_ok",
            "c3_below_75pct_52wh",
            "c3_ok",
            "c4_momentum_3m_pct",
            "c4_momentum_6m_pct",
            "c4_above_sma50",
            "c4_ok",
            "c5_volume_ratio",
            "c5_ok",
            "close_vs_52wh_ratio",
            "sma50",
            "avg_volume_30d",
            "avg_volume_90d",
            "trailing_pe",
            "peg_ratio",
            "ttm_eps",
            "eps_growth_pct",
            "dividend_yield",
            "payout_ratio",
            "margin_stability_4q",
            "dividend_stability_1y",
            "event_count",
            "event_flags",
            "source_publication_date",
            "computed_at",
        ]
    ]


def upsert_instrument(symbol: str, info: dict) -> None:
    symbol = _normalize_symbol(symbol)
    now = _utcnow()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO market_instruments (
            symbol, name, short_name, long_name, sector, industry, country,
            exchange, currency, quote_type, shares_outstanding, market_cap,
            enterprise_value, first_seen_at, last_profile_refresh, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'yfinance')
        ON CONFLICT(symbol) DO UPDATE SET
            name=excluded.name,
            short_name=excluded.short_name,
            long_name=excluded.long_name,
            sector=excluded.sector,
            industry=excluded.industry,
            country=excluded.country,
            exchange=excluded.exchange,
            currency=excluded.currency,
            quote_type=excluded.quote_type,
            shares_outstanding=excluded.shares_outstanding,
            market_cap=excluded.market_cap,
            enterprise_value=excluded.enterprise_value,
            last_profile_refresh=excluded.last_profile_refresh,
            source=excluded.source
        """,
        (
            symbol,
            info.get("shortName") or info.get("longName") or symbol,
            _safe_text(info.get("shortName")),
            _safe_text(info.get("longName")),
            _safe_text(info.get("sector")),
            _safe_text(info.get("industry")),
            _safe_text(info.get("country")),
            _safe_text(info.get("exchange")),
            _safe_text(info.get("currency")),
            _safe_text(info.get("quoteType")),
            _safe_float(info.get("sharesOutstanding")),
            _safe_float(info.get("marketCap")),
            _safe_float(info.get("enterpriseValue")),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()


def store_price_history(symbol: str, history: pd.DataFrame, currency: Optional[str]) -> int:
    symbol = _normalize_symbol(symbol)
    frame = _prepare_history_frame(history)
    if frame.empty:
        return 0

    fetched_at = _utcnow()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            (
                symbol,
                row["trade_date"].strftime("%Y-%m-%d"),
                _safe_float(row.get("Open")),
                _safe_float(row.get("High")),
                _safe_float(row.get("Low")),
                _safe_float(row.get("Close")),
                _safe_float(row.get("Adj Close", row.get("Close"))),
                _safe_float(row.get("Volume")),
                _safe_float(row.get("Dividends")),
                _safe_float(row.get("Stock Splits")),
                currency,
                fetched_at,
            )
        )

    conn = _connect()
    conn.executemany(
        """
        INSERT OR REPLACE INTO market_price_history (
            symbol, trade_date, open, high, low, close, adj_close, volume,
            dividends, stock_splits, currency, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def store_fundamental_snapshot(symbol: str, info: dict, as_of_date: Optional[str] = None) -> None:
    symbol = _normalize_symbol(symbol)
    snapshot_date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")
    fetched_at = _utcnow()
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO market_fundamental_snapshots (
            symbol, as_of_date, name, sector, industry, currency, current_price,
            market_cap, enterprise_value, shares_outstanding, trailing_pe,
            forward_pe, peg_ratio, price_to_book, price_to_sales,
            enterprise_to_revenue, enterprise_to_ebitda, gross_margins,
            operating_margins, profit_margins, revenue_growth, earnings_growth,
            return_on_equity, debt_to_equity, total_revenue, gross_profits,
            ebitda, free_cashflow, operating_cashflow, trailing_eps,
            dividend_rate, dividend_yield, payout_ratio, beta,
            fifty_two_week_high, fifty_two_week_low, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            snapshot_date,
            info.get("shortName") or info.get("longName") or symbol,
            _safe_text(info.get("sector")),
            _safe_text(info.get("industry")),
            _safe_text(info.get("currency")),
            _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            _safe_float(info.get("marketCap")),
            _safe_float(info.get("enterpriseValue")),
            _safe_float(info.get("sharesOutstanding")),
            _safe_float(info.get("trailingPE")),
            _safe_float(info.get("forwardPE")),
            _safe_float(info.get("pegRatio")),
            _safe_float(info.get("priceToBook")),
            _safe_float(info.get("priceToSalesTrailing12Months")),
            _safe_float(info.get("enterpriseToRevenue")),
            _safe_float(info.get("enterpriseToEbitda")),
            _safe_float(info.get("grossMargins")),
            _safe_float(info.get("operatingMargins")),
            _safe_float(info.get("profitMargins")),
            _safe_float(info.get("revenueGrowth")),
            _safe_float(info.get("earningsGrowth")),
            _safe_float(info.get("returnOnEquity")),
            _safe_float(info.get("debtToEquity")),
            _safe_float(info.get("totalRevenue")),
            _safe_float(info.get("grossProfits")),
            _safe_float(info.get("ebitda")),
            _safe_float(info.get("freeCashflow")),
            _safe_float(info.get("operatingCashflow")),
            _safe_float(info.get("trailingEps")),
            _safe_float(info.get("dividendRate")),
            _safe_float(info.get("dividendYield")),
            _safe_float(info.get("payoutRatio")),
            _safe_float(info.get("beta")),
            _safe_float(info.get("fiftyTwoWeekHigh")),
            _safe_float(info.get("fiftyTwoWeekLow")),
            fetched_at,
        ),
    )
    conn.commit()
    conn.close()


def store_daily_feature_series(symbol: str, feature_frame: pd.DataFrame) -> int:
    symbol = _normalize_symbol(symbol)
    if feature_frame.empty:
        return 0

    rows = []
    for row in feature_frame.itertuples(index=False):
        rows.append(
            (
                symbol,
                row.feature_date,
                _safe_text(row.name),
                _safe_text(row.sector),
                _safe_float(row.price),
                _safe_text(row.currency),
                _safe_float(row.fx_rate_to_usd),
                _safe_int(row.values_in_usd),
                _safe_float(row.market_cap_b),
                _safe_int(row.score),
                _safe_text(row.signal),
                _safe_text(row.trend),
                _safe_float(row.rsi),
                _safe_float(row.volume_mean_usd),
                _safe_text(row.domaine),
                _safe_text(row.cap_range),
                _safe_text(row.param_key),
                _safe_float(row.score_over_threshold),
                _safe_text(row.consensus),
                _safe_float(row.consensus_mean),
                _safe_float(row.seuil_achat),
                _safe_float(row.seuil_vente),
                _safe_text(row.selected_param_key),
                _safe_float(row.dprice),
                _safe_float(row.var5j_pct),
                _safe_float(row.drsi),
                _safe_float(row.dvolrel),
                _safe_float(row.rev_growth_pct),
                _safe_float(row.ebitda_yield_pct),
                _safe_float(row.fcf_yield_pct),
                _safe_float(row.de_ratio),
                _safe_float(row.market_cap_b_usd),
                _safe_float(row.roe_pct),
                _safe_float(row.fiabilite),
                _safe_int(row.nb_trades),
                _safe_int(row.gagnants),
                _safe_float(row.gain_total_usd),
                _safe_float(row.gain_moyen_usd),
                _safe_text(row.detection_time),
                _safe_float(row.c1_revenue_growth_pct),
                _safe_int(row.c1_ok),
                _safe_float(row.c2_gross_margin_pct),
                _safe_int(row.c2_ok),
                _safe_int(row.c3_undervaluation_hits),
                _safe_int(row.c3_pe_ok),
                _safe_int(row.c3_peg_ok),
                _safe_int(row.c3_below_75pct_52wh),
                _safe_int(row.c3_ok),
                _safe_float(row.c4_momentum_3m_pct),
                _safe_float(row.c4_momentum_6m_pct),
                _safe_int(row.c4_above_sma50),
                _safe_int(row.c4_ok),
                _safe_float(row.c5_volume_ratio),
                _safe_int(row.c5_ok),
                _safe_float(row.close_vs_52wh_ratio),
                _safe_float(row.sma50),
                _safe_float(row.avg_volume_30d),
                _safe_float(row.avg_volume_90d),
                _safe_float(row.trailing_pe),
                _safe_float(row.peg_ratio),
                _safe_float(row.ttm_eps),
                _safe_float(row.eps_growth_pct),
                _safe_float(row.dividend_yield),
                _safe_float(row.payout_ratio),
                _safe_float(row.margin_stability_4q),
                _safe_float(row.dividend_stability_1y),
                _safe_int(row.event_count),
                _safe_text(row.event_flags),
                _safe_text(row.source_publication_date),
                _safe_text(row.computed_at),
            )
        )

    conn = _connect()
    conn.executemany(
        """
        INSERT OR REPLACE INTO market_feature_series (
            symbol, feature_date, name, sector, price, currency, fx_rate_to_usd,
            values_in_usd, market_cap_b, score, signal, trend, rsi, volume_mean_usd,
            domaine, cap_range, param_key, score_over_threshold, consensus,
            consensus_mean, seuil_achat, seuil_vente, selected_param_key, dprice,
            var5j_pct, drsi, dvolrel, rev_growth_pct, ebitda_yield_pct,
            fcf_yield_pct, de_ratio, market_cap_b_usd, roe_pct, fiabilite,
            nb_trades, gagnants, gain_total_usd, gain_moyen_usd, detection_time,
            c1_revenue_growth_pct, c1_ok, c2_gross_margin_pct, c2_ok,
            c3_undervaluation_hits, c3_pe_ok, c3_peg_ok, c3_below_75pct_52wh,
            c3_ok, c4_momentum_3m_pct, c4_momentum_6m_pct, c4_above_sma50,
            c4_ok, c5_volume_ratio, c5_ok, close_vs_52wh_ratio, sma50,
            avg_volume_30d, avg_volume_90d, trailing_pe, peg_ratio, ttm_eps,
            eps_growth_pct, dividend_yield, payout_ratio, margin_stability_4q,
            dividend_stability_1y, event_count, event_flags,
            source_publication_date, computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def fetch_and_store_symbol_series(symbol: str, start_date: str = DEFAULT_FEATURE_START_DATE) -> dict:
    ensure_market_data_schema()
    symbol = _normalize_symbol(symbol)
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    history = ticker.history(start=start_date, auto_adjust=False, actions=True)
    if history is None or history.empty:
        raise ValueError(f"No history available for {symbol}")

    get_fundamental_metrics(symbol, use_cache=False)
    upsert_instrument(symbol, info)
    store_fundamental_snapshot(symbol, info)

    currency = _safe_text(info.get("currency"))
    price_rows = store_price_history(symbol, history, currency)
    feature_frame = _build_daily_feature_frame(symbol, history, info)
    feature_rows = store_daily_feature_series(symbol, feature_frame)

    return {
        "symbol": symbol,
        "name": info.get("shortName") or info.get("longName") or symbol,
        "price_rows": price_rows,
        "feature_rows": feature_rows,
        "start_date": start_date,
        "end_date": feature_frame["feature_date"].iloc[-1] if not feature_frame.empty else None,
    }


def _register_run(symbols: Iterable[str], start_date: str) -> int:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO market_ingestion_runs (started_at, symbols, start_date, status)
        VALUES (?, ?, ?, 'running')
        """,
        (_utcnow(), ",".join(symbols), start_date),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return int(run_id or 0)


def _finish_run(run_id: int, status: str, details: str) -> None:
    conn = _connect()
    conn.execute(
        """
        UPDATE market_ingestion_runs
        SET finished_at = ?, status = ?, details = ?
        WHERE run_id = ?
        """,
        (_utcnow(), status, details, run_id),
    )
    conn.commit()
    conn.close()


def bootstrap_market_database(
    symbols: Iterable[str] = DEFAULT_BOOTSTRAP_SYMBOLS,
    start_date: str = DEFAULT_FEATURE_START_DATE,
) -> list[dict]:
    ensure_market_data_schema()
    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols]
    run_id = _register_run(normalized_symbols, start_date)
    results = []
    failures = []

    for symbol in normalized_symbols:
        try:
            results.append(fetch_and_store_symbol_series(symbol, start_date=start_date))
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")

    status = "completed" if not failures else "partial"
    details = "; ".join(failures) if failures else f"Stored {len(results)} symbol(s)"
    _finish_run(run_id, status, details)

    if failures:
        raise RuntimeError(details)
    return results


def get_symbol_storage_summary(symbol: str) -> dict:
    symbol = _normalize_symbol(symbol)
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM market_price_history WHERE symbol = ?", (symbol,))
    price_rows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM market_feature_series WHERE symbol = ?", (symbol,))
    feature_rows = cursor.fetchone()[0]
    cursor.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM market_price_history WHERE symbol = ?",
        (symbol,),
    )
    min_date, max_date = cursor.fetchone()
    conn.close()
    return {
        "symbol": symbol,
        "price_rows": price_rows,
        "feature_rows": feature_rows,
        "start_date": min_date,
        "end_date": max_date,
    }


def _parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _get_instrument_profile(symbol: str) -> dict:
    symbol = _normalize_symbol(symbol)
    conn = _connect()
    row = conn.execute(
        """
        SELECT symbol, name, short_name, long_name, sector, industry, country,
               exchange, currency, quote_type, shares_outstanding, market_cap,
               enterprise_value, last_profile_refresh
        FROM market_instruments
        WHERE symbol = ?
        """,
        (symbol,),
    ).fetchone()
    conn.close()
    if row is None:
        return {}

    out = dict(row)
    # Alignement sur les clés attendues côté yfinance-like
    out["shortName"] = out.get("short_name") or out.get("name")
    out["longName"] = out.get("long_name") or out.get("name")
    out["sharesOutstanding"] = out.get("shares_outstanding")
    out["marketCap"] = out.get("market_cap")
    out["enterpriseValue"] = out.get("enterprise_value")
    out["quoteType"] = out.get("quote_type")
    return out


def get_latest_feature_row(symbol: str) -> Optional[dict]:
    """Retourne la dernière ligne quotidienne calculée pour un symbole."""
    symbol = _normalize_symbol(symbol)
    conn = _connect()
    row = conn.execute(
        """
        SELECT f.*, i.name AS instrument_name, i.short_name, i.long_name,
               i.sector AS instrument_sector, i.market_cap AS instrument_market_cap,
               i.currency AS instrument_currency
        FROM market_feature_series f
        LEFT JOIN market_instruments i ON i.symbol = f.symbol
        WHERE f.symbol = ?
        ORDER BY f.feature_date DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def get_symbol_last_trade_date(symbol: str) -> Optional[str]:
    symbol = _normalize_symbol(symbol)
    conn = _connect()
    row = conn.execute(
        "SELECT MAX(trade_date) AS last_trade_date FROM market_price_history WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return row["last_trade_date"]


def _get_last_profile_refresh(symbol: str) -> Optional[datetime]:
    symbol = _normalize_symbol(symbol)
    conn = _connect()
    row = conn.execute(
        "SELECT last_profile_refresh FROM market_instruments WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    value = row["last_profile_refresh"]
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def refresh_symbol_incremental(
    symbol: str,
    min_refresh_hours: int = 20,
    recalc_window_days: int = 420,
    bootstrap_start_date: str = DEFAULT_FEATURE_START_DATE,
    force: bool = False,
) -> dict:
    """
    Met à jour un symbole avec une stratégie incrémentale quotidienne.

    - Evite les téléchargements massifs récurrents.
    - Recalcule une fenêtre glissante suffisante pour les indicateurs techniques.
    - Upsert les données dans les tables normalisées.
    """
    ensure_market_data_schema()
    symbol = _normalize_symbol(symbol)

    latest_row = get_latest_feature_row(symbol)
    last_trade_date = get_symbol_last_trade_date(symbol)
    now = datetime.utcnow()

    if not force and latest_row is not None:
        last_refresh = _get_last_profile_refresh(symbol)
        if last_refresh is not None:
            age_hours = (now - last_refresh).total_seconds() / 3600.0
            if age_hours < float(min_refresh_hours):
                return {
                    "symbol": symbol,
                    "status": "skipped_recent_refresh",
                    "price_rows": 0,
                    "feature_rows": 0,
                    "last_trade_date": last_trade_date,
                }

    start_date = bootstrap_start_date
    parsed_last = _parse_iso_date(last_trade_date)
    parsed_bootstrap = _parse_iso_date(bootstrap_start_date)
    if parsed_last is not None:
        start_candidate = parsed_last - timedelta(days=max(90, recalc_window_days))
        if parsed_bootstrap is None:
            start_date = start_candidate.strftime("%Y-%m-%d")
        else:
            start_date = max(start_candidate, parsed_bootstrap).strftime("%Y-%m-%d")

    ticker = yf.Ticker(symbol)
    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    if not info:
        info = _get_instrument_profile(symbol)

    history = ticker.history(start=start_date, auto_adjust=False, actions=True)
    if history is None or history.empty:
        if latest_row is not None:
            return {
                "symbol": symbol,
                "status": "no_new_history_keep_existing",
                "price_rows": 0,
                "feature_rows": 0,
                "last_trade_date": last_trade_date,
            }
        raise ValueError(f"No history available for {symbol}")

    # Favorise le cache fondamental local pour limiter la pression API.
    try:
        get_fundamental_metrics(symbol, use_cache=True, allow_stale=False)
    except Exception:
        # Ne bloque pas l'ingestion quotidienne si les fondamentaux échouent ponctuellement.
        pass

    upsert_instrument(symbol, info)
    if info:
        store_fundamental_snapshot(symbol, info)

    currency = _safe_text(info.get("currency"))
    price_rows = store_price_history(symbol, history, currency)
    feature_frame = _build_daily_feature_frame(symbol, history, info)
    feature_rows = store_daily_feature_series(symbol, feature_frame)

    refreshed_last_date = None
    if not feature_frame.empty:
        refreshed_last_date = feature_frame["feature_date"].iloc[-1]

    return {
        "symbol": symbol,
        "status": "updated",
        "price_rows": price_rows,
        "feature_rows": feature_rows,
        "last_trade_date": refreshed_last_date,
    }


ensure_market_data_schema()

# ---------------------------------------------------------------------------
# Rebinding final: l'API publique doit utiliser market_store (Parquet + DuckDB)
# ---------------------------------------------------------------------------
import market_store as _market_store_backend  # noqa: E402

DEFAULT_FEATURE_START_DATE = _market_store_backend.DEFAULT_FEATURE_START_DATE
DEFAULT_BOOTSTRAP_SYMBOLS = _market_store_backend.DEFAULT_BOOTSTRAP_SYMBOLS
PARQUET_DIR = _market_store_backend.PARQUET_DIR
DEFAULT_FX_CURRENCIES = _market_store_backend.DEFAULT_FX_CURRENCIES

ensure_market_data_schema = _market_store_backend.ensure_market_data_schema
upsert_instrument = _market_store_backend.upsert_instrument
store_price_history = _market_store_backend.store_price_history
store_fundamental_snapshot = _market_store_backend.store_fundamental_snapshot
store_daily_feature_series = _market_store_backend.store_daily_feature_series
fetch_and_store_symbol_series = _market_store_backend.fetch_and_store_symbol_series
bootstrap_market_database = _market_store_backend.bootstrap_market_database
get_symbol_storage_summary = _market_store_backend.get_symbol_storage_summary
get_latest_feature_row = _market_store_backend.get_latest_feature_row
get_symbol_last_trade_date = _market_store_backend.get_symbol_last_trade_date
refresh_symbol_incremental = _market_store_backend.refresh_symbol_incremental
query_features = _market_store_backend.query_features
ensure_fx_rates_daily_history = _market_store_backend.ensure_fx_rates_daily_history

_build_daily_feature_frame = _market_store_backend._build_daily_feature_frame
_get_rate_to_usd = _market_store_backend._get_rate_to_usd
_normalize_symbol = _market_store_backend._normalize_symbol
_safe_float = _market_store_backend._safe_float
_safe_int = _market_store_backend._safe_int
_safe_text = _market_store_backend._safe_text
_compute_rsi = _market_store_backend._compute_rsi
_cap_range_from_market_cap_b = _market_store_backend._cap_range_from_market_cap_b
_safe_score_signal = _market_store_backend._safe_score_signal
_prepare_history_frame = _market_store_backend._prepare_history_frame
_build_quarter_feature_points = _market_store_backend._build_quarter_feature_points

# Crée les répertoires Parquet à l'import via la version market_store.
ensure_market_data_schema()