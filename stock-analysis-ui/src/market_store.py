"""
Parquet + DuckDB market data warehouse.

Architecture cible pour >1 000 stocks / 30 ans d'historique.

Arborescence sur disque
-----------------------
PARQUET_DIR/
  instruments/
    instruments.parquet          # profils statiques (1 ligne / stock)
  prices/
    symbol=AAPL/part0.parquet    # prix OHLCV (partitionné par symbole)
    symbol=MSFT/part0.parquet
    ...
  features/
    symbol=AAPL/part0.parquet    # features quotidiennes (partitionné par symbole)
    ...
  fundamentals/
    symbol=AAPL/part0.parquet    # snapshots fondamentaux (partitionné par symbole)
    ...

API publique identique à cache_db.py pour compatibilité descendante :
  ensure_market_data_schema()
  upsert_instrument(symbol, info)
  store_price_history(symbol, history, currency)
  store_fundamental_snapshot(symbol, info, as_of_date)
  store_daily_feature_series(symbol, feature_frame)
  fetch_and_store_symbol_series(symbol, start_date)
  bootstrap_market_database(symbols, start_date)
  get_symbol_storage_summary(symbol)
  get_latest_feature_row(symbol)
  get_symbol_last_trade_date(symbol)
  refresh_symbol_incremental(symbol, ...)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from config import DB_PATH
from fundamentals_cache import get_all_quarters_sorted, get_fundamental_metrics

# ---------------------------------------------------------------------------
# Constantes
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
DEFAULT_FX_CURRENCIES = (
    "USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "HKD", "INR",
)

# Répertoire racine du store Parquet : même dossier que stock_analysis.db
_SRC_DIR = Path(DB_PATH).parent
PARQUET_DIR = _SRC_DIR / "market_parquet"

# ---------------------------------------------------------------------------
# Helpers internes (identiques à cache_db.py pour cohérence)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
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


def _to_usd(value, fx_rate_to_usd: float):
    amount = _safe_float(value)
    if amount is None:
        return None
    return amount * fx_rate_to_usd

# ---------------------------------------------------------------------------
# Gestion de l'arborescence Parquet
# ---------------------------------------------------------------------------

def _symbol_dir(category: str, symbol: str) -> Path:
    """Retourne le dossier partitionné d'un symbole pour une catégorie donnée."""
    safe = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    return PARQUET_DIR / category / f"symbol={safe}"


def _parquet_path(category: str, symbol: str) -> Path:
    return _symbol_dir(category, symbol) / "part0.parquet"


def _instruments_path() -> Path:
    return PARQUET_DIR / "instruments" / "instruments.parquet"


def _fx_rates_daily_path() -> Path:
    return PARQUET_DIR / "fx_rates_daily" / "fx_rates_daily.parquet"


def ensure_market_data_schema() -> None:
    """Crée les répertoires Parquet si nécessaire (équivalent CREATE TABLE)."""
    for sub in ("instruments", "prices", "features", "fundamentals", "fx_rates_daily"):
        (PARQUET_DIR / sub).mkdir(parents=True, exist_ok=True)


def _fetch_fx_history_to_usd(currency: str, start_date: str, end_date: str) -> pd.DataFrame:
    cur = str(currency or "USD").strip().upper()
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        return pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])

    if cur == "USD":
        days = pd.date_range(start=start_ts, end=end_ts, freq="D")
        return pd.DataFrame({
            "trade_date": days.strftime("%Y-%m-%d"),
            "currency": "USD",
            "rate_to_usd": 1.0,
            "fetched_at": _utcnow(),
        })

    pair = f"{cur}USD=X"
    try:
        hist = yf.download(
            pair,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            return pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])

        close_series = None
        if isinstance(hist.columns, pd.MultiIndex):
            if ("Close", pair) in hist.columns:
                close_series = hist[("Close", pair)]
            elif "Close" in hist.columns.get_level_values(0):
                close_df = hist.xs("Close", axis=1, level=0)
                if isinstance(close_df, pd.DataFrame) and not close_df.empty:
                    close_series = close_df.iloc[:, 0]
        else:
            if "Close" in hist.columns:
                close_series = hist["Close"]

        if close_series is None:
            return pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])

        fx = pd.DataFrame({
            "trade_date": pd.to_datetime(close_series.index, errors="coerce").strftime("%Y-%m-%d"),
            "rate_to_usd": pd.to_numeric(close_series, errors="coerce"),
        })
        fx = fx.dropna(subset=["trade_date", "rate_to_usd"])
        fx = fx[fx["rate_to_usd"] > 0]
        if fx.empty:
            return pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])

        return pd.DataFrame({
            "trade_date": fx["trade_date"].astype(str),
            "currency": cur,
            "rate_to_usd": fx["rate_to_usd"].astype(float),
            "fetched_at": _utcnow(),
        })
    except Exception:
        return pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])


def ensure_fx_rates_daily_history(
    currencies: Iterable[str] = DEFAULT_FX_CURRENCIES,
    years: int = 5,
    min_refresh_hours: int = 20,
    force: bool = False,
) -> dict:
    """Assure un historique FX journalier vers USD pour les devises demandées."""
    ensure_market_data_schema()
    path = _fx_rates_daily_path()
    now = datetime.utcnow()
    requested = [str(c).strip().upper() for c in currencies if str(c).strip()]
    requested = list(dict.fromkeys(requested))
    if "USD" not in requested:
        requested.insert(0, "USD")

    existing = pd.DataFrame(columns=["trade_date", "currency", "rate_to_usd", "fetched_at"])
    if path.exists():
        existing = pd.read_parquet(path)
        if not force and not existing.empty and "fetched_at" in existing.columns:
            try:
                last_refresh = pd.to_datetime(existing["fetched_at"], errors="coerce").max()
                if pd.notna(last_refresh):
                    age_h = (now - last_refresh.to_pydatetime()).total_seconds() / 3600.0
                    if age_h < float(min_refresh_hours):
                        return {
                            "status": "skipped_recent_refresh",
                            "currencies": requested,
                            "rows_total": int(len(existing)),
                            "rows_added": 0,
                            "last_refresh": str(last_refresh),
                        }
            except Exception:
                pass

    start_date = (now - timedelta(days=max(365, int(years) * 365))).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    new_parts = []
    for cur in requested:
        cur_start = start_date
        if not existing.empty and "currency" in existing.columns and "trade_date" in existing.columns:
            cur_rows = existing[existing["currency"].astype(str).str.upper() == cur]
            if not cur_rows.empty:
                last_trade = pd.to_datetime(cur_rows["trade_date"], errors="coerce").max()
                if pd.notna(last_trade):
                    cur_start = max(
                        pd.to_datetime(start_date),
                        last_trade - timedelta(days=7),
                    ).strftime("%Y-%m-%d")

        part = _fetch_fx_history_to_usd(cur, cur_start, end_date)
        if not part.empty:
            new_parts.append(part)

    if new_parts:
        incoming = pd.concat(new_parts, ignore_index=True)
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    else:
        combined = existing

    if not combined.empty:
        combined["currency"] = combined["currency"].astype(str).str.upper()
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        combined["rate_to_usd"] = pd.to_numeric(combined["rate_to_usd"], errors="coerce")
        combined = combined.dropna(subset=["trade_date", "currency", "rate_to_usd"])
        combined = combined[combined["rate_to_usd"] > 0]
        combined = combined.sort_values(["currency", "trade_date", "fetched_at"])
        combined = combined.drop_duplicates(subset=["currency", "trade_date"], keep="last")

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))

    return {
        "status": "updated",
        "currencies": requested,
        "rows_total": int(len(combined)),
        "rows_added": int(sum(len(p) for p in new_parts)),
        "start_date": start_date,
        "end_date": end_date,
    }


def _get_fx_rates_daily_series(currency: str, dates: pd.Series) -> pd.Series:
    cur = str(currency or "USD").strip().upper()
    if dates is None or len(dates) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(dates, errors="coerce")
    if cur == "USD":
        return pd.Series(1.0, index=idx.index, dtype=float)

    path = _fx_rates_daily_path()
    if not path.exists():
        ensure_fx_rates_daily_history(currencies=DEFAULT_FX_CURRENCIES, years=5, min_refresh_hours=20, force=True)
    if not path.exists():
        return pd.Series(1.0, index=idx.index, dtype=float)

    fx = pd.read_parquet(path)
    if fx.empty:
        return pd.Series(1.0, index=idx.index, dtype=float)

    fx = fx[fx["currency"].astype(str).str.upper() == cur]
    if fx.empty:
        ensure_fx_rates_daily_history(currencies=[cur], years=5, min_refresh_hours=0, force=True)
        fx = pd.read_parquet(path)
        fx = fx[fx["currency"].astype(str).str.upper() == cur]
    if fx.empty:
        return pd.Series(1.0, index=idx.index, dtype=float)

    fx["trade_date"] = pd.to_datetime(fx["trade_date"], errors="coerce")
    fx["rate_to_usd"] = pd.to_numeric(fx["rate_to_usd"], errors="coerce")
    fx = fx.dropna(subset=["trade_date", "rate_to_usd"]).sort_values("trade_date")
    if fx.empty:
        return pd.Series(1.0, index=idx.index, dtype=float)

    merged = pd.merge_asof(
        pd.DataFrame({"trade_date": idx}),
        fx[["trade_date", "rate_to_usd"]].sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    )
    rates = pd.to_numeric(merged["rate_to_usd"], errors="coerce")
    rates = rates.ffill().fillna(1.0)
    rates.index = idx.index
    return rates.astype(float)


# ---------------------------------------------------------------------------
# DuckDB : connexion légère en mémoire pour les requêtes ad-hoc
# ---------------------------------------------------------------------------

def _duckdb_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Exécute une requête DuckDB sur les fichiers Parquet et retourne un DataFrame."""
    con = duckdb.connect(database=":memory:")
    try:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Instruments (profils statiques)
# ---------------------------------------------------------------------------

def upsert_instrument(symbol: str, info: dict) -> None:
    symbol = _normalize_symbol(symbol)
    ensure_market_data_schema()
    path = _instruments_path()
    now = _utcnow()
    currency = _safe_text(info.get("currency")) or "USD"
    fx_rate_to_usd = _get_rate_to_usd(currency)
    market_cap = _safe_float(info.get("marketCap"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))

    new_row = pd.DataFrame([{
        "symbol": symbol,
        "name": info.get("shortName") or info.get("longName") or symbol,
        "short_name": _safe_text(info.get("shortName")),
        "long_name": _safe_text(info.get("longName")),
        "sector": _safe_text(info.get("sector")),
        "industry": _safe_text(info.get("industry")),
        "country": _safe_text(info.get("country")),
        "exchange": _safe_text(info.get("exchange")),
        "currency": currency,
        "fx_rate_to_usd": fx_rate_to_usd,
        "values_in_usd": 1,
        "quote_type": _safe_text(info.get("quoteType")),
        "shares_outstanding": _safe_float(info.get("sharesOutstanding")),
        "market_cap": market_cap,
        "market_cap_usd": _to_usd(market_cap, fx_rate_to_usd),
        "enterprise_value": enterprise_value,
        "enterprise_value_usd": _to_usd(enterprise_value, fx_rate_to_usd),
        "first_seen_at": now,
        "last_profile_refresh": now,
        "source": "yfinance",
    }])

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["symbol"] != symbol]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))


def _get_instrument_profile(symbol: str) -> dict:
    symbol = _normalize_symbol(symbol)
    path = _instruments_path()
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    row = df[df["symbol"] == symbol]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    r["shortName"] = r.get("short_name") or r.get("name")
    r["longName"] = r.get("long_name") or r.get("name")
    r["sharesOutstanding"] = r.get("shares_outstanding")
    r["marketCap"] = r.get("market_cap")
    r["enterpriseValue"] = r.get("enterprise_value")
    r["quoteType"] = r.get("quote_type")
    return r


def _get_last_profile_refresh(symbol: str) -> Optional[datetime]:
    symbol = _normalize_symbol(symbol)
    path = _instruments_path()
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["symbol", "last_profile_refresh"])
    row = df[df["symbol"] == symbol]
    if row.empty:
        return None
    val = row.iloc[0]["last_profile_refresh"]
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val))
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Prix OHLCV
# ---------------------------------------------------------------------------

def _prepare_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy().reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True).dt.tz_localize(None)
    return frame.rename(columns={date_col: "trade_date"}).sort_values("trade_date").reset_index(drop=True)


def store_price_history(symbol: str, history: pd.DataFrame, currency: Optional[str]) -> int:
    symbol = _normalize_symbol(symbol)
    frame = _prepare_history_frame(history)
    if frame.empty:
        return 0

    ensure_market_data_schema()
    path = _parquet_path("prices", symbol)
    fetched_at = _utcnow()

    rows = pd.DataFrame({
        "symbol": symbol,
        "trade_date": frame["trade_date"].dt.strftime("%Y-%m-%d"),
        "open": frame.get("Open", pd.Series(dtype=float)).values,
        "high": frame.get("High", pd.Series(dtype=float)).values,
        "low": frame.get("Low", pd.Series(dtype=float)).values,
        "close": frame.get("Close", pd.Series(dtype=float)).values,
        "adj_close": frame.get("Adj Close", frame.get("Close", pd.Series(dtype=float))).values,
        "volume": frame.get("Volume", pd.Series(dtype=float)).values,
        "dividends": frame.get("Dividends", pd.Series(dtype=float)).values,
        "stock_splits": frame.get("Stock Splits", pd.Series(dtype=float)).values,
        "currency": currency,
        "fetched_at": fetched_at,
    })

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[~existing["trade_date"].isin(rows["trade_date"])]
        combined = pd.concat([existing, rows], ignore_index=True).sort_values("trade_date")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = rows

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
    return len(rows)


def get_symbol_last_trade_date(symbol: str) -> Optional[str]:
    symbol = _normalize_symbol(symbol)
    path = _parquet_path("prices", symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["trade_date"])
    if df.empty:
        return None
    return df["trade_date"].max()

# ---------------------------------------------------------------------------
# Snapshots fondamentaux
# ---------------------------------------------------------------------------

def store_fundamental_snapshot(symbol: str, info: dict, as_of_date: Optional[str] = None) -> None:
    symbol = _normalize_symbol(symbol)
    ensure_market_data_schema()
    path = _parquet_path("fundamentals", symbol)
    snapshot_date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")
    currency = _safe_text(info.get("currency")) or "USD"
    fx_rate_to_usd = _get_rate_to_usd(currency)

    current_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _safe_float(info.get("marketCap"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    total_revenue = _safe_float(info.get("totalRevenue"))
    gross_profits = _safe_float(info.get("grossProfits"))
    ebitda = _safe_float(info.get("ebitda"))
    free_cashflow = _safe_float(info.get("freeCashflow"))
    operating_cashflow = _safe_float(info.get("operatingCashflow"))
    trailing_eps = _safe_float(info.get("trailingEps"))
    dividend_rate = _safe_float(info.get("dividendRate"))
    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow"))

    new_row = pd.DataFrame([{
        "symbol": symbol,
        "as_of_date": snapshot_date,
        "name": info.get("shortName") or info.get("longName") or symbol,
        "sector": _safe_text(info.get("sector")),
        "industry": _safe_text(info.get("industry")),
        "currency": currency,
        "fx_rate_to_usd": fx_rate_to_usd,
        "values_in_usd": 1,
        "current_price": current_price,
        "current_price_usd": _to_usd(current_price, fx_rate_to_usd),
        "market_cap": market_cap,
        "market_cap_usd": _to_usd(market_cap, fx_rate_to_usd),
        "enterprise_value": enterprise_value,
        "enterprise_value_usd": _to_usd(enterprise_value, fx_rate_to_usd),
        "shares_outstanding": _safe_float(info.get("sharesOutstanding")),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "peg_ratio": _safe_float(info.get("pegRatio")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "price_to_sales": _safe_float(info.get("priceToSalesTrailing12Months")),
        "enterprise_to_revenue": _safe_float(info.get("enterpriseToRevenue")),
        "enterprise_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "gross_margins": _safe_float(info.get("grossMargins")),
        "operating_margins": _safe_float(info.get("operatingMargins")),
        "profit_margins": _safe_float(info.get("profitMargins")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "return_on_equity": _safe_float(info.get("returnOnEquity")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "total_revenue": total_revenue,
        "total_revenue_usd": _to_usd(total_revenue, fx_rate_to_usd),
        "gross_profits": gross_profits,
        "gross_profits_usd": _to_usd(gross_profits, fx_rate_to_usd),
        "ebitda": ebitda,
        "ebitda_usd": _to_usd(ebitda, fx_rate_to_usd),
        "free_cashflow": free_cashflow,
        "free_cashflow_usd": _to_usd(free_cashflow, fx_rate_to_usd),
        "operating_cashflow": operating_cashflow,
        "operating_cashflow_usd": _to_usd(operating_cashflow, fx_rate_to_usd),
        "trailing_eps": trailing_eps,
        "trailing_eps_usd": _to_usd(trailing_eps, fx_rate_to_usd),
        "dividend_rate": dividend_rate,
        "dividend_rate_usd": _to_usd(dividend_rate, fx_rate_to_usd),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "payout_ratio": _safe_float(info.get("payoutRatio")),
        "beta": _safe_float(info.get("beta")),
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_high_usd": _to_usd(fifty_two_week_high, fx_rate_to_usd),
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_low_usd": _to_usd(fifty_two_week_low, fx_rate_to_usd),
        "fetched_at": _utcnow(),
    }])

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["as_of_date"] != snapshot_date]
        combined = pd.concat([existing, new_row], ignore_index=True).sort_values("as_of_date")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_row

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))

# ---------------------------------------------------------------------------
# Feature series quotidiennes (cœur du store)
# ---------------------------------------------------------------------------

def _build_quarter_feature_points(symbol: str) -> pd.DataFrame:
    quarters = get_all_quarters_sorted(symbol)
    if not quarters:
        return pd.DataFrame(columns=[
            "source_publication_date", "c1_revenue_growth_pct",
            "c2_gross_margin_pct", "ttm_eps", "eps_growth_pct", "margin_stability_4q",
        ])
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

    return qdf[[
        "source_publication_date", "c1_revenue_growth_pct", "c2_gross_margin_pct",
        "ttm_eps", "eps_growth_pct", "margin_stability_4q",
    ]].sort_values("source_publication_date")


def _build_daily_feature_frame(symbol: str, history: pd.DataFrame, info: dict) -> pd.DataFrame:
    frame = _prepare_history_frame(history)
    if frame.empty:
        return frame

    currency = str(info.get("currency") or "USD").upper()
    fx_rates = _get_fx_rates_daily_series(currency, frame["trade_date"])
    fx_rate_to_usd = _safe_float(fx_rates.iloc[-1]) if not fx_rates.empty else _get_rate_to_usd(currency)
    if fx_rate_to_usd is None or fx_rate_to_usd <= 0:
        fx_rate_to_usd = _get_rate_to_usd(currency)
        fx_rates = pd.Series(fx_rate_to_usd, index=frame.index, dtype=float)

    frame["price"] = frame["Close"].astype(float) * fx_rates.astype(float)
    frame["currency"] = currency
    frame["fx_rate_to_usd"] = fx_rates.astype(float)
    frame["values_in_usd"] = 1
    frame["avg_volume_30d"] = frame["Volume"].rolling(30, min_periods=10).mean()
    frame["avg_volume_90d"] = frame["Volume"].rolling(90, min_periods=30).mean()
    frame["c5_volume_ratio"] = frame["avg_volume_30d"] / frame["avg_volume_90d"]
    frame["volume_mean_usd"] = frame["avg_volume_30d"] * frame["price"]
    frame["sma50"] = frame["price"].rolling(50, min_periods=20).mean()
    frame["c4_momentum_3m_pct"] = frame["price"].pct_change(63, fill_method=None) * 100.0
    frame["c4_momentum_6m_pct"] = frame["price"].pct_change(126, fill_method=None) * 100.0
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
        for col in ["source_publication_date", "c1_revenue_growth_pct", "c2_gross_margin_pct",
                    "ttm_eps", "eps_growth_pct", "margin_stability_4q"]:
            frame[col] = pd.NA

    # ttm_eps provient des états trimestriels (devise native). On garde la valeur locale
    # et on convertit en USD pour les calculs avec un prix déjà converti en USD.
    frame["ttm_eps_local"] = pd.to_numeric(frame["ttm_eps"], errors="coerce")
    frame["ttm_eps"] = frame["ttm_eps_local"] * fx_rate_to_usd

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
        _flag(frame["c1_ok"]) + _flag(frame["c2_ok"]) + _flag(frame["c3_ok"])
        + _flag(frame["c4_ok"]) + _flag(frame["c5_ok"])
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

    frame["dprice"] = frame["price"].pct_change(1, fill_method=None) * 100.0
    frame["var5j_pct"] = frame["price"].pct_change(5, fill_method=None) * 100.0
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
    ebitda_yield_pct = (ebitda_usd / denom_ebitda * 100.0) if (ebitda_usd and denom_ebitda and denom_ebitda > 0) else pd.NA
    fcf_yield_pct = (fcf_usd / market_cap_usd * 100.0) if (fcf_usd and market_cap_usd and market_cap_usd > 0) else pd.NA

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

    columns = [
        "feature_date", "name", "sector", "price", "currency", "fx_rate_to_usd",
        "values_in_usd", "market_cap_b", "score", "signal", "trend", "rsi",
        "volume_mean_usd", "domaine", "cap_range", "param_key", "score_over_threshold",
        "consensus", "consensus_mean", "seuil_achat", "seuil_vente", "selected_param_key",
        "dprice", "var5j_pct", "drsi", "dvolrel", "rev_growth_pct", "ebitda_yield_pct",
        "fcf_yield_pct", "de_ratio", "market_cap_b_usd", "roe_pct", "fiabilite",
        "nb_trades", "gagnants", "gain_total_usd", "gain_moyen_usd", "detection_time",
        "c1_revenue_growth_pct", "c1_ok", "c2_gross_margin_pct", "c2_ok",
        "c3_undervaluation_hits", "c3_pe_ok", "c3_peg_ok", "c3_below_75pct_52wh",
        "c3_ok", "c4_momentum_3m_pct", "c4_momentum_6m_pct", "c4_above_sma50",
        "c4_ok", "c5_volume_ratio", "c5_ok", "close_vs_52wh_ratio", "sma50",
        "avg_volume_30d", "avg_volume_90d", "trailing_pe", "peg_ratio", "ttm_eps",
        "ttm_eps_local",
        "eps_growth_pct", "dividend_yield", "payout_ratio", "margin_stability_4q",
        "dividend_stability_1y", "event_count", "event_flags", "source_publication_date",
        "computed_at",
    ]
    return frame[[c for c in columns if c in frame.columns]]


def store_daily_feature_series(symbol: str, feature_frame: pd.DataFrame) -> int:
    symbol = _normalize_symbol(symbol)
    if feature_frame.empty:
        return 0

    ensure_market_data_schema()
    path = _parquet_path("features", symbol)
    df = feature_frame.copy()
    df.insert(0, "symbol", symbol)

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[~existing["feature_date"].isin(df["feature_date"])]
        combined = pd.concat([existing, df], ignore_index=True).sort_values("feature_date")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = df.sort_values("feature_date")

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
    return len(df)


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def get_latest_feature_row(symbol: str) -> Optional[dict]:
    symbol = _normalize_symbol(symbol)
    path = _parquet_path("features", symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df.sort_values("feature_date").iloc[-1].to_dict()


def get_symbol_storage_summary(symbol: str) -> dict:
    symbol = _normalize_symbol(symbol)
    price_path = _parquet_path("prices", symbol)
    feature_path = _parquet_path("features", symbol)

    price_rows = 0
    min_date = max_date = None
    if price_path.exists():
        df_p = pd.read_parquet(price_path, columns=["trade_date"])
        price_rows = len(df_p)
        if not df_p.empty:
            min_date = df_p["trade_date"].min()
            max_date = df_p["trade_date"].max()

    feature_rows = 0
    if feature_path.exists():
        df_f = pd.read_parquet(feature_path, columns=["feature_date"])
        feature_rows = len(df_f)

    return {
        "symbol": symbol,
        "price_rows": price_rows,
        "feature_rows": feature_rows,
        "start_date": min_date,
        "end_date": max_date,
    }


def query_features(
    symbols: List[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Requête multi-symboles sur les features via DuckDB.
    Plus rapide que pandas pour les scans larges (>100 stocks).

    Exemple :
        df = query_features(["AAPL", "MSFT"], start_date="2026-04-18")
    """
    ensure_market_data_schema()
    pattern = str(PARQUET_DIR / "features" / "*/part0.parquet")
    col_clause = "*" if not columns else ", ".join(columns)

    filters = []
    if symbols:
        quoted = ", ".join(f"'{s}'" for s in [_normalize_symbol(s) for s in symbols])
        filters.append(f"symbol IN ({quoted})")
    if start_date:
        filters.append(f"feature_date >= '{start_date}'")
    if end_date:
        filters.append(f"feature_date <= '{end_date}'")

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT {col_clause} FROM read_parquet('{pattern}', hive_partitioning=false) {where} ORDER BY feature_date DESC, symbol ASC"

    try:
        return _duckdb_query(sql)
    except Exception:
        # Fallback si aucun fichier n'existe encore
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pipeline d'ingestion complet
# ---------------------------------------------------------------------------

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


def _parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def refresh_symbol_incremental(
    symbol: str,
    min_refresh_hours: int = 20,
    recalc_window_days: int = 420,
    bootstrap_start_date: str = DEFAULT_FEATURE_START_DATE,
    force: bool = False,
) -> dict:
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

    try:
        get_fundamental_metrics(symbol, use_cache=True, allow_stale=False)
    except Exception:
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


def bootstrap_market_database(
    symbols: Iterable[str] = DEFAULT_BOOTSTRAP_SYMBOLS,
    start_date: str = DEFAULT_FEATURE_START_DATE,
) -> list[dict]:
    ensure_market_data_schema()
    normalized = [_normalize_symbol(s) for s in symbols]
    results = []
    failures = []

    for sym in normalized:
        try:
            results.append(fetch_and_store_symbol_series(sym, start_date=start_date))
        except Exception as exc:
            failures.append(f"{sym}: {exc}")

    if failures:
        raise RuntimeError("; ".join(failures))
    return results


# ---------------------------------------------------------------------------
# Initialisation automatique à l'import
# ---------------------------------------------------------------------------
ensure_market_data_schema()
