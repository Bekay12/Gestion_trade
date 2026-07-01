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
import threading
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

# Verrous fichiers partagés (instruments.parquet écrit par plusieurs threads)
_INSTRUMENTS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers internes (identiques à cache_db.py pour cohérence)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _safe_read_parquet(path, **kwargs) -> pd.DataFrame:
    """Lit un fichier Parquet ; si le fichier est corrompu, le supprime et retourne un DataFrame vide."""
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:
        print(f"[market_store] fichier Parquet corrompu détecté, suppression : {path} ({exc})")
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        return pd.DataFrame()


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
        existing = _safe_read_parquet(path)
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

    fx = _safe_read_parquet(path)
    if fx.empty:
        return pd.Series(1.0, index=idx.index, dtype=float)

    fx = fx[fx["currency"].astype(str).str.upper() == cur]
    if fx.empty:
        ensure_fx_rates_daily_history(currencies=[cur], years=5, min_refresh_hours=0, force=True)
        fx = _safe_read_parquet(path)
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

    with _INSTRUMENTS_LOCK:
        if path.exists():
            existing = _safe_read_parquet(path)
            existing = existing[existing["symbol"] != symbol] if not existing.empty else pd.DataFrame(columns=new_row.columns)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))


def _get_instrument_profile(symbol: str) -> dict:
    symbol = _normalize_symbol(symbol)
    path = _instruments_path()
    if not path.exists():
        return {}
    df = _safe_read_parquet(path)
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
    df = _safe_read_parquet(path, columns=["symbol", "last_profile_refresh"])
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
        existing = _safe_read_parquet(path)
        existing = existing[~existing["trade_date"].isin(rows["trade_date"])] if not existing.empty else pd.DataFrame(columns=rows.columns)
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
    df = _safe_read_parquet(path, columns=["trade_date"])
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
        existing = _safe_read_parquet(path)
        existing = existing[existing["as_of_date"] != snapshot_date] if not existing.empty else pd.DataFrame(columns=new_row.columns)
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


# ---------------------------------------------------------------------------
# Calendrier entreprise : snapshot depuis yfinance
# ---------------------------------------------------------------------------

def _fetch_calendar_snapshot(ticker: "yf.Ticker") -> dict:
    """Télécharge toutes les données calendrier/analyste disponibles sur un ticker yfinance.

    Retourne un dict plat utilisable comme scalar dans _build_daily_feature_frame.
    Tous les champs sont protégés par try/except pour rester robuste.
    """
    cal: dict = {}

    # ── 1. calendar (prochaine date earnings + estimations consensus) ─────
    try:
        raw = ticker.calendar
        if raw is not None:
            if isinstance(raw, dict):
                cal_dict = raw
            elif isinstance(raw, pd.DataFrame):
                cal_dict = raw.to_dict()
            else:
                cal_dict = {}

            # Earnings Date (peut être une liste ou une valeur unique)
            ed = cal_dict.get("Earnings Date")
            if ed is not None:
                if isinstance(ed, (list, tuple)) and len(ed) > 0:
                    ed = ed[0]
                elif isinstance(ed, dict):
                    ed = list(ed.values())[0] if ed else None
                try:
                    ed_dt = pd.to_datetime(ed, utc=True).tz_localize(None)
                    cal["next_earnings_date"] = ed_dt.strftime("%Y-%m-%d")
                except Exception:
                    cal["next_earnings_date"] = None
            else:
                cal["next_earnings_date"] = None

            cal["earnings_eps_estimate_avg"]  = _safe_float(cal_dict.get("Earnings Average"))
            cal["earnings_eps_estimate_low"]  = _safe_float(cal_dict.get("Earnings Low"))
            cal["earnings_eps_estimate_high"] = _safe_float(cal_dict.get("Earnings High"))
            cal["earnings_revenue_estimate_avg"]  = _safe_float(cal_dict.get("Revenue Average"))
            cal["earnings_revenue_estimate_low"]  = _safe_float(cal_dict.get("Revenue Low"))
            cal["earnings_revenue_estimate_high"] = _safe_float(cal_dict.get("Revenue High"))

            # Dividend Date / Ex-Dividend Date
            for src_key, dst_key in [("Dividend Date", "next_dividend_date"),
                                      ("Ex-Dividend Date", "next_ex_dividend_date")]:
                raw_d = cal_dict.get(src_key)
                if raw_d is not None:
                    if isinstance(raw_d, dict):
                        raw_d = list(raw_d.values())[0] if raw_d else None
                    try:
                        cal[dst_key] = pd.to_datetime(raw_d, utc=True).tz_localize(None).strftime("%Y-%m-%d")
                    except Exception:
                        cal[dst_key] = None
                else:
                    cal[dst_key] = None
    except Exception:
        cal.setdefault("next_earnings_date", None)
        cal.setdefault("earnings_eps_estimate_avg", None)
        cal.setdefault("earnings_eps_estimate_low", None)
        cal.setdefault("earnings_eps_estimate_high", None)
        cal.setdefault("earnings_revenue_estimate_avg", None)
        cal.setdefault("earnings_revenue_estimate_low", None)
        cal.setdefault("earnings_revenue_estimate_high", None)
        cal.setdefault("next_dividend_date", None)
        cal.setdefault("next_ex_dividend_date", None)

    # ── 2. analyst_price_targets ─────────────────────────────────────────
    try:
        apt = ticker.analyst_price_targets
        if isinstance(apt, dict):
            cal["target_mean_price"]   = _safe_float(apt.get("mean") or apt.get("targetMeanPrice"))
            cal["target_high_price"]   = _safe_float(apt.get("high") or apt.get("targetHighPrice"))
            cal["target_low_price"]    = _safe_float(apt.get("low")  or apt.get("targetLowPrice"))
            cal["target_median_price"] = _safe_float(apt.get("median") or apt.get("targetMedianPrice"))
        else:
            raise ValueError("not a dict")
    except Exception:
        info_ref = {}
        try:
            info_ref = ticker.info or {}
        except Exception:
            pass
        cal["target_mean_price"]   = _safe_float(info_ref.get("targetMeanPrice"))
        cal["target_high_price"]   = _safe_float(info_ref.get("targetHighPrice"))
        cal["target_low_price"]    = _safe_float(info_ref.get("targetLowPrice"))
        cal["target_median_price"] = _safe_float(info_ref.get("targetMedianPrice"))

    # ── 3. recommendations_summary ───────────────────────────────────────
    try:
        rs = ticker.recommendations_summary
        if rs is not None and not rs.empty:
            row0 = rs.iloc[0]
            cal["rec_strong_buy"]  = int(row0.get("strongBuy",  0) or 0)
            cal["rec_buy"]         = int(row0.get("buy",         0) or 0)
            cal["rec_hold"]        = int(row0.get("hold",        0) or 0)
            cal["rec_sell"]        = int(row0.get("sell",        0) or 0)
            cal["rec_strong_sell"] = int(row0.get("strongSell", 0) or 0)
        else:
            raise ValueError("empty")
    except Exception:
        for k in ("rec_strong_buy", "rec_buy", "rec_hold", "rec_sell", "rec_strong_sell"):
            cal.setdefault(k, None)

    # ── 4. last split info (depuis info) ─────────────────────────────────
    try:
        info_s = ticker.info or {}
        raw_split_date = info_s.get("lastSplitDate")
        if raw_split_date:
            cal["last_split_date"] = pd.to_datetime(raw_split_date, unit="s", utc=True).tz_localize(None).strftime("%Y-%m-%d")
        else:
            cal["last_split_date"] = None
        cal["last_split_factor"] = _safe_text(info_s.get("lastSplitFactor"))
    except Exception:
        cal.setdefault("last_split_date", None)
        cal.setdefault("last_split_factor", None)

    return cal


# ---------------------------------------------------------------------------
# Feature frame quotidien — toutes features
# ---------------------------------------------------------------------------

def _build_daily_feature_frame(
    symbol: str,
    history: pd.DataFrame,
    info: dict,
    calendar_data: Optional[dict] = None,
) -> pd.DataFrame:
    """Construit la série de features journalières pour un symbole.

    Paramètres
    ----------
    symbol        : ticker normalisé
    history       : OHLCV + Dividends + Stock Splits depuis yfinance
    info          : dict yfinance.Ticker.info
    calendar_data : dict retourné par _fetch_calendar_snapshot() (optionnel)
    """
    frame = _prepare_history_frame(history)
    if frame.empty:
        return frame

    cal = calendar_data or {}

    # ── Prix & FX ────────────────────────────────────────────────────────
    currency = str(info.get("currency") or "USD").upper()
    fx_rates = _get_fx_rates_daily_series(currency, frame["trade_date"])
    fx_rate_to_usd = _safe_float(fx_rates.iloc[-1]) if not fx_rates.empty else _get_rate_to_usd(currency)
    if fx_rate_to_usd is None or fx_rate_to_usd <= 0:
        fx_rate_to_usd = _get_rate_to_usd(currency)
        fx_rates = pd.Series(fx_rate_to_usd, index=frame.index, dtype=float)

    price = frame["Close"].astype(float) * fx_rates.astype(float)
    frame["price"]         = price
    frame["currency"]      = currency
    frame["fx_rate_to_usd"] = fx_rates.astype(float)
    frame["values_in_usd"] = 1

    # ── Volume ───────────────────────────────────────────────────────────
    vol = frame["Volume"].astype(float)
    frame["avg_volume_5d"]      = vol.rolling(5,  min_periods=2).mean()
    frame["avg_volume_30d"]     = vol.rolling(30, min_periods=10).mean()
    frame["avg_volume_90d"]     = vol.rolling(90, min_periods=30).mean()
    frame["c5_volume_ratio"]    = frame["avg_volume_30d"] / frame["avg_volume_90d"]
    frame["volume_mean_usd"]    = frame["avg_volume_30d"] * price
    frame["volume_spike_ratio"] = vol / frame["avg_volume_30d"]

    # ── Trend / Moving averages ──────────────────────────────────────────
    frame["sma20"]  = price.rolling(20,  min_periods=10).mean()
    frame["sma50"]  = price.rolling(50,  min_periods=20).mean()
    frame["sma200"] = price.rolling(200, min_periods=60).mean()
    frame["ema12"]  = price.ewm(span=12, adjust=False).mean()
    frame["ema26"]  = price.ewm(span=26, adjust=False).mean()
    frame["macd"]        = frame["ema12"] - frame["ema26"]
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"]   = frame["macd"] - frame["macd_signal"]

    # ── Momentum ─────────────────────────────────────────────────────────
    frame["momentum_1m_pct"]    = price.pct_change(21,  fill_method=None) * 100.0
    frame["c4_momentum_3m_pct"] = price.pct_change(63,  fill_method=None) * 100.0
    frame["c4_momentum_6m_pct"] = price.pct_change(126, fill_method=None) * 100.0
    frame["momentum_12m_pct"]   = price.pct_change(252, fill_method=None) * 100.0
    frame["dprice"]    = price.pct_change(1, fill_method=None) * 100.0
    frame["var5j_pct"] = price.pct_change(5, fill_method=None) * 100.0

    # ── 52-semaine ───────────────────────────────────────────────────────
    frame["rolling_52w_high"]    = price.rolling(252, min_periods=20).max()
    frame["rolling_52w_low"]     = price.rolling(252, min_periods=20).min()
    frame["close_vs_52wh_ratio"] = price / frame["rolling_52w_high"]
    _range_52w = (frame["rolling_52w_high"] - frame["rolling_52w_low"]).replace(0, pd.NA)
    frame["high_low_52w_pct"]    = (price - frame["rolling_52w_low"]) / _range_52w * 100.0
    frame["price_vs_sma200_pct"] = (price - frame["sma200"]) / frame["sma200"].replace(0, pd.NA) * 100.0

    # ── Volatilité ───────────────────────────────────────────────────────
    log_ret = price.apply(lambda x: x).pct_change(fill_method=None)
    frame["realized_vol_21d"] = log_ret.rolling(21, min_periods=10).std() * (252 ** 0.5) * 100.0
    frame["realized_vol_63d"] = log_ret.rolling(63, min_periods=21).std() * (252 ** 0.5) * 100.0

    # ATR 14 (en devise locale puis converti)
    if "High" in frame.columns and "Low" in frame.columns:
        hi = frame["High"].astype(float) * fx_rates.astype(float)
        lo = frame["Low"].astype(float)  * fx_rates.astype(float)
        prev_close = price.shift(1)
        tr = pd.concat([
            hi - lo,
            (hi - prev_close).abs(),
            (lo - prev_close).abs(),
        ], axis=1).max(axis=1)
        frame["atr_14"] = tr.ewm(span=14, adjust=False).mean()
    else:
        frame["atr_14"] = pd.NA

    # Bollinger Bands (SMA20 ± 2σ)
    bb_std = price.rolling(20, min_periods=10).std()
    frame["bollinger_upper"] = frame["sma20"] + 2.0 * bb_std
    frame["bollinger_lower"] = frame["sma20"] - 2.0 * bb_std
    bb_range = (frame["bollinger_upper"] - frame["bollinger_lower"]).replace(0, pd.NA)
    frame["bollinger_pct_b"]  = (price - frame["bollinger_lower"]) / bb_range

    # ── RSI ──────────────────────────────────────────────────────────────
    frame["rsi"]   = _compute_rsi(price)
    frame["drsi"]  = frame["rsi"].diff(1)
    frame["dvolrel"] = (frame["c5_volume_ratio"] - 1.0) * 100.0

    # ── Dividendes ───────────────────────────────────────────────────────
    if "Dividends" in frame.columns:
        trailing_dividend = frame["Dividends"].rolling(252, min_periods=20).sum()
        frame["dividend_stability_1y"] = trailing_dividend.rolling(126, min_periods=20).std()
    else:
        frame["dividend_stability_1y"] = pd.NA

    # ── Données trimestrielles (via fundamentals_cache) ──────────────────
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

    frame["ttm_eps_local"] = pd.to_numeric(frame["ttm_eps"], errors="coerce")
    frame["ttm_eps"]       = frame["ttm_eps_local"] * fx_rate_to_usd

    frame["trailing_pe"] = price / frame["ttm_eps"]
    frame.loc[frame["ttm_eps"].isna() | (frame["ttm_eps"] <= 0), "trailing_pe"] = pd.NA
    frame["peg_ratio"] = frame["trailing_pe"] / frame["eps_growth_pct"]
    frame.loc[frame["eps_growth_pct"].isna() | (frame["eps_growth_pct"] <= 0), "peg_ratio"] = pd.NA

    # ── Critères de sélection (C1→C5) ────────────────────────────────────
    frame["c1_ok"] = frame["c1_revenue_growth_pct"] > 20.0
    frame["c2_ok"] = frame["c2_gross_margin_pct"] > 30.0
    frame["c3_pe_ok"]             = frame["trailing_pe"].between(0.0, 25.0, inclusive="neither")
    frame["c3_peg_ok"]            = frame["peg_ratio"].between(0.0, 1.5, inclusive="neither")
    frame["c3_below_75pct_52wh"]  = frame["close_vs_52wh_ratio"] < 0.75
    frame["c3_undervaluation_hits"] = (
        _flag(frame["c3_pe_ok"])
        + _flag(frame["c3_peg_ok"])
        + _flag(frame["c3_below_75pct_52wh"])
    )
    frame["c3_ok"]         = frame["c3_undervaluation_hits"] >= 2
    frame["c4_above_sma50"] = price > frame["sma50"]
    frame["c4_ok"] = (
        frame["c4_momentum_3m_pct"].between(8.0, 60.0, inclusive="neither")
        & frame["c4_above_sma50"]
        & (frame["c4_momentum_6m_pct"] < 150.0)
    )
    frame["c5_ok"] = frame["c5_volume_ratio"] > 1.20

    # ── Capitalisation ────────────────────────────────────────────────────
    # Defragmenter le DataFrame après les rolling calculations et merge_asof
    # pour éviter les PerformanceWarnings lors des affectations de colonnes scalaires.
    frame = frame.copy()
    shares_outstanding = _safe_float(info.get("sharesOutstanding"))
    frame["market_cap_b"] = pd.NA
    if shares_outstanding and shares_outstanding > 0:
        frame["market_cap_b"] = (price * shares_outstanding) / 1_000_000_000.0

    # ── Info statiques (snapshot) ─────────────────────────────────────────
    name   = info.get("shortName") or info.get("longName") or symbol
    sector = info.get("sector") or "N/A"
    frame["name"]    = name
    frame["sector"]  = sector
    frame["domaine"] = sector

    # Dividende & payout
    frame["dividend_yield"] = _safe_float(info.get("dividendYield"))
    frame["payout_ratio"]   = _safe_float(info.get("payoutRatio"))

    # Valorisation
    market_cap_usd = _safe_float(info.get("marketCap"))
    if market_cap_usd:
        market_cap_usd = market_cap_usd * fx_rate_to_usd
    ev_usd    = _safe_float(info.get("enterpriseValue"))
    if ev_usd:
        ev_usd = ev_usd * fx_rate_to_usd
    ebitda_usd = _safe_float(info.get("ebitda"))
    if ebitda_usd:
        ebitda_usd = ebitda_usd * fx_rate_to_usd
    fcf_usd = _safe_float(info.get("freeCashflow"))
    if fcf_usd:
        fcf_usd = fcf_usd * fx_rate_to_usd

    denom_ebitda = ev_usd if ev_usd and ev_usd > 0 else market_cap_usd
    frame["ebitda_yield_pct"] = (ebitda_usd / denom_ebitda * 100.0) if (ebitda_usd and denom_ebitda and denom_ebitda > 0) else pd.NA
    frame["fcf_yield_pct"]    = (fcf_usd / market_cap_usd * 100.0)  if (fcf_usd and market_cap_usd and market_cap_usd > 0) else pd.NA
    frame["de_ratio"]         = _safe_float(info.get("debtToEquity"))
    roe = _safe_float(info.get("returnOnEquity"))
    frame["roe_pct"]          = (roe * 100.0) if roe is not None else pd.NA
    frame["market_cap_b_usd"] = (market_cap_usd / 1e9) if market_cap_usd else frame["market_cap_b"]

    # Nouvelles métriques fondamentales issues de info
    frame["forward_pe"]   = _safe_float(info.get("forwardPE"))
    frame["forward_eps"]  = _safe_float(info.get("forwardEps"))
    frame["book_value"]   = _safe_float(info.get("bookValue"))
    frame["current_ratio"] = _safe_float(info.get("currentRatio"))
    frame["quick_ratio"]   = _safe_float(info.get("quickRatio"))
    frame["short_ratio"]   = _safe_float(info.get("shortRatio"))

    spof = _safe_float(info.get("shortPercentOfFloat"))
    frame["short_percent_float"] = (spof * 100.0) if spof is not None else pd.NA

    hpi = _safe_float(info.get("heldPercentInsiders"))
    frame["held_pct_insiders"] = (hpi * 100.0) if hpi is not None else pd.NA

    hpinst = _safe_float(info.get("heldPercentInstitutions"))
    frame["held_pct_institutions"] = (hpinst * 100.0) if hpinst is not None else pd.NA

    eqg = _safe_float(info.get("earningsQuarterlyGrowth"))
    frame["earnings_quarterly_growth_pct"] = (eqg * 100.0) if eqg is not None else pd.NA

    rg = _safe_float(info.get("revenueGrowth"))
    frame["revenue_growth_info_pct"] = (rg * 100.0) if rg is not None else pd.NA

    frame["operating_margins_pct"] = (_safe_float(info.get("operatingMargins")) or 0.0) * 100.0 or pd.NA
    frame["ebitda_margins_pct"]    = (_safe_float(info.get("ebitdaMargins"))    or 0.0) * 100.0 or pd.NA
    frame["gross_margins_pct"]     = (_safe_float(info.get("grossMargins"))     or 0.0) * 100.0 or pd.NA
    frame["profit_margins_pct"]    = (_safe_float(info.get("profitMargins"))    or 0.0) * 100.0 or pd.NA

    roa = _safe_float(info.get("returnOnAssets"))
    frame["return_on_assets_pct"] = (roa * 100.0) if roa is not None else pd.NA
    frame["revenue_per_share"]    = _safe_float(info.get("revenuePerShare"))

    tc = _safe_float(info.get("totalCash"))
    td = _safe_float(info.get("totalDebt"))
    frame["total_cash_b"] = (tc * fx_rate_to_usd / 1e9) if tc else pd.NA
    frame["total_debt_b"] = (td * fx_rate_to_usd / 1e9) if td else pd.NA
    if tc is not None and td is not None:
        frame["net_cash_b"] = ((tc - td) * fx_rate_to_usd / 1e9)
    else:
        frame["net_cash_b"] = pd.NA

    wk52c = _safe_float(info.get("52WeekChange"))
    frame["week52_change_pct"] = (wk52c * 100.0) if wk52c is not None else pd.NA

    sp52c = _safe_float(info.get("SandP52WeekChange"))
    frame["sp500_52w_change_pct"] = (sp52c * 100.0) if sp52c is not None else pd.NA

    fs = _safe_float(info.get("floatShares"))
    frame["float_shares_m"] = (fs / 1e6) if fs else pd.NA

    # ── Critères « Sichere Unternehmen » (S1→S7), tous depuis info ─────────
    # Répliqués depuis Sichere_Unternehmen_scan.py pour un screening 0-requête.
    _EUR_USD = 1.08  # conversion approx. pour le seuil « cap > 10 Mrd € »
    _beta = _safe_float(info.get("beta"))
    frame["beta"] = _beta if _beta is not None else pd.NA

    _total_revenue = _safe_float(info.get("totalRevenue"))
    _fcf_raw = _safe_float(info.get("freeCashflow"))
    if _fcf_raw is None:
        _ocf = _safe_float(info.get("operatingCashflow"))
        _capex = _safe_float(info.get("capitalExpenditures"))
        _fcf_raw = (_ocf - abs(_capex)) if (_ocf is not None and _capex is not None) else None
    _fcf_margin = (_fcf_raw / _total_revenue * 100.0) if (_fcf_raw is not None and _total_revenue and _total_revenue > 0) else None
    frame["fcf_margin_pct"] = _fcf_margin if _fcf_margin is not None else pd.NA

    _earnings_growth = _safe_float(info.get("earningsGrowth"))
    frame["earnings_growth_pct"] = (_earnings_growth * 100.0) if _earnings_growth is not None else pd.NA

    _mc_b_usd_val = frame["market_cap_b_usd"].iloc[0]
    _mc_b_usd = float(_mc_b_usd_val) if not pd.isna(_mc_b_usd_val) else None
    _de  = _safe_float(info.get("debtToEquity"))
    _dy  = _safe_float(info.get("dividendYield")) or _safe_float(info.get("trailingAnnualDividendYield"))
    _eqg = _safe_float(info.get("earningsQuarterlyGrowth"))
    _rg  = _safe_float(info.get("revenueGrowth"))

    s1 = bool(_mc_b_usd is not None and (_mc_b_usd / _EUR_USD) > 10.0)
    s2 = bool(_de is not None and _de < 100.0)
    s3 = bool(_beta is not None and 0.0 < _beta < 0.8)
    s4 = bool(_dy is not None and _dy > 0.0)
    s5 = bool(_fcf_margin is not None and _fcf_margin > 5.0)
    _g6 = _eqg if _eqg is not None else (_earnings_growth if _earnings_growth is not None else _rg)
    s6 = bool(_fcf_raw is not None and _fcf_raw > 0 and _g6 is not None and _g6 > 0)
    if _rg is not None and _earnings_growth is not None:
        s7 = bool(_rg * 100.0 > 3.0 and _earnings_growth * 100.0 > 3.0)
    elif _rg is not None or _earnings_growth is not None:
        _vals = [g for g in (_rg, _earnings_growth) if g is not None]
        s7 = bool((sum(_vals) / len(_vals)) * 100.0 > 3.0)
    else:
        s7 = False
    for _col, _val in (("s1_ok", s1), ("s2_ok", s2), ("s3_ok", s3), ("s4_ok", s4),
                       ("s5_ok", s5), ("s6_ok", s6), ("s7_ok", s7)):
        frame[_col] = _val
    frame["secure_score"] = int(s1) + int(s2) + int(s3) + int(s4) + int(s5) + int(s6) + int(s7)

    # Prix cibles analystes (depuis info si calendar non disponible)
    frame["target_mean_price"]   = cal.get("target_mean_price")   or _safe_float(info.get("targetMeanPrice"))
    frame["target_high_price"]   = cal.get("target_high_price")   or _safe_float(info.get("targetHighPrice"))
    frame["target_low_price"]    = cal.get("target_low_price")    or _safe_float(info.get("targetLowPrice"))
    frame["target_median_price"] = cal.get("target_median_price") or _safe_float(info.get("targetMedianPrice"))
    frame["analyst_count"]       = _safe_float(info.get("numberOfAnalystOpinions"))
    frame["recommendation_mean"] = cal.get("recommendation_mean") or _safe_float(info.get("recommendationMean"))
    frame["recommendation_key"]  = cal.get("recommendation_key")  or _safe_text(info.get("recommendationKey"))

    # Upside vs cible consensus
    tmean = frame["target_mean_price"].iloc[0] if not pd.isna(frame["target_mean_price"].iloc[0]) else None
    if tmean is not None:
        tmean_usd = tmean * fx_rate_to_usd
        frame["upside_to_target_pct"] = (tmean_usd - price) / price.replace(0, pd.NA) * 100.0
    else:
        frame["upside_to_target_pct"] = pd.NA

    # Comptages recommandations
    for rec_col in ("rec_strong_buy", "rec_buy", "rec_hold", "rec_sell", "rec_strong_sell"):
        frame[rec_col] = cal.get(rec_col)

    # ── Features calendrier (snapshot, calculées par row) ─────────────────
    ned = cal.get("next_earnings_date")
    frame["next_earnings_date"] = ned
    if ned:
        try:
            ned_dt = pd.to_datetime(ned)
            frame["days_to_next_earnings"] = (ned_dt - frame["trade_date"]).dt.days
        except Exception:
            frame["days_to_next_earnings"] = pd.NA
    else:
        frame["days_to_next_earnings"] = pd.NA

    frame["earnings_eps_estimate_avg"]      = cal.get("earnings_eps_estimate_avg")
    frame["earnings_eps_estimate_low"]      = cal.get("earnings_eps_estimate_low")
    frame["earnings_eps_estimate_high"]     = cal.get("earnings_eps_estimate_high")
    frame["earnings_revenue_estimate_avg"]  = cal.get("earnings_revenue_estimate_avg")
    frame["earnings_revenue_estimate_low"]  = cal.get("earnings_revenue_estimate_low")
    frame["earnings_revenue_estimate_high"] = cal.get("earnings_revenue_estimate_high")

    frame["next_dividend_date"]    = cal.get("next_dividend_date")
    frame["next_ex_dividend_date"] = cal.get("next_ex_dividend_date")
    ndd = cal.get("next_dividend_date")
    if ndd:
        try:
            ndd_dt = pd.to_datetime(ndd)
            frame["days_to_next_dividend"] = (ndd_dt - frame["trade_date"]).dt.days
        except Exception:
            frame["days_to_next_dividend"] = pd.NA
    else:
        frame["days_to_next_dividend"] = pd.NA

    frame["last_split_date"]   = cal.get("last_split_date")
    frame["last_split_factor"] = cal.get("last_split_factor")

    # ── Score global & signaux ────────────────────────────────────────────
    frame["score"] = (
        _flag(frame["c1_ok"]) + _flag(frame["c2_ok"]) + _flag(frame["c3_ok"])
        + _flag(frame["c4_ok"]) + _flag(frame["c5_ok"])
    )
    frame["trend"] = frame.apply(
        lambda r: "Hausse" if (_safe_float(r.get("price")) or 0.0) >= (_safe_float(r.get("sma50")) or 0.0) else "Baisse",
        axis=1,
    )
    frame["signal"] = frame["score"].apply(_safe_score_signal)

    default_seuil_achat = 4.2
    default_seuil_vente = -0.5
    frame["seuil_achat"]        = default_seuil_achat
    frame["seuil_vente"]        = default_seuil_vente
    frame["score_over_threshold"] = frame["score"] / default_seuil_achat
    frame["consensus"]          = "Neutre"
    frame["consensus_mean"]     = pd.NA

    frame["rev_growth_pct"] = frame["c1_revenue_growth_pct"]
    frame["cap_range"]      = frame["market_cap_b"].apply(_cap_range_from_market_cap_b)
    frame["param_key"]      = frame["domaine"].fillna("Inconnu").astype(str) + "_" + frame["cap_range"].fillna("Unknown").astype(str)
    frame["selected_param_key"] = frame["param_key"]

    frame["event_count"] = 0
    frame["event_flags"] = None
    frame["fiabilite"]   = pd.NA
    frame["nb_trades"]   = 0
    frame["gagnants"]    = 0
    frame["gain_total_usd"] = 0.0
    frame["gain_moyen_usd"] = 0.0
    frame["computed_at"]    = _utcnow()
    frame["feature_date"]   = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["detection_time"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["source_publication_date"] = pd.to_datetime(
        frame["source_publication_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # ── Liste finale des colonnes (ordre fixe) ────────────────────────────
    # TOTAL : 140 features (+ symbol ajouté par store_daily_feature_series = 141 colonnes Parquet)
    columns = [
        # Identifiant & meta
        "feature_date", "name", "sector", "domaine",
        # Prix & change
        "price", "dprice", "var5j_pct",
        # Devise & FX
        "currency", "fx_rate_to_usd", "values_in_usd",
        # Capitalisation
        "market_cap_b", "market_cap_b_usd",
        # Score & signal
        "score", "signal", "trend",
        # RSI & dérivés
        "rsi", "drsi",
        # Volume
        "volume_mean_usd", "avg_volume_5d", "avg_volume_30d", "avg_volume_90d",
        "c5_volume_ratio", "volume_spike_ratio", "dvolrel",
        # Moyennes mobiles
        "sma20", "sma50", "sma200", "ema12", "ema26",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # Momentum
        "momentum_1m_pct", "c4_momentum_3m_pct", "c4_momentum_6m_pct", "momentum_12m_pct",
        # 52 semaines
        "rolling_52w_high", "rolling_52w_low", "close_vs_52wh_ratio",
        "high_low_52w_pct", "price_vs_sma200_pct",
        # Volatilité
        "realized_vol_21d", "realized_vol_63d", "atr_14",
        # Bollinger
        "bollinger_upper", "bollinger_lower", "bollinger_pct_b",
        # Critères C1→C5
        "c1_revenue_growth_pct", "c1_ok",
        "c2_gross_margin_pct", "c2_ok",
        "c3_undervaluation_hits", "c3_pe_ok", "c3_peg_ok", "c3_below_75pct_52wh", "c3_ok",
        "c4_above_sma50", "c4_ok",
        "c5_ok",
        # Critères Sichere Unternehmen S1→S7 + score
        "beta", "fcf_margin_pct", "earnings_growth_pct",
        "s1_ok", "s2_ok", "s3_ok", "s4_ok", "s5_ok", "s6_ok", "s7_ok", "secure_score",
        # Valorisation
        "trailing_pe", "forward_pe", "peg_ratio",
        "ttm_eps", "ttm_eps_local", "forward_eps", "eps_growth_pct",
        "ebitda_yield_pct", "fcf_yield_pct",
        "de_ratio", "roe_pct", "return_on_assets_pct",
        "book_value", "current_ratio", "quick_ratio",
        "revenue_per_share", "total_cash_b", "total_debt_b", "net_cash_b",
        # Marges
        "gross_margins_pct", "operating_margins_pct", "ebitda_margins_pct", "profit_margins_pct",
        # Croissance
        "rev_growth_pct", "revenue_growth_info_pct",
        "earnings_quarterly_growth_pct", "margin_stability_4q",
        # Dividendes
        "dividend_yield", "payout_ratio", "dividend_stability_1y",
        # Actionnariat
        "short_ratio", "short_percent_float",
        "held_pct_insiders", "held_pct_institutions", "float_shares_m",
        # Comparaison marché
        "week52_change_pct", "sp500_52w_change_pct",
        # Analystes — prix cibles
        "target_mean_price", "target_high_price", "target_low_price", "target_median_price",
        "upside_to_target_pct", "analyst_count",
        "recommendation_mean", "recommendation_key",
        "rec_strong_buy", "rec_buy", "rec_hold", "rec_sell", "rec_strong_sell",
        # Calendrier — prochains événements
        "next_earnings_date", "days_to_next_earnings",
        "earnings_eps_estimate_avg", "earnings_eps_estimate_low", "earnings_eps_estimate_high",
        "earnings_revenue_estimate_avg", "earnings_revenue_estimate_low", "earnings_revenue_estimate_high",
        "next_dividend_date", "next_ex_dividend_date", "days_to_next_dividend",
        "last_split_date", "last_split_factor",
        # Optimisation / backtest
        "cap_range", "param_key", "score_over_threshold",
        "consensus", "consensus_mean", "seuil_achat", "seuil_vente", "selected_param_key",
        "fiabilite", "nb_trades", "gagnants", "gain_total_usd", "gain_moyen_usd",
        # Événements
        "event_count", "event_flags",
        # PIT fondamentaux
        "source_publication_date",
        # Timestamp
        "detection_time", "computed_at",
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
        existing = _safe_read_parquet(path)
        existing = existing[~existing["feature_date"].isin(df["feature_date"])] if not existing.empty else pd.DataFrame(columns=df.columns)
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
    df = _safe_read_parquet(path)
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
        df_p = _safe_read_parquet(price_path, columns=["trade_date"])
        price_rows = len(df_p)
        if not df_p.empty:
            min_date = df_p["trade_date"].min()
            max_date = df_p["trade_date"].max()

    feature_rows = 0
    if feature_path.exists():
        df_f = _safe_read_parquet(feature_path, columns=["feature_date"])
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


def get_latest_event_dates(symbols: List[str] | None = None) -> pd.DataFrame:
    """Dernière ligne par symbole avec les dates d'événements (earnings / ex-dividende),
    lue depuis le store features en UNE requête DuckDB.

    Tolère les schémas hétérogènes (anciens fichiers écrits avant l'ajout des
    colonnes calendrier) grâce à union_by_name : les colonnes manquantes
    deviennent NULL au lieu de faire échouer la requête.

    Colonnes retournées : symbol, feature_date, next_earnings_date, next_ex_dividend_date.
    DataFrame vide si aucune donnée calendrier n'est encore stockée.
    """
    ensure_market_data_schema()
    pattern = str(PARQUET_DIR / "features" / "*/part0.parquet")
    sql = f"""
        SELECT symbol, feature_date, next_earnings_date, next_ex_dividend_date
        FROM read_parquet('{pattern}', union_by_name=true)
        QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY feature_date DESC) = 1
    """
    try:
        df = _duckdb_query(sql)
    except Exception:
        return pd.DataFrame(
            columns=["symbol", "feature_date", "next_earnings_date", "next_ex_dividend_date"]
        )
    if symbols:
        wanted = {_normalize_symbol(s) for s in symbols}
        df = df[df["symbol"].isin(wanted)]
    return df


def get_latest_features(symbols: List[str] | None = None) -> pd.DataFrame:
    """Dernière ligne (feature_date la plus récente) par symbole, TOUTES colonnes,
    en UNE requête DuckDB. union_by_name=true tolère les schémas hétérogènes
    (fichiers anciens sans certaines colonnes → NULL). Base des screeners store-based.
    DataFrame vide si le store est vide.
    """
    ensure_market_data_schema()
    pattern = str(PARQUET_DIR / "features" / "*/part0.parquet")
    sql = f"""
        SELECT *
        FROM read_parquet('{pattern}', union_by_name=true)
        QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY feature_date DESC) = 1
    """
    try:
        df = _duckdb_query(sql)
    except Exception:
        return pd.DataFrame()
    if symbols and not df.empty:
        wanted = {_normalize_symbol(s) for s in symbols}
        df = df[df["symbol"].isin(wanted)]
    return df


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

    calendar_data = _fetch_calendar_snapshot(ticker)
    currency = _safe_text(info.get("currency"))
    price_rows = store_price_history(symbol, history, currency)
    feature_frame = _build_daily_feature_frame(symbol, history, info, calendar_data=calendar_data)
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
    fast_bootstrap: bool = True,
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

    # Pour un nouveau symbole (jamais en DB), on commence par un téléchargement
    # de 36 mois (3y) identique aux scans Big_Growth/Combined — donne suffisamment
    # d'historique pour le rolling 52w, le momentum 6m et les features trimestrielles.
    # Le remplissage historique complet depuis bootstrap_start_date peut se faire
    # lors d'un run suivant (fast_bootstrap=False) ou en tâche de fond.
    is_new_symbol = (latest_row is None)
    if is_new_symbol and fast_bootstrap:
        history = ticker.history(period="3y", auto_adjust=False, actions=True)
        if history is None or history.empty:
            raise ValueError(f"No history available for {symbol}")
        try:
            get_fundamental_metrics(symbol, use_cache=True, allow_stale=False)
        except Exception:
            pass
        upsert_instrument(symbol, info)
        if info:
            store_fundamental_snapshot(symbol, info)
        calendar_data = _fetch_calendar_snapshot(ticker)
        currency = _safe_text(info.get("currency"))
        price_rows = store_price_history(symbol, history, currency)
        feature_frame = _build_daily_feature_frame(symbol, history, info, calendar_data=calendar_data)
        feature_rows = store_daily_feature_series(symbol, feature_frame)
        return {
            "symbol": symbol,
            "status": "bootstrapped_fast",
            "price_rows": price_rows,
            "feature_rows": feature_rows,
            "last_trade_date": feature_frame["feature_date"].iloc[-1] if not feature_frame.empty else None,
        }

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

    calendar_data = _fetch_calendar_snapshot(ticker)
    currency = _safe_text(info.get("currency"))
    price_rows = store_price_history(symbol, history, currency)
    feature_frame = _build_daily_feature_frame(symbol, history, info, calendar_data=calendar_data)
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
# Financial cache (remplace le cache pickle de config.py)
# ---------------------------------------------------------------------------
# Stocke des snapshots arbitraires de données financières (dict) par symbole
# et par type.  Chaque ligne = un symbole × un cache_type × un horodatage.
# Arborescence : market_parquet/financial_cache/cache_type=<type>/symbol=<SYM>/part0.parquet


def _financial_cache_path(symbol: str, cache_type: str) -> Path:
    safe_sym = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_type = cache_type.replace("/", "_")
    return PARQUET_DIR / "financial_cache" / f"cache_type={safe_type}" / f"symbol={safe_sym}" / "part0.parquet"


def get_financial_cache(symbol: str, cache_type: str = "financial", ttl_hours: int = 24) -> Optional[dict]:
    """Lit un snapshot financier depuis le store Parquet (remplace config.get_pickle_cache).

    Retourne le dict le plus récent si son âge est inférieur à *ttl_hours*, sinon None.
    """
    symbol = _normalize_symbol(symbol)
    path = _financial_cache_path(symbol, cache_type)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        df["_fetched_at_dt"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
        latest = df.sort_values("_fetched_at_dt").iloc[-1]
        age_h = (datetime.utcnow() - latest["_fetched_at_dt"].to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600.0
        if age_h >= float(ttl_hours):
            return None
        row = latest.drop(labels=["_fetched_at_dt"]).to_dict()
        # Désérialiser les colonnes JSON (valeurs stockées comme str)
        result: dict = {}
        for k, v in row.items():
            if k in ("symbol", "cache_type", "fetched_at"):
                continue
            if isinstance(v, str):
                try:
                    result[k] = __import__("json").loads(v)
                except Exception:
                    result[k] = v
            elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                result[k] = None
            else:
                result[k] = v
        return result or None
    except Exception:
        return None


def save_financial_cache(symbol: str, data: dict, cache_type: str = "financial") -> bool:
    """Sauvegarde un snapshot financier dans le store Parquet (remplace config.save_pickle_cache).

    Les valeurs scalaires (int, float, str, bool) sont stockées directement.
    Les valeurs composites (dict, list) sont sérialisées en JSON dans une colonne TEXT.
    """
    symbol = _normalize_symbol(symbol)
    ensure_market_data_schema()
    path = _financial_cache_path(symbol, cache_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    flat: dict = {"symbol": symbol, "cache_type": cache_type, "fetched_at": _utcnow()}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            flat[k] = _json.dumps(v, default=str)
        elif v is None:
            flat[k] = None
        else:
            flat[k] = v

    new_row = pd.DataFrame([flat])
    if path.exists():
        existing = _safe_read_parquet(path)
        # Garder les 4 derniers snapshots pour traçabilité, supprimer les plus anciens
        if not existing.empty:
            existing = existing.sort_values("fetched_at").tail(4)
        combined = pd.concat([existing, new_row], ignore_index=True) if not existing.empty else new_row
    else:
        combined = new_row

    try:
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Timeline cache (remplace timeline_cache.py SQLite)
# ---------------------------------------------------------------------------
# Stocke les données point-in-time (earnings, recommendations, insider) par symbole.
# Arborescence :
#   market_parquet/timeline/earnings/symbol=<SYM>/part0.parquet
#   market_parquet/timeline/recommendations/symbol=<SYM>/part0.parquet
#   market_parquet/timeline/insider/symbol=<SYM>/part0.parquet


def _timeline_path(category: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    return PARQUET_DIR / "timeline" / category / f"symbol={safe}" / "part0.parquet"


def _upsert_timeline(path: Path, incoming: pd.DataFrame, key_cols: list[str]) -> int:
    """Merge *incoming* dans *path* en dédupliquant sur *key_cols*."""
    if incoming.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _safe_read_parquet(path)
        if not existing.empty:
            key_str = incoming[key_cols].astype(str).apply("|".join, axis=1)
            ex_key_str = existing[key_cols].astype(str).apply("|".join, axis=1)
            existing = existing[~ex_key_str.isin(key_str)]
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
    else:
        combined = incoming
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
    return len(incoming)


def store_timeline_earnings(symbol: str, df: pd.DataFrame) -> int:
    """Stocke les earnings (EPS, surprise) en Parquet.

    Colonnes attendues : date, eps_estimate, eps_actual, surprise_pct.
    """
    symbol = _normalize_symbol(symbol)
    if df is None or df.empty:
        return 0
    ensure_market_data_schema()
    df = df.copy()
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["symbol", "date"])
    for col in ("eps_estimate", "eps_actual", "surprise_pct"):
        if col not in df.columns:
            df[col] = None
    df = df[["symbol", "date", "eps_estimate", "eps_actual", "surprise_pct"]]
    return _upsert_timeline(_timeline_path("earnings", symbol), df, ["symbol", "date"])


def store_timeline_recommendations(symbol: str, df: pd.DataFrame) -> int:
    """Stocke les recommandations analystes en Parquet.

    Colonnes attendues : date, firm, to_grade, from_grade, action.
    """
    symbol = _normalize_symbol(symbol)
    if df is None or df.empty:
        return 0
    ensure_market_data_schema()
    df = df.copy()
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "firm"]).drop_duplicates(subset=["symbol", "date", "firm"])
    for col in ("to_grade", "from_grade", "action"):
        if col not in df.columns:
            df[col] = None
    df = df[["symbol", "date", "firm", "to_grade", "from_grade", "action"]]
    return _upsert_timeline(_timeline_path("recommendations", symbol), df, ["symbol", "date", "firm"])


def store_timeline_insider(symbol: str, df: pd.DataFrame) -> int:
    """Stocke les transactions insider en Parquet.

    Colonnes attendues : date, shares, value, transaction_text.
    """
    symbol = _normalize_symbol(symbol)
    if df is None or df.empty:
        return 0
    ensure_market_data_schema()
    df = df.copy()
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["symbol", "date", "transaction_text"])
    for col in ("shares", "value", "transaction_text"):
        if col not in df.columns:
            df[col] = None
    df = df[["symbol", "date", "shares", "value", "transaction_text"]]
    return _upsert_timeline(_timeline_path("insider", symbol), df, ["symbol", "date", "transaction_text"])


def get_timeline_pit_data(symbol: str, target_date: str, lookback_days: int = 120) -> dict:
    """Retourne les données point-in-time (PIT) connues à *target_date*.

    Réplique l'API de TimelineCache.get_pit_timeline_data() mais depuis Parquet.
    """
    symbol = _normalize_symbol(symbol)
    # Valider le format de date
    datetime.strptime(target_date, "%Y-%m-%d")
    cutoff_dt = pd.to_datetime(target_date)
    lookback_dt = cutoff_dt - timedelta(days=lookback_days)

    # -- Earnings : dernier connu avant ou à target_date --
    earn_path = _timeline_path("earnings", symbol)
    latest_surprise = 0.0
    if earn_path.exists():
        df_e = _safe_read_parquet(earn_path)
        if not df_e.empty:
            df_e["_date"] = pd.to_datetime(df_e["date"], errors="coerce")
            past = df_e[df_e["_date"] <= cutoff_dt].sort_values("_date")
            if not past.empty:
                latest_surprise = _safe_float(past.iloc[-1].get("surprise_pct")) or 0.0

    # -- Recommandations : nombre d'upgrades dans la fenêtre glissante --
    rec_path = _timeline_path("recommendations", symbol)
    upgrades_count = 0
    if rec_path.exists():
        df_r = _safe_read_parquet(rec_path)
        if not df_r.empty:
            df_r["_date"] = pd.to_datetime(df_r["date"], errors="coerce")
            window = df_r[
                (df_r["_date"] <= cutoff_dt)
                & (df_r["_date"] >= lookback_dt)
                & df_r["action"].str.lower().isin(["up", "init"])
            ]
            upgrades_count = len(window)

    return {
        "latest_earnings_surprise": latest_surprise,
        "recent_upgrades_count": upgrades_count,
    }


def update_timeline_data(symbol: str) -> None:
    """Télécharge et stocke les données timeline depuis yfinance (earnings + reco + insider).

    Remplace TimelineCache.update_timeline_data().
    """
    symbol = _normalize_symbol(symbol)
    ensure_market_data_schema()
    ticker = yf.Ticker(symbol)

    # --- Earnings ---
    try:
        earnings = ticker.earnings_dates
        if earnings is not None and not earnings.empty:
            df_earn = earnings.reset_index().rename(columns={
                "Earnings Date": "date",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
                "Surprise(%)": "surprise_pct",
            })
            df_earn["date"] = pd.to_datetime(df_earn["date"], errors="coerce", utc=True).dt.tz_localize(None)
            store_timeline_earnings(symbol, df_earn)
    except Exception:
        pass

    # --- Recommandations analystes ---
    try:
        recom = ticker.upgrades_downgrades
        if recom is not None and not recom.empty:
            df_rec = recom.reset_index().rename(columns={
                "GradeDate": "date", "Firm": "firm", "ToGrade": "to_grade",
                "FromGrade": "from_grade", "Action": "action",
            })
            df_rec["date"] = pd.to_datetime(df_rec["date"], errors="coerce", utc=True).dt.tz_localize(None)
            store_timeline_recommendations(symbol, df_rec)
    except Exception:
        pass

    # --- Transactions insider ---
    try:
        insider = ticker.insider_transactions
        if insider is not None and not insider.empty:
            df_ins = insider.reset_index().rename(columns={
                "Start Date": "date", "Shares": "shares",
                "Value": "value", "Text": "transaction_text",
            })
            if "date" not in df_ins.columns and "Date" in df_ins.columns:
                df_ins = df_ins.rename(columns={"Date": "date"})
            df_ins["date"] = pd.to_datetime(df_ins["date"], errors="coerce", utc=True).dt.tz_localize(None)
            if "transaction_text" not in df_ins.columns:
                df_ins["transaction_text"] = df_ins.get("Text", pd.Series("", index=df_ins.index)).astype(str)
            store_timeline_insider(symbol, df_ins)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Initialisation automatique à l'import
# ---------------------------------------------------------------------------
ensure_market_data_schema()
