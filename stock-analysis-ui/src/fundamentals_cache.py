"""
Fundamentals extraction and caching for trading system.
Provides yfinance-backed fetching with SQLite caching and TTL (24h default).

Quarterly data is stored EVOLUTIVELY: once a past quarter is fetched, it is
never requested again from yfinance.  Only the most recent (potentially
incomplete) quarter is refreshed.
"""

import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

try:
    from config import DB_PATH
except ImportError:
    DB_PATH = str(Path(__file__).parent.resolve() / 'stock_analysis.db')

FUNDAMENTALS_CACHE_TTL_HOURS = 24
# TTL for snapshot (point-in-time) data like market cap, ROE, etc.
SNAPSHOT_TTL_HOURS = 24

def _ensure_fundamentals_table():
    """Create all fundamentals cache tables if not exists (idempotent)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Legacy table — kept for backward compatibility
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                rev_growth REAL,
                eps_growth REAL,
                gross_margin REAL,
                fcf_yield REAL,
                de_ratio REAL,
                roe REAL,
                ocf_yield REAL,
                net_margin REAL,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_symbol UNIQUE (symbol)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_fetched ON fundamental_metrics(symbol, fetched_at DESC)')

        # ── Evolutionary quarterly storage ──
        # Each (symbol, quarter_date, source) row is immutable once stored.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quarterly_financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quarter_date TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'income',
                total_revenue REAL,
                gross_profit REAL,
                operating_income REAL,
                net_income REAL,
                ebitda REAL,
                diluted_eps REAL,
                cost_of_revenue REAL,
                operating_expense REAL,
                research_and_development REAL,
                free_cash_flow REAL,
                operating_cash_flow REAL,
                capital_expenditure REAL,
                total_debt REAL,
                stockholders_equity REAL,
                total_assets REAL,
                cash_and_equivalents REAL,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, quarter_date, source)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qf_symbol ON quarterly_financials(symbol, quarter_date DESC)')

        # ── Evolutionary annual storage ──
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annual_financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                fiscal_date TEXT NOT NULL,
                total_revenue REAL,
                gross_profit REAL,
                operating_income REAL,
                net_income REAL,
                ebitda REAL,
                diluted_eps REAL,
                cost_of_revenue REAL,
                operating_expense REAL,
                research_and_development REAL,
                free_cash_flow REAL,
                operating_cash_flow REAL,
                capital_expenditure REAL,
                total_debt REAL,
                stockholders_equity REAL,
                total_assets REAL,
                cash_and_equivalents REAL,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, fiscal_date)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_af_symbol ON annual_financials(symbol, fiscal_date DESC)')

        # ── Snapshot metrics (point-in-time, refreshed regularly) ──
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshot_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                market_cap REAL,
                enterprise_value REAL,
                roe REAL,
                de_ratio REAL,
                gross_margins REAL,
                profit_margins REAL,
                revenue_growth REAL,
                ebitda REAL,
                free_cashflow REAL,
                trailing_eps REAL,
                sector TEXT,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snap_symbol ON snapshot_metrics(symbol)')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Error ensuring fundamentals table: {e}")
        return False

def get_fundamental_metrics(symbol: str, lookback_quarters: int = 4, use_cache: bool = True, allow_stale: bool = False) -> Dict[str, Optional[float]]:
    """
    Extract fundamental metrics for a symbol via yfinance with local SQLite caching.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        lookback_quarters: Number of quarters for growth calculations (default: 4 = YoY)
        use_cache: If True, use cached values within TTL; if False, always fetch fresh
        allow_stale: If True, return expired cache data without fetching (useful during optimization)
    
    Returns:
        Dict with keys: rev_growth, eps_growth, gross_margin, fcf_yield, de_ratio, roe, ocf_yield, net_margin
        Values are floats or None (if not available).
        Example:
        {
            'rev_growth': 12.5,          # %
            'eps_growth': 15.3,          # %
            'gross_margin': 42.1,        # %
            'fcf_yield': 3.2,            # %
            'de_ratio': 0.45,            # ratio
            'roe': 18.5,                 # %
            'ocf_yield': 4.1,            # %
            'net_margin': 8.3            # %
        }
    """
    
    # Ensure table exists
    _ensure_fundamentals_table()
    
    # Check cache (fresh)
    if use_cache:
        cached = _get_cached_metrics(symbol)
        if cached:
            return cached
    
    # If allow_stale, return expired cache data without network fetch
    if allow_stale:
        stale = _get_cached_metrics(symbol, ignore_ttl=True)
        if stale:
            return stale
        # No cache at all — return empty metrics (no network call)
        return {
            'rev_growth': None, 'eps_growth': None, 'gross_margin': None,
            'fcf_yield': None, 'de_ratio': None, 'roe': None,
            'ocf_yield': None, 'net_margin': None,
        }
    
    # Fetch fresh from yfinance
    metrics = _fetch_metrics_from_yfinance(symbol, lookback_quarters)
    
    # Save to cache
    if metrics:
        _save_metrics_to_cache(symbol, metrics)
    
    return metrics

def _get_cached_metrics(symbol: str, ignore_ttl: bool = False) -> Optional[Dict[str, Optional[float]]]:
    """Check cache for symbol; return if within TTL (or any age if ignore_ttl), else None."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if ignore_ttl:
            cursor.execute('''
                SELECT rev_growth, eps_growth, gross_margin, fcf_yield, de_ratio, roe, ocf_yield, net_margin
                FROM fundamental_metrics
                WHERE symbol = ?
            ''', (symbol.upper(),))
        else:
            ttl_cutoff = datetime.now() - timedelta(hours=FUNDAMENTALS_CACHE_TTL_HOURS)
            cursor.execute('''
                SELECT rev_growth, eps_growth, gross_margin, fcf_yield, de_ratio, roe, ocf_yield, net_margin
                FROM fundamental_metrics
                WHERE symbol = ? AND fetched_at > ?
            ''', (symbol.upper(), ttl_cutoff.isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'rev_growth': row['rev_growth'],
                'eps_growth': row['eps_growth'],
                'gross_margin': row['gross_margin'],
                'fcf_yield': row['fcf_yield'],
                'de_ratio': row['de_ratio'],
                'roe': row['roe'],
                'ocf_yield': row['ocf_yield'],
                'net_margin': row['net_margin'],
            }
        return None
    except Exception as e:
        print(f"⚠️ Error reading cache for {symbol}: {e}")
        return None

def _fetch_metrics_from_yfinance(symbol: str, lookback_quarters: int = 4) -> Dict[str, Optional[float]]:
    """Fetch fundamentals from yfinance, storing quarterly data evolutively.
    Past quarters already in DB are NOT re-fetched."""
    metrics = {
        'rev_growth': None,
        'eps_growth': None,
        'gross_margin': None,
        'fcf_yield': None,
        'de_ratio': None,
        'roe': None,
        'ocf_yield': None,
        'net_margin': None,
    }
    
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # ── Store/update snapshot (point-in-time) metrics ──
        _save_snapshot(symbol, info)

        # ── Snapshot-derived metrics ──
        if 'grossMargins' in info and info['grossMargins'] is not None:
            try:
                metrics['gross_margin'] = float(info['grossMargins']) * 100
            except (TypeError, ValueError):
                pass
        
        if 'profitMargins' in info and info['profitMargins'] is not None:
            try:
                metrics['net_margin'] = float(info['profitMargins']) * 100
            except (TypeError, ValueError):
                pass
        
        if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
            try:
                metrics['roe'] = float(info['returnOnEquity']) * 100
            except (TypeError, ValueError):
                pass
        
        if 'debtToEquity' in info and info['debtToEquity'] is not None:
            try:
                metrics['de_ratio'] = float(info['debtToEquity'])
            except (TypeError, ValueError):
                pass

        # ── Evolutively store quarterly data ──
        _store_quarterly_data(symbol, ticker)

        # ── Evolutively store annual data (fallback for PIT backtest) ──
        _store_annual_data(symbol, ticker)

        # ── Compute derived metrics from stored quarterly data ──
        quarters = get_quarterly_history(symbol, limit=lookback_quarters + 1)
        
        if len(quarters) >= 2:
            # Revenue Growth (YoY if enough quarters, else QoQ)
            most_recent = quarters[0]
            if len(quarters) > lookback_quarters:
                compare_to = quarters[lookback_quarters]
            else:
                compare_to = quarters[-1]
            
            rev_now = most_recent.get('total_revenue')
            rev_then = compare_to.get('total_revenue')
            if rev_now and rev_then and rev_then != 0:
                metrics['rev_growth'] = ((rev_now - rev_then) / abs(rev_then)) * 100
            
            # EPS Growth
            eps_now = most_recent.get('diluted_eps')
            eps_then = compare_to.get('diluted_eps')
            if eps_now and eps_then and eps_then != 0:
                metrics['eps_growth'] = ((eps_now - eps_then) / abs(eps_then)) * 100
        
        # FCF Yield and OCF Yield from most recent quarter
        if quarters:
            recent = quarters[0]
            market_cap = float(info.get('marketCap', 0) or 0)
            
            fcf = recent.get('free_cash_flow')
            if fcf and market_cap > 0:
                metrics['fcf_yield'] = (float(fcf) / market_cap) * 100
            
            ocf = recent.get('operating_cash_flow')
            if ocf and market_cap > 0:
                metrics['ocf_yield'] = (float(ocf) / market_cap) * 100
    
    except Exception as e:
        print(f"⚠️ Error fetching yfinance data for {symbol}: {e}")
    
    return metrics


# ===================================================================
# EVOLUTIONARY QUARTERLY STORAGE
# ===================================================================

def _get_stored_quarter_dates(symbol: str) -> set:
    """Return set of quarter_date strings already stored for this symbol."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT DISTINCT quarter_date FROM quarterly_financials WHERE symbol = ?',
            (symbol.upper(),)
        )
        dates = {row[0] for row in cursor.fetchall()}
        conn.close()
        return dates
    except Exception:
        return set()


def _store_quarterly_data(symbol: str, ticker):
    """Fetch and store quarterly data from yfinance.
    Only NEW quarters (not yet in DB) are inserted — past data is immutable."""
    symbol = symbol.upper()
    stored_dates = _get_stored_quarter_dates(symbol)
    
    rows_to_insert = []
    
    try:
        # ── Income Statement ──
        qf = ticker.quarterly_financials
        qc = ticker.quarterly_cashflow
        qb = ticker.quarterly_balance_sheet
        
        # Collect all quarter dates across all statements
        all_dates = set()
        for df in [qf, qc, qb]:
            if df is not None and not df.empty:
                for col in df.columns:
                    all_dates.add(str(col.date()))
        
        for qdate in all_dates:
            if qdate in stored_dates:
                continue  # Already stored — skip
            
            row = {
                'symbol': symbol,
                'quarter_date': qdate,
                'source': 'combined',
            }
            
            # Extract from income statement
            if qf is not None and not qf.empty:
                ts_col = _find_column_by_date(qf, qdate)
                if ts_col is not None:
                    row['total_revenue'] = _safe_val(qf, 'Total Revenue', ts_col)
                    row['gross_profit'] = _safe_val(qf, 'Gross Profit', ts_col)
                    row['operating_income'] = _safe_val(qf, 'Operating Income', ts_col)
                    row['net_income'] = _safe_val(qf, 'Net Income', ts_col)
                    row['ebitda'] = _safe_val(qf, 'EBITDA', ts_col)
                    row['diluted_eps'] = _safe_val(qf, 'Diluted EPS', ts_col)
                    row['cost_of_revenue'] = _safe_val(qf, 'Cost Of Revenue', ts_col)
                    row['operating_expense'] = _safe_val(qf, 'Operating Expense', ts_col)
                    row['research_and_development'] = _safe_val(qf, 'Research And Development', ts_col)
            
            # Extract from cash flow
            if qc is not None and not qc.empty:
                ts_col = _find_column_by_date(qc, qdate)
                if ts_col is not None:
                    row['free_cash_flow'] = _safe_val(qc, 'Free Cash Flow', ts_col)
                    row['operating_cash_flow'] = _safe_val(qc, 'Operating Cash Flow', ts_col)
                    row['capital_expenditure'] = _safe_val(qc, 'Capital Expenditure', ts_col)
            
            # Extract from balance sheet
            if qb is not None and not qb.empty:
                ts_col = _find_column_by_date(qb, qdate)
                if ts_col is not None:
                    row['total_debt'] = _safe_val(qb, 'Total Debt', ts_col)
                    row['stockholders_equity'] = _safe_val(qb, 'Stockholders Equity', ts_col)
                    row['total_assets'] = _safe_val(qb, 'Total Assets', ts_col)
                    row['cash_and_equivalents'] = _safe_val(qb, 'Cash And Cash Equivalents', ts_col)
            
            rows_to_insert.append(row)
    
    except Exception as e:
        print(f"⚠️ Error extracting quarterly data for {symbol}: {e}")
    
    # Bulk insert new quarters
    if rows_to_insert:
        _bulk_insert_quarters(rows_to_insert)
        print(f"📊 {symbol}: {len(rows_to_insert)} nouveau(x) trimestre(s) stocké(s) en DB")


def _store_annual_data(symbol: str, ticker):
    """Fetch and store annual financial data from yfinance.
    Only NEW fiscal years (not yet in DB) are inserted — past data is immutable.
    Annual data covers ~5 years and serves as fallback for PIT when quarterly
    data is unavailable."""
    symbol = symbol.upper()

    # Get already-stored fiscal dates for this symbol
    stored_dates = set()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT DISTINCT fiscal_date FROM annual_financials WHERE symbol = ?',
            (symbol,)
        )
        stored_dates = {row[0] for row in cursor.fetchall()}
        conn.close()
    except Exception:
        pass

    rows_to_insert = []

    try:
        af = ticker.financials           # Annual income statement
        ac = ticker.cashflow             # Annual cash flow
        ab = ticker.balance_sheet        # Annual balance sheet

        # Collect all fiscal year-end dates across all statements
        all_dates = set()
        for df in [af, ac, ab]:
            if df is not None and not df.empty:
                for col in df.columns:
                    all_dates.add(str(col.date()))

        for fdate in all_dates:
            if fdate in stored_dates:
                continue  # Already stored

            row = {'symbol': symbol, 'fiscal_date': fdate}

            # Income statement
            if af is not None and not af.empty:
                ts_col = _find_column_by_date(af, fdate)
                if ts_col is not None:
                    row['total_revenue'] = _safe_val(af, 'Total Revenue', ts_col)
                    row['gross_profit'] = _safe_val(af, 'Gross Profit', ts_col)
                    row['operating_income'] = _safe_val(af, 'Operating Income', ts_col)
                    row['net_income'] = _safe_val(af, 'Net Income', ts_col)
                    row['ebitda'] = _safe_val(af, 'EBITDA', ts_col)
                    row['diluted_eps'] = _safe_val(af, 'Diluted EPS', ts_col)
                    row['cost_of_revenue'] = _safe_val(af, 'Cost Of Revenue', ts_col)
                    row['operating_expense'] = _safe_val(af, 'Operating Expense', ts_col)
                    row['research_and_development'] = _safe_val(af, 'Research And Development', ts_col)

            # Cash flow
            if ac is not None and not ac.empty:
                ts_col = _find_column_by_date(ac, fdate)
                if ts_col is not None:
                    row['free_cash_flow'] = _safe_val(ac, 'Free Cash Flow', ts_col)
                    row['operating_cash_flow'] = _safe_val(ac, 'Operating Cash Flow', ts_col)
                    row['capital_expenditure'] = _safe_val(ac, 'Capital Expenditure', ts_col)

            # Balance sheet
            if ab is not None and not ab.empty:
                ts_col = _find_column_by_date(ab, fdate)
                if ts_col is not None:
                    row['total_debt'] = _safe_val(ab, 'Total Debt', ts_col)
                    row['stockholders_equity'] = _safe_val(ab, 'Stockholders Equity', ts_col)
                    row['total_assets'] = _safe_val(ab, 'Total Assets', ts_col)
                    row['cash_and_equivalents'] = _safe_val(ab, 'Cash And Cash Equivalents', ts_col)

            rows_to_insert.append(row)

    except Exception as e:
        print(f"⚠️ Error extracting annual data for {symbol}: {e}")

    # Bulk insert
    if rows_to_insert:
        _bulk_insert_annual(rows_to_insert)
        print(f"📊 {symbol}: {len(rows_to_insert)} année(s) stockée(s) en DB")


def _bulk_insert_annual(rows: List[dict]):
    """Insert multiple annual rows into DB."""
    fields = [
        'symbol', 'fiscal_date',
        'total_revenue', 'gross_profit', 'operating_income', 'net_income',
        'ebitda', 'diluted_eps', 'cost_of_revenue', 'operating_expense',
        'research_and_development', 'free_cash_flow', 'operating_cash_flow',
        'capital_expenditure', 'total_debt', 'stockholders_equity',
        'total_assets', 'cash_and_equivalents'
    ]
    placeholders = ', '.join(['?'] * len(fields))
    cols = ', '.join(fields)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for row in rows:
            values = tuple(row.get(f) for f in fields)
            cursor.execute(
                f'INSERT OR IGNORE INTO annual_financials ({cols}) VALUES ({placeholders})',
                values
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error bulk inserting annual data: {e}")


def get_all_annual_sorted(symbol: str) -> List[dict]:
    """Retrieve ALL stored annual data for a symbol, sorted by fiscal_date ASC.
    Used as fallback for point-in-time when quarterly data is unavailable."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM annual_financials
            WHERE symbol = ?
            ORDER BY fiscal_date ASC
        ''', (symbol.upper(),))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _find_column_by_date(df, date_str: str):
    """Find the DataFrame column matching a date string."""
    for col in df.columns:
        if str(col.date()) == date_str:
            return col
    return None


def _safe_val(df, row_name: str, col):
    """Safely extract a value from a DataFrame cell."""
    try:
        if row_name in df.index:
            val = df.loc[row_name, col]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return None


def _bulk_insert_quarters(rows: List[dict]):
    """Insert multiple quarterly rows into DB."""
    fields = [
        'symbol', 'quarter_date', 'source',
        'total_revenue', 'gross_profit', 'operating_income', 'net_income',
        'ebitda', 'diluted_eps', 'cost_of_revenue', 'operating_expense',
        'research_and_development', 'free_cash_flow', 'operating_cash_flow',
        'capital_expenditure', 'total_debt', 'stockholders_equity',
        'total_assets', 'cash_and_equivalents'
    ]
    placeholders = ', '.join(['?'] * len(fields))
    cols = ', '.join(fields)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for row in rows:
            values = tuple(row.get(f) for f in fields)
            cursor.execute(
                f'INSERT OR IGNORE INTO quarterly_financials ({cols}) VALUES ({placeholders})',
                values
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error bulk inserting quarters: {e}")


def get_quarterly_history(symbol: str, limit: int = 8) -> List[dict]:
    """Retrieve stored quarterly data for a symbol, most recent first.
    Returns list of dicts with all stored fields."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM quarterly_financials
            WHERE symbol = ?
            ORDER BY quarter_date DESC
            LIMIT ?
        ''', (symbol.upper(), limit))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_all_quarters_sorted(symbol: str) -> List[dict]:
    """Retrieve ALL stored quarterly data for a symbol, sorted by quarter_date ASC.
    Used for point-in-time backtest lookups."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM quarterly_financials
            WHERE symbol = ?
            ORDER BY quarter_date ASC
        ''', (symbol.upper(),))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _annuals_to_pseudo_quarters(annuals: List[dict]) -> List[dict]:
    """Convert annual financial records into pseudo-quarterly records.

    Each annual record is split into 4 pseudo-quarters (Q1–Q4) with dates
    spread at -9, -6, -3, and 0 months from the fiscal year-end.

    Flow metrics (income, cash flow) are divided by 4.
    Stock metrics (balance sheet) remain unchanged.
    Ratios and margins computed later will be identical since both
    numerator and denominator are divided by the same factor.

    Returns pseudo-quarters sorted by quarter_date ASC.
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # Flow fields: divide by 4
    FLOW_FIELDS = [
        'total_revenue', 'gross_profit', 'operating_income', 'net_income',
        'ebitda', 'diluted_eps', 'cost_of_revenue', 'operating_expense',
        'research_and_development', 'free_cash_flow', 'operating_cash_flow',
        'capital_expenditure',
    ]
    # Stock fields: keep as-is
    STOCK_FIELDS = [
        'total_debt', 'stockholders_equity', 'total_assets',
        'cash_and_equivalents',
    ]

    pseudo_quarters = []

    for annual in annuals:
        fiscal_date_str = annual.get('fiscal_date', annual.get('quarter_date', ''))
        try:
            fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            continue

        # Create 4 pseudo-quarters ending at -9m, -6m, -3m, 0m from fiscal year-end
        for q_offset in [9, 6, 3, 0]:
            q_date = fiscal_date - relativedelta(months=q_offset)
            pq = {
                'symbol': annual.get('symbol'),
                'quarter_date': q_date.strftime('%Y-%m-%d'),
                '_pseudo_quarterly': True,  # marker for debugging
            }

            # Divide flow metrics by 4
            for field in FLOW_FIELDS:
                val = annual.get(field)
                pq[field] = val / 4.0 if val is not None else None

            # Keep stock metrics unchanged
            for field in STOCK_FIELDS:
                pq[field] = annual.get(field)

            pseudo_quarters.append(pq)

    # Sort by quarter_date ASC
    pseudo_quarters.sort(key=lambda x: x['quarter_date'])
    return pseudo_quarters


def compute_pit_fundamentals(quarters_sorted: List[dict], as_of_date: str,
                              publication_delay_days: int = 45,
                              lookback_quarters: int = 4,
                              annuals_sorted: List[dict] = None) -> Optional[Dict[str, Optional[float]]]:
    """Compute fundamental metrics using only data available at a given date.

    This eliminates look-ahead bias by only considering quarters whose
    publication date (quarter_date + publication_delay_days) is <= as_of_date.

    When quarterly data is unavailable, falls back to annual data converted
    into pseudo-quarterly records (flow metrics ÷ 4, stock metrics unchanged)
    with a 90-day publication delay. This ensures all metrics are directly
    comparable regardless of source:
      - Growth: (rev_q / rev_q-4) works identically since both are ÷4
      - Margins: (GP/4) / (Rev/4) = GP/Rev (identical)
      - ROE: (NI/4) / Equity = quarterly ROE (same treatment as real quarters)
      - FCF/OCF yields: trailing 4Q sum = 4 × (FCF/4) = full-year FCF

    Args:
        quarters_sorted: ALL quarters for a symbol, sorted by quarter_date ASC.
        as_of_date: The simulation date as 'YYYY-MM-DD' string.
        publication_delay_days: Assumed delay for quarterly results (default: 45d).
        lookback_quarters: How many quarters back for growth (4 = YoY).
        annuals_sorted: ALL annual records sorted by fiscal_date ASC (fallback).

    Returns:
        Dict with rev_growth, eps_growth, roe, fcf_yield, de_ratio, gross_margin,
        net_margin, ocf_yield — or None if no data available at that date.
    """
    from datetime import datetime, timedelta

    try:
        cutoff = datetime.strptime(as_of_date, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None

    # ── Try quarterly data first ──
    available_q = []
    for q in (quarters_sorted or []):
        try:
            qd = datetime.strptime(q['quarter_date'], '%Y-%m-%d')
            if qd + timedelta(days=publication_delay_days) <= cutoff:
                available_q.append(q)
        except (ValueError, TypeError, KeyError):
            continue

    if available_q:
        return _compute_metrics_from_periods(available_q, lookback_quarters)

    # ── Fallback: convert annual data to pseudo-quarters ──
    ANNUAL_PUB_DELAY = 90
    available_a = []
    for a in (annuals_sorted or []):
        try:
            fd = datetime.strptime(a.get('fiscal_date', a.get('quarter_date', '')), '%Y-%m-%d')
            if fd + timedelta(days=ANNUAL_PUB_DELAY) <= cutoff:
                available_a.append(a)
        except (ValueError, TypeError, KeyError):
            continue

    if available_a:
        pseudo_q = _annuals_to_pseudo_quarters(available_a)
        # Filter pseudo-quarters by the same publication delay logic
        usable_pq = []
        for pq in pseudo_q:
            try:
                qd = datetime.strptime(pq['quarter_date'], '%Y-%m-%d')
                if qd + timedelta(days=ANNUAL_PUB_DELAY) <= cutoff:
                    usable_pq.append(pq)
            except (ValueError, TypeError, KeyError):
                continue
        if usable_pq:
            return _compute_metrics_from_periods(usable_pq, lookback_quarters)

    # ── Last resort: use oldest available data despite date mismatch ──
    # Better to have stale fundamentals than none at all during backtest.
    # Priority: real quarterly > pseudo-quarterly from annuals
    all_q = list(quarters_sorted or [])
    if all_q:
        return _compute_metrics_from_periods(all_q, lookback_quarters)

    all_annuals = list(annuals_sorted or [])
    if all_annuals:
        pseudo_q = _annuals_to_pseudo_quarters(all_annuals)
        if pseudo_q:
            return _compute_metrics_from_periods(pseudo_q, lookback_quarters)

    return None


def _compute_metrics_from_periods(periods: List[dict], lookback_quarters: int = 4) -> Dict[str, Optional[float]]:
    """Compute fundamental metrics from quarterly periods.

    All periods (real or pseudo-quarterly from annual conversion) are treated
    uniformly. Growth is computed over lookback_quarters (default 4 = YoY).
    FCF/OCF yields use trailing 4 quarters sum.
    """
    most_recent = periods[-1]

    metrics = {
        'rev_growth': None,
        'eps_growth': None,
        'gross_margin': None,
        'fcf_yield': None,
        'de_ratio': None,
        'roe': None,
        'ocf_yield': None,
        'net_margin': None,
        'revenue_growth': None,
        'earnings_growth': None,
        'debt_to_equity': None,
    }

    # ── Growth metrics (compare to lookback_quarters ago) ──
    if len(periods) > lookback_quarters:
        compare_to = periods[-(lookback_quarters + 1)]
    elif len(periods) >= 2:
        compare_to = periods[0]
    else:
        compare_to = None

    if compare_to:
        rev_now = most_recent.get('total_revenue')
        rev_then = compare_to.get('total_revenue')
        if rev_now and rev_then and rev_then != 0:
            metrics['rev_growth'] = ((rev_now - rev_then) / abs(rev_then)) * 100
            metrics['revenue_growth'] = metrics['rev_growth']

        eps_now = most_recent.get('diluted_eps')
        eps_then = compare_to.get('diluted_eps')
        if eps_now and eps_then and eps_then != 0:
            metrics['eps_growth'] = ((eps_now - eps_then) / abs(eps_then)) * 100
            metrics['earnings_growth'] = metrics['eps_growth']

    # ── Margin metrics from most recent period ──
    rev = most_recent.get('total_revenue')
    gp = most_recent.get('gross_profit')
    ni = most_recent.get('net_income')
    if rev and rev != 0:
        if gp is not None:
            metrics['gross_margin'] = (gp / abs(rev)) * 100
        if ni is not None:
            metrics['net_margin'] = (ni / abs(rev)) * 100

    # ── ROE ──
    equity = most_recent.get('stockholders_equity')
    if ni and equity and equity != 0:
        metrics['roe'] = (ni / abs(equity)) * 100

    # ── Debt-to-Equity ──
    debt = most_recent.get('total_debt')
    if debt is not None and equity and equity != 0:
        metrics['de_ratio'] = debt / abs(equity)
        metrics['debt_to_equity'] = metrics['de_ratio']

    # ── FCF & OCF yields (trailing 4 quarters) ──
    total_assets = most_recent.get('total_assets')
    approx_cap = total_assets if total_assets and total_assets > 0 else None

    trailing = periods[-min(4, len(periods)):]
    total_fcf = sum(q.get('free_cash_flow') or 0 for q in trailing)
    total_ocf = sum(q.get('operating_cash_flow') or 0 for q in trailing)
    if approx_cap:
        if total_fcf != 0:
            metrics['fcf_yield'] = (total_fcf / approx_cap) * 100
        if total_ocf != 0:
            metrics['ocf_yield'] = (total_ocf / approx_cap) * 100

    return metrics


# ===================================================================
# SNAPSHOT (POINT-IN-TIME) STORAGE
# ===================================================================

def _save_snapshot(symbol: str, info: dict):
    """Save current point-in-time metrics (market cap, ROE, etc.)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO snapshot_metrics
            (symbol, market_cap, enterprise_value, roe, de_ratio, gross_margins,
             profit_margins, revenue_growth, ebitda, free_cashflow, trailing_eps,
             sector, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol.upper(),
            info.get('marketCap'),
            info.get('enterpriseValue'),
            info.get('returnOnEquity'),
            info.get('debtToEquity'),
            info.get('grossMargins'),
            info.get('profitMargins'),
            info.get('revenueGrowth'),
            info.get('ebitda'),
            info.get('freeCashflow'),
            info.get('trailingEps'),
            info.get('sector', 'Inconnu'),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error saving snapshot for {symbol}: {e}")


def get_snapshot(symbol: str, max_age_hours: int = SNAPSHOT_TTL_HOURS) -> Optional[dict]:
    """Get cached snapshot metrics if fresh enough."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        cursor.execute('''
            SELECT * FROM snapshot_metrics
            WHERE symbol = ? AND fetched_at > ?
        ''', (symbol.upper(), cutoff))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None

def _save_metrics_to_cache(symbol: str, metrics: Dict[str, Optional[float]]):
    """Save metrics to cache table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO fundamental_metrics
            (symbol, rev_growth, eps_growth, gross_margin, fcf_yield, de_ratio, roe, ocf_yield, net_margin, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol.upper(),
            metrics.get('rev_growth'),
            metrics.get('eps_growth'),
            metrics.get('gross_margin'),
            metrics.get('fcf_yield'),
            metrics.get('de_ratio'),
            metrics.get('roe'),
            metrics.get('ocf_yield'),
            metrics.get('net_margin'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error saving metrics cache for {symbol}: {e}")

def clear_fundamentals_cache(symbol: Optional[str] = None, older_than_hours: int = 48):
    """
    Clear outdated fundamentals cache.
    
    Args:
        symbol: If provided, clear only that symbol; if None, clear all old entries
        older_than_hours: Clear entries older than this many hours (default: 48h)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        if symbol:
            cursor.execute('DELETE FROM fundamental_metrics WHERE symbol = ?', (symbol.upper(),))
        else:
            cursor.execute('DELETE FROM fundamental_metrics WHERE fetched_at < ?', (cutoff.isoformat(),))
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        if deleted > 0:
            print(f"✅ Cleared {deleted} outdated fundamentals cache entries")
        
        return deleted
    except Exception as e:
        print(f"⚠️ Error clearing cache: {e}")
        return 0

# Initialize on module load
_ensure_fundamentals_table()
