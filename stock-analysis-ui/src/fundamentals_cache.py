"""
Fundamentals extraction and caching for trading system.
Provides yfinance-backed fetching with SQLite caching and TTL (24h default).
"""

import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

DB_PATH = 'stock_analysis.db'
FUNDAMENTALS_CACHE_TTL_HOURS = 24

def _ensure_fundamentals_table():
    """Create fundamentals cache table if not exists (idempotent)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                rev_growth REAL,           -- Revenue growth % (YoY)
                eps_growth REAL,           -- EPS growth % (YoY)
                gross_margin REAL,        -- Gross margin %
                fcf_yield REAL,           -- Free cash flow yield %
                de_ratio REAL,            -- Debt-to-equity ratio
                roe REAL,                 -- Return on equity %
                ocf_yield REAL,           -- Operating cash flow yield %
                net_margin REAL,          -- Net profit margin %
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_symbol UNIQUE (symbol)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_fetched ON fundamental_metrics(symbol, fetched_at DESC)')
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
    """Fetch fundamentals from yfinance."""
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
        
        # 1. Gross Margin (%)
        if 'grossMargins' in info:
            try:
                margins = info['grossMargins']
                if margins and len(margins) > 0:
                    metrics['gross_margin'] = float(margins[0]) * 100
            except (TypeError, ValueError):
                pass
        
        # 2. Net Profit Margin (%)
        if 'profitMargins' in info:
            try:
                metrics['net_margin'] = float(info['profitMargins']) * 100
            except (TypeError, ValueError):
                pass
        
        # 3. Return on Equity (%)
        if 'returnOnEquity' in info:
            try:
                metrics['roe'] = float(info['returnOnEquity']) * 100
            except (TypeError, ValueError):
                pass
        
        # 4. Debt-to-Equity Ratio
        if 'debtToEquity' in info:
            try:
                metrics['de_ratio'] = float(info['debtToEquity'])
            except (TypeError, ValueError):
                pass
        
        # 5. Revenue Growth (YoY from quarterly financials)
        try:
            financials = ticker.quarterly_financials
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue']
                if len(revenues) >= lookback_quarters:
                    # Most recent quarter
                    rev_now = float(revenues.iloc[0])
                    # Quarter 4 periods ago
                    rev_then = float(revenues.iloc[lookback_quarters - 1])
                    if rev_then != 0:
                        growth = ((rev_now - rev_then) / rev_then) * 100
                        metrics['rev_growth'] = growth
        except Exception:
            pass
        
        # 6. EPS Growth (YoY from info)
        try:
            if 'trailingEps' in info and 'epsTrailingTwelveMonths' in info:
                eps_now = float(info['trailingEps'])
                eps_then = float(info['epsTrailingTwelveMonths'])
                if eps_then != 0:
                    growth = ((eps_now - eps_then) / eps_then) * 100
                    metrics['eps_growth'] = growth
        except Exception:
            pass
        
        # 7. Free Cash Flow Yield (%)
        try:
            cashflow = ticker.quarterly_cashflow
            if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                fcf = float(cashflow.loc['Free Cash Flow'].iloc[0])
                market_cap = float(info.get('marketCap', 1))
                if market_cap > 0:
                    fcf_yield = (fcf / market_cap) * 100
                    metrics['fcf_yield'] = fcf_yield
        except Exception:
            pass
        
        # 8. Operating Cash Flow Yield (%)
        try:
            cashflow = ticker.quarterly_cashflow
            if not cashflow.empty and 'Operating Cash Flow' in cashflow.index:
                ocf = float(cashflow.loc['Operating Cash Flow'].iloc[0])
                market_cap = float(info.get('marketCap', 1))
                if market_cap > 0:
                    ocf_yield = (ocf / market_cap) * 100
                    metrics['ocf_yield'] = ocf_yield
        except Exception:
            pass
    
    except Exception as e:
        print(f"⚠️ Error fetching yfinance data for {symbol}: {e}")
    
    return metrics

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
