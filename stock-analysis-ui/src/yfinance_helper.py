"""
Helper for yfinance downloads on Render with retry and simple disk cache.
"""
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "cache"))
CACHE_DIR.mkdir(exist_ok=True)


def _create_session_with_retry(max_retries: int = 3) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=max_retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_from_cache(symbol: str, max_age_days: int = 7):
    cache_file = CACHE_DIR / f"{symbol}_data.pkl"
    if cache_file.exists():
        try:
            age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if age_days < max_age_days:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    print(f"[CACHE] {symbol} hit (age: {age_days}d)")
                    return data
        except Exception as e:
            print(f"[CACHE] read error {symbol}: {e}")
    return None


def _save_to_cache(symbol: str, data):
    try:
        cache_file = CACHE_DIR / f"{symbol}_data.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[CACHE] {symbol} saved")
    except Exception as e:
        print(f"[CACHE] save error {symbol}: {e}")


def _download_with_retry(symbol: str, period: str = "1mo"):
    cached = _get_from_cache(symbol)
    if cached is not None:
        return {symbol: cached}

    session = _create_session_with_retry(max_retries=3)
    for attempt in range(3):
        try:
            print(f"[DL] {symbol} attempt {attempt + 1}/3")
            data = yf.download(symbol, period=period, session=session, progress=False)
            if data is not None and not data.empty:
                _save_to_cache(symbol, data)
                return {symbol: data}
        except Exception as e:
            print(f"[DL] error {symbol}: {type(e).__name__}")
            if attempt < 2:
                wait = 2 ** attempt
                time.sleep(wait)
    print(f"[DL] failed {symbol}")
    return None


def download_stock_data(symbols, period: str = "1mo"):
    if isinstance(symbols, str):
        symbols = [symbols]

    result = {}
    for sym in symbols:
        data = _download_with_retry(sym, period=period)
        if data:
            result.update(data)
    return result if result else None


def downloadstockdata(symbols, period: str = "1mo"):
    # backward compatible alias
    return download_stock_data(symbols, period=period)
