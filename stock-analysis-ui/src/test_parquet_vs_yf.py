#!/usr/bin/env python3
"""
Test de cohérence : Parquet store vs yfinance en direct.

Compare les prix OHLCV et les données fondamentales des 7 derniers jours
pour un ou plusieurs symboles.

Usage
-----
  python test_parquet_vs_yf.py                  # teste AAPL, MSFT, NVDA
  python test_parquet_vs_yf.py AAPL TSLA ENR.DE # symboles personnalisés
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ajoute le répertoire src au path si le script est lancé depuis un sous-dossier
_SRC = Path(__file__).parent.resolve()
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import market_store as ms

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA"]
TOLERANCE_PCT   = 0.01   # 0.01 % de tolérance sur les prix (arrondi float)
WEEK_AGO        = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
TODAY           = datetime.utcnow().strftime("%Y-%m-%d")

OK   = "OK  "
FAIL = "FAIL"
WARN = "WARN"
SEP  = "─" * 70


def _pct_diff(a, b) -> float | None:
    """Différence relative en %. Retourne None si l'un des deux est None/NaN."""
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return None
        return abs(fa - fb) / abs(fb) * 100.0
    except (TypeError, ValueError):
        return None


def _status(pct: float | None) -> str:
    if pct is None:
        return WARN
    return OK if pct <= TOLERANCE_PCT else FAIL


# ──────────────────────────────────────────────
# 1. Test prix OHLCV (table prices/)
# ──────────────────────────────────────────────
def test_prices(symbol: str) -> dict:
    print(f"\n{'─'*10} PRIX  {symbol} {'─'*10}")
    results = {"symbol": symbol, "section": "prices", "rows_checked": 0, "failures": []}

    # Lecture Parquet
    price_path = ms._parquet_path("prices", symbol)
    if not price_path.exists():
        print(f"  {WARN} Pas de fichier Parquet pour {symbol} — lancer refresh_symbol_incremental d'abord.")
        results["failures"].append("parquet_missing")
        return results

    pq_df = pd.read_parquet(price_path)
    pq_df["trade_date"] = pd.to_datetime(pq_df["trade_date"])
    pq_week = pq_df[pq_df["trade_date"] >= WEEK_AGO].sort_values("trade_date")

    if pq_week.empty:
        print(f"  {WARN} Aucune donnée dans les 7 derniers jours en Parquet.")
        results["failures"].append("no_recent_data_parquet")
        return results

    # Lecture yfinance directe
    ticker = yf.Ticker(symbol)
    yf_hist = ticker.history(start=WEEK_AGO, auto_adjust=False, actions=False)
    if yf_hist is None or yf_hist.empty:
        print(f"  {WARN} yfinance n'a pas retourné de données pour {symbol}.")
        results["failures"].append("no_yf_data")
        return results

    yf_hist = yf_hist.reset_index()
    yf_hist["Date"] = pd.to_datetime(yf_hist["Date"], utc=True).dt.tz_localize(None)
    yf_hist["trade_date_str"] = yf_hist["Date"].dt.strftime("%Y-%m-%d")
    pq_week["trade_date_str"] = pq_week["trade_date"].dt.strftime("%Y-%m-%d")

    common_dates = set(pq_week["trade_date_str"]) & set(yf_hist["trade_date_str"])
    if not common_dates:
        print(f"  {WARN} Aucune date commune entre Parquet et yfinance.")
        results["failures"].append("no_common_dates")
        return results

    cols_to_check = [("close", "Close"), ("open", "Open"), ("high", "High"),
                     ("low", "Low"), ("volume", "Volume")]
    row_ok = row_fail = 0

    for date_str in sorted(common_dates):
        pq_row = pq_week[pq_week["trade_date_str"] == date_str].iloc[0]
        yf_row = yf_hist[yf_hist["trade_date_str"] == date_str].iloc[0]

        line_parts = [f"  {date_str}"]
        date_ok = True
        for pq_col, yf_col in cols_to_check:
            pq_val = pq_row.get(pq_col)
            yf_val = yf_row.get(yf_col)
            diff   = _pct_diff(pq_val, yf_val)
            status = _status(diff)
            if status == FAIL:
                date_ok = False
                results["failures"].append(f"{date_str}.{pq_col}")
            diff_str = f"{diff:.4f}%" if diff is not None else "N/A"
            line_parts.append(f"{pq_col}={pq_val:.2f}/yf={yf_val:.2f}({status},{diff_str})")

        print("  |  ".join(line_parts))
        if date_ok:
            row_ok += 1
        else:
            row_fail += 1

    results["rows_checked"] = len(common_dates)
    print(f"  => {row_ok}/{len(common_dates)} jours OK  |  {row_fail} FAIL")
    return results


# ──────────────────────────────────────────────
# 2. Test fondamentaux (table fundamentals/)
# ──────────────────────────────────────────────
def test_fundamentals(symbol: str) -> dict:
    print(f"\n{'─'*10} FONDAMENTAUX  {symbol} {'─'*10}")
    results = {"symbol": symbol, "section": "fundamentals", "failures": []}

    fund_path = ms._parquet_path("fundamentals", symbol)
    if not fund_path.exists():
        print(f"  {WARN} Pas de snapshot fondamental en Parquet.")
        results["failures"].append("parquet_missing")
        return results

    pq_df  = pd.read_parquet(fund_path).sort_values("as_of_date")
    latest = pq_df.iloc[-1].to_dict()
    pq_date = latest.get("as_of_date", "?")

    # Données yfinance directes
    info = yf.Ticker(symbol).info or {}

    fields = [
        ("trailing_pe",     "trailingPE",       5.0),   # ±5%
        ("forward_pe",      "forwardPE",        5.0),
        ("price_to_book",   "priceToBook",      5.0),
        ("gross_margins",   "grossMargins",     1.0),
        ("profit_margins",  "profitMargins",    1.0),
        ("beta",            "beta",             2.0),
        ("dividend_yield",  "dividendYield",    1.0),
    ]

    any_fail = False
    print(f"  Snapshot Parquet du : {pq_date}")
    for pq_col, yf_key, tol in fields:
        pq_val = latest.get(pq_col)
        yf_val = info.get(yf_key)
        if pq_val is None and yf_val is None:
            print(f"  {WARN}  {pq_col:22s}  pq=None  yf=None  (tous deux absents)")
            continue
        diff   = _pct_diff(pq_val, yf_val)
        status = OK if (diff is not None and diff <= tol) else (WARN if diff is None else FAIL)
        if status == FAIL:
            any_fail = True
            results["failures"].append(pq_col)
        diff_str = f"{diff:.2f}%" if diff is not None else "N/A"
        print(f"  {status}  {pq_col:22s}  pq={pq_val}  yf={yf_val}  diff={diff_str}")

    if any_fail:
        print("  => Des écarts dépassent la tolérance (les ratios yf varient intra-day).")
    else:
        print("  => Fondamentaux cohérents avec yfinance.")
    return results


# ──────────────────────────────────────────────
# 3. Test instruments (table instruments.parquet)
# ──────────────────────────────────────────────
def test_instruments(symbol: str) -> dict:
    print(f"\n{'─'*10} INSTRUMENTS  {symbol} {'─'*10}")
    results = {"symbol": symbol, "section": "instruments", "failures": []}

    inst_path = ms._instruments_path()
    if not inst_path.exists():
        print(f"  {WARN} Fichier instruments.parquet absent.")
        results["failures"].append("parquet_missing")
        return results

    df  = pd.read_parquet(inst_path)
    row = df[df["symbol"] == ms._normalize_symbol(symbol)]
    if row.empty:
        print(f"  {WARN} Symbole absent de instruments.parquet.")
        results["failures"].append("symbol_missing")
        return results

    pq = row.iloc[0].to_dict()
    info = yf.Ticker(symbol).info or {}

    fields = [
        ("sector",    "sector"),
        ("currency",  "currency"),
        ("exchange",  "exchange"),
    ]
    for pq_col, yf_key in fields:
        pq_val = str(pq.get(pq_col) or "").strip()
        yf_val = str(info.get(yf_key) or "").strip()
        match  = pq_val.lower() == yf_val.lower()
        status = OK if match else WARN
        if not match:
            results["failures"].append(pq_col)
        print(f"  {status}  {pq_col:12s}  pq='{pq_val}'  yf='{yf_val}'")

    return results


# ──────────────────────────────────────────────
# Bootstrap : ingère les symboles manquants
# ──────────────────────────────────────────────
def bootstrap_if_needed(symbols: list[str]) -> None:
    missing = [
        sym for sym in symbols
        if not ms._parquet_path("prices", sym).exists()
    ]
    if not missing:
        return
    print(f"\n  [BOOTSTRAP] Ingestion Parquet pour : {missing}")
    print("  (téléchargement 36 mois d'historique depuis yfinance…)\n")
    for sym in missing:
        try:
            result = ms.refresh_symbol_incremental(sym, force=True, fast_bootstrap=True)
            status = result.get("status", "?")
            rows_p = result.get("price_rows", 0)
            rows_f = result.get("feature_rows", 0)
            print(f"  {sym:12s}  status={status}  prix={rows_p} lignes  features={rows_f} lignes")
        except Exception as exc:
            print(f"  {sym:12s}  ERREUR bootstrap : {exc}")
    print()


# ──────────────────────────────────────────────
# Runner principal
# ──────────────────────────────────────────────
def run_all(symbols: list[str]) -> None:
    print(SEP)
    print(f"  TEST PARQUET vs YFINANCE — période {WEEK_AGO} → {TODAY}")
    print(f"  Symboles : {symbols}")
    print(f"  Store Parquet : {ms.PARQUET_DIR}")
    print(SEP)

    bootstrap_if_needed(symbols)

    all_failures: list[str] = []

    for sym in symbols:
        sym = ms._normalize_symbol(sym)
        r1 = test_prices(sym)
        r2 = test_fundamentals(sym)
        r3 = test_instruments(sym)
        for r in (r1, r2, r3):
            for f in r.get("failures", []):
                all_failures.append(f"{sym}/{r['section']}/{f}")

    print(f"\n{SEP}")
    if not all_failures:
        print("  RÉSULTAT GLOBAL : TOUT OK — Parquet cohérent avec yfinance.")
    else:
        print(f"  RÉSULTAT GLOBAL : {len(all_failures)} problème(s) détecté(s) :")
        for f in all_failures:
            print(f"    - {f}")
    print(SEP)


if __name__ == "__main__":
    syms = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    run_all(syms)
