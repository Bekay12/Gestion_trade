#!/usr/bin/env python3
"""Bootstrap the SQLite market warehouse for a seed list of symbols."""

from __future__ import annotations

import argparse

from cache_db import (
    DEFAULT_BOOTSTRAP_SYMBOLS,
    DEFAULT_FEATURE_START_DATE,
    bootstrap_market_database,
    get_symbol_storage_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap market warehouse from yfinance")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_BOOTSTRAP_SYMBOLS),
        help="Symbols to ingest. Default: AAPL MSFT NVDA TSM 02020.HK ENR.DE",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_FEATURE_START_DATE,
        help="Historical start date for daily series. Default: 2016-01-01",
    )
    args = parser.parse_args()

    results = bootstrap_market_database(symbols=args.symbols, start_date=args.start_date)
    print("Market database bootstrap complete")
    for result in results:
        summary = get_symbol_storage_summary(result["symbol"])
        print(
            f"{summary['symbol']}: {summary['price_rows']} price rows, "
            f"{summary['feature_rows']} feature rows, "
            f"{summary['start_date']} -> {summary['end_date']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())