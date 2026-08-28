# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`core/` is the analysis engine being **migrated out of `../qsi.py`**, module by module. Each
file's docstring lists the `qsi.py` functions still pending migration. When you move a
function here, keep `qsi.py` re-exporting it so existing `from qsi import …` callers don't
break.

## Modules

| Module | Responsibility | Network/DB? |
|---|---|---|
| `indicators.py` | Pure technical indicators (MACD, etc.) | none |
| `cache.py` | In-process LRU caches (`_BoundedCache`, `DERIV_CACHE`, `TA_CACHE`) to memoize indicator calcs | none |
| `io.py` | Signal I/O — evolutive CSV save, cached data helpers | disk |
| `fx.py` | Currency normalization + FX daily-rate cache | store |
| `symbols.py` | Symbol metadata (sector, market cap, consensus) | store |
| `signals.py` | Trading-signal generation + optimal-parameter extraction | store |
| `analysis.py` | High-level multi-symbol orchestration (analyse & display) | yfinance/store |
| `charts.py` | Unified price/volume/signal matplotlib charts | none |
| `finviz_screeners.py` | **Market-wide** Finviz presets (`PRESETS`, `run_preset`) — 1 Finviz req, discovers tickers across the whole US market | Finviz |
| `store_screeners.py` | **Store-only** views (`SCREENERS`: combined, golden_cross) — 0 yfinance req, local catalogue only | DuckDB |

## Screener design (don't collapse the two)

The two screener families are **intentionally separate**:
- `store_screeners.py` = catalogue-limited, zero yfinance cost. Kept only for the views Finviz
  cannot reproduce (bi-score profiles, recent golden cross).
- `finviz_screeners.py` = whole-market discovery. Presets map a strategy → a Finviz
  `filter_dict` (exact keys/options from `finvizfinance.constants.filter_dict`).

Both return `{title, headers, rows}` with the symbol in column 0 and the company name in
column 1, consumed by `ScreenerResultsDialog`. Name and country columns come from
`market_store.get_name_map()` / `get_country_map()` (store, best-effort, 0 network) or
natively from the Finviz response (`Company`, `Country`).

`run_screen()` is the only allowed entry point into finvizfinance — it owns both the curl_cffi
session and the ticker-cell sanitizing (see the note in `../CLAUDE.md`).

Keep pure-calc modules (`indicators`, `cache`, `charts`) free of network and DB calls so they
stay unit-testable offline.
