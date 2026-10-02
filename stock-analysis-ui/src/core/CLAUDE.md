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
| `store_screeners.py` | **Store-only** views (`SCREENERS`: `combined`, `dual_star`, `golden_cross`), 0 yfinance req, local catalogue only. `combined` = « Combined pur », profiles incl. `Dual Champion*` | DuckDB |
| `combined_finviz.py` | **Hybrid** « Finviz + Combined »: `run_finviz_combined(limit, progress)` runs the `dual_star` preset (1 Finviz req), then `Combined_scan.analyze_safe` per ticker (about 4 yfinance req each). `progress(i, total, symbol)` returns True to cancel. Rows sorted star first | Finviz + yfinance |
| `scan_fondamentaux.py` | Shared code of the standalone fundamental scanners (`Combined_scan`, `Sichere_Unternehmen_scan`, `Big_Growth_scan`, `AI_Implement/news_monitor_combined`): one yfinance throttle/retry layer, FX to EUR, annual statements, criteria G1-G5 / S1-S7, profile. Fix a criterion here, never in a script | yfinance |

## Screener design (don't collapse the three)

The screener forms are **intentionally separate**:
- `store_screeners.py` = catalogue-limited, zero yfinance cost. Kept only for the views Finviz
  cannot reproduce (bi-score profiles, recent golden cross).
- `finviz_screeners.py` = whole-market discovery. Presets map a strategy → a Finviz
  `filter_dict` (exact keys/options from `finvizfinance.constants.filter_dict`). The preset
  `dual_star` only emulates the core of Dual Champion* (P/E < 25, PEG < 2, quarter > +10 %,
  above SMA50, dividend > 0): Finviz filters are AND-only and cannot express "G >= 3 and
  S >= 5", and its PEG comes from analyst forecasts. Its output is a list to confirm, never a
  selection (measured 02.10.2026: 8 stars out of 53 tickers; G4 and S4 hold, G3 about 43 %).
- `combined_finviz.py` = Finviz discovers, the live Combined sorts. It is the only module that
  spends yfinance requests per ticker inside a screener, so keep it behind an explicit click
  with a cancellable progress callback, and never call it from a loop or at startup.

Both return `{title, headers, rows}` with the symbol in column 0 and the company name in
column 1, consumed by `ScreenerResultsDialog`. Name and country columns come from
`market_store.get_name_map()` / `get_country_map()` (store, best-effort, 0 network) or
natively from the Finviz response (`Company`, `Country`).

`run_screen()` is the only allowed entry point into finvizfinance — it owns both the curl_cffi
session and the ticker-cell sanitizing (see the note in `../CLAUDE.md`).

The star is defined once, in `scan_fondamentaux.est_etoile(g3_ok, g4_ok, s4_ok)` (Dual Champion
that also meets G3, G4 and S4); `DUAL_ETOILE` is its profile label. `store_screeners` and
`combined_finviz` both flag it through the `Profil` value, not a separate column. Profile order
in lists: Dual*, Dual, Pure Safe, Pure Growth, Balanced.

Keep pure-calc modules (`indicators`, `cache`, `charts`) free of network and DB calls so they
stay unit-testable offline.
