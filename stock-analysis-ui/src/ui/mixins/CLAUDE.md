# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Mixins mixed into `ui.main_window.MainWindow`. **`self` is the `MainWindow`** — these methods
use its widgets (`self.symbol_input`, `self.screener_combo`, …) and helpers (`self._status`,
`self._present_screener_results`). They are not standalone classes; don't instantiate them.

## Files

- `screeners.py` — `ScreenersMixin`. Dispatches the screener combo in `_show_yahoo_screener`:
  `_store_*` → store engine, `_fvw_*` → Finviz market-wide, `_finviz_gapper` → gapper,
  `_events_48h*` → earnings/ex-div calendar. `_country_map(symbols)` does a best-effort store
  country lookup (0 requests, N/A off-catalogue).
- `export.py` — `ExportMixin`. Result/report export.

## When editing screeners

- Respect the yfinance budget: store screeners cost 0 yfinance calls, Finviz costs 1 request;
  never add a per-symbol yfinance loop.
- finvizfinance returns `Change` as a fraction — multiply by 100 before display.
- Don't test `QProgressDialog.wasCanceled()` after `close()` (it's always True then).
- Results go to `self._present_screener_results(title, headers, rows)`, symbol in column 0.
