# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Application source root. **The app runs with CWD = this directory** (the launcher `chdir`s
here), so every relative path resolves from `src/`.

## Where things live

| Path | Role |
|---|---|
| [ui/](ui/) | PyQt5 GUI — `main_window.py` (composed from `mixins/`), `dialogs.py`, `workers.py` |
| [core/](core/) | Analysis engine being migrated out of `qsi.py` (indicators, signals, screeners, fx, io…) |
| `qsi.py` | **Legacy façade** re-exporting `core/`; callers import from here. Migrate logic *into* `core/`, keep the re-export. |
| `market_store.py` | Parquet+DuckDB warehouse (real backend). `market_parquet/features/symbol=…/` partitions. |
| `cache_db.py` | Re-exports public API from `market_store`. No implementation. Zero DB connections opened on import. |
| `config.py` | Central paths/constants — import cache dirs & globals from here, don't hardcode. |
| `symbol_manager.py`, `fundamentals_cache.py`, `timeline_cache.py`, `sector_normalizer.py` | Symbol catalogue, fundamentals cache (TTL, evolutive quarters), timelines, sector name normalization |
| `*_scan.py` (`Big_Growth_scan`, `Sichere_Unternehmen_scan`, `Combined_scan`) | Standalone CLI batch screeners: fetch yfinance per symbol → write Parquet store → CSV |
| `api.py`, `background_worker.py` | Optional Flask REST + daily-signal worker (online layer) |
| `pdf_generator.py` | reportlab PDF report generation |
| `trading_c_acceleration/` | Compiled C backtest module; bypassed with `QSI_DISABLE_C_ACCELERATION=1` |
| `market_parquet/`, `cache*/`, `data_cache/`, `Results/`, `signaux/` | Data & output artifacts — **not source** |

## Rules that bite

- **yfinance rate limits are real.** Prefer `market_store` reads or a single batched
  `yf.download(chunk, group_by="ticker", threads=True)` (MultiIndex when
  `columns.nlevels > 1`). Never loop yfinance per symbol when the store already has the data.
- **finvizfinance** needs `lxml` installed and a `curl_cffi` `Session(impersonate="chrome")`
  swapped into `finvizfinance.util.session`, or it crashes / gets bot-blocked. Its `Change`
  field is a **fraction** (0.4776 = 47.76%) — multiply by 100 for display.
- `qsi.py` sets a non-interactive matplotlib backend only if none is set — don't force `Agg`
  after the Qt app has initialized a GUI backend.
