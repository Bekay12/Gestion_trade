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
| `market_store.py` | Parquet+DuckDB warehouse (real backend). `market_parquet/features/symbol=…/` **and** `instruments/symbol=…/` partitions — one file per symbol, never a shared file (a shared one silently lost profiles across processes). |
| `cache_db.py` | Re-exports public API from `market_store`. No implementation. Zero DB connections opened on import. |
| `config.py` | Central paths/constants — import cache dirs & globals from here, don't hardcode. |
| `symbol_manager.py`, `fundamentals_cache.py`, `timeline_cache.py`, `sector_normalizer.py` | Symbol catalogue, fundamentals cache (TTL, evolutive quarters), timelines, sector name normalization |
| `*_scan.py` (`Big_Growth_scan`, `Sichere_Unternehmen_scan`, `Combined_scan`, `Valley_scan`) | Standalone CLI batch screeners: fetch yfinance per symbol → write Parquet store → CSV. The first three share `core/scan_fondamentaux.py` (criteria G1-G5 / S1-S7, FX, throttle): edit a criterion there, not in the script |
| `Combined_backtest.py`, `Valley_backtest.py` | Point-in-time backtests of the Combined profiles (incl. Dual Champion*) and of the Valley signals. Read the local store (0 yfinance request except one grouped index download). Outputs in `backtests/`, see [backtests/README.md](backtests/README.md) |
| `api.py`, `background_worker.py` | Optional Flask REST + daily-signal worker (online layer) |
| `pdf_generator.py` | reportlab PDF report generation |
| `trading_c_acceleration/` | Compiled C backtest module; bypassed with `QSI_DISABLE_C_ACCELERATION=1` |
| `backtests/` | Backtest reports and CSVs (tracked, except `*.parquet` caches) |
| `market_parquet/`, `cache*/`, `data_cache/`, `Results/`, `signaux/` | Data & output artifacts, **not source**; `Results/Screeners/` receives one timestamped CSV per displayed screener result |

## Rules that bite

- **yfinance rate limits are real.** Prefer `market_store` reads or a single batched
  `yf.download(chunk, group_by="ticker", threads=True)` (MultiIndex when
  `columns.nlevels > 1`). Never loop yfinance per symbol when the store already has the data.
- **finvizfinance** needs `lxml` installed and a `curl_cffi` `Session(impersonate="chrome")`
  swapped into `finvizfinance.util.session`, or it crashes / gets bot-blocked. Its `Change`
  field is a **fraction** (0.4776 = 47.76%) — multiply by 100 for display.
- **Never call finvizfinance directly — go through `core.finviz_screeners.run_screen()`.**
  finvizfinance builds each cell from `td.text`, and Finviz's ticker cell holds a letter
  avatar before the symbol link, so a raw `Overview()` returns `IIESC` for `IESC`. `run_screen`
  reads `data-boxover-ticker` (and strips `a.company-ticker`) and carries the curl_cffi session.
- **Instrument profiles need a real name.** yfinance answers a nonexistent ticker with a
  non-empty `info` that has no `shortName`/`longName` (sometimes a numeric-named "YHD" fund),
  so `ensure_instrument_profiles()` gates on `_info_sans_identite()`. Without it, ghost
  profiles land in the store and count as *fresh*, so they are never refreshed.
- `qsi.py` sets a non-interactive matplotlib backend only if none is set — don't force `Agg`
  after the Qt app has initialized a GUI backend.
- **Do not copy a G or S criterion into a new script.** Import `core.scan_fondamentaux`.
  `market_store.py` still carries an older copy for the store columns (S5-S7 quarterly, S1 at a
  fixed 1.08 USD rate, G3 often 0): treat the store's `secure_score` and `c1..c5` as
  indicative, and confirm with the live `Combined_scan.py`.
- **A backtest result is only as good as its point-in-time discipline.** Never feed a statement
  published after T, and cite the dates T it covers (see `backtests/README.md`).
