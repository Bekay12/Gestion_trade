# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Desktop stock-analysis tool: a PyQt5 GUI over a technical/fundamental analysis engine,
a Parquet+DuckDB market-data warehouse, and market screeners. The runnable app lives in
[stock-analysis-ui/](stock-analysis-ui/); everything else at the root is data, archives, or
compiled-content (`Academy-Germain-final/`, `Archives/`, `*.csv`, `*.log`, cache dirs).

The active codebase is **[stock-analysis-ui/src/](stock-analysis-ui/src/)** — start there.

## Environment

- Python **3.10**, virtualenv at repo root: **`.venv_new`** (`.venv_new/bin/python`).
- Dependencies: [stock-analysis-ui/requirements.txt](stock-analysis-ui/requirements.txt)
  (PyQt5, pandas/numpy, yfinance, finvizfinance + curl_cffi + lxml, duckdb, matplotlib, ta).

## Commands

Run from the `stock-analysis-ui/` directory unless noted.

```bash
# Launch the GUI (chdir's into src/ itself)
python launch_stock_analysis.py

# Tests — the "safe" subset used by CI (no network, no real DB)
pytest -m "not integration"
# Full suite incl. network/DB tests (run manually)
pytest -m integration
# A single test
pytest src/tests/test_symbol_manager.py::test_name

# Build the optional C acceleration module (from repo root)
python setup.py                 # wraps trading_c_acceleration/setup.py build_ext --inplace
```

### Environment flags (set for headless / CI / offline)

| Flag | Effect |
|---|---|
| `QSI_DISABLE_C_ACCELERATION=1` | Skip the compiled `trading_c` backtest module (pure-Python path) |
| `QSI_CONSENSUS_OFFLINE=1` | Disable network consensus lookups **only** — the desktop UI sets this unconditionally at startup, so it is *not* a general "no network" switch |
| `QSI_DISABLE_PROFILE_FETCH=1` | Stop `ensure_instrument_profiles()` from topping up instrument profiles (country, name, sector, beta…) after a screener. Set by default in the test conftest |
| `QT_QPA_PLATFORM=offscreen` | Run PyQt headless (required for GUI-touching tests) |

CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs `pytest -m "not integration"`
with the first two flags set, on Python 3.10.

## Architecture (big picture)

**Entry point** → [launch_stock_analysis.py](stock-analysis-ui/launch_stock_analysis.py)
puts `src/` and `src/ui/` on `sys.path`, `chdir`s into `src/`, then opens
`ui.main_window.MainWindow`. **All relative paths assume the CWD is `src/`.**

**`MainWindow` is composed from mixins**, not a monolith:
`ui/main_window.py` + `ScreenersMixin` ([ui/mixins/screeners.py](stock-analysis-ui/src/ui/mixins/screeners.py))
+ `ExportMixin` ([ui/mixins/export.py](stock-analysis-ui/src/ui/mixins/export.py)). Add screener/
export behavior to the mixin, not the window.

**`qsi.py` is a legacy façade under active migration into `core/`.** Callers still
`from qsi import ...`, but the implementations are being moved module-by-module into
[src/core/](stock-analysis-ui/src/core/) (indicators, cache, io, fx, symbols, signals,
analysis, charts). When touching analysis logic, prefer the `core/` submodule; keep the
`qsi` re-export working.

**Two-tier market-data warehouse:**
- [market_store.py](stock-analysis-ui/src/market_store.py) — the real backend: Parquet on
  disk (`market_parquet/features/symbol=…/` Hive partitions) queried via DuckDB. Built for
  5000+ symbols / 30y. `get_latest_features()` uses `union_by_name=true` + a `QUALIFY
  row_number()` window to tolerate heterogeneous per-symbol schemas.
- [cache_db.py](stock-analysis-ui/src/cache_db.py) — a **compatibility shim** that forwards
  the old SQLite API to `market_store`. Don't add logic here; add it to `market_store`.

**Two screener families** (this distinction matters — see the memory note):
- [core/store_screeners.py](stock-analysis-ui/src/core/store_screeners.py) — reads the local
  Parquet store, **0 yfinance requests**, but limited to the local catalogue. Kept only for
  the 2 views Finviz can't reproduce (Combined profiles, Golden Cross).
- [core/finviz_screeners.py](stock-analysis-ui/src/core/finviz_screeners.py) — queries the
  **entire US market** via Finviz (1 request, 0 yfinance) to *discover* new tickers.
  Requires `lxml` (else finvizfinance crashes) and a `curl_cffi` impersonated session to
  bypass bot-blocking.

**yfinance request budget is a hard constraint.** The user hits yfinance rate limits;
screeners and features must minimize per-symbol yfinance calls and prefer the Parquet/DuckDB
store. Never add a per-symbol yfinance loop where a store read or a batched
`yf.download(chunk, group_by="ticker")` would do.

**Standalone CLI scan scripts** (`src/*_scan.py`: `Big_Growth_scan.py`,
`Sichere_Unternehmen_scan.py`, `Combined_scan.py`) fetch yfinance per symbol *and* write into
the Parquet store, producing CSVs. They are the batch counterpart to the in-app store
screeners.

**Optional online layer:** [api.py](stock-analysis-ui/src/api.py) (Flask REST) and
[background_worker.py](stock-analysis-ui/src/background_worker.py) (daily signal computation)
serve signals outside the desktop app.

## Qt gotcha (bit us before)

`QProgressDialog.close()` sets `wasCanceled()` to **True**. Never test `wasCanceled()` after
`close()` — it always returns early. See the comment in `_show_finviz_gapper_screener`.

## Conventions

Naming, docstrings, logging tags, and error handling follow the workspace rules in
`~/.claude/rules/` (French default for user-facing strings; `logging.getLogger(__name__)`
with a `[TAG]` prefix; type hints on signatures). There is **no CI/CD deploy** — run tests
manually before pushing.
