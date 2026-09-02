# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scope: the **stock-analysis desktop application**. See the root
[CLAUDE.md](../CLAUDE.md) for the full architecture; this note covers running and testing
from *this* directory.

## Layout

- [launch_stock_analysis.py](launch_stock_analysis.py) — GUI entry point. Adds `src/` to the
  path, `chdir`s into `src/`, opens `ui.main_window.MainWindow`.
- [src/](src/) — all application code (see [src/CLAUDE.md](src/CLAUDE.md)).
- [requirements.txt](requirements.txt) — pinned deps; installed into the root `.venv_new`.
- [pytest.ini](pytest.ini) — `testpaths = src/tests`; defines the `integration` marker.
- `data_cache/`, `cache_logs/` — runtime artifacts, not source.

## Commands (run from here)

```bash
python launch_stock_analysis.py          # launch the app
pytest -m "not integration"              # CI-safe subset (no network, no real DB)
pytest -m integration                    # network/DB tests, run manually
pytest src/tests/test_price_features.py  # a single file
```

For headless/GUI-touching tests set `QT_QPA_PLATFORM=offscreen QSI_DISABLE_C_ACCELERATION=1
QSI_CONSENSUS_OFFLINE=1`.

## Tests

The `integration` marker means a test **hits the network (yfinance) and/or mutates a real
local SQLite DB** — excluded by default. Anything running in CI must stay off the network and
off real state; patch yfinance and use fixtures. Test fixtures live in
[src/tests/conftest.py](src/tests/conftest.py).
