# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pytest suite (`testpaths = src/tests`, files `test_*.py`).

## Running

From `stock-analysis-ui/`:

```bash
pytest -m "not integration"                          # default CI-safe subset
pytest -m integration                                # network/DB tests, manual only
pytest src/tests/test_symbol_manager.py::test_name   # single test
```

## The `integration` marker

Marks a test that **hits the network (yfinance) and/or mutates a real local SQLite DB**. It is
excluded from the default run and from CI. Everything unmarked must run offline against
fixtures/patches — no live yfinance, no real DB writes.

## conftest

[conftest.py](conftest.py) puts `src/` on `sys.path` and sets `QSI_DISABLE_C_ACCELERATION=1`
and `QSI_CONSENSUS_OFFLINE=1` (via `setdefault`) to avoid C-acceleration segfaults and
network consensus calls during import. GUI-touching tests additionally need
`QT_QPA_PLATFORM=offscreen`.

New tests default to **not** `integration`: patch yfinance and use a temp DB so the suite
stays deterministic and offline.
