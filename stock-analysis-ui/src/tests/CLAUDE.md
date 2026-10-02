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

## Scanner and backtest tests (all offline)

| File | Covers |
|---|---|
| `test_scanner_criteres.py` | Criteria G1-G5 / S1-S7, dividend suspension, compounded growth, the star, criteria shared by the three scanners |
| `test_scanner_devises.py` | FX to EUR, retargeted at `core.scan_fondamentaux` and `news_monitor_combined` |
| `test_combined_backtest.py` | Point-in-time bricks: no statement published after T, dividends recovered from adjusted prices, currency factor, refusal of an unreached horizon, split jumps |
| `test_valley_scan.py`, `test_valley_backtest.py` | Valley signals and their point-in-time replay |
| `test_valley_screener.py`, `test_valley_screener_ui.py` | Valley screener output contract, signal filtering, progress callback, and the combo dispatch |
| `test_combined_finviz.py` | « Finviz + Combined »: star first, no extra ⭐ column, cancel keeps what was analysed, a ticker the Combined rejects is skipped |
| `test_screener_archive.py` | CSV archive of displayed results, written even when the dialog is cancelled; « Combined pur » keeps no ⭐ column |

Rule for these: patch `run_preset`, `Combined_scan.analyze_safe` and any yfinance call; a
screener or backtest test must never reach Finviz or Yahoo.
