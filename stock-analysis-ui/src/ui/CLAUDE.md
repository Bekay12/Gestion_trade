# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PyQt5 GUI layer.

## Structure

- `main_window.py` — `MainWindow`, the single top-level window. It is **composed from
  mixins** (`ScreenersMixin`, `ExportMixin` from [mixins/](mixins/)), so its methods are
  spread across those files. `_SCREENER_LABELS` defines the screener combo entries.
- `mixins/` — feature groups mixed into `MainWindow` (see [mixins/CLAUDE.md](mixins/CLAUDE.md)).
- `dialogs.py` — `ScreenerResultsDialog` (interactive triable table + checkboxes; injects
  checked symbols into the analysis field) and other dialogs.
- `workers.py` — QThread/worker plumbing for off-UI-thread work.

## Rules

- Add screener/export behavior to the relevant **mixin**, not to `main_window.py`.
- Any screener result feeds `_present_screener_results(title, headers, rows)` with the symbol
  in column 0 and the company name in column 1 (header exactly `"Nom"` — the dialog truncates
  that column and moves the full text to a tooltip).
- **`merged_table` columns are addressed by key, never by index**: `MERGED_COLUMNS` /
  `MERGED_COL['clé']` in `main_window.py` is the single source of the layout, and the
  multi-criteria comparison table derives its columns from it. Adding a column means adding a
  tuple there, nothing else.
- **Numeric cells are `CelluleNumerique`, not `QTableWidgetItem`.** It displays
  `formater_nombre()` (≤ `DECIMALES_MAX` decimals, no scientific notation) and keeps the real
  value in `.valeur`. Never sort or recompute from the cell text — it is rounded; read
  `valeur_cellule(item)`. Setting a number on `Qt.EditRole` does **not** work here:
  `QTableWidgetItem` folds `EditRole` into the display role, which is what made every numeric
  column sort lexicographically (`10.2` before `9.5`).
- **`QProgressDialog.close()` sets `wasCanceled()` to True** — never branch on
  `wasCanceled()` after `close()`, or the handler always returns before displaying results.
  (This silently broke the Finviz gapper once.)
- Long work must go through a worker / `ThreadPoolExecutor` with a timeout, not block the UI
  thread. Wrap `setOverrideCursor` in try/finally so the cursor is always restored.
