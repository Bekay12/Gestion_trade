# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Currently an **empty namespace package** — `__init__.py` is intentionally blank and there are
no modules yet. It exists as the home for cross-cutting helpers not tied to `core/` (analysis)
or `ui/` (GUI).

When adding a utility here: keep it dependency-light and side-effect-free (no network, no DB,
no Qt) so `core/` and `ui/` can both import it, and follow the workspace naming/docstring
rules (`snake_case`, type hints on signatures, module docstring stating purpose).
