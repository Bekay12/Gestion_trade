# Scan cassure midi 2026-10-15 — echec

Le scan a echoue avec `ModuleNotFoundError: No module named 'core'` car `scripts/cassure_scan.py` (ligne 35) importe `core.finviz_screeners` depuis le chemin absolu code en dur `APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"`, qui n'existe pas dans cet environnement (le depot est ici sous `/home/user/Gestion_trade`).
