# Scan cassure midi 2026-10-14 - echec

Le scan a echoue avec `ModuleNotFoundError: No module named 'core'` car `cassure_scan.py` (ligne 35) code en dur `APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"`, un chemin qui n'existe pas dans cet environnement (le depot est clone sous `/home/user/Gestion_trade`), alors que le module `core/finviz_screeners.py` existe bien sous `/home/user/Gestion_trade/stock-analysis-ui/src`.
