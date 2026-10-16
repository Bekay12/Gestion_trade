# Scan premarket 2026-10-16 — echec

Le scan n'a produit aucun candidat : `scanner_finviz()` dans
`Trading_Agent/skills/gap-trading-germain/scripts/gap_scan.py` insere
`APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"` (chemin
absolu code en dur, ligne 41) dans `sys.path` avant `from core.finviz_screeners
import run_screen` ; ce chemin n'existe pas dans cet environnement d'execution
(le depot est ici sous `/home/user/Gestion_trade`), ce qui a leve
`ModuleNotFoundError: No module named 'core'` avant toute requete reseau ou
analyse de marche.
