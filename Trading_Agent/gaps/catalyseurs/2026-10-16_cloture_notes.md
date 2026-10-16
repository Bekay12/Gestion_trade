# Scan cloture 2026-10-16 — echec

Les deux scans (`cassure_scan.py --min-hausse 5 --limit 50` et `gap_scan.py
--mode close --min-gap 5`) ont echoue avant toute requete reseau ou analyse
de marche.

Cause : `scanner_finviz()`, dans `scripts/cassure_scan.py` (ligne 94) et dans
`scripts/gap_scan.py` (ligne 293), insere dans `sys.path` la constante
`APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"`
(chemin absolu code en dur, ligne 35 / 41) avant `from core.finviz_screeners
import run_screen`. Ce chemin n'existe pas dans cet environnement
d'execution (le depot est ici sous `/home/user/Gestion_trade`), ce qui leve
`ModuleNotFoundError: No module named 'core'` dans les deux scripts.

Aucun fichier de detection n'a ete produit pour 2026-10-16. Meme blocage que
celui deja consigne dans `2026-10-16_premarket_notes.md` ce matin.
