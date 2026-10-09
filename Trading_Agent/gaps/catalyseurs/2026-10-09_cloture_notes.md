Scan cassure (cassure_scan.py --min-hausse 5 --limit 50) echoue : import de core.finviz_screeners via le chemin absolu code en dur APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src", qui n'existe pas dans cet environnement d'execution (ModuleNotFoundError: No module named 'core').

Scan close (gap_scan.py --mode close --min-gap 5) echoue pour la meme raison : meme import core.finviz_screeners via le meme chemin absolu code en dur APP_SRC, ModuleNotFoundError: No module named 'core'.

Aucun fichier JSON de detection n'a ete produit pour les deux scripts (echec avant ecriture).
