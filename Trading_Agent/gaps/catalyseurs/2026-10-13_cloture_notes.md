Scan cassure (cassure_scan.py --min-hausse 5 --limit 50) echoue : la requete vers finviz.com (core.finviz_screeners.run_screen / finvizfinance) a ete refusee par le proxy reseau de l'environnement d'execution (CONNECT finviz.com:443 rejete avec HTTP 403, politique d'egress bloquant ce host), aucun candidat n'a pu etre recupere.

Scan close (gap_scan.py --mode close --min-gap 5) echoue pour la meme raison : meme appel core.finviz_screeners.run_screen vers finviz.com, meme rejet 403 du proxy d'egress.

Aucun fichier JSON de detection n'a ete produit pour les deux scripts (echec avant ecriture, cote reseau et non cote donnees).
