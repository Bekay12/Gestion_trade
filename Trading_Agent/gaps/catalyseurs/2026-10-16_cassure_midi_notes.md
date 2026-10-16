# Scan cassure midi — 2026-10-16

Echec : `cassure_scan.py` n'a pas pu s'executer. L'appel a Finviz (finvizfinance -> curl_cffi) a echoue avec `CurlError: CONNECT tunnel failed, response 403` — le proxy reseau de l'environnement d'execution bloque le domaine finviz.com, donc aucun candidat n'a pu etre recupere.
