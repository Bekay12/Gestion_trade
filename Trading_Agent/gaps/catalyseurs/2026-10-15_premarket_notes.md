# Scan premarket 2026-10-15 — echec

Le scan n'a produit aucun candidat : la requete vers finviz.com (screener Finviz,
`core.finviz_screeners.run_screen`) a ete refusee par le proxy reseau de cet
environnement d'execution (curl_cffi.requests.exceptions.ProxyError, `CONNECT
tunnel failed, response 403` — finviz.com n'est pas un hote autorise pour cette
session), avant toute analyse de marche ou de catalyseur.
