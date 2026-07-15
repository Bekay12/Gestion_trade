"""
Génération de signaux de trading et extraction des paramètres optimaux.
À migrer depuis qsi.py :
  extract_best_parameters, get_trading_signal, resolve_symbol_scoring_context,
  get_cap_range_for_symbol, BEST_PARAM_EXTRAS, PRICE_FEATURE_WINDOW.

Migration bloquée par l'état global partagé (BEST_PARAM_EXTRAS, DERIV_CACHE,
TA_CACHE) — à faire après avoir stabilisé core/cache.py comme source unique.
"""
# TODO: migrer extract_best_parameters() depuis qsi.py
# TODO: migrer get_trading_signal() depuis qsi.py
# TODO: migrer resolve_symbol_scoring_context() depuis qsi.py
