"""
cache_db - shim de compatibilite vers market_store (Parquet + DuckDB).

Ce module n'implemente rien. Il ne fait que re-exporter l'API publique de
market_store.py, afin que les imports historiques `from cache_db import ...`
continuent de fonctionner sans modification.

Le backend reel est market_store.py : Parquet partitionne par symbole, interroge
via DuckDB, dimensionne pour 5 000+ titres sur 30 ans d'historique.

Toute logique nouvelle va dans market_store.py, jamais ici. Une version
anterieure de ce fichier conservait ~1 200 lignes d'implementation SQLite
heritee entre les imports de tete et un bloc de reliaison final : ces
definitions etaient masquees deux fois et n'etaient donc jamais executees, mais
rien ne le signalait au lecteur. Modifier une fonction au milieu du fichier
n'avait aucun effet sur le comportement de l'application.

Ce module n'ouvre aucune base a l'import. L'ancienne version appelait
`ensure_market_data_schema()` au niveau module, avant la reliaison, ce qui
resolvait vers la version SQLite locale et ouvrait `stock_analysis.db` des le
simple import. market_store initialise son propre stockage a la premiere
requete ; il n'y a rien a preparer ici.
"""

from __future__ import annotations

from market_store import (  # noqa: F401  (re-export public volontaire)
    # Constantes
    DEFAULT_BOOTSTRAP_SYMBOLS,
    DEFAULT_FEATURE_START_DATE,
    DEFAULT_FX_CURRENCIES,
    PARQUET_DIR,
    # Schema et ecriture
    ensure_market_data_schema,
    upsert_instrument,
    store_price_history,
    store_fundamental_snapshot,
    store_daily_feature_series,
    fetch_and_store_symbol_series,
    bootstrap_market_database,
    refresh_symbol_incremental,
    # Lecture
    query_features,
    get_latest_feature_row,
    get_symbol_last_trade_date,
    get_symbol_storage_summary,
    # Taux de change
    ensure_fx_rates_daily_history,
    # Cache financier (remplace l'ancien stockage pickle)
    get_financial_cache,
    save_financial_cache,
    # Timeline
    get_timeline_pit_data,
    store_timeline_earnings,
    store_timeline_insider,
    store_timeline_recommendations,
    update_timeline_data,
    # Helpers internes — re-exportes car les scripts *_scan.py les importent
    _build_daily_feature_frame,
    _build_quarter_feature_points,
    _cap_range_from_market_cap_b,
    _compute_rsi,
    _get_rate_to_usd,
    _normalize_symbol,
    _prepare_history_frame,
    _safe_float,
    _safe_int,
    _safe_score_signal,
    _safe_text,
)

__all__ = [
    "DEFAULT_BOOTSTRAP_SYMBOLS",
    "DEFAULT_FEATURE_START_DATE",
    "DEFAULT_FX_CURRENCIES",
    "PARQUET_DIR",
    "ensure_market_data_schema",
    "upsert_instrument",
    "store_price_history",
    "store_fundamental_snapshot",
    "store_daily_feature_series",
    "fetch_and_store_symbol_series",
    "bootstrap_market_database",
    "refresh_symbol_incremental",
    "query_features",
    "get_latest_feature_row",
    "get_symbol_last_trade_date",
    "get_symbol_storage_summary",
    "ensure_fx_rates_daily_history",
    "get_financial_cache",
    "save_financial_cache",
    "get_timeline_pit_data",
    "store_timeline_earnings",
    "store_timeline_insider",
    "store_timeline_recommendations",
    "update_timeline_data",
    "_build_daily_feature_frame",
    "_build_quarter_feature_points",
    "_cap_range_from_market_cap_b",
    "_compute_rsi",
    "_get_rate_to_usd",
    "_normalize_symbol",
    "_prepare_history_frame",
    "_safe_float",
    "_safe_int",
    "_safe_score_signal",
    "_safe_text",
]
