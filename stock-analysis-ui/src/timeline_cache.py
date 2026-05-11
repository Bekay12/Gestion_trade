"""
Backend Parquet pour les données temporelles Point-In-Time (earnings, recommandations, insider).

Ce module délègue entièrement vers market_store.py (Parquet + DuckDB).
L'API publique de TimelineCache est conservée à l'identique pour ne pas
casser les imports existants.

Ancienne implémentation : SQLite (timeline_earnings, timeline_recommendations, timeline_insider).
Nouveau backend : market_parquet/timeline/<category>/symbol=<SYM>/part0.parquet
"""

import logging
from datetime import datetime
from typing import Dict, Any

import market_store as ms

logger = logging.getLogger(__name__)


class TimelineCache:
    """Wrapper de compatibilité — délègue vers market_store (Parquet)."""

    def __init__(self, db_path: str = "stock_analysis.db"):
        # db_path conservé pour compatibilité des signatures, non utilisé.
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def update_timeline_data(self, symbol: str) -> None:
        """Télécharge et stocke les données temporelles depuis yfinance (Parquet)."""
        try:
            ms.update_timeline_data(symbol)
        except Exception as e:
            logger.error(f"Erreur mise à jour timeline {symbol}: {e}")

    # ------------------------------------------------------------------
    # Lecture Point-In-Time
    # ------------------------------------------------------------------

    def get_pit_timeline_data(self, symbol: str, target_date: str, lookback_days: int = 120) -> Dict[str, Any]:
        """Retourne les données PIT connues à *target_date* (depuis Parquet)."""
        return ms.get_timeline_pit_data(symbol, target_date, lookback_days)


if __name__ == "__main__":
    cache = TimelineCache()
    cache.update_timeline_data("AAPL")
    print(cache.get_pit_timeline_data("AAPL", "2023-10-01"))
