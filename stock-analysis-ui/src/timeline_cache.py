import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TimelineCache:
    def __init__(self, db_path: str = "stock_analysis.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialise les tables pour les données temporelles (Point-In-Time)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table des Earnings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_earnings (
                    symbol TEXT,
                    date TEXT,
                    eps_estimate REAL,
                    eps_actual REAL,
                    surprise_pct REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Table des Recommandations Analystes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_recommendations (
                    symbol TEXT,
                    date TEXT,
                    firm TEXT,
                    to_grade TEXT,
                    from_grade TEXT,
                    action TEXT,
                    PRIMARY KEY (symbol, date, firm)
                )
            """)
            
            # Table des Transactions Insider
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_insider (
                    symbol TEXT,
                    date TEXT,
                    shares INTEGER,
                    value REAL,
                    transaction_text TEXT,
                    PRIMARY KEY (symbol, date, transaction_text)
                )
            """)
            conn.commit()

    def update_timeline_data(self, symbol: str) -> None:
        """Télécharge et met à jour les données temporelles depuis yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Earnings
                earnings = ticker.earnings_dates
                if earnings is not None and not earnings.empty:
                    df_earn = earnings.reset_index()
                    df_earn = df_earn.rename(columns={
                        'Earnings Date': 'date', 
                        'EPS Estimate': 'eps_estimate', 
                        'Reported EPS': 'eps_actual', 
                        'Surprise(%)': 'surprise_pct'
                    })
                    # Format date to YYYY-MM-DD
                    df_earn['date'] = df_earn['date'].dt.strftime('%Y-%m-%d')
                    df_earn['symbol'] = symbol
                    df_earn = df_earn[['symbol', 'date', 'eps_estimate', 'eps_actual', 'surprise_pct']]
                    df_earn = df_earn.dropna(subset=['date']).drop_duplicates(subset=['symbol', 'date'])
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO timeline_earnings
                        (symbol, date, eps_estimate, eps_actual, surprise_pct)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        list(df_earn.itertuples(index=False, name=None))
                    )
                
                # Analyst Recommendations
                recom = ticker.upgrades_downgrades
                if recom is not None and not recom.empty:
                    df_rec = recom.reset_index()
                    df_rec = df_rec.rename(columns={
                        'GradeDate': 'date', 'Firm': 'firm', 'ToGrade': 'to_grade', 
                        'FromGrade': 'from_grade', 'Action': 'action'
                    })
                    df_rec['date'] = df_rec['date'].dt.strftime('%Y-%m-%d')
                    df_rec['symbol'] = symbol
                    df_rec = df_rec[['symbol', 'date', 'firm', 'to_grade', 'from_grade', 'action']]
                    df_rec = df_rec.dropna(subset=['date', 'firm']).drop_duplicates(subset=['symbol', 'date', 'firm'])
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO timeline_recommendations
                        (symbol, date, firm, to_grade, from_grade, action)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        list(df_rec.itertuples(index=False, name=None))
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données temporelles pour {symbol}: {e}")

    def get_pit_timeline_data(self, symbol: str, target_date: str, lookback_days: int = 120) -> Dict[str, Any]:
        """
        Récupère les données temporelles strictes Point-In-Time (PIT) 
        connues exactement à la date `target_date`.
        """
        # Validation de la date cible (format attendu YYYY-MM-DD)
        datetime.strptime(target_date, "%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Dernier Earning connu avant ou à target_date
            cursor.execute("""
                SELECT eps_actual, eps_estimate, surprise_pct 
                FROM timeline_earnings 
                WHERE symbol = ? AND date <= ? 
                ORDER BY date DESC LIMIT 1
            """, (symbol, target_date))
            earn_res = cursor.fetchone()
            
            # Nombre d'upgrades dans les X derniers jours
            cursor.execute("""
                SELECT COUNT(*) FROM timeline_recommendations 
                                WHERE symbol = ?
                                    AND date <= ?
                                    AND date >= date(?, '-' || ? || ' day')
                                    AND lower(coalesce(action, '')) IN ('up', 'init')
                        """, (symbol, target_date, target_date, lookback_days))
            rec_res = cursor.fetchone()
            
            return {
                "latest_earnings_surprise": earn_res[2] if earn_res else 0.0,
                "recent_upgrades_count": rec_res[0] if rec_res else 0,
            }

if __name__ == "__main__":
    # Test unitaire rapide
    cache = TimelineCache()
    cache.update_timeline_data("AAPL")
    print(cache.get_pit_timeline_data("AAPL", "2023-10-01"))
