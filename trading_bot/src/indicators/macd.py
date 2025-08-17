"""
Indicateur MACD (Moving Average Convergence Divergence).
Migration de votre fonction calculate_macd().
"""
import pandas as pd
import numpy as np
from typing import Tuple

from config.settings import config

class MACDIndicator:
    """
    Indicateur MACD.
    Migration exacte de votre fonction calculate_macd().
    """

    def __init__(self, fast: int = None, slow: int = None, signal: int = None):
        """
        Initialize MACD indicator.

        Args:
            fast: Période EMA rapide (défaut: 12 de votre config).
            slow: Période EMA lente (défaut: 26 de votre config).
            signal: Période ligne de signal (défaut: 9 de votre config).
        """
        self.fast = fast or config.indicators.macd_fast
        self.slow = slow or config.indicators.macd_slow  
        self.signal_period = signal or config.indicators.macd_signal

    def calculate(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule le MACD et sa ligne de signal.
        Migration exacte de votre fonction calculate_macd().

        Args:
            prices: Série des prix de clôture.

        Returns:
            Tuple (macd, signal_line).
        """
        # Migration exacte de votre code
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()

        return macd, signal_line

    def get_latest_values(self, prices: pd.Series) -> Tuple[float, float, float, float]:
        """
        Retourne les dernières valeurs MACD pour l'analyse.
        Utile pour vos conditions de trading.

        Returns:
            Tuple (last_macd, prev_macd, last_signal, prev_signal).
        """
        macd, signal_line = self.calculate(prices)

        if len(macd) < 2:
            return 0.0, 0.0, 0.0, 0.0

        last_macd = float(macd.iloc[-1])
        prev_macd = float(macd.iloc[-2])
        last_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])

        return last_macd, prev_macd, last_signal, prev_signal

    def is_bullish_crossover(self, prices: pd.Series) -> bool:
        """Détecte un croisement haussier MACD."""
        last_macd, prev_macd, last_signal, prev_signal = self.get_latest_values(prices)
        return prev_macd < prev_signal and last_macd > last_signal

    def is_bearish_crossover(self, prices: pd.Series) -> bool:
        """Détecte un croisement baissier MACD."""
        last_macd, prev_macd, last_signal, prev_signal = self.get_latest_values(prices)
        return prev_macd > prev_signal and last_macd < last_signal

# Fonction de compatibilité (pour faciliter la migration)
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Fonction de compatibilité avec votre code existant.
    Migration directe de votre calculate_macd().
    """
    indicator = MACDIndicator(fast, slow, signal)
    return indicator.calculate(prices)
