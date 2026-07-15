"""
Calculs d'indicateurs techniques purs (pas de réseau, pas de DB).
"""
import pandas as pd


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal (retourne des Series)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_rsi_scalar(prices, period: int = 14) -> float:
    """Retourne la dernière valeur RSI (scalaire)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def calculate_macd_scalar(prices, fast: int = 12, slow: int = 26) -> float:
    """Retourne la dernière valeur MACD (scalaire)."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return float(macd.iloc[-1]) if not macd.empty else 0.0


def calculate_bollinger_extreme(prices, period: int = 20, num_std: int = 2) -> bool:
    """Retourne True si le dernier prix touche une bande de Bollinger."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    current = prices.iloc[-1]
    return bool(current >= upper.iloc[-1] or current <= lower.iloc[-1])
