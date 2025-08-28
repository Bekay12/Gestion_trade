# -*- coding: utf-8 -*-
"""
Indicateur RSI (Relative Strength Index).
"""
import pandas as pd
import numpy as np
import ta
from typing import Union

from .base_indicator import BaseIndicator

class RSIIndicator(BaseIndicator):
    """Indicateur RSI avec vos param�tres exacts."""
    
    def __init__(self):
        super().__init__("RSI")
        self.default_window = 17  # Votre valeur exacte
    
    def calculate(self, data: pd.Series, window: int = None, **kwargs) -> pd.Series:
        """Calcule le RSI."""
        if not self.validate_data(data, min_length=2):
            return pd.Series(dtype=float, index=data.index)
        
        window = window or self.default_window
        
        try:
            # Utiliser ta (votre m�thode exacte)
            rsi = ta.momentum.RSIIndicator(close=data, window=window).rsi()
            return rsi.fillna(50.0)  # Valeur neutre pour les NaN
        except Exception:
            # Fallback: calcul manuel
            return self._calculate_manual_rsi(data, window)
    
    def _calculate_manual_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """Calcul manuel du RSI en cas d'�chec de ta."""
        delta = data.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        avg_gains = gains.rolling(window=window, min_periods=1).mean()
        avg_losses = losses.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi.fillna(50.0)

# Instance globale
rsi_indicator = RSIIndicator()

# Fonction de compatibilit�
def calculate_rsi(prices: pd.Series, window: int = 17) -> pd.Series:
    """Fonction de compatibilit�."""
    return rsi_indicator.calculate(prices, window)
