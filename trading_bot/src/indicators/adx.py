"""
Indicateur ADX (Average Directional Index).
"""
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from typing import Dict

from .base_indicator import BaseIndicator

class ADXIndicator_Custom(BaseIndicator):
    """Indicateur ADX avec vos paramètres exacts."""
    
    def __init__(self):
        super().__init__("ADX")
        self.default_window = 14  # Votre valeur exacte
        self.strong_threshold = 25.0  # Votre seuil pour tendance forte
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                 window: int = None, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule l'ADX.
        
        Args:
            high: Prix les plus hauts.
            low: Prix les plus bas.
            close: Prix de clôture.
            window: Période de calcul.
            
        Returns:
            Dict avec adx, di_plus, di_minus.
        """
        if not all(self.validate_data(series, min_length=window or self.default_window) 
                  for series in [high, low, close]):
            empty_series = pd.Series(dtype=float, index=close.index)
            return {
                'adx': empty_series,
                'di_plus': empty_series,
                'di_minus': empty_series
            }
        
        window = window or self.default_window
        
        try:
            adx_indicator = ADXIndicator(high=high, low=low, close=close, window=window)
            
            return {
                'adx': adx_indicator.adx(),
                'di_plus': adx_indicator.adx_pos(),
                'di_minus': adx_indicator.adx_neg()
            }
            
        except Exception:
            # Fallback simple
            empty_series = pd.Series(dtype=float, index=close.index)
            return {
                'adx': empty_series,
                'di_plus': empty_series,
                'di_minus': empty_series
            }
    
    def get_trend_strength(self, adx_value: float) -> str:
        """Détermine la force de la tendance."""
        if adx_value >= self.strong_threshold:
            return "STRONG"
        elif adx_value >= 20:
            return "MODERATE"
        else:
            return "WEAK"

# Instance globale
adx_indicator = ADXIndicator_Custom()
