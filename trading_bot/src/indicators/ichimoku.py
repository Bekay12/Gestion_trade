"""
Indicateur Ichimoku Cloud.
"""
import pandas as pd
import numpy as np
from ta.trend import IchimokuIndicator
from typing import Dict

from .base_indicator import BaseIndicator

class IchimokuIndicator_Custom(BaseIndicator):
    """Indicateur Ichimoku avec vos paramètres exacts."""
    
    def __init__(self):
        super().__init__("Ichimoku")
        self.default_conversion = 9   # Votre valeur exacte
        self.default_base = 26       # Votre valeur exacte
        self.default_span = 52       # Votre valeur exacte
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 conversion_window: int = None, base_window: int = None, 
                 span_window: int = None, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule l'Ichimoku Cloud.
        
        Args:
            high: Prix les plus hauts.
            low: Prix les plus bas.
            close: Prix de clôture.
            conversion_window: Période conversion line.
            base_window: Période base line.
            span_window: Période span.
            
        Returns:
            Dict avec conversion_line, base_line, span_a, span_b.
        """
        if not all(self.validate_data(series, min_length=52) 
                  for series in [high, low, close]):
            empty_series = pd.Series(dtype=float, index=close.index)
            return {
                'conversion_line': empty_series,
                'base_line': empty_series,
                'span_a': empty_series,
                'span_b': empty_series
            }
        
        conversion_window = conversion_window or self.default_conversion
        base_window = base_window or self.default_base
        span_window = span_window or self.default_span
        
        try:
            ichimoku = IchimokuIndicator(
                high=high, low=low, 
                window1=conversion_window, 
                window2=base_window, 
                window3=span_window
            )
            
            return {
                'conversion_line': ichimoku.ichimoku_conversion_line(),
                'base_line': ichimoku.ichimoku_base_line(),
                'span_a': ichimoku.ichimoku_a(),
                'span_b': ichimoku.ichimoku_b()
            }
            
        except Exception:
            # Fallback simple
            empty_series = pd.Series(dtype=float, index=close.index)
            return {
                'conversion_line': empty_series,
                'base_line': empty_series,
                'span_a': empty_series,
                'span_b': empty_series
            }

# Instance globale
ichimoku_indicator = IchimokuIndicator_Custom()
