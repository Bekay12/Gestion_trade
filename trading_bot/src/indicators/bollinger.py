"""
Indicateur Bollinger Bands.
"""
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from typing import Dict

from .base_indicator import BaseIndicator

class BollingerIndicator(BaseIndicator):
    # -*- coding: utf-8 -*-
    """
    Indicateur Bollinger Bands avec vos param�tres exacts.
    """
    
    def __init__(self):
        super().__init__("BollingerBands")
        self.default_window = 20  # Votre valeur exacte
        self.default_std_dev = 2  # Votre valeur exacte
    
    def calculate(self, data: pd.Series, window: int = None, std_dev: int = None, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule les Bollinger Bands.
        
        Args:
            data: Prix de cl�ture.
            window: P�riode de calcul (d�faut: 20).
            std_dev: Nombre d'�carts-types (d�faut: 2).
            
        Returns:
            Dictionnaire avec upper, lower, middle, percent.
        """
        if not self.validate_data(data, min_length=20):
            empty_series = pd.Series(dtype=float, index=data.index)
            return {
                'upper': empty_series,
                'lower': empty_series,
                'middle': empty_series,
                'percent': empty_series
            }
        
        window = window or self.default_window
        std_dev = std_dev or self.default_std_dev
        
        try:
            # Utiliser ta (votre m�thode)
            bb = BollingerBands(close=data, window=window, window_dev=std_dev)
            
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            middle = bb.bollinger_mavg()
            percent = bb.bollinger_pband()
            
            return {
                'upper': upper,
                'lower': lower,
                'middle': middle,
                'percent': percent
            }
            
        except Exception:
            # Calcul manuel en fallback
            return self._calculate_manual_bb(data, window, std_dev)
    
    def _calculate_manual_bb(self, data: pd.Series, window: int, std_dev: int) -> Dict[str, pd.Series]:
        """Calcul manuel des Bollinger Bands."""
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent = (data - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'percent': percent
        }
    
    def get_signals(self, bb_data: Dict[str, pd.Series], prices: pd.Series) -> dict:
        """
        G�n�re les signaux Bollinger Bands.
        
        Args:
            bb_data: Donn�es Bollinger calcul�es.
            prices: Prix actuels.
            
        Returns:
            Dictionnaire avec les signaux.
        """
        if len(prices) < 2:
            return {'percent': 0.5, 'position': 'MIDDLE'}
        
        current_percent = float(bb_data['percent'].iloc[-1]) if not bb_data['percent'].empty else 0.5
        
        if current_percent <= 0.0:
            position = 'LOWER_BREAK'
        elif current_percent >= 1.0:
            position = 'UPPER_BREAK'
        elif current_percent <= 0.2:
            position = 'NEAR_LOWER'
        elif current_percent >= 0.8:
            position = 'NEAR_UPPER'
        else:
            position = 'MIDDLE'
        
        return {
            'percent': current_percent,
            'position': position,
            'squeeze': self._is_squeeze(bb_data),
            'expansion': self._is_expansion(bb_data)
        }
    
    def _is_squeeze(self, bb_data: Dict[str, pd.Series], lookback: int = 20) -> bool:
        """D�tecte un squeeze (bandes qui se resserrent)."""
        if len(bb_data['upper']) < lookback:
            return False
        
        recent_width = (bb_data['upper'] - bb_data['lower']).tail(lookback)
        current_width = recent_width.iloc[-1]
        avg_width = recent_width.mean()
        
        return current_width < (avg_width * 0.8)
    
    def _is_expansion(self, bb_data: Dict[str, pd.Series], lookback: int = 20) -> bool:
        """D�tecte une expansion (bandes qui s'�largissent)."""
        if len(bb_data['upper']) < lookback:
            return False
        
        recent_width = (bb_data['upper'] - bb_data['lower']).tail(lookback)
        current_width = recent_width.iloc[-1]
        avg_width = recent_width.mean()
        
        return current_width > (avg_width * 1.2)

# Instance globale
bollinger_indicator = BollingerIndicator()

# Fonction de compatibilit�
def calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Fonction de compatibilit�."""
    return bollinger_indicator.calculate(prices, window, std_dev)
