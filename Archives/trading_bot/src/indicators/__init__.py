# -*- coding: utf-8 -*-
"""
Indicateurs techniques - Version compl�te.
"""
from .base_indicator import BaseIndicator
from .macd import MACDIndicator, calculate_macd #, macd_indicator
from .rsi import RSIIndicator, rsi_indicator, calculate_rsi
from .bollinger import BollingerIndicator, bollinger_indicator, calculate_bollinger_bands
from .adx import ADXIndicator_Custom, adx_indicator
from .ichimoku import IchimokuIndicator_Custom, ichimoku_indicator
from .manager import IndicatorManager, indicator_manager

__all__ = [
    'BaseIndicator',
    'MACDIndicator', 'calculate_macd', #, 'macd_indicator'
    'RSIIndicator', 'rsi_indicator', 'calculate_rsi',
    'BollingerIndicator', 'bollinger_indicator', 'calculate_bollinger_bands',
    'ADXIndicator_Custom', 'adx_indicator',
    'IchimokuIndicator_Custom', 'ichimoku_indicator',
    'IndicatorManager', 'indicator_manager'
]
