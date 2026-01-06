"""
Gestionnaire unifié des indicateurs techniques.
Intègre tous vos indicateurs de get_trading_signal().
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, Tuple
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator, IchimokuIndicator

from config.settings import config
from .macd import MACDIndicator

class IndicatorManager:
    """
    Gestionnaire unifié des indicateurs techniques.
    Regroupe tous les indicateurs utilisés dans votre get_trading_signal().
    """

    def __init__(self):
        self.config = config.indicators
        self.macd_indicator = MACDIndicator()
        self._cached_results = {}

    def calculate_all_indicators(self, prices: pd.Series, volumes: pd.Series = None) -> Dict[str, Any]:
        """
        Calcule tous les indicateurs nécessaires.
        Migration de tous vos calculs dans get_trading_signal().

        Args:
            prices: Série des prix de clôture.
            volumes: Série des volumes (optionnel).

        Returns:
            Dictionnaire avec tous les indicateurs calculés.
        """
        results = {}

        # MACD (votre fonction calculate_macd)
        macd, signal_line = self.macd_indicator.calculate(prices)
        results['macd'] = macd
        results['signal_line'] = signal_line

        # RSI (votre configuration window=17)
        rsi = ta.momentum.RSIIndicator(close=prices, window=self.config.rsi_period).rsi()
        results['rsi'] = rsi

        # EMAs (vos spans: 20, 50, 200)
        results['ema20'] = prices.ewm(span=self.config.ema20_span, adjust=False).mean()
        results['ema50'] = prices.ewm(span=self.config.ema50_span, adjust=False).mean()

        if len(prices) >= self.config.ema200_span:
            results['ema200'] = prices.ewm(span=self.config.ema200_span, adjust=False).mean()
        else:
            results['ema200'] = results['ema50']  # Fallback comme dans votre code

        # Bollinger Bands (vos paramètres: window=20, window_dev=2)
        bb = BollingerBands(close=prices, window=self.config.bb_window, window_dev=self.config.bb_std_dev)
        results['bb_upper'] = bb.bollinger_hband()
        results['bb_lower'] = bb.bollinger_lband()
        results['bb_percent'] = (prices - results['bb_lower']) / (results['bb_upper'] - results['bb_lower'])

        # ADX (votre configuration: window=14)
        adx_indicator = ADXIndicator(high=prices, low=prices, close=prices, window=self.config.adx_window)
        results['adx'] = adx_indicator.adx()

        # Ichimoku Cloud (vos paramètres: window1=9, window2=26, window3=52)
        ichimoku = IchimokuIndicator(
            high=prices, low=prices, 
            window1=self.config.ichimoku_conversion,
            window2=self.config.ichimoku_base,
            window3=self.config.ichimoku_span
        )
        results['ichimoku_base'] = ichimoku.ichimoku_base_line()
        results['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

        # Momentum 10 jours (de votre code)
        results['momentum_10'] = prices.pct_change(10)

        # Volatilité (votre calcul)
        results['volatility'] = prices.pct_change().rolling(20).std()

        # Volume si fourni
        if volumes is not None:
            results['volume_mean_30'] = volumes.rolling(window=30).mean()
            results['volume_std_30'] = volumes.rolling(window=30).std()

        # Ratio de Sharpe court terme (votre calcul)
        returns = prices.pct_change()
        if returns.std() > 0:
            results['sharpe'] = returns.mean() / returns.std() * np.sqrt(252)
        else:
            results['sharpe'] = pd.Series(0, index=prices.index)

        return results

    def get_latest_values(self, prices: pd.Series, volumes: pd.Series = None) -> Dict[str, float]:
        """
        Retourne les dernières valeurs de tous les indicateurs.
        Directement utilisable pour vos conditions de trading.

        Returns:
            Dictionnaire avec les valeurs les plus récentes.
        """
        indicators = self.calculate_all_indicators(prices, volumes)

        latest = {}

        # Prix et EMAs (vos conversions explicites en float)
        latest['last_close'] = float(prices.iloc[-1])
        latest['last_ema20'] = float(indicators['ema20'].iloc[-1])
        latest['last_ema50'] = float(indicators['ema50'].iloc[-1])
        latest['last_ema200'] = float(indicators['ema200'].iloc[-1])

        # MACD (vos valeurs last et prev)
        if len(indicators['macd']) >= 2:
            latest['last_macd'] = float(indicators['macd'].iloc[-1])
            latest['prev_macd'] = float(indicators['macd'].iloc[-2])
            latest['last_signal'] = float(indicators['signal_line'].iloc[-1])
            latest['prev_signal'] = float(indicators['signal_line'].iloc[-2])
        else:
            latest.update({'last_macd': 0.0, 'prev_macd': 0.0, 'last_signal': 0.0, 'prev_signal': 0.0})

        # RSI (vos valeurs last et prev)
        if len(indicators['rsi']) >= 2:
            latest['last_rsi'] = float(indicators['rsi'].iloc[-1])
            latest['prev_rsi'] = float(indicators['rsi'].iloc[-2])
            latest['delta_rsi'] = latest['last_rsi'] - latest['prev_rsi']
        else:
            latest.update({'last_rsi': 50.0, 'prev_rsi': 50.0, 'delta_rsi': 0.0})

        # Bollinger Bands
        if len(indicators['bb_percent']) > 0:
            latest['last_bb_percent'] = float(indicators['bb_percent'].iloc[-1])
        else:
            latest['last_bb_percent'] = 0.5

        # ADX
        if len(indicators['adx']) > 0:
            latest['last_adx'] = float(indicators['adx'].iloc[-1])
        else:
            latest['last_adx'] = 0

        # Ichimoku
        if len(indicators['ichimoku_base']) > 0:
            latest['last_ichimoku_base'] = float(indicators['ichimoku_base'].iloc[-1])
            latest['last_ichimoku_conversion'] = float(indicators['ichimoku_conversion'].iloc[-1])
        else:
            latest['last_ichimoku_base'] = latest['last_close']
            latest['last_ichimoku_conversion'] = latest['last_close']

        # Volatilité (votre conversion en scalaire)
        if isinstance(indicators['volatility'], pd.Series) and not indicators['volatility'].empty:
            latest['volatility'] = float(indicators['volatility'].iloc[-1])
        else:
            latest['volatility'] = 0.05

        # Volume si disponible
        if volumes is not None and len(volumes) >= 30:
            latest['volume_mean'] = float(indicators['volume_mean_30'].iloc[-1])
            latest['volume_std'] = float(indicators['volume_std_30'].iloc[-1])
            latest['current_volume'] = float(volumes.iloc[-1])
        else:
            volume_mean = float(volumes.mean()) if volumes is not None and len(volumes) > 0 else 0.0
            latest.update({
                'volume_mean': volume_mean,
                'volume_std': 0.0,
                'current_volume': float(volumes.iloc[-1]) if volumes is not None and len(volumes) > 0 else 0.0
            })

        return latest

    def get_performance_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calcule les métriques de performance (vos calculs de variation).

        Returns:
            Dictionnaire avec les variations sur différentes périodes.
        """
        metrics = {}

        # Performance long terme (votre code)
        if len(prices) >= 30:
            metrics['variation_30j'] = ((prices.iloc[-1] - prices.iloc[-30]) / prices.iloc[-30] * 100)
        else:
            metrics['variation_30j'] = np.nan

        if len(prices) >= 180:
            metrics['variation_180j'] = ((prices.iloc[-1] - prices.iloc[-180]) / prices.iloc[-180] * 100)
        else:
            metrics['variation_180j'] = np.nan

        return metrics

# Instance globale
indicator_manager = IndicatorManager()
