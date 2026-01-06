"""
Générateur de signaux de trading.
Migration de votre fonction get_trading_signal().
"""
import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, Optional

from config.settings import config
from src.indicators.manager import IndicatorManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SignalGenerator:
    """
    Générateur de signaux de trading.
    Migration complète de votre fonction get_trading_signal().
    """

    def __init__(self):
        self.indicator_manager = IndicatorManager()
        self.config = config.trading

    def generate_signal(self, prices: pd.Series, volumes: pd.Series, sector: str = "default",
                       domain_coeffs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Génère un signal de trading.
        Migration complète de votre fonction get_trading_signal().

        Args:
            prices: Série des prix de clôture.
            volumes: Série des volumes.
            sector: Secteur de l'actif.
            domain_coeffs: Coefficients par domaine (optionnel).

        Returns:
            Dictionnaire avec le signal et les métriques.
        """
        # Validation des données (votre logique)
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if isinstance(volumes, pd.DataFrame):
            volumes = volumes.squeeze()

        if len(prices) < self.config.min_data_points:
            return {
                'signal': 'INSUFFICIENT_DATA',
                'price': None,
                'trend': None,
                'rsi': None,
                'volume_mean': None,
                'score': None
            }

        # Calculer tous les indicateurs
        latest_values = self.indicator_manager.get_latest_values(prices, volumes)
        performance = self.indicator_manager.get_performance_metrics(prices)

        # Validation des derniers points (votre logique)
        if 'last_macd' not in latest_values or 'last_rsi' not in latest_values:
            return {
                'signal': 'MISSING_DATA',
                'price': None,
                'trend': None,
                'rsi': None,
                'volume_mean': None,
                'score': None
            }

        # Calculer le score (votre logique complète)
        score_result = self._calculate_score(latest_values, performance, sector, domain_coeffs)

        # Déterminer le signal final (votre logique de seuils)
        signal = self._determine_signal(score_result['score'], sector, domain_coeffs)

        return {
            'signal': signal,
            'price': latest_values['last_close'],
            'trend': latest_values['last_close'] > latest_values['last_ema20'],
            'rsi': round(latest_values['last_rsi'], 2),
            'volume_mean': round(latest_values['volume_mean'], 2),
            'score': round(score_result['score'], 3),
            'details': score_result['details']
        }

    def _calculate_score(self, latest: Dict, performance: Dict, sector: str,
                        domain_coeffs: Optional[Dict] = None) -> Dict:
        """
        Calcule le score de trading.
        Migration exacte de votre logique de scoring complexe.
        """
        # Récupérer les coefficients (votre logique extract_best_parameters)
        coeffs, thresholds = self._get_coefficients(sector, domain_coeffs)
        a1, a2, a3, a4, a5, a6, a7, a8 = coeffs

        # Multiplicateurs (votre logique)
        m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0

        # ADX strong trend (votre condition)
        if latest['last_adx'] > config.indicators.adx_strong_threshold:
            m1 = 1.5

        # Vérification du volume (votre logique Z-score)
        if latest['volume_std'] > 0:
            z = (latest['current_volume'] - latest['volume_mean']) / latest['volume_std']
            if z > 1.75:
                m2 = 1.5
            elif z < -1.75:
                m2 = 0.7

        volume_ratio = latest['current_volume'] / latest['volume_mean'] if latest['volume_mean'] > 0 else 0
        if volume_ratio > 1.5:
            m3 = 1.5
        elif volume_ratio < 0.5:
            m3 = 0.7

        # Initialiser le score
        score = 0.0
        details = {}

        # RSI : Signaux haussiers (votre logique exacte)
        rsi_cross_up = latest['prev_rsi'] < 30 and latest['last_rsi'] >= 30
        rsi_cross_mid = latest['prev_rsi'] < 50 and latest['last_rsi'] >= 50
        rsi_cross_down = latest['prev_rsi'] > 65 and latest['last_rsi'] <= 65
        rsi_ok = 40 < latest['last_rsi'] < 75

        if rsi_cross_up:
            score += a1
            details['rsi_cross_up'] = a1

        if latest['delta_rsi'] > 3:
            contribution = m3 * a2
            score += contribution
            details['rsi_momentum_up'] = contribution

        if rsi_cross_mid:
            score += a3
            details['rsi_cross_mid'] = a3

        # RSI : Signaux baissiers
        if rsi_cross_down:
            score -= a1
            details['rsi_cross_down'] = -a1

        if latest['delta_rsi'] < -3:
            contribution = m3 * a2
            score -= contribution
            details['rsi_momentum_down'] = -contribution

        if rsi_ok:
            score += a4
            details['rsi_ok'] = a4
        else:
            score -= a4
            details['rsi_not_ok'] = -a4

        # EMA : Structure de tendance (votre logique)
        ema_structure_up = (latest['last_close'] > latest['last_ema20'] > 
                           latest['last_ema50'] > latest['last_ema200'])
        ema_structure_down = (latest['last_close'] < latest['last_ema20'] < 
                             latest['last_ema50'] < latest['last_ema200'])

        if ema_structure_up:
            contribution = m1 * a5
            score += contribution
            details['ema_structure_up'] = contribution

        if ema_structure_down:
            contribution = m1 * a5
            score -= contribution
            details['ema_structure_down'] = -contribution

        # MACD : Croisements (votre logique)
        is_macd_cross_up = latest['prev_macd'] < latest['prev_signal'] and latest['last_macd'] > latest['last_signal']
        is_macd_cross_down = latest['prev_macd'] > latest['prev_signal'] and latest['last_macd'] < latest['last_signal']

        if is_macd_cross_up:
            score += a6
            details['macd_cross_up'] = a6

        if is_macd_cross_down:
            score -= a6
            details['macd_cross_down'] = -a6

        # Volume (votre logique)
        is_volume_ok = latest['volume_mean'] > self.config.volume_threshold
        if is_volume_ok:
            contribution = m2 * a6
            score += contribution
            details['volume_ok'] = contribution
        else:
            contribution = m2 * a6
            score -= contribution
            details['volume_not_ok'] = -contribution

        # Performance passée (votre logique)
        is_variation_ok = (not np.isnan(performance['variation_30j']) and 
                          performance['variation_30j'] > self.config.variation_threshold)
        if is_variation_ok:
            score += a7
            details['variation_ok'] = a7
        else:
            score -= a7
            details['variation_not_ok'] = -a7

        # Conditions avancées (votre logique complète)
        strong_uptrend = (latest['last_close'] > latest['last_ichimoku_base'] and 
                         latest['last_close'] > latest['last_ichimoku_conversion'])
        strong_downtrend = (latest['last_close'] < latest['last_ichimoku_base'] and 
                           latest['last_close'] < latest['last_ichimoku_conversion'])
        adx_strong_trend = latest['last_adx'] > config.indicators.adx_strong_threshold

        if strong_uptrend:
            contribution = m2 * a5
            score += contribution
            details['strong_uptrend'] = contribution

        if latest['last_bb_percent'] < 0.4:
            contribution = m3 * a4
            score += contribution
            details['bb_oversold'] = contribution

        if strong_downtrend:
            contribution = m2 * a5
            score -= contribution
            details['strong_downtrend'] = -contribution

        if latest['last_bb_percent'] > 0.6:
            contribution = m3 * a4
            score -= contribution
            details['bb_overbought'] = -contribution

        # Conditions d'achat/vente renforcées (votre logique)
        buy_conditions = (
            (is_macd_cross_up or ema_structure_up) and
            (rsi_cross_up or rsi_cross_mid) and
            (latest['last_rsi'] < 65) and
            (latest['last_bb_percent'] < 0.7) and
            (strong_uptrend or adx_strong_trend) and
            (latest['volume_mean'] > self.config.volume_threshold) and
            (is_variation_ok if not np.isnan(performance.get('variation_30j', np.nan)) else True)
        )

        sell_conditions = (
            (is_macd_cross_down or ema_structure_down) and
            (rsi_cross_down or latest['last_rsi'] > 70) and
            (latest['last_rsi'] > 35) and
            (latest['last_bb_percent'] > 0.3) and
            (strong_downtrend or adx_strong_trend) and
            (latest['volume_mean'] > self.config.volume_threshold)
        )

        if buy_conditions:
            score += a8
            details['buy_conditions'] = a8

        if sell_conditions:
            score -= a8
            details['sell_conditions'] = -a8

        # Volatilité (votre ajustement final)
        if latest['volatility'] > 0.05:
            m4 = 0.75

        score *= m4
        if m4 != 1.0:
            details['volatility_adjustment'] = f"x{m4}"

        return {'score': score, 'details': details}

    def _get_coefficients(self, sector: str, domain_coeffs: Optional[Dict] = None) -> Tuple:
        """Récupère les coefficients pour un secteur."""
        default_coeffs = (1.75, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75)
        default_thresholds = (config.trading.default_buy_threshold, config.trading.default_sell_threshold)

        if domain_coeffs and sector in domain_coeffs:
            return domain_coeffs[sector], default_thresholds

        # Utiliser les paramètres optimisés depuis la configuration
        coeffs, thresholds, _ = config.get_sector_parameters(sector)
        return coeffs, thresholds

    def _determine_signal(self, score: float, sector: str, domain_coeffs: Optional[Dict] = None) -> str:
        """Détermine le signal final basé sur le score."""
        coeffs, thresholds = self._get_coefficients(sector, domain_coeffs)

        if score >= thresholds[0]:
            return "BUY"
        elif score <= thresholds[1]:
            return "SELL"
        else:
            return "HOLD"

# Instance globale
signal_generator = SignalGenerator()

# Fonction de compatibilité
def get_trading_signal(prices: pd.Series, volumes: pd.Series, domaine: str,
                      domain_coeffs: Optional[Dict] = None,
                      variation_seuil: float = -20, volume_seuil: int = 100000) -> Tuple:
    """
    Fonction de compatibilité avec votre code existant.
    Migration de votre get_trading_signal().
    """
    result = signal_generator.generate_signal(prices, volumes, domaine, domain_coeffs)

    # Convertir pour correspondre à votre format de retour original
    signal_map = {"BUY": "ACHAT", "SELL": "VENTE", "HOLD": "NEUTRE"}
    signal = signal_map.get(result['signal'], result['signal'])

    return (signal, result['price'], result['trend'], 
            result['rsi'], result['volume_mean'], result['score'])
