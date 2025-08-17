"""
Moteur de backtesting.
Migration de votre fonction backtest_signals().
"""
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional

from config.settings import config
from src.signals.signal_generator import SignalGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BacktestEngine:
    """
    Moteur de backtesting pour évaluer les stratégies.
    Migration complète de votre fonction backtest_signals().
    """

    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.config = config.trading

    def run_backtest(self, prices: Union[pd.Series, pd.DataFrame], 
                     volumes: Union[pd.Series, pd.DataFrame],
                     sector: str = "default", 
                     position_size: float = None,
                     transaction_cost: float = None,
                     domain_coeffs: Optional[Dict] = None) -> Dict:
        """
        Effectue un backtest sur une série de prix.
        Migration complète de votre fonction backtest_signals().

        Args:
            prices: Série ou DataFrame des prix de clôture.
            volumes: Série ou DataFrame des volumes.
            sector: Secteur de l'actif.
            position_size: Montant investi par trade.
            transaction_cost: Frais de transaction par trade.
            domain_coeffs: Coefficients par domaine.

        Returns:
            Dict avec les métriques de performance.
        """
        # Valeurs par défaut
        if position_size is None:
            position_size = self.config.position_size
        if transaction_cost is None:
            transaction_cost = self.config.transaction_cost

        # Validation des entrées (votre logique exacte)
        if not isinstance(prices, (pd.Series, pd.DataFrame)) or not isinstance(volumes, (pd.Series, pd.DataFrame)):
            return self._empty_result()

        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if isinstance(volumes, pd.DataFrame):
            volumes = volumes.squeeze()

        if (len(prices) < self.config.min_data_points or 
            len(volumes) < self.config.min_data_points or 
            prices.isna().any() or volumes.isna().any()):
            return self._empty_result()

        # Pré-calculer les signaux pour toute la série (votre logique)
        signals = []
        logger.debug(f"Calcul des signaux pour {len(prices)} points de données")

        for i in range(self.config.min_data_points, len(prices)):
            try:
                result = self.signal_generator.generate_signal(
                    prices[:i], volumes[:i], sector, domain_coeffs
                )
                # Mapper vers vos signaux originaux
                signal_map = {"BUY": "ACHAT", "SELL": "VENTE", "HOLD": "NEUTRE"}
                signal = signal_map.get(result['signal'], result['signal'])
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Erreur calcul signal à l'index {i}: {e}")
                signals.append("NEUTRE")

        signals = pd.Series(signals, index=prices.index[self.config.min_data_points:])

        # Simuler les trades (votre logique exacte)
        positions = []

        for i in range(len(signals)):
            current_signal = signals.iloc[i]

            if current_signal == "ACHAT":
                # Ouvrir une nouvelle position
                positions.append({
                    "entry": prices.iloc[i + self.config.min_data_points],
                    "entry_idx": i + self.config.min_data_points,
                    "type": "buy"
                })

            elif current_signal == "VENTE" and positions:
                # Fermer la dernière position ouverte
                last_position = positions[-1]
                if "exit" not in last_position:
                    last_position["exit"] = prices.iloc[i + self.config.min_data_points]
                    last_position["exit_idx"] = i + self.config.min_data_points

        # Calculer les métriques (votre logique exacte)
        return self._calculate_metrics(positions, position_size, transaction_cost)

    def _calculate_metrics(self, positions: list, position_size: float, transaction_cost: float) -> Dict:
        """
        Calcule les métriques de performance.
        Migration de votre logique de calcul de métriques.
        """
        nb_trades = 0
        nb_gagnants = 0
        gain_total = 0.0
        gains = []
        portfolio_values = [position_size]  # Suivi de la valeur du portefeuille

        for pos in positions:
            if "exit" in pos:
                nb_trades += 1
                entry = pos["entry"]
                exit_price = pos["exit"]

                # Calcul du rendement (votre logique)
                rendement = (exit_price - entry) / entry

                # Ajuster pour les frais de transaction (entrée + sortie) - votre logique
                gain = position_size * rendement * (1 - 2 * transaction_cost)

                gain_total += gain
                gains.append(gain)

                if gain > 0:
                    nb_gagnants += 1

                portfolio_values.append(portfolio_values[-1] + gain)

        # Calculer le drawdown maximum (votre logique)
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        drawdown_max = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0

        # Résultat dans votre format exact
        return {
            "trades": nb_trades,
            "gagnants": nb_gagnants,
            "taux_reussite": (nb_gagnants / nb_trades * 100) if nb_trades else 0,
            "gain_total": round(gain_total, 2),
            "gain_moyen": round(np.mean(gains), 2) if gains else 0.0,
            "drawdown_max": round(drawdown_max, 2),

            # Métriques supplémentaires
            "total_positions": len(positions),
            "open_positions": len([p for p in positions if "exit" not in p]),
            "portfolio_values": portfolio_values,
            "individual_gains": gains
        }

    def _empty_result(self) -> Dict:
        """Retourne un résultat vide (votre format)."""
        return {
            "trades": 0,
            "gagnants": 0,
            "taux_reussite": 0,
            "gain_total": 0.0,
            "gain_moyen": 0.0,
            "drawdown_max": 0.0,
            "total_positions": 0,
            "open_positions": 0,
            "portfolio_values": [],
            "individual_gains": []
        }

    def batch_backtest(self, stock_data: Dict[str, Dict], sector: str = "default",
                      domain_coeffs: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Effectue un backtest sur plusieurs symboles.
        Utile pour l'optimisation par secteur.
        """
        results = {}

        for symbol, data in stock_data.items():
            try:
                prices = data['Close']
                volumes = data['Volume']

                result = self.run_backtest(
                    prices, volumes, sector, 
                    domain_coeffs=domain_coeffs
                )

                results[symbol] = result
                logger.debug(f"Backtest {symbol}: {result['trades']} trades, "
                           f"{result['taux_reussite']:.1f}% réussite")

            except Exception as e:
                logger.error(f"Erreur backtest {symbol}: {e}")
                results[symbol] = self._empty_result()

        return results

# Instance globale
backtest_engine = BacktestEngine()

# Fonction de compatibilité
def backtest_signals(prices: Union[pd.Series, pd.DataFrame], 
                    volumes: Union[pd.Series, pd.DataFrame],
                    domaine: str, montant: float = 50, 
                    transaction_cost: float = 0.01,
                    domain_coeffs: Optional[Dict] = None) -> Dict:
    """
    Fonction de compatibilité avec votre code existant.
    Migration directe de votre backtest_signals().
    """
    return backtest_engine.run_backtest(
        prices, volumes, domaine, montant, transaction_cost, domain_coeffs
    )
