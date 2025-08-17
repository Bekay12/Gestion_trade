"""
Optimisateur de coefficients par secteur.
Migration complète de votre optimisateur_boucle.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
from tqdm import tqdm
from collections import deque
from typing import List, Dict, Tuple, Optional

from config.settings import config
from src.data.providers.yahoo_provider import YahooProvider
from src.backtesting.backtest_engine import BacktestEngine
from src.optimization.parameter_manager import ParameterManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SectorOptimizer:
    """
    Optimisateur de coefficients par secteur.
    Migration complète de votre fonction optimize_sector_coefficients().
    """

    def __init__(self):
        self.yahoo_provider = YahooProvider()
        self.backtest_engine = BacktestEngine()
        self.parameter_manager = ParameterManager()
        self.opt_config = config.optimization

    def get_sector(self, symbol: str) -> str:
        """
        Récupère le secteur d'une action avec logs pour diagnostic.
        Migration exacte de votre fonction get_sector().
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'ℹ️Inconnu!!')
            logger.info(f"📋 {symbol}: Secteur = {sector}")
            return sector
        except Exception as e:
            logger.warning(f"⚠️ Erreur pour {symbol}: {e}")
            return 'ℹ️Inconnu!!'

    def optimize_sector_coefficients(self, sector_symbols: List[str], domain: str, 
                                   period: str = '1y', n_iterations: int = 20, 
                                   montant: float = 50, transaction_cost: float = 1.0,
                                   initial_thresholds: Tuple[float, float] = (4.20, -0.5), 
                                   max_cycles: int = 2, convergence_threshold: float = 0.1) -> Tuple:
        """
        Optimise les coefficients et seuils en quatre étapes (initiale + trois itératives).
        Migration complète de votre fonction optimize_sector_coefficients().

        Args:
            sector_symbols: Liste des symboles du secteur.
            domain: Nom du secteur.
            period: Période des données.
            n_iterations: Nombre d'itérations par étape.
            montant: Montant investi par trade.
            transaction_cost: Frais de transaction.
            initial_thresholds: Seuils initiaux.
            max_cycles: Nombre maximum de cycles.
            convergence_threshold: Seuil de convergence.

        Returns:
            Tuple (best_coeffs, best_gain_total, best_success_rate, best_thresholds).
        """
        if not sector_symbols:
            logger.warning(f"🚫 Secteur {domain} vide, ignoré")
            return None, 0.0, 0.0, initial_thresholds

        # Télécharger les données du secteur
        stock_data = self.yahoo_provider.download_batch(sector_symbols, period=period)

        if not stock_data:
            logger.error(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
            return None, 0.0, 0.0, initial_thresholds

        for symbol, data in stock_data.items():
            logger.info(f"📊 {symbol}: {len(data['Close'])} points de données")

        best_gain_total = -float('inf')
        best_coeffs = None
        best_thresholds = initial_thresholds
        best_success_rate = 0.0
        best_trades = 0

        # Étape initiale : Optimiser coefficients, seuil_achat, et seuil_vente simultanément
        logger.info(f"🔍 Étape initiale : Optimisation globale pour {domain}")

        tested_configs = deque(maxlen=self.opt_config.max_tested_configs)
        doublons = 0

        with tqdm(total=n_iterations, desc=f"🔍 Étape initiale - Global {domain}", unit="it") as pbar:
            for iteration in range(n_iterations):
                # Générer des paramètres aléatoirement (votre logique exacte)
                coeffs = tuple(np.round(np.random.uniform(
                    self.opt_config.coeff_min, self.opt_config.coeff_max, 8), 2))
                seuil_achat = np.round(np.random.uniform(
                    self.opt_config.buy_threshold_min, self.opt_config.buy_threshold_max), 2)
                seuil_vente = np.round(np.random.uniform(
                    self.opt_config.sell_threshold_min, self.opt_config.sell_threshold_max), 2)

                config_key = (coeffs, seuil_achat, seuil_vente)

                if config_key in tested_configs:
                    doublons += 1
                    continue

                tested_configs.append(config_key)

                # Évaluer cette configuration
                total_gain, total_trades, total_success = self._evaluate_configuration(
                    stock_data, domain, coeffs, (seuil_achat, seuil_vente), 
                    montant, transaction_cost
                )

                avg_gain = total_gain / len(stock_data) if stock_data else 0.0
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

                if iteration == 0:
                    logger.info(f"🔍 Étape initiale - Optimisation {domain} démarrée avec {len(stock_data)} symboles")
                    logger.info(f"📈 Itération 1: Gain moyen = {avg_gain:.2f}, Trades = {total_trades}, Taux de réussite = {success_rate:.2f}%")

                if avg_gain > best_gain_total:
                    best_gain_total = avg_gain
                    best_coeffs = coeffs
                    best_thresholds = (seuil_achat, seuil_vente)
                    best_success_rate = success_rate
                    best_trades = total_trades

                pbar.set_postfix({
                    'Gain_moy': f"{best_gain_total:.2f}",
                    'Success_Rate': f"{success_rate:.2f}%",
                    'Seuil_Achat': f"{seuil_achat:.2f}",
                    'Seuil_Vente': f"{seuil_vente:.2f}",
                    'Trades': total_trades
                })
                pbar.update(1)

        logger.info(f"📊 Étape initiale - Nombre de doublons évités : {doublons}")

        if best_coeffs is None:
            logger.error(f"🚨 Étape initiale - Aucun coefficient optimisé pour {domain}: aucun signal ACHAT/VENTE généré")
            return None, 0.0, 0.0, initial_thresholds

        logger.info(f"✅ Étape initiale - Meilleurs coefficients : {best_coeffs}, Seuils : {best_thresholds}, Gain moyen = {best_gain_total:.2f}")

        # Boucle itérative pour les trois étapes (votre logique exacte)
        cycle = 0
        while cycle < max_cycles:
            logger.info(f"\n🔄 Cycle {cycle + 1}/{max_cycles} pour {domain}")

            # Étape 1 : Optimiser les coefficients avec seuils fixes
            best_coeffs = self._optimize_coefficients(
                stock_data, domain, best_coeffs, best_thresholds, 
                n_iterations, montant, transaction_cost, cycle + 1
            )

            # Étape 2 : Optimiser le seuil d'achat avec coefficients et seuil de vente fixes
            new_seuil_achat = self._optimize_buy_threshold(
                stock_data, domain, best_coeffs, best_thresholds, 
                n_iterations, montant, transaction_cost, cycle + 1
            )

            # Étape 3 : Optimiser le seuil de vente avec coefficients et seuil d'achat fixes
            new_seuil_vente = self._optimize_sell_threshold(
                stock_data, domain, best_coeffs, (new_seuil_achat, best_thresholds[1]), 
                n_iterations, montant, transaction_cost, cycle + 1
            )

            current_thresholds = (new_seuil_achat, new_seuil_vente)

            # Évaluer la nouvelle configuration
            total_gain, total_trades, total_success = self._evaluate_configuration(
                stock_data, domain, best_coeffs, current_thresholds, 
                montant, transaction_cost
            )

            current_gain_total = total_gain / len(stock_data) if stock_data else 0.0
            current_success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

            # Mise à jour des meilleurs paramètres
            if current_gain_total > best_gain_total:
                improvement = current_gain_total - best_gain_total
                best_gain_total = current_gain_total
                best_thresholds = current_thresholds
                best_success_rate = current_success_rate
                best_trades = total_trades

                logger.info(f"✅ Cycle {cycle + 1} - Amélioration du gain : {improvement:.2f}")

                # Sauvegarder les résultats
                self.parameter_manager.save_parameters(
                    domain, best_coeffs, best_gain_total, 
                    best_success_rate, best_trades, best_thresholds
                )
            else:
                logger.info(f"✅ Cycle {cycle + 1} - Aucune amélioration significative, arrêt anticipé")
                break

            # Critère de convergence
            if cycle > 0 and improvement < convergence_threshold:
                logger.info(f"✅ Cycle {cycle + 1} - Convergence atteinte (amélioration < {convergence_threshold})")
                break

            cycle += 1

        if best_coeffs is None:
            logger.error(f"🚨 Aucun coefficient optimisé pour {domain}: aucun signal ACHAT/VENTE généré")
        else:
            # Sauvegarde finale
            self.parameter_manager.save_parameters(
                domain, best_coeffs, best_gain_total, 
                best_success_rate, best_trades, best_thresholds
            )

        return best_coeffs, best_gain_total, best_success_rate, best_thresholds

    def _evaluate_configuration(self, stock_data: Dict, domain: str, coeffs: Tuple, 
                               thresholds: Tuple, montant: float, transaction_cost: float) -> Tuple[float, int, int]:
        """Évalue une configuration de paramètres."""
        total_gain = 0.0
        total_trades = 0
        total_success = 0

        domain_coeffs = {domain: coeffs}

        for symbol, data in stock_data.items():
            prices = data['Close']
            volumes = data['Volume']

            result = self.backtest_engine.run_backtest(
                prices, volumes, domain,
                domain_coeffs=domain_coeffs,
                position_size=montant,
                transaction_cost=transaction_cost
            )

            total_gain += result['gain_total']
            total_trades += result['trades']
            total_success += result['gagnants']

        return total_gain, total_trades, total_success

    def _optimize_coefficients(self, stock_data: Dict, domain: str, current_coeffs: Tuple,
                              thresholds: Tuple, n_iterations: int, montant: float, 
                              transaction_cost: float, cycle: int) -> Tuple:
        """Optimise les coefficients avec seuils fixes."""
        logger.info(f"🔍 Étape 1 : Optimisation des coefficients avec seuils {thresholds}")

        best_coeffs = current_coeffs
        best_gain = -float('inf')
        tested_configs = deque(maxlen=self.opt_config.max_tested_configs)
        doublons = 0

        with tqdm(total=n_iterations, desc=f"🔍 Cycle {cycle} - Étape 1 - Coeffs {domain}", unit="it") as pbar:
            for iteration in range(n_iterations):
                coeffs = tuple(np.round(np.random.uniform(
                    self.opt_config.coeff_min, self.opt_config.coeff_max, 8), 2))

                if coeffs in tested_configs:
                    doublons += 1
                    continue

                tested_configs.append(coeffs)

                total_gain, total_trades, total_success = self._evaluate_configuration(
                    stock_data, domain, coeffs, thresholds, montant, transaction_cost
                )

                avg_gain = total_gain / len(stock_data) if stock_data else 0.0
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

                if iteration == 0:
                    logger.info(f"🔍 Cycle {cycle} - Étape 1 - Optimisation {domain} démarrée avec {len(stock_data)} symboles")
                    logger.info(f"📈 Itération 1: Gain moyen = {avg_gain:.2f}, Trades = {total_trades}, Taux de réussite = {success_rate:.2f}%")

                if avg_gain > best_gain:
                    best_gain = avg_gain
                    best_coeffs = coeffs

                pbar.set_postfix({
                    'Gain_moy': f"{best_gain:.2f}",
                    'Success_Rate': f"{success_rate:.2f}%",
                    'Trades': total_trades
                })
                pbar.update(1)

        logger.info(f"📊 Cycle {cycle} - Étape 1 - Nombre de doublons évités : {doublons}")
        logger.info(f"✅ Cycle {cycle} - Étape 1 - Meilleurs coefficients : {best_coeffs}, Gain moyen = {best_gain:.2f}")

        return best_coeffs

    def _optimize_buy_threshold(self, stock_data: Dict, domain: str, coeffs: Tuple,
                               thresholds: Tuple, n_iterations: int, montant: float,
                               transaction_cost: float, cycle: int) -> float:
        """Optimise le seuil d'achat."""
        logger.info(f"🔍 Étape 2 : Optimisation du seuil d'achat avec coefficients {coeffs} et seuil de vente {thresholds[1]}")

        best_threshold = thresholds[0]
        best_gain = -float('inf')
        tested_configs = deque(maxlen=self.opt_config.max_tested_configs)
        doublons = 0

        with tqdm(total=n_iterations, desc=f"🔍 Cycle {cycle} - Étape 2 - Seuil Achat {domain}", unit="it") as pbar:
            for iteration in range(n_iterations):
                seuil_achat = np.round(np.random.uniform(
                    self.opt_config.buy_threshold_min, self.opt_config.buy_threshold_max), 2)

                if seuil_achat in tested_configs:
                    doublons += 1
                    continue

                tested_configs.append(seuil_achat)

                total_gain, total_trades, total_success = self._evaluate_configuration(
                    stock_data, domain, coeffs, (seuil_achat, thresholds[1]), 
                    montant, transaction_cost
                )

                avg_gain = total_gain / len(stock_data) if stock_data else 0.0
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

                if iteration == 0:
                    logger.info(f"🔍 Cycle {cycle} - Étape 2 - Optimisation {domain} démarrée")
                    logger.info(f"📈 Itération 1: Gain moyen = {avg_gain:.2f}, Trades = {total_trades}")

                if avg_gain > best_gain:
                    best_gain = avg_gain
                    best_threshold = seuil_achat

                pbar.set_postfix({
                    'Gain_moy': f"{best_gain:.2f}",
                    'Seuil_Achat': f"{seuil_achat:.2f}",
                    'Trades': total_trades
                })
                pbar.update(1)

        logger.info(f"📊 Cycle {cycle} - Étape 2 - Nombre de doublons évités : {doublons}")
        logger.info(f"✅ Cycle {cycle} - Étape 2 - Meilleur seuil d'achat : {best_threshold}, Gain moyen = {best_gain:.2f}")

        return best_threshold

    def _optimize_sell_threshold(self, stock_data: Dict, domain: str, coeffs: Tuple,
                                thresholds: Tuple, n_iterations: int, montant: float,
                                transaction_cost: float, cycle: int) -> float:
        """Optimise le seuil de vente."""
        logger.info(f"🔍 Étape 3 : Optimisation du seuil de vente avec coefficients {coeffs} et seuil d'achat {thresholds[0]}")

        best_threshold = thresholds[1]
        best_gain = -float('inf')
        tested_configs = deque(maxlen=self.opt_config.max_tested_configs)
        doublons = 0

        with tqdm(total=n_iterations, desc=f"🔍 Cycle {cycle} - Étape 3 - Seuil Vente {domain}", unit="it") as pbar:
            for iteration in range(n_iterations):
                seuil_vente = np.round(np.random.uniform(
                    self.opt_config.sell_threshold_min, self.opt_config.sell_threshold_max), 2)

                if seuil_vente in tested_configs:
                    doublons += 1
                    continue

                tested_configs.append(seuil_vente)

                total_gain, total_trades, total_success = self._evaluate_configuration(
                    stock_data, domain, coeffs, (thresholds[0], seuil_vente), 
                    montant, transaction_cost
                )

                avg_gain = total_gain / len(stock_data) if stock_data else 0.0
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

                if iteration == 0:
                    logger.info(f"🔍 Cycle {cycle} - Étape 3 - Optimisation {domain} démarrée")
                    logger.info(f"📈 Itération 1: Gain moyen = {avg_gain:.2f}, Trades = {total_trades}")

                if avg_gain > best_gain:
                    best_gain = avg_gain
                    best_threshold = seuil_vente

                pbar.set_postfix({
                    'Gain_moy': f"{best_gain:.2f}",
                    'Seuil_Vente': f"{seuil_vente:.2f}",
                    'Trades': total_trades
                })
                pbar.update(1)

        logger.info(f"📊 Cycle {cycle} - Étape 3 - Nombre de doublons évités : {doublons}")
        logger.info(f"✅ Cycle {cycle} - Étape 3 - Meilleur seuil de vente : {best_threshold}, Gain moyen = {best_gain:.2f}")

        return best_threshold

# Instance globale
sector_optimizer = SectorOptimizer()

# Fonction de compatibilité
def optimize_sector_coefficients(sector_symbols: List[str], domain: str, period: str = '1y', 
                               n_iterations: int = 20, montant: float = 50, transaction_cost: float = 1.0,
                               initial_thresholds: Tuple[float, float] = (4.20, -0.5), 
                               max_cycles: int = 2, convergence_threshold: float = 0.1) -> Tuple:
    """Fonction de compatibilité avec votre code existant."""
    return sector_optimizer.optimize_sector_coefficients(
        sector_symbols, domain, period, n_iterations, montant, transaction_cost,
        initial_thresholds, max_cycles, convergence_threshold
    )
