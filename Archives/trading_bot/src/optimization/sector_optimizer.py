"""
Optimisateur de coefficients par secteur - Version orientée objet.
Migration de votre optimisateur_boucle avec amélirations.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import deque
from typing import Dict, Tuple, List, Optional
import yfinance as yf

# Imports depuis votre nouvelle architecture
from config.settings import config
from src.data.providers.yahoo_provider import YahooProvider
from src.backtesting.backtest_engine import BacktestEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SectorOptimizer:
    """
    Optimisateur de coefficients par secteur.
    Basé sur votre optimisateur_boucle avec architecture en classes.
    """
    
    def __init__(self, 
                 data_provider: Optional[YahooProvider] = None,
                 backtest_engine: Optional[BacktestEngine] = None):
        self.data_provider = data_provider or YahooProvider()
        self.backtest_engine = backtest_engine or BacktestEngine()
        
        # Configuration depuis votre optimisateur
        self.opt_config = config.optimization
        self.trading_config = config.trading
        
        # Cache des configurations testées (comme dans votre code)
        self.tested_configs = deque(maxlen=self.opt_config.max_tested_configs)
        
    def get_sector(self, symbol: str) -> str:
        """
        Récupère le secteur d'une action.
        Même logique que votre fonction get_sector().
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
    
    def optimize_sector_coefficients(self, 
                                   sector_symbols: List[str], 
                                   sector_name: str,
                                   period: str = '1y') -> Tuple[Optional[Tuple], float, float, Tuple[float, float]]:
        """
        Optimise les coefficients et seuils pour un secteur.
        Migration directe de votre fonction optimize_sector_coefficients().
        """
        if not sector_symbols:
            logger.warning(f"🚫 Secteur {sector_name} vide, ignoré")
            return None, 0.0, 0.0, (config.trading.default_buy_threshold, 
                                   config.trading.default_sell_threshold)
        
        # Télécharger les données (même logique que votre code)
        logger.info(f"📊 Téléchargement des données pour {sector_name}")
        stock_data = self.data_provider.download_batch(sector_symbols, period)
        
        if not stock_data:
            logger.error(f"🚨 Aucune donnée téléchargée pour le secteur {sector_name}")
            return None, 0.0, 0.0, (config.trading.default_buy_threshold,
                                   config.trading.default_sell_threshold)
        
        # Affichage des données (comme dans votre code)
        for symbol, data in stock_data.items():
            logger.info(f"📊 {symbol}: {len(data['Close'])} points de données")
        
        # Variables de tracking
        best_gain_total = -float('inf')
        best_coeffs = None
        best_thresholds = (config.trading.default_buy_threshold, 
                          config.trading.default_sell_threshold)
        best_success_rate = 0.0
        best_trades = 0
        
        # ÉTAPE INITIALE : Optimisation globale (votre logique)
        logger.info(f"🔍 Étape initiale : Optimisation globale pour {sector_name}")
        best_coeffs, best_gain_total, best_success_rate, best_thresholds, best_trades = \
            self._optimize_global(stock_data, sector_name)
        
        if best_coeffs is None:
            logger.error(f"🚨 Aucun coefficient optimisé pour {sector_name}")
            return None, 0.0, 0.0, (config.trading.default_buy_threshold,
                                   config.trading.default_sell_threshold)
        
        # BOUCLES ITÉRATIVES (votre logique des 3 étapes)
        cycle = 0
        while cycle < self.opt_config.max_cycles:
            logger.info(f"🔄 Cycle {cycle + 1}/{self.opt_config.max_cycles} pour {sector_name}")
            
            # Étape 1: Optimiser coefficients
            best_coeffs, best_gain_total, best_success_rate, best_trades = \
                self._optimize_coefficients(stock_data, sector_name, best_thresholds, cycle)
            
            # Étape 2: Optimiser seuil d'achat
            best_thresholds, best_gain_total, best_success_rate, best_trades = \
                self._optimize_buy_threshold(stock_data, sector_name, best_coeffs, 
                                           best_thresholds, cycle)
            
            # Étape 3: Optimiser seuil de vente
            best_thresholds, best_gain_total, best_success_rate, best_trades = \
                self._optimize_sell_threshold(stock_data, sector_name, best_coeffs, 
                                            best_thresholds, cycle)
            
            # Critère de convergence (comme dans votre code)
            if cycle > 0:
                improvement = best_gain_total - getattr(self, '_previous_gain', best_gain_total)
                if improvement < self.opt_config.convergence_threshold:
                    logger.info(f"✅ Convergence atteinte pour {sector_name}")
                    break
            
            self._previous_gain = best_gain_total
            cycle += 1
        
        # Sauvegarder les résultats
        if best_coeffs:
            self._save_results(sector_name, best_coeffs, best_gain_total, 
                             best_success_rate, best_trades, best_thresholds)
        
        return best_coeffs, best_gain_total, best_success_rate, best_thresholds
    
    def _optimize_global(self, stock_data: Dict, sector_name: str) -> Tuple:
        """Étape d'optimisation globale initiale."""
        best_gain = -float('inf')
        best_coeffs = None
        best_thresholds = None
        best_success_rate = 0.0
        best_trades = 0
        doublons = 0
        
        with tqdm(total=self.opt_config.n_iterations, 
                 desc=f"🔍 Global {sector_name}", unit="it") as pbar:
            
            for iteration in range(self.opt_config.n_iterations):
                # Génération aléatoire (même logique que votre code)
                coeffs = tuple(np.round(np.random.uniform(
                    self.opt_config.coeff_min, self.opt_config.coeff_max, 8), 2))
                
                seuil_achat = np.round(np.random.uniform(
                    self.opt_config.buy_threshold_min, self.opt_config.buy_threshold_max), 2)
                
                seuil_vente = np.round(np.random.uniform(
                    self.opt_config.sell_threshold_min, self.opt_config.sell_threshold_max), 2)
                
                config_key = (coeffs, seuil_achat, seuil_vente)
                
                # Éviter les doublons (votre logique)
                if config_key in self.tested_configs:
                    doublons += 1
                    continue
                
                self.tested_configs.append(config_key)
                
                # Évaluation sur tous les symboles
                total_gain, total_trades, total_success = self._evaluate_configuration(
                    stock_data, sector_name, coeffs, seuil_achat, seuil_vente)
                
                avg_gain = total_gain / len(stock_data) if stock_data else 0.0
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
                
                # Log première itération (comme votre code)
                if iteration == 0:
                    logger.info(f"📈 Première itération: Gain={avg_gain:.2f}, Trades={total_trades}, Taux={success_rate:.2f}%")
                
                # Mise à jour du meilleur
                if avg_gain > best_gain:
                    best_gain = avg_gain
                    best_coeffs = coeffs
                    best_thresholds = (seuil_achat, seuil_vente)
                    best_success_rate = success_rate
                    best_trades = total_trades
                
                # Mise à jour barre de progression
                pbar.set_postfix({
                    'Gain': f"{best_gain:.2f}",
                    'Success': f"{best_success_rate:.1f}%",
                    'Trades': best_trades
                })
                pbar.update(1)
        
        logger.info(f"📊 Doublons évités : {doublons}")
        logger.info(f"✅ Meilleurs coefficients : {best_coeffs}, Seuils : {best_thresholds}")
        
        return best_coeffs, best_gain, best_success_rate, best_thresholds, best_trades
    
    def _optimize_coefficients(self, stock_data: Dict, sector_name: str, 
                              fixed_thresholds: Tuple, cycle: int) -> Tuple:
        """Optimise uniquement les coefficients avec seuils fixes."""
        # Implémentation similaire à votre Étape 1
        # ... (code similaire à _optimize_global mais pour coefficients uniquement)
        pass
    
    def _optimize_buy_threshold(self, stock_data: Dict, sector_name: str, 
                               fixed_coeffs: Tuple, current_thresholds: Tuple, cycle: int) -> Tuple:
        """Optimise le seuil d'achat avec coefficients fixes."""
        # Implémentation de votre Étape 2
        # ... 
        pass
    
    def _optimize_sell_threshold(self, stock_data: Dict, sector_name: str, 
                                fixed_coeffs: Tuple, current_thresholds: Tuple, cycle: int) -> Tuple:
        """Optimise le seuil de vente avec coefficients et seuil d'achat fixes."""
        # Implémentation de votre Étape 3
        # ...
        pass
    
    def _evaluate_configuration(self, stock_data: Dict, sector_name: str, 
                               coeffs: Tuple, seuil_achat: float, seuil_vente: float) -> Tuple[float, int, int]:
        """Évalue une configuration sur tous les symboles du secteur."""
        total_gain = 0.0
        total_trades = 0
        total_success = 0
        
        for symbol, data in stock_data.items():
            prices = data['Close']
            volumes = data['Volume']
            
            # Utiliser votre backtest_engine
            result = self.backtest_engine.run_backtest(
                prices=prices,
                volumes=volumes,
                sector=sector_name,
                coefficients=coeffs,
                buy_threshold=seuil_achat,
                sell_threshold=seuil_vente,
                position_size=self.trading_config.position_size
            )
            
            total_gain += result['gain_total']
            total_trades += result['trades']
            total_success += result['winners']
        
        return total_gain, total_trades, total_success
    
    def _save_results(self, sector: str, coeffs: Tuple, gain_total: float, 
                     success_rate: float, total_trades: int, thresholds: Tuple):
        """
        Sauvegarde les résultats d'optimisation.
        Même logique que votre save_optimization_results().
        """
        results = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Sector': sector,
            'Gain_moy': gain_total,
            'Success_Rate': success_rate,
            'Trades': total_trades,
            'Seuil_Achat': thresholds[0],
            'Seuil_Vente': thresholds[1],
            **{f'a{i+1}': coeffs[i] for i in range(8)}
        }
        
        df_new = pd.DataFrame([results])
        csv_path = config.paths.get_results_file_path(config.paths.optimization_csv)
        
        try:
            # Vérification des doublons (votre logique)
            if csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                last_entry = df_existing[df_existing['Sector'] == sector].tail(1)
                
                if not last_entry.empty:
                    # Vérifier si identique (votre logique)
                    same_params = all([
                        last_entry['Gain_moy'].iloc[0] == gain_total,
                        last_entry['Success_Rate'].iloc[0] == success_rate,
                        last_entry['Trades'].iloc[0] == total_trades,
                        last_entry['Seuil_Achat'].iloc[0] == thresholds[0],
                        last_entry['Seuil_Vente'].iloc[0] == thresholds[1],
                        *[last_entry[f'a{i+1}'].iloc[0] == coeffs[i] for i in range(8)]
                    ])
                    
                    if same_params:
                        logger.info(f"📝 Entrée identique pour {sector}, ignorée")
                        return
            
            # Sauvegarder
            df_new.to_csv(csv_path, mode='a', 
                         header=not csv_path.exists(), index=False)
            logger.info(f"📝 Résultats sauvegardés pour {sector}")
            
        except Exception as e:
            logger.error(f"⚠️ Erreur sauvegarde: {e}")

class OptimizationRunner:
    """
    Runner principal pour l'optimisation.
    Remplace votre script principal dans optimisateur_boucle.
    """
    
    def __init__(self):
        self.optimizer = SectorOptimizer()
        self.data_provider = YahooProvider()
        
    def run_optimization(self, symbols_file: str = "optimisation_symbols.txt", 
                        period: str = '1y') -> Dict[str, Tuple]:
        """
        Lance l'optimisation complète.
        Remplace votre logique __main__.
        """
        # Charger les symboles
        symbols_path = config.paths.get_symbol_file_path(symbols_file)
        symbols = self._load_symbols(symbols_path)
        
        if not symbols:
            logger.error(f"❌ Aucun symbole dans {symbols_file}")
            return {}
        
        # Créer dictionnaire des secteurs (votre logique)
        sectors = self._create_sector_dictionary(symbols)
        
        # Optimiser chaque secteur
        optimized_coeffs = {}
        
        for sector_name, sector_symbols in sectors.items():
            if not sector_symbols:
                logger.warning(f"🚫 Secteur {sector_name} vide, ignoré")
                continue
            
            logger.info(f"🚀 Optimisation du secteur: {sector_name}")
            
            coeffs, gain_total, success_rate, thresholds = \
                self.optimizer.optimize_sector_coefficients(
                    sector_symbols, sector_name, period
                )
            
            if coeffs:
                optimized_coeffs[sector_name] = coeffs
                logger.info(f"✅ {sector_name}: Gain={gain_total:.2f}, Taux={success_rate:.2f}%")
        
        # Affichage final (comme votre code)
        self._display_results(optimized_coeffs)
        
        return optimized_coeffs
    
    def _load_symbols(self, file_path: Path) -> List[str]:
        """Charge la liste des symboles depuis un fichier."""
        try:
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            # Supprimer doublons (comme dans votre code)
            return list(dict.fromkeys(symbols))
        except Exception as e:
            logger.error(f"❌ Erreur lecture {file_path}: {e}")
            return []
    
    def _create_sector_dictionary(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Crée le dictionnaire des secteurs.
        Même logique que votre code principal.
        """
        sectors = {
            "Technology": [],
            "Healthcare": [],
            "Financial Services": [],
            "Consumer Cyclical": [],
            "Industrials": [],
            "Energy": [],
            "Basic Materials": [],
            "Communication Services": [],
            "Consumer Defensive": [],
            "Utilities": [],
            "Real Estate": [],
            "ℹ️Inconnu!!": []
        }
        
        # Assigner les symboles aux secteurs
        logger.info("📋 Classification des symboles par secteur...")
        for symbol in symbols:
            sector = self.optimizer.get_sector(symbol)
            if sector in sectors:
                sectors[sector].append(symbol)
            else:
                sectors["ℹ️Inconnu!!"].append(symbol)
        
        # Affichage (comme votre code)
        logger.info("📋 Assignation des secteurs:")
        for sector, syms in sectors.items():
            if syms:
                logger.info(f"  {sector}: {syms}")
        
        return sectors
    
    def _display_results(self, optimized_coeffs: Dict[str, Tuple]):
        """Affiche les résultats finaux (comme votre code)."""
        logger.info("="*50)
        logger.info("🎯 RÉSULTATS D'OPTIMISATION")
        logger.info("="*50)
        
        logger.info("Dictionnaire optimisé pour domain_coeffs:")
        logger.info("{")
        for sector, coeffs in optimized_coeffs.items():
            logger.info(f"  '{sector}': {coeffs},")
        logger.info("}")

# Point d'entrée principal
if __name__ == "__main__":
    runner = OptimizationRunner()
    results = runner.run_optimization()
    
    logger.info(f"🏁 Optimisation terminée pour {len(results)} secteurs")