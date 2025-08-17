"""
Analyseur de signaux - Analyse des signaux sur plusieurs symboles.
Migration de votre fonction analyse_signaux_populaires().
"""
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
import yfinance as yf

from config.settings import config
from src.data.providers.yahoo_provider import YahooProvider
from src.signals.signal_generator import SignalGenerator
from src.backtesting.backtest_engine import BacktestEngine
from src.signals.signal_saver import SignalSaver
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SignalAnalyzer:
    """
    Analyseur de signaux pour actions multiples.
    Migration complète de votre fonction analyse_signaux_populaires().
    """
    
    def __init__(self):
        self.yahoo_provider = YahooProvider()
        self.signal_generator = SignalGenerator()
        self.backtest_engine = BacktestEngine()
        self.signal_saver = SignalSaver()
        self.config = config.trading
    
    def analyze_popular_signals(self, popular_symbols: List[str], mes_symbols: List[str],
                               period: str = "12mo", display_charts: bool = True,
                               chunk_size: int = 20, verbose: bool = True,
                               save_csv: bool = True, plot_all: bool = False) -> Dict[str, Any]:
        """
        Analyse les signaux pour les actions populaires.
        Migration complète de votre fonction analyse_signaux_populaires().
        
        Args:
            popular_symbols: Liste des symboles populaires.
            mes_symbols: Liste de vos symboles personnels.
            period: Période d'analyse.
            display_charts: Afficher les graphiques.
            chunk_size: Taille des lots de traitement.
            verbose: Mode verbeux.
            save_csv: Sauvegarder en CSV.
            plot_all: Afficher tous les graphiques.
            
        Returns:
            Dictionnaire avec tous les résultats d'analyse.
        """
        if verbose:
            logger.info("🔍 Analyse des signaux pour actions populaires...")
        
        # Extraire les meilleurs paramètres (votre logique)
        best_parameters = config._load_sector_parameters()
        
        if verbose:
            logger.info("Extraction des meilleurs paramètres depuis le CSV:")
            logger.info("Dictionnaire des meilleurs paramètres:")
            for sector, (coeffs, thresholds, gain_moy) in best_parameters.items():
                logger.info(f"'{sector}': coefficients={coeffs}, thresholds={thresholds}, Gain_moy={gain_moy:.2f}")
        
        # Analyser les signaux
        signals = []
        
        for i in range(0, len(popular_symbols), chunk_size):
            chunk = popular_symbols[i:i+chunk_size]
            
            if verbose:
                logger.info(f"🔎 Lot {i//chunk_size + 1}: {', '.join(chunk)}")
            
            try:
                # Télécharger les données pour ce lot
                chunk_data = self.yahoo_provider.download_batch(chunk, period)
                
                for symbol, stock_data in chunk_data.items():
                    prices = stock_data['Close']
                    volumes = stock_data['Volume']
                    
                    if len(prices) < self.config.min_data_points:
                        continue
                    
                    # Récupérer le secteur (votre logique)
                    try:
                        info = yf.Ticker(symbol).info
                        sector = info.get("sector", "ℹ️Inconnu!!")
                    except Exception:
                        sector = "ℹ️Inconnu!!"
                    
                    # Générer le signal
                    result = self.signal_generator.generate_signal(prices, volumes, sector)
                    
                    # Mapper vers vos signaux originaux
                    signal_map = {"BUY": "ACHAT", "SELL": "VENTE", "HOLD": "NEUTRE"}
                    signal = signal_map.get(result['signal'], result['signal'])
                    
                    if signal != "NEUTRE":
                        signals.append({
                            'Symbole': symbol,
                            'Signal': signal,
                            'Score': result['score'],
                            'Prix': result['price'],
                            'Tendance': "Hausse" if result['trend'] else "Baisse",
                            'RSI': result['rsi'],
                            'Domaine': sector,
                            'Volume moyen': result['volume_mean']
                        })
                        
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️ Erreur: {str(e)}")
            
            time.sleep(0.5)  # Pause entre les lots
        
        # Organiser les signaux (votre logique)
        organized_signals = self._organize_signals(signals)
        
        # Afficher les résultats
        if signals and verbose:
            self._display_results(organized_signals, mes_symbols)
        elif verbose:
            logger.info("ℹ️ Aucun signal fort détecté parmi les actions populaires")
            return {}
        
        # Effectuer le backtest sur les signaux détectés
        backtest_results = self._perform_backtest_analysis(signals, period, verbose)
        
        # Filtrer les signaux fiables
        reliable_signals = self._filter_reliable_signals(signals, backtest_results)
        
        # Sauvegarde CSV
        if reliable_signals and save_csv:
            self._save_signals(reliable_signals, mes_symbols, verbose)
        
        return {
            "signals": signals,
            "organized_signals": organized_signals,
            "backtest_results": backtest_results,
            "reliable_signals": reliable_signals,
            "best_parameters": best_parameters
        }
    
    def _organize_signals(self, signals: List[Dict]) -> Dict:
        """Organise les signaux par type et tendance."""
        organized = {"ACHAT": {"Hausse": [], "Baisse": []}, "VENTE": {"Hausse": [], "Baisse": []}}
        
        for s in signals:
            if s['Signal'] in organized and s['Tendance'] in organized[s['Signal']]:
                organized[s['Signal']][s['Tendance']].append(s)
        
        # Trier par prix
        for signal_type in ["ACHAT", "VENTE"]:
            for tendance in ["Hausse", "Baisse"]:
                if organized[signal_type][tendance]:
                    organized[signal_type][tendance].sort(key=lambda x: x['Prix'])
        
        return organized
    
    def _display_results(self, organized_signals: Dict, mes_symbols: List[str]):
        """Affiche les résultats formatés (votre logique exacte)."""
        logger.info("=" * 115)
        logger.info("RÉSULTATS DES SIGNAUX")
        logger.info("=" * 115)
        logger.info(f"{'Symbole':<8} {'Signal':<8} {'Score':<7} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Volume moyen':<15} {'Domaine':<24} Analyse")
        logger.info("-" * 115)
        
        for signal_type in ["ACHAT", "VENTE"]:
            for tendance in ["Hausse", "Baisse"]:
                if organized_signals[signal_type][tendance]:
                    logger.info(f"\\n------------------------------------ Signal {signal_type} | Tendance {tendance} ------------------------------------")
                    
                    for s in organized_signals[signal_type][tendance]:
                        special_marker = "‼️ " if s['Symbole'] in mes_symbols else ""
                        analysis = ""
                        
                        if s['Signal'] == "ACHAT":
                            analysis += "RSI bas" if s['RSI'] < 40 else ""
                            analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                        else:
                            analysis += "RSI élevé" if s['RSI'] > 60 else ""
                            analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""
                        
                        logger.info(f" {s['Symbole']:<8} {s['Signal']}{special_marker:<3} {s['Score']:<7.2f} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Volume moyen']:<15,.0f} {s['Domaine']:<24} {analysis}")
        
        logger.info("=" * 115)
    
    def _perform_backtest_analysis(self, signals: List[Dict], period: str, verbose: bool) -> List[Dict]:
        """Effectue l'analyse de backtest sur tous les signaux."""
        backtest_results = []
        total_trades = 0
        total_gagnants = 0
        total_gain = 0.0
        
        for s in signals:
            try:
                # Télécharger les données pour ce symbole
                stock_data = self.yahoo_provider.download_batch([s['Symbole']], period)
                if s['Symbole'] not in stock_data:
                    continue
                
                symbol_data = stock_data[s['Symbole']]
                prices = symbol_data['Close']
                volumes = symbol_data['Volume']
                
                # Récupérer le secteur pour le backtest
                try:
                    info = yf.Ticker(s['Symbole']).info
                    sector = info.get("sector", "Inconnu")
                except Exception:
                    sector = "Inconnu"
                
                if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
                    if verbose:
                        logger.warning(f"{s['Symbole']:<8} : Données insuffisantes pour le backtest")
                    continue
                
                # Effectuer le backtest
                result = self.backtest_engine.run_backtest(prices, volumes, sector)
                
                result_entry = {
                    "Symbole": s['Symbole'],
                    "trades": result['trades'],
                    "gagnants": result['gagnants'],
                    "taux_reussite": result['taux_reussite'],
                    "gain_total": result['gain_total'],
                    "gain_moyen": result['gain_moyen'],
                    "drawdown_max": result['drawdown_max']
                }
                
                backtest_results.append(result_entry)
                total_trades += result['trades']
                total_gagnants += result['gagnants']
                total_gain += result['gain_total']
                
            except Exception as e:
                if verbose:
                    logger.error(f"{s['Symbole']:<8} : Erreur backtest ({e})")
        
        # Trier par taux de réussite
        backtest_results.sort(key=lambda x: x['taux_reussite'], reverse=True)
        
        # Afficher les résultats de backtest
        if verbose and backtest_results:
            self._display_backtest_results(backtest_results, total_trades, total_gagnants, total_gain)
        
        return backtest_results
    
    def _display_backtest_results(self, backtest_results: List[Dict], total_trades: int, 
                                 total_gagnants: int, total_gain: float):
        """Affiche les résultats du backtest."""
        for res in backtest_results:
            logger.info(f"{res['Symbole']:<8} | Trades: {res['trades']:<2} | "
                       f"Gagnants: {res['gagnants']:<2} | "
                       f"Taux réussite: {res['taux_reussite']:.0f}% | "
                       f"Gain total: {res['gain_total']:.2f} $ | "
                       f"Gain moyen: {res['gain_moyen']:.2f} $ | "
                       f"Drawdown max: {res['drawdown_max']:.2f}%")
        
        cout_par_trade = 1.0
        cout_total_trades = total_trades * cout_par_trade
        total_investi_reel = len(backtest_results) * 50
        gain_total_reel = total_gain - cout_total_trades
        
        if total_trades > 0:
            taux_global = total_gagnants / total_trades * 100
            logger.info("\\n" + "=" * 115)
            logger.info("🌍 Résultat global :")
            logger.info(f" - Taux de réussite = {taux_global:.1f}%")
            logger.info(f" - Nombre de trades = {total_trades}")
            logger.info(f" - Total investi réel = {total_investi_reel:.2f} $ (50 $ par action analysée)")
            logger.info(f" - Coût total des trades = {cout_total_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
            logger.info(f" - Gain total brut = {total_gain:.2f} $")
            logger.info(f" - Gain total net (après frais) = {gain_total_reel:.2f} $")
            logger.info("=" * 115)
    
    def _filter_reliable_signals(self, signals: List[Dict], backtest_results: List[Dict]) -> List[Dict]:
        """Filtre les signaux fiables (>=60% taux de réussite ou non évalués)."""
        # Créer un ensemble des symboles fiables
        reliable_symbols = set()
        for res in backtest_results:
            if res['taux_reussite'] >= 60 or res['trades'] == 0:
                reliable_symbols.add(res['Symbole'])
        
        # Filtrer et enrichir les signaux
        reliable_signals = []
        for s in signals:
            if s['Symbole'] in reliable_symbols:
                # Ajouter le taux de fiabilité
                taux_fiabilite = next(
                    (res['taux_reussite'] for res in backtest_results if res['Symbole'] == s['Symbole']),
                    "N/A"
                )
                
                signal_copy = s.copy()
                signal_copy['Fiabilite'] = taux_fiabilite
                reliable_signals.append(signal_copy)
        
        return reliable_signals
    
    def _save_signals(self, reliable_signals: List[Dict], mes_symbols: List[str], verbose: bool):
        """Sauvegarde les signaux fiables."""
        if verbose:
            logger.info(f"💾 Sauvegarde de {len(reliable_signals)} signaux validés par le backtest...")
        
        # Sauvegarde principale
        self.signal_saver.save_to_evolutive_csv(reliable_signals)
        
        # Sauvegarde spéciale pour vos symboles personnels
        mes_signaux_valides = [s for s in reliable_signals if s['Symbole'] in mes_symbols]
        if mes_signaux_valides:
            special_filename = "mes_signaux_fiables_.csv"
            if verbose:
                logger.info(f"💠 Sauvegarde de {len(mes_signaux_valides)} signaux personnels fiables dans {special_filename}")
            self.signal_saver.save_to_evolutive_csv(mes_signaux_valides, special_filename)

# Instance globale
signal_analyzer = SignalAnalyzer()

# Fonction de compatibilité
def analyse_signaux_populaires(popular_symbols: List[str], mes_symbols: List[str],
                              period: str = "12mo", afficher_graphiques: bool = True,
                              chunk_size: int = 20, verbose: bool = True,
                              save_csv: bool = True, plot_all: bool = False) -> Dict[str, Any]:
    """Fonction de compatibilité avec votre code existant."""
    return signal_analyzer.analyze_popular_signals(
        popular_symbols, mes_symbols, period, afficher_graphiques, 
        chunk_size, verbose, save_csv, plot_all
    )