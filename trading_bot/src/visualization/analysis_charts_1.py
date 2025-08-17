"""
Analyseur et afficheur de graphiques techniques.
Migration de votre fonction analyse_et_affiche().
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from src.data.providers.yahoo_provider import YahooProvider
from src.visualization.chart_plotter import ChartPlotter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnalysisCharts:
    """
    Analyseur et afficheur de graphiques techniques.
    Migration complète de votre fonction analyse_et_affiche().
    """
    
    def __init__(self):
        self.yahoo_provider = YahooProvider()
        self.chart_plotter = ChartPlotter()
    
    def analyse_et_affiche(self, symbols: List[str], period: str = "12mo"):
        """
        Télécharge les données pour les symboles donnés et affiche les graphiques d'analyse technique.
        Migration exacte de votre fonction analyse_et_affiche().
        
        Args:
            symbols: Liste des symboles à analyser.
            period: Période des données.
        """
        logger.info("⏳ Téléchargement des données...")
        
        # Télécharger les données (votre logique)
        data = self.yahoo_provider.download_batch(symbols, period)
        
        if not data:
            logger.error("❌ Aucune donnée valide disponible. Vérifiez les symboles ou la connexion internet.")
            return
        
        num_plots = len(data)
        
        # Créer la figure avec sous-graphiques (votre logique exacte)
        fig, axes = self.chart_plotter.create_multi_chart_figure(num_plots)
        
        if num_plots == 0:
            logger.error("❌ Aucun symbole valide à afficher")
            return
        
        # Tracer chaque symbole (votre logique exacte)
        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            
            logger.info(f"📊 Traitement de {symbol}...")
            
            show_xaxis = (i == len(data) - 1)  # True seulement pour le dernier subplot
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)
        
        # Finaliser et afficher (votre logique exacte)
        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.show_figure()
    
    def display_signal_charts(self, signals: List[Dict], period: str = "12mo", 
                             max_charts: int = 5, signal_type: str = "ACHAT"):
        """
        Affiche les graphiques pour des signaux spécifiques.
        Inspiré de votre logique d'affichage des top signaux.
        
        Args:
            signals: Liste des signaux.
            period: Période des données.
            max_charts: Nombre maximum de graphiques.
            signal_type: Type de signal ("ACHAT" ou "VENTE").
        """
        # Filtrer les signaux par type
        filtered_signals = [s for s in signals if s.get('Signal') == signal_type][:max_charts]
        
        if not filtered_signals:
            logger.warning(f"Aucun signal {signal_type} à afficher")
            return
        
        logger.info(f"\\nAffichage des graphiques pour les {len(filtered_signals)} premiers signaux {signal_type} détectés...")
        
        # Télécharger les données pour ces signaux
        symbols = [s['Symbole'] for s in filtered_signals]
        data = self.yahoo_provider.download_batch(symbols, period)
        
        if not data:
            logger.error("❌ Aucune donnée disponible pour les signaux")
            return
        
        # Créer la figure
        fig, axes = self.chart_plotter.create_multi_chart_figure(len(data))
        
        # Tracer chaque graphique
        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            
            show_xaxis = (i == len(data) - 1)
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)
        
        # Finaliser et afficher
        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.show_figure()
    
    def save_analysis_charts(self, symbols: List[str], period: str = "12mo", 
                           filename: str = "analysis_charts.png"):
        """
        Sauvegarde les graphiques d'analyse au lieu de les afficher.
        
        Args:
            symbols: Liste des symboles.
            period: Période des données.
            filename: Nom du fichier de sauvegarde.
        """
        logger.info("⏳ Génération des graphiques pour sauvegarde...")
        
        data = self.yahoo_provider.download_batch(symbols, period)
        
        if not data:
            logger.error("❌ Aucune donnée valide disponible")
            return
        
        num_plots = len(data)
        fig, axes = self.chart_plotter.create_multi_chart_figure(num_plots)
        
        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            
            show_xaxis = (i == len(data) - 1)
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)
        
        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.save_figure(filename)
        plt.close(fig)  # Libérer la mémoire

# Instance globale
analysis_charts = AnalysisCharts()

# Fonction de compatibilité
def analyse_et_affiche(symbols: List[str], period: str = "12mo"):
    """Fonction de compatibilité avec votre code existant."""
    analysis_charts.analyse_et_affiche(symbols, period)
'''

analysis_charts_file = Path("trading_bot/src/visualization/analysis_charts.py")
analysis_charts_file.write_text(analysis_charts_code)

# __init__.py pour visualization
visualization_init = '''"""
Modules de visualisation et graphiques.
"""
from .chart_plotter import ChartPlotter, chart_plotter, plot_unified_chart
from .analysis_charts import AnalysisCharts, analysis_charts, analyse_et_affiche

__all__ = [
    'ChartPlotter', 'chart_plotter', 'plot_unified_chart',
    'AnalysisCharts', 'analysis_charts', 'analyse_et_affiche'
]
'''

visualization_init_file = Path("trading_bot/src/visualization/__init__.py")
visualization_init_file.write_text(visualization_init)

# 19. Scripts utilitaires
Path("trading_bot/scripts").mkdir(parents=True, exist_ok=True)

# Script run_optimization.py
run_optimization_script = '''#!/usr/bin/env python3
"""
Script de lancement de l'optimisation.
Remplace votre optimisateur_boucle en tant que script standalone.
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.optimization_runner import optimization_runner
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Lance l'optimisation complète."""
    logger.info("🚀 Démarrage du script d'optimisation")
    
    try:
        results = optimization_runner.run_full_optimization()
        
        if results:
            logger.info(f"✅ Optimisation terminée avec succès: {len(results)} secteurs optimisés")
        else:
            logger.warning("⚠️ Aucun résultat d'optimisation obtenu")
    
    except Exception as e:
        logger.error(f"❌ Erreur durant l'optimisation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

run_optimization_file = Path("trading_bot/scripts/run_optimization.py")
run_optimization_file.write_text(run_optimization_script)

# Script run_analysis.py
run_analysis_script = '''#!/usr/bin/env python3
"""
Script de lancement de l'analyse de signaux.
Remplace votre test.py en tant que script standalone.
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.utils.cache import DataCacheManager
from src.utils.logger import get_logger
from config.settings import config

logger = get_logger(__name__)

def main():
    """Lance l'analyse des signaux."""
    logger.info("🔍 Démarrage du script d'analyse")
    
    try:
        # Charger les symboles
        symbol_manager = SymbolFileManager()
        test_symbols = symbol_manager.load_symbols_from_txt("test_symbols.txt")
        mes_symbols = symbol_manager.load_symbols_from_txt("mes_symbols.txt")
        
        if not test_symbols:
            logger.error("❌ Aucun symbole de test trouvé")
            return
        
        logger.info(f"📋 Analyse de {len(test_symbols)} symboles de test et {len(mes_symbols)} symboles personnels")
        
        # Précharger le cache
        cache_manager = DataCacheManager()
        period = config.trading.default_period
        
        logger.info("⏳ Préchargement du cache...")
        cache_manager.preload_cache(test_symbols + mes_symbols, period)
        
        # Lancer l'analyse
        results = signal_analyzer.analyze_popular_signals(
            test_symbols, 
            mes_symbols, 
            period, 
            plot_all=True
        )
        
        if results.get('signals'):
            logger.info(f"✅ Analyse terminée: {len(results['signals'])} signaux détectés")
        else:
            logger.warning("⚠️ Aucun signal détecté")
    
    except Exception as e:
        logger.error(f"❌ Erreur durant l'analyse: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()