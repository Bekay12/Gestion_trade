"""
Trading Bot - Modules principaux.
"""
# Import des modules principaux pour faciliter l'utilisation

from src.utils.logger import get_logger, setup_logging
from src.utils.file_manager import SymbolFileManager
from src.utils.cache import DataCacheManager

from src.data.providers.yahoo_provider import YahooProvider
from src.indicators.manager import IndicatorManager
from src.signals.signal_generator import SignalGenerator
from src.signals.signal_analyzer import SignalAnalyzer
from src.backtesting.backtest_engine import BacktestEngine
from src.optimization.optimization_runner import OptimizationRunner
from src.visualization.analysis_charts import AnalysisCharts

# Configuration
from config.settings import config

__version__ = "1.0.0"
__author__ = "Trading Bot User"

__all__ = [
    # Configuration
    'config',
    
    # Utilitaires
    'get_logger', 'setup_logging',
    'SymbolFileManager',
    'DataCacheManager',
    
    # Données
    'YahooProvider',
    
    # Indicateurs
    'IndicatorManager',
    
    # Signaux
    'SignalGenerator',
    'SignalAnalyzer',
    
    # Backtesting
    'BacktestEngine',
    
    # Optimisation
    'OptimizationRunner',
    
    # Visualisation
    'AnalysisCharts',
]