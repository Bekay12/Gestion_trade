"""
Trading Bot - Modules principaux.
"""
# Import des modules principaux pour faciliter l'utilisation

from .utils.logger import get_logger, setup_logging
from .utils.file_manager import SymbolFileManager
from .utils.cache import DataCacheManager, data_cache_manager

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
    'DataCacheManager', 'data_cache_manager',
]