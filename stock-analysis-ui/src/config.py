"""
Configuration centralisée pour le projet stock-analysis.
Tous les chemins et constantes globales en un seul endroit.
"""

from pathlib import Path
import os

# ===================================================================
# CHEMINS BASE
# ===================================================================

# Utiliser un chemin absolu basé sur le répertoire de ce fichier config.py
_CONFIG_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _CONFIG_DIR  # Alias pour compatibilité
DB_PATH = str(_CONFIG_DIR / "stock_analysis.db")
OPTIMIZATION_DB_PATH = str(_CONFIG_DIR / "signaux" / "optimization_hist.db")
CACHE_DIR = _CONFIG_DIR / "cache_data"
DATA_CACHE_DIR = _CONFIG_DIR / "data_cache"
CACHE_LOGS_DIR = _CONFIG_DIR / "cache_logs"
SIGNALS_DIR = _CONFIG_DIR / "signaux"

# Créer les répertoires s'ils n'existent pas
CACHE_DIR.mkdir(exist_ok=True)
DATA_CACHE_DIR.mkdir(exist_ok=True)
CACHE_LOGS_DIR.mkdir(exist_ok=True)
SIGNALS_DIR.mkdir(exist_ok=True)

# ===================================================================
# CACHE DISQUE
# ===================================================================

SECTOR_CACHE_FILE = CACHE_DIR / "sector_cache.json"
CACHE_INDEX_FILE = CACHE_DIR / "cache_index.json"

# ===================================================================
# FICHIERS DE SYMBOLES
# ===================================================================

POPULAR_SYMBOLS_FILE = str(_CONFIG_DIR / "popular_symbols.txt")
PERSONAL_SYMBOLS_FILE = str(_CONFIG_DIR / "mes_symbols.txt")
OPTIMIZATION_SYMBOLS_FILE = str(_CONFIG_DIR / "optimisation_symbols.txt")
SP500_SYMBOLS_FILE = str(_CONFIG_DIR / "sp500_symbols.txt")

# ===================================================================
# PARAMETRES DE CACHE
# ===================================================================

SECTOR_TTL_DAYS = 30
SECTOR_TTL_UNKNOWN_DAYS = 7

CACHE_TTL_FINANCIAL_DAYS = 30
CACHE_TTL_MARKET_DATA_DAYS = 1

# ===================================================================
# PARAMETRES DE CAPITALISATION
# ===================================================================

# Seuils de classification de la market cap (en milliards $)
CAP_RANGE_THRESHOLDS = {
    'Small': (0, 2),
    'Mid': (2, 10),
    'Large': (10, 200),
    'Mega': (200, float('inf'))
}

CAP_RANGE_LABELS = ['Small', 'Mid', 'Large', 'Mega', 'Unknown']

# ===================================================================
# FICHIERS SIGNAUX ET RESULTATS
# ===================================================================

OPTIMIZATION_CSV = SIGNALS_DIR / "optimization_hist_4stp.csv"
SIGNAUX_CSV = "signaux_trading.csv"

# ===================================================================
# PARAMETERS PAR DEFAUT
# ===================================================================

DEFAULT_TRAINING_MONTHS = 12
DEFAULT_MIN_HOLD_DAYS = 14
DEFAULT_VOLUME_MIN = 100000
DEFAULT_RELIABILITY_THRESHOLD = 60.0
DEFAULT_TRAILING_MONTHS = 9
DEFAULT_RECALC_RELIABILITY_EVERY = 5


# ===================================================================
# CACHE UTILITIES (PICKLE)
# ===================================================================

import pandas as pd
from datetime import datetime, timedelta

def get_pickle_cache(symbol: str, cache_type: str = 'financial', ttl_hours: int = 24) -> pd.DataFrame:
    """Charge depuis le cache pickle s'il existe et n'est pas expire.
    
    Args:
        symbol: Ticker
        cache_type: Type de cache ('financial', 'consensus', etc.)
        ttl_hours: Time-to-live en heures
    
    Returns:
        Data si cache valide, None sinon
    """
    try:
        cache_file = DATA_CACHE_DIR / f"{symbol}_{cache_type}.pkl"
        if cache_file.exists():
            age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if age_hours < ttl_hours:
                return pd.read_pickle(cache_file)
    except Exception:
        pass
    return None

def save_pickle_cache(data: pd.DataFrame, symbol: str, cache_type: str = 'financial') -> bool:
    """Sauvegarde data dans le cache pickle.
    
    Args:
        data: Data a sauvegarder
        symbol: Ticker
        cache_type: Type de cache
    
    Returns:
        True si succes, False sinon
    """
    try:
        cache_file = DATA_CACHE_DIR / f"{symbol}_{cache_type}.pkl"
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(data, cache_file)
        return True
    except Exception:
        return False

