"""
Configuration centralisée pour le projet stock-analysis.
Tous les chemins et constantes globales en un seul endroit.
"""

from pathlib import Path

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
# FALLBACK CONTROLS (UI + API)
# ===================================================================

# Autoriser le fallback de domaine lorsque le secteur est "Inconnu"
DOMAIN_FALLBACK_ENABLED = False

# Autoriser le fallback de cap_range lorsque la capitalisation est "Unknown"
CAP_FALLBACK_ENABLED = False


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
