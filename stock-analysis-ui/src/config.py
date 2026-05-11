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
MARKET_DATA_DB_PATH = DB_PATH
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
# CACHE UTILITIES — backend Parquet (via market_store)
# ===================================================================
# Les anciennes fonctions get_pickle_cache / save_pickle_cache sont
# conservées pour compatibilité des imports existants, mais délèguent
# désormais vers market_store.get_financial_cache /
# market_store.save_financial_cache (Parquet, market_parquet/financial_cache/).
#
# Le cache pickle (data_cache/*.pkl) n'est plus utilisé en écriture.
# Les fichiers pkl existants restent lisibles pendant la période de
# transition via le fallback ci-dessous.

import pandas as pd
from datetime import datetime, timedelta


def get_pickle_cache(symbol: str, cache_type: str = 'financial', ttl_hours: int = 24):
    """Lit le cache financier depuis Parquet (remplace pickle).

    Fallback automatique vers l'ancien fichier pkl si le cache Parquet
    n'existe pas encore pour ce symbole.
    """
    # Tentative 1 : Parquet via market_store
    try:
        import market_store as _ms
        result = _ms.get_financial_cache(symbol, cache_type=cache_type, ttl_hours=ttl_hours)
        if result is not None:
            return result
    except Exception:
        pass

    # Tentative 2 : fallback lecture pickle legacy (lecture seule, pas d'écriture)
    try:
        cache_file = DATA_CACHE_DIR / f"{symbol}_{cache_type}.pkl"
        if cache_file.exists():
            age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if age_hours < ttl_hours:
                return pd.read_pickle(cache_file)
    except Exception:
        pass

    return None


def save_pickle_cache(data, symbol: str, cache_type: str = 'financial') -> bool:
    """Sauvegarde le cache financier en Parquet (remplace pickle).

    *data* peut être un dict ou un DataFrame.  Les DataFrames sont
    convertis en dict avant stockage (première ligne).
    """
    try:
        import market_store as _ms
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return False
            payload = data.iloc[0].to_dict()
        elif isinstance(data, dict):
            payload = data
        else:
            return False
        return _ms.save_financial_cache(symbol, payload, cache_type=cache_type)
    except Exception:
        return False

