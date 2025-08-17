"""
Gestionnaire de cache pour les données boursières.
Migration de vos fonctions get_cached_data et preload_cache.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import pickle

from config.settings import config
from .logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    Gestionnaire de cache pour les données boursières.
    Migration directe de vos fonctions de cache.
    """

    def __init__(self):
        self.cache_dir = config.paths.cache_dir
        self.max_age_hours = config.trading.cache_max_age_hours
        self.max_workers = config.trading.max_workers

    def get_cached_data(self, symbol: str, period: str, 
                       max_age_hours: Optional[int] = None,
                       offline_mode: bool = False) -> pd.DataFrame:
        """
        Récupère les données en cache si elles existent et sont récentes, sinon télécharge.
        Migration directe de votre fonction get_cached_data().

        Args:
            symbol: Symbole boursier (ex: 'AAPL').
            period: Période des données (ex: '1y').
            max_age_hours: Âge maximum du cache en heures.
            offline_mode: Mode hors ligne.

        Returns:
            pd.DataFrame avec les données, ou DataFrame vide si échec.
        """
        if max_age_hours is None:
            max_age_hours = self.max_age_hours

        cache_file = self.cache_dir / f"{symbol}_{period}.pkl"

        # Vérifier le cache existant
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            age_hours = file_age.total_seconds() / 3600

            if age_hours < max_age_hours or offline_mode:
                try:
                    return pd.read_pickle(cache_file)
                except Exception as e:
                    logger.warning(f"Erreur lecture cache {symbol}: {e}")

        if offline_mode:
            # Si offline, charger le cache même s'il est vieux
            if cache_file.exists():
                try:
                    return pd.read_pickle(cache_file)
                except Exception as e:
                    logger.error(f"Erreur cache offline {symbol}: {e}")
            else:
                logger.warning(f"Pas de cache disponible pour {symbol} ({period}) en mode hors ligne.")
            return pd.DataFrame()

        # Sinon, télécharger et mettre en cache
        try:
            logger.info(f"Téléchargement {symbol} ({period})...")
            data = yf.download(symbol, period=period)

            if not data.empty:
                # Sauvegarder en cache
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                data.to_pickle(cache_file)
                logger.debug(f"Cache sauvegardé: {cache_file}")

            return data

        except Exception as e:
            logger.error(f"Erreur téléchargement {symbol}: {e}")

            # Essayer de charger un cache ancien
            if cache_file.exists():
                try:
                    logger.info(f"Utilisation cache ancien pour {symbol}")
                    return pd.read_pickle(cache_file)
                except Exception as cache_error:
                    logger.error(f"Erreur cache ancien {symbol}: {cache_error}")

            return pd.DataFrame()

    def preload_batch(self, symbols: List[str], period: str) -> None:
        """
        Pré-charge le cache pour une liste de symboles.
        Migration de votre fonction preload_cache().

        Args:
            symbols: Liste des symboles à pré-charger.
            period: Période des données.
        """
        logger.info(f"Pré-chargement du cache pour {len(symbols)} symboles...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.get_cached_data, symbol, period) 
                for symbol in symbols
            ]

            # Attendre que tous les téléchargements soient terminés
            completed = 0
            for future in futures:
                try:
                    result = future.result()
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Pré-chargement: {completed}/{len(symbols)} complétés")
                except Exception as e:
                    logger.warning(f"Erreur pré-chargement: {e}")

        logger.info(f"✅ Pré-chargement terminé: {completed}/{len(symbols)} symboles")

    def clear_cache(self, older_than_days: int = 7) -> None:
        """
        Nettoie le cache des fichiers anciens.

        Args:
            older_than_days: Supprimer les fichiers plus anciens que N jours.
        """
        if not self.cache_dir.exists():
            return

        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                file_date = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_date < cutoff_date:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Erreur suppression cache {cache_file}: {e}")

        logger.info(f"🧹 Cache nettoyé: {removed_count} fichiers supprimés")

    def get_cache_stats(self) -> dict:
        """Retourne des statistiques sur le cache."""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        total_files = 0
        total_size = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                total_files += 1
                total_size += cache_file.stat().st_size
            except Exception:
                continue

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024*1024), 2),
            "cache_dir": str(self.cache_dir)
        }
