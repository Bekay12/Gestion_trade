"""
Provider de données Yahoo Finance.
Migration de votre fonction download_stock_data().
"""
import pandas as pd
import yfinance as yf
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time

from config.settings import config
from src.utils.logger import get_logger
from src.utils.cache import DataCacheManager

logger = get_logger(__name__)

class YahooProvider:
    """
    Provider de données Yahoo Finance.
    Migration complète de votre fonction download_stock_data().
    """
    
    def __init__(self):
        self.cache_manager = DataCacheManager()
        self.batch_size = config.data.batch_size
        self.timeout = config.data.timeout
        self.max_workers = config.data.max_workers

        
        # Périodes valides (de votre code)
        self.valid_periods = [
            '1d', '5d', '1mo', '3mo', '6mo', '12mo', '1y', '14mo', '15mo', '18mo','21mo','24mo', 
            '2y', '5y', '10y', 'ytd', 'max'
        ]
        
        # Suffixes Yahoo valides (de votre code)
        self.yahoo_suffixes = (
            '.HK', '.DE', '.PA', '.AS', '.SW', '.L', '.TO', '.V', '.MI', '.AX', '.SI',
            '.KQ', '.T', '.OL', '.HE', '.ST', '.CO', '.SA', '.MX', '.TW', '.JO', 
            '.SZ', '.NZ', '.KS', '.PL', '.IR', '.MC', '.VI', '.BK', '.SS', '.SG', 
            '.F', '.BE', '.CN', '.TA', '-USD', '=F'
        )
    
    def download_batch(self, symbols: List[str], period: str) -> Dict[str, Dict[str, pd.Series]]:
        """
        Télécharge les données pour une liste de symboles.
        Migration complète de votre fonction download_stock_data().
        
        Args:
            symbols: Liste des symboles boursiers (ex: ['AAPL', 'MSFT']).
            period: Période des données (ex: '1y', '6mo', '1mo').
            
        Returns:
            Dictionnaire avec les données valides: {'symbol': {'Close': pd.Series, 'Volume': pd.Series}}.
        """
        valid_data = {}
        
        # Valider la période
        if period not in self.valid_periods:
            logger.error(f"🚨 Période invalide: {period}. Valeurs possibles: {self.valid_periods}")
            return valid_data
        
        # Filtrer les symboles potentiellement invalides (votre logique)
        valid_symbols = [s for s in symbols 
                        if s and ('.' not in s or s.endswith(self.yahoo_suffixes))]
        
        if len(valid_symbols) < len(symbols):
            invalid = set(symbols) - set(valid_symbols)
            logger.warning(f"🚨 Symboles ignorés (format invalide): {invalid}")
        
        # Diviser les symboles en lots (votre logique)
        symbol_batches = [valid_symbols[i:i + self.batch_size] 
                         for i in range(0, len(valid_symbols), self.batch_size)]
        
        for batch_idx, batch in enumerate(symbol_batches):
            logger.info(f"📊 Traitement lot {batch_idx + 1}/{len(symbol_batches)}: {len(batch)} symboles")
            
            # Essayer téléchargement groupé
            all_data = self._download_batch_group(batch, period)
            
            # Traiter chaque symbole du lot
            for symbol in batch:
                try:
                    # Extraire les données du téléchargement groupé ou du cache
                    if all_data is not None and symbol in all_data:
                        data = all_data[symbol]
                    else:
                        data = self.cache_manager.get_cached_data(symbol, period)
                    
                    # Validation des données (votre logique)
                    validated_data = self._validate_data(symbol, data)
                    if validated_data is not None:
                        valid_data[symbol] = validated_data
                        
                except Exception as e:
                    logger.error(f"🚨 Erreur pour {symbol}: {e}")
            
            # Pause entre les lots pour éviter les limites API
            if batch_idx < len(symbol_batches) - 1:
                time.sleep(1)
        
        logger.info(f"✅ Données téléchargées pour {len(valid_data)} symboles")
        return valid_data
    
    def _download_batch_group(self, batch: List[str], period: str) -> Dict:
        """Téléchargement groupé avec gestion d'erreurs."""
        try:
            # Téléchargement groupé (votre logique)
            all_data = yf.download(
                list(batch),
                period=period,
                group_by='ticker',
                progress=False,
                threads=True,  # Activer le multithreading
                timeout=self.timeout  # Timeout pour éviter les blocages
            )
            return all_data
            
        except Exception as e:
            logger.warning(f"🚨 Erreur téléchargement groupé pour lot {batch[:5]}...: {e}")
            return None
    
    def _validate_data(self, symbol: str, data) -> Dict[str, pd.Series]:
        """
        Valide et nettoie les données d'un symbole.
        Reprend votre logique de validation.
        """
        # Validation des données
        if data is None or data.empty:
            logger.warning(f"🚨 Aucune donnée pour {symbol}")
            return None
        
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            logger.warning(f"🚨 Données incomplètes pour {symbol}: colonnes manquantes")
            return None
        
        # Vérifier la longueur minimale pour get_trading_signal
        if len(data) < config.trading.min_data_points:
            logger.warning(f"🚨 Données insuffisantes pour {symbol} ({len(data)} points)")
            return None
        
        # Nettoyer les données (votre logique)
        clean_data = data[['Close', 'Volume']].copy()
        clean_data['Close'] = clean_data['Close'].ffill()  # Remplir les NaN dans Close
        clean_data['Volume'] = clean_data['Volume'].fillna(0)  # Remplir les NaN dans Volume par 0
        
        # Vérifier les NaN restants
        if clean_data['Close'].isna().all() or clean_data['Volume'].isna().all():
            logger.warning(f"🚨 Données invalides pour {symbol}: trop de valeurs manquantes")
            return None
        
        # Convertir en Series si nécessaire (votre logique)
        return {
            'Close': clean_data['Close'].squeeze(),
            'Volume': clean_data['Volume'].squeeze()
        }
    
    def download_single(self, symbol: str, period: str) -> Dict[str, pd.Series]:
        """Télécharge les données pour un seul symbole."""
        result = self.download_batch([symbol], period)
        return result.get(symbol, {})

# Instance globale
yahoo_provider = YahooProvider()

# Fonction de compatibilité
def download_stock_data(symbols: List[str], period: str) -> Dict[str, Dict[str, pd.Series]]:
    """Fonction de compatibilité avec votre code existant."""
    return yahoo_provider.download_batch(symbols, period)