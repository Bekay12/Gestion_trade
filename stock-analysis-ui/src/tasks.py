"""
Celery Tasks: tâches asynchrones pour analyse, backtest, download
Exécutées par le worker séparé pour ne pas bloquer l'API
"""
import os
import sys
import logging
from datetime import timedelta
from pathlib import Path

from celery import Celery, group
from celery.schedules import crontab
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery app
app = Celery('stock_analysis')
app.conf.update(
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_soft_time_limit=300,  # 5 min soft limit
    task_time_limit=600,        # 10 min hard limit
)

# Periodic tasks (Celery Beat schedule)
app.conf.beat_schedule = {
    'refresh-popular-symbols-hourly': {
        'task': 'src.tasks.refresh_popular_symbols',
        'schedule': crontab(minute=0),  # Chaque heure
    },
    'cache-sp500-daily': {
        'task': 'src.tasks.cache_sp500_symbols',
        'schedule': crontab(hour=9, minute=0),  # 9h chaque matin (heure serveur)
    },
}


@app.task(bind=True, name='src.tasks.download_stock_data')
def download_stock_data_task(self, symbol: str, period: str = '1mo'):
    """
    Télécharge les données d'un symbole (tâche longue)
    Utilise le cache yfinance_helper avec retry automatique
    """
    try:
        from yfinance_helper import download_stock_data
        
        logger.info(f"[TASK] download_stock_data: {symbol} ({period})")
        result = download_stock_data([symbol], period=period)
        
        if result and symbol in result:
            logger.info(f"[TASK] ✓ {symbol} téléchargé ({len(result[symbol])} lignes)")
            return {
                'status': 'success',
                'symbol': symbol,
                'rows': len(result[symbol]),
                'task_id': self.request.id
            }
        else:
            logger.error(f"[TASK] ✗ Aucune donnée pour {symbol}")
            raise Exception(f"No data for {symbol}")
            
    except Exception as e:
        logger.error(f"[TASK] Erreur download {symbol}: {e}")
        # Retry avec backoff exponentiel (3 tentatives)
        raise self.retry(exc=e, countdown=2 ** self.request.retries, max_retries=3)


@app.task(bind=True, name='src.tasks.analyze_symbol')
def analyze_symbol_task(self, symbol: str, period: str = '1mo'):
    """
    Analyse complète d'un symbole (download + calcul signal)
    Tâche longue exécutée dans le worker
    """
    try:
        from yfinance_helper import download_stock_data
        from qsi import get_trading_signal, get_cap_range_for_symbol, extract_best_parameters
        
        logger.info(f"[TASK] analyze_symbol: {symbol}")
        
        # Télécharger
        stock_data_dict = download_stock_data([symbol], period=period)
        if not stock_data_dict or symbol not in stock_data_dict:
            raise Exception(f"Cannot download {symbol}")
        
        stock_data = stock_data_dict[symbol]
        prices = stock_data['Close']
        volumes = stock_data['Volume']
        
        # Récupérer secteur
        try:
            info = yf.Ticker(symbol).info
            domaine = info.get("sector", "Inconnu")
        except:
            domaine = "Inconnu"
        
        # Cap range & fallbacks
        cap_range = get_cap_range_for_symbol(symbol)
        if cap_range == "Unknown" or not cap_range:
            best_params = extract_best_parameters()
            for fb in ["Large", "Mid", "Mega"]:
                if f"{domaine}_{fb}" in best_params:
                    cap_range = fb
                    break
        
        # Fallback domaine
        if domaine == "Inconnu":
            best_params = extract_best_parameters()
            for fb in ["Technology", "Healthcare", "Financial Services"]:
                if fb in best_params:
                    domaine = fb
                    break
        
        # Seuils optimisés
        seuil_achat, seuil_vente = None, None
        best_params = extract_best_parameters()
        param_key = f"{domaine}_{cap_range}" if cap_range else domaine
        if param_key in best_params:
            params = best_params[param_key]
            if len(params) > 2 and params[2]:
                th = params[2]
                if isinstance(th, (tuple, list)) and len(th) >= 2:
                    seuil_achat, seuil_vente = float(th[0]), float(th[1])
        
        # Signal
        sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
            prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol,
            cap_range=cap_range, seuil_achat=seuil_achat, seuil_vente=seuil_vente
        )
        
        logger.info(f"[TASK] ✓ {symbol}: {sig} (score: {score})")
        return {
            'status': 'success',
            'symbol': symbol,
            'signal': sig,
            'score': float(score) if score else 0.0,
            'task_id': self.request.id
        }
        
    except Exception as e:
        logger.error(f"[TASK] Erreur analyze {symbol}: {e}")
        raise self.retry(exc=e, countdown=5, max_retries=2)


@app.task(name='src.tasks.refresh_popular_symbols')
def refresh_popular_symbols():
    """
    Pré-calcule les signaux des symboles populaires (job cron)
    Exécuté toutes les heures - remplit le cache pour requêtes rapides
    """
    try:
        from config import SIGNALS_DIR
        from pathlib import Path
        
        popular = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B', 'JNJ', 'V']
        
        logger.info(f"[CRON] Refresh {len(popular)} popular symbols")
        
        # Lancer les analyses en parallèle
        tasks = group([analyze_symbol_task.s(sym, '6mo') for sym in popular])
        results = tasks.apply_async()
        
        logger.info(f"[CRON] ✓ {len(popular)} analyses lancées en arrière-plan")
        return {'status': 'scheduled', 'symbols': popular}
        
    except Exception as e:
        logger.error(f"[CRON] Erreur refresh: {e}")
        return {'status': 'error', 'error': str(e)}


@app.task(name='src.tasks.cache_sp500_symbols')
def cache_sp500_symbols():
    """
    Pré-télécharge le top 20 du S&P 500 (job cron quotidien)
    """
    try:
        from yfinance_helper import download_stock_data
        
        sp500_top = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK.B', 'TSLA', 'META', 
                     'JNJ', 'V', 'WMT', 'MA', 'PG', 'COST', 'AVGO', 'JPM', 'ABBV', 'ACN', 'NFLX', 'CRM']
        
        logger.info(f"[CRON] Cache S&P500 top 20")
        
        # Télécharger en une opération
        result = download_stock_data(sp500_top, period='2y')
        
        if result:
            logger.info(f"[CRON] ✓ {len(result)} symboles cachés")
            return {'status': 'success', 'symbols': list(result.keys())}
        else:
            raise Exception("Download failed")
            
    except Exception as e:
        logger.error(f"[CRON] Erreur cache S&P500: {e}")
        return {'status': 'error', 'error': str(e)}
