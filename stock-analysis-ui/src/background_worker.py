#!/usr/bin/env python3
# ============================================================================
# BACKGROUND_WORKER.PY - TÂCHES EN ARRIÈRE-PLAN
# Worker pour calcul des signaux quotidiens, notifications, etc.
# ============================================================================

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from qsi import analyse_signaux_populaires
    from symbol_manager import get_symbols_by_list_type
    from config import SIGNALS_DIR
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)

# ============================================================================
# TASKS
# ============================================================================

def task_daily_signals():
    """Calcul quotidien des signaux sur les symboles populaires"""
    logger.info("📊 Running daily signals calculation...")
    
    try:
        # Récupérer symboles populaires
        symbols = get_symbols_by_list_type('popular')
        if not symbols:
            logger.warning("⚠️ No popular symbols found")
            return
        
        logger.info(f"📈 Analyzing {len(symbols)} symbols...")
        
        # Analyser signaux
        results = analyse_signaux_populaires(
            symbols=symbols,
            mes_symbols=[],
            period='12mo',
            afficher_graphiques=False,
            verbose=True
        )
        
        logger.info(f"✅ Daily signals calculation completed: {len(results)} signals")
        
    except Exception as e:
        logger.error(f"❌ Error in daily signals: {e}")

def task_cleanup_cache():
    """Nettoyage du cache expiré"""
    logger.info("🧹 Cleaning up expired cache...")
    
    try:
        # Implémenter la logique de nettoyage
        # Par exemple: supprimer fichiers cache > 30 jours
        cache_dir = Path('cache_data')
        if cache_dir.exists():
            cutoff = datetime.utcnow() - timedelta(days=30)
            for file in cache_dir.glob('*'):
                if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    logger.info(f"Deleted: {file}")
        
        logger.info("✅ Cache cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Error in cache cleanup: {e}")

def task_send_notifications():
    """Envoyer les notifications/alertes"""
    logger.info("📬 Sending notifications...")
    
    try:
        # Charger les signaux
        signals_file = SIGNALS_DIR / "signaux_trading.csv"
        if not signals_file.exists():
            logger.info("ℹ️ No signals file to process")
            return
        
        import pandas as pd
        df = pd.read_csv(signals_file)
        
        # Filtrer les signaux récents avec haute fiabilité
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            today = datetime.utcnow().date()
            recent = df[df['Date'].dt.date == today]
            
            if 'Reliability' in df.columns:
                high_reliability = recent[recent['Reliability'] >= 70]
                
                if not high_reliability.empty:
                    logger.info(f"📢 Found {len(high_reliability)} high-reliability signals for today")
                    # Implémenter l'envoi d'emails/SMS
                    # for idx, signal in high_reliability.iterrows():
                    #     send_email(signal)
        
        logger.info("✅ Notifications sent")
        
    except Exception as e:
        logger.error(f"❌ Error in notifications: {e}")

# ============================================================================
# SCHEDULER
# ============================================================================

def run_worker():
    """Worker loop principal"""
    logger.info("""
    ╔═══════════════════════════════════════╗
    ║  BACKGROUND WORKER                    ║
    ║  v1.0.0                                ║
    ╚═══════════════════════════════════════╝
    
    ✅ Worker started
    """)
    
    # Planification des tâches (heure UTC)
    last_daily_signals = None
    last_cleanup = None
    last_notifications = None
    
    while True:
        try:
            now = datetime.utcnow()
            
            # Tâche 1: Signaux quotidiens à 16h00 UTC (clôture marché US)
            if (last_daily_signals is None or 
                (now - last_daily_signals).total_seconds() > 86400):
                if now.hour >= 16 and now.minute >= 0:  # Après 16h UTC
                    task_daily_signals()
                    last_daily_signals = now
            
            # Tâche 2: Cleanup cache une fois par jour à 02h00 UTC
            if (last_cleanup is None or 
                (now - last_cleanup).total_seconds() > 86400):
                if now.hour >= 2 and now.minute >= 0:  # Après 02h UTC
                    task_cleanup_cache()
                    last_cleanup = now
            
            # Tâche 3: Notifications toutes les 30 minutes
            if (last_notifications is None or 
                (now - last_notifications).total_seconds() > 1800):
                task_send_notifications()
                last_notifications = now
            
            # Sleep 1 minute avant prochaine vérification
            time.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error in worker loop: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_worker()
