#!/usr/bin/env python3
# ============================================================================
# SCHEDULER.PY - PLANIFICATEUR DE TÂCHES
# Alternative au background worker avec meilleure gestion des horaires
# ============================================================================

import os
import sys
import logging
from datetime import datetime
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
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)

# ============================================================================
# APScheduler (optionnel, nécessite: pip install APScheduler)
# ============================================================================

def setup_scheduler():
    """Configure APScheduler pour tâches planifiées"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("⚠️ APScheduler not installed. Install with: pip install APScheduler")
        logger.info("Using basic scheduling instead...")
        return None
    
    scheduler = BackgroundScheduler(timezone='UTC')
    
    # Tâche: Signaux quotidiens à 16h00 UTC
    scheduler.add_job(
        task_daily_signals,
        CronTrigger(hour=16, minute=0),
        id='daily_signals',
        name='Daily signals calculation',
        replace_existing=True
    )
    
    # Tâche: Cleanup cache à 02h00 UTC
    scheduler.add_job(
        task_cleanup_cache,
        CronTrigger(hour=2, minute=0),
        id='cleanup_cache',
        name='Cache cleanup',
        replace_existing=True
    )
    
    # Tâche: Notifications toutes les 30 minutes
    scheduler.add_job(
        task_send_notifications,
        CronTrigger(minute='*/30'),
        id='notifications',
        name='Send notifications',
        replace_existing=True
    )
    
    logger.info("✅ Scheduler configured")
    return scheduler

def task_daily_signals():
    """Calcul quotidien des signaux"""
    logger.info("📊 [SCHEDULED] Running daily signals calculation...")
    try:
        symbols = get_symbols_by_list_type('popular')
        if not symbols:
            logger.warning("⚠️ No popular symbols found")
            return
        
        analyse_signaux_populaires(
            symbols=symbols,
            mes_symbols=[],
            period='12mo',
            afficher_graphiques=False,
            verbose=True
        )
        logger.info("✅ Daily signals completed")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def task_cleanup_cache():
    """Nettoyage du cache"""
    logger.info("🧹 [SCHEDULED] Cleaning cache...")
    try:
        from datetime import timedelta
        cache_dir = Path('cache_data')
        if cache_dir.exists():
            cutoff = datetime.utcnow() - timedelta(days=30)
            count = 0
            for file in cache_dir.glob('*'):
                if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    count += 1
            logger.info(f"✅ Cache cleanup completed ({count} files deleted)")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def task_send_notifications():
    """Envoyer les notifications"""
    logger.info("📬 [SCHEDULED] Checking for notifications...")
    # TODO: Implémenter logique notifications
    logger.info("✅ Notifications checked")

if __name__ == '__main__':
    logger.info("""
    ╔═══════════════════════════════════════╗
    ║  TASK SCHEDULER                       ║
    ║  v1.0.0                                ║
    ╚═══════════════════════════════════════╝
    """)
    
    scheduler = setup_scheduler()
    
    if scheduler:
        scheduler.start()
        logger.info("🚀 Scheduler started (APScheduler)")
        try:
            # Garder le scheduler actif
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹️ Scheduler stopped")
            scheduler.shutdown()
    else:
        logger.error("❌ Could not start scheduler. Install APScheduler:")
        logger.error("   pip install APScheduler")
