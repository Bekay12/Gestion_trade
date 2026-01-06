#!/usr/bin/env python3
"""
Script de lancement de l'analyse de signaux.
Remplace votre test.py en tant que script standalone.
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.utils.cache import DataCacheManager
from src.utils.logger import get_logger
from config.settings import config

logger = get_logger(__name__)

def main():
    """Lance l'analyse des signaux."""
    logger.info("🔍 Démarrage du script d'analyse")

    try:
        # Charger les symboles
        symbol_manager = SymbolFileManager()
        test_symbols = symbol_manager.load_symbols_from_txt("test_symbols.txt")
        mes_symbols = symbol_manager.load_symbols_from_txt("mes_symbols.txt")

        if not test_symbols:
            logger.error("❌ Aucun symbole de test trouvé")
            return

        logger.info(f"📋 Analyse de {len(test_symbols)} symboles de test et {len(mes_symbols)} symboles personnels")

        # Précharger le cache
        cache_manager = DataCacheManager()
        period = config.trading.default_period

        logger.info("⏳ Préchargement du cache...")
        cache_manager.preload_cache(test_symbols + mes_symbols, period)

        # Lancer l'analyse
        results = signal_analyzer.analyze_popular_signals(
            test_symbols, 
            mes_symbols, 
            period, 
            plot_all=True
        )

        if results.get('signals'):
            logger.info(f"✅ Analyse terminée: {len(results['signals'])} signaux détectés")
        else:
            logger.warning("⚠️ Aucun signal détecté")

    except Exception as e:
        logger.error(f"❌ Erreur durant l'analyse: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
