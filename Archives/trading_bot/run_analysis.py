#!/usr/bin/env python3
"""
Script d'entrée pour lancer l'analyse des signaux.
Usage: python run_analysis.py
"""
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.signals.signal_analyzer import signal_analyzer
from src.utils.logger import setup_logging, get_logger
from src.utils.file_manager import SymbolFileManager
from config.settings import config

def main():
    """Lance l'analyse des signaux."""
    # Configuration des logs
    setup_logging(config.logging)
    logger = get_logger(__name__)

    logger.info("🔍 Lancement de l'analyse des signaux")

    try:
        # Charger les symboles
        symbol_manager = SymbolFileManager()
        popular_symbols = symbol_manager.load_symbols_from_txt("popular_symbols.txt")
        mes_symbols = symbol_manager.load_symbols_from_txt("mes_symbols.txt")

        if not popular_symbols:
            logger.error("❌ Aucun symbole populaire trouvé")
            return 1

        # Lancer l'analyse
        results = signal_analyzer.analyze_popular_signals(
            popular_symbols=popular_symbols,
            mes_symbols=mes_symbols,
            period="12mo",
            display_charts=True,
            verbose=True,
            save_csv=True
        )

        if results.get("signals"):
            n_signals = len(results["signals"])
            logger.info(f"✅ Analyse terminée - {n_signals} signal(s) détecté(s)")
        else:
            logger.info("ℹ️ Aucun signal détecté")

    except Exception as e:
        logger.error(f"❌ Erreur durant l'analyse: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
