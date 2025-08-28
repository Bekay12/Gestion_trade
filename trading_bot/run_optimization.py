#!/usr/bin/env python3
"""
Script d'entrée pour lancer l'optimisation complète.
Usage: python run_optimization.py
"""
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.optimization_runner import optimization_runner
from src.utils.logger import setup_logging, get_logger
from config.settings import config

def main():
    """Lance l'optimisation complète."""
    # Configuration des logs
    setup_logging(config.logging)
    logger = get_logger(__name__)

    logger.info("🚀 Lancement de l'optimisation complète")

    try:
        # Lancer l'optimisation
        results = optimization_runner.run_full_optimization("optimisation_symbols.txt")

        if results:
            logger.info("✅ Optimisation terminée avec succès")
            print(f"📈 {len(results)} secteurs optimisés")
        else:
            logger.warning("⚠️ Aucun résultat d'optimisation")

    except Exception as e:
        logger.error(f"❌ Erreur durant l'optimisation: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
