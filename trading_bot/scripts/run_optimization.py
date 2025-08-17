#!/usr/bin/env python3
"""
Script de lancement de l'optimisation.
Remplace votre optimisateur_boucle en tant que script standalone.
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.optimization_runner import optimization_runner
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Lance l'optimisation complète."""
    logger.info("🚀 Démarrage du script d'optimisation")

    try:
        results = optimization_runner.run_full_optimization()

        if results:
            logger.info(f"✅ Optimisation terminée avec succès: {len(results)} secteurs optimisés")
        else:
            logger.warning("⚠️ Aucun résultat d'optimisation obtenu")

    except Exception as e:
        logger.error(f"❌ Erreur durant l'optimisation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
