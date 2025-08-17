#!/usr/bin/env python3
"""
Point d'entrée principal du trading bot.
Lance l'analyse ou l'optimisation selon les arguments.
"""
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire du projet au path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from src.utils.logger import setup_logging, get_logger
from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.optimization.optimization_runner import optimization_runner
from src.visualization.analysis_charts import analysis_charts

def setup_environment():
    """Configure l'environnement."""
    # Configurer les logs
    setup_logging(config.logging)

    # S'assurer que les dossiers existent
    config._create_directories()

    return get_logger(__name__)

def run_analysis(symbols_file: str = "test_symbols.txt", personal_symbols_file: str = "mes_symbols.txt",
                period: str = None, show_charts: bool = True):
    """Lance l'analyse des signaux."""
    logger = get_logger(__name__)
    logger.info("🔍 Lancement de l'analyse des signaux")

    try:
        symbol_manager = SymbolFileManager()

        # Charger les symboles
        test_symbols = symbol_manager.load_symbols_from_txt(symbols_file)
        personal_symbols = symbol_manager.load_symbols_from_txt(personal_symbols_file)

        if not test_symbols:
            logger.error(f"❌ Aucun symbole trouvé dans {symbols_file}")
            return False

        period = period or config.trading.default_period
        logger.info(f"📊 Analyse de {len(test_symbols)} symboles (période: {period})")

        # Lancer l'analyse
        results = signal_analyzer.analyze_popular_signals(
            test_symbols, 
            personal_symbols, 
            period=period,
            display_charts=show_charts,
            verbose=True,
            save_csv=True
        )

        if results.get('signals'):
            logger.info(f"✅ Analyse terminée: {len(results['signals'])} signaux détectés")
            return True
        else:
            logger.warning("⚠️ Aucun signal détecté")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur durant l'analyse: {e}")
        return False

def run_optimization(symbols_file: str = "optimisation_symbols.txt"):
    """Lance l'optimisation des paramètres."""
    logger = get_logger(__name__)
    logger.info("🚀 Lancement de l'optimisation")

    try:
        results = optimization_runner.run_full_optimization(symbols_file)

        if results:
            logger.info(f"✅ Optimisation terminée: {len(results)} secteurs optimisés")
            return True
        else:
            logger.warning("⚠️ Aucun résultat d'optimisation")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur durant l'optimisation: {e}")
        return False

def run_charts(symbols_file: str = "test_symbols.txt", period: str = None):
    """Affiche uniquement les graphiques."""
    logger = get_logger(__name__)
    logger.info("📊 Affichage des graphiques")

    try:
        symbol_manager = SymbolFileManager()
        symbols = symbol_manager.load_symbols_from_txt(symbols_file)

        if not symbols:
            logger.error(f"❌ Aucun symbole trouvé dans {symbols_file}")
            return False

        period = period or config.trading.default_period
        analysis_charts.analyse_et_affiche(symbols, period)

        logger.info("✅ Graphiques affichés")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur affichage graphiques: {e}")
        return False

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Trading Bot - Analyse technique et optimisation")
    parser.add_argument('command', choices=['analysis', 'optimization', 'charts'], 
                       help='Commande à exécuter')
    parser.add_argument('--symbols', '-s', default=None,
                       help='Fichier des symboles à analyser')
    parser.add_argument('--personal-symbols', '-p', default="mes_symbols.txt",
                       help='Fichier des symboles personnels')
    parser.add_argument('--period', default=None,
                       help='Période d\'analyse (ex: 12mo, 1y, 2y)')
    parser.add_argument('--no-charts', action='store_true',
                       help='Ne pas afficher les graphiques')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbeux')

    args = parser.parse_args()

    # Configuration de base
    logger = setup_environment()
    logger.info(f"🤖 Trading Bot - Commande: {args.command}")

    success = False

    if args.command == 'analysis':
        symbols_file = args.symbols or "test_symbols.txt"
        success = run_analysis(
            symbols_file, 
            args.personal_symbols,
            args.period,
            not args.no_charts
        )

    elif args.command == 'optimization':
        symbols_file = args.symbols or "optimisation_symbols.txt"
        success = run_optimization(symbols_file)

    elif args.command == 'charts':
        symbols_file = args.symbols or "test_symbols.txt"
        success = run_charts(symbols_file, args.period)

    if success:
        logger.info("🎉 Exécution terminée avec succès")
        sys.exit(0)
    else:
        logger.error("💥 Exécution échouée")
        sys.exit(1)

if __name__ == "__main__":
    main()