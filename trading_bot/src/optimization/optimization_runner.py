"""
Lanceur d'optimisation complète pour tous les secteurs.
Migration de la logique principale de votre optimisateur_boucle.
"""
import pandas as pd
from typing import List, Dict
from pathlib import Path

from config.settings import config
from src.utils.file_manager import SymbolFileManager
from src.optimization.optimizer import SectorOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OptimizationRunner:
    """
    Lanceur d'optimisation pour tous les secteurs.
    Migration de la logique principale de votre script optimisateur_boucle.
    """

    def __init__(self):
        self.symbol_manager = SymbolFileManager()
        self.sector_optimizer = SectorOptimizer()
        self.opt_config = config.optimization

        # Secteurs disponibles (votre liste exacte)
        self.available_sectors = {
            "Technology": [],
            "Healthcare": [],
            "Financial Services": [],
            "Consumer Cyclical": [],
            "Industrials": [],
            "Energy": [],
            "Basic Materials": [],
            "Communication Services": [],
            "Consumer Defensive": [],
            "Utilities": [],
            "Real Estate": [],
            "ℹ️Inconnu!!": []
        }

    def run_full_optimization(self, symbols_file: str = "optimisation_symbols.txt") -> Dict[str, tuple]:
        """
        Lance l'optimisation complète pour tous les secteurs.
        Migration de votre logique __main__ dans optimisateur_boucle.

        Args:
            symbols_file: Fichier contenant les symboles à optimiser.

        Returns:
            Dictionnaire des coefficients optimisés par secteur.
        """
        logger.info("🚀 Démarrage de l'optimisation complète des secteurs")

        # Charger les symboles (votre logique)
        symbols = self.symbol_manager.load_symbols_from_txt(symbols_file)

        if not symbols:
            logger.error(f"❌ Aucun symbole trouvé dans {symbols_file}")
            return {}

        # Supprimer les doublons tout en conservant l'ordre
        symbols = list(dict.fromkeys(symbols))
        logger.info(f"📋 {len(symbols)} symboles uniques chargés pour l'optimisation")

        # Assigner les symboles aux secteurs (votre logique exacte)
        sectors = self._assign_symbols_to_sectors(symbols)

        # Lancer l'optimisation pour chaque secteur
        optimized_coeffs = {}

        for sector, sector_symbols in sectors.items():
            if not sector_symbols:
                logger.info(f"🚫 Secteur {sector} vide, ignoré")
                continue

            logger.info(f"\n🔧 Optimisation du secteur: {sector} ({len(sector_symbols)} symboles)")

            try:
                coeffs, gain_total, success_rate, thresholds = self.sector_optimizer.optimize_sector_coefficients(
                    sector_symbols, 
                    sector, 
                    period=self.opt_config.period,
                    n_iterations=self.opt_config.n_iterations,
                    montant=self.opt_config.position_size,
                    transaction_cost=self.opt_config.transaction_cost,
                    max_cycles=self.opt_config.max_cycles,
                    convergence_threshold=self.opt_config.convergence_threshold
                )

                if coeffs:
                    optimized_coeffs[sector] = coeffs

                    logger.info(f"\n📊 Résultats pour {sector}:")
                    logger.info(f"  Meilleurs coefficients: {coeffs}")
                    logger.info(f"  Meilleurs seuils (achat, vente): {thresholds}")
                    logger.info(f"  Gain total moyen: {gain_total:.2f}")
                    logger.info(f"  Taux de réussite: {success_rate:.2f}%")

            except Exception as e:
                logger.error(f"❌ Erreur optimisation secteur {sector}: {e}")
                continue

        # Afficher le résumé final (votre format)
        self._display_final_summary(optimized_coeffs)

        logger.info("✅ Optimisation complète terminée")
        return optimized_coeffs

    def _assign_symbols_to_sectors(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Assigne les symboles aux secteurs dynamiquement.
        Migration de votre logique d'assignation.
        """
        sectors = self.available_sectors.copy()

        logger.info("\n📋 Assignation des secteurs:")

        for symbol in symbols:
            sector = self.sector_optimizer.get_sector(symbol)

            if sector in sectors:
                sectors[sector].append(symbol)
            else:
                sectors["ℹ️Inconnu!!"].append(symbol)

        # Afficher l'assignation (votre format)
        for sector, syms in sectors.items():
            if syms:  # N'afficher que les secteurs non vides
                logger.info(f"  {sector}: {syms}")

        return sectors

    def _display_final_summary(self, optimized_coeffs: Dict[str, tuple]):
        """
        Affiche le résumé final au format de votre code original.
        """
        if not optimized_coeffs:
            logger.warning("⚠️ Aucun coefficient optimisé")
            return

        logger.info("\n" + "="*80)
        logger.info("🎯 RÉSUMÉ FINAL DE L'OPTIMISATION")
        logger.info("="*80)

        logger.info("\nDictionnaire optimisé pour domain_coeffs:")
        logger.info("{")
        for sector, coeffs in optimized_coeffs.items():
            logger.info(f"    '{sector}': {coeffs},")
        logger.info("}")

        logger.info(f"\n📈 {len(optimized_coeffs)} secteurs optimisés avec succès")
        logger.info("="*80)

    def optimize_single_sector(self, sector: str, symbols: List[str]) -> tuple:
        """
        Optimise un secteur spécifique.

        Args:
            sector: Nom du secteur.
            symbols: Liste des symboles du secteur.

        Returns:
            Tuple (coeffs, gain_total, success_rate, thresholds).
        """
        logger.info(f"🔧 Optimisation du secteur spécifique: {sector}")

        return self.sector_optimizer.optimize_sector_coefficients(
            symbols, 
            sector,
            period=self.opt_config.period,
            n_iterations=self.opt_config.n_iterations,
            montant=self.opt_config.position_size,
            transaction_cost=self.opt_config.transaction_cost,
            max_cycles=self.opt_config.max_cycles,
            convergence_threshold=self.opt_config.convergence_threshold
        )

# Instance globale
optimization_runner = OptimizationRunner()
