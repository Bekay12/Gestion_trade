"""
Analyseur et afficheur de graphiques techniques.
Migration de votre fonction analyse_et_affiche().
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from src.data.providers.yahoo_provider import YahooProvider
from src.visualization.chart_plotter import ChartPlotter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnalysisCharts:
    """
    Analyseur et afficheur de graphiques techniques.
    Migration complète de votre fonction analyse_et_affiche().
    """

    def __init__(self):
        self.yahoo_provider = YahooProvider()
        self.chart_plotter = ChartPlotter()

    def analyse_et_affiche(self, symbols: List[str], period: str = "12mo"):
        """
        Télécharge les données pour les symboles donnés et affiche les graphiques d'analyse technique.
        Migration exacte de votre fonction analyse_et_affiche().

        Args:
            symbols: Liste des symboles à analyser.
            period: Période des données.
        """
        logger.info("⏳ Téléchargement des données...")

        # Télécharger les données (votre logique)
        data = self.yahoo_provider.download_batch(symbols, period)

        if not data:
            logger.error("❌ Aucune donnée valide disponible. Vérifiez les symboles ou la connexion internet.")
            return

        num_plots = len(data)

        # Créer la figure avec sous-graphiques (votre logique exacte)
        fig, axes = self.chart_plotter.create_multi_chart_figure(num_plots)

        if num_plots == 0:
            logger.error("❌ Aucun symbole valide à afficher")
            return

        # Tracer chaque symbole (votre logique exacte)
        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            logger.info(f"📊 Traitement de {symbol}...")

            show_xaxis = (i == len(data) - 1)  # True seulement pour le dernier subplot
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)

        # Finaliser et afficher (votre logique exacte)
        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.show_figure()

    def display_signal_charts(self, signals: List[Dict], period: str = "12mo", 
                             max_charts: int = 5, signal_type: str = "ACHAT"):
        """
        Affiche les graphiques pour des signaux spécifiques.
        Inspiré de votre logique d'affichage des top signaux.

        Args:
            signals: Liste des signaux.
            period: Période des données.
            max_charts: Nombre maximum de graphiques.
            signal_type: Type de signal ("ACHAT" ou "VENTE").
        """
        # Filtrer les signaux par type
        filtered_signals = [s for s in signals if s.get('Signal') == signal_type][:max_charts]

        if not filtered_signals:
            logger.warning(f"Aucun signal {signal_type} à afficher")
            return

        logger.info(f"\nAffichage des graphiques pour les {len(filtered_signals)} premiers signaux {signal_type} détectés...")

        # Télécharger les données pour ces signaux
        symbols = [s['Symbole'] for s in filtered_signals]
        data = self.yahoo_provider.download_batch(symbols, period)

        if not data:
            logger.error("❌ Aucune donnée disponible pour les signaux")
            return

        # Créer la figure
        fig, axes = self.chart_plotter.create_multi_chart_figure(len(data))

        # Tracer chaque graphique
        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            show_xaxis = (i == len(data) - 1)
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)

        # Finaliser et afficher
        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.show_figure()

    def save_analysis_charts(self, symbols: List[str], period: str = "12mo", 
                           filename: str = "analysis_charts.png"):
        """
        Sauvegarde les graphiques d'analyse au lieu de les afficher.

        Args:
            symbols: Liste des symboles.
            period: Période des données.
            filename: Nom du fichier de sauvegarde.
        """
        logger.info("⏳ Génération des graphiques pour sauvegarde...")

        data = self.yahoo_provider.download_batch(symbols, period)

        if not data:
            logger.error("❌ Aucune donnée valide disponible")
            return

        num_plots = len(data)
        fig, axes = self.chart_plotter.create_multi_chart_figure(num_plots)

        for i, (symbol, stock_data) in enumerate(data.items()):
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            show_xaxis = (i == len(data) - 1)
            self.chart_plotter.plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)

        self.chart_plotter.finalize_figure(fig)
        self.chart_plotter.save_figure(filename)
        plt.close(fig)  # Libérer la mémoire

# Instance globale
analysis_charts = AnalysisCharts()

# Fonction de compatibilité
def analyse_et_affiche(symbols: List[str], period: str = "12mo"):
    """Fonction de compatibilité avec votre code existant."""
    analysis_charts.analyse_et_affiche(symbols, period)
