"""
Traceur de graphiques unifiés.
Migration de votre fonction plot_unified_chart().
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from typing import Optional

from src.indicators.macd import calculate_macd
from src.signals.signal_generator import get_trading_signal
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChartPlotter:
    """
    Traceur de graphiques techniques unifiés.
    Migration complète de votre fonction plot_unified_chart().
    """

    def __init__(self):
        self.figure_size = (14, 5)
        self.colors = {
            'price': 'tab:blue',
            'ema20': 'orange',
            'ema50': 'purple',
            'sma50': 'green',
            'macd': 'tab:purple',
            'signal': 'tab:orange'
        }

    def plot_unified_chart(self, symbol: str, prices: pd.Series, volumes: pd.Series, 
                          ax: plt.Axes, show_xaxis: bool = False) -> plt.Axes:
        """
        Trace un graphique unifié avec prix, MACD et RSI intégré.
        Migration exacte de votre fonction plot_unified_chart().

        Args:
            symbol: Symbole boursier.
            prices: Série des prix.
            volumes: Série des volumes.
            ax: Axes matplotlib.
            show_xaxis: Afficher l'axe des x.

        Returns:
            Axe secondaire créé pour MACD.
        """
        # Vérification du format des prix (votre logique exacte)
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices, name=symbol)

        # Calcul des indicateurs (votre logique exacte)
        ema20 = prices.ewm(span=20, adjust=False).mean()
        ema50 = prices.ewm(span=50, adjust=False).mean()
        sma50 = prices.rolling(window=50).mean() if len(prices) >= 50 else pd.Series()
        macd, signal_line = calculate_macd(prices)

        # Calcul du RSI avec vérification des données (votre logique exacte)
        try:
            rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
        except Exception as e:
            logger.warning(f"⚠️ Erreur RSI pour {symbol}: {str(e)}")
            rsi = pd.Series(np.zeros(len(prices)), index=prices.index)

        # Tracé des prix sur l'axe principal (votre logique exacte)
        color = self.colors['price']
        ax.plot(prices.index, prices, label='Prix', color=color, linewidth=1.8)
        ax.plot(ema20.index, ema20, label='EMA20', linestyle='--', 
                color=self.colors['ema20'], linewidth=1.4)
        ax.plot(ema50.index, ema50, label='EMA50', linestyle='-.', 
                color=self.colors['ema50'], linewidth=1.4)

        if not sma50.empty:
            ax.plot(sma50.index, sma50, label='SMA50', linestyle=':', 
                   color=self.colors['sma50'], linewidth=1.4)

        ax.set_ylabel('Prix', color=color, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Création d'un axe secondaire pour MACD (votre logique exacte)
        ax2 = ax.twinx()

        # Tracé MACD (votre logique exacte)
        macd_color = self.colors['macd']
        ax2.plot(macd.index, macd, label='MACD', color=macd_color, linewidth=1.2)
        ax2.plot(signal_line.index, signal_line, label='Signal', 
                color=self.colors['signal'], linewidth=1.2)

        ax2.fill_between(
            macd.index, 0, macd - signal_line,
            where=(macd - signal_line) >= 0,
            facecolor='green', alpha=0.3, interpolate=True
        )

        ax2.fill_between(
            macd.index, 0, macd - signal_line,
            where=(macd - signal_line) < 0,
            facecolor='red', alpha=0.3, interpolate=True
        )

        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('MACD', color=macd_color, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=macd_color)

        # Tracé RSI en arrière-plan avec axvspan (votre logique exacte)
        for i in range(1, len(prices)):
            start = prices.index[i-1]
            end = prices.index[i]
            rsi_val = rsi.iloc[i-1]

            if rsi_val > 70:
                color = 'lightcoral'
            elif rsi_val < 30:
                color = 'lightgreen'
            else:
                color = 'lightgray'

            ax.axvspan(start, end, facecolor=color, alpha=0.1, zorder=-1)

        # Ajout des légendes (votre logique exacte)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

        # Récupérer les informations et signaux (votre logique exacte)
        info = yf.Ticker(symbol).info
        domaine = info.get("sector", "Inconnu")

        # Ajout des signaux trading (votre logique exacte)
        signal, last_price, trend, last_rsi, volume_moyen, score = get_trading_signal(
            prices, volumes, domaine=domaine
        )

        # Calcul de la progression en pourcentage (votre logique exacte)
        if len(prices) > 1:
            progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
        else:
            progression = 0.0

        if last_price is not None:
            trend_symbol = "Haussière" if trend else "Baissière"
            rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
            signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'

            title = (
                f"{symbol} | Prix: {last_price:.2f} | Signal: {signal} ({score}) | "
                f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
                f"Progression: {progression:+.2f}% | Vol. moyen: {volume_moyen:,.0f} units "
            )

            ax.set_title(title, fontsize=12, fontweight='bold', color=signal_color)

        # Affichage de l'axe du temps (votre logique exacte)
        if not show_xaxis:
            ax.set_xticklabels([])
            ax2.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', which='both', labelbottom=True)
            ax2.tick_params(axis='x', which='both', labelbottom=True)

        return ax2

    def create_multi_chart_figure(self, num_charts: int) -> tuple:
        """
        Crée une figure avec plusieurs sous-graphiques.

        Args:
            num_charts: Nombre de graphiques.

        Returns:
            Tuple (figure, axes).
        """
        figsize = (self.figure_size[0], self.figure_size[1] * num_charts)
        fig, axes = plt.subplots(num_charts, 1, figsize=figsize, sharex=False)

        if num_charts == 1:
            axes = [axes]

        return fig, axes

    def finalize_figure(self, fig: plt.Figure):
        """
        Finalise une figure avec les paramètres de votre code original.

        Args:
            fig: Figure matplotlib.
        """
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, hspace=0.152, bottom=0.032)

    def show_figure(self):
        """Affiche la figure."""
        plt.show()

    def save_figure(self, filename: str, dpi: int = 300):
        """
        Sauvegarde la figure.

        Args:
            filename: Nom du fichier.
            dpi: Résolution.
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Graphique sauvegardé: {filename}")

# Instance globale
chart_plotter = ChartPlotter()

# Fonction de compatibilité
def plot_unified_chart(symbol: str, prices: pd.Series, volumes: pd.Series, 
                      ax: plt.Axes, show_xaxis: bool = False) -> plt.Axes:
    """Fonction de compatibilité avec votre code existant."""
    return chart_plotter.plot_unified_chart(symbol, prices, volumes, ax, show_xaxis)
