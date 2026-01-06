#!/usr/bin/env python3
"""
Interface graphique PyQt5 pour le trading bot.
Adapte l'ancienne UI (main_window.py) à la nouvelle architecture.
"""
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt

# Ajouter la racine du projet au PYTHONPATH pour les imports relatifs
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from src.utils.logger import setup_logging, get_logger
from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.visualization.analysis_charts import analysis_charts

# Configure logging
setup_logging(config.logging)
logger = get_logger(__name__)

class MainWindow(QMainWindow):
    """Fenêtre principale de l'interface graphique."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot - Analyse Technique")
        self.setGeometry(100, 100, 900, 600)

        # Gestionnaire de symboles
        self.symbol_manager = SymbolFileManager()

        # Données initiales
        self.default_period = config.trading.default_period
        self.popular_symbols = self.symbol_manager.load_symbols_from_txt("popular_symbols.txt")
        self.personal_symbols = self.symbol_manager.load_symbols_from_txt("mes_symbols.txt")

        # Construction de l'UI
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Titre
        title = QLabel("🤖 Trading Bot – Interface Graphique")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title)

        # Ligne de symboles individuels
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symboles (ex: AAPL, MSFT, GOOGL)")
        analyze_btn = QPushButton("Analyser")
        analyze_btn.clicked.connect(self._on_analyze)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Symboles:"))
        row1.addWidget(self.symbol_input)
        row1.addWidget(analyze_btn)
        main_layout.addLayout(row1)

        # Période
        self.period_input = QLineEdit(self.default_period)
        self.period_input.setPlaceholderText("Période (ex: 12mo)")
        row_period = QHBoxLayout()
        row_period.addWidget(QLabel("Période:"))
        row_period.addWidget(self.period_input)
        main_layout.addLayout(row_period)

        # Symboles populaires / personnels
        self.popular_symbols_input = QLineEdit(", ".join(self.popular_symbols))
        self.personal_symbols_input = QLineEdit(", ".join(self.personal_symbols))

        row_pop = QHBoxLayout()
        row_pop.addWidget(QLabel("Symboles populaires:"))
        row_pop.addWidget(self.popular_symbols_input)
        main_layout.addLayout(row_pop)

        row_pers = QHBoxLayout()
        row_pers.addWidget(QLabel("Mes symboles:"))
        row_pers.addWidget(self.personal_symbols_input)
        main_layout.addLayout(row_pers)

        # Bouton analyse signaux populaires
        signals_btn = QPushButton("Analyser Signaux Populaires ✅")
        signals_btn.clicked.connect(self._on_popular_signals)
        main_layout.addWidget(signals_btn)

        # Espace résultat
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Actions UI
    # ------------------------------------------------------------------
    def _on_analyze(self):
        symbols = self._parse_symbols(self.symbol_input.text())
        if not symbols:
            self._warn("Entrez au moins un symbole à analyser.")
            return

        period = self._validate_period(self.period_input.text())
        if not period:
            return

        logger.info(f"Analyse graphique pour {len(symbols)} symboles (période {period})")
        analysis_charts.analyse_et_affiche(symbols, period)
        self.status_label.setText(f"✅ Analyse graphique terminée pour {len(symbols)} symbole(s).")

    def _on_popular_signals(self):
        pop_symbols = self._parse_symbols(self.popular_symbols_input.text())
        per_symbols = self._parse_symbols(self.personal_symbols_input.text())
        if not pop_symbols:
            self._warn("Entrez des symboles populaires.")
            return

        period = self._validate_period(self.period_input.text())
        if not period:
            return

        logger.info("Analyse des signaux populaires…")
        result = signal_analyzer.analyze_popular_signals(
            pop_symbols, per_symbols, period=period, display_charts=True, verbose=True
        )
        n_signals = len(result.get("signals", []))
        self.status_label.setText(f"✅ Analyse signaux populaires terminée – {n_signals} signal(s) détecté(s).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_symbols(self, text: str):
        return [s.strip().upper() for s in text.split(',') if s.strip()]

    def _validate_period(self, period: str):
        valid_periods = config.data.valid_periods
        if period not in valid_periods:
            self._warn(f"Période invalide. Choisissez parmi: {', '.join(valid_periods)}")
            return None
        return period

    def _warn(self, message: str):
        QMessageBox.warning(self, "Attention", message)
        self.status_label.setText(f"⚠️ {message}")

# ----------------------------------------------------------------------
# Lancement de l'application
# ----------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
