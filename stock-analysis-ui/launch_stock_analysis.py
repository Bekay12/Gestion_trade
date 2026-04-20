#!/usr/bin/env python3
"""
Lanceur principal de l'application Stock Analysis Tool.
Ce script doit être exécuté pour ouvrir l'interface graphique.
"""
import sys
import os

# Assurer que le dossier src est dans le path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

UI_DIR = os.path.join(SRC_DIR, 'ui')
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

# Changer le répertoire de travail vers src pour les chemins relatifs
os.chdir(SRC_DIR)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow
from cache_db import ensure_fx_rates_daily_history


def main():
    # Précharge les taux FX journaliers (5 ans) avec TTL de 20h pour le backtest.
    try:
        fx_refresh = ensure_fx_rates_daily_history(min_refresh_hours=20, years=5, force=False)
        print(f"[FX] {fx_refresh.get('status')} | rows={fx_refresh.get('rows_total')} | added={fx_refresh.get('rows_added')}")
    except Exception as exc:
        print(f"[FX] refresh skipped due to error: {exc}")

    app = QApplication(sys.argv)
    
    # Nom de l'application visible dans la barre des tâches
    app.setApplicationName("Stock Analysis Tool")
    app.setApplicationDisplayName("Stock Analysis Tool")
    app.setOrganizationName("Bekay")
    
    # Icône de l'application (si disponible)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Créer et afficher la fenêtre principale
    window = MainWindow()
    window.setWindowTitle("Stock Analysis Tool")
    window.showMaximized()  # Plein écran (maximisé)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
