"""
Nouvelle version de main_window.py avec les fonctionnalités yfinance améliorées
"""

import sys
import os
import math
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSpacerItem, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QInputDialog, QListWidget, QListWidgetItem,
    QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QComboBox, QHeaderView, QSpinBox, QCheckBox
)
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import yfinance as yf
import pandas as pd

# Ensure project root is on sys.path
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from qsi import (
    analyse_signaux_populaires, analyse_et_affiche, popular_symbols,
    mes_symbols, period, download_stock_data, backtest_signals,
    plot_unified_chart, get_trading_signal, generate_trade_events
)

class AnalysisThread(QThread):
    # [Code existant de AnalysisThread inchangé]
    pass

class DownloadThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, symbols, period="12mo", do_backtest=False):
        super().__init__()
        self.symbols = symbols
        self.period = period
        self.do_backtest = do_backtest

    def run(self):
        try:
            # Redirection temporaire de print pour les messages de progression
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.progress.emit(message)
                original_print(*args, **kwargs)

            import builtins
            builtins.print = custom_print

            try:
                # Télécharger les données de prix
                data = download_stock_data(self.symbols, self.period)
                result = {'data': data}

                # Enrichir avec les données yfinance
                tickers_info = {}
                for symbol in self.symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        # Récupérer les dates importantes
                        calendar = ticker.calendar
                        earnings_date = None
                        if calendar is not None and not calendar.empty:
                            earnings_date = calendar.iloc[0]['Earnings Date'] if 'Earnings Date' in calendar.columns else None
                        
                        # Récupérer l'historique des dividendes
                        dividends = ticker.dividends
                        next_div_date = None
                        if not dividends.empty:
                            last_div_date = dividends.index[-1]
                            # Calculer la prochaine date estimée (simpliste)
                            if len(dividends) >= 2:
                                avg_interval = (dividends.index[-1] - dividends.index[-2]).days
                                next_div_date = last_div_date + pd.Timedelta(days=avg_interval)
                        
                        # Analyse des recommandations
                        analysis = ticker.analysis if hasattr(ticker, 'analysis') else None
                        target_price = None
                        if analysis is not None and hasattr(analysis, 'info'):
                            target_price = analysis.info.get('targetMeanPrice')
                        elif 'targetMeanPrice' in info:
                            target_price = info['targetMeanPrice']
                        
                        # Stocker toutes les infos
                        tickers_info[symbol] = {
                            'info': info,
                            'earnings_date': earnings_date,
                            'next_dividend_date': next_div_date,
                            'target_price': target_price,
                            'last_dividend_date': last_div_date if not dividends.empty else None,
                            'last_dividend_amount': dividends.iloc[-1] if not dividends.empty else None
                        }

                    except Exception as e:
                        print(f"Erreur lors de la récupération des infos pour {symbol}: {str(e)}")
                        continue

                result['tickers_info'] = tickers_info

                if self.do_backtest and data:
                    backtests = []
                    for symbol, stock_data in data.items():
                        try:
                            prices = stock_data['Close']
                            volumes = stock_data['Volume']
                            
                            # Utiliser le secteur depuis les infos yfinance
                            domaine = 'Inconnu'
                            if symbol in tickers_info:
                                info = tickers_info[symbol]['info']
                                domaine = info.get('sector', 'Inconnu')

                            bt = backtest_signals(prices, volumes, domaine, montant=50)
                            backtests.append({'Symbole': symbol, **bt})
                            
                        except Exception:
                            continue

                    result['backtest_results'] = backtests

                self.finished.emit(result)

            finally:
                builtins.print = original_print

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.setup_ui()
        self.current_results = []
        self.tickers_info = {}  # Pour stocker les infos yfinance

    def setup_ui(self):
        # [Code existant de setup_ui inchangé jusqu'à la définition des colonnes du tableau]
        
        # Mettre à jour les colonnes du tableau avec les nouvelles informations
        merged_columns = [
            'Date résultats',  # Nouvelle colonne
            'Symbole', 'Signal', 'Score', 'Prix', 'Tendance', 'RSI', 
            'Volume moyen', 'Domaine', 'Fiabilite', 'Nb Trades', 'Gagnants',
            'Rev. Growth (%)', 'Gross Margin (%)', 'FCF (B$)', 'D/E Ratio', 
            'Market Cap (B$)', 'Prix Objectif',  # Nouvelle colonne
            'dPrice', 'dMACD', 'dRSI', 'dVolRel',
            'Gain total ($)', 'Gain moyen ($)'
        ]
        
        self.merged_table.setColumnCount(len(merged_columns))
        self.merged_table.setHorizontalHeaderLabels(merged_columns)
        self.merged_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.merged_table.setSortingEnabled(True)
        
        # [Reste du code setup_ui inchangé]

    def update_results_table(self):
        """Mise à jour du tableau avec les nouvelles colonnes et données yfinance"""
        if not hasattr(self, 'current_results'):
            return

        self.merged_table.setRowCount(0)
        results_to_display = getattr(self, 'filtered_results', self.current_results)
        bt_map = getattr(self, 'backtest_map', {})
        tickers_info = getattr(self, 'tickers_info', {})

        for signal in results_to_display:
            try:
                row = self.merged_table.rowCount()
                self.merged_table.insertRow(row)
                sym = signal.get('Symbole', '')

                # Colonne Date résultats (nouvelle)
                earnings_date = ''
                if sym in tickers_info:
                    ed = tickers_info[sym].get('earnings_date')
                    if ed:
                        if isinstance(ed, (pd.Timestamp, datetime)):
                            earnings_date = ed.strftime('%Y-%m-%d')
                        else:
                            earnings_date = str(ed)
                self.merged_table.setItem(row, 0, QTableWidgetItem(earnings_date))

                # [Colonnes existantes inchangées]
                # ... [Code existant pour les autres colonnes]

                # Prix Objectif (nouvelle colonne après Market Cap)
                target_price = 'N/A'
                if sym in tickers_info:
                    tp = tickers_info[sym].get('target_price')
                    if tp is not None:
                        target_price = f"{float(tp):.2f}"
                item = QTableWidgetItem(target_price)
                if target_price != 'N/A':
                    item.setData(Qt.EditRole, float(target_price))
                self.merged_table.setItem(row, 16, item)  # Ajuster l'index selon vos colonnes

                # [Reste des colonnes existantes]
                # ... [Code existant pour les dernières colonnes]

            except Exception:
                continue

    def plot_unified_chart_extended(self, symbol, prices, volumes, ax, show_xaxis=True):
        """Version étendue de plot_unified_chart avec dates importantes"""
        # D'abord appeler la fonction de base
        plot_unified_chart(symbol, prices, volumes, ax, show_xaxis=show_xaxis)
        
        # Ajouter les dates importantes si disponibles
        if symbol in self.tickers_info:
            info = self.tickers_info[symbol]
            
            # Dates de résultats
            if info.get('earnings_date'):
                ed = info['earnings_date']
                if isinstance(ed, (pd.Timestamp, datetime)):
                    # Trouver le prix le plus proche pour la position verticale
                    try:
                        closest_price = prices.asof(ed)
                        ax.axvline(x=ed, color='purple', linestyle='--', alpha=0.5)
                        ax.text(ed, closest_price, 'Résultats', rotation=90, 
                               verticalalignment='bottom', color='purple')
                    except Exception:
                        pass
            
            # Dates de dividendes
            if info.get('next_dividend_date'):
                nd = info['next_dividend_date']
                if isinstance(nd, (pd.Timestamp, datetime)):
                    try:
                        closest_price = prices.asof(nd)
                        ax.axvline(x=nd, color='green', linestyle='--', alpha=0.5)
                        ax.text(nd, closest_price, 'Div. estimé', rotation=90, 
                               verticalalignment='bottom', color='green')
                    except Exception:
                        pass
            
            # Dernier dividende
            if info.get('last_dividend_date') and info.get('last_dividend_amount'):
                ld = info['last_dividend_date']
                amount = info['last_dividend_amount']
                if isinstance(ld, (pd.Timestamp, datetime)):
                    try:
                        closest_price = prices.asof(ld)
                        ax.axvline(x=ld, color='green', linestyle='--', alpha=0.5)
                        ax.text(ld, closest_price, f'Div. {amount:.2f}$', rotation=90, 
                               verticalalignment='bottom', color='green')
                    except Exception:
                        pass

    def on_download_complete(self, result):
        # Sauvegarder les informations yfinance
        self.tickers_info = result.get('tickers_info', {})
        
        # [Reste du code on_download_complete avec appel à plot_unified_chart_extended]
        # Remplacer les appels à plot_unified_chart par plot_unified_chart_extended
        # ... [Code existant modifié]