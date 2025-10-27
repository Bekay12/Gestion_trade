from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QInputDialog, QListWidget, QListWidgetItem,
    QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QComboBox, QHeaderView, QSpinBox, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import os
import math

# Ensure project `src` root is on sys.path
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)
from qsi import analyse_signaux_populaires, analyse_et_affiche, popular_symbols, mes_symbols, period
from qsi import download_stock_data, backtest_signals, plot_unified_chart, get_trading_signal
from qsi import analyse_et_affiche

class AnalysisThread(QThread):
    """Thread for running long analyses without blocking the GUI."""
    finished = pyqtSignal(dict)  # Emitted when analysis completes
    error = pyqtSignal(str)      # Emitted on error
    progress = pyqtSignal(str)   # Emitted to update progress message
    
    def __init__(self, symbols, mes_symbols, period="12mo"):
        super().__init__()
        self.symbols = symbols
        self.mes_symbols = mes_symbols
        self.period = period
    
    def run(self):
        try:
            # Patch the print function to capture progress
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.progress.emit(message)
                original_print(*args, **kwargs)
            
            import builtins
            builtins.print = custom_print
            
            try:
                result = analyse_signaux_populaires(
                    self.symbols,
                    self.mes_symbols,
                    period=self.period,
                    plot_all=False  # Désactive l'affichage des graphiques
                )
                self.finished.emit(result)
            finally:
                builtins.print = original_print
                
        except Exception as e:
            self.error.emit(str(e))

class DownloadThread(QThread):
    """Thread to download stock data (and optionally run backtests) without blocking UI.

    Emits finished with a dict: { 'data': {...}, 'backtest_results': [...] }
    """
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
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.progress.emit(message)
                original_print(*args, **kwargs)

            import builtins
            builtins.print = custom_print

            try:
                data = download_stock_data(self.symbols, self.period)
                result = { 'data': data }

                if self.do_backtest and data:
                    backtests = []
                    for symbol, stock_data in data.items():
                        try:
                            prices = stock_data['Close']
                            volumes = stock_data['Volume']
                            # domain info best-effort
                            try:
                                import yfinance as yf
                                info = yf.Ticker(symbol).info
                                domaine = info.get('sector', 'Inconnu')
                            except Exception:
                                domaine = 'Inconnu'

                            bt = backtest_signals(prices, volumes, domaine, montant=50)
                            backtests.append({ 'Symbole': symbol, **bt })
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
        
    def setup_ui(self):
        # Titre
        title_label = QLabel("Stock Analysis Tool")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(title_label)
        
        # Input de symbole
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        self.layout.addWidget(self.symbol_input)
        
        # Listes de symboles
        lists_container = QHBoxLayout()
        
        # Liste populaire
        popular_layout = QVBoxLayout()
        popular_layout.addWidget(QLabel("Symboles populaires:"))
        self.popular_list = QListWidget()
        # Reduce list height to free more room for plots
        self.popular_list.setMaximumHeight(110)
        for s in popular_symbols:
            if s:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, s)
                self.popular_list.addItem(item)
        popular_layout.addWidget(self.popular_list)
        
        pop_btns = QHBoxLayout()
        self.pop_add_btn = QPushButton("Ajouter")
        self.pop_del_btn = QPushButton("Supprimer")
        self.pop_show_btn = QPushButton("Afficher")
        pop_btns.addWidget(self.pop_add_btn)
        pop_btns.addWidget(self.pop_del_btn)
        pop_btns.addWidget(self.pop_show_btn)
        popular_layout.addLayout(pop_btns)
        
        lists_container.addLayout(popular_layout)
        
        # Liste personnelle
        mes_layout = QVBoxLayout()
        mes_layout.addWidget(QLabel("Mes symboles:"))
        self.mes_list = QListWidget()
        # Reduce list height to free more room for plots
        self.mes_list.setMaximumHeight(110)
        for s in mes_symbols:
            if s:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, s)
                self.mes_list.addItem(item)
        mes_layout.addWidget(self.mes_list)
        
        mes_btns = QHBoxLayout()
        self.mes_add_btn = QPushButton("Ajouter")
        self.mes_del_btn = QPushButton("Supprimer")
        self.mes_show_btn = QPushButton("Afficher")
        mes_btns.addWidget(self.mes_add_btn)
        mes_btns.addWidget(self.mes_del_btn)
        mes_btns.addWidget(self.mes_show_btn)
        mes_layout.addLayout(mes_btns)
        
        lists_container.addLayout(mes_layout)
        self.layout.addLayout(lists_container)
        
        # Période d'analyse
        period_layout = QHBoxLayout()
        period_layout.addWidget(QLabel("Période d'analyse:"))
        self.period_input = QLineEdit(period)
        period_layout.addWidget(self.period_input)
        self.layout.addLayout(period_layout)
        
        # Boutons d'analyse dans un layout horizontal
        buttons_layout = QHBoxLayout()
        
        # Bouton d'analyse simple
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_stock)
        buttons_layout.addWidget(self.analyze_button)
        
        # Bouton de backtest
        self.backtest_button = QPushButton("Analyze and Backtest")
        self.backtest_button.clicked.connect(self.analyse_and_backtest)
        buttons_layout.addWidget(self.backtest_button)
        
        # Bouton analyse populaire
        self.popular_signals_button = QPushButton("Analyser mouvements fiables (populaires)")
        self.popular_signals_button.clicked.connect(self.analyze_popular_signals)
        buttons_layout.addWidget(self.popular_signals_button)
        
        self.layout.addLayout(buttons_layout)
        
        # Options de tri
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Trier par:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Prix (croissant)", 
            "Prix (décroissant)",
            "Score (croissant)",
            "Score (décroissant)",
            "RSI (croissant)",
            "RSI (décroissant)",
            "Volume (croissant)",
            "Volume (décroissant)",
            "Fiabilité (croissant)",
            "Fiabilité (décroissant)"
        ])
        self.sort_combo.currentIndexChanged.connect(self.sort_results)
        sort_layout.addWidget(self.sort_combo)
        self.layout.addLayout(sort_layout)

        # Filtre de fiabilité (pour backtest / signaux populaires)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Fiabilité min (%):"))
        self.min_fiabilite_spin = QSpinBox()
        self.min_fiabilite_spin.setRange(0, 100)
        self.min_fiabilite_spin.setValue(60)
        filter_layout.addWidget(self.min_fiabilite_spin)

        self.include_non_eval_chk = QCheckBox("Inclure non évaluées")
        self.include_non_eval_chk.setChecked(True)
        filter_layout.addWidget(self.include_non_eval_chk)

        self.layout.addLayout(filter_layout)
        
        # Zone de défilement pour les graphiques (embeddés dans l'interface)
        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        # Make plots area expand and prefer more vertical space
        self.plots_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plots_container = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.plots_scroll.setWidget(self.plots_container)
        # Increase the minimum height so plots get priority
        self.plots_scroll.setMinimumHeight(520)
        self.layout.addWidget(self.plots_scroll)

        # Tableau de résultats
        self.results_table = QTableWidget()
        # Keep results table but reduce its vertical footprint so plots get more space
        self.results_table.setMinimumHeight(120)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Add derivative columns + Fiabilité to the results
        self.results_table.setColumnCount(13)
        self.results_table.setHorizontalHeaderLabels([
            "Symbole", "Signal", "Score", "Prix", "Tendance",
            "RSI", "Volume moyen", "Domaine",
            "Fiabilité", "dPrice", "dMACD", "dRSI", "dVolRel"
        ])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.layout.addWidget(self.results_table)
        
        # Connexions des boutons
        self.pop_add_btn.clicked.connect(lambda: self.add_symbol(self.popular_list, "popular_symbols.txt"))
        self.pop_del_btn.clicked.connect(lambda: self.remove_selected(self.popular_list, "popular_symbols.txt"))
        self.pop_show_btn.clicked.connect(lambda: self.show_selected(self.popular_list))
        self.mes_add_btn.clicked.connect(lambda: self.add_symbol(self.mes_list, "mes_symbols.txt"))
        self.mes_del_btn.clicked.connect(lambda: self.remove_selected(self.mes_list, "mes_symbols.txt"))
        self.mes_show_btn.clicked.connect(lambda: self.show_selected(self.mes_list))
        
    def add_symbol(self, list_widget, filename):
        text, ok = QInputDialog.getText(self, "Ajouter symbole", "Symbole (ex: AAPL):")
        if ok and text:
            symbol = text.strip().upper()
            if not symbol:
                return
            exists = any(list_widget.item(i).text() == symbol for i in range(list_widget.count()))
            if exists:
                QMessageBox.information(self, "Info", f"{symbol} existe déjà dans la liste")
                return
            item = QListWidgetItem(symbol)
            item.setData(Qt.UserRole, symbol)
            list_widget.addItem(item)
            try:
                from qsi import save_symbols_to_txt
                symbols = [list_widget.item(i).data(Qt.UserRole) if list_widget.item(i).data(Qt.UserRole) is not None else list_widget.item(i).text() for i in range(list_widget.count())]
                save_symbols_to_txt(symbols, filename)
            except Exception:
                pass

    def remove_selected(self, list_widget, filename):
        items = list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Veuillez sélectionner au moins un symbole à supprimer")
            return
        for it in items:
            list_widget.takeItem(list_widget.row(it))
        try:
            from qsi import save_symbols_to_txt
            symbols = [list_widget.item(i).data(Qt.UserRole) if list_widget.item(i).data(Qt.UserRole) is not None else list_widget.item(i).text() for i in range(list_widget.count())]
            save_symbols_to_txt(symbols, filename)
        except Exception:
            pass

    def show_selected(self, list_widget):
        items = list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Veuillez sélectionner au moins un symbole à afficher")
            return
        symbols = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in items]
        self.symbol_input.setText(", ".join(symbols))

    def analyze_popular_signals(self):
        popular_symbols = [self.popular_list.item(i).text() for i in range(self.popular_list.count())]
        mes_symbols = [self.mes_list.item(i).text() for i in range(self.mes_list.count())]
        period = self.period_input.text().strip()
        
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer une période d'analyse valide (ex: 12mo)")
            return
        
        self.analyze_button.setEnabled(False)
        
        # Afficher la progression
        self.progress = QProgressDialog("Analyse en cours...", "Annuler", 0, 0, self)
        self.progress.setWindowTitle("Analyse")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setMinimumWidth(400)
        
        # Lancer l'analyse dans un thread
        self.analysis_thread = AnalysisThread(popular_symbols, mes_symbols, period)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.start()

    def clear_plots(self):
        # remove all widgets from plots_layout
        for i in reversed(range(self.plots_layout.count())):
            w = self.plots_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

    def on_download_complete(self, result):
        # Called when the DownloadThread finishes
        # Re-enable buttons
        self.analyze_button.setEnabled(True)
        self.backtest_button.setEnabled(True)
        self.popular_signals_button.setEnabled(True)

        if self.progress:
            self.progress.close()

        data = result.get('data', {}) if isinstance(result, dict) else {}
        backtests = result.get('backtest_results', []) if isinstance(result, dict) else []
        # Build result rows (collect data first, then filter & render plots only for filtered symbols)
        self.current_results = []

        for symbol, stock_data in data.items():
            try:
                prices = stock_data['Close']
                volumes = stock_data['Volume']

                # domain best-effort
                try:
                    import yfinance as yf
                    info = yf.Ticker(symbol).info
                    domaine = info.get('sector', 'Inconnu')
                except Exception:
                    domaine = 'Inconnu'

                try:
                    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(prices, volumes, domaine=domaine, return_derivatives=True)
                except Exception:
                    sig, last_price, trend, last_rsi, volume_mean, score = ("NEUTRE", 0.0, False, 0.0, 0.0, 0.0)
                    derivatives = {}

                row_info = {
                    'Symbole': symbol,
                    'Signal': sig,
                    'Score': score,
                    'Prix': last_price,
                    'Tendance': 'Hausse' if trend else 'Baisse',
                    'RSI': last_rsi,
                    'Domaine': domaine,
                    'Volume moyen': volume_mean,
                    'dPrice': round(derivatives.get('price_slope', 0.0), 6),
                    'dMACD': round(derivatives.get('macd_slope', 0.0), 6),
                    'dRSI': round(derivatives.get('rsi_slope', 0.0), 6),
                    'dVolRel': round(derivatives.get('volume_slope_rel', 0.0), 6)
                }

                self.current_results.append(row_info)
            except Exception:
                continue

        # Attach fiabilite from backtests if present
        if backtests:
            bt_map = {b['Symbole']: b for b in backtests}
            for r in self.current_results:
                sym = r['Symbole']
                if sym in bt_map:
                    r['Fiabilite'] = bt_map[sym].get('taux_reussite', 'N/A')
                else:
                    r['Fiabilite'] = 'N/A'
        else:
            for r in self.current_results:
                r['Fiabilite'] = 'N/A'

        # Apply fiabilité filter as requested by user
        min_f = getattr(self, 'min_fiabilite_spin', None)
        include_non = getattr(self, 'include_non_eval_chk', None)
        try:
            min_val = int(min_f.value()) if min_f is not None else 60
        except Exception:
            min_val = 60
        try:
            include_non_eval = bool(include_non.isChecked()) if include_non is not None else True
        except Exception:
            include_non_eval = True

        filtered = []
        for r in self.current_results:
            fiab = r.get('Fiabilite', 'N/A')
            try:
                if fiab == 'N/A':
                    if include_non_eval:
                        filtered.append(r)
                else:
                    # numeric
                    if float(fiab) >= float(min_val):
                        filtered.append(r)
            except Exception:
                # fallback include
                if include_non_eval:
                    filtered.append(r)

        # Replace the symbol lists area with the filtered symbols (popular list) to match user's request
        try:
            self.popular_list.clear()
            self.mes_list.clear()
            for r in filtered:
                self.popular_list.addItem(QListWidgetItem(r['Symbole']))
        except Exception:
            pass

        # Render plots only for filtered symbols (embedded). If embedding fails, fallback to external plots.
        self.clear_plots()
        filtered_symbols = [r['Symbole'] for r in filtered]
        try:
            for i, sym in enumerate(filtered_symbols):
                stock_data = data.get(sym)
                if not stock_data:
                    continue
                prices = stock_data['Close']
                volumes = stock_data['Volume']

                fig = Figure(figsize=(10, 3))
                ax = fig.add_subplot(111)
                show_xaxis = True if i == len(filtered_symbols) - 1 else False
                try:
                    plot_unified_chart(sym, prices, volumes, ax, show_xaxis=show_xaxis)
                except Exception:
                    ax.plot(prices.index, prices.values)
                    ax.set_title(sym)

                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(280)
                self.plots_layout.addWidget(canvas)
        except Exception:
            # Fallback: external plotting (analyse_et_affiche shows plots in separate windows)
            try:
                if filtered_symbols:
                    analyse_et_affiche(filtered_symbols, period=self.period_input.text().strip() or '12mo')
            except Exception:
                pass

        # Before updating table, make popular_list display include fiabilité in text
        try:
            self.popular_list.clear()
            for r in filtered:
                fiab = r.get('Fiabilite', 'N/A')
                display = f"{r['Symbole']} ({fiab})"
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, r['Symbole'])
                self.popular_list.addItem(item)
        except Exception:
            pass

        # Finalize results displayed in table
        self.current_results = filtered
        self.update_results_table()

    def on_analysis_progress(self, message):
        if self.progress:
            self.progress.setLabelText(message)
            QApplication.processEvents()

    def on_analysis_complete(self, result):
        # Re-enable all buttons
        self.analyze_button.setEnabled(True)
        self.backtest_button.setEnabled(True)
        self.popular_signals_button.setEnabled(True)
        
        if self.progress:
            self.progress.close()
        
        # Stocker les résultats
        self.current_results = result.get('signals', [])
        
        # Afficher les résultats
        self.update_results_table()

    def on_analysis_error(self, error_msg):
        self.analyze_button.setEnabled(True)
        if self.progress:
            self.progress.close()
        
        QMessageBox.critical(self, "Erreur", f"Erreur pendant l'analyse:\n{error_msg}")

    def analyze_stock(self):
        # Get list of symbols from input
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()]
        if not symbols:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer au moins un symbole")
            return

        # Get analysis period
        period = self.period_input.text().strip()
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer une période d'analyse valide (ex: 12mo)")
            return

        # Disable buttons during analysis
        self.analyze_button.setEnabled(False)
        self.backtest_button.setEnabled(False)
        self.popular_signals_button.setEnabled(False)

        # Progress dialog
        self.progress = QProgressDialog("Téléchargement et analyse...", "Annuler", 0, 0, self)
        self.progress.setWindowTitle("Analyse")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setMinimumWidth(400)

        # Launch download thread (no backtest)
        self.download_thread = DownloadThread(symbols, period, do_backtest=False)
        self.download_thread.finished.connect(self.on_download_complete)
        self.download_thread.error.connect(self.on_analysis_error)
        self.download_thread.progress.connect(self.on_analysis_progress)
        self.download_thread.start()

    def analyse_and_backtest(self):
        # Get list of symbols from input
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()]
        if not symbols:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer au moins un symbole")
            return

        period = self.period_input.text().strip()
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer une période d'analyse valide (ex: 12mo)")
            return

        # Disable buttons during analysis
        self.analyze_button.setEnabled(False)
        self.backtest_button.setEnabled(False)
        self.popular_signals_button.setEnabled(False)

        # Progress dialog
        self.progress = QProgressDialog("Téléchargement et backtest...", "Annuler", 0, 0, self)
        self.progress.setWindowTitle("Backtest")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setMinimumWidth(400)

        # Launch download thread with backtest true
        self.download_thread = DownloadThread(symbols, period, do_backtest=True)
        self.download_thread.finished.connect(self.on_download_complete)
        self.download_thread.error.connect(self.on_analysis_error)
        self.download_thread.progress.connect(self.on_analysis_progress)
        self.download_thread.start()
    
    def update_results_table(self):
        self.results_table.setRowCount(0)  # Clear table
        if not hasattr(self, 'current_results'):
            return
            
        for signal in self.current_results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            # Ajouter les données
            self.results_table.setItem(row, 0, QTableWidgetItem(signal['Symbole']))
            self.results_table.setItem(row, 1, QTableWidgetItem(signal['Signal']))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{signal['Score']:.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{signal['Prix']:.2f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(signal['Tendance']))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{signal['RSI']:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{signal['Volume moyen']:,.0f}"))
            self.results_table.setItem(row, 7, QTableWidgetItem(signal['Domaine']))
            # Fiabilité then derivative columns (may be missing on older results)
            fiab = signal.get('Fiabilite', 'N/A')
            try:
                fiab_text = f"{float(fiab):.0f}%" if fiab != 'N/A' else 'N/A'
            except Exception:
                fiab_text = str(fiab)
            self.results_table.setItem(row, 8, QTableWidgetItem(fiab_text))

            dprice = signal.get('dPrice', 0.0)
            dmacd = signal.get('dMACD', 0.0)
            drsi = signal.get('dRSI', 0.0)
            dvol = signal.get('dVolRel', 0.0)
            self.results_table.setItem(row, 9, QTableWidgetItem(f"{dprice:.6f}"))
            self.results_table.setItem(row, 10, QTableWidgetItem(f"{dmacd:.6f}"))
            self.results_table.setItem(row, 11, QTableWidgetItem(f"{drsi:.6f}"))
            self.results_table.setItem(row, 12, QTableWidgetItem(f"{dvol:.6f}"))
    
    def sort_results(self, index):
        if not hasattr(self, 'current_results'):
            return
            
        sort_options = {
            0: ('Prix', False),      # Prix croissant
            1: ('Prix', True),       # Prix décroissant
            2: ('Score', False),     # Score croissant
            3: ('Score', True),      # Score décroissant
            4: ('RSI', False),       # RSI croissant
            5: ('RSI', True),        # RSI décroissant
            6: ('Volume moyen', False),  # Volume croissant
            7: ('Volume moyen', True),    # Volume décroissant
            8: ('Fiabilite', False),      # Fiabilité croissant
            9: ('Fiabilite', True)        # Fiabilité décroissant
        }
        
        if index in sort_options:
            key, reverse = sort_options[index]

            def keyfn(x):
                v = x.get(key, None)
                # handle 'N/A' and missing
                if v is None:
                    return float('-inf') if not reverse else float('inf')
                if isinstance(v, (int, float)):
                    return float(v)
                try:
                    return float(v)
                except Exception:
                    # If Fiabilite is 'N/A' or other text - treat as very small
                    return float('-inf') if not reverse else float('inf')

            self.current_results.sort(key=keyfn, reverse=reverse)
            self.update_results_table()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    analyse_signaux_populaires(popular_symbols, mes_symbols, period=period, plot_all=True)
    window.setWindowTitle("Stock Analysis Tool")
    window.show()
    sys.exit(app.exec_())