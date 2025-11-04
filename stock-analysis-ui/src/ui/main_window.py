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
import sys
import os
import math

# Ensure project `src` root is on sys.path
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)
from qsi import analyse_signaux_populaires, analyse_et_affiche, popular_symbols, mes_symbols, period
from qsi import download_stock_data, backtest_signals, plot_unified_chart, get_trading_signal, generate_trade_events


class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    def __init__(self, symbols, mes_symbols, period="12mo"):
        super().__init__()
        self.symbols = symbols
        self.mes_symbols = mes_symbols
        self.period = period
        self._stop_requested = False
    def run(self):
        try:
            # ...
            while not self._stop_requested:
                # Run analysis without opening matplotlib GUIs (we'll embed charts in the main thread)
                result = analyse_signaux_populaires(
                    self.symbols,
                    self.mes_symbols,
                    period=self.period,
                    afficher_graphiques=False,
                    plot_all=False,
                    verbose=False
                )
                self.finished.emit(result)
                break  # ou return
        except Exception as e:
            self.error.emit(str(e))
    def stop(self):
        self._stop_requested = True


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

        self.current_results = []

        
    def setup_ui(self):
        # Title
        # title_label = QLabel("Stock Analysis Tool")
        # title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        # self.layout.addWidget(title_label)

        # Input de symbole
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        self.layout.addWidget(self.symbol_input)

        # Listes de symboles
        lists_container = QHBoxLayout()
        lists_container.setSpacing(24)  # Ajuste ce chiffre, ex: 24px entre les trois zones

        # Liste populaire
        popular_layout = QHBoxLayout()
        popular_listcol = QVBoxLayout()
        popular_layout.addWidget(QLabel("Symboles populaires:"))
        self.popular_list = QListWidget()
        self.popular_list.setMaximumHeight(70)
        for s in popular_symbols:
            if s:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, s)
                self.popular_list.addItem(item)
        self.popular_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        popular_listcol.addWidget(self.popular_list)
        popular_layout.addLayout(popular_listcol)

        pop_btns = QVBoxLayout()
        pop_btns.setSpacing(2)
        self.pop_add_btn = QPushButton("Ajouter")
        self.pop_del_btn = QPushButton("Supprimer")
        self.pop_show_btn = QPushButton("Afficher")
        pop_btns.addWidget(self.pop_add_btn)
        pop_btns.addWidget(self.pop_del_btn)
        pop_btns.addWidget(self.pop_show_btn)
        popular_layout.addLayout(pop_btns)

        lists_container.addLayout(popular_layout)
       
        lists_container.addItem(QSpacerItem(48, 20, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum))  # espace “élastique” mais raisonnable

        # Liste personnelle
        mes_layout = QHBoxLayout()
        mes_listcol = QVBoxLayout()
        mes_layout.addWidget(QLabel("Mes symboles:"))
        self.mes_list = QListWidget()
        self.mes_list.setMaximumHeight(70)
        for s in mes_symbols:
            if s:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, s)
                self.mes_list.addItem(item)
        self.mes_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        mes_listcol.addWidget(self.mes_list)
        mes_layout.addLayout(mes_listcol)

        mes_btns = QVBoxLayout()
        mes_btns.setSpacing(2)
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

        self.popular_list.setMaximumWidth(280)
        self.mes_list.setMaximumWidth(280)


        # Boutons d'analyse
        buttons_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_stock)
        buttons_layout.addWidget(self.analyze_button)
        self.backtest_button = QPushButton("Analyze and Backtest")
        self.backtest_button.clicked.connect(self.analyse_and_backtest)
        buttons_layout.addWidget(self.backtest_button)
        self.popular_signals_button = QPushButton("Analyser mouvements fiables (populaires)")
        self.popular_signals_button.clicked.connect(self.analyze_popular_signals)
        buttons_layout.addWidget(self.popular_signals_button)
        # Toggle details (collapse/expand bottom panels)
        self.toggle_bottom_btn = QPushButton("Masquer détails")
        self.toggle_bottom_btn.setCheckable(True)
        self.toggle_bottom_btn.clicked.connect(self.toggle_bottom)
        buttons_layout.addWidget(self.toggle_bottom_btn)
        self.layout.addLayout(buttons_layout)

        # # Options de tri
        # sort_layout = QHBoxLayout()
        # sort_layout.addWidget(QLabel("Trier par:"))
        # self.sort_combo = QComboBox()
        # self.sort_combo.addItems([
        #     "Prix (croissant)", "Prix (décroissant)",
        #     "Score (croissant)", "Score (décroissant)",
        #     "RSI (croissant)", "RSI (décroissant)",
        #     "Volume (croissant)", "Volume (décroissant)",
        #     "Fiabilité (croissant)", "Fiabilité (décroissant)",
        #     "Rev. Growth (%) (décroissant)", "Rev. Growth (%) (croissant)",
        #     "Gross Margin (%) (décroissant)", "Gross Margin (%) (croissant)",
        #     "FCF (B$) (décroissant)", "D/E Ratio (croissant)",
        #     "Market Cap (B$) (décroissant)", "Market Cap (B$) (croissant)",
        # ])
        # self.sort_combo.currentIndexChanged.connect(self.sort_results)
        # sort_layout.addWidget(self.sort_combo)
        # self.layout.addLayout(sort_layout)

        # # Filtres
        # filterlayout = QHBoxLayout()
        # filterlayout.addWidget(QLabel("Fiabilité min (%):"))
        # self.min_fiabilite_spin = QSpinBox()
        # self.min_fiabilite_spin.setRange(0, 100)
        # self.min_fiabilite_spin.setValue(60)
        # filterlayout.addWidget(self.min_fiabilite_spin)
        # filterlayout.addWidget(QLabel("  Nb trades min:"))
        # self.min_trades_spin = QSpinBox()
        # self.min_trades_spin.setRange(0, 100)
        # self.min_trades_spin.setValue(5)
        # filterlayout.addWidget(self.min_trades_spin)
        # self.include_none_val_chk = QCheckBox("Inclure non évalués")
        # self.include_none_val_chk.setChecked(True)
        # filterlayout.addWidget(self.include_none_val_chk)
        # self.layout.addLayout(filterlayout)

        # Plots area
        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        self.plots_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plots_container = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.plots_scroll.setWidget(self.plots_container)
        self.plots_scroll.setMinimumHeight(390)

        # Use a vertical splitter so plots remain visible
        from PyQt5.QtWidgets import QSplitter, QTextEdit
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.plots_scroll)

        # Bottom container
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(80)
        self.summary_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_layout.addWidget(self.summary_text)
        # Single merged table combining signals + backtest metrics
        self.merged_table = QTableWidget()
        self.merged_table.setMinimumHeight(280)
        self.merged_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        merged_columns = [
        'Symbole','Signal','Score','Prix','Tendance','RSI','Volume moyen','Domaine',
        'Fiabilite','Nb Trades','Gagnants',
        # COLONNES FINANCIÈRES
        'Rev. Growth (%)','Gross Margin (%)','FCF (B$)','D/E Ratio','Market Cap (B$)',
        # COLONNES DERIVÉES
        'dPrice','dMACD','dRSI','dVolRel',
        # COLONNES BACKTEST
        # 'Trades','Gagnants','Taux réussite (%)','Gain total ($)','Gain moyen ($)'#,'Drawdown max (%)'
        'Gain total ($)','Gain moyen ($)'#,'Drawdown max (%)'
        ]

        self.merged_table.setColumnCount(len(merged_columns))
        self.merged_table.setHorizontalHeaderLabels(merged_columns)
        self.merged_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Allow sorting by clicking headers (we also provide numeric data via Qt.EditRole)
        self.merged_table.setSortingEnabled(True)
        bottom_layout.addWidget(self.merged_table)

        # Keep a reference to bottom container to toggle visibility later
        self.bottom_container = bottom_container

        self.splitter.addWidget(bottom_container)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        # Try to set an initial sensible ratio so plots are visible by default
        try:
            total_h = max(600, self.height())
            top_h = int(total_h * 0.72)
            bottom_h = total_h - top_h
            self.splitter.setSizes([top_h, bottom_h])
        except Exception:
            pass

        self.layout.addWidget(self.splitter)

        # Connexions des boutons
        self.pop_add_btn.clicked.connect(lambda: self.add_symbol(self.popular_list, "popular_symbols.txt"))
        self.pop_del_btn.clicked.connect(lambda: self.remove_selected(self.popular_list, "popular_symbols.txt"))
        self.pop_show_btn.clicked.connect(lambda: self.show_selected(self.popular_list))
        self.mes_add_btn.clicked.connect(lambda: self.add_symbol(self.mes_list, "mes_symbols.txt"))
        self.mes_del_btn.clicked.connect(lambda: self.remove_selected(self.mes_list, "mes_symbols.txt"))
        self.mes_show_btn.clicked.connect(lambda: self.show_selected(self.mes_list))
    
    def validate_ticker(self, symbol):
        """Validation rapide mais moins fiable"""
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Vérifier juste la présence de 'regularMarketPrice' ou 'symbol'
            return info.get('regularMarketPrice') is not None or info.get('symbol') is not None
        except:
            return False


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
            
             # Afficher un dialogue de progression pendant la validation
            progress = QProgressDialog(f"Validation de {symbol}...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()
            
            is_valid = self.validate_ticker(symbol)
            progress.close()
            
            if not is_valid:
                QMessageBox.warning(
                    self, 
                    "Ticker invalide", 
                    f"Le symbole '{symbol}' n'est pas valide.\n"
                    "Vérifiez l'orthographe ou consultez Yahoo Finance."
                )
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
        # Use selected items if any, otherwise fallback to full lists
        selected_pop = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.popular_list.selectedItems()]
        popular_symbols = selected_pop if selected_pop else [self.popular_list.item(i).text() for i in range(self.popular_list.count())]
        selected_mes = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.mes_list.selectedItems()]
        mes_symbols = selected_mes if selected_mes else [self.mes_list.item(i).text() for i in range(self.mes_list.count())]
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
                    ticker_obj = yf.Ticker(symbol)
                    info = ticker_obj.info
                    domaine = info.get('sector', 'Inconnu')

                    # Try to extract financial metrics (robust fallbacks)
                    market_cap_val = 0.0
                    debt_to_equity_val = 0.0
                    gross_margin_val = 0.0
                    fcf_val = 0.0
                    rev_growth_val = 0.0

                    try:
                        mc = info.get('marketCap') or info.get('market_cap') or 0
                        if mc:
                            market_cap_val = float(mc) / 1e9
                    except Exception:
                        market_cap_val = 0.0

                    try:
                        debt_to_equity_val = float(info.get('debtToEquity') or info.get('debtToEquityRatio') or 0.0)
                    except Exception:
                        debt_to_equity_val = 0.0

                    try:
                        gm = info.get('grossMargins')
                        if gm is None:
                            gm = info.get('grossMargin')
                        if gm is not None:
                            gross_margin_val = float(gm) * 100.0 if abs(gm) <= 1 else float(gm)
                    except Exception:
                        gross_margin_val = 0.0

                    try:
                        # free cash flow naming varies
                        fcf_val = float(info.get('freeCashflow') or info.get('freeCashFlow') or 0.0)
                        # try cashflow statement if missing
                        if not fcf_val:
                            try:
                                cf = ticker_obj.cashflow
                                # common index names: 'Free Cash Flow' or 'freeCashFlow'
                                for candidate in ['Free Cash Flow', 'freeCashFlow', 'FreeCashFlow']:
                                    if candidate in cf.index:
                                        fcf_val = float(cf.loc[candidate].iloc[0])
                                        break
                            except Exception:
                                pass
                        # convert to billions for display
                        if fcf_val:
                            fcf_val = fcf_val / 1e9
                    except Exception:
                        fcf_val = 0.0

                    try:
                        # revenue growth via financials: compare most recent two periods
                        try:
                            fin = ticker_obj.financials
                            # common labels for revenue
                            rev_candidates = ['Total Revenue', 'Revenue', 'totalRevenue', 'TotalRevenue']
                            rev_vals = None
                            for cand in rev_candidates:
                                if cand in fin.index:
                                    rev_vals = fin.loc[cand].dropna()
                                    break
                            if rev_vals is not None and len(rev_vals) >= 2:
                                rev_growth_val = ((float(rev_vals.iloc[0]) - float(rev_vals.iloc[1])) / float(rev_vals.iloc[1])) * 100.0
                        except Exception:
                            rev_growth_val = 0.0
                    except Exception:
                        rev_growth_val = 0.0
                except Exception:
                    domaine = 'Inconnu'

                try:
                    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
                        prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol  # Ajouter symbol
                    )
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
                    'dVolRel': round(derivatives.get('volume_slope_rel', 0.0), 6),
                    # NOUVELLES COLONNES FINANCIÈRES
                    'Rev. Growth (%)': round(float(derivatives.get('rev_growth_val')) if derivatives.get('rev_growth_val') else float(rev_growth_val or 0.0), 2),
                    'Gross Margin (%)': round(float(derivatives.get('gross_margin_val')) if derivatives.get('gross_margin_val') else float(gross_margin_val or 0.0), 2),
                    'FCF (B$)': round(float(derivatives.get('fcf_val')) if derivatives.get('fcf_val') else float(fcf_val or 0.0), 2),
                    'D/E Ratio': round(float(derivatives.get('debt_to_equity_val')) if derivatives.get('debt_to_equity_val') else float(debt_to_equity_val or 0.0), 2),
                    'Market Cap (B$)': round(float(derivatives.get('market_cap_val')) if derivatives.get('market_cap_val') else float(market_cap_val or 0.0), 2)
                }

                self.current_results.append(row_info)
            except Exception:
                continue

        # Attach fiabilite AND nb_trades from backtests if present
        if backtests:
            bt_map = {b['Symbole']: b for b in backtests}
            for r in self.current_results:
                sym = r['Symbole']
                if sym in bt_map:
                    r['Fiabilite'] = bt_map[sym].get('taux_reussite', 'N/A')
                    r['NbTrades'] = bt_map[sym].get('trades', 0)
                else:
                    r['Fiabilite'] = 'N/A'
                    r['NbTrades'] = 0
        else:
            for r in self.current_results:
                r['Fiabilite'] = 'N/A'
                r['NbTrades'] = 0


        # Apply fiabilité and nb trades filters
        min_f = getattr(self, 'min_fiabilite_spin', None)
        min_t = getattr(self, 'min_trades_spin', None)
        include_non = getattr(self, 'include_none_val_chk', None)
        try:
            min_val = int(min_f.value()) if min_f is not None else 60
        except Exception:
            min_val = 60
        try:
            min_trades = int(min_t.value()) if min_t is not None else 5
        except Exception:
            min_trades = 5
        try:
            include_none_val = bool(include_non.isChecked()) if include_non is not None else True
        except Exception:
            include_none_val = True

        filtered = []
        for r in self.current_results:
            fiab = r.get('Fiabilite', 'N/A')
            nb_trades = r.get('NbTrades', 0)

            # Filtre sur nb trades
            try:
                # If nb_trades == 0 (no backtest), allow inclusion when include_none_val is True
                if int(nb_trades) == 0:
                    if not include_none_val:
                        continue
                else:
                    if int(nb_trades) < min_trades:
                        continue
            except Exception:
                # If we can't parse nb_trades, only include when include_none_val is True
                if not include_none_val:
                    continue

            # Filtre sur fiabilité
            try:
                if fiab == 'N/A':
                    if include_none_val:
                        filtered.append(r)
                else:
                    if float(fiab) >= float(min_val):
                        filtered.append(r)
            except Exception:
                if include_none_val:
                    filtered.append(r)

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
                canvas.setMinimumHeight(200)
                self.plots_layout.addWidget(canvas)
        except Exception:
            # Fallback: external plotting (analyse_et_affiche shows plots in separate windows)
            try:
                if filtered_symbols:
                    analyse_et_affiche(filtered_symbols, period=self.period_input.text().strip() or '12mo')
            except Exception:
                pass

        # NOTE: Do not replace the user's popular/mes lists with filtered results.
        # Instead, we can optionally update item tooltips to show fiabilité without
        # modifying the list contents. Keep original lists intact so the user
        # doesn't lose their configured popular symbols.
        try:
            # Map fiabilité by symbol for quick lookup
            fiab_map = {r['Symbole']: r.get('Fiabilite', 'N/A') for r in filtered}
            # Update tooltip for items in popular_list only (non-destructive)
            for i in range(self.popular_list.count()):
                item = self.popular_list.item(i)
                sym = item.data(Qt.UserRole) if item.data(Qt.UserRole) is not None else item.text()
                if sym in fiab_map:
                    item.setToolTip(f"Fiabilité: {fiab_map[sym]}")
                else:
                    item.setToolTip("")
            # Do the same for mes_list
            for i in range(self.mes_list.count()):
                item = self.mes_list.item(i)
                sym = item.data(Qt.UserRole) if item.data(Qt.UserRole) is not None else item.text()
                if sym in fiab_map:
                    item.setToolTip(f"Fiabilité: {fiab_map[sym]}")
                else:
                    item.setToolTip("")
        except Exception:
            pass

        # Finalize results displayed in table
        self.update_results_table()
        # If backtest results present, render the backtest summary and table
        try:
            backtests = result.get('backtest_results', []) if isinstance(result, dict) else []
            if backtests:
                # current_results contains signal rows with Domaine if available
                self.render_backtest_summary_and_table(backtests, self.current_results)
        except Exception:
            pass

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
        
        # Enrichir les résultats (calculer les dérivées si non présentes)
        try:
            # If signals came without derivatives, compute them by downloading data per symbol
            for r in self.current_results:
                # Only compute if dPrice not present or zero
                if not r.get('dPrice') or float(r.get('dPrice', 0)) == 0.0:
                    sym = r.get('Symbole')
                    if not sym:
                        continue
                    try:
                        data = download_stock_data([sym], self.period_input.text().strip() or '12mo').get(sym)
                        if data is None:
                            continue
                        prices = data['Close']
                        volumes = data['Volume']
                        try:
                            _sig, _last_price, _trend, _last_rsi, _vol_mean, _score, derivatives = get_trading_signal(prices, volumes, domaine=r.get('Domaine', 'Inconnu'), return_derivatives=True,symbol=sym)
                        except Exception:
                            derivatives = {}

                        r['dPrice'] = round(derivatives.get('price_slope', 0.0), 6)
                        r['dMACD'] = round(derivatives.get('macd_slope', 0.0), 6)
                        r['dRSI'] = round(derivatives.get('rsi_slope', 0.0), 6)
                        r['dVolRel'] = round(derivatives.get('volume_slope_rel', 0.0), 6)
                    except Exception:
                        # leave defaults
                        r.setdefault('dPrice', 0.0)
                        r.setdefault('dMACD', 0.0)
                        r.setdefault('dRSI', 0.0)
                        r.setdefault('dVolRel', 0.0)
        except Exception:
            pass

        # Afficher les résultats
        self.update_results_table()
        # Also embed the final analysis charts (top buys / sells) returned by the analysis
        try:
            # Clear existing plots
            self.clear_plots()

            top_buys = result.get('top_achats_fiables', []) if isinstance(result, dict) else []
            top_sells = result.get('top_ventes_fiables', []) if isinstance(result, dict) else []

            # Helper to embed a list of symbols as canvases
            def embed_symbol_list(symbol_list, title_prefix=""):
                if not symbol_list:
                    return
                for i, s in enumerate(symbol_list):
                    sym = s['Symbole'] if isinstance(s, dict) and 'Symbole' in s else s
                    try:
                        stock_data = download_stock_data([sym], period=self.period_input.text().strip() or '12mo').get(sym)
                        if not stock_data:
                            continue
                        prices = stock_data['Close']
                        volumes = stock_data['Volume']

                        fig = Figure(figsize=(10, 3))
                        ax = fig.add_subplot(111)
                        show_xaxis = True
                        try:
                            plot_unified_chart(sym, prices, volumes, ax, show_xaxis=show_xaxis)
                        except Exception:
                            ax.plot(prices.index, prices.values)
                            ax.set_title(sym)

                        # Add trade markers based on the lightweight simulator
                        try:
                            try:
                                info = None
                                import yfinance as yf
                                info = yf.Ticker(sym).info
                                domaine = info.get('sector', 'Inconnu') if info else 'Inconnu'
                            except Exception:
                                domaine = 'Inconnu'

                            events = generate_trade_events(prices, volumes, domaine)
                            for ev in events:
                                if ev.get('type') == 'BUY':
                                    ax.scatter(ev['date'], ev['price'], marker='^', s=80, color='green', edgecolor='black', zorder=6)
                                elif ev.get('type') == 'SELL':
                                    ax.scatter(ev['date'], ev['price'], marker='v', s=80, color='red', edgecolor='black', zorder=6)
                        except Exception:
                            pass

                        canvas = FigureCanvas(fig)
                        canvas.setMinimumHeight(240)
                        self.plots_layout.addWidget(canvas)
                    except Exception:
                        continue

            # Embed buys then sells (if any)
            embed_symbol_list(top_buys, "Top ACHAT")
            embed_symbol_list(top_sells, "Top VENTE")
        except Exception:
            # If anything fails, silently ignore — table already updated
            pass
        # Render backtest summary/table if present in the result
        try:
            backtests = result.get('backtest_results', []) if isinstance(result, dict) else []
            signals = result.get('signals', []) if isinstance(result, dict) else []
            if backtests:
                self.render_backtest_summary_and_table(backtests, signals)
        except Exception:
            pass

    def on_analysis_error(self, error_msg):
        self.analyze_button.setEnabled(True)
        if self.progress:
            self.progress.close()
        
        QMessageBox.critical(self, "Erreur", f"Erreur pendant l'analyse:\n{error_msg}")

    def analyze_stock(self):
        # Get list of symbols from input or from selection in lists
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()]
        if not symbols:
            # If no manual input, use selected items from the lists (popular first, then mes)
            selected = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.popular_list.selectedItems()]
            if not selected:
                selected = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.mes_list.selectedItems()]
            symbols = [s.strip().upper() for s in selected if s]
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
        # For consistency with 'Analyser mouvements fiables', run the full
        # analyse_signaux_populaires pipeline (which includes backtests) and
        # embed the same charts + detailed backtest info in the UI.

        # Get list of symbols from input or from selection in lists
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()]
        if not symbols:
            selected = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.popular_list.selectedItems()]
            if not selected:
                selected = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.mes_list.selectedItems()]
            symbols = [s.strip().upper() for s in selected if s]
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
        self.progress = QProgressDialog("Analyse et backtest en cours...", "Annuler", 0, 0, self)
        self.progress.setWindowTitle("Analyse + Backtest")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setMinimumWidth(400)

        # Use the AnalysisThread which calls analyse_signaux_populaires (no plt.show()
        # in background). Pass the selected symbols as the "popular_symbols" input
        # so the function analyzes/backtests those symbols and returns the same
        # result structure used for the "Analyser mouvements fiables" flow.
        selected_pop = symbols
        selected_mes = []

        self.analysis_thread = AnalysisThread(selected_pop, selected_mes, period)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.start()

    def update_results_table(self):
        """Fill the merged table (`self.merged_table`) with current results plus backtest metrics.
        Numeric values are stored via Qt.EditRole to enable correct numeric sorting."""
        if not hasattr(self, 'current_results'):
            return

        self.merged_table.setRowCount(0)
        results_to_display = getattr(self, 'filtered_results', self.current_results)
        bt_map = getattr(self, 'backtest_map', {})

        for signal in results_to_display:
            try:
                row = self.merged_table.rowCount()
                self.merged_table.insertRow(row)
                sym = signal.get('Symbole', '')

                # Basic columns
                self.merged_table.setItem(row, 0, QTableWidgetItem(str(sym)))
                self.merged_table.setItem(row, 1, QTableWidgetItem(str(signal.get('Signal', ''))))

                score = float(signal.get('Score', 0.0) or 0.0)
                item = QTableWidgetItem(f"{score:.2f}")
                item.setData(Qt.EditRole, score)
                self.merged_table.setItem(row, 2, item)

                prix = float(signal.get('Prix', 0.0) or 0.0)
                item = QTableWidgetItem(f"{prix:.2f}")
                item.setData(Qt.EditRole, prix)
                self.merged_table.setItem(row, 3, item)

                self.merged_table.setItem(row, 4, QTableWidgetItem(str(signal.get('Tendance', ''))))

                rsi = float(signal.get('RSI', 0.0) or 0.0)
                item = QTableWidgetItem(f"{rsi:.2f}")
                item.setData(Qt.EditRole, rsi)
                self.merged_table.setItem(row, 5, item)

                vol = signal.get('Volume moyen', 0.0) or 0.0
                item = QTableWidgetItem(f"{vol:,.0f}")
                item.setData(Qt.EditRole, float(vol))
                self.merged_table.setItem(row, 6, item)

                self.merged_table.setItem(row, 7, QTableWidgetItem(str(signal.get('Domaine', ''))))

                # Fiabilite and NbTrades (from signal or backtest)
                fiab = signal.get('Fiabilite')
                nb_trades = signal.get('NbTrades')
                # if missing, check backtest map
                bt = bt_map.get(sym, {})
                if (fiab is None or fiab == 'N/A') and bt:
                    fiab = bt.get('taux_reussite', 'N/A')
                if (nb_trades is None or nb_trades == 0) and bt:
                    nb_trades = bt.get('trades', 0)

                # Fiabilité display
                if fiab is None or fiab == 'N/A':
                    fiab_text = 'N/A'
                    fiab_val = None
                else:
                    try:
                        fiab_val = float(fiab)
                        fiab_text = f"{fiab_val:.0f}%"
                    except Exception:
                        fiab_text = str(fiab)
                        fiab_val = None

                item = QTableWidgetItem(fiab_text)
                if fiab_val is not None:
                    item.setData(Qt.EditRole, fiab_val)
                self.merged_table.setItem(row, 8, item)

                # NbTrades
                try:
                    nb_int = int(nb_trades or 0)
                except Exception:
                    nb_int = 0
                item = QTableWidgetItem(str(nb_int))
                item.setData(Qt.EditRole, nb_int)
                self.merged_table.setItem(row, 9, item)

                # Gagnants
                gagnants = int(bt.get('gagnants', 0)) if bt else 0
                item = QTableWidgetItem(str(gagnants))
                item.setData(Qt.EditRole, gagnants)
                self.merged_table.setItem(row, 10, item)
                
                # Colonnes Techniques 
                # Colonne 14: Rev. Growth (%)
                rev_growth = float(signal.get('Rev. Growth (%)', 0.0) or 0.0)
                item = QTableWidgetItem(f"{rev_growth:.2f}")
                item.setData(Qt.EditRole, rev_growth)
                self.merged_table.setItem(row, 11, item)

                # Colonne 15: Gross Margin (%)
                gross_margin = float(signal.get('Gross Margin (%)', 0.0) or 0.0)
                item = QTableWidgetItem(f"{gross_margin:.2f}")
                item.setData(Qt.EditRole, gross_margin)
                self.merged_table.setItem(row, 12, item)

                # Colonne 16: FCF (B$)
                fcf = float(signal.get('FCF (B$)', 0.0) or 0.0)
                item = QTableWidgetItem(f"{fcf:.2f}")
                item.setData(Qt.EditRole, fcf)
                self.merged_table.setItem(row, 13, item)

                # Colonne 17: D/E Ratio
                de_ratio = float(signal.get('D/E Ratio', 0.0) or 0.0)
                item = QTableWidgetItem(f"{de_ratio:.2f}")
                item.setData(Qt.EditRole, de_ratio)
                self.merged_table.setItem(row, 14, item)

                # Colonne 18: Market Cap (B$)
                market_cap = float(signal.get('Market Cap (B$)', 0.0) or 0.0)
                item = QTableWidgetItem(f"{market_cap:.2f}")
                item.setData(Qt.EditRole, market_cap)
                self.merged_table.setItem(row, 15, item)

                # Derivatives
                dprice = float(signal.get('dPrice', 0.0) or 0.0)
                item = QTableWidgetItem(f"{dprice:.6f}")
                item.setData(Qt.EditRole, dprice)
                self.merged_table.setItem(row, 16, item)

                dmacd = float(signal.get('dMACD', 0.0) or 0.0)
                item = QTableWidgetItem(f"{dmacd:.6f}")
                item.setData(Qt.EditRole, dmacd)
                self.merged_table.setItem(row, 17, item)

                drsi = float(signal.get('dRSI', 0.0) or 0.0)
                item = QTableWidgetItem(f"{drsi:.6f}")
                item.setData(Qt.EditRole, drsi)
                self.merged_table.setItem(row, 18, item)

                dvol = float(signal.get('dVolRel', 0.0) or 0.0)
                item = QTableWidgetItem(f"{dvol:.6f}")
                item.setData(Qt.EditRole, dvol)
                self.merged_table.setItem(row, 19, item)

                # Backtest metrics (if available)
                trades = int(bt.get('trades', 0)) if bt else 0
                taux = float(bt.get('taux_reussite', 0.0)) if bt else 0.0
                gain_total = float(bt.get('gain_total', 0.0)) if bt else 0.0
                gain_moy = float(bt.get('gain_moyen', 0.0)) if bt else 0.0
                #drawdown = float(bt.get('drawdown_max', 0.0)) if bt else 0.0

                # Place backtest metrics in their proper columns (19-23)
                # item = QTableWidgetItem(str(trades))
                # item.setData(Qt.EditRole, trades)
                # self.merged_table.setItem(row, 19, item)

                # item = QTableWidgetItem(f"{taux:.1f}")
                # item.setData(Qt.EditRole, taux)
                # self.merged_table.setItem(row, 21, item)

                item = QTableWidgetItem(f"{gain_total:.2f}")
                item.setData(Qt.EditRole, gain_total)
                self.merged_table.setItem(row, 20, item)

                item = QTableWidgetItem(f"{gain_moy:.2f}")
                item.setData(Qt.EditRole, gain_moy)
                self.merged_table.setItem(row, 21, item)

                # item = QTableWidgetItem(f"{drawdown:.2f}")
                # item.setData(Qt.EditRole, drawdown)
                # self.merged_table.setItem(row, 19, item)

            except Exception:
                continue
    
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
            9: ('Fiabilite', True),       # Fiabilité décroissant
            10: ('rev_growth_val', True),  # Rev. Growth décroissant
            11: ('rev_growth_val', False), # Rev. Growth croissant
            12: ('gross_margin_val', True),# Gross Margin décroissant
            13: ('gross_margin_val', False),# Gross Margin croissant
            14: ('market_cap_val', True),   # Market Cap décroissant
            15: ('market_cap_val', False),  # Market Cap croissant
            16: ('fcf_val', True),         # FCF décroissant
            17: ('debt_to_equity_val', False) # D/E Ratio croissant
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

            def keyfn(x):
                v = x.get(key, None)
                if v is None or v == 'N/A':
                    return float('-inf') if not reverse else float('inf')
                try:
                    return float(v)
                except Exception:
                    return float('-inf') if not reverse else float('inf')

    def toggle_bottom(self, checked: bool):
        """Hide/show the bottom summary/backtest/results panel."""
        try:
            if checked:
                # hide bottom container and expand plots
                if hasattr(self, 'bottom_container'):
                    self.bottom_container.setVisible(False)
                self.toggle_bottom_btn.setText("Afficher détails")
                try:
                    self.splitter.setSizes([self.height(), 0])
                except Exception:
                    pass
            else:
                if hasattr(self, 'bottom_container'):
                    self.bottom_container.setVisible(True)
                self.toggle_bottom_btn.setText("Masquer détails")
                try:
                    total_h = max(600, self.height())
                    top_h = int(total_h * 0.72)
                    bottom_h = total_h - top_h
                    self.splitter.setSizes([top_h, bottom_h])
                except Exception:
                    pass
        except Exception:
            pass

    def render_backtest_summary_and_table(self, backtest_results: list, signals: list):
        """Build the summary text and populate an internal backtest map used by the merged table.

        This no longer writes to a separate backtest table; instead it stores results in
        `self.backtest_map` and injects Fiabilité/NbTrades back into `self.current_results` so
        the merged table can display and sort them.
        """
        try:
            total_trades = 0
            total_gagnants = 0
            total_gain = 0.0

            # Map symbol to domain from signals list
            domain_map = {s.get('Symbole'): s.get('Domaine', 'Inconnu') for s in signals if isinstance(s, dict)}

            domain_stats = {}

            # Build backtest map
            self.backtest_map = {}
            for br in backtest_results:
                sym = br.get('Symbole')
                trades = int(br.get('trades', 0))
                gagnants = int(br.get('gagnants', 0))
                gain_total = float(br.get('gain_total', 0.0))
                gain_moyen = float(br.get('gain_moyen', 0.0))
                #drawdown = float(br.get('drawdown_max', 0.0))
                taux = float(br.get('taux_reussite', 0.0))

                self.backtest_map[sym] = {
                    'trades': trades,
                    'gagnants': gagnants,
                    'taux_reussite': taux,
                    'gain_total': gain_total,
                    'gain_moyen': gain_moyen,
                    #'drawdown_max': drawdown
                }

                total_trades += trades
                total_gagnants += gagnants
                total_gain += gain_total

                # Domain aggregation
                domain = domain_map.get(sym, 'Inconnu')
                if domain not in domain_stats:
                    domain_stats[domain] = {'trades': 0, 'gagnants': 0, 'gain': 0.0}
                domain_stats[domain]['trades'] += trades
                domain_stats[domain]['gagnants'] += gagnants
                domain_stats[domain]['gain'] += gain_total

            taux_global = (total_gagnants / total_trades * 100) if total_trades else 0.0

            # Inject Fiabilite/NbTrades into current_results so merged table shows them
            try:
                for r in getattr(self, 'current_results', []):
                    sym = r.get('Symbole')
                    bt = self.backtest_map.get(sym)
                    if bt:
                        r['Fiabilite'] = bt.get('taux_reussite', 'N/A')
                        r['NbTrades'] = bt.get('trades', 0)
                    else:
                        r.setdefault('Fiabilite', 'N/A')
                        r.setdefault('NbTrades', 0)
            except Exception:
                pass

            # Build summary text
            lines = []
            lines.append(f"🌍 Résultat global :\n - Taux de réussite = {taux_global:.1f}%\n - Nombre de trades = {total_trades}\n - Gain total brut = {total_gain:.2f} $")
            lines.append("\n📊 Taux de réussite par domaine:")
            for dom, stats in sorted(domain_stats.items(), key=lambda x: -x[1]['trades']):
                trades = stats['trades']
                gagnants = stats['gagnants']
                taux = (gagnants / trades * 100) if trades else 0.0
                gain_dom = stats['gain']
                lines.append(f" - {dom}: Trades={trades} | Gagnants={gagnants} | Taux={taux:.1f}% | Gain brut={gain_dom:.2f} $")

            self.summary_text.setPlainText('\n'.join(lines))

            # Refresh merged table to display updated fiabilite/nb trades
            try:
                self.update_results_table()
            except Exception:
                pass

        except Exception:
            try:
                self.summary_text.setPlainText('')
            except Exception:
                pass
    
    def closeEvent(self, event):
        # Stop analysis thread if running
        try:
            if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
                self.analysis_thread.stop()
                self.analysis_thread.wait(2000)
        except Exception:
            pass
        try:
            if hasattr(self, 'download_thread') and self.download_thread.isRunning():
                self.download_thread.quit()
                self.download_thread.wait(2000)
        except Exception:
            pass
        event.accept()

    def clear_plots(self):
        for i in reversed(range(self.plots_layout.count())):
            w = self.plots_layout.itemAt(i).widget()
            if w:
                if hasattr(w, 'figure'):
                    w.figure.clear()
                    try:
                        w.close()
                    except Exception:
                        pass
                w.setParent(None)
        import gc
        gc.collect()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    #analyse_signaux_populaires(popular_symbols, mes_symbols, period=period, plot_all=True)
    window.setWindowTitle("Stock Analysis Tool")
    window.show()
    sys.exit(app.exec_())