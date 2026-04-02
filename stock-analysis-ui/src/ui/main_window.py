from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSpacerItem, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QInputDialog, QListWidget, QListWidgetItem,
    QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QComboBox, QHeaderView, QSpinBox, QCheckBox, QTabWidget, QTextEdit
)
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor
import io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import dates as mdates
import sys
import os
import math
from datetime import datetime
import yfinance as yf

# Ensure project `src` root is on sys.path
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)
from qsi import analyse_signaux_populaires, analyse_et_affiche, load_symbols_from_txt, period
from qsi import download_stock_data, backtest_signals, plot_unified_chart, get_trading_signal
import qsi
from trading_c_acceleration.qsi_optimized import extract_best_parameters

try:
    from symbol_manager import get_symbols_by_list_type, get_recent_symbols
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    SYMBOL_MANAGER_AVAILABLE = False


class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    def __init__(self, symbols, mes_symbols, period="12mo", analysis_id=0, min_holding_days=7):
        super().__init__()
        self.symbols = symbols
        self.mes_symbols = mes_symbols
        self.period = period
        self._stop_requested = False
        self.analysis_id = analysis_id  # 🔧 Identifiant d'analyse
        self.min_holding_days = max(1, int(min_holding_days))
    def run(self):
        try:
            import builtins
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.progress.emit(message)
                # Force flush to keep console output visible when running UI from a terminal
                kwargs_with_flush = dict(kwargs)
                kwargs_with_flush.setdefault('flush', True)
                original_print(*args, **kwargs_with_flush)

            builtins.print = custom_print

            while not self._stop_requested:
                # Run analysis without opening matplotlib GUIs; keep verbose to surface progress
                # Get the reliability threshold from the main window spinbox if available
                fiab_threshold = 30  # Default value
                try:
                    # Access the spinbox value from main window (passed via parent reference)
                    if hasattr(self, 'parent') and hasattr(self.parent(), 'fiab_threshold_spin'):
                        fiab_threshold = self.parent().fiab_threshold_spin.value()
                except Exception:
                    pass
                
                result = analyse_signaux_populaires(
                    self.symbols,
                    self.mes_symbols,
                    period=self.period,
                    afficher_graphiques=False,
                    plot_all=False,
                    verbose=True,
                    taux_reussite_min=fiab_threshold,
                    min_holding_days=self.min_holding_days
                )
                if not self._stop_requested:
                    # 🔧 Ajouter l'identifiant d'analyse au résultat
                    result['_analysis_id'] = self.analysis_id
                    self.finished.emit(result)
                break  # ou return
        except Exception as e:
            if not self._stop_requested:
                self.error.emit(str(e))
        finally:
            try:
                builtins.print = original_print
            except Exception:
                pass
    def stop(self):
        self._stop_requested = True


class DownloadThread(QThread):
    """Thread to download stock data and run backtests with V2.0 optimized parameters"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, symbols, period="12mo", do_backtest=False, analysis_id=0, min_holding_days=7):
        super().__init__()
        self.symbols = symbols
        self.period = period
        self.do_backtest = do_backtest
        self._stop_requested = False
        self.analysis_id = analysis_id  # 🔧 Identifiant d'analyse
        self.min_holding_days = max(1, int(min_holding_days))

    def run(self):
        try:
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.progress.emit(message)
                # Force flush to keep console output visible when running UI from a terminal
                kwargs_with_flush = dict(kwargs)
                kwargs_with_flush.setdefault('flush', True)
                original_print(*args, **kwargs_with_flush)

            import builtins
            builtins.print = custom_print

            try:
                if self._stop_requested:
                    return
                    
                data = download_stock_data(self.symbols, self.period)
                result = { 'data': data }

                if self._stop_requested:
                    return

                if self.do_backtest and data:
                    backtests = []
                    for symbol, stock_data in data.items():
                        if self._stop_requested:
                            return
                            
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

                            # ✨ Extraire les paramètres optimisés depuis la SQLite
                            try:
                                best_params = extract_best_parameters()
                            except Exception:
                                best_params = {}

                            coeffs, feature_thresholds, globals_thresholds, _, _ = best_params.get(domaine, (None, None, (4.2, -0.5), None, {}))
                            domain_coeffs = {domaine: coeffs} if coeffs else None
                            
                            # ✨ V2.0: Utiliser les paramètres optimisés si disponibles
                            backtest_kwargs = {
                                'prices': prices,
                                'volumes': volumes,
                                'domaine': domaine,
                                'montant': 50,
                                'domain_coeffs': domain_coeffs,
                                'domain_thresholds': {domaine: feature_thresholds} if feature_thresholds else None,
                                'min_holding_bars': self.min_holding_days,
                            }
                            
                            bt = backtest_signals(**backtest_kwargs)
                            
                            # Debug: vérifier si le backtest retourne des trades
                            if bt.get('trades', 0) == 0:
                                self.progress.emit(f"  ⚠️ {symbol}: Aucun trade détecté (domaine={domaine})")
                            
                            backtests.append({ 'Symbole': symbol, **bt })
                        except Exception as e:
                            self.progress.emit(f"  ⚠️ Erreur backtest {symbol}: {e}")
                            continue

                    result['backtest_results'] = backtests

                if not self._stop_requested:
                    # 🔧 Ajouter l'identifiant d'analyse au résultat
                    result['_analysis_id'] = self.analysis_id
                    self.finished.emit(result)
            finally:
                builtins.print = original_print

        except Exception as e:
            if not self._stop_requested:
                self.error.emit(str(e))

    def stop(self):
        """Stop the thread gracefully"""
        self._stop_requested = True

class LogCapture:
    """Captures stdout/stderr and writes both to QTextEdit and to original stdout/stderr.
    Thread-safe implementation using direct append (QTextEdit is thread-safe for append)."""
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.original_stdout = sys.__stdout__
        self.original_stderr = sys.__stderr__
    
    def write(self, message):
        """Write message to QTextEdit and original stdout."""
        try:
            if message and message.strip():
                # Append to QTextEdit (thread-safe)
                self.text_edit.append(message.rstrip())
                # Also print to original stdout (terminal)
                self.original_stdout.write(message)
                self.original_stdout.flush()
        except Exception:
            # Fail silently to avoid breaking print calls
            try:
                self.original_stdout.write(message)
                self.original_stdout.flush()
            except Exception:
                pass
    
    def flush(self):
        """Flush the buffer."""
        try:
            self.original_stdout.flush()
        except Exception:
            pass
    
    def isatty(self):
        """Required for some code that checks if stdout is a TTY."""
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        # 🔄 Synchroniser personal et optimization vers popular au démarrage
        if SYMBOL_MANAGER_AVAILABLE:
            try:
                from symbol_manager import sync_all_to_popular
                stats = sync_all_to_popular()
                if stats['total'] > 0:
                    print(f"🔄 Auto-sync au démarrage: {stats['total']} symboles ajoutés à popular")
            except Exception as e:
                print(f"⚠️ Erreur sync auto popular: {e}")

        # Charger les listes au démarrage (SQLite si dispo, sinon txt)
        self.popular_symbols_data = self._load_symbols_preferred("popular_symbols.txt", "popular")
        self.mes_symbols_data = self._load_symbols_preferred("mes_symbols.txt", "personal")
        self.optim_symbols_data = self._load_symbols_preferred("optimisation_symbols.txt", "optimization")
        
        # Tabs-based UI: results-focused navigation
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Analyze (input + quick summary)
        self.analyze_container = QWidget()
        self.layout = QVBoxLayout(self.analyze_container)
        self.tabs.addTab(self.analyze_container, "Analyser")

        # Tab 2: Results (detailed table of analysis results)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.tabs.addTab(self.results_container, "Résultats")

        # Tab 3: Charts (per-symbol graphs and metrics)
        self.charts_container = QWidget()
        self.charts_layout = QVBoxLayout(self.charts_container)
        self.charts_scroll = QScrollArea()
        self.charts_scroll.setWidgetResizable(True)
        self.charts_scroll_widget = QWidget()
        self.charts_scroll_layout = QVBoxLayout(self.charts_scroll_widget)
        self.charts_scroll.setWidget(self.charts_scroll_widget)
        self.charts_layout.addWidget(self.charts_scroll)
        self.tabs.addTab(self.charts_container, "Graphiques")

        # Tab 4: Comparisons (multi-symbol visualizations)
        self.comparisons_container = QWidget()
        self.comparisons_layout = QVBoxLayout(self.comparisons_container)
        self.comparisons_layout.addWidget(QLabel("📈 Comparaisons entre symboles (heatmaps, scatter)"))
        self.tabs.addTab(self.comparisons_container, "Comparaisons")

        # Tab 5: Logs (display stdout/stderr)
        self.logs_container = QWidget()
        self.logs_layout = QVBoxLayout(self.logs_container)
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setStyleSheet("font-family: monospace; font-size: 9pt;")
        self.logs_layout.addWidget(QLabel("📝 Logs du système"))
        self.logs_layout.addWidget(self.logs_text)
        self.tabs.addTab(self.logs_container, "Logs")
        
        # Setup log capture to redirect stdout/stderr to logs_text AND terminal
        try:
            self.log_capture = LogCapture(self.logs_text)
            sys.stdout = self.log_capture
            sys.stderr = self.log_capture
            print("✅ Log capture activé - les messages s'affichent ici et dans le terminal")
        except Exception as e:
            print(f"⚠️ Erreur initialisation LogCapture: {e}")

        # Build analyze tab UI
        self.setup_ui()

        self.current_results = []
        self._analysis_id = 0  # 🔧 Identifiant unique pour chaque analyse
    
    def add_log(self, message: str):
        """Ajouter un message à l'onglet Logs (sans redirection de stdout)."""
        if hasattr(self, 'logs_text'):
            self.logs_text.append(message)

    def _load_symbols_preferred(self, filename: str, list_type: str):
        """Charge depuis SQLite si possible, sinon depuis le fichier txt."""
        symbols = []
        if SYMBOL_MANAGER_AVAILABLE:
            try:
                symbols = get_symbols_by_list_type(list_type, active_only=True)
                print(f"✅ {list_type}: {len(symbols)} symboles chargés depuis SQLite")
            except Exception as e:
                print(f"⚠️ Erreur chargement SQLite pour {list_type}: {e}")
                symbols = []
        if not symbols:
            try:
                symbols = load_symbols_from_txt(filename, use_sqlite=True)
                print(f"✅ {list_type}: {len(symbols)} symboles chargés depuis fichier/SQLite")
            except Exception as e:
                print(f"⚠️ Erreur chargement {filename}: {e}")
                symbols = []
        # Dédupe en conservant l'ordre
        return list(dict.fromkeys([s for s in symbols if s]))
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
        popular_sorted = sorted(self.popular_symbols_data)
        popular_layout = QHBoxLayout()
        popular_listcol = QVBoxLayout()
        self.popular_label = QLabel()
        self.popular_label.setAlignment(Qt.AlignCenter)
        self.popular_label.setWordWrap(True)
        popular_layout.addWidget(self.popular_label)
        self.popular_list = QListWidget()
        self.popular_list.setMaximumHeight(70)
        for s in popular_sorted:
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
        mes_sorted = sorted(self.mes_symbols_data)
        mes_layout = QHBoxLayout()
        mes_listcol = QVBoxLayout()
        self.mes_label = QLabel()
        self.mes_label.setAlignment(Qt.AlignCenter)
        self.mes_label.setWordWrap(True)
        mes_layout.addWidget(self.mes_label)
        self.mes_list = QListWidget()
        self.mes_list.setMaximumHeight(70)
        for s in mes_sorted:
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

        lists_container.addItem(QSpacerItem(48, 20, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum))

        # ========== NOUVELLES LISTES OPTIMISATION ==========
        # Liste 1 : 30 symboles ALÉATOIRES
        random_layout = QHBoxLayout()
        random_listcol = QVBoxLayout()
        self.random_label = QLabel()
        self.random_label.setAlignment(Qt.AlignCenter)
        self.random_label.setWordWrap(True)
        random_layout.addWidget(self.random_label)
        self.random_list = QListWidget()
        self.random_list.setMaximumHeight(70)
        self.random_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        random_listcol.addWidget(self.random_list)
        random_layout.addLayout(random_listcol)
        
        random_btns = QVBoxLayout()
        random_btns.setSpacing(2)
        self.random_refresh_btn = QPushButton("🔄 Nouveau")
        self.random_refresh_btn.clicked.connect(self.refresh_random_symbols)
        self.random_show_btn = QPushButton("Afficher")
        self.random_all_btn = QPushButton("📋 Tout sélect.")
        self.random_all_btn.clicked.connect(lambda: self._select_all_items(self.random_list))
        random_btns.addWidget(self.random_refresh_btn)
        random_btns.addWidget(self.random_show_btn)
        random_btns.addWidget(self.random_all_btn)
        random_layout.addLayout(random_btns)
        
        lists_container.addLayout(random_layout)
        
        lists_container.addItem(QSpacerItem(48, 5, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum))
        
        # Liste 2 : 30 derniers SYMBOLES AJOUTÉS
        recent_layout = QHBoxLayout()
        recent_listcol = QVBoxLayout()
        self.recent_label = QLabel()
        self.recent_label.setAlignment(Qt.AlignCenter)
        self.recent_label.setWordWrap(True)
        recent_layout.addWidget(self.recent_label)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(70)
        self.recent_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        recent_listcol.addWidget(self.recent_list)
        recent_layout.addLayout(recent_listcol)
        
        recent_btns = QVBoxLayout()
        recent_btns.setSpacing(2)
        self.recent_show_btn = QPushButton("Afficher")
        self.recent_all_btn = QPushButton("📋 Tout sélect.")
        self.recent_all_btn.clicked.connect(lambda: self._select_all_items(self.recent_list))
        recent_btns.addWidget(self.recent_show_btn)
        recent_btns.addWidget(self.recent_all_btn)
        recent_layout.addLayout(recent_btns)
        
        lists_container.addLayout(recent_layout)
        
        lists_container.addItem(QSpacerItem(48, 20, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum))
        
        # Bouton OPTIMISATION HYBRIDE (ouvre la fenêtre dédiée)
        optim_button_layout = QVBoxLayout()
        self.optimization_window_btn = QPushButton("🚀 OPTIMISATION HYBRIDE")
        self.optimization_window_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 12px; font-size: 13px;"
        )
        self.optimization_window_btn.clicked.connect(self.open_optimization_window)
        optim_button_layout.addWidget(self.optimization_window_btn)
        lists_container.addLayout(optim_button_layout)
        self.layout.addLayout(lists_container)

        top_controls = QHBoxLayout()

        # Période d'analyse à gauche (menu déroulant)
        top_controls.addWidget(QLabel("Période d'analyse:"))
        self.period_input = QComboBox()
        self.period_input.setMinimumWidth(220)
        period_options = [
            ("3mo",  "3 mois   — ~63 points (journalier)"),
            ("6mo",  "6 mois   — ~126 points (journalier)"),
            ("1y",   "1 an     — ~252 points (journalier)"),
            ("18mo", "18 mois  — ~378 points (journalier)"),
            ("2y",   "2 ans    — ~504 points (journalier)"),
            ("3y",   "3 ans    — ~756 points (journalier)"),
            ("4y",   "4 ans    — ~1 008 points (journalier)"),
            ("5y",   "5 ans    — ~1 260 points (journalier)"),
            ("10y",  "10 ans   — ~2 520 points (journalier)"),
            ("max",  "Max      — historique complet (journalier)"),
        ]
        default_index = 0
        for i, (value, label) in enumerate(period_options):
            self.period_input.addItem(label, userData=value)
            if value == period:
                default_index = i
        self.period_input.setCurrentIndex(default_index)
        top_controls.addWidget(self.period_input)

        top_controls.addSpacing(24)  # Petit espace pour l'esthétique

        # Boutons d'analyse sur la même ligne
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_stock)
        top_controls.addWidget(self.analyze_button)

        self.backtest_button = QPushButton("Analyze and Backtest")
        self.backtest_button.clicked.connect(self.analyse_and_backtest)
        top_controls.addWidget(self.backtest_button)

        # Seuil minimum de fiabilité pour le backtest (à droite du bouton backtest)
        top_controls.addWidget(QLabel("Seuil fiabilité:"))
        self.fiab_threshold_spin = QSpinBox()
        self.fiab_threshold_spin.setMinimum(0)
        self.fiab_threshold_spin.setMaximum(100)
        self.fiab_threshold_spin.setValue(30)
        self.fiab_threshold_spin.setSuffix("%")
        self.fiab_threshold_spin.setMaximumWidth(80)
        self.fiab_threshold_spin.setToolTip("Seuil minimum de fiabilité pour filtrer les résultats du backtest")
        top_controls.addWidget(self.fiab_threshold_spin)

        # Durée minimale de détention (en jours de bourse / barres actives)
        top_controls.addWidget(QLabel("Durée min position:"))
        self.min_hold_days_spin = QSpinBox()
        self.min_hold_days_spin.setMinimum(1)
        self.min_hold_days_spin.setMaximum(60)
        self.min_hold_days_spin.setValue(7)
        self.min_hold_days_spin.setSuffix(" j")
        self.min_hold_days_spin.setMaximumWidth(80)
        self.min_hold_days_spin.setToolTip("Nombre minimum de jours actifs avant d'autoriser une vente en backtest")
        top_controls.addWidget(self.min_hold_days_spin)

        top_controls.addSpacing(24)  # Petit espace pour l'esthétique

        self.popular_signals_button = QPushButton("Analyse de mes symboles")
        self.popular_signals_button.clicked.connect(self.analyze_popular_signals)
        top_controls.addWidget(self.popular_signals_button)

        self.toggle_bottom_btn = QPushButton("Masquer détails")
        self.toggle_bottom_btn.setCheckable(True)
        self.toggle_bottom_btn.clicked.connect(self.toggle_bottom)
        top_controls.addWidget(self.toggle_bottom_btn)
        
        # Bouton pour basculer entre mode online/offline
        self.offline_mode_btn = QPushButton("🌐 Mode: ONLINE")
        self.offline_mode_btn.setCheckable(True)
        self.offline_mode_btn.setChecked(False)
        self.offline_mode_btn.clicked.connect(self.toggle_offline_mode)
        self.offline_mode_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        top_controls.addWidget(self.offline_mode_btn)
        
        # 💾 Bouton pour sauvegarder les graphiques en PDF
        self.save_pdf_btn = QPushButton("💾 Sauvegarder (PDF)")
        self.save_pdf_btn.setToolTip("Sauvegarder tous les graphiques de l'analyse en PDF")
        self.save_pdf_btn.clicked.connect(self.export_results_pdf)
        top_controls.addWidget(self.save_pdf_btn)

        self.layout.addLayout(top_controls)

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

        # Bottom container for Analyze tab summary
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(80)
        self.summary_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_layout.addWidget(self.summary_text)
        
        # Single merged table combining signals + backtest metrics (will be shown in Results tab)
        self.merged_table = QTableWidget()
        self.merged_table.setMinimumHeight(600)
        self.merged_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        merged_columns = [
        'Symbole','Signal','Score','Prix','Tendance','RSI','Volume\nmoyen','Domaine','Cap\nRange','Score/\nSeuil',
        'Fiabilite\n(%)','Nb\nTrades','Gagnants',
        # COLONNES FINANCIÈRES
        'Rev.\nGrowth(%)','EBITDA\nYield(%)','FCF\nYield(%)','D/E\nRatio','Market\nCap(B$)','ROE\n(%)',
        # COLONNES DERIVÉES
        'dPrice','dMACD','dRSI','dVol\nRel',
        # COLONNES BACKTEST
        'Gain\ntotal($)','Gain\nmoyen($)',
        # INFO
        'Consensus', 'Earn\nSurp(%)', 'Ana\nUpgrad'
        ]
        # Add table to Results tab, not Analyze tab
        # 🔧 Boutons d'export dans l'onglet Résultats
        export_buttons_layout = QHBoxLayout()
        export_buttons_layout.addStretch()
        self.export_csv_btn = QPushButton("📥 Exporter (CSV)")
        self.export_csv_btn.setToolTip("Exporter les résultats en fichier CSV")
        self.export_csv_btn.clicked.connect(self.export_results_csv)
        self.export_excel_btn = QPushButton("📊 Exporter (Excel)")
        self.export_excel_btn.setToolTip("Exporter les résultats en fichier Excel")
        self.export_excel_btn.clicked.connect(self.export_results_excel)
        export_buttons_layout.addWidget(self.export_csv_btn)
        export_buttons_layout.addWidget(self.export_excel_btn)
        
        self.results_layout.addWidget(QLabel("📋 Résultats détaillés de l'analyse"))
        self.results_layout.addLayout(export_buttons_layout)
        self.results_layout.addWidget(self.merged_table)

        self.merged_table.setColumnCount(len(merged_columns))
        self.merged_table.setHorizontalHeaderLabels(merged_columns)
        self.merged_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Réduire la hauteur des en-têtes pour 2 lignes maximum
        self.merged_table.horizontalHeader().setMinimumHeight(40)
        # Centrer les en-têtes horizontalement et verticalement
        header_style = """
            QHeaderView::section {
                padding: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
        """
        self.merged_table.horizontalHeader().setStyleSheet(header_style)
        # Allow sorting by clicking headers (we also provide numeric data via Qt.EditRole)
        self.merged_table.setSortingEnabled(True)
        
        # Keep bottom_container for Analyze tab (only summary_text)
        # NOTA: self.merged_table is now in Results tab, not in Analyze tab
        bottom_layout.addWidget(self.summary_text)  # Only summary in Analyze tab
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
        
        # Callbacks pour les nouvelles listes
        self.random_show_btn.clicked.connect(lambda: self.show_selected(self.random_list))
        self.recent_show_btn.clicked.connect(lambda: self.show_selected(self.recent_list))
        
        self._update_list_counts()
        
        # Charger les listes aléatoires et récentes au démarrage
        self.refresh_random_symbols()
        self.load_recent_symbols()
    
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


    def _map_list_type(self, filename: str) -> str:
        lower = filename.lower()
        if 'mes_symbol' in lower:
            return 'personal'
        if 'optimisation' in lower or 'optimization' in lower:
            return 'optimization'
        return 'popular'

    def add_symbol(self, list_widget, filename):
        """Ajoute un ou plusieurs symboles (séparés par des virgules) à la liste.
        Les symboles sont validés, ajoutés individuellement, et la liste est 
        triée alphabétiquement. Si c'est mes_symbols, ils sont aussi ajoutés 
        automatiquement aux symboles populaires.
        """
        text, ok = QInputDialog.getText(
            self, 
            "Ajouter symbole(s)", 
            "Symbole(s) (ex: AAPL ou AAPL, MSFT, GOOGL):"
        )
        
        if ok and text:
            # Identifier si c'est la liste mes_symbols
            is_mes_list = (list_widget == self.mes_list)
            main_list = list_widget
            secondary_list = self.popular_list if is_mes_list else None
            
            # Parser les symboles séparés par des virgules
            symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
            
            if not symbols:
                return
            
            # Ajouter chaque symbole individuellement
            added_symbols = []
            
            for symbol in symbols:
                if not symbol:
                    continue
                
                # Vérifier que le symbole n'existe pas déjà
                exists_main = any(
                    main_list.item(i).text() == symbol 
                    for i in range(main_list.count())
                )
                
                if exists_main:
                    QMessageBox.information(
                        self, 
                        "Info", 
                        f"{symbol} existe déjà dans la liste principale"
                    )
                    continue
                
                # Validation du ticker
                progress = QProgressDialog(
                    f"Validation de {symbol}...", 
                    None, 0, 0, self
                )
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
                        f"Le symbole '{symbol}' n'est pas valide.\\n"
                        "Vérifiez l'orthographe ou consultez Yahoo Finance."
                    )
                    continue
                
                # Ajouter à la liste principale
                item = QListWidgetItem(symbol)
                item.setData(Qt.UserRole, symbol)
                main_list.addItem(item)
                added_symbols.append(symbol)
                
                # Si c'est mes_symbols, ajouter aussi automatiquement aux populaires
                if is_mes_list and secondary_list:
                    exists_secondary = any(
                        secondary_list.item(i).text() == symbol 
                        for i in range(secondary_list.count())
                    )
                    
                    if not exists_secondary:
                        item_pop = QListWidgetItem(symbol)
                        item_pop.setData(Qt.UserRole, symbol)
                        secondary_list.addItem(item_pop)
            
            # Trier les deux listes alphabétiquement
            self._sort_list_alphabetically(main_list)
            if secondary_list:
                self._sort_list_alphabetically(secondary_list)
            
            # Sauvegarder les listes triées
            try:
                from qsi import save_symbols_to_txt
                
                symbols_main = [
                    main_list.item(i).data(Qt.UserRole) 
                    if main_list.item(i).data(Qt.UserRole) is not None 
                    else main_list.item(i).text() 
                    for i in range(main_list.count())
                ]
                save_symbols_to_txt(symbols_main, filename)
                
                # 🔧 Synchroniser avec SQLite après sauvegarde txt
                if SYMBOL_MANAGER_AVAILABLE:
                    try:
                        from symbol_manager import sync_txt_to_sqlite
                        list_type = self._map_list_type(filename)
                        sync_txt_to_sqlite(filename, list_type=list_type)
                        print(f"✅ SQLite synchronisé pour {filename}")
                    except Exception as e:
                        print(f"⚠️ Erreur lors de la sync SQLite: {e}")
                
                if secondary_list:
                    filename_secondary = "popular_symbols.txt"
                    symbols_secondary = [
                        secondary_list.item(i).data(Qt.UserRole) 
                        if secondary_list.item(i).data(Qt.UserRole) is not None 
                        else secondary_list.item(i).text() 
                        for i in range(secondary_list.count())
                    ]
                    save_symbols_to_txt(symbols_secondary, filename_secondary)
                    
                    # 🔧 Synchroniser la liste secondaire avec SQLite
                    if SYMBOL_MANAGER_AVAILABLE:
                        try:
                            from symbol_manager import sync_txt_to_sqlite
                            sync_txt_to_sqlite(filename_secondary, list_type='popular')
                            print(f"✅ SQLite synchronisé pour {filename_secondary}")
                        except Exception as e:
                            print(f"⚠️ Erreur lors de la sync SQLite: {e}")

                # Rafraîchir les compteurs après ajouts/sauvegardes
                self._update_list_counts()
            
            except Exception:
                pass

    def _sort_list_alphabetically(self, list_widget):
        """Trie les éléments d'une QListWidget alphabétiquement."""
        items = []
        
        # Récupérer tous les éléments
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            text = item.text()
            data = item.data(Qt.UserRole)
            items.append((text, data))
        
        # Trier alphabétiquement
        items.sort(key=lambda x: x)
        
        # Vider la liste
        list_widget.clear()
        
        # Réajouter les éléments triés
        for text, data in items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, data)
            list_widget.addItem(item)

        # Mettre à jour les compteurs après réinjection
        self._update_list_counts()

    def _update_list_counts(self):
        """Met à jour les libellés avec le nombre d'éléments de chaque liste."""
        try:
            pop_count = self.popular_list.count() if hasattr(self, "popular_list") else 0
            mes_count = self.mes_list.count() if hasattr(self, "mes_list") else 0
            optim_count = self.optim_list.count() if hasattr(self, "optim_list") else 0
            if hasattr(self, "popular_label"):
                self.popular_label.setText(f"Symboles\npopulaires ({pop_count})")
            if hasattr(self, "mes_label"):
                self.mes_label.setText(f"Mes\nsymboles ({mes_count})")
            if hasattr(self, "optim_label"):
                self.optim_label.setText(f"Symboles\noptimisation ({optim_count})")
        except Exception:
            pass


    def remove_selected(self, list_widget, filename):
        items = list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Veuillez sélectionner au moins un symbole à supprimer")
            return
        for it in items:
            list_widget.takeItem(list_widget.row(it))
        self._update_list_counts()
        try:
            from qsi import save_symbols_to_txt
            symbols = [list_widget.item(i).data(Qt.UserRole) if list_widget.item(i).data(Qt.UserRole) is not None else list_widget.item(i).text() for i in range(list_widget.count())]
            save_symbols_to_txt(symbols, filename)
            
            # 🔧 Synchroniser avec SQLite après suppression
            if SYMBOL_MANAGER_AVAILABLE:
                try:
                    from symbol_manager import sync_txt_to_sqlite
                    # Déterminer le type de liste pour SQLite
                    list_type = self._map_list_type(filename)
                    sync_txt_to_sqlite(filename, list_type=list_type)
                    print(f"✅ SQLite synchronisé (suppression) pour {filename}")
                except Exception as e:
                    print(f"⚠️ Erreur lors de la sync SQLite: {e}")
        except Exception:
            pass
        self._update_list_counts()

    def show_selected(self, list_widget):
        items = list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Veuillez sélectionner au moins un symbole à afficher")
            return
        symbols = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in items]
        self.symbol_input.setText(", ".join(symbols))

    def preview_cleaned_optimization(self):
        """Affiche un aperçu des groupes nettoyés (sector × cap) en priorisant les symboles ajoutés manuellement."""
        if not SYMBOL_MANAGER_AVAILABLE:
            QMessageBox.warning(self, "SQLite requis", "Le nettoyage nécessite SQLite/symbol_manager.")
            return
        try:
            from symbol_manager import (
                get_all_sectors,
                get_all_cap_ranges,
                get_symbols_by_sector_and_cap,
            )
            from optimisateur_hybride import clean_sector_cap_groups

            list_type = "optimization"
            sectors = get_all_sectors(list_type=list_type)
            caps = get_all_cap_ranges(list_type=list_type)

            sector_cap_ranges = {}
            for sec in sectors:
                buckets = {}
                for cap in caps:
                    syms = get_symbols_by_sector_and_cap(sec, cap, list_type=list_type, active_only=True)
                    if syms:
                        buckets[cap] = syms
                if buckets:
                    sector_cap_ranges[sec] = buckets

            if not sector_cap_ranges:
                QMessageBox.information(self, "Aperçu nettoyage", "Aucune donnée d'optimisation trouvée.")
                return

            # Nouvelle logique : FIXE (mes_symbols) + ALÉATOIRE (popular), min=6, max=15
            cleaned = clean_sector_cap_groups(sector_cap_ranges, ttl_days=0, min_symbols=6, max_symbols=15, fixed_ratio=0.18)

            # Prioriser les symboles ajoutés manuellement : ils restent en tête et ne sont pas élagués en premier
            manual_order = [self.optim_list.item(i).text() for i in range(self.optim_list.count())]
            manual_set = set(manual_order)
            for sec, buckets in cleaned.items():
                for cap, syms in buckets.items():
                    manual_first = [s for s in manual_order if s in syms]
                    rest = [s for s in syms if s not in manual_set]
                    cleaned[sec][cap] = manual_first + rest

            # Aplatir en liste unique (ordre: secteurs triés, cap triés, avec priorité manuelle déjà appliquée)
            seen = set()
            flat_cleaned = []
            for sec in sorted(cleaned.keys()):
                for cap in sorted(cleaned[sec].keys()):
                    for s in cleaned[sec][cap]:
                        if s not in seen:
                            seen.add(s)
                            flat_cleaned.append(s)
            
            # Garantir minimum 300 symboles au total
            MIN_TOTAL_SYMBOLS = 300
            if len(flat_cleaned) < MIN_TOTAL_SYMBOLS:
                needed = MIN_TOTAL_SYMBOLS - len(flat_cleaned)
                try:
                    from symbol_manager import get_all_popular_symbols
                    import random
                    all_popular = get_all_popular_symbols(max_count=1000, exclude_symbols=seen)
                    random.shuffle(all_popular)  # Randomiser pour diversité
                    additional = all_popular[:needed]
                    flat_cleaned.extend(additional)
                    print(f"   📊 Complément pour atteindre {MIN_TOTAL_SYMBOLS} symboles : +{len(additional)} depuis popular_symbols")
                except Exception as e:
                    print(f"   ⚠️ Impossible d'ajouter des symboles supplémentaires : {e}")

            # Mettre à jour la QList optimisation avec la version nettoyée
            self.optim_list.clear()
            for s in flat_cleaned:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, s)
                self.optim_list.addItem(item)

            # Sauvegarder dans le fichier + SQLite
            try:
                from qsi import save_symbols_to_txt
                save_symbols_to_txt(flat_cleaned, "optimisation_symbols.txt")
                if SYMBOL_MANAGER_AVAILABLE:
                    from symbol_manager import sync_txt_to_sqlite
                    sync_txt_to_sqlite("optimisation_symbols.txt", list_type="optimization")
            except Exception as e:
                QMessageBox.warning(self, "Avertissement", f"Nettoyage appliqué mais sauvegarde non confirmée: {e}")

            # Rafraîchir compteurs
            self._update_list_counts()

            lines = [f"=== Résumé des groupes nettoyés (optimisation) ==="]
            lines.append(f"TOTAL : {len(flat_cleaned)} symboles (minimum garanti : 300)")
            lines.append("")
            for sec in sorted(cleaned.keys()):
                for cap in sorted(cleaned[sec].keys()):
                    syms = cleaned[sec][cap]
                    preview = ", ".join(syms[:18]) + (" …" if len(syms) > 18 else "")
                    lines.append(f"{sec} × {cap}: {len(syms)} -> {preview}")

            QMessageBox.information(self, "Nettoyage appliqué", "\n".join(lines))
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'afficher l'aperçu du nettoyage: {e}")

    def analyze_popular_signals(self):
        # Analyse + Backtest uniquement sur mes symboles
        selected_mes = [it.data(Qt.UserRole) if it.data(Qt.UserRole) is not None else it.text() for it in self.mes_list.selectedItems()]
        mes_symbols = selected_mes if selected_mes else [self.mes_list.item(i).text() for i in range(self.mes_list.count())]
        
        period = self.period_input.currentData()
        
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une période d'analyse")
            return
        
        if not mes_symbols:
            QMessageBox.warning(self, "Erreur", "Aucun symbole disponible dans 'Mes symboles'")
            return
        
        # 🔧 Incrémenter l'ID d'analyse et stopper les threads précédents
        self._analysis_id += 1
        current_id = self._analysis_id
        print(f"\n🚀 Nouvelle analyse populaire lancée #ID={current_id}")
        
        # 🔧 Réinitialiser les données complètement
        self.current_results = []
        if hasattr(self, 'filtered_results'):
            delattr(self, 'filtered_results')
        if hasattr(self, 'backtest_map'):
            delattr(self, 'backtest_map')
        
        # Stop any running analysis thread before starting a new one
        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                # Disconnect signals to prevent old thread from updating UI
                self.analysis_thread.finished.disconnect()
                self.analysis_thread.error.disconnect()
                self.analysis_thread.progress.disconnect()
            except Exception:
                pass
            
            # Stop the thread
            if self.analysis_thread.isRunning():
                self.analysis_thread.stop()
                self.analysis_thread.wait(2000)  # Wait max 2 seconds
                print(f"⚠️ Ancien analysis thread arrêté")
        
        # Disable buttons during analysis
        self.analyze_button.setEnabled(False)
        self.backtest_button.setEnabled(False)
        self.popular_signals_button.setEnabled(False)
        
        # Afficher la progression
        self.progress = QProgressDialog("Analyse + Backtest de mes symboles en cours...", "Annuler", 0, 0, self)
        self.progress.setWindowTitle("Analyse de mes symboles")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setMinimumWidth(400)
        
        # Lancer l'analyse+backtest dans un thread (mes_symbols en premier arg, [] en deuxième)
        min_holding_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
        self.analysis_thread = AnalysisThread(mes_symbols, [], period, analysis_id=current_id, min_holding_days=min_holding_days)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.start()
        print(f"📊 Analysis thread démarré avec ID={current_id}")

    def clear_plots(self):
        # remove all widgets from plots_layout
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

    def on_download_complete(self, result):
        # 🔧 Vérifier que ce résultat appartient à l'analyse actuelle
        received_id = result.get('_analysis_id', 0) if isinstance(result, dict) else 0
        if received_id != self._analysis_id:
            print(f"⚠️ Résultat ignoré: ID={received_id}, ID actuel={self._analysis_id}")
            return
        
        # Called when the DownloadThread finishes
        # Re-enable buttons
        self.analyze_button.setEnabled(True)
        self.backtest_button.setEnabled(True)
        self.popular_signals_button.setEnabled(True)

        if self.progress:
            self.progress.close()
        
        # 🔧 Supprimer filtered_results si existant pour forcer l'utilisation de current_results
        if hasattr(self, 'filtered_results'):
            delattr(self, 'filtered_results')

        data = result.get('data', {}) if isinstance(result, dict) else {}
        backtests = result.get('backtest_results', []) if isinstance(result, dict) else []
        # Build result rows (collect data first, then filter & render plots only for filtered symbols)
        self.current_results = []

        for symbol, stock_data in data.items():
            try:
                prices = stock_data['Close']
                volumes = stock_data['Volume']

                # ✅ Récupération du secteur AVANT l'analyse (cohérence avec backtest)
                sig = "NEUTRE"
                last_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
                trend = False
                last_rsi = 0.0
                volume_mean = float(volumes.mean()) if len(volumes) > 0 else 0.0
                score = 0.0
                derivatives = {}
                cap_range = qsi.get_cap_range_for_symbol(symbol)
                
                # Récupérer le secteur depuis le cache ou yfinance
                try:
                    if qsi.OFFLINE_MODE:
                        if qsi.get_pickle_cache is not None:
                            fin_cache = qsi.get_pickle_cache(symbol, 'financial', ttl_hours=24*365)
                            domaine = fin_cache.get('sector', 'Inconnu') if fin_cache else "Inconnu"
                        else:
                            domaine = "Inconnu"
                    else:
                        info = yf.Ticker(symbol).info
                        domaine = info.get("sector", "Inconnu")
                    
                    # ✅ NEW: Normaliser le secteur pour cohérence avec la DB
                    from sector_normalizer import normalize_sector
                    domaine_raw = domaine
                    domaine = normalize_sector(domaine)
                    if domaine_raw != domaine:
                        print(f"🔄 {symbol}: Secteur normalisé: '{domaine_raw}' -> '{domaine}'")
                    else:
                        print(f"🔍 DEBUG {symbol}: secteur = {domaine}")
                except Exception as e:
                    domaine = "Inconnu"
                    print(f"⚠️ DEBUG {symbol}: erreur récupération secteur: {e}")
                
                # ✅ NEW: Améliorer le fallback cap_range en 2 étapes
                from config import CAP_FALLBACK_ENABLED
                original_cap_range = cap_range
                
                if CAP_FALLBACK_ENABLED and (cap_range == "Unknown" or not cap_range):
                    best_params_all = qsi.extract_best_parameters()
                    
                    # ✅ ÉTAPE 1: Essayer de trouver dans la DB les cap_ranges valides pour ce secteur
                    print(f"🔍 {symbol}: Recherche cap_range pour {domaine}...")
                    try:
                        import sqlite3
                        db_path = 'symbols.db'
                        if os.path.exists(db_path):
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT DISTINCT cap_range FROM symbols 
                                WHERE sector = ? AND cap_range IS NOT NULL AND cap_range != 'Unknown'
                                LIMIT 10
                            """, (domaine,))
                            db_caps = [row[0] for row in cursor.fetchall()]
                            conn.close()
                            
                            # Prioriser l'ordre logique: Small, Mid, Large, Mega
                            cap_priority = ['Small', 'Mid', 'Large', 'Mega']
                            for cap in cap_priority:
                                if cap in db_caps:
                                    test_key = f"{domaine}_{cap}"
                                    if test_key in best_params_all:
                                        cap_range = cap
                                        print(f"✅ {symbol}: Cap_range trouvé en DB: {cap}")
                                        break
                    except Exception as e:
                        print(f"⚠️ {symbol}: Erreur recherche DB cap_range: {e}")
                    
                    # ✅ ÉTAPE 2: Si toujours Unknown, essayer les fallbacks standards
                    if cap_range == "Unknown" or not cap_range:
                        for fallback_cap in ["Large", "Mid", "Small", "Mega"]:
                            test_key = f"{domaine}_{fallback_cap}"
                            if test_key in best_params_all:
                                cap_range = fallback_cap
                                print(f"✅ {symbol}: Cap_range fallback: {fallback_cap}")
                                break
                    
                    if cap_range != original_cap_range:
                        print(f"🔄 {symbol}: Cap_range ajusté: '{original_cap_range}' -> '{cap_range}'")
                
                # ✅ Appliquer fallback universel pour "Inconnu" (même logique que backtest) - configurable
                original_domaine = domaine
                from config import DOMAIN_FALLBACK_ENABLED
                if DOMAIN_FALLBACK_ENABLED and domaine == "Inconnu":
                    best_params_all = qsi.extract_best_parameters()
                    for fallback_sector in ["Technology", "Healthcare", "Financial Services"]:
                        if fallback_sector in best_params_all:
                            domaine = fallback_sector
                            break
                    if domaine == "Inconnu" and best_params_all:
                        first_key = list(best_params_all.keys())[0]
                        domaine = first_key.split('_')[0] if '_' in first_key else first_key
                    print(f"🔄 DEBUG {symbol}: fallback appliqué {original_domaine} -> {domaine}")
                
                # ✅ Extraire les seuils globaux optimisés
                seuil_achat_opt = None
                seuil_vente_opt = None
                best_params_all = qsi.extract_best_parameters()
                # Chercher la clé optimale : secteur_cap ou secteur seul
                param_key = None
                if cap_range and cap_range != "Unknown":
                    test_key = f"{domaine}_{cap_range}"
                    if test_key in best_params_all:
                        param_key = test_key
                if not param_key and domaine in best_params_all:
                    param_key = domaine
                
                if param_key and param_key in best_params_all:
                    params = best_params_all[param_key]
                    if len(params) > 2 and params[2]:
                        globals_th = params[2]
                        if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                            seuil_achat_opt = float(globals_th[0])
                            seuil_vente_opt = float(globals_th[1])
                
                try:
                    # Un seul appel avec le bon domaine (ou fallback) et seuils globaux optimisés
                    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
                        prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol, cap_range=cap_range,
                        seuil_achat=seuil_achat_opt, seuil_vente=seuil_vente_opt
                    )
                    
                    # 🔍 Debug: vérifier si les métriques financières sont présentes
                    if not derivatives.get('rev_growth_val') and not derivatives.get('market_cap_val'):
                        print(f"⚠️ {symbol}: Métriques financières manquantes dans derivatives")
                        print(f"   Clés disponibles: {list(derivatives.keys())}")
                except Exception as e:
                    # Log l'erreur mais continue avec les valeurs par défaut
                    print(f"⚠️ Erreur get_trading_signal pour {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
                    derivatives = {}

                # Timeline metrics (earnings surprise and analyst upgrades)
                timeline_surprise = 'N/A'
                timeline_upgrades = 0
                timeline_err = None
                try:
                    from timeline_cache import TimelineCache
                    if not hasattr(self, '_timeline_cache'):
                        timeline_db_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..', 'stock_analysis.db')
                        )
                        self._timeline_cache = TimelineCache(db_path=timeline_db_path)

                    last_dt = prices.index[-1]
                    if hasattr(last_dt, 'strftime'):
                        target_date = last_dt.strftime('%Y-%m-%d')
                    else:
                        target_date = str(last_dt)[:10]

                    # Refresh timeline data before reading strict PIT metrics.
                    try:
                        self._timeline_cache.update_timeline_data(symbol)
                    except Exception as refresh_err:
                        print(f"⚠️ Timeline refresh failed for {symbol}: {refresh_err}")

                    pit_timeline = self._timeline_cache.get_pit_timeline_data(symbol, target_date)
                    if isinstance(pit_timeline, dict):
                        surprise_val = pit_timeline.get('latest_earnings_surprise', None)
                        if surprise_val is not None:
                            timeline_surprise = round(float(surprise_val), 2)
                        timeline_upgrades = int(pit_timeline.get('recent_upgrades_count', 0) or 0)
                except Exception as e:
                    timeline_err = e

                if timeline_err is not None:
                    print(f"⚠️ Timeline enrich failed for {symbol}: {timeline_err}")
                    # Fallback PIT read using today's date, without forcing refresh.
                    try:
                        from timeline_cache import TimelineCache
                        if not hasattr(self, '_timeline_cache'):
                            timeline_db_path = os.path.abspath(
                                os.path.join(os.path.dirname(__file__), '..', 'stock_analysis.db')
                            )
                            self._timeline_cache = TimelineCache(db_path=timeline_db_path)
                        fallback_date = datetime.now().strftime('%Y-%m-%d')
                        pit_timeline = self._timeline_cache.get_pit_timeline_data(symbol, fallback_date)
                        if isinstance(pit_timeline, dict):
                            surprise_val = pit_timeline.get('latest_earnings_surprise', None)
                            if surprise_val is not None:
                                timeline_surprise = round(float(surprise_val), 2)
                            timeline_upgrades = int(pit_timeline.get('recent_upgrades_count', 0) or 0)
                    except Exception as fallback_err:
                        print(f"⚠️ Timeline fallback failed for {symbol}: {fallback_err}")

                if not hasattr(self, '_timeline_debug_count'):
                    self._timeline_debug_count = 0
                if self._timeline_debug_count < 12:
                    print(
                        f"🧪 TIMELINE {symbol}: surprise={timeline_surprise} upgrades={timeline_upgrades}"
                    )
                    self._timeline_debug_count += 1

                row_info = {
                    'Symbole': symbol,
                    'Signal': sig,
                    'Score': score,
                    'Prix': last_price,
                    'Tendance': 'Hausse' if trend else 'Baisse',
                    'RSI': last_rsi,
                    'DomaineOriginal': original_domaine,
                    'Domaine': domaine,
                    'CapRange': cap_range,
                    'Volume moyen': volume_mean,
                    # Consensus (stable via cache/offline fallback)
                    'Consensus': (qsi.get_consensus(symbol) or {}).get('label', 'Neutre'),
                    'ConsensusMean': (qsi.get_consensus(symbol) or {}).get('mean', None),
                    'dPrice': round((derivatives.get('price_slope_rel') or 0.0) * 100, 2),
                    'dMACD': round((derivatives.get('macd_slope_rel') or 0.0) * 100, 2),
                    'dRSI': round((derivatives.get('rsi_slope_rel') or 0.0) * 100, 2),
                    'dVolRel': round((derivatives.get('volume_slope_rel') or 0.0) * 100, 2),
                    # ✅ Métriques financières simples - protection contre None
                    'Rev. Growth (%)': round(float((derivatives.get('rev_growth_val') or 0.0)), 2),
                    'EBITDA Yield (%)': round(float((derivatives.get('ebitda_yield_pct') or 0.0)), 2),
                    'FCF Yield (%)': round(float((derivatives.get('fcf_yield_pct') or 0.0)), 2),
                    'D/E Ratio': round(float((derivatives.get('debt_to_equity') or 0.0)), 2),
                    'Market Cap (B$)': round(float((derivatives.get('market_cap_val') or 0.0)), 2),
                    'ROE (%)': round(float((derivatives.get('roe_val') or 0.0)), 2),
                    'Earn Surp (%)': timeline_surprise,
                    'Ana Upgrad': timeline_upgrades
                }

                self.current_results.append(row_info)
            except Exception as e:
                # ✅ Ne jamais ignorer silencieusement - ajouter au moins les données de base
                print(f"❌ Erreur critique pour {symbol}: {e}")
                try:
                    row_info = {
                        'Symbole': symbol,
                        'Signal': 'ERREUR',
                        'Score': 0.0,
                        'Prix': float(stock_data['Close'].iloc[-1]) if 'Close' in stock_data else 0.0,
                        'Tendance': 'N/A',
                        'RSI': 0.0,
                        'Domaine': 'Inconnu',
                        'CapRange': qsi.get_cap_range_for_symbol(symbol),
                        'Volume moyen': 0.0,
                        'dPrice': 0.0,
                        'dMACD': 0.0,
                        'dRSI': 0.0,
                        'dVolRel': 0.0,
                        'Rev. Growth (%)': 0.0,
                        'EBITDA Yield (%)': 0.0,
                        'FCF Yield (%)': 0.0,
                        'D/E Ratio': 0.0,
                        'Market Cap (B$)': 0.0,
                        'ROE (%)': 0.0,
                        'Earn Surp (%)': 'N/A',
                        'Ana Upgrad': 0
                    }
                    self.current_results.append(row_info)
                except Exception:
                    pass

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

            # Filtre sur nb trades - afficher TOUS les stocks inclus ceux avec 0 trades
            try:
                # Toujours inclure les résultats, même avec 0 trades
                # On affiche simplement l'indicateur 0 trades sans filtrer
                if int(nb_trades) > 0 and int(nb_trades) < min_trades:
                    # Si il y a des trades mais moins que le minimum, filtrer
                    if not include_none_val:
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

                row = next((r for r in filtered if r.get('Symbole') == sym), {})
                score_val = row.get('Score')
                precomp = {
                    'signal': row.get('Signal'),
                    'last_price': row.get('Prix'),
                    'trend': row.get('Tendance'),
                    'last_rsi': row.get('RSI'),
                    'volume_moyen': row.get('Volume moyen'),
                    'score': score_val,
                    'domaine': row.get('Domaine'),
                    'cap_range': row.get('CapRange'),
                }

                fig = self._build_symbol_figure_with_score(sym, prices, volumes, precomp=precomp, events=[])

                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(340)
                self.plots_layout.addWidget(canvas)
        except Exception:
            # Fallback: external plotting (analyse_et_affiche shows plots in separate windows)
            try:
                if filtered_symbols:
                    analyse_et_affiche(filtered_symbols, period=self.period_input.currentData() or '12mo')
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
        # 🔧 Vérifier que ce résultat appartient à l'analyse actuelle
        received_id = result.get('_analysis_id', 0) if isinstance(result, dict) else 0
        if received_id != self._analysis_id:
            print(f"⚠️ Résultat ignoré: ID={received_id}, ID actuel={self._analysis_id}")
            return
        
        # Re-enable all buttons
        self.analyze_button.setEnabled(True)
        self.backtest_button.setEnabled(True)
        self.popular_signals_button.setEnabled(True)
        
        if self.progress:
            self.progress.close()
        
        # 🔧 Supprimer filtered_results si existant pour forcer l'utilisation de current_results
        if hasattr(self, 'filtered_results'):
            delattr(self, 'filtered_results')
        
        # Stocker les résultats
        self.current_results = result.get('signals', [])
        
        # 🔧 Stocker les résultats du backtest dans une map pour accès rapide
        backtest_results = result.get('backtest_results', []) if isinstance(result, dict) else []
        self.backtest_map = {b['Symbole']: b for b in backtest_results} if backtest_results else {}
        
        # 🔧 Ajouter les données de backtest aux signaux
        for signal in self.current_results:
            sym = signal.get('Symbole')
            if sym in self.backtest_map:
                bt = self.backtest_map[sym]
                signal['Fiabilite'] = bt.get('taux_reussite', 'N/A')
                signal['NbTrades'] = bt.get('trades', 0)
                signal['Gagnants'] = bt.get('gagnants', 0)
                signal['Gain_total'] = bt.get('gain_total', 0.0)
                signal['Gain_moyen'] = bt.get('gain_moyen', 0.0)
                signal['Drawdown_max'] = bt.get('drawdown_max', 0.0)
            else:
                signal.setdefault('Fiabilite', 'N/A')
                signal.setdefault('NbTrades', 0)
                signal.setdefault('Gagnants', 0)
                signal.setdefault('Gain_total', 0.0)
                signal.setdefault('Gain_moyen', 0.0)
                signal.setdefault('Drawdown_max', 0.0)
        
        # 🔧 Initialiser les colonnes par défaut pour tous les signaux
        for r in self.current_results:
            r.setdefault('CapRange', qsi.get_cap_range_for_symbol(r.get('Symbole', '')))
            r.setdefault('dPrice', 0.0)
            r.setdefault('dMACD', 0.0)
            r.setdefault('dRSI', 0.0)
            r.setdefault('dVolRel', 0.0)
            r.setdefault('Rev. Growth (%)', 0.0)
            r.setdefault('EBITDA Yield (%)', 0.0)
            r.setdefault('FCF Yield (%)', 0.0)
            r.setdefault('D/E Ratio', 0.0)
            r.setdefault('Market Cap (B$)', 0.0)
            r.setdefault('ROE (%)', 0.0)
            r.setdefault('Earn Surp (%)', 'N/A')
            r.setdefault('Ana Upgrad', 0)
        
        # 🔧 Charger les meilleurs paramètres une seule fois
        try:
            from qsi import extract_best_parameters
            self.best_parameters = extract_best_parameters()
        except Exception:
            self.best_parameters = {}
        
        # ✅ OPTIMISATION: Réutiliser les données déjà téléchargées depuis result
        try:
            # Récupérer les données déjà disponibles
            existing_data = result.get('data', {}) if isinstance(result, dict) else {}
            
            for r in self.current_results:
                sym = r.get('Symbole')
                if not sym:
                    continue
                    
                # Calculer les dérivées techniques et métriques financières si manquantes
                need_derivatives = not r.get('dPrice') or float(r.get('dPrice', 0)) == 0.0
                need_financials = not r.get('Market Cap (B$)') or float(r.get('Market Cap (B$)', 0)) == 0.0
                need_timeline = (r.get('Earn Surp (%)', 'N/A') in ('N/A', '', None)) or int(r.get('Ana Upgrad', 0) or 0) == 0

                # Timeline enrichment does not require market candles; fetch it first.
                if need_timeline:
                    try:
                        from timeline_cache import TimelineCache
                        if not hasattr(self, '_timeline_cache'):
                            timeline_db_path = os.path.abspath(
                                os.path.join(os.path.dirname(__file__), '..', 'stock_analysis.db')
                            )
                            self._timeline_cache = TimelineCache(db_path=timeline_db_path)
                        try:
                            self._timeline_cache.update_timeline_data(sym)
                        except Exception as refresh_err:
                            print(f"⚠️ Timeline refresh failed for {sym}: {refresh_err}")

                        # Use today's date for PIT when market data is unavailable.
                        target_date = datetime.now().strftime('%Y-%m-%d')
                        pit_timeline = self._timeline_cache.get_pit_timeline_data(sym, target_date)
                        if isinstance(pit_timeline, dict):
                            surprise_val = pit_timeline.get('latest_earnings_surprise', None)
                            if surprise_val is None:
                                r['Earn Surp (%)'] = 'N/A'
                            else:
                                r['Earn Surp (%)'] = round(float(surprise_val), 2)
                            r['Ana Upgrad'] = int(pit_timeline.get('recent_upgrades_count', 0) or 0)
                            if not hasattr(self, '_timeline_debug_count'):
                                self._timeline_debug_count = 0
                            if self._timeline_debug_count < 12:
                                print(
                                    f"🧪 TIMELINE {sym}: surprise={r.get('Earn Surp (%)')} upgrades={r.get('Ana Upgrad')} target={target_date}"
                                )
                                self._timeline_debug_count += 1
                    except Exception as e:
                        print(f"⚠️ Timeline enrich failed for {sym}: {e}")
                        r.setdefault('Earn Surp (%)', 'N/A')
                        r.setdefault('Ana Upgrad', 0)
                
                if need_derivatives or need_financials:
                    try:
                        # Réutiliser les données en mémoire au lieu de re-télécharger
                        stock_data = existing_data.get(sym)
                        if stock_data is None:
                            # Seulement télécharger si vraiment absent
                            stock_data = download_stock_data([sym], self.period_input.currentData() or '12mo').get(sym)
                        
                        if stock_data is None:
                            continue
                            
                        prices = stock_data['Close']
                        volumes = stock_data['Volume']
                        try:
                            _sig, _last_price, _trend, _last_rsi, _vol_mean, _score, derivatives = get_trading_signal(
                                prices, volumes,
                                domaine=r.get('Domaine', 'Inconnu'),
                                return_derivatives=True,
                                symbol=sym,
                                cap_range=r.get('CapRange')
                            )
                        except Exception:
                            derivatives = {}

                        # Dérivées techniques (relatives en %)
                        if need_derivatives:
                            r['dPrice'] = round(derivatives.get('price_slope_rel', 0.0) * 100, 2)
                            r['dMACD'] = round(derivatives.get('macd_slope_rel', 0.0) * 100, 2)
                            r['dRSI'] = round(derivatives.get('rsi_slope_rel', 0.0) * 100, 2)
                            r['dVolRel'] = round(derivatives.get('volume_slope_rel', 0.0) * 100, 2)
                        
                        # ✅ Métriques financières simples
                        r['Rev. Growth (%)'] = round(derivatives.get('rev_growth_val', 0.0), 2)
                        r['EBITDA Yield (%)'] = round(derivatives.get('ebitda_yield_pct', 0.0), 2)
                        r['FCF Yield (%)'] = round(derivatives.get('fcf_yield_pct', 0.0), 2)
                        r['D/E Ratio'] = round(derivatives.get('debt_to_equity', 0.0), 2)
                        r['Market Cap (B$)'] = round(derivatives.get('market_cap_val', 0.0), 2)
                    except Exception:
                        # leave defaults
                        if need_derivatives:
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

            # 🔧 Map des événements issus du backtest (même source que les stats)
            backtests = result.get('backtest_results', []) if isinstance(result, dict) else []
            events_map = {bt.get('Symbole'): bt.get('events', []) for bt in backtests}

            # ✅ OPTIMISATION: Récupérer les données existantes
            existing_data = result.get('data', {}) if isinstance(result, dict) else {}
            
            # Helper to embed a list of symbols as canvases
            def embed_symbol_list(symbol_list, title_prefix=""):
                if not symbol_list:
                    return
                for i, s in enumerate(symbol_list):
                    sym = s['Symbole'] if isinstance(s, dict) and 'Symbole' in s else s
                    try:
                        # Réutiliser les données en mémoire
                        stock_data = existing_data.get(sym)
                        if stock_data is None:
                            # Seulement télécharger si vraiment absent
                            stock_data = download_stock_data([sym], period=self.period_input.currentData() or '12mo').get(sym)
                        if not stock_data:
                            continue
                        prices = stock_data['Close']
                        volumes = stock_data['Volume']

                        # Prélever les données pré-calculées (table ou item courant)
                        pre_row = next((r for r in self.current_results if r.get('Symbole') == sym), s if isinstance(s, dict) else {})
                        score_val = pre_row.get('Score')
                        precomp = {
                            'signal': pre_row.get('Signal'),
                            'last_price': pre_row.get('Prix'),
                            'trend': pre_row.get('Tendance'),
                            'last_rsi': pre_row.get('RSI'),
                            'volume_moyen': pre_row.get('Volume moyen'),
                            'score': score_val,
                            'domaine': pre_row.get('Domaine'),
                            'cap_range': pre_row.get('CapRange'),
                        }

                        # Add trade markers based on events déjà calculés par le backtest
                        events = events_map.get(sym, [])
                        if len(events) == 0:
                            print(f"⚠️ {sym}: Aucun événement généré")
                        else:
                            print(f"✅ {sym}: {len(events)} événement(s) trouvé(s)")

                        fig = self._build_symbol_figure_with_score(sym, prices, volumes, precomp=precomp, events=events)

                        canvas = FigureCanvas(fig)
                        canvas.setMinimumHeight(340)
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
        period = self.period_input.currentData()
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une période d'analyse")
            return

        # 🔧 Incrémenter l'ID d'analyse et stopper les threads précédents
        self._analysis_id += 1
        current_id = self._analysis_id
        print(f"\n🚀 Nouvelle analyse lancée #ID={current_id}")
        
        # 🔧 Réinitialiser les données complètement
        self.current_results = []
        if hasattr(self, 'filtered_results'):
            delattr(self, 'filtered_results')
        if hasattr(self, 'backtest_map'):
            delattr(self, 'backtest_map')
        
        # Stop any running download thread before starting a new one
        if hasattr(self, 'download_thread') and self.download_thread is not None:
            try:
                # Disconnect signals to prevent old thread from updating UI
                self.download_thread.finished.disconnect()
                self.download_thread.error.disconnect()
                self.download_thread.progress.disconnect()
            except Exception:
                pass
            
            # Stop the thread
            if self.download_thread.isRunning():
                self.download_thread.stop()
                self.download_thread.wait(2000)  # Wait max 2 seconds
                print(f"⚠️ Ancien download thread arrêté")

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
        min_holding_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
        self.download_thread = DownloadThread(symbols, period, do_backtest=False, analysis_id=current_id, min_holding_days=min_holding_days)
        self.download_thread.finished.connect(self.on_download_complete)
        self.download_thread.error.connect(self.on_analysis_error)
        self.download_thread.progress.connect(self.on_analysis_progress)
        self.download_thread.start()
        print(f"📥 Download thread démarré avec ID={current_id}")

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

        period = self.period_input.currentData()
        if not period:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une période d'analyse")
            return

        # 🔧 Incrémenter l'ID d'analyse et stopper les threads précédents
        self._analysis_id += 1
        current_id = self._analysis_id
        print(f"\n🚀 Nouvelle analyse backtest lancée #ID={current_id}")
        
        # 🔧 Réinitialiser les données complètement
        self.current_results = []
        if hasattr(self, 'filtered_results'):
            delattr(self, 'filtered_results')
        if hasattr(self, 'backtest_map'):
            delattr(self, 'backtest_map')
        
        # Stop any running analysis thread before starting a new one
        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                # Disconnect signals to prevent old thread from updating UI
                self.analysis_thread.finished.disconnect()
                self.analysis_thread.error.disconnect()
                self.analysis_thread.progress.disconnect()
            except Exception:
                pass
            
            # Stop the thread
            if self.analysis_thread.isRunning():
                self.analysis_thread.stop()
                self.analysis_thread.wait(2000)  # Wait max 2 seconds
                print(f"⚠️ Ancien analysis thread arrêté")

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

        min_holding_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
        self.analysis_thread = AnalysisThread(selected_pop, selected_mes, period, analysis_id=current_id, min_holding_days=min_holding_days)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.start()
        print(f"📊 Analysis backtest thread démarré avec ID={current_id}")

    def update_results_table(self):
        """Fill the merged table (`self.merged_table`) with current results plus backtest metrics.
        Numeric values are stored via Qt.EditRole to enable correct numeric sorting."""
        if not hasattr(self, 'current_results'):
            return

        sorting_was_enabled = self.merged_table.isSortingEnabled()
        if sorting_was_enabled:
            self.merged_table.setSortingEnabled(False)

        self.merged_table.setRowCount(0)
        # 🔧 Toujours utiliser current_results (filtered_results est supprimé à chaque nouvelle analyse)
        raw_results = self.current_results
        
        # Appliquer le filtre de fiabilité minimum si un seuil est défini
        min_fiab_threshold = self.fiab_threshold_spin.value() if hasattr(self, 'fiab_threshold_spin') else 30
        filtered_results = []
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            fiab_val = r.get('Fiabilite', 'N/A')
            if fiab_val == 'N/A':
                # Toujours inclure si pas de données de fiabilité
                filtered_results.append(r)
            else:
                try:
                    fiab_num = float(fiab_val) if isinstance(fiab_val, (int, float, str)) else 0.0
                    if fiab_num >= min_fiab_threshold:
                        filtered_results.append(r)
                except (ValueError, TypeError):
                    # Inclure si conversion échoue
                    filtered_results.append(r)
        
        results_to_display = [r for r in filtered_results if str(r.get('Symbole', '')).strip()]
        # 🔧 Garantir les champs backtest et dérivées pour éviter des cellules vides
        for r in results_to_display:
            r.setdefault('Fiabilite', 'N/A')
            r.setdefault('NbTrades', 0)
            r.setdefault('Gagnants', 0)
            r.setdefault('Gain_total', 0.0)
            r.setdefault('Gain_moyen', 0.0)
            r.setdefault('Drawdown_max', 0.0)
            # Dérivées techniques
            r.setdefault('dPrice', 0.0)
            r.setdefault('dMACD', 0.0)
            r.setdefault('dRSI', 0.0)
            r.setdefault('dVolRel', 0.0)
            # Données financières
            r.setdefault('Rev. Growth (%)', 0.0)
            r.setdefault('EBITDA Yield (%)', 0.0)
            r.setdefault('FCF Yield (%)', 0.0)
            r.setdefault('D/E Ratio', 0.0)
            r.setdefault('Market Cap (B$)', 0.0)
            r.setdefault('ROE (%)', 0.0)
        
        bt_map = getattr(self, 'backtest_map', {})

        # Helper de conversion robuste pour éviter qu'une valeur vide casse la ligne
        def safe_float(val, default=0.0):
            try:
                if val is None:
                    return default
                if isinstance(val, str) and val.strip() == '':
                    return default
                return float(val)
            except Exception:
                return default

        def safe_int(val, default=0):
            try:
                if val is None:
                    return default
                if isinstance(val, str) and val.strip() == '':
                    return default
                return int(val)
            except Exception:
                return default

        for signal in results_to_display:
            row = -1
            try:
                row = self.merged_table.rowCount()
                self.merged_table.insertRow(row)
                sym = signal.get('Symbole', '')

                # Basic columns
                self.merged_table.setItem(row, 0, QTableWidgetItem(str(sym)))
                self.merged_table.setItem(row, 1, QTableWidgetItem(str(signal.get('Signal', ''))))

                if signal.get('Signal', '') == 'ACHAT':
                    self.merged_table.item(row, 1).setForeground(QColor(0, 128, 0))  # Vert
                elif signal.get('Signal', '') == 'VENTE':
                    self.merged_table.item(row, 1).setForeground(QColor(255, 0, 0))  # Rouge

                score = safe_float(signal.get('Score', 0.0))
                item = QTableWidgetItem(f"{score:.2f}")
                item.setData(Qt.EditRole, score)
                self.merged_table.setItem(row, 2, item)

                prix = safe_float(signal.get('Prix', 0.0))
                item = QTableWidgetItem(f"{prix:.2f}")
                item.setData(Qt.EditRole, prix)
                self.merged_table.setItem(row, 3, item)

                self.merged_table.setItem(row, 4, QTableWidgetItem(str(signal.get('Tendance', ''))))

                rsi = safe_float(signal.get('RSI', 0.0))
                item = QTableWidgetItem(f"{rsi:.2f}")
                item.setData(Qt.EditRole, rsi)
                # RSI: < 30 = excellent (survente, opportunité achat), 30-40 = bon, 40-60 = neutre, 60-70 = attention, > 70 = mauvais (surachat)
                if rsi < 30:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent (survente)
                elif rsi < 40:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif rsi <= 60:
                    item.setForeground(QColor(255, 165, 0))  # Orange : zone neutre
                elif rsi <= 70:
                    item.setForeground(QColor(255, 100, 0))  # Orange foncé : attention (surachat imminent)
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais (surachat)
                self.merged_table.setItem(row, 5, item)

                vol = safe_float(signal.get('Volume moyen', 0.0))
                item = QTableWidgetItem(f"{vol:,.0f}")
                item.setData(Qt.EditRole, float(vol))
                # Volume: plus de volume = meilleure liquidité
                if vol > 5000000:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent volume
                elif vol > 1000000:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon volume
                elif vol > 100000:
                    item.setForeground(QColor(255, 165, 0))  # Orange : volume moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : faible liquidité
                self.merged_table.setItem(row, 6, item)

                self.merged_table.setItem(row, 7, QTableWidgetItem(str(signal.get('Domaine', ''))))

                self.merged_table.setItem(row, 8, QTableWidgetItem(str(signal.get('CapRange', ''))))

                
                # Score/Seuil ratio (dynamique selon domaine_cap et signe du score)
                score_val = safe_float(signal.get('Score', 0.0))
                domaine = str(signal.get('Domaine', ''))
                cap_range = str(signal.get('CapRange', ''))
                
                # Récupérer les seuils optimisés pour domaine_cap
                seuil_achat = 4.2  # Défaut
                seuil_vente = -0.5  # Défaut (négatif pour les seuils de vente)
                try:
                    best_params_all = getattr(self, 'best_parameters', {})
                    param_key = None
                    # Chercher d'abord domaine_cap
                    if cap_range and cap_range != "Unknown":
                        test_key = f"{domaine}_{cap_range}"
                        if test_key in best_params_all:
                            param_key = test_key
                    # Sinon fallback sur domaine seul
                    if not param_key and domaine in best_params_all:
                        param_key = domaine

                    if param_key and param_key in best_params_all:
                        params = best_params_all[param_key]
                        if len(params) > 2 and params[2]:
                            globals_th = params[2]
                            if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                                seuil_achat = float(globals_th[0]) if globals_th[0] else 4.2
                                seuil_vente = float(globals_th[1]) if globals_th[1] else -0.5
                except Exception as e:
                    print(f"⚠️ Erreur lecture seuils pour {sym}: {e}")
                    pass

                # Calculer le ratio selon le signe du score
                try:
                    if score_val > 0:
                        ratio = score_val / seuil_achat if seuil_achat and seuil_achat != 0 else 0.0
                    elif score_val < 0:
                        ratio = score_val / seuil_vente if seuil_vente and seuil_vente != 0 else 0.0
                    else:
                        ratio = 0.0
                except Exception as e:
                    print(f"⚠️ Erreur calcul ratio pour {sym}: {e}")
                    ratio = 0.0
                
                item = QTableWidgetItem(f"{ratio:.2f}")
                item.setData(Qt.EditRole, ratio)
                # Harmoniser : ratio > 1 = excellent (dépassé le seuil), 0.5-1 = moyen, <0.5 = mauvais
                if ratio > 1.5:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif ratio > 1.0:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon (dépasse le seuil)
                elif ratio > 0.5:
                    item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais
                self.merged_table.setItem(row, 9, item)
                
                # Fiabilite and NbTrades (from signal or backtest)
                try:
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
                        # Harmoniser : >75% excellent, 50-75% bon, 30-50% moyen, <30% mauvais
                        if fiab_val >= 75:
                            item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                        elif fiab_val >= 50:
                            item.setForeground(QColor(34, 139, 34))  # Vert : bon
                        elif fiab_val >= 30:
                            item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                        else:
                            item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais
                    self.merged_table.setItem(row, 10, item)
                except Exception as e:
                    print(f"⚠️ Erreur colonne Fiabilité (10) pour {sym}: {e}")
                    self.merged_table.setItem(row, 10, QTableWidgetItem("N/A"))

                # NbTrades
                try:
                    nb_int = safe_int(nb_trades, 0)
                    item = QTableWidgetItem(str(nb_int))
                    item.setData(Qt.EditRole, nb_int)
                    self.merged_table.setItem(row, 11, item)
                except Exception as e:
                    print(f"⚠️ Erreur colonne NbTrades (11) pour {sym}: {e}")
                    self.merged_table.setItem(row, 11, QTableWidgetItem("0"))

                # Gagnants
                try:
                    gagnants = int(bt.get('gagnants', 0)) if bt else 0
                    item = QTableWidgetItem(str(gagnants))
                    item.setData(Qt.EditRole, gagnants)
                    self.merged_table.setItem(row, 12, item)
                except Exception as e:
                    print(f"⚠️ Erreur colonne Gagnants (12) pour {sym}: {e}")
                    self.merged_table.setItem(row, 12, QTableWidgetItem("0"))
                
                # Colonnes Financières simples
                # Colonne 13: Rev. Growth (%)
                rev_growth = safe_float(signal.get('Rev. Growth (%)', 0.0))
                item = QTableWidgetItem(f"{rev_growth:.2f}")
                item.setData(Qt.EditRole, rev_growth)
                # Harmoniser : croissance revenue positive = bon
                if rev_growth > 20:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif rev_growth > 5:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif rev_growth > 0:
                    item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais (négatif)
                self.merged_table.setItem(row, 13, item)

                # Colonne 14: EBITDA Yield (%) - avec couleurs
                ebitda = safe_float(signal.get('EBITDA Yield (%)', 0.0))
                item = QTableWidgetItem(f"{ebitda:.2f}")
                item.setData(Qt.EditRole, ebitda)
                # Harmoniser : EBITDA positif élevé = bon
                if ebitda > 15:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif ebitda > 8:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif ebitda > 0:
                    item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais (négatif)
                self.merged_table.setItem(row, 14, item)

                # Colonne 15: FCF Yield (%)
                fcf = safe_float(signal.get('FCF Yield (%)', 0.0))
                item = QTableWidgetItem(f"{fcf:.2f}")
                item.setData(Qt.EditRole, fcf)
                # Harmoniser : FCF positif = bon
                if fcf > 10:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif fcf > 3:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif fcf > 0:
                    item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais (négatif)
                self.merged_table.setItem(row, 15, item)

                # Colonne 16: D/E Ratio (bas = bon)
                de_ratio = safe_float(signal.get('D/E Ratio', 0.0))
                item = QTableWidgetItem(f"{de_ratio:.2f}")
                item.setData(Qt.EditRole, de_ratio)
                # Harmoniser : ratio bas = excellent (moins d'endettement), ratio haut = mauvais
                if de_ratio < 0.5:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif de_ratio < 1.5:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif de_ratio < 2.5:
                    item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais (trop endetté)
                self.merged_table.setItem(row, 16, item)

                # Colonne 16: Market Cap (B$)
                market_cap = safe_float(signal.get('Market Cap (B$)', 0.0))
                item = QTableWidgetItem(f"{market_cap:.2f}")
                item.setData(Qt.EditRole, market_cap)
                self.merged_table.setItem(row, 17, item)

                # Colonne 17: ROE (%)
                roe = safe_float(signal.get('ROE (%)', 0.0))
                item = QTableWidgetItem(f"{roe:.2f}")
                item.setData(Qt.EditRole, roe)
                # Colorer basé sur ROE : vert si > 15%, orange si 10-15%, rouge si < 10%
                if roe > 15:
                    item.setForeground(QColor(0, 128, 0))  # Vert : excellent
                elif roe > 10:
                    item.setForeground(QColor(255, 165, 0))  # Orange : bon
                elif roe > 5:
                    item.setForeground(QColor(255, 140, 0))  # Orange clair : acceptable
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : faible
                self.merged_table.setItem(row, 18, item)

                # Derivatives (colonnes 19-22)
                dprice = safe_float(signal.get('dPrice', 0.0))
                item = QTableWidgetItem(f"{dprice:.3f}")
                item.setData(Qt.EditRole, dprice)
                self.merged_table.setItem(row, 19, item)

                dmacd = safe_float(signal.get('dMACD', 0.0))
                item = QTableWidgetItem(f"{dmacd:.3f}")
                item.setData(Qt.EditRole, dmacd)
                self.merged_table.setItem(row, 20, item)

                drsi = safe_float(signal.get('dRSI', 0.0))
                item = QTableWidgetItem(f"{drsi:.3f}")
                item.setData(Qt.EditRole, drsi)
                self.merged_table.setItem(row, 21, item)

                dvol = safe_float(signal.get('dVolRel', 0.0))
                item = QTableWidgetItem(f"{dvol:.3f}")
                item.setData(Qt.EditRole, dvol)
                self.merged_table.setItem(row, 22, item)

                # Backtest metrics (if available)
                # Colonnes 23-24 pour Gain total et Gain moyen
                try:
                    trades = int(bt.get('trades', 0)) if bt else 0
                    taux = float(bt.get('taux_reussite', 0.0)) if bt else 0.0
                    gain_total = float(bt.get('gain_total', 0.0)) if bt else 0.0
                    gain_moy = float(bt.get('gain_moyen', 0.0)) if bt else 0.0

                    item = QTableWidgetItem(f"{gain_total:.2f}")
                    item.setData(Qt.EditRole, gain_total)
                    # Harmoniser : gain > 0 = bon
                    if gain_total > 200:
                        item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                    elif gain_total > 50:
                        item.setForeground(QColor(34, 139, 34))  # Vert : bon
                    elif gain_total > 0:
                        item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                    else:
                        item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais
                    self.merged_table.setItem(row, 23, item)

                    item = QTableWidgetItem(f"{gain_moy:.2f}")
                    item.setData(Qt.EditRole, gain_moy)
                    # Harmoniser : gain moyen > 0 = bon
                    if gain_moy > 20:
                        item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                    elif gain_moy > 5:
                        item.setForeground(QColor(34, 139, 34))  # Vert : bon
                    elif gain_moy > 0:
                        item.setForeground(QColor(255, 165, 0))  # Orange : moyen
                    else:
                        item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais
                    self.merged_table.setItem(row, 24, item)
                except Exception as e:
                    print(f"⚠️ Erreur colonnes Gain (23-24) pour {sym}: {e}")
                    self.merged_table.setItem(row, 23, QTableWidgetItem("0.00"))
                    self.merged_table.setItem(row, 24, QTableWidgetItem("0.00"))

                # Consensus (text column at index 25)
                consensus = signal.get('Consensus', 'N/A')
                # Debug: vérifier si le Consensus existe vraiment
                if row == 0:  # Afficher seulement pour la première ligne
                    print(f"🔍 DEBUG Consensus - Symbol: {signal.get('Symbole')}, Consensus value: '{consensus}'")
                item = QTableWidgetItem(str(consensus))
                # Colorer selon consensus
                consensus_lower = str(consensus).lower()
                if 'strong buy' in consensus_lower or 'achat fort' in consensus_lower:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé : excellent
                elif 'buy' in consensus_lower or 'achat' in consensus_lower:
                    item.setForeground(QColor(34, 139, 34))  # Vert : bon
                elif 'hold' in consensus_lower or 'conserver' in consensus_lower or 'neutre' in consensus_lower:
                    item.setForeground(QColor(255, 165, 0))  # Orange : neutre
                elif 'sell' in consensus_lower or 'vente' in consensus_lower:
                    item.setForeground(QColor(255, 0, 0))  # Rouge : mauvais
                self.merged_table.setItem(row, 25, item)

                # Earnings Surprise (column 26)
                surprise_raw = signal.get('Earn Surp (%)', signal.get('latest_earnings_surprise', 'N/A'))
                try:
                    if surprise_raw in (None, '', 'N/A'):
                        surprise_item = QTableWidgetItem('N/A')
                    else:
                        surprise_val = float(surprise_raw)
                        surprise_item = QTableWidgetItem(f"{surprise_val:.2f}%")
                        surprise_item.setData(Qt.EditRole, surprise_val)
                        if surprise_val > 10:
                            surprise_item.setForeground(QColor(0, 128, 0))
                        elif surprise_val > 0:
                            surprise_item.setForeground(QColor(34, 139, 34))
                        elif surprise_val < -5:
                            surprise_item.setForeground(QColor(255, 0, 0))
                        else:
                            surprise_item.setForeground(QColor(255, 165, 0))
                    self.merged_table.setItem(row, 26, surprise_item)
                except Exception:
                    self.merged_table.setItem(row, 26, QTableWidgetItem('N/A'))

                # Analyst Upgrades (column 27)
                upgrades_raw = signal.get('Ana Upgrad', signal.get('recent_upgrades_count', 0))
                try:
                    upgrades_val = int(upgrades_raw or 0)
                    upgrades_item = QTableWidgetItem(str(upgrades_val))
                    upgrades_item.setData(Qt.EditRole, upgrades_val)
                    if upgrades_val >= 2:
                        upgrades_item.setForeground(QColor(0, 128, 0))
                    elif upgrades_val == 1:
                        upgrades_item.setForeground(QColor(34, 139, 34))
                    else:
                        upgrades_item.setForeground(QColor(255, 165, 0))
                    self.merged_table.setItem(row, 27, upgrades_item)
                except Exception:
                    self.merged_table.setItem(row, 27, QTableWidgetItem('0'))

                # item = QTableWidgetItem(f"{drawdown:.2f}")
                # item.setData(Qt.EditRole, drawdown)
                # self.merged_table.setItem(row, 19, item)

            except Exception as e:
                print(f"❌ Erreur remplissage table pour {signal.get('Symbole', '?')}: {e}")
                import traceback
                traceback.print_exc()
                if row >= 0:
                    self.merged_table.removeRow(row)
                continue

        if sorting_was_enabled:
            self.merged_table.setSortingEnabled(True)
        
        # Mettre à jour les onglets Graphiques et Comparaisons après remplissage de la table
        try:
            self.populate_charts_tab()
        except Exception as e:
            print(f"⚠️ Erreur lors de la mise à jour de l'onglet Graphiques: {e}")
        
        try:
            self.populate_comparisons_tab()
        except Exception as e:
            print(f"⚠️ Erreur lors de la mise à jour de l'onglet Comparaisons: {e}")
    
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
            16: ('roe_val', True),         # ROE décroissant
            17: ('roe_val', False),        # ROE croissant
            18: ('fcf_val', True),         # FCF décroissant
            19: ('debt_to_equity_val', False) # D/E Ratio croissant
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
    
    def toggle_offline_mode(self):
        """Bascule entre le mode online et offline"""
        is_offline = self.offline_mode_btn.isChecked()
        qsi.OFFLINE_MODE = is_offline
        
        if is_offline:
            self.offline_mode_btn.setText("📴 Mode: OFFLINE")
            self.offline_mode_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
            self.summary_text.append("\n⚠️ Mode OFFLINE activé - Utilisation du cache uniquement")
        else:
            self.offline_mode_btn.setText("🌐 Mode: ONLINE")
            self.offline_mode_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            self.summary_text.append("\n✅ Mode ONLINE activé - Téléchargement si cache obsolète")

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
            min_hold_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
            lines.append(
                f"🌍 Résultat global :\n"
                f" - Taux de réussite = {taux_global:.1f}%\n"
                f" - Nombre de trades = {total_trades}\n"
                f" - Gain total brut = {total_gain:.2f} $\n"
                f" - Durée min position = {min_hold_days} jour(s) actif(s)"
            )
            lines.append("\n📊 Taux de réussite par domaine:")
            for dom, stats in sorted(domain_stats.items(), key=lambda x: -x[1]['trades']):
                trades = stats['trades']
                gagnants = stats['gagnants']
                taux = (gagnants / trades * 100) if trades > 0 else 0.0
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

    def _compute_score_series(self, prices, volumes, domaine='Inconnu', cap_range=None, symbol=None):
        """Calcule l'evolution du score sur la fenetre analysee (jours actifs)."""
        score_dates = []
        score_values = []
        start_idx = 50

        if len(prices) <= start_idx:
            return score_dates, score_values

        # Precision maximale: calcul quotidien (chaque jour actif) pour eviter
        # les artefacts visuels d'interpolation sur les seuils.
        step = 1

        for i in range(start_idx, len(prices), step):
            try:
                _sig, _last_price, _trend, _last_rsi, _vol_mean, score, _ = get_trading_signal(
                    prices.iloc[:i + 1],
                    volumes.iloc[:i + 1],
                    domaine=domaine,
                    cap_range=cap_range,
                    symbol=symbol,
                    return_derivatives=True,
                )
                score_dates.append(prices.index[i])
                score_values.append(float(score))
            except Exception:
                continue

        # Assurer un point final (dernier jour) pour la lecture visuelle
        if score_dates and score_dates[-1] != prices.index[-1]:
            try:
                _sig, _last_price, _trend, _last_rsi, _vol_mean, score, _ = get_trading_signal(
                    prices,
                    volumes,
                    domaine=domaine,
                    cap_range=cap_range,
                    symbol=symbol,
                    return_derivatives=True,
                )
                score_dates.append(prices.index[-1])
                score_values.append(float(score))
            except Exception:
                pass

        return score_dates, score_values

    def _get_global_thresholds_for_symbol(self, domaine='Inconnu', cap_range=None):
        """Retourne (seuil_achat, seuil_vente) optimises pour secteur/cap, avec fallback par defaut."""
        default_buy = 4.2
        default_sell = -0.5
        try:
            best_params = getattr(self, 'best_parameters', None) or extract_best_parameters()
            selected_key = None

            if cap_range:
                comp_key = f"{domaine}_{cap_range}"
                if comp_key in best_params:
                    selected_key = comp_key

            if not selected_key and domaine in best_params:
                selected_key = domaine

            if selected_key:
                _coeffs, _thresholds, globals_thresholds, _gain, _extras = best_params[selected_key]
                buy_thr = float(globals_thresholds[0])
                sell_thr = float(globals_thresholds[1])
                return buy_thr, sell_thr
        except Exception:
            pass

        return default_buy, default_sell

    def _build_symbol_figure_with_score(self, sym, prices, volumes, precomp=None, events=None):
        """Construit une figure: trace principal + score au fil du temps en dessous."""
        precomp = precomp or {}
        events = events or []

        fig = Figure(figsize=(10, 7.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.18)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_score = fig.add_subplot(gs[1, 0], sharex=ax_main)

        score_val = precomp.get('score')
        try:
            # Use show_xaxis=True to avoid plot_unified_chart clearing shared x tick labels.
            plot_unified_chart(sym, prices, volumes, ax_main, show_xaxis=True, score_override=score_val, precomputed=precomp)
        except Exception:
            ax_main.plot(prices.index, prices.values, color='black', linewidth=1.2)
            if isinstance(score_val, (int, float)):
                ax_main.set_title(f"{sym} | Score: {score_val:.2f}")
            else:
                ax_main.set_title(sym)

        # Hide x labels on top panel only; keep bottom panel date labels visible.
        ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        for ev in events:
            if ev.get('type') == 'BUY':
                ax_main.scatter(ev['date'], ev['price'], marker='^', s=80, color='green', edgecolor='black', zorder=6)
            elif ev.get('type') == 'SELL':
                ax_main.scatter(ev['date'], ev['price'], marker='v', s=80, color='red', edgecolor='black', zorder=6)

        score_dates, score_values = self._compute_score_series(
            prices,
            volumes,
            domaine=precomp.get('domaine', 'Inconnu'),
            cap_range=precomp.get('cap_range'),
            symbol=sym,
        )
        buy_thr, sell_thr = self._get_global_thresholds_for_symbol(
            domaine=precomp.get('domaine', 'Inconnu'),
            cap_range=precomp.get('cap_range'),
        )

        if score_dates and score_values:
            ax_score.plot(score_dates, score_values, color='#1565C0', linewidth=1.6, label='Score')
            ax_score.axhline(y=buy_thr, color='green', linestyle='--', alpha=0.5, linewidth=1.0, label=f'Seuil Achat ({buy_thr:.2f})')
            ax_score.axhline(y=sell_thr, color='red', linestyle='--', alpha=0.5, linewidth=1.0, label=f'Seuil Vente ({sell_thr:.2f})')
            ax_score.legend(loc='upper left', fontsize=8, frameon=True)
        else:
            ax_score.text(0.5, 0.5, 'Score indisponible', transform=ax_score.transAxes,
                          ha='center', va='center', fontsize=9)

        ax_score.set_ylabel('Score', fontsize=9)
        ax_score.set_xlabel('Date', fontsize=9)
        ax_score.text(
            0.01,
            0.98,
            'Seuils appliques au score (pas au prix)',
            transform=ax_score.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            color='#424242',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5),
        )
        ax_score.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax_score.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_score.xaxis.get_major_locator()))
        ax_score.tick_params(axis='x', labelrotation=0, labelsize=8)
        ax_score.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig




    def compute_domain_stats(self):
        """Agrège les résultats par domaine depuis la merged_table."""
        try:
            if not hasattr(self, 'merged_table') or self.merged_table.rowCount() == 0:
                return {'global': {}, 'by_domain': {}}
            
            domain_stats = {}
            total_trades = 0
            total_gagnants = 0
            total_gain = 0.0
            
            for row in range(self.merged_table.rowCount()):
                try:
                    # Colonne 7: Domaine
                    domaine_item = self.merged_table.item(row, 7)
                    domaine = domaine_item.text() if domaine_item and domaine_item.text().strip() else 'Inconnu'
                    
                    # Colonne 11: Nb Trades
                    trades_item = self.merged_table.item(row, 11)
                    nb_trades = int(trades_item.data(Qt.EditRole)) if trades_item and trades_item.data(Qt.EditRole) is not None else 0
                    
                    # Colonne 12: Gagnants
                    gagnants_item = self.merged_table.item(row, 12)
                    gagnants = int(gagnants_item.data(Qt.EditRole)) if gagnants_item and gagnants_item.data(Qt.EditRole) is not None else 0
                    
                    # Colonne 24: Gain total ($) ✅ FIX: était 22 (dRSI), maintenant 24 (Gain total)
                    gain_item = self.merged_table.item(row, 24)
                    gain = float(gain_item.data(Qt.EditRole)) if gain_item and gain_item.data(Qt.EditRole) is not None else 0.0
                    
                    if domaine not in domain_stats:
                        domain_stats[domaine] = {'trades': 0, 'gagnants': 0, 'gain': 0.0}
                    
                    domain_stats[domaine]['trades'] += nb_trades
                    domain_stats[domaine]['gagnants'] += gagnants
                    domain_stats[domaine]['gain'] += gain
                    
                    total_trades += nb_trades
                    total_gagnants += gagnants
                    total_gain += gain
                except Exception as e:
                    print(f"⚠️ Erreur compute_domain_stats row {row}: {e}")
                    continue
            
            # Calculer taux de réussite par domaine
            for domaine in domain_stats:
                trades = domain_stats[domaine]['trades']
                gagnants = domain_stats[domaine]['gagnants']
                taux = (gagnants / trades * 100) if trades > 0 else 0.0
                domain_stats[domaine]['taux'] = taux
            
            total_taux = (total_gagnants / total_trades * 100) if total_trades > 0 else 0.0
            
            return {
                'global': {
                    'trades': total_trades,
                    'gagnants': total_gagnants,
                    'taux': total_taux,
                    'gain': total_gain
                },
                'by_domain': domain_stats
            }
        except Exception as e:
            print(f"❌ Erreur compute_domain_stats: {e}")
            import traceback
            traceback.print_exc()
            return {'global': {}, 'by_domain': {}}
    
    def populate_charts_tab(self):
        """Génère les graphiques de comparaison par domaine dans l'onglet Graphiques."""
        try:
            # Nettoyer l'onglet Graphiques
            while self.charts_scroll_layout.count() > 0:
                widget = self.charts_scroll_layout.takeAt(0).widget()
                if widget:
                    widget.deleteLater()
            
            stats = self.compute_domain_stats()
            if not stats['by_domain']:
                self.charts_scroll_layout.addWidget(QLabel("Aucun résultat à afficher. Analysez d'abord des symboles."))
                return
            
            # Titre + résumé global compact (sur une seule ligne)
            title = QLabel("📊 Analyse par domaine")
            title.setStyleSheet("font-weight: bold; font-size: 12px;")
            
            global_info = stats['global']
            min_hold_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
            summary_text = (
                f"🌍 Résultat global: Taux={global_info['taux']:.1f}% | Trades={global_info['trades']} | "
                f"Gain=${global_info['gain']:.2f} | Durée min={min_hold_days}j"
            )
            summary_label = QLabel(summary_text)
            summary_label.setStyleSheet("background-color: #f0f0f0; padding: 6px; border-radius: 4px; font-size: 10px;")
            
            # Graphiques matplotlib (sans tableau textuel)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # Préparer les données pour les graphiques
            domains = list(stats['by_domain'].keys())
            taux_list = [stats['by_domain'][d]['taux'] for d in domains]
            trades_list = [stats['by_domain'][d]['trades'] for d in domains]
            gain_list = [stats['by_domain'][d]['gain'] for d in domains]
            
            # Compter le nombre de symboles par domaine
            symbols_per_domain = {}
            for row in range(self.merged_table.rowCount()):
                try:
                    domaine_item = self.merged_table.item(row, 7)
                    domaine = domaine_item.text() if domaine_item else 'Inconnu'
                    symbols_per_domain[domaine] = symbols_per_domain.get(domaine, 0) + 1
                except Exception:
                    continue
            
            symbols_list = [symbols_per_domain.get(d, 0) for d in domains]
            
            # Calculer la rentabilité annualisée en % par secteur
            # Capital investi: 50€ par stock
            period_str = self.period_input.currentData() if hasattr(self, 'period_input') else "12mo"
            
            # Convertir la période en années
            if 'y' in period_str:
                years = float(period_str.replace('y', ''))
            elif 'mo' in period_str:
                months = float(period_str.replace('mo', ''))
                years = months / 12.0
            elif 'd' in period_str:
                days = float(period_str.replace('d', ''))
                years = days / 365.0
            else:
                years = 1.0  # Par défaut 1 an
            
            rentabilite_annuelle_pct_list = []
            for d in domains:
                trades = stats['by_domain'][d]['trades']
                gain = stats['by_domain'][d]['gain']
                nb_symbols = symbols_per_domain.get(d, 1)
                
                # Capital investi: 50€ par stock
                capital_investi = nb_symbols * 50.0
                
                # Rendement total en %
                rendement_total_pct = (gain / capital_investi) * 100 if capital_investi > 0 else 0.0
                
                # Annualiser le rendement
                rendement_annuel_pct = rendement_total_pct / years if years > 0 else rendement_total_pct
                
                rentabilite_annuelle_pct_list.append(rendement_annuel_pct)
            
            # Créer figure avec moins de graphiques pour plus de clarté (2 lignes x 2 colonnes)
            fig = Figure(figsize=(16, 10), dpi=90)
            
            # Subplot 1: Bar chart taux de réussite
            ax1 = fig.add_subplot(2, 2, 1)
            colors = ['green' if t >= 60 else 'orange' if t >= 40 else 'red' for t in taux_list]
            bars1 = ax1.bar(domains, taux_list, color=colors, alpha=0.75, edgecolor='black', linewidth=1.2)
            ax1.set_ylabel('Taux de réussite (%)', fontweight='bold', fontsize=11)
            ax1.set_title('Taux de réussite', fontweight='bold', fontsize=13, pad=10)
            ax1.set_ylim(0, 105)
            ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.tick_params(axis='x', rotation=45, labelsize=9)
            ax1.tick_params(axis='y', labelsize=9)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            for i, v in enumerate(taux_list):
                ax1.text(i, v + 3, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
            
            # Subplot 2: Bar chart gain total
            ax2 = fig.add_subplot(2, 2, 2)
            colors_gain = ['green' if g > 0 else 'red' for g in gain_list]
            bars2 = ax2.bar(domains, gain_list, color=colors_gain, alpha=0.75, edgecolor='black', linewidth=1.2)
            ax2.set_ylabel('Gain total ($)', fontweight='bold', fontsize=11)
            ax2.set_title('Gain brut total', fontweight='bold', fontsize=13, pad=10)
            ax2.axhline(y=0, color='black', linewidth=1.5)
            ax2.tick_params(axis='x', rotation=45, labelsize=9)
            ax2.tick_params(axis='y', labelsize=9)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            for i, v in enumerate(gain_list):
                offset = 20 if v > 0 else -40
                ax2.text(i, v + offset, f'${v:.0f}', ha='center', fontsize=10, fontweight='bold')
            
            # Subplot 3: Bar chart rentabilité annuelle en %
            ax3 = fig.add_subplot(2, 2, 3)
            colors_rentabilite = ['darkgreen' if r > 50 else 'green' if r > 10 else 'orange' if r > 0 else 'red' for r in rentabilite_annuelle_pct_list]
            bars3 = ax3.bar(domains, rentabilite_annuelle_pct_list, color=colors_rentabilite, alpha=0.75, edgecolor='black', linewidth=1.2)
            ax3.set_ylabel('Rentabilité annuelle (%)', fontweight='bold', fontsize=11)
            ax3.set_title('Rentabilité annualisée', fontweight='bold', fontsize=13, pad=10)
            ax3.axhline(y=0, color='black', linewidth=1.5)
            ax3.axhline(y=10, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax3.tick_params(axis='x', rotation=45, labelsize=9)
            ax3.tick_params(axis='y', labelsize=9)
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            for i, v in enumerate(rentabilite_annuelle_pct_list):
                offset = max(abs(v) * 0.1, 3) if v > 0 else -max(abs(v) * 0.1, 5)
                ax3.text(i, v + offset, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
            
            # Subplot 4: Distribution trades + symboles combinée
            ax4 = fig.add_subplot(2, 2, 4)
            x_pos = range(len(domains))
            width = 0.35
            
            # Normaliser pour affichage sur même échelle (éviter division par zéro)
            max_trades = max(trades_list) if trades_list and max(trades_list) > 0 else 1
            max_symbols = max(symbols_list) if symbols_list and max(symbols_list) > 0 else 1
            trades_normalized = [t / max_trades * 100 if max_trades > 0 else 0 for t in trades_list]
            symbols_normalized = [s / max_symbols * 100 if max_symbols > 0 else 0 for s in symbols_list]
            
            bars4a = ax4.bar([x - width/2 for x in x_pos], trades_normalized, width, 
                            label='Trades', color='steelblue', alpha=0.75, edgecolor='black', linewidth=1)
            bars4b = ax4.bar([x + width/2 for x in x_pos], symbols_normalized, width,
                            label='Symboles', color='coral', alpha=0.75, edgecolor='black', linewidth=1)
            
            ax4.set_ylabel('Distribution (normalisée)', fontweight='bold', fontsize=11)
            ax4.set_title('Trades et Symboles par secteur', fontweight='bold', fontsize=13, pad=10)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
            ax4.tick_params(axis='y', labelsize=9)
            ax4.legend(loc='upper right', fontsize=10)
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Ajouter les valeurs réelles au-dessus des barres
            for i, (t, s) in enumerate(zip(trades_list, symbols_list)):
                ax4.text(i - width/2, trades_normalized[i] + 3, str(int(t)), 
                        ha='center', fontsize=8, fontweight='bold')
                ax4.text(i + width/2, symbols_normalized[i] + 3, str(int(s)), 
                        ha='center', fontsize=8, fontweight='bold')
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Ajouter à l'onglet Graphiques
            self.charts_scroll_layout.addWidget(title)
            self.charts_scroll_layout.addWidget(summary_label)
            self.charts_scroll_layout.addWidget(canvas)
            self.charts_scroll_layout.addStretch()
            
        except Exception as e:
            print(f"❌ Erreur populate_charts_tab: {e}")
            import traceback
            traceback.print_exc()
    
    def populate_comparisons_tab(self):
        """Onglet Comparaisons: permet de sélectionner jusqu'à 15 symboles et les comparer."""
        try:
            # Nettoyer l'onglet Comparaisons
            while self.comparisons_layout.count() > 0:
                widget = self.comparisons_layout.takeAt(0).widget()
                if widget:
                    widget.deleteLater()
            
            # Titre
            title = QLabel("📊 Comparaison personnalisée de symboles (max 15)")
            title.setStyleSheet("font-weight: bold; font-size: 12px;")
            self.comparisons_layout.addWidget(title)
            
            # Récupérer tous les symboles disponibles
            all_symbols = []
            for row in range(self.merged_table.rowCount()):
                try:
                    sym = self.merged_table.item(row, 0).text()
                    if sym:
                        all_symbols.append(sym)
                except Exception:
                    continue
            
            if not all_symbols:
                self.comparisons_layout.addWidget(QLabel("Aucun symbole disponible pour la comparaison."))
                return
            
            # Conteneur de sélection avec scroll
            selection_container = QWidget()
            selection_layout = QVBoxLayout(selection_container)
            
            # Label pour sélection avec boutons rapides
            select_header_layout = QHBoxLayout()
            select_label = QLabel("✓ Sélectionnez jusqu'à 15 symboles:")
            select_label.setStyleSheet("font-weight: bold; font-size: 10px;")
            select_header_layout.addWidget(select_label)
            select_header_layout.addStretch()
            
            # ✅ Boutons de sélection rapide
            select_all_btn = QPushButton("Tout sélectionner")
            select_all_btn.setMaximumWidth(120)
            select_all_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 9px; padding: 4px;")
            
            deselect_all_btn = QPushButton("Tout désélectionner")
            deselect_all_btn.setMaximumWidth(130)
            deselect_all_btn.setStyleSheet("background-color: #757575; color: white; font-size: 9px; padding: 4px;")
            
            select_header_layout.addWidget(select_all_btn)
            select_header_layout.addWidget(deselect_all_btn)
            
            select_label_widget = QWidget()
            select_label_widget.setLayout(select_header_layout)
            selection_layout.addWidget(select_label_widget)
            
            # Sélecteur de date pour comparaison historique
            date_container = QWidget()
            date_layout = QHBoxLayout(date_container)
            date_label = QLabel("📅 Date de référence (optionnel):")
            date_label.setStyleSheet("font-size: 10px;")
            
            from PyQt5.QtWidgets import QDateEdit
            from PyQt5.QtCore import QDate
            self.comparison_date_edit = QDateEdit()
            self.comparison_date_edit.setCalendarPopup(True)
            self.comparison_date_edit.setDate(QDate.currentDate())
            self.comparison_date_edit.setMaximumDate(QDate.currentDate())
            self.comparison_date_edit.setMinimumDate(QDate(2020, 1, 1))
            self.comparison_date_edit.setDisplayFormat("dd/MM/yyyy")
            
            self.use_historical_check = QCheckBox("Comparer avec données historiques")
            self.use_historical_check.setToolTip("Analyser les symboles à la date sélectionnée et voir l'évolution réelle depuis")
            
            date_layout.addWidget(date_label)
            date_layout.addWidget(self.comparison_date_edit)
            date_layout.addWidget(self.use_historical_check)
            date_layout.addStretch()
            selection_layout.addWidget(date_container)
            
            # Scroll area pour les checkboxes
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(120)
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            
            # Créer les checkboxes
            checkboxes = {}
            for sym in sorted(all_symbols):
                cb = QCheckBox(sym)
                checkboxes[sym] = cb
                checkbox_layout.addWidget(cb)
            
            scroll.setWidget(checkbox_container)
            selection_layout.addWidget(scroll)
            
            # ✅ Connexions des boutons de sélection rapide
            def on_select_all():
                for cb in checkboxes.values():
                    cb.setChecked(True)
            
            def on_deselect_all():
                for cb in checkboxes.values():
                    cb.setChecked(False)
            
            select_all_btn.clicked.connect(on_select_all)
            deselect_all_btn.clicked.connect(on_deselect_all)
            
            # Boutons d'action
            button_layout = QHBoxLayout()
            compare_btn = QPushButton("Comparer les symboles sélectionnés")
            compare_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            reset_btn = QPushButton("Réinitialiser")
            reset_btn.setStyleSheet("background-color: #f44336; color: white;")
            
            button_layout.addWidget(compare_btn)
            button_layout.addWidget(reset_btn)
            selection_layout.addLayout(button_layout)
            
            self.comparisons_layout.addWidget(selection_container)
            
            # Zone de résultats de comparaison (initialement vide)
            results_scroll = QScrollArea()
            results_scroll.setWidgetResizable(True)
            self.comparison_results = QWidget()
            self.comparison_results_layout = QVBoxLayout(self.comparison_results)
            results_scroll.setWidget(self.comparison_results)
            
            self.comparisons_layout.addWidget(results_scroll)
            self.comparisons_layout.addStretch()
            
            # Connexions des boutons
            def on_compare():
                selected = [sym for sym, cb in checkboxes.items() if cb.isChecked()]
                if not selected:
                    QMessageBox.warning(self, "Erreur", "Sélectionnez au moins 1 symbole pour comparer")
                    return
                if len(selected) > 15:
                    QMessageBox.warning(self, "Erreur", "Maximum 15 symboles à la fois")
                    return
                
                # Nettoyer les résultats précédents
                while self.comparison_results_layout.count() > 0:
                    w = self.comparison_results_layout.takeAt(0).widget()
                    if w:
                        w.deleteLater()
                
                # Check if historical comparison requested
                use_historical = self.use_historical_check.isChecked()
                if use_historical:
                    selected_date = self.comparison_date_edit.date()
                    historical_date_str = selected_date.toString("yyyy-MM-dd")
                    self._generate_historical_comparison_table(selected, historical_date_str)
                else:
                    # Générer le tableau comparatif
                    self._generate_comparison_table(selected)
            
            def on_reset():
                for cb in checkboxes.values():
                    cb.setChecked(False)
                while self.comparisons_layout.count() > 0:
                    w = self.comparisons_layout.takeAt(0).widget()
                    if w:
                        w.deleteLater()
            
            compare_btn.clicked.connect(on_compare)
            reset_btn.clicked.connect(on_reset)
            
        except Exception as e:
            print(f"❌ Erreur populate_comparisons_tab: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_comparison_table(self, symbols_to_compare):
        """Génère un tableau comparatif pour les symboles sélectionnés avec classement par pertinence."""
        try:
            # Récupérer les données pour chaque symbole
            symbols_data = {}
            for row in range(self.merged_table.rowCount()):
                try:
                    sym = self.merged_table.item(row, 0).text()
                    if sym in symbols_to_compare:
                        score = float(self.merged_table.item(row, 2).text()) if self.merged_table.item(row, 2) else 0.0
                        prix = float(self.merged_table.item(row, 3).text()) if self.merged_table.item(row, 3) else 0.0
                        rsi = float(self.merged_table.item(row, 5).text()) if self.merged_table.item(row, 5) else 0.0
                        domaine = self.merged_table.item(row, 7).text() if self.merged_table.item(row, 7) else 'N/A'
                        score_seuil = float(self.merged_table.item(row, 9).text()) if self.merged_table.item(row, 9) else 0.0
                        fiab_text = self.merged_table.item(row, 10).text() if self.merged_table.item(row, 10) else 'N/A'
                        fiab = float(fiab_text.replace('%', '')) if fiab_text != 'N/A' else 0.0
                        trades = int(self.merged_table.item(row, 11).text()) if self.merged_table.item(row, 11) else 0
                        gagnants = int(self.merged_table.item(row, 12).text()) if self.merged_table.item(row, 12) else 0
                        # ✅ Ajouter EBITDA (colonne 14 dans merged_table)
                        ebitda_text = self.merged_table.item(row, 14).text() if self.merged_table.item(row, 14) else '0'
                        ebitda = float(ebitda_text) if ebitda_text else 0.0
                        gain = float(self.merged_table.item(row, 22).text()) if self.merged_table.item(row, 22) else 0.0
                        consensus = self.merged_table.item(row, 25).text() if self.merged_table.item(row, 25) else 'N/A'
                        
                        symbols_data[sym] = {
                            'Score': score,
                            'Prix': prix,
                            'RSI': rsi,
                            'Domaine': domaine,
                            'Score/Seuil': score_seuil,
                            'Fiabilité (%)': fiab,
                            'Nb Trades': trades,
                            'Gagnants': gagnants,
                            'EBITDA Yield (%)': ebitda,  # ✅ Ajouté
                            'Gain ($)': gain,
                            'Consensus': consensus
                        }
                except Exception:
                    continue
            
            # Calculer un score de pertinence pour chaque symbole
            pertinence_scores = {}
            for sym, data in symbols_data.items():
                # Score de pertinence = combinaison pondérée des métriques
                # Facteurs: Score/Seuil (25%), Fiabilité (25%), EBITDA (20%), Gain (15%), RSI (10%), Consensus (5%)
                score_factor = min(data['Score/Seuil'], 2.0) * 25  # Max 50 points
                fiab_factor = data['Fiabilité (%)'] * 0.25  # Max 25 points
                
                # ✅ EBITDA a une influence majeure (20%)
                ebitda_val = data.get('EBITDA Yield (%)', 0.0)
                ebitda_factor = min(max(ebitda_val / 10, 0), 2) * 10  # Max 20 points (positif surtout)
                
                gain_factor = min(max(data['Gain ($)'] / 100, 0), 1.5) * 10  # Max 15 points
                rsi_factor = abs(50 - data['RSI']) * 0.1  # Proche de 50 = meilleur
                consensus_factor = 5 if data['Consensus'] != 'N/A' else 0
                
                pertinence = score_factor + fiab_factor + ebitda_factor + gain_factor + rsi_factor + consensus_factor
                pertinence_scores[sym] = pertinence
            
            # Classer par pertinence (décroissant)
            sorted_symbols = sorted(symbols_data.keys(), key=lambda x: pertinence_scores[x], reverse=True)
            
            # Créer un tableau QTableWidget pour afficher la comparaison
            table = QTableWidget()
            columns = ['Rang', 'Symbole', 'Domaine', 'Score', 'Score/Seuil', 'Fiabilité (%)', 'Nb Trades', 
                      'Gagnants', 'RSI', 'Prix', 'EBITDA (%)', 'Gain ($)', 'Consensus', 'Pertinence']
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            # Remplir le tableau
            for rank, sym in enumerate(sorted_symbols, 1):
                data = symbols_data[sym]
                pertinence = pertinence_scores[sym]
                
                row = table.rowCount()
                table.insertRow(row)
                
                # Rang
                item = QTableWidgetItem(str(rank))
                if rank == 1:
                    item.setBackground(QColor(144, 238, 144))  # Vert pour 1er
                elif rank == 2:
                    item.setBackground(QColor(211, 211, 211))  # Gris pour 2ème
                table.setItem(row, 0, item)
                
                # Symbole
                table.setItem(row, 1, QTableWidgetItem(sym))
                
                # Domaine
                table.setItem(row, 2, QTableWidgetItem(data['Domaine']))
                
                # Score
                item = QTableWidgetItem(f"{data['Score']:.2f}")
                item.setData(Qt.EditRole, data['Score'])
                table.setItem(row, 3, item)
                
                # Score/Seuil
                item = QTableWidgetItem(f"{data['Score/Seuil']:.2f}")
                item.setData(Qt.EditRole, data['Score/Seuil'])
                table.setItem(row, 4, item)
                
                # Fiabilité
                item = QTableWidgetItem(f"{data['Fiabilité (%)']:.1f}%")
                item.setData(Qt.EditRole, data['Fiabilité (%)'])
                table.setItem(row, 5, item)
                
                # Nb Trades
                item = QTableWidgetItem(str(data['Nb Trades']))
                item.setData(Qt.EditRole, data['Nb Trades'])
                table.setItem(row, 6, item)
                
                # Gagnants
                item = QTableWidgetItem(str(data['Gagnants']))
                item.setData(Qt.EditRole, data['Gagnants'])
                table.setItem(row, 7, item)
                
                # RSI
                item = QTableWidgetItem(f"{data['RSI']:.1f}")
                item.setData(Qt.EditRole, data['RSI'])
                table.setItem(row, 8, item)
                
                # Prix
                item = QTableWidgetItem(f"${data['Prix']:.2f}")
                item.setData(Qt.EditRole, data['Prix'])
                table.setItem(row, 9, item)
                
                # EBITDA ✅ Ajout avec couleurs
                ebitda = data.get('EBITDA Yield (%)', 0.0)
                item = QTableWidgetItem(f"{ebitda:.2f}")
                item.setData(Qt.EditRole, ebitda)
                # Appliquer les couleurs selon les seuils
                if ebitda > 15:
                    item.setForeground(QColor(0, 128, 0))  # Vert foncé
                elif ebitda > 8:
                    item.setForeground(QColor(34, 139, 34))  # Vert
                elif ebitda > 0:
                    item.setForeground(QColor(255, 165, 0))  # Orange
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge
                table.setItem(row, 10, item)
                
                # Gain
                color = Qt.green if data['Gain ($)'] > 0 else Qt.red if data['Gain ($)'] < 0 else Qt.white
                item = QTableWidgetItem(f"${data['Gain ($)']:.2f}")
                item.setData(Qt.EditRole, data['Gain ($)'])
                item.setBackground(color)
                table.setItem(row, 11, item)
                
                # Consensus
                table.setItem(row, 12, QTableWidgetItem(data['Consensus']))
                
                # Pertinence
                item = QTableWidgetItem(f"{pertinence:.1f}")
                item.setData(Qt.EditRole, pertinence)
                item.setBackground(Qt.yellow)
                table.setItem(row, 13, item)
            
            table.setSortingEnabled(True)
            table.setMinimumHeight(300)
            
            # Ajouter un résumé
            summary = QLabel(f"📊 Comparaison de {len(sorted_symbols)} symbole(s) | 🥇 Meilleur: {sorted_symbols[0]} (Pertinence: {pertinence_scores[sorted_symbols[0]]:.1f})")
            summary.setStyleSheet("background-color: #e3f2fd; padding: 6px; border-radius: 4px; font-weight: bold;")
            
            self.comparison_results_layout.addWidget(summary)
            self.comparisons_layout.addWidget(table)
            
        except Exception as e:
            print(f"❌ Erreur _generate_comparison_table: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"Erreur: {e}")
            self.comparison_results_layout.addWidget(error_label)
    
    def _generate_historical_comparison_table(self, symbols_to_compare, historical_date):
        """
        Génère un tableau de comparaison historique avec analyse complète.
        Télécharge intelligemment 18 mois de données pour le backtest annuel.
        Évite les retéléchargements en utilisant un cache intelligent.
        """
        try:
            from datetime import datetime, timedelta
            import pandas as pd
            import yfinance as yf
            from pathlib import Path
            
            # Créer un répertoire pour le cache historique
            cache_dir = Path("data_cache/historical")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Afficher le chargement
            loading_label = QLabel(f"⏳ Téléchargement intelligent des données (18 mois) pour {historical_date}...")
            loading_label.setStyleSheet("font-size: 11px; color: blue; padding: 10px;")
            self.comparison_results_layout.addWidget(loading_label)
            QApplication.processEvents()
            
            # Convertir la date
            target_date = datetime.strptime(historical_date, "%Y-%m-%d")
            today = datetime.now()
            
            # Vérifier que c'est une date passée
            if target_date >= today:
                QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une date passée")
                loading_label.deleteLater()
                return
            
            # Période à télécharger : 18 mois avant la date cible
            # (12 mois de backtest + 6 mois pour les indicateurs)
            dl_start_date = target_date - timedelta(days=550)  # ~18 mois
            dl_end_date = target_date + timedelta(days=1)  # Inclure la date cible
            
            # Stocker les données historiques et actuelles
            historical_data = {}
            actual_performance = {}
            
            # Fonction interne pour gérer le cache intelligent
            def get_or_download_data(symbol, start, end):
                """Récupère du cache ou télécharge avec gestion intelligente"""
                cache_file = cache_dir / f"{symbol}_hist.pkl"
                
                # Vérifier si le cache existe et est suffisamment complet
                if cache_file.exists():
                    try:
                        cached_df = pd.read_pickle(cache_file)
                        cached_meta = cache_dir / f"{symbol}_hist_meta.txt"
                        
                        # Vérifier les metadata (période couverte)
                        if cached_meta.exists():
                            with open(cached_meta, 'r') as f:
                                meta = f.read().strip()
                                cached_start, cached_end = meta.split('|')
                                cached_start = datetime.strptime(cached_start, "%Y-%m-%d")
                                cached_end = datetime.strptime(cached_end, "%Y-%m-%d")
                                
                                # Si le cache couvre la période requise
                                if cached_start <= start and cached_end >= end:
                                    print(f"✓ Cache valide pour {symbol}")
                                    return cached_df
                    except Exception as e:
                        print(f"⚠️ Erreur lecture cache {symbol}: {e}")
                
                # Télécharger les données manquantes
                print(f"📥 Téléchargement {symbol} ({start.strftime('%Y-%m-%d')} à {end.strftime('%Y-%m-%d')})")
                try:
                    df = yf.download(
                        symbol, 
                        start=start.strftime("%Y-%m-%d"), 
                        end=end.strftime("%Y-%m-%d"), 
                        progress=False,
                        timeout=30
                    )
                    
                    if not df.empty:
                        # Sauvegarder en cache avec metadata
                        df.to_pickle(cache_file)
                        with open(cache_dir / f"{symbol}_hist_meta.txt", 'w') as f:
                            f.write(f"{start.strftime('%Y-%m-%d')}|{end.strftime('%Y-%m-%d')}")
                        print(f"💾 Cache sauvegardé pour {symbol}")
                        return df
                except Exception as e:
                    print(f"❌ Erreur téléchargement {symbol}: {e}")
                    return None
                
                return None
            
            # Télécharger les données pour tous les symboles
            for idx, symbol in enumerate(symbols_to_compare):
                try:
                    # Mettre à jour le label de progression
                    loading_label.setText(
                        f"⏳ Traitement {symbol} ({idx+1}/{len(symbols_to_compare)}) - "
                        f"Téléchargement intelligent..."
                    )
                    QApplication.processEvents()
                    
                    # Récupérer/télécharger les données
                    df = get_or_download_data(symbol, dl_start_date, dl_end_date)
                    
                    if df is None or df.empty:
                        print(f"⚠️ Pas de données pour {symbol}")
                        continue
                    
                    # Normaliser l'index
                    if df.index.name is None or df.index.name != 'Date':
                        df = df.reset_index()
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                    
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    
                    # Trouver le prix à la date cible (ou le plus proche)
                    closest_date = df.index[df.index <= target_date].max() if any(df.index <= target_date) else None
                    
                    if closest_date is None:
                        print(f"⚠️ Pas de données pour {symbol} avant {historical_date}")
                        continue
                    
                    # Données à la date historique et aujourd'hui
                    historical_price = df.loc[closest_date]['Close']
                    current_price = df.iloc[-1]['Close']
                    
                    # Calculer la performance réelle (%)
                    performance_pct = ((current_price - historical_price) / historical_price) * 100
                    
                    # Historique jusqu'à la date cible
                    historical_idx = df.index.get_loc(closest_date)
                    hist_df = df.iloc[:historical_idx+1].copy()
                    
                    # === CALCUL DES INDICATEURS À LA DATE HISTORIQUE ===
                    
                    # 1. RSI (14 jours)
                    hist_rsi = self._calculate_rsi(hist_df['Close'], period=14)
                    
                    # 2. MACD
                    hist_macd = self._calculate_macd(hist_df['Close'])
                    
                    # 3. Bandes de Bollinger (20 jours)
                    hist_bb = self._calculate_bollinger_bands(hist_df['Close'], period=20)
                    
                    # 4. Volume Relatif (20 jours)
                    if len(hist_df) >= 20:
                        avg_vol = hist_df['Volume'].rolling(window=20).mean().iloc[-1]
                        current_vol = hist_df['Volume'].iloc[-1]
                        vol_rel = (current_vol / avg_vol) if avg_vol > 0 else 1.0
                    else:
                        vol_rel = 1.0
                    
                    # 5. Tendance de prix (30 jours)
                    if len(hist_df) >= 30:
                        price_trend = hist_df['Close'].iloc[-30:].pct_change().mean() * 100
                    else:
                        price_trend = 0.0
                    
                    # 6. Volatilité (20 jours)
                    if len(hist_df) >= 20:
                        volatility = hist_df['Close'].pct_change().rolling(window=20).std().iloc[-1] * 100
                    else:
                        volatility = 0.0
                    
                    # === CALCUL DU SCORE DE PERTINENCE HISTORIQUE ===
                    
                    # Score basé sur les indicateurs
                    rsi_score = 0
                    if hist_rsi < 30:
                        rsi_score = 25  # Survendu (bon signal d'achat)
                    elif hist_rsi > 70:
                        rsi_score = 0   # Suracheté (mauvais)
                    else:
                        rsi_score = 15  # Neutre
                    
                    # Score MACD
                    macd_score = 15 if hist_macd > 0 else 0
                    
                    # Score Bollinger Bands
                    bb_score = 10 if hist_bb else 0
                    
                    # Score Volume
                    vol_score = 10 if vol_rel > 1.2 else 5
                    
                    # Score Tendance
                    trend_score = 15 if price_trend > 0 else 5
                    
                    # Score Volatilité (haute volatilité = plus d'opportunité)
                    vol_std_score = 10 if volatility > 2.0 else 5
                    
                    # Score basé sur performance réelle (validation)
                    actual_score = min(25, max(0, performance_pct / 10))  # Max 25 points
                    
                    # Total
                    pertinence_score = (rsi_score + macd_score + bb_score + 
                                       vol_score + trend_score + vol_std_score + actual_score)
                    
                    historical_data[symbol] = {
                        'Date': closest_date.strftime("%Y-%m-%d"),
                        'Prix Historique': float(historical_price),
                        'Prix Actuel': float(current_price),
                        'Performance (%)': float(performance_pct),
                        'RSI': float(hist_rsi),
                        'MACD': float(hist_macd),
                        'Volume Rel': float(vol_rel),
                        'Tendance (%)': float(price_trend),
                        'Volatilité (%)': float(volatility),
                        'Pertinence': float(pertinence_score)
                    }
                    
                    actual_performance[symbol] = performance_pct
                    
                except Exception as e:
                    print(f"❌ Erreur pour {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Supprimer le label de chargement
            loading_label.deleteLater()
            
            if not historical_data:
                error_label = QLabel("❌ Aucune donnée historique disponible pour les symboles sélectionnés")
                error_label.setStyleSheet("color: red; padding: 10px;")
                self.comparison_results_layout.addWidget(error_label)
                return
            
            # Trier par performance réelle (meilleure performance = meilleur)
            sorted_symbols = sorted(historical_data.keys(), key=lambda x: actual_performance[x], reverse=True)
            
            # Créer le tableau
            table = QTableWidget()
            table.setColumnCount(11)
            table.setRowCount(len(sorted_symbols))
            table.setHorizontalHeaderLabels([
                'Rang', 'Symbole', 'Date', 'Prix Historique', 'Prix Actuel',
                'Performance (%)', 'RSI', 'MACD', 'Vol. Rel', 'Volatilité (%)', 'Avis'
            ])
            
            for row, symbol in enumerate(sorted_symbols):
                data = historical_data[symbol]
                
                # Rang
                item = QTableWidgetItem(str(row + 1))
                if row == 0:
                    item.setBackground(QColor(144, 238, 144))  # Vert pour 1er
                elif row == 1:
                    item.setBackground(QColor(211, 211, 211))  # Gris pour 2ème
                table.setItem(row, 0, item)
                
                # Symbole
                table.setItem(row, 1, QTableWidgetItem(symbol))
                
                # Date d'analyse
                table.setItem(row, 2, QTableWidgetItem(data['Date']))
                
                # Prix historique
                item = QTableWidgetItem(f"${data['Prix Historique']:.2f}")
                item.setData(Qt.EditRole, data['Prix Historique'])
                table.setItem(row, 3, item)
                
                # Prix actuel
                item = QTableWidgetItem(f"${data['Prix Actuel']:.2f}")
                item.setData(Qt.EditRole, data['Prix Actuel'])
                table.setItem(row, 4, item)
                
                # Performance
                perf = data['Performance (%)']
                item = QTableWidgetItem(f"{perf:+.1f}%")
                item.setData(Qt.EditRole, perf)
                if perf > 0:
                    item.setForeground(QColor(0, 128, 0))  # Vert
                else:
                    item.setForeground(QColor(255, 0, 0))  # Rouge
                table.setItem(row, 5, item)
                
                # RSI
                rsi = data['RSI']
                item = QTableWidgetItem(f"{rsi:.1f}")
                item.setData(Qt.EditRole, rsi)
                if rsi < 30 or rsi > 70:
                    item.setForeground(QColor(255, 140, 0))  # Orange (extrême)
                table.setItem(row, 6, item)
                
                # MACD
                macd = data['MACD']
                item = QTableWidgetItem(f"{macd:+.4f}")
                item.setData(Qt.EditRole, macd)
                table.setItem(row, 7, item)
                
                # Volume Relatif
                vol = data['Volume Rel']
                item = QTableWidgetItem(f"{vol:.2f}x")
                item.setData(Qt.EditRole, vol)
                table.setItem(row, 8, item)
                
                # Volatilité
                vol_std = data['Volatilité (%)']
                item = QTableWidgetItem(f"{vol_std:.2f}%")
                item.setData(Qt.EditRole, vol_std)
                table.setItem(row, 9, item)
                
                # Avis (justification)
                avis = self._generate_historical_verdict(data)
                table.setItem(row, 10, QTableWidgetItem(avis))
            
            table.setSortingEnabled(True)
            table.setMinimumHeight(350)
            table.resizeColumnsToContents()
            
            # Résumé et statistiques
            best_symbol = sorted_symbols[0]
            best_perf = actual_performance[best_symbol]
            worst_symbol = sorted_symbols[-1]
            worst_perf = actual_performance[worst_symbol]
            avg_perf = sum(actual_performance.values()) / len(actual_performance)
            winners = sum(1 for p in actual_performance.values() if p > 0)
            
            summary = QLabel(
                f"📈 Historique ({historical_date}) | "
                f"🥇 {best_symbol} ({best_perf:+.1f}%) | "
                f"📊 Moyenne: {avg_perf:+.1f}% | "
                f"✓ {winners}/{len(actual_performance)} gagnants"
            )
            summary.setStyleSheet("background-color: #fff9c4; padding: 8px; border-radius: 4px; font-weight: bold;")
            
            info_label = QLabel(
                f"💡 Analyse complète : 18 mois téléchargés intelligemment (12 mois backtest + 6 mois indicateurs). "
                f"Le cache est utilisé pour éviter les retéléchargements. "
                f"Les symboles sont classés par performance réelle depuis {historical_date}."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("background-color: #e1f5fe; padding: 6px; border-radius: 4px; font-size: 9px;")
            
            self.comparison_results_layout.addWidget(summary)
            self.comparisons_layout.addWidget(info_label)
            self.comparisons_layout.addWidget(table)
            
        except Exception as e:
            print(f"❌ Erreur _generate_historical_comparison_table: {e}")
            import traceback
            traceback.print_exc()
            error_label = QLabel(f"Erreur: {e}")
            self.comparisons_layout.addWidget(error_label)
    
    def _calculate_rsi(self, prices, period=14):
        """Calcule le RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return float(macd.iloc[-1]) if not macd.empty else 0.0
    
    def _calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """Calcule les Bandes de Bollinger et retourne si le prix est en extrême"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        current_price = prices.iloc[-1]
        # Retourne True si le prix touche un extrême
        return current_price >= upper_band.iloc[-1] or current_price <= lower_band.iloc[-1]
    
    def _generate_historical_verdict(self, data):
        """Génère un avis pointu basé sur les indicateurs"""
        avis_parts = []
        
        rsi = data['RSI']
        if rsi < 30:
            avis_parts.append("↓ Survendu")
        elif rsi > 70:
            avis_parts.append("↑ Suracheté")
        
        if data['MACD'] > 0:
            avis_parts.append("▲ MACD+")
        else:
            avis_parts.append("▼ MACD-")
        
        if data['Volume Rel'] > 1.5:
            avis_parts.append("📈 Vol▲")
        
        perf = data['Performance (%)']
        if perf > 20:
            avis_parts.append("🚀 +20%")
        elif perf < -10:
            avis_parts.append("📉 -10%")
        
        return " | ".join(avis_parts) if avis_parts else "Neutre"

    def refresh_random_symbols(self):
        """Charger 30 symboles ALÉATOIRES depuis popular_symbols."""
        try:
            import random
            all_symbols = self.popular_symbols_data
            if not all_symbols:
                QMessageBox.warning(self, "Erreur", "Aucun symbole populaire disponible")
                return
            
            random_30 = random.sample(all_symbols, min(30, len(all_symbols)))
            
            self.random_list.clear()
            for sym in random_30:
                item = QListWidgetItem(sym)
                item.setData(Qt.UserRole, sym)
                self.random_list.addItem(item)
            
            self.random_label.setText(f"🎲 Aléatoires\n({len(random_30)} symboles)")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les symboles aléatoires: {e}")
    
    def load_recent_symbols(self):
        """Charger les 30 derniers SYMBOLES AJOUTÉS depuis la base de données."""
        try:
            if SYMBOL_MANAGER_AVAILABLE:
                # Récupérer les 30 derniers symboles directement de la BDD, triés par date d'ajout
                recent = get_recent_symbols(limit=30, active_only=True)
            else:
                # Fallback: si symbol_manager n'est pas disponible, utiliser les listes en fichier
                recent = []
                seen = set()

                for sym in self.mes_symbols_data:
                    if sym not in seen:
                        recent.append(sym)
                        seen.add(sym)

                for sym in reversed(self.popular_symbols_data):
                    if sym not in seen and len(recent) < 30:
                        recent.append(sym)
                        seen.add(sym)

                recent = recent[:30]

            self.recent_list.clear()
            for sym in recent:
                item = QListWidgetItem(sym)
                item.setData(Qt.UserRole, sym)
                self.recent_list.addItem(item)

            self.recent_label.setText(f"🔥 Récents\n({len(recent)} symboles)")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les symboles récents: {e}")
    
    def open_optimization_window(self):
        """Ouvrir la fenêtre dédiée à l'optimisation hybride."""
        try:
            from optimization_window import OptimizationWindow
            self.optim_window = OptimizationWindow(parent=self)
            self.optim_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir la fenêtre d'optimisation: {e}")

    def _select_all_items(self, list_widget):
        """Sélectionner tous les éléments d'une QListWidget"""
        list_widget.selectAll()

    def _get_clean_columns_and_data(self):
        """Filtrer les colonnes vides ou contenant uniquement des 0"""
        if not self.current_results:
            return [], []
        
        # Récupérer toutes les colonnes
        all_columns = []
        for col in range(self.merged_table.columnCount()):
            header = self.merged_table.horizontalHeaderItem(col)
            if header:
                all_columns.append(header.text())
        
        if not all_columns:
            all_columns = list(self.current_results[0].keys())
        
        # Identifier les colonnes à conserver
        valid_columns = []
        for col in all_columns:
            has_valid_data = False
            for result in self.current_results:
                value = result.get(col, '')
                # Vérifier si la colonne a des données non vides et pas 0
                if value and value != '' and value != 0 and value != '0' and value != 'N/A':
                    try:
                        float_val = float(value) if isinstance(value, str) else value
                        if float_val != 0:
                            has_valid_data = True
                            break
                    except (ValueError, TypeError):
                        if value not in ('', 'N/A', 0, '0'):
                            has_valid_data = True
                            break
            
            if has_valid_data:
                valid_columns.append(col)
        
        print(f"📊 Colonnes filtrées: {len(all_columns)} → {len(valid_columns)} (suppression de {len(all_columns) - len(valid_columns)} colonnes vides/zéro)")
        return valid_columns, self.current_results

    def export_results_csv(self):
        """Exporter les résultats actuels en fichier CSV avec auto-save dans Results"""
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "Erreur", "Aucun résultat à exporter. Veuillez d'abord lancer une analyse.")
            return
        
        import csv
        from datetime import datetime
        from pathlib import Path
        
        try:
            # Nettoyer les colonnes vides/zéro
            clean_columns, data = self._get_clean_columns_and_data()
            
            if not clean_columns:
                QMessageBox.warning(self, "Erreur", "Aucune colonne avec données valides à exporter")
                return
            
            # Créer le dossier Results avec chemins relatifs
            results_dir = Path(__file__).parent.parent / "Results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Générer le nom du fichier avec timestamp
            filename = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = results_dir / filename
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=clean_columns)
                writer.writeheader()
                
                for result in data:
                    # Créer une ligne en prenant les valeurs existantes
                    row = {}
                    for field in clean_columns:
                        row[field] = result.get(field, '')
                    writer.writerow(row)
            
            QMessageBox.information(self, "Succès", f"✅ Résultats exportés avec succès!\n\nDossier: Results\nFichier: {filename}\nColonnes: {len(clean_columns)}")
            print(f"✅ Résultats exportés en CSV: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export CSV:\n{e}")
            print(f"❌ Erreur export CSV: {e}")
            import traceback
            traceback.print_exc()

    def export_results_excel(self):
        """Exporter les résultats actuels en fichier Excel avec auto-save dans Results"""
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "Erreur", "Aucun résultat à exporter. Veuillez d'abord lancer une analyse.")
            return
        
        from datetime import datetime
        from pathlib import Path
        
        try:
            # Vérifier si openpyxl est disponible
            try:
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            except ImportError:
                QMessageBox.warning(
                    self,
                    "Bibliothèque manquante",
                    "openpyxl n'est pas installé.\n\nVeuillez installer avec:\npip install openpyxl\n\nOu utiliser l'export CSV à la place."
                )
                return
            
            # Créer le dossier Results avec chemins relatifs
            results_dir = Path(__file__).parent.parent / "Results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Générer le nom du fichier avec timestamp
            filename = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            file_path = results_dir / filename
            
            # Nettoyer les colonnes vides/zéro
            columns, data = self._get_clean_columns_and_data()
            
            if not columns:
                QMessageBox.warning(self, "Erreur", "Aucune colonne avec données valides à exporter")
                return
            
            # Créer un classeur
            workbook = openpyxl.Workbook()
            worksheet = workbook.active
            worksheet.title = "Résultats"
            
            # En-têtes
            for col_idx, col_name in enumerate(columns, 1):
                cell = worksheet.cell(row=1, column=col_idx)
                cell.value = col_name
                # Style d'en-tête
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Données
            for row_idx, result in enumerate(data, 2):
                for col_idx, col_name in enumerate(columns, 1):
                    value = result.get(col_name, '')
                    cell = worksheet.cell(row=row_idx, column=col_idx)
                    cell.value = value
                    
                    # Formatage selon le type
                    if isinstance(value, (int, float)):
                        cell.alignment = Alignment(horizontal="right")
                    
                    # Coloration selon la valeur pour certaines colonnes
                    if col_name == "Signal":
                        if value == "ACHAT":
                            cell.font = Font(color="00B050")  # Vert
                        elif value == "VENTE":
                            cell.font = Font(color="FF0000")  # Rouge
                    elif col_name in ["Fiabilite", "Rev. Growth (%)", "EBITDA Yield (%)"]:
                        try:
                            val = float(value) if value else 0
                            if val > 0:
                                cell.font = Font(color="00B050")  # Vert
                            elif val < 0:
                                cell.font = Font(color="FF0000")  # Rouge
                        except:
                            pass
                    
                    # Bordures légères
                    thin_border = Border(
                        left=Side(style='thin'),
                        right=Side(style='thin'),
                        top=Side(style='thin'),
                        bottom=Side(style='thin')
                    )
                    cell.border = thin_border
            
            # Ajuster les largeurs de colonnes
            for col_idx, col_name in enumerate(columns, 1):
                max_length = max(
                    len(str(col_name)),
                    max(len(str(r.get(col_name, ''))) for r in data) if data else 0
                )
                adjusted_width = min(max_length + 2, 50)  # Max 50 pour lisibilité
                worksheet.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = adjusted_width
            
            # Geler la première ligne
            worksheet.freeze_panes = "A2"
            
            # Sauvegarder
            workbook.save(str(file_path))
            
            QMessageBox.information(self, "Succès", f"✅ Résultats exportés avec succès!\n\nDossier: Results\nFichier: {filename}\nColonnes: {len(columns)}")
            print(f"✅ Résultats exportés en Excel: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'export Excel:\n{e}")
            print(f"❌ Erreur export Excel: {e}")
            import traceback
            traceback.print_exc()

    def export_results_pdf(self):
        """Exporter tous les graphiques de l'analyse en PDF avec infos du tableau intégrées"""
        try:
            if not hasattr(self, 'current_results') or not self.current_results:
                QMessageBox.warning(self, "Erreur", "Aucun résultat à exporter. Veuillez d'abord lancer une analyse.")
                return
            
            # Construire les colonnes PDF directement depuis les clés des données
            # (éviter _get_clean_columns_and_data qui utilise les headers UI avec \n)
            all_data_keys = []
            for result in self.current_results:
                for key in result.keys():
                    if key not in all_data_keys:
                        all_data_keys.append(key)
            
            # Filtrer : garder uniquement les colonnes qui ont au moins 1 valeur non-vide
            clean_columns = []
            for col in all_data_keys:
                for result in self.current_results:
                    value = result.get(col, '')
                    if value is not None and value != '' and value != 'N/A':
                        try:
                            if isinstance(value, (int, float)) and value != 0:
                                clean_columns.append(col)
                                break
                            elif isinstance(value, str) and value not in ('0', '0.0', '0.00', ''):
                                clean_columns.append(col)
                                break
                        except (ValueError, TypeError):
                            clean_columns.append(col)
                            break
            
            if not clean_columns:
                QMessageBox.warning(self, "Erreur", "Aucune colonne avec données valides à exporter")
                return
            
            print(f"📊 PDF Export: {len(all_data_keys)} clés données → {len(clean_columns)} colonnes retenues")
            
            # Importer le générateur de PDF
            from pdf_generator import PDFReportGenerator
            
            # Créer le générateur et exporter
            generator = PDFReportGenerator()
            min_hold_days = self.min_hold_days_spin.value() if hasattr(self, 'min_hold_days_spin') else 7
            report_meta = {
                'min_holding_days': int(min_hold_days),
            }
            pdf_path = generator.export_pdf(
                self.plots_layout,
                self.current_results,
                clean_columns,
                report_meta=report_meta,
            )
            
            if pdf_path:
                # Extraire juste le nom du fichier pour le message
                from pathlib import Path
                filename = Path(pdf_path).name
                QMessageBox.information(
                    self, 
                    "Succès", 
                    f"✅ PDF créé avec succès!\n\n"
                    f"Dossier: Results\nFichier: {filename}"
                )
            else:
                QMessageBox.critical(self, "Erreur", "Impossible de créer le PDF")
                
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la création du PDF:\n{e}")
            print(f"❌ Erreur export PDF: {e}")
            import traceback
            traceback.print_exc()


# Ensure the application only launches when run directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Stock Analysis Tool")
    window.show()
    sys.exit(app.exec_())

    #TODO:
    # - Ajouter un bouton pour exporter les resultats (CSV/Excel)
    # - Ajouter dates d'annonces / résultats dans les signaux (ex: earnings date)
    # - harmoniser l'affichage des plots (embedded + external)
    # - améliorer le threading / gestion des erreurs
    # - Ajouter le earning dates et tous les autres nouveaux criteres a l'analyse et au backtest
    # - Ajouter un bouton pour choisir si backup des resultats avant analyse ou pas