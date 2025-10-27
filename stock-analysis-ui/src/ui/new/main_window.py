from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QInputDialog, QListWidget, QListWidgetItem,
    QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QComboBox, QHeaderView
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import sys
import os
import math

# Ensure project `src` root is on sys.path
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)
from qsi import analyse_signaux_populaires, popular_symbols, mes_symbols, period

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
        self.popular_list.setMaximumHeight(200)
        for s in popular_symbols:
            if s:
                self.popular_list.addItem(QListWidgetItem(s))
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
        self.mes_list.setMaximumHeight(200)
        for s in mes_symbols:
            if s:
                self.mes_list.addItem(QListWidgetItem(s))
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
        
        # Bouton d'analyse
        self.analyze_button = QPushButton("Analyser mouvements fiables (populaires)")
        self.analyze_button.clicked.connect(self.analyze_popular_signals)
        self.layout.addWidget(self.analyze_button)
        
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
            "Volume (décroissant)"
        ])
        self.sort_combo.currentIndexChanged.connect(self.sort_results)
        sort_layout.addWidget(self.sort_combo)
        self.layout.addLayout(sort_layout)
        
        # Tableau de résultats
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Symbole", "Signal", "Score", "Prix", "Tendance",
            "RSI", "Volume moyen", "Domaine"
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
            list_widget.addItem(QListWidgetItem(symbol))
            try:
                from qsi import save_symbols_to_txt
                symbols = [list_widget.item(i).text() for i in range(list_widget.count())]
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
            symbols = [list_widget.item(i).text() for i in range(list_widget.count())]
            save_symbols_to_txt(symbols, filename)
        except Exception:
            pass

    def show_selected(self, list_widget):
        items = list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Veuillez sélectionner au moins un symbole à afficher")
            return
        symbols = [it.text() for it in items]
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

    def on_analysis_progress(self, message):
        if self.progress:
            self.progress.setLabelText(message)
            QApplication.processEvents()

    def on_analysis_complete(self, result):
        self.analyze_button.setEnabled(True)
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
            7: ('Volume moyen', True)    # Volume décroissant
        }
        
        if index in sort_options:
            key, reverse = sort_options[index]
            self.current_results.sort(key=lambda x: x[key], reverse=reverse)
            self.update_results_table()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())