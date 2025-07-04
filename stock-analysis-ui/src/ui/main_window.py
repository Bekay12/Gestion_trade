from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QInputDialog
import sys
import os
import math
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from qsi import analyse_et_affiche, analyse_signaux_populaires, popular_symbols, mes_symbols

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis Tool")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.title_label = QLabel("Welcome to the Stock Analysis Tool")
        self.layout.addWidget(self.title_label)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        self.layout.addWidget(self.symbol_input)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_stock)
        self.layout.addWidget(self.analyze_button)

        # Nouveau bouton pour signaux populaires
        self.popular_signals_button = QPushButton("Analyser mouvements fiables (populaires)")
        self.popular_signals_button.clicked.connect(self.analyze_popular_signals)
        self.layout.addWidget(self.popular_signals_button)

        # Champs pour symboles populaires
        self.popular_symbols_input = QLineEdit(", ".join(popular_symbols))
        self.popular_symbols_input.setPlaceholderText("Symboles populaires (ex: AAPL, MSFT, GOOG)")
        self.layout.addWidget(QLabel("Symboles populaires :"))
        self.layout.addWidget(self.popular_symbols_input)

        # Champs pour mes symboles
        self.mes_symbols_input = QLineEdit(", ".join(mes_symbols))
        self.mes_symbols_input.setPlaceholderText("Mes symboles (ex: TSLA, NVDA)")
        self.layout.addWidget(QLabel("Mes symboles :"))
        self.layout.addWidget(self.mes_symbols_input)

        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

    def analyze_stock(self):
        # Permet d'entrer plusieurs symboles séparés par des virgules
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()]
        if not symbols:
            self.result_label.setText("Please enter at least one stock symbol.")
            return

        group_size = 5
        n_groups = math.ceil(len(symbols) / group_size)
        for i in range(n_groups):
            group = symbols[i*group_size:(i+1)*group_size]
            # analyse_et_affiche doit accepter une liste de symboles
            # et afficher les graphiques sur une même figure
            plt.figure(figsize=(14, 5))
            analyse_et_affiche(group, period="12mo")
            plt.suptitle(f"Stocks {', '.join(group)}")
            plt.show()

        self.result_label.setText(f"Analyzed {len(symbols)} stock(s) in {n_groups} window(s).")

    def analyze_popular_signals(self):
        popular_symbols = [s.strip().upper() for s in self.popular_symbols_input.text().split(",") if s.strip()]
        mes_symbols = [s.strip().upper() for s in self.mes_symbols_input.text().split(",") if s.strip()]
        result = analyse_signaux_populaires(popular_symbols, mes_symbols)
        self.result_label.setText(str(result))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())