from qsi import load_symbols_from_txt, analyse_signaux_populaires, modify_symbols_file,extract_best_parameters, period
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import time
import csv

test_symbols = load_symbols_from_txt("test_symbols.txt")
mes_symbols = load_symbols_from_txt("mes_symbols.txt")

print("\nExtraction des meilleurs paramètres depuis le CSV:")
best_parameters = extract_best_parameters()
print("\nDictionnaire des meilleurs paramètres:")
print("{")
for sector, (coeffs, thresholds) in best_parameters.items():
        print(f"    '{sector}': (coefficients={coeffs}, thresholds={thresholds}),")
print("}")

analyse_signaux_populaires(test_symbols, mes_symbols, period, plot_all=True)