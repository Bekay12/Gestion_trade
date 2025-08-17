from qsi import load_symbols_from_txt, analyse_signaux_populaires, modify_symbols_file,preload_cache, period
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import time
import csv

test_symbols = load_symbols_from_txt("test_symbols.txt")
mes_symbols = load_symbols_from_txt("mes_symbols.txt")
modify_symbols_file("optimisation_symbols.txt", ["BATS.L", "ULVR.L"],action='add')  # Exclure CDNS de l'analyse
preload_cache(test_symbols + mes_symbols, period)
analyse_signaux_populaires(test_symbols, mes_symbols, period, plot_all=True)