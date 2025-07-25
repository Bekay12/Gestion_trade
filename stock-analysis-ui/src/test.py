from qsi import analyse_et_affiche, analyse_signaux_populaires, get_trading_signal, popular_symbols, mes_symbols, period
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import time
import csv

test_symbols = list(dict.fromkeys([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "BABA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CSCO",
    "SHOP", "SNAP", "PFE", "TGT", "CVS", "WFC", "RHM.DE", "SAP.DE", "BAS.DE", "ALV.DE", "BMW.DE", "VOW3.DE", "SMTC", "RWE.DE",
    "DTE.DE", "DBK.DE", "LHA.DE", "FME.DE", "BAYN.DE", "LIN.DE", "ENR.DE", "VNA.DE", "1COV.DE", "FRE.DE", "HEN3.DE", "HEI.DE",
    "AMC", "BB", "NOK", "RBLX", "PLTR", "FSLY", "CRWD", "OKTA", "SGMT", "DOCU", "PINS", "SPOT", "LYFT", "VOW.DE", "GLW", "TMO",
    "AIR.PA", "BNP.PA", "SAN.PA", "ENGI.PA", "CAP.PA", "WELL", "FVRR", "VICI", "ETOR", "ABR", "MOH.BE", "KSS","XLK", "PFC.L",
    "VEEV", "LEN", "PHM", "DHI", "KBH", "TOL", "NVR", "RMAX", "BURL", "TJX", "ROST", "VYGR","TLRY", "FSK", "PSEC", "OPEN", "O",
    "GLD", "SLV", "GDX", "GDXJ", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLC", "XLI", "XLB", "XLC", "XLV", "XLI", "XLP", "XLY", "Z",
    "PHYS", "FNV.TO", "WDO.TO", "BOE", "JOBY", "LAC", "PLL", "ALB", "SQM", "RIOT", "MARA", "HUT", "BITF", "VKTX", "CRSR", "KO"
    ]))

analyse_signaux_populaires(test_symbols, mes_symbols, period, plot_all=True)