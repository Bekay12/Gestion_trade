from qsi import analyse_et_affiche, analyse_signaux_populaires, get_trading_signal, popular_symbols, mes_symbols, period
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import time
import csv

test_symbols = list(dict.fromkeys([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "BABA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CSCO", "WMT", "VZ", "KO", "SGMT",
     "ZM", "SHOP", "SNAP", "PFE", "TGT", "CVS", "WFC", "RHM.DE", "SAP.DE", "BAS.DE", "ALV.DE", "BMW.DE", "VOW3.DE", "SMTC", "ZS", "ZTS",
    "DTE.DE", "DBK.DE", "LHA.DE", "FME.DE", "BAYN.DE", "LIN.DE", "ENR.DE", "VNA.DE", "1COV.DE", "FRE.DE", "HEN3.DE", "HEI.DE", "RWE.DE", "VOW.DE", "GLW", "TMO",
     "AMC", "BB", "NOK", "RBLX", "PLTR", "FSLY", "CRWD", "OKTA", "Z", "DOCU", "PINS", "SPOT", "LYFT",
     "AIR.PA", "BNP.PA", "SAN.PA", "ENGI.PA", "CAP.PA", "WELL", "O", "VICI", "ETOR", "ABR", "MOH.BE", "KSS",
    "VEEV", "LEN", "PHM", "DHI", "KBH", "TOL", "NVR", "RMAX", "BURL", "TJX", "ROST", "VYGR","TLRY", "FSK", "PSEC",
    "MAIN", "ARCC", "ORC", "GBDC", "FDUS", "ALHF.PA", "PBR",
    "GLD", "SLV", "GDX", "GDXJ", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLC", "XLI", "XLB", "XLC", "XLV", "XLI", "XLP", "XLY","XLK",
     "PHYS", "FNV.TO", "WDO.TO", "BOE", "JOBY", "LAC", "PLL", "ALB", "SQM", "RIOT", "MARA", "HUT", "BITF", "VKTX", "CRSR", "PFC.L", "OPEN", "FVRR"
    ]))

analyse_signaux_populaires(test_symbols, mes_symbols, period, plot_all=True)