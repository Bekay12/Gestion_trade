# qsi.py - Analyse technique unifiée pour les actions avec MACD et RSI et gestion intelligente du cache

# Ce script télécharge les données boursières, calcule les indicateurs techniques et affiche

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import time
import csv
from matplotlib import dates as mdates
import logging
import warnings
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
sys.path.append("C:\\Users\\berti\\Desktop\\Mes documents\\Gestion_trade\\stock-analysis-ui\\src\\trading_c_acceleration")
from qsi_optimized import backtest_signals, extract_best_parameters

# Supprimer les avertissements FutureWarning de yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration du logger
logging.basicConfig(level=logging.INFO, filename='stock_analysis.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def save_to_evolutive_csv(signals, filename="signaux_trading.csv"):
    """
    Sauvegarde les signaux dans un CSV évolutif qui conserve l'historique
    - Crée le fichier s'il n'existe pas
    - Ajoute de nouveaux signaux
    - Met à jour les signaux existants
    - Conserve l'historique des changements
    """
    if not signals:
        return

    # Préparer les données avec le nouveau champ de fiabilité
    header = [
        'Symbole', 'Signal', 'Score', 'Prix', 'Tendance',
        'RSI', 'Volume moyen', 'Domaine', 'Fiabilite', 'Detection_Time'
    ]

    rows = []
    for s in signals:
        # Formater la fiabilité
        fiabilite = s.get('Fiabilite', 'N/A')
        if isinstance(fiabilite, float):
            fiabilite = f"{fiabilite:.1f}%"

        rows.append([
            s['Symbole'],
            s['Signal'],
            f"{s['Score']:.2f}",
            f"{s['Prix']:.4f}",
            s['Tendance'],
            f"{s['RSI']:.2f}",
            f"{s['Volume moyen']:,.0f}",
            s['Domaine'],
            fiabilite,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

    # Création du DataFrame à partir des signaux actuels
    df_new = pd.DataFrame(signals)
    if df_new.empty:
        return

    # Ajout d'un timestamp pour le moment de détection
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new['detection_time'] = detection_time

    script_dir = Path(__file__).parent
    signals_dir = script_dir / "signaux"

    # Construire les chemins pour le fichier principal et l'archive
    file_path = signals_dir / filename

    # Vérifier si le fichier existe déjà
    if file_path.exists():
        try:
            # Lire l'historique existant
            df_old = pd.read_csv(file_path)
            # Fusionner les nouveaux signaux avec l'historique
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            # Supprimer les doublons en gardant la dernière version
            df_combined = df_combined.sort_values(
                by=['detection_time', 'Symbole', 'Fiabilite'],
                ascending=[True, False]
            )
            df_clean = df_combined.drop_duplicates(
                subset=['Symbole', 'Signal', 'Prix', 'RSI'],
                keep='first'
            )
        except Exception as e:
            print(f"⚠️ Erreur lecture CSV: {e}")
            df_clean = df_new
    else:
        df_clean = df_new

    # Sauvegarde avec vérification de la structure
    try:
        # Créer le dossier si nécessaire
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder avec date de mise à jour dans le nom
        timestamp = datetime.now().strftime("%Y%m%d")
        base_name = Path(filename).stem  # enlève l'extension
        archive_file = signals_dir / f"{base_name}_{timestamp}.csv"
        df_clean.to_csv(archive_file, index=False)

        # Sauvegarde principale
        df_clean.to_csv(filename, index=False)
        print(f"💾 Signaux sauvegardés: {filename} (archive: {archive_file})")

    except Exception as e:
        print(f"🚨 Erreur sauvegarde CSV: {e}")

from typing import Tuple, Dict, Union, List

def extract_best_parameters(csv_path: str = 'signaux/optimization_hist_4stp.csv') -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, float]]]:
    """
    Extrait les meilleurs coefficients et seuils pour chaque secteur à partir du CSV, basés sur le Gain_moy maximal.

    Args:
        csv_path (str): Chemin vers le CSV contenant l'historique d'optimisation.

    Returns:
        Dict[str, Tuple[Tuple[float, ...], Tuple[float, float]]]: Dictionnaire avec pour chaque secteur
        un tuple (coefficients, seuils), où coefficients est (a1, a2, ..., a8) et seuils est (Seuil_Achat, Seuil_Vente).
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("🚫 CSV vide, aucun paramètre extrait")
            return {}

        required_columns = ['Sector', 'Gain_moy', 'Success_Rate', 'Trades', 'Seuil_Achat', 'Seuil_Vente', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"🚫 Colonnes manquantes dans le CSV : {missing}")
            return {}

        # Trier par Gain_moy (descendant), Success_Rate (descendant), Trades (descendant), Timestamp (descendant)
        df_sorted = df.sort_values(by=['Sector', 'Gain_moy', 'Success_Rate', 'Trades', 'Timestamp'], ascending=[True, False, False, False, False])

        # Prendre la première entrée par secteur (la meilleure)
        best_params = df_sorted.groupby('Sector').first().reset_index()

        result = {}
        for _, row in best_params.iterrows():
            sector = row['Sector']
            coefficients = tuple(row[f'a{i+1}'] for i in range(8))
            thresholds = (row['Seuil_Achat'], row['Seuil_Vente'])
            gain_moy = row['Gain_moy']
            result[sector] = (coefficients, thresholds, gain_moy)
            # print(f"📊 Meilleurs paramètres pour {sector}: Coefficients={coefficients}, Seuils={thresholds}, Gain_moy={row['Gain_moy']:.2f}")

        return result

    except FileNotFoundError:
        print(f"🚫 Fichier CSV {csv_path} non trouvé")
        return {}
    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction des paramètres: {e}")
        return {}

def get_trading_signal(prices, volumes, domaine, domain_coeffs=None,
                      variation_seuil=-20, volume_seuil=100000, return_derivatives: bool = False, symbol: str = None):
    """Détermine les signaux de trading avec validation des données"""
    # Conversion explicite en Series scalaires
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()

    if len(prices) < 50:
        return "Données insuffisantes", None, None, None, None, None

    # Calcul des indicateurs
    macd, signal_line = calculate_macd(prices)
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema50 = prices.ewm(span=50, adjust=False).mean()
    ema200 = prices.ewm(span=200, adjust=False).mean() if len(prices) >= 200 else ema50

    # Validation des derniers points
    if len(macd) < 2 or len(rsi) < 1:
        return "Données récentes manquantes", None, None, None, None, None

    # CORRECTION 1: Conversion explicite en valeurs scalaires
    last_close = float(prices.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1]) if len(prices) >= 200 else last_ema50
    last_rsi = float(rsi.iloc[-1])
    last_macd = float(macd.iloc[-1])
    prev_macd = float(macd.iloc[-2])
    last_signal = float(signal_line.iloc[-1])
    prev_signal = float(signal_line.iloc[-2])
    prev_rsi = float(rsi.iloc[-2])
    delta_rsi = last_rsi - prev_rsi

    # Performance long terme
    variation_30j = ((last_close - float(prices.iloc[-30])) / float(prices.iloc[-30]) * 100) if len(prices) >= 30 else np.nan
    variation_180j = ((last_close - float(prices.iloc[-180])) / float(prices.iloc[-180]) * 100) if len(prices) >= 180 else np.nan

    # Calcul du volume moyen sur 30 jours
    if len(volumes) >= 30:
        volume_mean = float(volumes.rolling(window=30).mean().iloc[-1])
        volume_std = float(volumes.rolling(window=30).std().iloc[-1])
    else:
        volume_mean = float(volumes.mean()) if len(volumes) > 0 else 0.0
        volume_std = 0.0

    current_volume = float(volumes.iloc[-1])

    # Nouveaux indicateurs pour confirmation
    from ta.volatility import BollingerBands
    from ta.trend import ADXIndicator, IchimokuIndicator

    # Bollinger Bands
    bb = BollingerBands(close=prices, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_percent = (prices - bb_lower) / (bb_upper - bb_lower)
    last_bb_percent = float(bb_percent.iloc[-1]) if len(bb_percent) > 0 else 0.5

    # ADX (Force de la tendance)
    adx_indicator = ADXIndicator(high=prices, low=prices, close=prices, window=14)
    adx = adx_indicator.adx()
    last_adx = float(adx.iloc[-1]) if len(adx) > 0 else 0

    # Ichimoku Cloud (Tendance globale)
    ichimoku = IchimokuIndicator(high=prices, low=prices, window1=9, window2=26, window3=52)
    ichimoku_base = ichimoku.ichimoku_base_line()
    ichimoku_conversion = ichimoku.ichimoku_conversion_line()
    last_ichimoku_base = float(ichimoku_base.iloc[-1]) if len(ichimoku_base) > 0 else last_close
    last_ichimoku_conversion = float(ichimoku_conversion.iloc[-1]) if len(ichimoku_conversion) > 0 else last_close

    # Conditions d'achat optimisées
    is_macd_cross_up = prev_macd < prev_signal and last_macd > last_signal
    is_macd_cross_down = prev_macd > prev_signal and last_macd < last_signal
    is_volume_ok = volume_mean > volume_seuil
    is_variation_ok = not np.isnan(variation_30j) and variation_30j > variation_seuil

    # CORRECTION 2: Utilisation de valeurs scalaires pour les comparaisons
    ema_structure_up = last_close > last_ema20 > last_ema50 > last_ema200
    ema_structure_down = last_close < last_ema20 < last_ema50 < last_ema200

    # RSI dynamique
    rsi_cross_up = prev_rsi < 30 and last_rsi >= 30
    rsi_cross_mid = prev_rsi < 50 and last_rsi >= 50
    rsi_cross_down = prev_rsi > 65 and last_rsi <= 65
    rsi_ok = last_rsi < 75 and last_rsi > 40

    # Conditions supplémentaires pour confirmation
    strong_uptrend = (last_close > last_ichimoku_base) and (last_close > last_ichimoku_conversion)
    strong_downtrend = (last_close < last_ichimoku_base) and (last_close < last_ichimoku_conversion)
    adx_strong_trend = last_adx > 25  # Tendance forte

    # Momentum 10 jours
    momentum_10 = prices.pct_change(10)

    # Volatilité
    volatility = prices.pct_change().rolling(20).std()

    # CORRECTION 3: Conversion de la volatilité en scalaire
    if isinstance(volatility, pd.Series) and not volatility.empty:
        volatility = float(volatility.iloc[-1])
    else:
        volatility = 0.05

    # Ratio de Sharpe à court terme
    returns = prices.pct_change()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    score = 0
    default_coeffs = (1.75, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75)
    thresholds = (4.20, -0.5)

    best_params = extract_best_parameters()
    if domain_coeffs:
        coeffs = domain_coeffs.get(domaine, default_coeffs)
    else:
        if domaine in best_params:
            coeffs, thresholds, gain_moyen = best_params[domaine]
        else:
            coeffs = default_coeffs

    a1, a2, a3, a4, a5, a6, a7, a8 = coeffs
    m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0

    if adx_strong_trend:
        m1 = 1.5

    # Vérification du volume
    z = (current_volume - volume_mean) / volume_std if volume_std > 0 else 0
    if z > 1.75:
        m2 = 1.5
    elif z < -1.75:
        m2 = 0.7

    volume_ratio = current_volume / volume_mean if volume_mean > 0 else 0
    if volume_ratio > 1.5:
        m3 = 1.5
    elif volume_ratio < 0.5:
        m3 = 0.7

    # RSI : Signaux haussiers
    if rsi_cross_up:
        score += a1
    if delta_rsi > 3:
        score += m3 * a2
    if rsi_cross_mid:
        score += a3

    # RSI : Signaux baissiers
    if rsi_cross_down:
        score -= a1
    if delta_rsi < -3:
        score -= m3 * a2

    if rsi_ok:
        score += a4
    else:
        score -= a4

    # EMA : Structure de tendance
    if ema_structure_up:
        score += m1 * a5
    if ema_structure_down:
        score -= m1 * a5

    # MACD : Croisements
    if is_macd_cross_up:
        score += a6
    if is_macd_cross_down:
        score -= a6

    # Volume
    if is_volume_ok:
        score += m2 * a6
    else:
        score -= m2 * a6

    # Performance passée
    if is_variation_ok:
        score += a7
    else:
        score -= a7

    # Conditions d'achat renforcées
    buy_conditions = (
        (is_macd_cross_up or ema_structure_up) and
        (rsi_cross_up or rsi_cross_mid) and
        (last_rsi < 65) and
        (last_bb_percent < 0.7) and
        (strong_uptrend or adx_strong_trend) and
        (volume_mean > volume_seuil) and
        (is_variation_ok if not np.isnan(variation_30j) else True)
    )

    # Conditions de vente renforcées
    sell_conditions = (
        (is_macd_cross_down or ema_structure_down) and
        (rsi_cross_down or last_rsi > 70) and
        (last_rsi > 35) and
        (last_bb_percent > 0.3) and
        (strong_downtrend or adx_strong_trend) and
        (volume_mean > volume_seuil)
    )

    if strong_uptrend:
        score += m2 * a5
    if last_bb_percent < 0.4:
        score += m3 * a4
    if buy_conditions:
        score += a8

    if strong_downtrend:
        score -= m2 * a5
    if last_bb_percent > 0.6:
        score -= m3 * a4
    if sell_conditions:
        score -= a8

    if volatility > 0.05:
        m4 = 0.75
    score *= m4

    # Interprétation du score
    if score >= thresholds[0]:
        signal = "ACHAT"
    elif score <= thresholds[1]:
        signal = "VENTE"
    else:
        signal = "NEUTRE"


    # Helper: compute robust numerical derivatives (slope) for series
    def compute_derivatives(series_dict, window: int = 8):
        """
        Compute slope (units per period) and relative slope (slope / last_value)
        using a linear fit (polyfit degree 1) on the last `window` points when possible.

        Returns a dict with keys like '<name>_slope' and '<name>_slope_rel'.
        """
        deriv = {}
        for name, ser in series_dict.items():
            try:
                if not isinstance(ser, (pd.Series, pd.DataFrame)):
                    arr = np.asarray(ser)
                else:
                    arr = ser.dropna().values.astype(float)

                n = len(arr)
                if n >= 2:
                    k = min(window, n)
                    y = arr[-k:]
                    x = np.arange(k, dtype=float)
                    # linear fit: slope is coefficient
                    try:
                        p = np.polyfit(x, y, 1)
                        slope = float(p[0])
                    except Exception:
                        # fallback to simple diff
                        slope = float(y[-1] - y[-2]) if k >= 2 else 0.0

                    last = float(arr[-1]) if n > 0 else 0.0
                    rel = float(slope / last) if last != 0 else 0.0
                else:
                    slope = 0.0
                    rel = 0.0

                deriv[f"{name}_slope"] = slope
                deriv[f"{name}_slope_rel"] = rel
            except Exception:
                deriv[f"{name}_slope"] = 0.0
                deriv[f"{name}_slope_rel"] = 0.0
        return deriv

    if return_derivatives:
        series_dict = {
            'price': prices,
            'ema20': ema20,
            'ema50': ema50,
            'macd': macd,
            'signal_line': signal_line,
            'rsi': rsi,
            'volume': volumes
        }
        derivatives = compute_derivatives(series_dict, window=8)
    
        # Ajouter les dérivées financières si symbol fourni
        # todo : a adapter pour eviter les try/except inutiles
        if symbol:
            try:
                fin_deriv = compute_financial_derivatives(symbol, lookback_quarters=4)
                derivatives.update(fin_deriv)
            except Exception:
                # Ajouter des valeurs par défaut si erreur
                derivatives['rev_growth_val'] = None
                derivatives['gross_margin_val'] = None
                derivatives['fcf_val'] = None
                derivatives['debt_to_equity_val'] = None
                derivatives['market_cap_val'] = None
        
        return signal, last_close, last_close > last_ema20, round(last_rsi, 2), round(volume_mean, 2), round(score, 3), derivatives

    return signal, last_close, last_close > last_ema20, round(last_rsi, 2), round(volume_mean, 2), round(score, 3)

# ====================================================================
# MÉTRIQUES FINANCIÈRES CLÉS ET LEURS DÉRIVÉES
# =======================================================================

def get_financial_metrics(symbol: str) -> dict:
    """
    Récupère les 5 métriques financières clés d'une action via yfinance.
    
    Retourne un dictionnaire avec:
    - revenue_growth: Croissance du chiffre d'affaires (%)
    - gross_margin: Marge brute (%)
    - free_cash_flow: Free Cash Flow (milliards $)
    - debt_to_equity: Ratio Dette/Équité
    - market_cap: Capitalisation boursière (milliards $)
    """
    metrics = {
        'revenue_growth': None,
        'gross_margin': None,
        'free_cash_flow': None,
        'debt_to_equity': None,
        'market_cap': None
    }
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 1. Capitalisation boursière
        market_cap = info.get('marketCap')
        if market_cap:
            metrics['market_cap'] = float(market_cap) / 1e9
        
        # 2. Ratio Dette/Équité
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            metrics['debt_to_equity'] = float(debt_to_equity)
        
        # États financiers trimestriels
        financials = ticker.quarterly_financials
        cashflow = ticker.quarterly_cashflow
        
        if not financials.empty:
            # 3. Croissance du chiffre d'affaires
            try:
                revenues = financials.loc['Total Revenue']
                if len(revenues) >= 2:
                    # ✅ CORRECTION 1: Ajouter  pour accéder à la première valeur
                    rev_growth = (((revenues.iloc[0] - revenues.iloc[-1]) / revenues.iloc[-1]) * 100)
                    metrics['revenue_growth'] = rev_growth
            except Exception:
                pass
            
            # 4. Marge brute
            try:
                gross_profit = financials.loc['Gross Profit']
                total_revenue = financials.loc['Total Revenue']
                if not gross_profit.empty and not total_revenue.empty:
                    # ✅ CORRECTION 2 & 3: Ajouter  pour accéder aux premières valeurs
                    latest_gp = gross_profit.iloc[0]
                    latest_rev = total_revenue.iloc[0]
                    if latest_rev != 0:
                        margin = (latest_gp / latest_rev * 100)
                        metrics['gross_margin'] = margin
            except Exception:
                pass
        
        # 5. Free Cash Flow
        if not cashflow.empty:
            try:
                if 'Free Cash Flow' in cashflow.index:
                    # ✅ CORRECTION 4: Ajouter  pour accéder à la première valeur
                    fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                    metrics['free_cash_flow'] = float(fcf) / 1e9
            except Exception:
                pass
    
    except Exception:
        pass
    
    return metrics

def compute_financial_derivatives(symbol: str, lookback_quarters: int = 4) -> dict:
    """
    Calcule les dérivées (pentes) des métriques financières sur les derniers trimestres.
    Idéal pour intégrer dans les calculs de score.
    """
    derivatives = {
    'rev_growth_slope': 0.0,
    'gross_margin_slope': 0.0,
    'fcf_slope': 0.0,
    'debt_to_equity_val': 0.0,
    'market_cap_val': 0.0,
    }

    try:
        ticker = yf.Ticker(symbol)
        financials = ticker.quarterly_financials
        cashflow = ticker.quarterly_cashflow
        info = ticker.info
        
        # Croissance du chiffre d'affaires - pente sur derniers trimestres
        if not financials.empty and 'Total Revenue' in financials.index:
            revenues = financials.loc['Total Revenue'].head(lookback_quarters)
            if len(revenues) >= 2:
                growth_rates = revenues.pct_change() * 100
                growth_rates = growth_rates.dropna()
                if len(growth_rates) >= 2:
                    x = np.arange(len(growth_rates), dtype=float)
                    y = growth_rates.values.astype(float)
                    try:
                        p = np.polyfit(x, y, 1)
                        derivatives['rev_growth_slope'] = float(p)
                    except Exception:
                        pass
        
        # Marge brute - pente sur derniers trimestres
        if not financials.empty:
            try:
                gp = financials.loc['Gross Profit'].head(lookback_quarters)
                rev = financials.loc['Total Revenue'].head(lookback_quarters)
                if len(gp) >= 2 and len(rev) >= 2:
                    margins = (gp / rev * 100).dropna()
                    if len(margins) >= 2:
                        x = np.arange(len(margins), dtype=float)
                        y = margins.values.astype(float)
                        p = np.polyfit(x, y, 1)
                        derivatives['gross_margin_slope'] = float(p)
            except Exception:
                pass
        
        # Free Cash Flow - pente sur derniers trimestres
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fcf_data = (cashflow.loc['Free Cash Flow'].head(lookback_quarters) / 1e9).dropna()
            if len(fcf_data) >= 2:
                x = np.arange(len(fcf_data), dtype=float)
                y = fcf_data.values.astype(float)
                try:
                    p = np.polyfit(x, y, 1)
                    derivatives['fcf_slope'] = float(p)
                except Exception:
                    pass
        
        # Ratio Dette/Équité et Market Cap (valeurs actuelles)
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            derivatives['debt_to_equity_val'] = float(debt_to_equity)
        
        market_cap = info.get('marketCap')
        if market_cap:
            derivatives['market_cap_val'] = float(market_cap) / 1e9

    except Exception:
        pass

    return derivatives

# ===================================================================
# SYSTÈME DE CACHE INTELLIGENT INTÉGRÉ
# ===================================================================

# Configuration du cache pour les données boursières
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Configuration globale pour le mode offline
OFFLINE_MODE = False

def load_symbol_lists():
    """Charge toutes les listes de symboles avec gestion d'erreurs robuste"""
    lists_config = {
        "mes_symbols.txt": {"max_age_hours": 6, "priority": 1, "context": "personal"},
        "popular_symbols.txt": {"max_age_hours": 12, "priority": 2, "context": "analysis"},
        "optimisation_symbols.txt": {"max_age_hours": 24, "priority": 3, "context": "optimization"},
        "test_symbols.txt": {"max_age_hours": 48, "priority": 4, "context": "test"}
    }
    
    symbol_lists = {}
    total_symbols = 0
    
    for filename, config in lists_config.items():
        try:
            filepath = Path(filename)
            symbols = set()
            
            if filepath.exists() and filepath.stat().st_size > 0:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        symbol = line.strip().upper()
                        if symbol and not symbol.startswith('#') and symbol.isalnum() or any(c in symbol for c in ['.', '-', '=']):
                            symbols.add(symbol)
                
                symbol_lists[filename] = {
                    "symbols": symbols,
                    "config": config,
                    "count": len(symbols)
                }
                total_symbols += len(symbols)
                # print(f"📋 {filename}: {len(symbols)} symboles chargés")
            else:
                # print(f"⚠️ {filename}: Fichier manquant ou vide")
                symbol_lists[filename] = {"symbols": set(), "config": config, "count": 0}
        
        except Exception as e:
            print(f"⚠️ Erreur lecture {filename}: {e}")
            symbol_lists[filename] = {"symbols": set(), "config": config, "count": 0}
    
    # Cache des listes en mémoire pour éviter de relire les fichiers
    load_symbol_lists._cached_lists = symbol_lists
    load_symbol_lists._cache_time = datetime.now()
    
    return symbol_lists

def get_symbol_classification(symbols: List[str]) -> Dict:
    """Classifie intelligemment les symboles selon les listes existantes"""
    
    # Utiliser le cache si récent (5 minutes)
    if hasattr(load_symbol_lists, '_cached_lists') and hasattr(load_symbol_lists, '_cache_time'):
        age = (datetime.now() - load_symbol_lists._cache_time).total_seconds()
        if age < 300:  # 5 minutes
            symbol_lists = load_symbol_lists._cached_lists
        else:
            symbol_lists = load_symbol_lists()
    else:
        symbol_lists = load_symbol_lists()
    
    if not symbols:
        return {"strategy": "default", "max_age_hours": 24, "source": "liste vide"}
    
    # Nettoyer et normaliser les symboles
    clean_symbols = []
    for s in symbols:
        if s and str(s).strip():
            clean_symbols.append(str(s).strip().upper())
    
    if not clean_symbols:
        return {"strategy": "default", "max_age_hours": 24, "source": "aucun symbole valide"}
    
    symbols_set = set(clean_symbols)
    
    # Analyser la correspondance avec chaque liste
    matches = {}
    for filename, list_info in symbol_lists.items():
        if list_info["count"] == 0:
            continue
        
        overlap = symbols_set.intersection(list_info["symbols"])
        overlap_ratio = len(overlap) / len(clean_symbols)
        
        matches[filename] = {
            "overlap_count": len(overlap),
            "overlap_ratio": overlap_ratio,
            "config": list_info["config"],
            "overlapping_symbols": overlap
        }
    
    # Détecter les nouveaux symboles
    all_known_symbols = set()
    for list_info in symbol_lists.values():
        all_known_symbols.update(list_info["symbols"])
    
    new_symbols = symbols_set - all_known_symbols
    
    # Stratégie de décision
    if matches:
        # Trouver la meilleure correspondance
        best_match = max(matches.items(), key=lambda x: x[1]["overlap_ratio"])
        best_filename, best_info = best_match
        
        if best_info["overlap_ratio"] >= 0.3:  # Seuil 30%
            # Correspondance suffisante
            config = best_info["config"]
            return {
                "strategy": "list_based",
                "max_age_hours": config["max_age_hours"],
                "source": f"{best_filename} ({best_info['overlap_ratio']:.1%})",
                "new_symbols": new_symbols,
                "total_symbols": len(clean_symbols),
                "known_symbols": len(clean_symbols) - len(new_symbols)
            }
    
    # Fallback pour symboles majoritairement nouveaux
    fallback_age = 24  # Défaut 24h pour nouveaux symboles
    
    # Ajustement contextuel intelligent basé sur les suffixes
    if any('.HK' in s for s in clean_symbols):
        fallback_age = 18  # Actions HK mises à jour moins fréquemment
    elif any(s.endswith('-USD') or s.endswith('=F') for s in clean_symbols):
        fallback_age = 8   # Crypto et futures plus volatiles
    elif any(len(s) <= 4 and s.isalpha() for s in clean_symbols):
        fallback_age = 12  # Actions US classiques
    
    return {
        "strategy": "context_fallback",
        "max_age_hours": fallback_age,
        "source": f"contexte (nouveaux symboles)",
        "new_symbols": new_symbols,
        "total_symbols": len(clean_symbols),
        "known_symbols": 0
    }

def log_new_symbols(new_symbols: set, context: str = "unknown"):
    """Log des nouveaux symboles pour traçabilité et analyse"""
    if not new_symbols:
        return
    
    try:
        log_dir = Path("cache_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "nouveaux_symboles.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            for symbol in new_symbols:
                f.write(f"{timestamp},{symbol},{context}\n")
        
        # Log condensé pour suivi en temps réel
        if len(new_symbols) <= 3:
            # print(f"🆕 Nouveaux: {list(new_symbols)}")
            pass
        else:
            # print(f"🆕 {len(new_symbols)} nouveaux symboles (voir cache_logs/nouveaux_symboles.log)")
            pass
            
    except Exception as e:
        # print(f"⚠️ Impossible de logger: {e}")
        pass

def analyze_cache_status(symbols: List[str], period: str, max_age_hours: int) -> Dict:
    """Analyse l'état du cache pour les symboles donnés"""
    fresh_count = 0
    stale_count = 0
    missing_count = 0
    
    for symbol in symbols:
        cache_file = CACHE_DIR / f"{symbol}_{period}.pkl"
        if cache_file.exists():
            try:
                age_hours = (datetime.now() - datetime.fromtimestamp(
                    cache_file.stat().st_mtime)).total_seconds() / 3600
                if age_hours <= max_age_hours:
                    fresh_count += 1
                else:
                    stale_count += 1
            except Exception:
                missing_count += 1
        else:
            missing_count += 1
    
    fresh_ratio = fresh_count / len(symbols) if symbols else 0
    
    return {
        "fresh_count": fresh_count,
        "stale_count": stale_count,
        "missing_count": missing_count,
        "fresh_ratio": fresh_ratio,
        "total_count": len(symbols)
    }

def get_cached_data(symbol: str, period: str, max_age_hours: int = 24, force_offline: bool = False) -> pd.DataFrame:
    """Récupère les données en cache si elles existent et sont récentes, sinon télécharge."""
    
    cache_file = CACHE_DIR / f"{symbol}_{period}.pkl"
    
    # Mode forcé offline ou global OFFLINE_MODE
    if force_offline or OFFLINE_MODE:
        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except Exception as e:
                print(f"⚠️ Erreur lecture cache {symbol}: {e}")
                return pd.DataFrame()
        else:
            # print(f"❌ Pas de cache disponible pour {symbol} en mode offline")
            return pd.DataFrame()
    
    # Vérifier si le cache existe et est frais
    if cache_file.exists():
        try:
            age_hours = (datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime)).total_seconds() / 3600
            
            if age_hours <= max_age_hours:
                return pd.read_pickle(cache_file)
            # else:
                # print(f"💾 Cache obsolète pour {symbol} ({age_hours:.1f}h > {max_age_hours}h)")
        except Exception as e:
            print(f"⚠️ Erreur cache {symbol}: {e}")
    
    # Télécharger et mettre en cache
    try:
        # print(f"🌐 Téléchargement {symbol}...")
        data = yf.download(symbol, period=period, progress=False)
        
        if not data.empty:
            data.to_pickle(cache_file)
            return data
        else:
            # print(f"❌ Aucune donnée reçue pour {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"🚨 Erreur téléchargement {symbol}: {e}")
        
        # Tentative de récupération avec cache obsolète
        if cache_file.exists():
            try:
                # print(f"🔄 Utilisation cache obsolète pour {symbol}")
                return pd.read_pickle(cache_file)
            except Exception:
                pass
        
        return pd.DataFrame()

def download_stock_data(symbols: List[str], period: str) -> Dict[str, Dict[str, pd.Series]]:
    """
    *** VERSION INTELLIGENTE INTÉGRÉE ***
    Télécharge automatiquement les données boursières avec gestion intelligente du cache.
    
    🎯 NOUVEAUTÉS:
    - Détection automatique des listes de symboles
    - Cache adaptatif selon le type de symbole
    - Gestion automatique des nouveaux symboles
    - Logs de traçabilité
    - Récupération d'urgence
    - API identique (transparente)
    
    Args:
        symbols: Liste des symboles boursiers (ex: ['AAPL', 'MSFT']).
        period: Période des données (ex: '1y', '6mo', '1mo').

    Returns:
        Dictionnaire avec les données valides: {'symbol': {'Close': pd.Series, 'Volume': pd.Series}}.
    """
    
    if not symbols:
        return {}
    
    # ÉTAPE 1: VALIDATION ET NETTOYAGE
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '12mo', '1y', "18mo", "24mo", '2y', '5y', '10y', 'ytd', 'max']
    if period not in valid_periods:
        print(f"🚨 Période invalide: {period}. Valeurs possibles: {valid_periods}")
        return {}
    
    YAHOO_SUFFIXES = ('.HK', '.DE', '.PA', '.AS', '.SW', '.L', '.TO', '.V', '.MI', '.AX', '.SI',
                     '.KQ', '.T', '.OL', '.HE', '.ST', '.CO', '.SA', '.MX', '.TW', '.JO', '.SZ', '.NZ', '.KS',
                     '.PL', '.IR', '.MC', '.VI', '.BK', '.SS', '.SG', '.F', '.BE', '.CN', '.TA', '-USD', '=F')
    
    # Filtrage et nettoyage intelligents
    clean_symbols = []
    for s in symbols:
        if not s or not str(s).strip():
            continue
        
        symbol = str(s).strip().upper()
        
        # Validation format Yahoo Finance
        if '.' in symbol and not symbol.endswith(YAHOO_SUFFIXES):
            # print(f"⚠️ Format potentiellement invalide: {symbol}")
            continue
        
        clean_symbols.append(symbol)
    
    if not clean_symbols:
        print("❌ Aucun symbole valide après nettoyage")
        return {}
    
    # ÉTAPE 2: CLASSIFICATION INTELLIGENTE
    classification = get_symbol_classification(clean_symbols)
    max_age_hours = classification["max_age_hours"]
    
    # ÉTAPE 3: ANALYSE DU CACHE
    cache_status = analyze_cache_status(clean_symbols, period, max_age_hours)
    
    # ÉTAPE 4: DÉCISION DOWNLOAD vs CACHE
    use_cache_mode = cache_status["fresh_ratio"] >= 0.7  # 70% frais
    
    # ÉTAPE 5: LOG DES NOUVEAUX SYMBOLES
    if "new_symbols" in classification and classification["new_symbols"]:
        log_new_symbols(classification["new_symbols"], classification.get("source", "unknown"))
    
    # ÉTAPE 6: AFFICHAGE STATUS (optionnel, pour debug)
    status_icon = "🚀" if use_cache_mode else "🌐"
    # print(f"{status_icon} {classification['source']}: {cache_status['fresh_count']}/{cache_status['total_count']} frais ({cache_status['fresh_ratio']:.1%})")
    
    if "new_symbols" in classification and classification["new_symbols"] and len(classification["new_symbols"]) <= 5:
        # print(f"🆕 Nouveaux: {list(classification['new_symbols'])[:3]}")
        pass
    
    # ÉTAPE 7: TÉLÉCHARGEMENT OPTIMISÉ
    valid_data = {}
    
    if use_cache_mode:
        # Mode cache prioritaire
        for symbol in clean_symbols:
            try:
                data = get_cached_data(symbol, period, max_age_hours, force_offline=True)
                if not data.empty and 'Close' in data.columns and 'Volume' in data.columns:
                    valid_data[symbol] = {
                        'Close': data['Close'].squeeze(),
                        'Volume': data['Volume'].squeeze()
                    }
            except Exception as e:
                # print(f"⚠️ Erreur cache pour {symbol}: {e}")
                pass
    
    # Compléter avec téléchargements manquants
    missing_symbols = [s for s in clean_symbols if s not in valid_data]
    
    if missing_symbols:
        # Traitement par batch pour optimiser
        batch_size = 50  # Réduit pour éviter timeouts
        for i in range(0, len(missing_symbols), batch_size):
            batch = missing_symbols[i:i + batch_size]
            
            # Téléchargement groupé avec fallback individuel
            try:
                all_data = yf.download(
                    batch,
                    period=period,
                    group_by='ticker',
                    progress=False,
                    threads=True,
                    timeout=20
                )
                
                # Extraction des données du batch
                for symbol in batch:
                    try:
                        if len(batch) == 1:
                            data = all_data
                        else:
                            data = all_data[symbol] if symbol in all_data.columns.levels[0] else None
                        
                        if data is not None and not data.empty:
                            # --- Normalisation des colonnes yfinance ---
                            # yfinance peut parfois retourner un DataFrame à colonnes MultiIndex
                            # (par ex. niveau sup: ticker ou 'Price'), ce qui empêche les tests
                            # "'Close' in data.columns". Ici on gère ces cas en extrayant la
                            # tranche correcte (data[symbol]) si présente, ou en aplatissant
                            # vers le niveau des noms de colonnes (Open/High/Low/Close/Volume).
                            try:
                                if hasattr(data, 'columns') and isinstance(data.columns, pd.MultiIndex):
                                    # Si le premier niveau contient le ticker, on extrait
                                    if symbol in data.columns.get_level_values(0):
                                        data = data[symbol]
                                    else:
                                        # Aplatir en ne conservant que le dernier niveau
                                        data.columns = data.columns.get_level_values(-1)
                            except Exception:
                                # Si la normalisation échoue, on continue et laissera
                                # la validation suivante filtrer le dataset.
                                pass

                            # Validation et nettoyage
                            if 'Close' in data.columns and 'Volume' in data.columns:
                                if len(data) >= 50:  # Minimum requis pour get_trading_signal
                                    # Nettoyage des NaN
                                    clean_data = data[['Close', 'Volume']].copy()
                                    clean_data['Close'] = clean_data['Close'].ffill()
                                    clean_data['Volume'] = clean_data['Volume'].fillna(0)
                                    
                                    if not clean_data['Close'].isna().all():
                                        # Sauvegarde en cache
                                        cache_file = CACHE_DIR / f"{symbol}_{period}.pkl"
                                        clean_data.to_pickle(cache_file)
                                        
                                        valid_data[symbol] = {
                                            'Close': clean_data['Close'].squeeze(),
                                            'Volume': clean_data['Volume'].squeeze()
                                        }
                    except Exception as e:
                        # print(f"⚠️ Erreur traitement {symbol}: {e}")
                        pass
                
            except Exception as e:
                # print(f"🚨 Erreur batch: {e}")
                
                # Fallback: téléchargements individuels
                for symbol in batch:
                    try:
                        data = get_cached_data(symbol, period, max_age_hours, force_offline=False)
                        if not data.empty and 'Close' in data.columns and 'Volume' in data.columns and len(data) >= 50:
                            # Nettoyage
                            clean_data = data[['Close', 'Volume']].copy()
                            clean_data['Close'] = clean_data['Close'].ffill()
                            clean_data['Volume'] = clean_data['Volume'].fillna(0)
                            
                            if not clean_data['Close'].isna().all():
                                valid_data[symbol] = {
                                    'Close': clean_data['Close'].squeeze(),
                                    'Volume': clean_data['Volume'].squeeze()
                                }
                    except Exception as e2:
                        # print(f"⚠️ Fallback échoué {symbol}: {e2}")
                        pass
    
    # ÉTAPE 8: RAPPORT FINAL
    success_rate = len(valid_data) / len(clean_symbols) if clean_symbols else 0
    
    if success_rate < 0.8:
        failed_symbols = set(clean_symbols) - set(valid_data.keys())
        if len(failed_symbols) <= 5:
            print(f"⚠️ Échec pour: {list(failed_symbols)}")
        else:
            print(f"⚠️ {len(failed_symbols)} symboles échoués")
    
    # Suggestion automatique d'amélioration
    if "new_symbols" in classification and classification["new_symbols"] and len(valid_data) > 0:
        successful_new = classification["new_symbols"].intersection(set(valid_data.keys()))
        if successful_new and len(successful_new) <= 3:
            # print(f"💡 Suggestion: Ajouter {list(successful_new)} à vos listes pour optimiser futures sessions")
            pass
    
    return valid_data

# def backtest_signals(prices: Union[pd.Series, pd.DataFrame], volumes: Union[pd.Series, pd.DataFrame],
#                      domaine: str, montant: float = 50, transaction_cost: float = 0.00, domain_coeffs=None, seuil_achat=None, seuil_vente=None) -> Dict:
#     """
#     Effectue un backtest sur la série de prix.
#     Un 'trade' correspond à un cycle complet ACHAT puis VENTE (entrée puis sortie).
#     Le gain est calculé pour chaque cycle, avec prise en compte des frais de transaction.

#     Args:
#         prices: Série ou DataFrame des prix de clôture.
#         volumes: Série ou DataFrame des volumes.
#         domaine: Secteur de l'actif (ex: 'Technology').
#         montant: Montant investi par trade (défaut: 50).
#         transaction_cost: Frais de transaction par trade (défaut: 0.1%).

#     Returns:
#         Dict avec les métriques: trades, gagnants, taux_reussite, gain_total, gain_moyen, drawdown_max.
#     """
#     # Validation des entrées
#     if not isinstance(prices, (pd.Series, pd.DataFrame)) or not isinstance(volumes, (pd.Series, pd.DataFrame)):
#         return {
#             "trades": 0,
#             "gagnants": 0,
#             "taux_reussite": 0,
#             "gain_total": 0.0,
#             "gain_moyen": 0.0,
#             "drawdown_max": 0.0
#         }

#     if isinstance(prices, pd.DataFrame):
#         prices = prices.squeeze()
#     if isinstance(volumes, pd.DataFrame):
#         volumes = volumes.squeeze()

#     if len(prices) < 50 or len(volumes) < 50 or prices.isna().any() or volumes.isna().any():
#         return {
#             "trades": 0,
#             "gagnants": 0,
#             "taux_reussite": 0,
#             "gain_total": 0.0,
#             "gain_moyen": 0.0,
#             "drawdown_max": 0.0
#         }

#     # Pré-calculer les signaux pour toute la série
#     signals = []
#     for i in range(50, len(prices)):
#         signal, *_ = get_trading_signal(prices[:i], volumes[:i], domaine, domain_coeffs=domain_coeffs,)
#         signals.append(signal)

#     signals = pd.Series(signals, index=prices.index[50:])

#     # Simuler les trades
#     positions = []
#     for i in range(len(signals)):
#         if signals.iloc[i] == "ACHAT":
#             positions.append({"entry": prices.iloc[i + 50], "entry_idx": i + 50, "type": "buy"})
#         elif signals.iloc[i] == "VENTE" and positions and "exit" not in positions[-1]:
#             positions[-1]["exit"] = prices.iloc[i + 50]
#             positions[-1]["exit_idx"] = i + 50

#     # Fermer les positions ouvertes avec le dernier prix
#     # if positions and "exit" not in positions[-1]:
#     #     positions[-1]["exit"] = prices.iloc[-1]
#     #     positions[-1]["exit_idx"] = len(prices) - 1

#     # Calculer les métriques
#     nb_trades = 0
#     nb_gagnants = 0
#     gain_total = 0.0
#     gains = []
#     portfolio_values = [montant]  # Suivi de la valeur du portefeuille

#     for pos in positions:
#         if "exit" in pos:
#             nb_trades += 1
#             entry = pos["entry"]
#             exit = pos["exit"]
#             rendement = (exit - entry) / entry

#             # Ajuster pour les frais de transaction (entrée + sortie)
#             gain = montant * rendement * (1 - 2 * transaction_cost)
#             gain_total += gain
#             gains.append(gain)

#             if gain > 0:
#                 nb_gagnants += 1

#             portfolio_values.append(portfolio_values[-1] + gain)

#     # Calculer le drawdown maximum
#     portfolio_series = pd.Series(portfolio_values)
#     rolling_max = portfolio_series.cummax()
#     drawdowns = (portfolio_series - rolling_max) / rolling_max
#     drawdown_max = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0

#     return {
#         "trades": nb_trades,
#         "gagnants": nb_gagnants,
#         "taux_reussite": (nb_gagnants / nb_trades * 100) if nb_trades else 0,
#         "gain_total": round(gain_total, 2),
#         "gain_moyen": round(np.mean(gains), 2) if gains else 0.0,
#         "drawdown_max": round(drawdown_max, 2)
#     }

def plot_unified_chart(symbol, prices, volumes, ax, show_xaxis=False):
    """Trace un graphique unifié avec prix, MACD et RSI intégré"""
    # Vérification du format des prix
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices, name=symbol)

    # Calcul des indicateurs
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema50 = prices.ewm(span=50, adjust=False).mean()
    sma50 = prices.rolling(window=50).mean() if len(prices) >= 50 else pd.Series()
    macd, signal_line = calculate_macd(prices)

    # Calcul du RSI avec vérification des données
    try:
        rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
    except Exception as e:
        print(f"⚠️ Erreur RSI pour {symbol}: {str(e)}")
        rsi = pd.Series(np.zeros(len(prices)), index=prices.index)

    # Tracé des prix sur l'axe principal
    color = 'tab:blue'
    ax.plot(prices.index, prices, label='Prix', color=color, linewidth=1.8)
    ax.plot(ema20.index, ema20, label='EMA20', linestyle='--', color='orange', linewidth=1.4)
    ax.plot(ema50.index, ema50, label='EMA50', linestyle='-.', color='purple', linewidth=1.4)
    if not sma50.empty:
        ax.plot(sma50.index, sma50, label='SMA50', linestyle=':', color='green', linewidth=1.4)

    ax.set_ylabel('Prix', color=color, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Création d'un axe secondaire pour MACD
    ax2 = ax.twinx()

    # Tracé MACD
    color = 'tab:purple'
    ax2.plot(macd.index, macd, label='MACD', color=color, linewidth=1.2)
    ax2.plot(signal_line.index, signal_line, label='Signal', color='tab:orange', linewidth=1.2)
    ax2.fill_between(
        macd.index, 0, macd - signal_line,
        where=(macd - signal_line) >= 0,
        facecolor='green', alpha=0.3, interpolate=True
    )
    ax2.fill_between(
        macd.index, 0, macd - signal_line,
        where=(macd - signal_line) < 0,
        facecolor='red', alpha=0.3, interpolate=True
    )
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('MACD', color=color, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color)

    # Tracé RSI en arrière-plan avec axvspan
    for i in range(1, len(prices)):
        start = prices.index[i-1]
        end = prices.index[i]
        rsi_val = rsi.iloc[i-1]
        if rsi_val > 70:
            color = 'lightcoral'
        elif rsi_val < 30:
            color = 'lightgreen'
        else:
            color = 'lightgray'
        ax.axvspan(start, end, facecolor=color, alpha=0.1, zorder=-1)

    # Ajout des légendes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

    info = yf.Ticker(symbol).info
    domaine = info.get("sector", "Inconnu")

    # Ajout des signaux trading (ne pas demander les dérivées ici — elles seront consommées par l'UI)
    signal, last_price, trend, last_rsi, volume_moyen, score = get_trading_signal(prices, volumes, domaine=domaine)

    # Calcul de la progression en pourcentage, en évitant la division par zéro ou NaN
    if len(prices) > 1 and not pd.isna(prices.iloc[0]) and prices.iloc[0] != 0:
        progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
        # Éventuellement arrondir ou limiter
        progression = round(progression, 2)
    else:
        progression = 0.0

    if last_price is not None:
        trend_symbol = "Haussière" if trend else "Baissière"
        rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
        signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'

        # Compose a compact derivative summary for the title
        title = (
            f"{symbol} | Prix: {last_price:.2f} | Signal: {signal} ({score}) | "
            f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
            f"Progression: {progression:+.2f}% | Vol. moyen: {volume_moyen:,.0f} units"
        )

        ax.set_title(title, fontsize=12, fontweight='bold', color=signal_color)

    # Affichage de l'axe du temps uniquement sur le dernier plot
    if not show_xaxis:
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax2.tick_params(axis='x', which='both', labelbottom=True)

    return ax2


def generate_trade_events(prices: pd.Series, volumes: pd.Series, domaine: str) -> List[Dict]:
    """
    Simule rapidement les signaux sur la série pour produire une liste d'événements de trade
    (entrée / sortie) utilisables pour l'affichage des marqueurs sur les graphiques.

    Retourne une liste d'événements de la forme:
      [{ 'date': Timestamp, 'type': 'BUY'|'SELL', 'price': float, 'idx': int }, ...]

    Cette fonction utilise la logique get_trading_signal sur des fenêtres progressives
    et ne remplace pas le backtest complet (agrégé) qui peut rester accéleré en C.
    """
    events = []
    try:
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if isinstance(volumes, pd.DataFrame):
            volumes = volumes.squeeze()

        n = len(prices)
        if n < 60:
            return events

        open_pos = False
        entry_idx = None

        # On commence à 50 (cohérent avec le backtest minimal)
        for i in range(50, n - 1):
            window_prices = prices.iloc[:i]
            window_vols = volumes.iloc[:i]
            try:
                sig, *_ = get_trading_signal(window_prices, window_vols, domaine)
            except Exception:
                sig = "NEUTRE"

            # Nous enregistrons l'exécution effective au prix suivant (i)
            exec_price = float(prices.iloc[i])
            exec_date = prices.index[i]

            if sig == "ACHAT" and not open_pos:
                open_pos = True
                entry_idx = i
                events.append({"date": exec_date, "type": "BUY", "price": exec_price, "idx": i})
            elif sig == "VENTE" and open_pos:
                open_pos = False
                events.append({"date": exec_date, "type": "SELL", "price": exec_price, "idx": i})

        # Optionnel: si position ouverte à la fin, on peut la fermer au dernier prix
        if open_pos and entry_idx is not None:
            last_idx = n - 1
            events.append({"date": prices.index[last_idx], "type": "SELL", "price": float(prices.iloc[-1]), "idx": last_idx})

    except Exception:
        return events

    return events

def analyse_et_affiche(symbols, period="12mo"):
    """
    Télécharge les données pour les symboles donnés et affiche les graphiques d'analyse technique.
    """
    print("⏳ Téléchargement des données...")
    data = download_stock_data(symbols, period)

    if not data:
        print("❌ Aucune donnée valide disponible. Vérifiez les symboles ou la connexion internet.")
        return

    num_plots = len(data)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots), sharex=False)

    if num_plots == 1:
        axes = [axes]
    elif num_plots == 0:
        print("❌ Aucun symbole valide à afficher")
        return

    for i, (symbol, stock_data) in enumerate(data.items()):
        prices = stock_data['Close']
        volumes = stock_data['Volume']
        print(f"📊 Traitement de {symbol}...")

        show_xaxis = (i == len(data) - 1)  # True seulement pour le dernier subplot
        plot_unified_chart(symbol, prices, volumes, axes[i], show_xaxis=show_xaxis)

        # Dessiner les marqueurs d'achat/vente générés par la simulation rapide
        try:
            try:
                info = yf.Ticker(symbol).info
                domaine = info.get("sector", "Inconnu")
            except Exception:
                domaine = "Inconnu"

            events = generate_trade_events(prices, volumes, domaine)
            for ev in events:
                if ev.get('type') == 'BUY':
                    axes[i].scatter(ev['date'], ev['price'], marker='^', s=80, color='green', edgecolor='black', zorder=6)
                    axes[i].annotate('BUY', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,8), ha='center', fontsize=8, color='green')
                elif ev.get('type') == 'SELL':
                    axes[i].scatter(ev['date'], ev['price'], marker='v', s=80, color='red', edgecolor='black', zorder=6)
                    axes[i].annotate('SELL', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,-10), ha='center', fontsize=8, color='red')
        except Exception:
            pass

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.152, bottom=0.032)
    plt.show()

def save_symbols_to_txt(symbols: List[str], filename: str):
    """Sauvegarde la liste de symboles dans un fichier texte"""
    with open(filename, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(symbol + '\n')

def load_symbols_from_txt(filename: str) -> List[str]:
    """Charge la liste de symboles depuis un fichier texte"""
    try:
        # Prefer files located next to this module (package root)
        script_dir = Path(__file__).parent
        file_path = script_dir / filename

        # If not found next to the module, fall back to given path (cwd-based)
        if not file_path.exists():
            alt_path = Path(filename)
            if alt_path.exists():
                file_path = alt_path

        # If still not found, create an empty file next to the module to avoid repeated errors
        if not file_path.exists():
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text('', encoding='utf-8')
                print(f"Notice: '{filename}' not found — created empty file at {file_path}")
            except Exception as e:
                # Fall back to returning empty list but report error
                print(f"Erreur de lecture/creation du fichier {filename} : {e}")
                return []

        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Erreur de lecture du fichier {filename} : {e}")
        return []

def modify_symbols_file(filename: str, symbols_to_change: List[str], action: str):
    """Modifie un fichier de symboles (ajout/suppression)"""
    try:
        # Charger les symboles existants
        with open(filename, 'r', encoding='utf-8') as f:
            existing_symbols = set(line.strip() for line in f if line.strip())

        initial_count = len(existing_symbols)
        added, removed = 0, 0

        if action == "add":
            for symbol in symbols_to_change:
                if symbol not in existing_symbols:
                    existing_symbols.add(symbol)
                    added += 1
        elif action == "remove":
            for symbol in symbols_to_change:
                if symbol in existing_symbols:
                    existing_symbols.remove(symbol)
                    removed += 1
        else:
            print("⚠️ Action invalide. Utilise 'add' ou 'remove'.")
            return

        # Sauvegarder la nouvelle liste
        with open(filename, 'w', encoding='utf-8') as f:
            for symbol in sorted(existing_symbols):
                f.write(symbol + '\n')

        print(f"✅ Fichier mis à jour : {filename}")
        print(f"🔼 Symboles ajoutés : {added}")
        print(f"🔽 Symboles retirés : {removed}")
        print(f"📊 Total actuel : {len(existing_symbols)} symboles")

    except Exception as e:
        print(f"❌ Erreur lors de la modification du fichier : {e}")

# ===================================================================
# FONCTIONS UTILITAIRES POUR MAINTENANCE DU CACHE
# ===================================================================

def cache_status_report():
    """Affiche un rapport détaillé de l'état du cache"""
    print("📊 RAPPORT ÉTAT DU CACHE")
    print("=" * 50)
    
    if not CACHE_DIR.exists():
        print("❌ Dossier cache inexistant")
        return
    
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    print(f"💾 Total fichiers cache: {len(cache_files)}")
    
    if not cache_files:
        print("📁 Cache vide")
        return
    
    # Analyse par âge
    now = datetime.now()
    age_categories = {"< 6h": 0, "6h-24h": 0, "1-7j": 0, "> 7j": 0}
    total_size = 0
    
    for cache_file in cache_files:
        try:
            age_hours = (now - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            if age_hours < 6:
                age_categories["< 6h"] += 1
            elif age_hours < 24:
                age_categories["6h-24h"] += 1
            elif age_hours < 168:  # 7 jours
                age_categories["1-7j"] += 1
            else:
                age_categories["> 7j"] += 1
        except Exception:
            continue
    
    print(f"📈 Répartition par âge:")
    for category, count in age_categories.items():
        print(f"   {category}: {count} fichiers")
    
    print(f"💽 Taille totale: {total_size:.2f} MB")
    
    # Symboles les plus récents
    recent_files = sorted(cache_files, key=lambda f: f.stat().st_mtime, reverse=True)[:10]
    print(f"\n🔥 10 plus récents:")
    for cache_file in recent_files:
        try:
            age_hours = (now - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            symbol = cache_file.stem.split('_')[0]
            print(f"   {symbol}: {age_hours:.1f}h")
        except Exception:
            continue

def cleanup_cache(max_age_days: int = 30):
    """Nettoie les fichiers cache trop anciens"""
    print(f"🧹 NETTOYAGE CACHE (> {max_age_days} jours)")
    print("=" * 40)
    
    if not CACHE_DIR.exists():
        print("❌ Dossier cache inexistant")
        return
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    cleaned_count = 0
    cleaned_size = 0
    
    for cache_file in cache_files:
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                cache_file.unlink()
                cleaned_count += 1
                cleaned_size += size_mb
        except Exception as e:
            print(f"⚠️ Erreur suppression {cache_file.name}: {e}")
    
    print(f"✅ Nettoyé: {cleaned_count} fichiers ({cleaned_size:.2f} MB)")

def warmup_cache(symbol_lists: List[str] = None, period: str = "1y"):
    """Pré-chauffe le cache avec les symboles des listes importantes"""
    if symbol_lists is None:
        symbol_lists = ["mes_symbols.txt", "popular_symbols.txt"]
    
    print("🔥 PRÉ-CHAUFFAGE DU CACHE")
    print("=" * 30)
    
    all_symbols = set()
    for list_file in symbol_lists:
        try:
            symbols = load_symbols_from_txt(list_file)
            all_symbols.update(symbols)
            print(f"📋 {list_file}: {len(symbols)} symboles")
        except Exception as e:
            print(f"⚠️ Erreur {list_file}: {e}")
    
    if not all_symbols:
        print("❌ Aucun symbole à pré-charger")
        return
    
    print(f"🚀 Pré-chargement de {len(all_symbols)} symboles...")
    
    # Pré-chargement avec barre de progression
    successful = 0
    for symbol in all_symbols:
        try:
            data = get_cached_data(symbol, period, max_age_hours=24)
            if not data.empty:
                successful += 1
            print(f"\r🔄 Progression: {successful}/{len(all_symbols)}", end="", flush=True)
        except Exception:
            continue
    
    print(f"\n✅ Pré-chargement terminé: {successful}/{len(all_symbols)} réussis")

# ===================================================================
# NOUVELLES FONCTIONS D'ANALYSE DES LOGS
# ===================================================================

def analyze_new_symbols_usage():
    """Analyse les nouveaux symboles les plus utilisés"""
    log_file = Path("cache_logs/nouveaux_symboles.log")
    
    if not log_file.exists():
        print("📊 Aucun log de nouveaux symboles trouvé")
        return
    
    print("📊 ANALYSE NOUVEAUX SYMBOLES")
    print("=" * 40)
    
    try:
        import pandas as pd
        df = pd.read_csv(log_file, names=['timestamp', 'symbol', 'context'])
        
        # Symboles les plus fréquents
        symbol_counts = df['symbol'].value_counts().head(10)
        print("🔥 Top 10 nouveaux symboles:")
        for symbol, count in symbol_counts.items():
            print(f"   {symbol}: {count} utilisations")
        
        # Usage par contexte
        print(f"\n📋 Usage par contexte:")
        context_counts = df['context'].value_counts()
        for context, count in context_counts.items():
            print(f"   {context}: {count} utilisations")
        
        # Suggestions d'ajout
        frequent_symbols = symbol_counts[symbol_counts >= 3].index.tolist()
        if frequent_symbols:
            print(f"\n💡 Suggérer d'ajouter aux listes:")
            for symbol in frequent_symbols[:5]:
                print(f"   {symbol} → Utilisé {symbol_counts[symbol]} fois")
        
    except Exception as e:
        print(f"⚠️ Erreur analyse: {e}")

# ======================================================================
# CONFIGURATION PRINCIPALE
# ======================================================================

# Variable globale pour la période d'analyse
period = "12mo"  # Variable globale pour la période d'analyse ne pas commenter

# Chargement des listes de symboles
popular_symbols = load_symbols_from_txt("popular_symbols.txt")
mes_symbols = load_symbols_from_txt("mes_symbols.txt")

#todo: ajouter le parametre taux de reussite minimum pour le backtest
def analyse_signaux_populaires(
    popular_symbols, mes_symbols,
    period="12mo", afficher_graphiques=True,
    chunk_size=20, verbose=True,
    save_csv=True, plot_all=False
):
    """
    Analyse les signaux pour les actions populaires, affiche les résultats, effectue le backtest,
    et affiche les graphiques pour les signaux fiables si demandé.
    Retourne un dictionnaire contenant les résultats principaux.
    """
    import matplotlib.pyplot as plt

    print("\nExtraction des meilleurs paramètres depuis le CSV:")
    best_parameters = extract_best_parameters()
    print("\nDictionnaire des meilleurs paramètres:")
    print("{")
    for sector, (coeffs, thresholds, gain_moy) in best_parameters.items():
        print(f" '{sector}': (coefficients={coeffs}, thresholds={thresholds}), 'Gain moy={gain_moy}/50),")
    print("}")

    if verbose:
        print("\n🔍 Analyse des signaux pour actions populaires...")

    signals = []
    for i in range(0, len(popular_symbols), chunk_size):
        chunk = popular_symbols[i:i+chunk_size]
        if verbose:
            print(f"\n🔎 Lot {i//chunk_size + 1}: {', '.join(chunk)}")

        try:
            chunk_data = download_stock_data(chunk, period)
            for symbol, stock_data in chunk_data.items():
                prices = stock_data['Close']
                volumes = stock_data['Volume']

                if len(prices) < 50:
                    continue

                try:
                    info = yf.Ticker(symbol).info
                    domaine = info.get("sector", "ℹ️Inconnu!!")
                except Exception:
                    domaine = "ℹ️Inconnu!!"

                signal, last_price, trend, last_rsi, volume_mean, score = get_trading_signal(prices, volumes, domaine)

                if signal != "NEUTRE":
                    signals.append({
                        'Symbole': symbol,
                        'Signal': signal,
                        'Score': score,
                        'Prix': last_price,
                        'Tendance': "Hausse" if trend else "Baisse",
                        'RSI': last_rsi,
                        'Domaine': domaine,
                        'Volume moyen': volume_mean
                    })

        except Exception as e:
            if verbose:
                print(f"⚠️ Erreur: {str(e)}")

        time.sleep(1)

    # Affichage des résultats
    signaux_tries = {"ACHAT": {"Hausse": [], "Baisse": []}, "VENTE": {"Hausse": [], "Baisse": []}}

    if signals:
        if verbose:
            print("\n" + "=" * 115)
            print("RÉSULTATS DES SIGNEAUX")
            print("=" * 115)
            print(f"{'Symbole':<8} {'Signal':<8} {'Score':<7} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Volume moyen':<15} {'Domaine':<24} Analyse")
            print("-" * 115)

        for s in signals:
            if s['Signal'] in signaux_tries and s['Tendance'] in signaux_tries[s['Signal']]:
                signaux_tries[s['Signal']][s['Tendance']].append(s)

        for signal_type in ["ACHAT", "VENTE"]:
            for tendance in ["Hausse", "Baisse"]:
                if signaux_tries[signal_type][tendance]:
                    signaux_tries[signal_type][tendance].sort(key=lambda x: x['Prix'])

                    if verbose:
                        print(f"\n------------------------------------ Signal {signal_type} | Tendance {tendance} ------------------------------------")

                    for s in signaux_tries[signal_type][tendance]:
                        special_marker = "‼️ " if s['Symbole'] in mes_symbols else ""
                        analysis = ""
                        if s['Signal'] == "ACHAT":
                            analysis += "RSI bas" if s['RSI'] < 40 else ""
                            analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                        else:
                            analysis += "RSI élevé" if s['RSI'] > 60 else ""
                            analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""

                        print(f" {s['Symbole']:<8}{s['Signal']}{special_marker:<3} {s['Score']:<7.2f} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Volume moyen']:<15,.0f} {s['Domaine']:<24} {analysis}")

        if verbose:
            print("=" * 115)
    else:
        if verbose:
            print("\nℹ️ Aucun signal fort détecté parmi les actions populaires")
        return {}

    # Résumé du backtest sur les signaux détectés
    total_trades = 0
    total_gagnants = 0
    total_gain = 0.0
    cout_par_trade = 1.0
    backtest_results = []

    for s in signals:
        try:
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            # Ajout : récupération du domaine pour le backtest
            try:
                info = yf.Ticker(s['Symbole']).info
                domaine = info.get("sector", "Inconnu")
            except Exception:
                domaine = "Inconnu"

            if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
                if verbose:
                    print(f"{s['Symbole']:<8} : Données insuffisantes pour le backtest")
                continue

            resultats = backtest_signals(prices, volumes, domaine, montant=50)
            backtest_results.append({
                "Symbole": s['Symbole'],
                "trades": resultats['trades'],
                "gagnants": resultats['gagnants'],
                "taux_reussite": resultats['taux_reussite'],
                "gain_total": resultats['gain_total'],
                "gain_moyen": resultats['gain_moyen'],
                "drawdown_max": resultats['drawdown_max']
            })

            total_trades += resultats['trades']
            total_gagnants += resultats['gagnants']
            total_gain += resultats['gain_total']

        except Exception as e:
            if verbose:
                print(f"{s['Symbole']:<8} : Erreur backtest ({e})")

    backtest_results.sort(key=lambda x: x['taux_reussite'], reverse=True)

    if verbose:
        for res in backtest_results:
            print(
                f"{res['Symbole']:<8} | Trades: {res['trades']:<2} | "
                f"Gagnants: {res['gagnants']:<2} | "
                f"Taux réussite: {res['taux_reussite']:.0f}% | "
                f"Gain total: {res['gain_total']:.2f} $"
                f" | Gain moyen: {res['gain_moyen']:.2f} $ | "
                f"Drawdown max: {res['drawdown_max']:.2f}%"
            )

    cout_total_trades = total_trades * cout_par_trade
    total_investi_reel = len(backtest_results) * 50
    gain_total_reel = total_gain - cout_total_trades

    if total_trades > 0:
        taux_global = total_gagnants / total_trades * 100
        print("\n" + "="*115)
        print(f"🌍 Résultat global :")
        print(f" - Taux de réussite = {taux_global:.1f}%")
        print(f" - Nombre de trades = {total_trades}")
        print(f" - Total investi réel = {total_investi_reel:.2f} $ (50 $ par action analysée)")
        print(f" - Coût total des trades = {cout_total_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
        print(f" - Gain total brut = {total_gain:.2f} $")
        print(f" - Gain total net (après frais) = {gain_total_reel:.2f} $")
        print("="*115)
    else:
        print("\nAucun trade détecté pour le calcul global.")

    # === Affichage des taux de réussite par catégorie de domaine (secteur) pour le backtest ===
    domaine_stats = {}
    for res in backtest_results:
        symbole = res['Symbole']
        # Retrouver le domaine associé à ce symbole
        domaine = next((s['Domaine'] for s in signals if s['Symbole'] == symbole), "Inconnu")

        if domaine not in domaine_stats:
            domaine_stats[domaine] = {"trades": 0, "gagnants": 0, "gain_total": 0.0}

        domaine_stats[domaine]["trades"] += res["trades"]
        domaine_stats[domaine]["gagnants"] += res["gagnants"]
        domaine_stats[domaine]["gain_total"] += res["gain_total"]

    print("\n" + "="*115)
    print("📊 Taux de réussite par catégorie de domaine (backtest) :")
    print("="*115)
    print(f"{'Domaine':<25} {'Trades':<8} {'Gagnants':<10} {'Taux de réussite':<15} {'Gain brut':<12} {'Gain net':<12} {'Rentab. brute':<15}")
    print("-"*115)

    cout_par_trade = 1.0

    # Variables pour le total
    total_trades = 0
    total_gagnants = 0
    total_gain_brut = 0.0

    for domaine, stats in sorted(domaine_stats.items(), key=lambda x: -x[1]["trades"]):
        taux = (stats["gagnants"] / stats["trades"] * 100) if stats["trades"] else 0
        gain_brut = stats["gain_total"]
        gain_net = gain_brut - stats["trades"] * cout_par_trade
        investi = stats["trades"] * 50 if stats["trades"] else 1
        rentab_brute = (gain_brut / investi * 100) if investi else 0

        print(f"{domaine:<25} {stats['trades']:<8} {stats['gagnants']:<10} {taux:>10.1f} % {gain_brut:>10.2f} {gain_net:>10.2f} {rentab_brute:>10.1f} %")

        total_trades += stats["trades"]
        total_gagnants += stats["gagnants"]
        total_gain_brut += gain_brut

    # Ligne TOTAL
    total_taux = (total_gagnants / total_trades * 100) if total_trades else 0
    total_gain_net = total_gain_brut - total_trades * cout_par_trade
    total_investi = total_trades * 50 if total_trades else 1
    total_rentab_brute = (total_gain_brut / total_investi * 100) if total_investi else 0

    print("-"*115)
    print(f"{'TOTAL':<25} {total_trades:<8} {total_gagnants:<10} {total_taux:>10.1f} % {total_gain_brut:>10.2f} {total_gain_net:>10.2f} {total_rentab_brute:>10.1f} %")
    print("="*115)

    # Évaluation supplémentaire : stratégie filtrée

    # Appliquer les conditions
    # todo: ajuster les conditions selon les besoins
    # todo: creer un parametre pour le taux de reussite minimal actuel=60
    # todo: en faire un parametre de la fonction analyse_signaux_populaires
    taux_reussite_min = 30  # Valeur par défaut
    filtres = [res for res in backtest_results if res['taux_reussite'] >= taux_reussite_min and res['gain_total'] > 0]
    nb_actions_filtrees = len(filtres)
    total_trades_filtre = sum(res['trades'] for res in filtres)
    total_gagnants_filtre = sum(res['gagnants'] for res in filtres)
    total_gain_filtre = sum(res['gain_total'] for res in filtres)
    cout_total_trades_filtre = total_trades_filtre * cout_par_trade
    total_investi_filtre = nb_actions_filtrees * 50
    gain_total_reel_filtre = total_gain_filtre - cout_total_trades_filtre

    if verbose:
        print("\n" + "="*115)
        print(f"🔎 Évaluation si investissement SEULEMENT sur les actions à taux de réussite >= {taux_reussite_min}% ET gain total positif :")
        print(f" - Nombre d'actions sélectionnées = {nb_actions_filtrees}")
        print(f" - Nombre de trades = {total_trades_filtre}")
        print(f" - Taux de réussite global = {(total_gagnants_filtre / total_trades_filtre * 100) if total_trades_filtre else 0:.1f}%")
        print(f" - Total investi réel = {total_investi_filtre:.2f} $ (50 $ par action sélectionnée)")
        print(f" - Coût total des trades = {cout_total_trades_filtre:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
        print(f" - Gain total brut = {total_gain_filtre:.2f} $")
        print(f" - Gain total net (après frais) = {gain_total_reel_filtre:.2f} $")
        print("="*115)

    # Tableau des signaux pour actions fiables (>={taux_reussite_min}% taux de réussite) ou non encore évaluables


    fiables_ou_non_eval = set()
    for res in backtest_results:
        symbole = res['Symbole']
        
        # Récupérer le signal correspondant à ce symbole
        signal_info = next((s for s in signals if s['Symbole'] == symbole), None)
        
        if signal_info:
            # Récupérer le seuil d'achat pour le domaine
            _, seuils, _ = best_parameters.get(signal_info['Domaine'], (None, (4.2, -0.5), None))
            seuil_achat = seuils[0]
            seuil_vente = seuils[1]
            
            if res['taux_reussite'] >= taux_reussite_min or (res['trades'] == 0 and (signal_info['Score'] > 2 * seuil_achat or signal_info['Score'] < 2 * seuil_vente)):
                fiables_ou_non_eval.add(res['Symbole'])


    if verbose:
        print("\n" + "=" * 115)
        # todo: ajuster le texte selon les conditions appliquées ci-dessus
        print(f"SIGNES UNIQUEMENT POUR ACTIONS FIABLES (>={taux_reussite_min}% taux de réussite) OU NON ÉVALUÉES")
        print("=" * 115)
        print(f"{'Symbole':<8} {'Signal':<8} {'Score':<7} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Volume moyen':<15} {'Domaine':<24} Analyse")
        print("-" * 115)

        for signal_type in ["ACHAT", "VENTE"]:
            for tendance in ["Hausse", "Baisse"]:
                filtered = [
                    s for s in signaux_tries[signal_type][tendance]
                    if s['Symbole'] in fiables_ou_non_eval
                ]

                if filtered and verbose:
                    print(f"\n------------------------------------ Signal {signal_type} | Tendance {tendance} ------------------------------------")

                    for s in filtered:
                        special_marker = "‼️ " if s['Symbole'] in mes_symbols else ""
                        analysis = ""
                        if s['Signal'] == "ACHAT":
                            analysis += "RSI bas" if s['RSI'] < 40 else ""
                            analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                        else:
                            analysis += "RSI élevé" if s['RSI'] > 60 else ""
                            analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""

                        print(f" {s['Symbole']:<8} {s['Signal']}{special_marker:<3} {s['Score']:<7.2f} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Volume moyen']:<15,.0f} {s['Domaine']:<24} {analysis}")

        if verbose:
            print("=" * 115)

    signaux_valides = []
    for s in signals:
        # Vérifier si le symbole est dans la liste des fiables ou non évalués
        if s['Symbole'] in fiables_ou_non_eval:
            # Ajouter le taux de fiabilité si disponible
            taux_fiabilite = next(
                (res['taux_reussite'] for res in backtest_results if res['Symbole'] == s['Symbole']),
                "N/A"
            )

            # Créer une copie enrichie du signal
            signal_valide = s.copy()
            signal_valide['Fiabilite'] = taux_fiabilite
            signaux_valides.append(signal_valide)

    # =================================================================
    # SAUVEGARDE DES SIGNAUX VALIDÉS
    # =================================================================
    if signaux_valides and save_csv:
        if verbose:
            print(f"\n💾 Sauvegarde de {len(signaux_valides)} signaux validés par le backtest...")

        # Sauvegarde principale
        save_to_evolutive_csv(signaux_valides)

        # Sauvegarde spéciale pour vos symboles personnels
        mes_signaux_valides = [s for s in signaux_valides if s['Symbole'] in mes_symbols]
        if mes_signaux_valides:
            special_filename = f"mes_signaux_fiables_.csv"
            if verbose:
                print(f"💠 Sauvegarde de {len(mes_signaux_valides)} signaux personnels fiables dans {special_filename}")
            save_to_evolutive_csv(mes_signaux_valides, special_filename)

    # Affichage des graphiques pour les 5 premiers signaux d'achat et de vente FIABLES
    top_achats_fiables = []
    top_ventes_fiables = []

    for signal_type in ["ACHAT", "VENTE"]:
        for tendance in ["Hausse", "Baisse"]:
            filtered = [
                s for s in signaux_tries[signal_type][tendance]
                if s['Symbole'] in fiables_ou_non_eval
            ]

            if signal_type == "ACHAT":
                top_achats_fiables.extend(filtered)
            else:
                top_ventes_fiables.extend(filtered)

    top_achats_fiables = top_achats_fiables[:15]
    top_ventes_fiables = top_ventes_fiables[:15]

    fiabilite_dict = {res['Symbole']: res['taux_reussite'] for res in backtest_results}

    if afficher_graphiques and top_achats_fiables:
        print("\nAffichage des graphiques pour les 5 premiers signaux d'ACHAT FIABLES détectés (sur une même figure)...")

        fig, axes = plt.subplots(len(top_achats_fiables), 1, figsize=(14, 5 * len(top_achats_fiables)), sharex=False)
        if len(top_achats_fiables) == 1:
            axes = [axes]

        for i, s in enumerate(top_achats_fiables):
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            show_xaxis = (i == len(top_achats_fiables) - 1)  # True seulement pour le dernier subplot
            plot_unified_chart(s['Symbole'], prices, volumes, axes[i], show_xaxis=show_xaxis)

            # Dessiner les marqueurs d'achat/vente
            try:
                try:
                    info = yf.Ticker(s['Symbole']).info
                    domaine = info.get("sector", "Inconnu")
                except Exception:
                    domaine = "Inconnu"

                events = generate_trade_events(prices, volumes, domaine)
                for ev in events:
                    if ev.get('type') == 'BUY':
                        axes[i].scatter(ev['date'], ev['price'], marker='^', s=80, color='green', edgecolor='black', zorder=6)
                        axes[i].annotate('BUY', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,8), ha='center', fontsize=8, color='green')
                    elif ev.get('type') == 'SELL':
                        axes[i].scatter(ev['date'], ev['price'], marker='v', s=80, color='red', edgecolor='black', zorder=6)
                        axes[i].annotate('SELL', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,-10), ha='center', fontsize=8, color='red')
            except Exception:
                pass

            if len(prices) > 1:
                progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
                if isinstance(progression, pd.Series):
                    progression = float(progression.iloc[0])
            else:
                progression = 0.0

            try:
                info = yf.Ticker(s['Symbole']).info
                domaine = info.get("sector", "Inconnu")
            except Exception:
                domaine = "Inconnu"

            signal, last_price, trend, last_rsi, volume_mean, score = get_trading_signal(prices, volumes, domaine=domaine)

            taux_fiabilite = fiabilite_dict.get(s['Symbole'], None)
            fiabilite_str = f" | Fiabilité: {taux_fiabilite:.0f}%" if taux_fiabilite is not None else ""

            if last_price is not None:
                trend_symbol = "Haussière" if trend else "Baissière"
                rsi_status = "SURACH" if last_rsi > 72.5 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
                signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'
                special_marker = " ‼️" if s['Symbole'] in mes_symbols else ""

                title = (
                    f"{special_marker} {s['Symbole']} | Prix: {last_price:.2f} | Signal: {signal}({score}) {fiabilite_str} | "
                    f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
                    f"Progression: {progression:+.2f}% | Vol. moyen: {s['Volume moyen']:,.0f} units {special_marker}"
                )

                axes[i].set_title(title, fontsize=12, fontweight='bold', color=signal_color)

        plt.tight_layout()
        plt.subplots_adjust(top=0.96, hspace=0.152, bottom=0.032)
        plt.show()

    if afficher_graphiques and top_ventes_fiables:
        print("\nAffichage des graphiques pour les 5 premiers signaux de VENTE FIABLES détectés (sur une même figure)...")

        fig, axes = plt.subplots(len(top_ventes_fiables), 1, figsize=(14, 5 * len(top_ventes_fiables)), sharex=False)
        if len(top_ventes_fiables) == 1:
            axes = [axes]

        for i, s in enumerate(top_ventes_fiables):
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']

            show_xaxis = (i == len(top_ventes_fiables) - 1)  # True seulement pour le dernier subplot
            plot_unified_chart(s['Symbole'], prices, volumes, axes[i], show_xaxis=show_xaxis)

            # Dessiner les marqueurs d'achat/vente
            try:
                try:
                    info = yf.Ticker(s['Symbole']).info
                    domaine = info.get("sector", "Inconnu")
                except Exception:
                    domaine = "Inconnu"

                events = generate_trade_events(prices, volumes, domaine)
                for ev in events:
                    if ev.get('type') == 'BUY':
                        axes[i].scatter(ev['date'], ev['price'], marker='^', s=80, color='green', edgecolor='black', zorder=6)
                        axes[i].annotate('BUY', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,8), ha='center', fontsize=8, color='green')
                    elif ev.get('type') == 'SELL':
                        axes[i].scatter(ev['date'], ev['price'], marker='v', s=80, color='red', edgecolor='black', zorder=6)
                        axes[i].annotate('SELL', (ev['date'], ev['price']), textcoords='offset points', xytext=(0,-10), ha='center', fontsize=8, color='red')
            except Exception:
                pass

            if len(prices) > 1:
                progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
                if isinstance(progression, pd.Series):
                    progression = float(progression.iloc[0])
            else:
                progression = 0.0

            try:
                info = yf.Ticker(s['Symbole']).info
                domaine = info.get("sector", "Inconnu")
            except Exception:
                domaine = "Inconnu"

            signal, last_price, trend, last_rsi, volume_mean, score = get_trading_signal(prices, volumes, domaine=domaine)

            taux_fiabilite = fiabilite_dict.get(s['Symbole'], None)
            fiabilite_str = f" | Fiabilité: {taux_fiabilite:.0f}%" if taux_fiabilite is not None else ""

            if last_price is not None:
                trend_symbol = "Haussière" if trend else "Baissière"
                rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
                signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'
                special_marker = " ‼️" if s['Symbole'] in mes_symbols else ""

                title = (
                    f"{special_marker} {s['Symbole']} | Prix: {last_price:.2f} | Signal: {signal}({score}) {fiabilite_str} | "
                    f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
                    f"Progression: {progression:+.2f}% | Vol. moyen: {s['Volume moyen']:,.0f} units {special_marker}"
                )

                axes[i].set_title(title, fontsize=12, fontweight='bold', color=signal_color)

        plt.tight_layout()
        plt.subplots_adjust(top=0.96, hspace=0.152, bottom=0.032)
        plt.show()

    # Retourne les résultats pour usage ultérieur
    return {
        "signals": signals,
        "signaux_valides": signaux_valides,
        "signaux_tries": signaux_tries,
        "backtest_results": backtest_results,
        "fiables_ou_non_eval": fiables_ou_non_eval,
        "top_achats_fiables": top_achats_fiables,
        "top_ventes_fiables": top_ventes_fiables
    }

# ===================================================================
# FONCTIONS DE MAINTENANCE ET DIAGNOSTIC
# ===================================================================

if __name__ == "__main__":

    print("🛠️ OUTILS DE DIAGNOSTIC QSI INTELLIGENTE")
    print("=" * 50)
    print("Fonctions disponibles:")
    print("- cache_status_report()       # État du cache")
    print("- analyze_new_symbols_usage() # Analyse nouveaux symboles")
    print("- cleanup_cache(30)           # Nettoie cache > 30j")
    print("- warmup_cache()              # Pré-charge cache")
    print("\nLe système de cache intelligent est maintenant actif!")
    print("Toutes vos utilisations de download_stock_data() sont optimisées automatiquement.")