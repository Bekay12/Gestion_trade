# qsi.py - Analyse technique unifiée pour les actions avec MACD et RSI
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
    
    # Vérifier si le fichier existe déjà
    file_path = Path(filename)
    if file_path.exists():
        try:
            # Lire l'historique existant
            df_old = pd.read_csv(file_path)
            
            # Fusionner les nouveaux signaux avec l'historique
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # Supprimer les doublons en gardant la dernière version
            df_combined = df_combined.sort_values(
                by=['Symbole', 'detection_time'],
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
        archive_file = f"signaux_trading_{timestamp}.csv"
        df_clean.to_csv(archive_file, index=False)
        
        # Sauvegarde principale
        df_clean.to_csv(filename, index=False)
        print(f"💾 Signaux sauvegardés: {filename} (archive: {archive_file})")
    except Exception as e:
        print(f"🚨 Erreur sauvegarde CSV: {e}")

def get_trading_signal(prices, volumes,  domaine, domain_coeffs=None,
                       seuil_achat=5.75, seuil_vente=-0.5, 
                       variation_seuil=-20, volume_seuil=100000):
    """Détermine les signaux de trading avec validation des données"""
    # Correction : assure que prices et volumes sont bien des Series 1D
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    if len(prices) < 50:
        return "Données insuffisantes", None, None, None, None, None  # <-- 6 valeurs
    
    # Calcul des indicateurs
    macd, signal_line = calculate_macd(prices)
    #rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema50 = prices.ewm(span=50, adjust=False).mean()
    ema200 = prices.ewm(span=200, adjust=False).mean() if len(prices) >= 200 else ema50
    
    # Validation des derniers points
    if len(macd) < 2 or len(rsi) < 1:
        return "Données récentes manquantes", None, None, None, None, None  # <-- 6 valeurs
    

    last_close = prices.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_macd = macd.iloc[-1]
    prev_macd = macd.iloc[-2]
    last_signal = signal_line.iloc[-1]
    prev_signal = signal_line.iloc[-2]
    prev_rsi = rsi.iloc[-2]
    delta_rsi = last_rsi - prev_rsi
    
   # Performance long terme
    variation_30j = ((last_close - prices.iloc[-30]) / prices.iloc[-30]) * 100 if len(prices) >= 30 else None
    variation_180j = ((last_close - prices.iloc[-180]) / prices.iloc[-180]) * 100 if len(prices) >= 180 else None

    # Calcul du volume moyen sur 30 jours
    if len(volumes) >= 30:
        volume_mean = volumes.rolling(window=30).mean().iloc[-1]
        volume_std = volumes.rolling(window=30).std().iloc[-1]

        if isinstance(volume_mean, pd.Series):
            volume_mean = float(volume_mean.iloc[0])
            volume_std = float(volume_std.iloc[0]) if not volume_std.empty else 0.0
        else:
            volume_mean = float(volume_mean)
            volume_std = float(volume_std) 
    else:
        volume_mean = float(volumes.mean()) if len(volumes) > 0 else 0.0

    current_volume = volumes.iloc[-1]
    
    # Nouveaux indicateurs pour confirmation
    from ta.volatility import BollingerBands
    from ta.trend import ADXIndicator, IchimokuIndicator
    
    # Bollinger Bands
    bb = BollingerBands(close=prices, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_percent = (prices - bb_lower) / (bb_upper - bb_lower)
    last_bb_percent = bb_percent.iloc[-1] if len(bb_percent) > 0 else 0.5
    
    # ADX (Force de la tendance)
    adx_indicator = ADXIndicator(high=prices, low=prices, close=prices, window=14)
    adx = adx_indicator.adx()
    last_adx = adx.iloc[-1] if len(adx) > 0 else 0
    
    # Ichimoku Cloud (Tendance globale)
    ichimoku = IchimokuIndicator(high=prices, low=prices, window1=9, window2=26, window3=52)
    ichimoku_base = ichimoku.ichimoku_base_line()
    ichimoku_conversion = ichimoku.ichimoku_conversion_line()
    last_ichimoku_base = ichimoku_base.iloc[-1] if len(ichimoku_base) > 0 else last_close
    last_ichimoku_conversion = ichimoku_conversion.iloc[-1] if len(ichimoku_conversion) > 0 else last_close
    
     # Conditions d’achat optimisées
    is_macd_cross_up = prev_macd < prev_signal and last_macd > last_signal
    is_macd_cross_down = prev_macd > prev_signal and last_macd < last_signal
    is_volume_ok = volume_mean > volume_seuil

    is_variation_ok = variation_30j is not np.nan and variation_30j > variation_seuil
    tendance_haussière = last_close > last_ema20 > last_ema50 > last_ema200

        # RSI dynamique
    rsi_cross_up = prev_rsi < 30 and last_rsi >= 30
    rsi_cross_mid = prev_rsi < 50 and last_rsi >= 50
    rsi_cross_down = prev_rsi > 65 and last_rsi <= 65  # Seuil abaissé
    rsi_ok = last_rsi < 75 and last_rsi > 40  # pas suracheté, mais dynamique

    # EMA tendance
    ema_structure_up = last_close > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
    ema_structure_down = last_close < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]
    
    # Conditions supplémentaires pour confirmation
    strong_uptrend = (last_close > last_ichimoku_base) and (last_close > last_ichimoku_conversion)
    strong_downtrend = (last_close < last_ichimoku_base) and (last_close < last_ichimoku_conversion)
    adx_strong_trend = last_adx > 25  # Tendance forte
    
    score = 0  # Initialisation du score
    ################################### Conditions d'achat et de vente optimisées ###################################
    # if is_macd_cross_up and is_volume_ok and is_variation_ok and rsi.iloc[-1] < 75 :
    #     score += 2  # Signal d'achat fort
    #     # signal = "ACHAT"
    # elif is_macd_cross_down and rsi.iloc[-1] > 30:
    #     score -= 2  # Signal de vente fort
    #     #signal = "VENTE"
    # else:
    #     signal = "NEUTRE"

    ########################### Methode alternative avec score de signal ###########################
    # Définir les coefficients par domaine

    # Valeurs par défaut si domaine inconnu
    default_coeffs = (1.75, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75)
    if domain_coeffs:
        coeffs = domain_coeffs.get(domaine, default_coeffs)
    else:
        domain_coeffs = {
            "Technology":      (2.0, 1.5, 1.3, 1.5, 1.8, 1.5, 1.4, 1.2),   # (2.0, 1.2, 1.7, 1.5, 2.0, 1.5, 1.2, 2.0),
            "Healthcare":      (1.8, 1.1, 1.6, 1.3, 1.8, 1.3, 1.3, 1.8),
            "Financial Services": (1.6, 1.9, 2.3, 1.2, 1.6, 1.1, 1.2, 1.6),  #(1.6, 0.9, 1.3, 1.1, 1.6, 1.1, 0.9, 1.6),
            "Consumer Cyclical":  (1.7, 1.0, 1.4, 1.2, 1.7, 1.2, 1.7, 1.7),  #(1.7, 1.0, 1.4, 1.2, 1.7, 1.2, 1.0, 1.7),
            "Industrials":     (1.7, 1.2, 1.2, 1.0, 1.5, 1.0, 1.2, 1.5), #(1.5, 0.8, 1.2, 1.0, 1.5, 1.0, 0.8, 1.5),
            "Energy":          (1.4, 1.7, 1.1, 1.1, 1.4, 0.9, 0.7, 1.4), #(1.4, 0.7, 1.1, 0.9, 1.4, 0.9, 0.7, 1.4),
            "Basic Materials": (1.3, 1.6, 1.0, 1.8, 1.3, 0.8, 0.6, 1.3),
            "Communication Services": (1.6, 1.0, 1.3, 1.1, 1.6, 1.1, 1.0, 1.6), #(1.6, 1.0, 1.3, 1.1, 1.6, 1.1, 1.0, 1.6),
            "Utilities":       (1.8, 1.5, 1.2, 1.3, 1.2, 1.3, 0.5, 1.2), #(1.2, 0.5, 0.9, 0.7, 1.2, 0.7, 0.5, 1.2),
            "Real Estate":     (1.1, 1.6, 1.2, 1.4, 1.1, 1.4, 0.4, 1.1), #(1.1, 0.4, 0.8, 0.6, 1.1, 0.6, 0.4, 1.1),
            # Ajoutez d'autres domaines si besoin
        }
        # Sélectionner les coefficients selon le domaine
        coeffs = domain_coeffs.get(domaine, default_coeffs)


    a1, a2, a3, a4, a5, a6, a7, a8 = coeffs
    m1, m2, m3 = 1.0, 1.0, 1.0  # Coefficients pour ajuster l'importance des conditions de volume et tendance

    if adx_strong_trend: m1 = 1.5

    # Vérification du volume : Mesure à quel point le volume actuel s’éloigne de sa moyenne.
    z = (current_volume - volume_mean) / volume_std if volume_std > 0 else 0
    if z > 1.75:
        m2= 1.5
    elif z < -1.75:
        m2= 0.7
    else:
        m2=1.0
    
    volume_ratio = current_volume / volume_mean if volume_mean > 0 else 0
    if volume_ratio > 1.5:
        m3 = 1.5 
    elif volume_ratio < 0.5:
        m3 = 0.7

    # RSI : Signaux haussiers
    if rsi_cross_up: score += a1
    if  delta_rsi > 3 : score += m3*a2 # accélération RSI rsi_surge  # Momentum haussier
    if rsi_cross_mid: score += a3

    # RSI : Signaux baissiers
    if rsi_cross_down: score -= a1
    if delta_rsi < -3: score -= m3 * a2  # RSI chute rapide # RSI chute rapidement
 

    if rsi_ok: score += a4  # RSI dans une zone saine
    else: score -= a4  # RSI trop extrême

    # EMA : Structure de tendance
    if ema_structure_up: score += m1*a5
    if ema_structure_down: score -= m1*a5

    # MACD : Croisements
    if is_macd_cross_up: score += a6
    if is_macd_cross_down: score -= a6

    # Volume
    if is_volume_ok: score += m2*a6
    else: score -= m2*a6  # Manque de volume → faible conviction

    # Performance passée
    if is_variation_ok: score += a7
    else: score -= a7  # Sous-performance

    #  ## 🟢 CONDITIONS D'ACHAT Bonus
    # if (
    #     (rsi_cross_up or delta_rsi > 2.5 or rsi_cross_mid) and
    #     (is_macd_cross_up or ema_structure_up) and
    #     is_volume_ok and is_variation_ok and last_rsi < 75
    # ):
    #     score += 1.5

    # ### 🔴 CONDITIONS DE VENTE BONUS
    # elif (
    #     (rsi_cross_down or last_rsi > 70) and
    #     (is_macd_cross_down or ema_structure_down)
    # ):
    #     score -= 1.5

    

     # Facteurs d'achat

     # Conditions d'achat renforcées
    buy_conditions = (
        (is_macd_cross_up or ema_structure_up) and
        (rsi_cross_up or rsi_cross_mid) and
        (last_rsi < 65) and
        (last_bb_percent < 0.7) and  # Pas en zone de surachat
        (strong_uptrend or adx_strong_trend) and
        (volume_mean > volume_seuil) and
        (variation_30j > variation_seuil if variation_30j else True)
    )
    
    # Conditions de vente renforcées
    sell_conditions = (
        (is_macd_cross_down or ema_structure_down) and
        (rsi_cross_down or last_rsi > 70) and
        (last_rsi > 35) and  # Éviter les surventes extrêmes
        (last_bb_percent > 0.3) and  # Pas en zone de survente
        (strong_downtrend or adx_strong_trend) and
        (volume_mean > volume_seuil)
    )

    
    if strong_uptrend: score += 1.5
    if last_bb_percent < 0.4: score += 1.0  # Zone de survente
    if buy_conditions: score += 1.75  # Conditions d'achat renforcées

    

    if strong_downtrend: score -= 1.5
    if last_bb_percent > 0.6: score -= 1.0  # Zone de surachat
    if sell_conditions: score -= 1.75  # Conditions de vente renforcées

    # Interprétation du score
    if score >= seuil_achat:
        signal = "ACHAT"
    elif score <= seuil_vente:
        signal = "VENTE"
    else:
        signal = "NEUTRE"
    
    
    return signal, last_close, last_close > last_ema20, round(last_rsi,2), round(volume_mean, 2), score

# Configuration du cache pour les données boursières
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_data(symbol: str, period: str, max_age_hours: int = 6) -> pd.DataFrame:
    """Récupère les données en cache si elles existent et sont récentes, sinon télécharge.
    
    Args:
        symbol: Symbole boursier (ex: 'AAPL').
        period: Période des données (ex: '1y').
        max_age_hours: Âge maximum du cache en heures.
    
    Returns:
        pd.DataFrame avec les données, ou DataFrame vide si échec.
    """
    cache_file = CACHE_DIR / f"{symbol}_{period}.pkl"
    
    # Vérifier si le cache existe et est récent
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=max_age_hours):
            try:
                data = pd.read_pickle(cache_file)
                if not data.empty and 'Close' in data.columns and 'Volume' in data.columns:
                    # Vérifier que les données ont assez de points
                    if len(data) >= 50:
                        return data
                    else:
                        print(f"🚨 Cache pour {symbol} contient trop peu de données ({len(data)} points)")
            except Exception as e:
                print(f"🚨 Erreur lors de la lecture du cache pour {symbol}: {e}")
    
    # Télécharger et mettre en cache
    try:
        data = yf.download(symbol, period=period, progress=False, timeout=15)
        if not data.empty and 'Close' in data.columns and 'Volume' in data.columns:
            data.to_pickle(cache_file)
            return data
        else:
            print(f"🚨 Données téléchargées vides ou incomplètes pour {symbol}")
            return pd.DataFrame()
    except Exception as e:
        print(f"🚨 Erreur lors du téléchargement individuel pour {symbol}: {e}")
        return pd.DataFrame()

def download_stock_data(symbols: List[str], period: str) -> Dict[str, Dict[str, pd.Series]]:
    """Version optimisée pour télécharger les données boursières avec cache.
    
    Args:
        symbols: Liste des symboles boursiers (ex: ['AAPL', 'MSFT']).
        period: Période des données (ex: '1y', '6mo', '1mo').
    
    Returns:
        Dictionnaire avec les données valides: {'symbol': {'Close': pd.Series, 'Volume': pd.Series}}.
    """
    valid_data = {}
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '12mo', '1y', "18mo", "24mo", '2y', '5y', '10y', 'ytd', 'max']
    
    # Valider la période
    if period not in valid_periods:
        print(f"🚨 Période invalide: {period}. Valeurs possibles: {valid_periods}")
        return valid_data
    
    YAHOO_SUFFIXES = ('.HK', '.DE', '.PA', '.AS', '.SW', '.L', '.TO', '.V', '.MI', '.AX', '.SI',
    '.KQ', '.T', '.OL', '.HE', '.ST', '.CO', '.SA', '.MX', '.TW', '.JO', '.SZ', '.NZ', '.KS',
    '.PL', '.IR', '.MC', '.VI', '.BK', '.SS', '.SG', '.F', '.BE', '.CN', '.TA',  '-USD', '=F')
    
    # Filtrer les symboles potentiellement invalides
    valid_symbols = [s for s in symbols if s and ('.' not in s or s.endswith(YAHOO_SUFFIXES))]
    if len(valid_symbols) < len(symbols):
        invalid = set(symbols) - set(valid_symbols)
        print(f"🚨 Symboles ignorés (format invalide): {invalid}")
    
    # Diviser les symboles en lots
    batch_size = 100  # Ajustez selon les limites de l'API
    symbol_batches = [valid_symbols[i:i + batch_size] for i in range(0, len(valid_symbols), batch_size)]
    
    for batch in symbol_batches:
        try:
            # Téléchargement groupé
            all_data = yf.download(
                list(batch),
                period=period,
                group_by='ticker',
                progress=False,
                threads=True,  # Activer le multithreading
                timeout=30  # Timeout pour éviter les blocages
            )
        except Exception as e:
            print(f"🚨 Erreur lors du téléchargement groupé pour le lot {batch[:5]}...: {e}")
            all_data = None
        
        for symbol in batch:
            try:
                # Extraire les données du téléchargement groupé ou du cache
                if all_data is not None and symbol in all_data:
                    data = all_data[symbol]
                else:
                    data = get_cached_data(symbol, period)
                
                # Validation des données
                if data is None or data.empty:
                    print(f"🚨 Aucune donnée pour {symbol}")
                    continue
                
                if 'Close' not in data.columns or 'Volume' not in data.columns:
                    print(f"🚨 Données incomplètes pour {symbol}: colonnes manquantes")
                    continue
                
                # Vérifier la longueur minimale pour get_trading_signal
                if len(data) < 50:
                    print(f"🚨 Données insuffisantes pour {symbol} ({len(data)} points)")
                    continue
                
                # Nettoyer les données
                data = data[['Close', 'Volume']].copy()
                data['Close'] = data['Close'].ffill()  # Remplir les NaN dans Close
                data['Volume'] = data['Volume'].fillna(0)  # Remplir les NaN dans Volume par 0
                
                # Vérifier les NaN restants
                if data['Close'].isna().all() or data['Volume'].isna().all():
                    print(f"🚨 Données invalides pour {symbol}: trop de valeurs manquantes")
                    continue
                
                # Convertir en Series si nécessaire
                valid_data[symbol] = {
                    'Close': data['Close'].squeeze(),
                    'Volume': data['Volume'].squeeze()
                }
            except Exception as e:
                print(f"🚨 Erreur pour {symbol}: {e}")
    
    return valid_data


def backtest_signals(prices: Union[pd.Series, pd.DataFrame], volumes: Union[pd.Series, pd.DataFrame], 
                    domaine: str, montant: float = 50, transaction_cost: float = 0.00) -> Dict:
    """
    Effectue un backtest sur la série de prix.
    Un 'trade' correspond à un cycle complet ACHAT puis VENTE (entrée puis sortie).
    Le gain est calculé pour chaque cycle, avec prise en compte des frais de transaction.

    Args:
        prices: Série ou DataFrame des prix de clôture.
        volumes: Série ou DataFrame des volumes.
        domaine: Secteur de l'actif (ex: 'Technology').
        montant: Montant investi par trade (défaut: 50).
        transaction_cost: Frais de transaction par trade (défaut: 0.1%).

    Returns:
        Dict avec les métriques: trades, gagnants, taux_reussite, gain_total, gain_moyen, drawdown_max.
    """
    # Validation des entrées
    if not isinstance(prices, (pd.Series, pd.DataFrame)) or not isinstance(volumes, (pd.Series, pd.DataFrame)):
        return {
            "trades": 0,
            "gagnants": 0,
            "taux_reussite": 0,
            "gain_total": 0.0,
            "gain_moyen": 0.0,
            "drawdown_max": 0.0
        }
    
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    
    if len(prices) < 50 or len(volumes) < 50 or prices.isna().any() or volumes.isna().any():
        return {
            "trades": 0,
            "gagnants": 0,
            "taux_reussite": 0,
            "gain_total": 0.0,
            "gain_moyen": 0.0,
            "drawdown_max": 0.0
        }

    # Pré-calculer les signaux pour toute la série
    signals = []
    for i in range(50, len(prices)):
        signal, *_ = get_trading_signal(prices[:i], volumes[:i], domaine)
        signals.append(signal)
    signals = pd.Series(signals, index=prices.index[50:])

    # Simuler les trades
    positions = []
    for i in range(len(signals)):
        if signals.iloc[i] == "ACHAT":
            positions.append({"entry": prices.iloc[i + 50], "entry_idx": i + 50, "type": "buy"})
        elif signals.iloc[i] == "VENTE" and positions and "exit" not in positions[-1]:
            positions[-1]["exit"] = prices.iloc[i + 50]
            positions[-1]["exit_idx"] = i + 50

    # Fermer les positions ouvertes avec le dernier prix
    # if positions and "exit" not in positions[-1]:
    #     positions[-1]["exit"] = prices.iloc[-1]
    #     positions[-1]["exit_idx"] = len(prices) - 1

    # Calculer les métriques
    nb_trades = 0
    nb_gagnants = 0
    gain_total = 0.0
    gains = []
    portfolio_values = [montant]  # Suivi de la valeur du portefeuille

    for pos in positions:
        if "exit" in pos:
            nb_trades += 1
            entry = pos["entry"]
            exit = pos["exit"]
            rendement = (exit - entry) / entry
            # Ajuster pour les frais de transaction (entrée + sortie)
            gain = montant * rendement * (1 - 2 * transaction_cost)
            gain_total += gain
            gains.append(gain)
            if gain > 0:
                nb_gagnants += 1
            portfolio_values.append(portfolio_values[-1] + gain)

    # Calculer le drawdown maximum
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    drawdown_max = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0

    return {
        "trades": nb_trades,
        "gagnants": nb_gagnants,
        "taux_reussite": (nb_gagnants / nb_trades * 100) if nb_trades else 0,
        "gain_total": round(gain_total, 2),
        "gain_moyen": round(np.mean(gains), 2) if gains else 0.0,
        "drawdown_max": round(drawdown_max, 2)
    }

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
    
    # Ajout des signaux trading
    signal, last_price, trend, last_rsi, volume_moyen, score = get_trading_signal(prices, volumes, domaine=domaine)
   
    
    # Calcul de la progression en pourcentage
    if len(prices) > 1:
        progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
    else:
        progression = 0.0

    if last_price is not None:
        trend_symbol = "Haussière" if trend else "Baissière"
        rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
        signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'

        title = (
            f"{symbol} | Prix: {last_price:.2f} | Signal: {signal} ({score}) | "
            f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
            f"Progression: {progression:+.2f}% | Vol. moyen: {volume_moyen:,.0f} units "
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

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.152,bottom=0.032)
    plt.show()

# ======================================================================
# CONFIGURATION PRINCIPALE
# ======================================================================

period = "12mo"  #variable globale pour la période d'analyse ne pas commenter 

# Exemple d'utilisation (décommente pour exécuter) :
# symbols = ["SYM", "LHA.DE", "INGA.AS", "FLEX", "ALDX"]

# analyse_et_affiche(symbols, period)

# ======================================================================
# SIGNEAUX POUR ACTIONS POPULAIRES (version simplifiée)
# ======================================================================
popular_symbols = list(dict.fromkeys([
    # "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "BABA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CSCO", "WMT", "VZ", "KO", "SGMT",
    # "PEP", "MRK", "T", "NKE", "XOM", "CVX", "ABT", "CRM", "IBM", "ORCL", "MCD", "TMDX", "BA", "CAT", "GS", "RTX", "MMM", "HON", "LMT", "SBUX", "ADBE", "SMSN.L",
    "EEM", "INTU", "NOW", "ZM", "SHOP", "SNAP", "PFE", "TGT", "CVS", "WFC", "RHM.DE", "SAP.DE", "BAS.DE", "ALV.DE", "BMW.DE", "VOW3.DE", "SMTC", "ZS", "ZTS",
    "DTE.DE", "DBK.DE", "LHA.DE", "FME.DE", "BAYN.DE", "LIN.DE", "ENR.DE", "VNA.DE", "1COV.DE", "FRE.DE", "HEN3.DE", "HEI.DE", "RWE.DE", "VOW.DE", "GLW", "TMO",
    "DHR", "BAX", "MDT", "GE", "NOC", "GD", "HII", "TXT", "LHX", "TDY", "CARR", "OTIS", "JCI", "INOD", "BIDU", "JD", "PDD", "TCEHY", "NTES", "BILI", "HL",
    # "XPEV", "LI", "NIO", "BYDDF", "GME", "AMC", "BB", "NOK", "RBLX", "PLTR", "FSLY", "CRWD", "OKTA", "Z", "DOCU", "PINS", "SPOT", "LYFT", "UBER", "SNOW", "TTWO",
    # "VRSN", "WDAY", "2318.HK", "2382.HK", "2388.HK", "2628.HK", "3328.HK", "3988.HK", "9988.HK", "2319.HK", "0700.HK", "3690.HK", "ADSK", "02020.HK", "ABG","AN",
    # "9618.HK", "1810.HK", "1211.HK", "1299.HK", "2386.HK", "2385.HK", "0005.HK", "0011.HK", "0027.HK", "0038.HK", "0066.HK", "0083.HK", "MU", "GILT",
    # "0101.HK", "01177.HK", "0120.HK", "LSEG.L", "VOD.L", "BP.L", "HSBA.L",  "GSK.L", "ULVR.L", "AZN.L", "RIO.L", "BATS.L", "ADYEN.AS", "ABBN.SW",
    # "ASML.AS", "PHIA.AS", "INGA.AS", "MC.PA", "OR.PA", "AIR.PA", "BNP.PA", "SAN.PA", "ENGI.PA", "CAP.PA", "WELL", "O", "VICI", "ETOR", "ABR", "MOH.BE", "KSS",
    # "PLD", "PSA", "AMT", "CCI", "DLR", "EXR", "EQR", "ESS", "AVB", "MAA", "UDR", "SBRA", "UNH", "HD", "MA", "PG", "LLY", "COST", "AVGO", "ABBV", "QCOM",
    # "LONN.SW", "NOVN.SW", "ROG.SW", "ZURN.SW", "UBSG.SW", "QBTQ.CN","GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F", "ZC=F", "ZS=F", "ZL=F",
    # "DDOG", "CRL", "EXAS", "ILMN", "INCY", "MELI", "MRNA", "NTLA", "REGN", "ROKU", "QSI", "SYM", "IONQ", "QBTS", "RGTI", "SMCI", "TSM", "ALDX", "CSX", "LRCX", 
    # "BIIB", "CDNS", "CTSH", "EA", "FTNT", "GILD", "IDXX", "MP", "MTCH", "MRVL", "PAYX", "PTON", "AAL", "UAL", "DAL", "LUV", "JBLU", "ALK", "FLEX", "CACI",  
    # "CRIS", "CYTK", "EXEL", "FATE", "INSM", "KPTI", "NBIX", "NTRA", "PGEN", "RGEN", "SAGE", "SNY", "TGTX", "ARCT", "AXSM", "BMRN", "KTOS","BTC-USD", "ETH-USD",
    # "LTC-USD", "SOL-USD", "LINK-USD", "ATOM-USD", "TRX-USD", "COMP-USD","VEEV", "LEN", "PHM", "DHI", "KBH", "TOL", "NVR", "RMAX", "BURL", "TJX", "ROST", "VYGR",
    "LTC", "SOL", "LINK", "ATOM", "TRX", "COMP", "BTC", "ETH", "LB", "FINV", "HMC", "TM", "F", "GM", "TSLA", "RIVN", "LCID", "CGC", "CRON", "TLRY", "FSK", "PSEC",
    "MAIN", "ARCC", "ORC", "GBDC", "FDUS", "ALHF.PA", "PBR",
    "GLD", "SLV", "GDX", "GDXJ", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLC", "XLI", "XLB", "XLC", "XLV", "XLI", "XLP", "XLY","XLK", "XBI", "XHB", "URBN", "ANF",
    # "EZPW", "HNI", "COLL","LMB", "SCSC","CAR", "CARG", "CARS", "CVNA", "SAH", "GPI", "PAG", "RUSHA", "RUSHB", "LAD", "KMX", "CARV","SBLK","GOGL.OL", "SFL",
    #  "VYX", "CCCC", "AG", "AGI", "AGL", "AGM", "AGO","AGQ", "AGS", "AGX","DE", "DEO", "DES", "RR.L","RMS.PA", "ARG.PA", "RNO.PA", "AIR.PA", "ML.PA",
    # "FRO", "DHT", "STNG", "TNK", "GASS", "GLNG", "CMRE", "DAC", "ZIM","XMTR","JAKK","PANW","ETN", "EMR", "PH", "SWK", "FAST", "PNR", "XYL", "AOS","DOCN",
    # "VMEO", "GETY", "PUM.DE", "ETSY", "SSTK", "UDMY", "TDOC", "BARC.L", "LLOY.L", "STAN.L", "IMB.L", "GRPN", "CCRD", "LEU", "UEC", "CCJ", "AEO", "XRT",
     "NEM", "HMY", "KGC", "SAND", "WPM", "FNV", "RGLD", "GFI", "AEM", "NXE", "AU", "SIL", "GDXU", "GDXD", "GLDM", "IAU", "SGOL", "CDE", "EXK", "AGI.TO",
     "PHYS", "FNV.TO", "WDO.TO", "BOE", "JOBY", "LAC", "PLL", "ALB", "SQM", "RIOT", "MARA", "HUT", "BITF", "VKTX", "CRSR", "PFC.L", "OPEN", "FVRR"
    ]))

mes_symbols = ["QSI", "GLD","SYM","INGA.AS", "FLEX", "ALDX", "TSM", "02020.HK", "ARCT", "CACI", "ERJ", "PYPL", "GLW", "MSFT",
               "TMDX", "GILT", "ENR.DE", "META", "AMD", "ASML.AS", "TBLA", "VOOG", "WELL", "SMSN.L", "BMRN", "GS", "BABA",
               "SMTC", "AFX.DE", "ABBN.SW", "QCOM", "MP", "TM", "SGMT", "AMZN", "INOD", "SMCI", "GOOGL", "MU", "ETOR","DBK.DE", 
               "DDOG", "OKTA", "AXSM", "EEM", "SPY", "HMY", "2318.HK", "RHM.DE", "NVDA", "QBTS", "SAP.DE", "V", "UEC"]

# popular_symbols = list(set(mes_symbols))

def analyse_signaux_populaires(
    popular_symbols,
    mes_symbols,
    period="12mo",
    afficher_graphiques=True,
    chunk_size=20,
    verbose=True,
    save_csv = True,
    plot_all=False
):
    """
    Analyse les signaux pour les actions populaires, affiche les résultats, effectue le backtest,
    et affiche les graphiques pour les signaux fiables si demandé.
    Retourne un dictionnaire contenant les résultats principaux.
    """
    import matplotlib.pyplot as plt

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
            print(f"  - Taux de réussite = {taux_global:.1f}%")
            print(f"  - Nombre de trades = {total_trades}")
            print(f"  - Total investi réel = {total_investi_reel:.2f} $ (50 $ par action analysée)")
            print(f"  - Coût total des trades = {cout_total_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
            print(f"  - Gain total brut = {total_gain:.2f} $")
            print(f"  - Gain total net (après frais) = {gain_total_reel:.2f} $")
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
        print(f"{domaine:<25} {stats['trades']:<8} {stats['gagnants']:<10} {taux:>10.1f} %   {gain_brut:>10.2f}   {gain_net:>10.2f}   {rentab_brute:>10.1f} %")
        total_trades += stats["trades"]
        total_gagnants += stats["gagnants"]
        total_gain_brut += gain_brut

    # Ligne TOTAL
    total_taux = (total_gagnants / total_trades * 100) if total_trades else 0
    total_gain_net = total_gain_brut - total_trades * cout_par_trade
    total_investi = total_trades * 50 if total_trades else 1
    total_rentab_brute = (total_gain_brut / total_investi * 100) if total_investi else 0
    print("-"*115)
    print(f"{'TOTAL':<25} {total_trades:<8} {total_gagnants:<10} {total_taux:>10.1f} %   {total_gain_brut:>10.2f}   {total_gain_net:>10.2f}   {total_rentab_brute:>10.1f} %")
    print("="*115)

    # Évaluation supplémentaire : stratégie filtrée
    filtres = [res for res in backtest_results if res['taux_reussite'] >= 60 and res['gain_total'] > 0]
    nb_actions_filtrees = len(filtres)
    total_trades_filtre = sum(res['trades'] for res in filtres)
    total_gagnants_filtre = sum(res['gagnants'] for res in filtres)
    total_gain_filtre = sum(res['gain_total'] for res in filtres)
    cout_total_trades_filtre = total_trades_filtre * cout_par_trade
    total_investi_filtre = nb_actions_filtrees * 50
    gain_total_reel_filtre = total_gain_filtre - cout_total_trades_filtre
    if verbose:
        print("\n" + "="*115)
        print("🔎 Évaluation si investissement SEULEMENT sur les actions à taux de réussite >= 60% ET gain total positif :")
        print(f"  - Nombre d'actions sélectionnées = {nb_actions_filtrees}")
        print(f"  - Nombre de trades = {total_trades_filtre}")
        print(f"  - Taux de réussite global = {(total_gagnants_filtre / total_trades_filtre * 100) if total_trades_filtre else 0:.1f}%")
        print(f"  - Total investi réel = {total_investi_filtre:.2f} $ (50 $ par action sélectionnée)")
        print(f"  - Coût total des trades = {cout_total_trades_filtre:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
        print(f"  - Gain total brut = {total_gain_filtre:.2f} $")
        print(f"  - Gain total net (après frais) = {gain_total_reel_filtre:.2f} $")
        print("="*115)

    # Tableau des signaux pour actions fiables (>=60% taux de réussite) ou non encore évaluables
    fiables_ou_non_eval = set()
    for res in backtest_results:
        if res['taux_reussite'] >= 60 or res['trades'] == 0:
            fiables_ou_non_eval.add(res['Symbole'])
    if verbose:
        print("\n" + "=" * 115)
        print("SIGNES UNIQUEMENT POUR ACTIONS FIABLES (>=60% taux de réussite) OU NON ÉVALUÉES")
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
            special_filename = f"mes_signaux_fiables_{datetime.now().strftime('%Y%m%d')}.csv"
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
    top_achats_fiables = top_achats_fiables[:5]
    top_ventes_fiables = top_ventes_fiables[:5]
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
        plt.subplots_adjust(top=0.96, hspace=0.152,bottom=0.032)
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
                    f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status})  | "
                    f"Progression: {progression:+.2f}% | Vol. moyen: {s['Volume moyen']:,.0f} units {special_marker}"
                )
                axes[i].set_title(title, fontsize=12, fontweight='bold', color=signal_color)
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, hspace=0.152,bottom=0.032)
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

# Pour utiliser la fonction sans exécution automatique :

#if __name__ == "__main__":
start_time = time.time()
resultats = analyse_signaux_populaires(popular_symbols, mes_symbols, period="12mo", afficher_graphiques=True)
end_time = time.time()
elapsed = end_time - start_time
print(f"\n⏱️ Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)")



# # ======================================================================
# # Évaluation dynamique : investissement uniquement si l'action est "fiable" au moment du signal
# # ======================================================================
# print("\n" + "="*115)
# print("🔎 Simulation dynamique : investissement SEULEMENT si l'action est déjà >50% réussite ET gain positif au moment du signal")
# print("="*115)
# print(f"{'Symbole':<8} {'Entrée':<10} {'Sortie':<10} {'Résultat':<8} {'Gain($)':<10} {'Taux%':<7} {'GainTot($)':<10}")

# total_dyn_trades = 0
# total_dyn_gagnants = 0
# total_dyn_gain = 0.0
# cout_total_dyn_trades = 0
# actions_dyn = set()

# for s in signals:
#     try:
#         prices = download_stock_data([s['Symbole']], period)[s['Symbole']]
#         positions = []
#         for i in range(50, len(prices)):  # Commence après 50 points pour avoir un historique suffisant
#             past_prices = prices[:i]
#             past_volumes = Volume[:i]
#             # On effectue le backtest sur les 50 derniers points
#             stats = backtest_signals(past_prices, montant=50)
#             # On vérifie la fiabilité à ce moment précis
#             # if stats['trades'] > 0 and stats['taux_reussite'] > 70 and stats['gain_total'] > 0:
#             if stats['trades'] > 0 and stats['taux_reussite'] > 60 and stats['gain_total'] > 0:
#                 signal, _, _, _ = get_trading_signal(past_prices, past_volumes)
#                 if signal == "ACHAT":
#                     entry = prices.iloc[i]
#                     entry_idx = i
#                     # Cherche la prochaine sortie (VENTE)
#                     for j in range(i+1, len(prices)):
#                         next_signal, _, _, _ = get_trading_signal(prices[:j], volumes[:j])
#                         if next_signal == "VENTE":
#                             exit = prices.iloc[j]
#                             rendement = (exit - entry) / entry
#                             gain = 50 * rendement
#                             total_dyn_trades += 1
#                             cout_total_dyn_trades += cout_par_trade
#                             total_dyn_gain += gain
#                             actions_dyn.add(s['Symbole'])
#                             if gain > 0:
#                                 total_dyn_gagnants += 1
#                             print(f"{s['Symbole']:<8} {entry:>10.2f} {exit:>10.2f} {'Gagnant' if gain>0 else 'Perdant':<8} {gain:>10.2f} {stats['taux_reussite']:>7.1f} {stats['gain_total']:>10.2f}")
#                             break  # Passe au prochain signal d'achat
#     except Exception as e:
#         print(f"{s['Symbole']:<8} : Erreur simulation dynamique ({e})")

# nb_dyn_actions = len(actions_dyn)
# total_investi_dyn = nb_dyn_actions * 50
# gain_total_reel_dyn = total_dyn_gain - cout_total_dyn_trades
# taux_dyn = (total_dyn_gagnants / total_dyn_trades * 100) if total_dyn_trades else 0

# print("="*115)
# print(f"  - Nombre d'actions sélectionnées dynamiquement = {nb_dyn_actions}")
# print(f"  - Nombre de trades dynamiques = {total_dyn_trades}")
# print(f"  - Taux de réussite global = {taux_dyn:.1f}%")
# print(f"  - Total investi réel = {total_investi_dyn:.2f} $ (50 $ par action sélectionnée)")
# print(f"  - Coût total des trades = {cout_total_dyn_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
# print(f"  - Gain total brut = {total_dyn_gain:.2f} $")
# print(f"  - Gain total net (après frais) = {gain_total_reel_dyn:.2f} $")
# print("="*115)



#todo :# 1. Ajouter une option pour sauvegarder les signaux dans un fichier CSV
# 2. Ajouter une option pour afficher les graphiques des 5 premiers signaux d'achat et de vente finalement détectés