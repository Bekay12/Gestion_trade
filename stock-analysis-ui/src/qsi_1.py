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
from concurrent.futures import ThreadPoolExecutor

# Supprimer les avertissements FutureWarning de yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration du logger
logging.basicConfig(level=logging.INFO, filename='stock_analysis.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

OFFLINE_MODE = False  # Changez à True pour travailler hors ligne

def set_offline_mode(offline=True):
    """Active ou désactive le mode hors ligne globalement"""
    global OFFLINE_MODE
    OFFLINE_MODE = offline
    if offline:
        print("🔌 Mode hors ligne activé - utilisation du cache uniquement")
    else:
        print("🌐 Mode en ligne activé - téléchargement autorisé")

def work_offline():
    """Passe en mode hors ligne"""
    set_offline_mode(True)

def work_online():
    """Passe en mode en ligne"""
    set_offline_mode(False)

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
                       variation_seuil=-20, volume_seuil=100000):
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
    
    return signal, last_close, last_close > last_ema20, round(last_rsi,2), round(volume_mean, 2), round(score,3)

# Configuration du cache pour les données boursières
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def smart_cache_download(symbols, period, context="optimization"):
    """Cache intelligent basé sur vos 4 listes existantes"""
    from datetime import datetime
    from pathlib import Path
    
    # Config adaptée à VOS fichiers existants
    file_ages = {
        "mes_symbols.txt": 6,           # 6h pour vos investissements
        "popular_symbols.txt": 12,      # 12h pour les populaires  
        "optimisation_symbols.txt": 24, # 24h pour optimisation
        "test_symbols.txt": 48          # 48h pour tests
    }
    
    # Trouver la liste correspondante
    symbols_set = set(symbols)
    best_age = 24  # Par défaut
    best_file = "défaut"
    
    for filename, max_age in file_ages.items():
        try:
            with open(filename, 'r') as f:
                file_symbols = set(line.strip() for line in f if line.strip())
                overlap = len(symbols_set.intersection(file_symbols))
                if overlap >= len(symbols) * 0.6:  # 60% de correspondance
                    best_age = max_age
                    best_file = filename
                    break
        except FileNotFoundError:
            continue
    
    # Vérifier le cache
    cache_dir = Path("data_cache")
    fresh_count = sum(1 for symbol in symbols 
                     if (cache_dir / f"{symbol}_{period}.pkl").exists() 
                     and (datetime.now() - datetime.fromtimestamp((cache_dir / f"{symbol}_{period}.pkl").stat().st_mtime)).total_seconds() < best_age * 3600)
    
    # Décision
    use_offline = (fresh_count / len(symbols)) >= 0.7
    
    print(f"{'🚀 Cache' if use_offline else '🌐 Téléchargement'} ({best_file}): {fresh_count}/{len(symbols)} frais")
    
    # Appliquer
    import qsi
    old_mode = getattr(qsi, 'OFFLINE_MODE', False)
    qsi.OFFLINE_MODE = use_offline
    
    try:
        return qsi.download_stock_data(symbols, period)
    finally:
        qsi.OFFLINE_MODE = old_mode


def get_cached_data(symbol: str, period: str, max_age_hours: int = 6, offline_mode=None) -> pd.DataFrame:
    """Récupère les données en cache si elles existent, sinon télécharge (sauf en mode hors ligne)."""
    
    # Utiliser le mode global si non spécifié
    if offline_mode is None:
        offline_mode = OFFLINE_MODE
    
    cache_file = CACHE_DIR / f"{symbol}_{period}.pkl"
    
    # Si le fichier de cache existe
    if cache_file.exists():
        try:
            data = pd.read_pickle(cache_file)
            
            # En mode hors ligne, toujours utiliser le cache
            if offline_mode:
                age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
                print(f"📁 Cache utilisé pour {symbol} (âge: {age_hours:.1f}h)")
                return data
            
            # En mode en ligne, vérifier l'âge du cache
            age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if age < max_age_hours:
                return data
        except Exception as e:
            print(f"⚠️ Erreur lecture cache pour {symbol}: {e}")
    
    # Si on est en mode hors ligne et qu'il n'y a pas de cache
    if offline_mode:
        print(f"❌ Pas de cache disponible pour {symbol} ({period}) en mode hors ligne.")
        return pd.DataFrame()

    # Mode en ligne : télécharger et mettre en cache
    try:
        data = yf.download(symbol, period=period, progress=False)
        if not data.empty:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data.to_pickle(cache_file)
        return data
    except Exception as e:
        print(f"🚨 Erreur téléchargement {symbol}: {e}")
        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except:
                pass
        return pd.DataFrame()

    
def preload_cache(symbols, period):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_cached_data, symbol, period) for symbol in symbols]
        for f in futures:
            f.result()  # force le téléchargement/cache

def download_stock_data(symbols: List[str], period: str) -> Dict[str, Dict[str, pd.Series]]:
    """Version optimisée pour télécharger les données boursières avec cache.
    
    Args:
        symbols: Liste des symboles boursiers (ex: ['AAPL', 'MSFT']).
        period: Période des données (ex: '1y', '6mo', '1mo').
    
    Returns:
        Dictionnaire avec les données valides: {'symbol': {'Close': pd.Series, 'Volume': pd.Series}}.
    """
        # Utiliser le mode global si non spécifié
    if offline_mode is None:
        offline_mode = OFFLINE_MODE
        
    if offline_mode:
        print("🔌 Mode hors ligne activé - utilisation du cache uniquement")

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
    
    if offline_mode:
        for symbol in valid_symbols:
            try:
                data = get_cached_data(symbol, period, offline_mode=True)
                # ... (reste de la logique de validation)
            except Exception as e:
                print(f"🚨 Erreur pour {symbol}: {e}")
        return valid_data
    
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
        
        # Extraire d'abord la liste des symboles non présents dans all_data
        symbols_to_fetch = [s for s in batch if all_data is None or s not in all_data]


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
                    domaine: str, montant: float = 50, transaction_cost: float = 0.00, domain_coeffs=None,seuil_achat=None, seuil_vente=None) -> Dict:
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
        signal, *_ = get_trading_signal(prices[:i], volumes[:i], domaine,domain_coeffs=domain_coeffs,)
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

def save_symbols_to_txt(symbols: List[str], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(symbol + '\n')

def load_symbols_from_txt(filename: str) -> List[str]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Erreur de lecture du fichier {filename} : {e}")
        return []
    
def modify_symbols_file(filename: str, symbols_to_change: List[str], action: str):
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
# ======================================================================
# CONFIGURATION PRINCIPALE
# ======================================================================
# SIGNEAUX POUR ACTIONS POPULAIRES (version simplifiée)
# ======================================================================
period = "12mo"  #variable globale pour la période d'analyse ne pas commenter 
popular_symbols = load_symbols_from_txt("popular_symbols.txt")
mes_symbols = load_symbols_from_txt("mes_symbols.txt")
# popular_symbols = list(set(mes_symbols))
#preload_cache(popular_symbols + mes_symbols, period)

def analyse_signaux_populaires(
    popular_symbols, mes_symbols,
    period="12mo", afficher_graphiques=True,
    chunk_size=20, verbose=True,
    save_csv = True, plot_all=False
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
            print(f"    '{sector}': (coefficients={coeffs}, thresholds={thresholds}), 'Gain moy={gain_moy}/50),")
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

# if __name__ == "__main__":
#     start_time = time.time()
#     resultats = analyse_signaux_populaires(popular_symbols, mes_symbols, period="12mo", afficher_graphiques=True)
#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"\n⏱️ Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)")



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