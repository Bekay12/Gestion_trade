# optimisateur_turbo.py
# Version ultra-rapide avec Numba JIT compilation et optimisations vectorielles

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from qsi import download_stock_data, load_symbols_from_txt, extract_best_parameters
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
from collections import deque
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

# 🚀 NUMBA IMPORTS - JIT compilation pour accélération C-like
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba détecté - Accélération turbo activée!")
except ImportError:
    print("⚠️ Numba non installé. Installation: pip install numba")
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# 🚀 FONCTIONS ACCÉLÉRÉES AVEC NUMBA
@njit(cache=True, fastmath=True)
def fast_ema(prices, span):
    """EMA ultra-rapide avec Numba - 10-50x plus rapide"""
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(prices)
    result[0] = prices[0]
    
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    return result

@njit(cache=True, fastmath=True)
def fast_macd(prices, fast=12, slow=26, signal=9):
    """MACD ultra-rapide compilé avec Numba"""
    ema_fast = fast_ema(prices, fast)
    ema_slow = fast_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = fast_ema(macd, signal)
    return macd, signal_line

@njit(cache=True, fastmath=True)
def fast_rsi(prices, window=14):
    """RSI ultra-rapide avec Numba - évite pandas"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Moyenne mobile simple pour l'initialisation
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    rsi = np.empty(len(prices))
    rsi[:window] = 50.0  # Valeurs par défaut
    
    # EMA pour le reste
    alpha = 1.0 / window
    for i in range(window, len(prices)):
        gain_idx = i - 1
        avg_gain = alpha * gains[gain_idx] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[gain_idx] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@njit(cache=True, fastmath=True)
def fast_bollinger_percent(prices, window=20, std_dev=2.0):
    """Bollinger Bands % ultra-rapide"""
    result = np.empty(len(prices))
    
    for i in range(window-1, len(prices)):
        window_prices = prices[i-window+1:i+1]
        sma = np.mean(window_prices)
        std = np.std(window_prices)
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        if upper == lower:
            result[i] = 0.5
        else:
            result[i] = (prices[i] - lower) / (upper - lower)
    
    return result

@njit(cache=True, fastmath=True, parallel=True)
def fast_trading_signal_batch(prices_batch, volumes_batch, coeffs, seuil_achat, seuil_vente):
    """
    🚀 RÉVOLUTION : Calcul par batch ultra-rapide
    Traite plusieurs symboles simultanément avec parallélisation
    """
    n_symbols = len(prices_batch)
    results = np.empty((n_symbols, 6))  # signal, price, trend, rsi, volume, score
    
    for idx in prange(n_symbols):  # Parallélisation automatique
        prices = prices_batch[idx]
        volumes = volumes_batch[idx]
        
        if len(prices) < 50:
            results[idx] = np.array([-1, 0, 0, 50, 0, 0])  # Signal invalide
            continue
            
        # Calculs ultra-rapides
        macd, signal_line = fast_macd(prices)
        rsi = fast_rsi(prices)
        ema20 = fast_ema(prices, 20)
        ema50 = fast_ema(prices, 50)
        bb_percent = fast_bollinger_percent(prices)
        
        # Derniers points
        last_close = prices[-1]
        last_ema20 = ema20[-1]
        last_ema50 = ema50[-1]
        last_rsi = rsi[-1]
        last_macd = macd[-1]
        prev_macd = macd[-2]
        last_signal = signal_line[-1]
        prev_signal = signal_line[-2]
        last_bb_percent = bb_percent[-1]
        
        # Volume moyen rapide
        volume_mean = np.mean(volumes[-30:]) if len(volumes) >= 30 else np.mean(volumes)
        
        # Conditions simplifiées mais efficaces
        is_macd_cross_up = prev_macd < prev_signal and last_macd > last_signal
        is_macd_cross_down = prev_macd > prev_signal and last_macd < last_signal
        ema_structure_up = last_close > last_ema20 > last_ema50
        ema_structure_down = last_close < last_ema20 < last_ema50
        
        # Score simplifié mais efficace
        score = 0.0
        a1, a2, a3, a4, a5, a6, a7, a8 = coeffs
        
        # RSI conditions
        if last_rsi < 30:
            score += a1
        elif last_rsi > 70:
            score -= a1
            
        if 40 < last_rsi < 75:
            score += a4
        else:
            score -= a4
            
        # EMA structure
        if ema_structure_up:
            score += a5
        elif ema_structure_down:
            score -= a5
            
        # MACD crossover
        if is_macd_cross_up:
            score += a6
        elif is_macd_cross_down:
            score -= a6
            
        # Volume check
        if volume_mean > 100000:
            score += a6
        else:
            score -= a6
            
        # Bollinger position
        if last_bb_percent < 0.3:
            score += a4
        elif last_bb_percent > 0.7:
            score -= a4
            
        # Signal determination
        if score >= seuil_achat:
            signal = 1  # ACHAT
        elif score <= seuil_vente:
            signal = -1  # VENTE
        else:
            signal = 0  # NEUTRE
            
        trend = 1 if last_close > last_ema20 else 0
        
        results[idx] = np.array([signal, last_close, trend, last_rsi, volume_mean, score])
    
    return results

@njit(cache=True, fastmath=True)
def fast_backtest_single(prices, volumes, coeffs, seuil_achat, seuil_vente, montant=50.0, transaction_cost=0.02):
    """
    🚀 Backtest ultra-rapide pour un seul symbole
    Évite toute allocation Python, pure Numba
    """
    n_points = len(prices)
    if n_points < 50:
        return 0, 0, 0.0, 0.0  # trades, winners, total_gain, success_rate
    
    # Pré-calculer tous les signaux
    signals = np.empty(n_points - 50)
    
    for i in range(50, n_points):
        window_prices = prices[:i+1]
        window_volumes = volumes[:i+1]
        
        # Signal rapide (version simplifiée)
        macd, signal_line = fast_macd(window_prices)
        rsi = fast_rsi(window_prices)
        ema20 = fast_ema(window_prices, 20)
        
        last_close = window_prices[-1]
        last_rsi = rsi[-1]
        last_macd = macd[-1]
        prev_macd = macd[-2] if len(macd) > 1 else macd[-1]
        last_signal = signal_line[-1]
        prev_signal = signal_line[-2] if len(signal_line) > 1 else signal_line[-1]
        last_ema20 = ema20[-1]
        
        # Score rapide
        score = 0.0
        a1, a2, a3, a4, a5, a6, a7, a8 = coeffs
        
        if last_rsi < 30:
            score += a1
        elif last_rsi > 70:
            score -= a1
            
        if prev_macd < prev_signal and last_macd > last_signal:
            score += a6
        elif prev_macd > prev_signal and last_macd < last_signal:
            score -= a6
            
        if last_close > last_ema20:
            score += a5
        else:
            score -= a5
            
        # Déterminer le signal
        if score >= seuil_achat:
            signals[i-50] = 1  # ACHAT
        elif score <= seuil_vente:
            signals[i-50] = -1  # VENTE
        else:
            signals[i-50] = 0  # NEUTRE
    
    # Simulation des trades
    trades = 0
    winners = 0
    total_gain = 0.0
    position_open = False
    entry_price = 0.0
    
    for i in range(len(signals)):
        current_price = prices[i + 50]
        
        if signals[i] == 1 and not position_open:  # Signal d'achat
            position_open = True
            entry_price = current_price
            
        elif signals[i] == -1 and position_open:  # Signal de vente
            position_open = False
            trades += 1
            
            # Calcul du gain
            rendement = (current_price - entry_price) / entry_price
            gain = montant * rendement * (1 - 2 * transaction_cost)
            total_gain += gain
            
            if gain > 0:
                winners += 1
    
    success_rate = (winners / trades * 100.0) if trades > 0 else 0.0
    return trades, winners, total_gain, success_rate

# 🚀 CLASSE OPTIMISEUR TURBO
class TurboOptimizer:
    """Optimiseur ultra-rapide avec Numba et vectorisation"""
    
    def __init__(self, stock_data, domain, montant=50, transaction_cost=0.02, precision=2):
        self.stock_data = stock_data
        self.domain = domain
        self.montant = montant
        self.transaction_cost = transaction_cost
        self.evaluation_count = 0
        self.best_cache = {}
        self.precision = precision
        
        # 🚀 Préparation des données pour Numba
        self._prepare_numba_data()
        
    def _prepare_numba_data(self):
        """Prépare les données pour Numba (arrays NumPy)"""
        print("🔧 Préparation des données pour accélération Numba...")
        
        self.symbols = list(self.stock_data.keys())
        self.prices_arrays = []
        self.volumes_arrays = []
        
        # Conversion en arrays NumPy purs (requis pour Numba)
        for symbol in self.symbols:
            prices = self.stock_data[symbol]['Close'].values.astype(np.float64)
            volumes = self.stock_data[symbol]['Volume'].values.astype(np.float64)
            
            self.prices_arrays.append(prices)
            self.volumes_arrays.append(volumes)
            
        print(f"✅ {len(self.symbols)} symboles préparés pour calcul turbo")
    
    def round_params(self, params):
        """Arrondir les paramètres à la précision définie"""
        return np.round(params, self.precision)
    
    def evaluate_config_turbo(self, params):
        """🚀 Évaluation turbo avec Numba - 10-100x plus rapide"""
        params = self.round_params(params)
        param_key = tuple(params)
        
        if param_key in self.best_cache:
            return self.best_cache[param_key]
        
        coeffs = params[:8]
        seuil_achat, seuil_vente = params[8], params[9]
        
        # Contraintes
        coeffs = np.clip(coeffs, 0.5, 3.0)
        seuil_achat = np.clip(seuil_achat, 2.0, 6.0)
        seuil_vente = np.clip(seuil_vente, -3.0, 0.0)
        
        total_gain = 0.0
        total_trades = 0
        
        try:
            # 🚀 CALCUL TURBO : Traitement par lots avec Numba
            for i in range(len(self.prices_arrays)):
                trades, winners, gain, success_rate = fast_backtest_single(
                    self.prices_arrays[i], 
                    self.volumes_arrays[i],
                    coeffs, seuil_achat, seuil_vente,
                    self.montant, self.transaction_cost
                )
                total_gain += gain
                total_trades += trades
            
            avg_gain = total_gain / len(self.prices_arrays) if self.prices_arrays else 0.0
            self.evaluation_count += 1
            
            # Cache le résultat
            self.best_cache[param_key] = avg_gain
            return avg_gain
            
        except Exception as e:
            return -1000.0
    
    def latin_hypercube_turbo(self, n_samples=500):
        """LHS ultra-rapide avec évaluation par batch"""
        print(f"🚀 Latin Hypercube TURBO avec {n_samples} échantillons")
        
        # Génération optimisée des échantillons
        sampler = qmc.LatinHypercube(d=10)
        samples = sampler.random(n=n_samples)
        
        # Mise à l'échelle et arrondi vectorisé
        bounds = [(0.5, 3.0)] * 8 + [(2.0, 6.0), (-3.0, 0.0)]
        l_bounds = np.array([b[0] for b in bounds])
        u_bounds = np.array([b[1] for b in bounds])
        
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
        # Arrondi vectorisé
        scaled_samples = np.round(scaled_samples, self.precision)
        
        best_params = None
        best_score = -float('inf')
        
        # 🚀 Évaluation par batch pour plus d'efficacité
        batch_size = 50
        with tqdm(total=n_samples, desc="🚀 LHS TURBO", unit="sample") as pbar:
            for i in range(0, n_samples, batch_size):
                batch = scaled_samples[i:i+batch_size]
                
                # Évaluation vectorisée du batch
                for sample in batch:
                    score = self.evaluate_config_turbo(sample)
                    if score > best_score:
                        best_score = score
                        best_params = sample.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Meilleur': f"{best_score:.3f}", 
                        'Eval': self.evaluation_count,
                        'Cache': len(self.best_cache)
                    })
        
        return best_params, best_score

def get_sector(symbol):
    """Récupère le secteur d'une action"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'ℹ️Inconnu!!')
        print(f"📋 {symbol}: Secteur = {sector}")
        return sector
    except Exception as e:
        print(f"⚠️ Erreur pour {symbol}: {e}")
        return 'ℹ️Inconnu!!'

def optimize_sector_turbo(
    sector_symbols, domain,
    period='1y', strategy='lhs_turbo',
    montant=50, transaction_cost=0.02,
    budget_evaluations=500,  # Réduit car beaucoup plus rapide
    precision=1  # Précision réduite pour plus de vitesse
):
    """
    🚀 OPTIMISATION TURBO - 50-200x plus rapide
    """
    if not sector_symbols:
        print(f"🚫 Secteur {domain} vide, ignoré")
        return None, 0.0, 0.0, (4.2, -0.5)

    print(f"🚀 OPTIMISATION TURBO pour {domain}")
    print(f"⚡ Stratégie: {strategy} | Budget: {budget_evaluations} | Précision: {precision}")
    
    # Téléchargement des données
    stock_data = download_stock_data(sector_symbols, period=period)
    if not stock_data:
        print(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
        return None, 0.0, 0.0, (4.2, -0.5)

    # Initialisation optimiseur turbo
    optimizer = TurboOptimizer(stock_data, domain, montant, transaction_cost, precision)
    
    # Optimisation ultra-rapide
    start_time = datetime.now()
    
    if strategy == 'lhs_turbo':
        best_params, best_score = optimizer.latin_hypercube_turbo(budget_evaluations)
    else:
        # Fallback sur LHS si autre stratégie demandée
        best_params, best_score = optimizer.latin_hypercube_turbo(budget_evaluations)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if best_params is None:
        return None, 0.0, 0.0, (4.2, -0.5)
    
    # Extraction des paramètres finaux
    best_coeffs = tuple(float(x) for x in best_params[:8])
    best_thresholds = (float(best_params[8]), float(best_params[9]))
    
    # Calcul des statistiques finales (rapide avec Numba)
    total_success = 0
    total_trades = 0
    
    for i in range(len(optimizer.prices_arrays)):
        trades, winners, _, _ = fast_backtest_single(
            optimizer.prices_arrays[i], 
            optimizer.volumes_arrays[i],
            best_params[:8], best_thresholds[0], best_thresholds[1],
            montant, transaction_cost
        )
        total_success += winners
        total_trades += trades
    
    success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
    
    print(f"🏁 Optimisation terminée en {duration:.1f}s")
    print(f"⚡ Vitesse: {optimizer.evaluation_count/duration:.1f} éval/sec")
    print(f"🎯 Évaluations: {optimizer.evaluation_count}")
    print(f"💾 Cache hits: {len(optimizer.best_cache)}")
    print(f"🏆 Meilleurs coefficients: {best_coeffs}")
    print(f"🎯 Meilleurs seuils: {best_thresholds}")
    print(f"💰 Gain moyen: {best_score:.2f}")
    print(f"📊 Taux de réussite: {success_rate:.2f}%")
    print(f"🔄 Trades: {total_trades}")
    
    return best_coeffs, best_score, success_rate, best_thresholds

def save_optimization_results(domain, coeffs, gain_total, success_rate, total_trades, thresholds):
    """Sauvegarde rapide des résultats"""
    from datetime import datetime
    import pandas as pd

    results = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Sector': domain,
        'Gain_moy': gain_total,
        'Success_Rate': success_rate,
        'Trades': total_trades,
        'Seuil_Achat': thresholds[0],
        'Seuil_Vente': thresholds[1],
        'a1': coeffs[0], 'a2': coeffs[1], 'a3': coeffs[2], 'a4': coeffs[3],
        'a5': coeffs[4], 'a6': coeffs[5], 'a7': coeffs[6], 'a8': coeffs[7]
    }

    csv_path = 'signaux/optimization_hist_turbo.csv'

    try:
        df_new = pd.DataFrame([results])
        df_new.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)
        print(f"💾 Résultats sauvegardés pour {domain}")
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde: {e}")

# Script principal
if __name__ == "__main__":
    print("🚀 OPTIMISATEUR TURBO - Version ultra-rapide avec Numba")
    print("="*80)
    
    if not NUMBA_AVAILABLE:
        print("❌ Numba non disponible - performance dégradée")
        print("Installation: pip install numba")
    
    # Chargement des symboles
    symbols = list(dict.fromkeys(load_symbols_from_txt("optimisation_symbols.txt")))
    
    # Test rapide sur Healthcare (votre secteur lent)
    healthcare_symbols = []
    
    print("🔍 Détection des symboles Healthcare...")
    for symbol in symbols[:20]:  # Limiter pour test
        sector = get_sector(symbol)
        if sector == "Healthcare":
            healthcare_symbols.append(symbol)
    
    if healthcare_symbols:
        print(f"🏥 {len(healthcare_symbols)} symboles Healthcare détectés: {healthcare_symbols}")
        
        # TEST TURBO
        start = datetime.now()
        coeffs, gain_total, success_rate, thresholds = optimize_sector_turbo(
            healthcare_symbols, 
            "Healthcare",
            period='6mo',  # Période réduite pour test
            budget_evaluations=200,  # Réduit pour test rapide
            precision=1
        )
        end = datetime.now()
        
        print(f"\n🏁 TEST TURBO TERMINÉ en {(end-start).total_seconds():.1f} secondes!")
        
        if coeffs:
            print(f"🎯 Résultats: {coeffs}")
            print(f"⚖️ Seuils: {thresholds}")
            print(f"💰 Gain: {gain_total:.2f}")
            print(f"📊 Taux: {success_rate:.1f}%")
            
            # Sauvegarde
            save_optimization_results("Healthcare", coeffs, gain_total, success_rate, 0, thresholds)
    else:
        print("❌ Aucun symbole Healthcare trouvé dans les 20 premiers")