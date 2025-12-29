# optimisateur_hybride_fixed.py
# Version optimisée avec limitation des décimales pour réduire l'espace de recherche

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path
from typing import Dict  # Type hints for annotations
sys.path.append("C:\\Users\\berti\\Desktop\\Mes documents\\Gestion_trade\\stock-analysis-ui\\src\\trading_c_acceleration")
from qsi import download_stock_data, load_symbols_from_txt, extract_best_parameters
from qsi_optimized import backtest_signals, backtest_signals_with_events
from tqdm import tqdm
import qsi  # Import module pour accès aux caches globaux
from joblib import Parallel, delayed  # Parallélisation par symboles
# import yfinance as yf  # Import paresseux - chargé seulement si nécessaire
from collections import deque
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

# CMA-ES (optionnel): accélère la recherche en haute dimension si installé
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

# 🎯 CONFIGURATION VALIDATION & ANTI-OVERFITTING
MIN_SYMBOLS_PER_GROUP = 5  # Minimum de symboles par cellule secteur×cap
MAX_SYMBOLS_PER_GROUP = 12  # Maximum de symboles par groupe (échantillonnage si >12)
TRAIN_MONTHS = 18  # Période d'entraînement (mois)
VAL_MONTHS = 4  # Période de validation (mois)
HOLDOUT_MONTHS = 2  # Hold-out final (mois récents)
# Total = 24 mois (compatible yfinance: '24mo')
MIN_GAIN_PER_TRADE = 1.0  # Seuil minimal de gain par trade ($)
MAX_DRAWDOWN_PCT = 15.0  # Seuil maximal de drawdown (%)
MIN_TRADES_PER_YEAR = 3  # Minimum de trades par an
IGNORED_GROUPS_LOG = "ignored_groups.log"  # Log des groupes ignorés
POPULAR_SYMBOLS_FILE = "popular_symbols.txt"  # Fichier des symboles populaires

# Import du gestionnaire de symboles SQLite
try:
    from symbol_manager import (
        init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type, get_all_sectors,
        get_all_cap_ranges, get_symbols_by_sector_and_cap, get_symbol_count, get_sector_cached,
        classify_cap_range_for_symbol
    )
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ symbol_manager non disponible, utilisation de la méthode classique")
    SYMBOL_MANAGER_AVAILABLE = False
    get_sector_cached = None
    classify_cap_range_for_symbol = None

def precalculate_features(stock_data: Dict[str, pd.DataFrame], domain: str):
    """🚀 Pré-calcule toutes les features TA et dérivées pour tous les symbols.
    
    Remplit TA_CACHE et DERIV_CACHE de qsi.py pour éviter les recalculs pendant l'optimisation.
    Pré-charge aussi les fundamentals du SQLite pour éviter les requêtes yfinance répétées.
    
    Args:
        stock_data: Dict {symbol: {'Close': Series, 'Volume': Series}}
        domain: Secteur courant (pour logging)
    """
    print(f"\n⚙️  Pré-calcul des features pour {len(stock_data)} symbols ({domain})...")
    
    # Pré-charger les fundamentals du SQLite pour tous les symbols
    fundamentals_preloaded = 0
    try:
        from fundamentals_cache import get_fundamental_metrics
        for symbol in stock_data.keys():
            try:
                _ = get_fundamental_metrics(symbol, use_cache=True)
                fundamentals_preloaded += 1
            except Exception:
                pass  # Silencieux si fundamentals indisponibles
    except ImportError:
        pass  # fundamentals_cache optionnel
    
    for symbol, data in stock_data.items():
        prices = data['Close']
        volumes = data['Volume']
        
        if len(prices) < 50:
            continue
        
        # Trigger le calcul et la mise en cache via get_trading_signal
        # Cette fonction remplit automatiquement TA_CACHE et DERIV_CACHE
        try:
            result = qsi.get_trading_signal(
                prices, volumes, domain,
                domain_coeffs=None, domain_thresholds=None,
                variation_seuil=-20, volume_seuil=100000,
                return_derivatives=False,
                symbol=symbol, cap_range=None,
                price_extras=None
            )
            # Debug: vérifier si le cache s'est rempli
            cache_key = (str(symbol), round(float(prices.iloc[-1]), 2), round(float(volumes.iloc[-1]) if len(volumes) > 0 else 0, -2), int(len(prices)))
            if cache_key in qsi.TA_CACHE:
                print(f"      ✓ {symbol}: TA_CACHE rempli")
            else:
                print(f"      ✗ {symbol}: TA_CACHE manquant!")
        except Exception as e:
            print(f"      ✗ {symbol}: Exception - {str(e)[:100]}")
    
    ta_cached = len(qsi.TA_CACHE)
    deriv_cached = len(qsi.DERIV_CACHE)
    print(f"   ✅ {ta_cached} entrées TA_CACHE, {deriv_cached} entrées DERIV_CACHE pré-calculées")
    if fundamentals_preloaded > 0:
        print(f"   ✅ {fundamentals_preloaded} symbols fundamentals pré-chargés du SQLite\n")
    else:
        print()

def get_sector(symbol, use_cache=True):
    """Recupere le secteur d'une action avec cache intelligent.
    Wrapper autour de symbol_manager.get_sector_cached()."""
    if get_sector_cached is not None:
        return get_sector_cached(symbol, use_cache=use_cache)
    
    # Fallback si symbol_manager non disponible
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.info.get('sector', 'Unknown')
    except Exception:
        return 'Unknown'

def classify_cap_range(symbol: str) -> str:
    """Classe la capitalisation en categories. 
    Wrapper autour de symbol_manager.classify_cap_range_for_symbol()."""
    if classify_cap_range_for_symbol is not None:
        return classify_cap_range_for_symbol(symbol)
    
    # Fallback si symbol_manager non disponible
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        market_cap = ticker.info.get('marketCap')
        if market_cap is None:
            return 'Unknown'
        market_cap_b = market_cap / 1e9
        if market_cap_b < 2:
            return 'Small'
        if market_cap_b < 10:
            return 'Mid'
        if market_cap_b < 200:
            return 'Large'
        return 'Mega'
    except Exception:
        return 'Unknown'

def clean_sector_cap_groups(sector_cap_ranges: dict, min_symbols: int = MIN_SYMBOLS_PER_GROUP, max_symbols: int = MAX_SYMBOLS_PER_GROUP) -> tuple:
    """🧼 Nettoie les groupes secteur×cap: complète <min, limite >max, ignore impossibles.
    
    Args:
        sector_cap_ranges: Dict {sector: {cap_range: [symbols]}}
        min_symbols: Nombre minimum de symboles par groupe
        max_symbols: Nombre maximum de symboles par groupe
    
    Returns:
        tuple: (cleaned_groups, ignored_groups_log)
    """
    cleaned = {}
    ignored_log = []
    completion_log = []
    
    # Utiliser cache SQLite (TTL 100j) pour eviter recalcul secteur/cap via yfinance
    try:
        from symbol_manager import get_cleaned_group_cache, save_cleaned_group_cache
        use_cache = SYMBOL_MANAGER_AVAILABLE
    except:
        use_cache = False
    
    print(f"   Verif cache (TTL 100j)...")
    cache_hits = 0
    cache_misses = 0
    
    for sector, buckets in sector_cap_ranges.items():
        cleaned[sector] = {}
        for cap_range, syms in buckets.items():
            if not syms:
                continue
            
            # Essayer le cache d'abord
            if use_cache:
                try:
                    cached = get_cleaned_group_cache(sector, cap_range, ttl_days=100)
                    if cached is not None:
                        cleaned[sector][cap_range] = cached
                        cache_hits += 1
                        continue
                except Exception:
                    pass
            
            cache_misses += 1
            current_syms = list(syms)
            
            # Completion si <min
            if len(current_syms) < min_symbols:
                candidates = []
                if SYMBOL_MANAGER_AVAILABLE:
                    try:
                        candidates = get_symbols_by_sector_and_cap(sector, cap_range, list_type='popular', active_only=True)
                    except Exception:
                        pass
                
                # Exclure ceux déjà présents
                candidates = [c for c in candidates if c not in current_syms]
                
                # Ajouter jusqu'à min_symbols
                needed = min_symbols - len(current_syms)
                added = candidates[:needed]
                current_syms.extend(added)
                
                if added:
                    completion_log.append({
                        'sector': sector,
                        'cap_range': cap_range,
                        'original_count': len(syms),
                        'added': added,
                        'final_count': len(current_syms)
                    })
            
            # 2️⃣ Vérification finale: si toujours <min_symbols, ignorer
            if len(current_syms) < min_symbols:
                reason = f"Trop peu de symboles ({len(current_syms)} < {min_symbols}, complétion échouée)"
                ignored_log.append({
                    'sector': sector,
                    'cap_range': cap_range,
                    'count': len(current_syms),
                    'symbols': current_syms,
                    'reason': reason
                })
                continue
            
            # 3️⃣ Limitation si >max_symbols: échantillonnage aléatoire
            if len(current_syms) > max_symbols:
                original_count = len(current_syms)
                current_syms = random.sample(current_syms, max_symbols)
                completion_log.append({
                    'sector': sector,
                    'cap_range': cap_range,
                    'action': 'limited',
                    'original_count': original_count,
                    'sampled': current_syms,
                    'final_count': max_symbols
                })
            
            # ✅ Groupe valide : sauvegarder en cache
            cleaned[sector][cap_range] = current_syms
            if use_cache:
                try:
                    save_cleaned_group_cache(sector, cap_range, current_syms)
                except Exception:
                    pass
        
        if not cleaned[sector]:
            del cleaned[sector]
    
    # Afficher stats cache
    if cache_hits + cache_misses > 0:
        print(f"      Cache hits: {cache_hits}, misses: {cache_misses}")
    
    return cleaned, ignored_log

def log_ignored_groups(ignored_log: list, log_file: str = IGNORED_GROUPS_LOG):
    """Consigner les groupes ignorés dans un fichier log."""
    if not ignored_log:
        return
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# Groupes ignorés - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Minimum: {MIN_SYMBOLS_PER_GROUP} symboles/groupe\n\n")
        
        for entry in ignored_log:
            f.write(f"Secteur: {entry['sector']}\n")
            f.write(f"Cap Range: {entry['cap_range']}\n")
            f.write(f"Symboles ({entry['count']}): {', '.join(entry['symbols'])}\n")
            f.write(f"Raison: {entry['reason']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"   📝 {len(ignored_log)} groupes ignorés logés dans {log_file}")

def split_data_temporal(stock_data: dict, train_months: int = TRAIN_MONTHS, val_months: int = VAL_MONTHS, holdout_months: int = HOLDOUT_MONTHS) -> tuple:
    """📅 Split temporel des données: train / val / holdout.
    
    Args:
        stock_data: Dict {symbol: {'Close': Series, 'Volume': Series}}
        train_months: Mois pour training
        val_months: Mois pour validation
        holdout_months: Mois pour hold-out final
    
    Returns:
        tuple: (train_data, val_data, holdout_data, dates_info)
            - train_data: Dict {symbol: {'Close': Series, 'Volume': Series}}
            - val_data: Dict {symbol: {'Close': Series, 'Volume': Series}}
            - holdout_data: Dict {symbol: {'Close': Series, 'Volume': Series}}
            - dates_info: Dict avec les dates de split
    """
    train_data = {}
    val_data = {}
    holdout_data = {}
    
    # Calculer les dates de split depuis la date la plus récente
    latest_date = None
    for symbol, data_dict in stock_data.items():
        close_series = data_dict['Close']
        if latest_date is None or close_series.index[-1] > latest_date:
            latest_date = close_series.index[-1]
    
    # Définir les bornes temporelles
    holdout_start = latest_date - timedelta(days=holdout_months * 30)
    val_start = holdout_start - timedelta(days=val_months * 30)
    train_start = val_start - timedelta(days=train_months * 30)
    
    dates_info = {
        'train_start': train_start,
        'val_start': val_start,
        'holdout_start': holdout_start,
        'latest_date': latest_date
    }
    
    # Splitter chaque symbole
    for symbol, data_dict in stock_data.items():
        close_series = data_dict['Close']
        volume_series = data_dict['Volume']
        
        # Train: [train_start, val_start)
        train_mask = (close_series.index >= train_start) & (close_series.index < val_start)
        train_data[symbol] = {
            'Close': close_series[train_mask].copy(),
            'Volume': volume_series[train_mask].copy()
        }
        
        # Val: [val_start, holdout_start)
        val_mask = (close_series.index >= val_start) & (close_series.index < holdout_start)
        val_data[symbol] = {
            'Close': close_series[val_mask].copy(),
            'Volume': volume_series[val_mask].copy()
        }
        
        # Holdout: [holdout_start, latest]
        holdout_mask = close_series.index >= holdout_start
        holdout_data[symbol] = {
            'Close': close_series[holdout_mask].copy(),
            'Volume': volume_series[holdout_mask].copy()
        }
    
    return train_data, val_data, holdout_data, dates_info

def validate_configs_on_split(configs_with_scores: list, val_data: dict, domain: str, 
                             montant: float, transaction_cost: float,
                             top_k: int = 20) -> list:
    """✅ Valide les top-K configs sur les données de validation.
    
    Args:
        configs_with_scores: List of (params, train_score) tuples
        val_data: Dict {symbol: DataFrame} pour validation
        domain: Secteur
        montant, transaction_cost: Paramètres backtest
        top_k: Nombre de configs à valider
    
    Returns:
        list: [(params, train_score, val_score, val_metrics), ...] trié par val_score
    """
    # Trier par train_score et prendre top-K
    sorted_configs = sorted(configs_with_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    validated = []
    print(f"\n✅ Validation de {len(sorted_configs)} meilleures configs sur période val...")
    
    for params, train_score in tqdm(sorted_configs, desc="🔍 Validation"):
        # Évaluer sur val
        val_gain = 0.0
        val_trades = 0
        val_successes = 0
        val_drawdown = 0.0
        
        coeffs = tuple(params[:8])
        feature_thresholds = tuple(params[8:16])
        seuil_achat = float(params[16])
        seuil_vente = float(params[17])
        
        for symbol, data in val_data.items():
            try:
                result, _ = backtest_signals_with_events(
                    data['Close'], data['Volume'], domain,
                    montant=montant, transaction_cost=transaction_cost,
                    domain_coeffs={domain: coeffs},
                    domain_thresholds={domain: feature_thresholds},
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    extra_params=None,
                    fundamentals_extras=None,
                    symbol_name=symbol
                )
                val_gain += result['gain_total']
                val_trades += result.get('trades', 0)
                if result.get('taux_reussite', 0) > 0:
                    val_successes += 1
                # TODO: calculer drawdown si disponible
            except Exception:
                continue
        
        avg_val_gain = val_gain / len(val_data) if val_data else 0.0
        gain_per_trade = val_gain / val_trades if val_trades > 0 else 0.0
        
        val_metrics = {
            'avg_gain': avg_val_gain,
            'total_trades': val_trades,
            'gain_per_trade': gain_per_trade,
            'success_rate': (val_successes / len(val_data) * 100) if val_data else 0.0,
            'drawdown': val_drawdown
        }
        
        validated.append((params, train_score, avg_val_gain, val_metrics))
    
    # Trier par val_score
    validated.sort(key=lambda x: x[2], reverse=True)
    return validated

def apply_validation_threshold(validated_configs: list, min_gain_per_trade: float = MIN_GAIN_PER_TRADE,
                              max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
                              min_trades_per_year: int = MIN_TRADES_PER_YEAR) -> list:
    """⛔ Filtre les configs qui ne passent pas les seuils de validation.
    
    Args:
        validated_configs: List from validate_configs_on_split
        min_gain_per_trade: Seuil minimal gain/trade
        max_drawdown_pct: Seuil maximal drawdown
        min_trades_per_year: Seuil minimal trades/an
    
    Returns:
        list: Configs filtrées qui passent les seuils
    """
    filtered = []
    for params, train_score, val_score, val_metrics in validated_configs:
        # Appliquer les seuils
        if val_metrics['gain_per_trade'] < min_gain_per_trade:
            continue
        if val_metrics['total_trades'] < min_trades_per_year / 2:  # Sur 6 mois = moitié de l'année
            continue
        # TODO: vérifier drawdown si disponible
        
        filtered.append((params, train_score, val_score, val_metrics))
    
    if not filtered:
        print("   ⚠️ Aucune config ne passe les seuils de validation!")
        # Fallback: retourner la meilleure même si elle échoue
        if validated_configs:
            return [validated_configs[0]]
    
    return filtered

def get_best_gain_csv(domain, csv_path='signaux/optimization_hist_4stp.csv'):
    """Récupère le meilleur gain moyen historique pour le secteur dans le CSV."""
    try:
        if pd.io.common.file_exists(csv_path):
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
            sector_data = df[df['Sector'] == domain]
            if not sector_data.empty:
                return sector_data['Gain_moy'].max()
    except Exception as e:
        print(f"⚠️ Erreur chargement CSV pour {domain}: {e}")
    return -float('inf')


def classify_cap_range(symbol: str) -> str:
    """Classe la capitalisation en 4 catégories (Small, Mid, Large, Mega) ou Unknown."""
    try:
        import yfinance as yf  # Import paresseux
        ticker = yf.Ticker(symbol)
        market_cap = ticker.info.get('marketCap')
        if market_cap is None:
            return 'Unknown'

        market_cap_b = market_cap / 1e9
        if market_cap_b < 2:
            return 'Small'
        if market_cap_b < 10:
            return 'Mid'
        if market_cap_b < 200:
            return 'Large'
        return 'Mega'
    except Exception:
        return 'Unknown'

# ----
# Top-level objective for SciPy DE (picklable on Windows)
def _de_objective(params, optimizer):
    """Objective wrapper for differential_evolution.
    Uses a top-level function so it can be pickled when workers are used.
    """
    try:
        rounded = optimizer.round_params(params)
        return -optimizer.evaluate_config(rounded)
    except Exception:
        # Penalize any evaluation failure to keep DE robust
        return 1000.0

def _evaluate_single_symbol(symbol: str, data: dict, domain: str, params: np.ndarray, 
                            montant: float, transaction_cost: float,
                            have_active_price: bool, have_active_fund: bool,
                            price_extras: dict, fundamentals_extras: dict) -> tuple:
    """🚀 Évalue le backtest pour UN SEUL symbole (appelé en parallèle).
    
    Args:
        symbol: Symbole boursier
        data: Dict avec 'Close' et 'Volume' Series
        domain: Secteur
        params: Vecteur de paramètres (coeffs, seuils)
        montant, transaction_cost: Paramètres backtest
        have_active_price, have_active_fund: Flags pour activer extras
        price_extras, fundamentals_extras: Dicts d'extras optionnels
    
    Returns:
        tuple: (gain_total, num_trades) pour ce symbole
    """
    # Extraire les 18 paramètres de base
    coeffs = tuple(params[:8])
    feature_thresholds = tuple(params[8:16])
    seuil_achat = float(params[16])
    seuil_vente = float(params[17])
    
    try:
        result, _ = backtest_signals_with_events(
            data['Close'], data['Volume'], domain,
            montant=montant, transaction_cost=transaction_cost,
            domain_coeffs={domain: coeffs},
            domain_thresholds={domain: feature_thresholds},
            seuil_achat=seuil_achat, seuil_vente=seuil_vente,
            extra_params=(price_extras if have_active_price else None),
            fundamentals_extras=(fundamentals_extras if have_active_fund else None),
            symbol_name=symbol
        )
        return result['gain_total'], result.get('trades', 0)  # 🚀 Retourner aussi les trades
    except Exception as e:
        return 0.0, 0  # Défaut si erreur

class HybridOptimizer:
    """Optimiseur hybride utilisant plusieurs stratégies d'optimisation avec limitation des décimales"""
    
    def __init__(self, stock_data, domain, montant=50, transaction_cost=1.0, precision=2, use_price_features: bool = False, use_fundamentals_features: bool = False, n_jobs: int = 1):
        self.stock_data = stock_data
        self.domain = domain
        self.montant = montant
        self.transaction_cost = transaction_cost
        self.evaluation_count = 0
        self.best_cache = {}
        self.precision = precision  # 🔧 NUOVO: Précision des paramètres (nombre de décimales)
        self.use_price_features = use_price_features
        self.use_fundamentals_features = use_fundamentals_features
        self.n_jobs = n_jobs  # 🚀 Nombre de workers pour parallélisation par symboles (1 = séquentiel)
        
        # 🔧 Définir les bounds (18 de base: 8 coeffs + 8 seuils + 2 globaux)
        # a3 (RSI cross mid) gelé, et les thresholds binaires inutiles sont figés
        coeff_bounds = [(0.5, 3.0)] * 8
        coeff_bounds[2] = (0.0, 0.0)  # a3 inutilisé (RSI cross mid)

        threshold_bounds = [
            (30.0, 70.0),   # threshold 0: RSI_threshold (index 8)
            (0.0, 0.0),     # threshold 1: MACD_threshold (gelé)
            (0.0, 0.0),     # threshold 2: EMA_threshold (gelé)
            (0.5, 2.5),     # threshold 3: Volume_threshold (index 11)
            (15.0, 35.0),   # threshold 4: ADX_threshold (index 12)
            (0.0, 0.0),     # threshold 5: Ichimoku_threshold (gelé)
            (0.5, 0.5),     # threshold 6: Bollinger_threshold (fixé au mid-band)
            (2.0, 6.0),     # threshold 7: Score_global_threshold (index 15)
        ]

        base_bounds = (
            coeff_bounds +
            threshold_bounds +
            [(2.0, 6.0)] +     # seuil_achat global (index 16)
            [(-6.0, -2.0)]     # seuil_vente global (index 17)
        )
        self.bounds = base_bounds
        
        if self.use_price_features:
            # Extra 6 params: flags (rounded to 0/1), weights a9/a10, thresholds th9/th10 on relative values
            price_bounds = [
                (0.0, 1.0),  # use_price_slope (index 18)
                (0.0, 1.0),  # use_price_acc   (index 19)
                (0.0, 3.0),  # a9 weight       (index 20)
                (0.0, 3.0),  # a10 weight      (index 21)
                (-0.05, 0.05),  # th9 on price_slope_rel (index 22)
                (-0.05, 0.05),  # th10 on price_acc_rel  (index 23)
            ]
            self.bounds += price_bounds
        
        if self.use_fundamentals_features:
            # Extra 10 params for fundamentals: 1 flag, 5 weights, 4 thresholds
            # Indices: 24+ (or 18+ if price_features disabled)
            fundamentals_bounds = [
                (0.0, 1.0),    # use_fundamentals flag (index 24 or 18)
                (0.0, 3.0),    # a_rev_growth weight
                (0.0, 3.0),    # a_eps_growth weight
                (0.0, 3.0),    # a_roe weight
                (0.0, 3.0),    # a_fcf_yield weight
                (0.0, 3.0),    # a_de_ratio weight
                (-30.0, 30.0),  # th_rev_growth
                (-30.0, 30.0),  # th_eps_growth
                (-30.0, 30.0),  # th_roe
                (-10.0, 10.0),  # th_fcf_yield
            ]
            self.bounds += fundamentals_bounds
        
        # ✨ V2.0: Charger les paramètres optimisés existants comme point de départ
        self.optimized_coeffs_loaded = False
        self.initial_coeffs = None
        self.initial_thresholds = None
        self.meilleur_trades = 0  # 🔧 Stocker le nombre de trades de la meilleure config
        
    def round_params(self, params):
        """🔧 NOUVEAU: Arrondir les paramètres à la précision définie"""
        return np.round(params, self.precision)
    
    def evaluate_config(self, params):
        """Évalue une configuration de paramètres avec arrondi"""
        # 🔧 MODIFIÉ: Arrondir les paramètres avant évaluation
        params = self.round_params(params)
        
        # Éviter les réévaluations inutiles avec précision réduite
        param_key = tuple(params)
        if param_key in self.best_cache:
            return self.best_cache[param_key]

        # Extraire les paramètres : 8 coeffs + 8 seuils feature + 2 seuils globaux
        coeffs = list(params[:8])
        feature_thresholds = list(params[8:16])  # 8 seuils individuels (certains figés)
        seuil_achat = float(params[16])  # Seuil global achat
        seuil_vente = float(params[17])  # Seuil global vente

        # Contraintes avec arrondi sur les coefficients
        coeffs = list(np.clip(self.round_params(coeffs), 0.5, 3.0))
        coeffs[2] = 0.0  # a3 gelé
        coeffs = tuple(coeffs)
        
        # Contraintes sur les seuils features
        feature_thresholds[0] = np.clip(round(feature_thresholds[0], self.precision), 30.0, 70.0)  # RSI_threshold
        feature_thresholds[1] = 0.0  # MACD_threshold gelé
        feature_thresholds[2] = 0.0  # EMA_threshold gelé
        feature_thresholds[3] = np.clip(round(feature_thresholds[3], self.precision), 0.5, 2.5)    # Volume_threshold
        feature_thresholds[4] = np.clip(round(feature_thresholds[4], self.precision), 15.0, 35.0)  # ADX_threshold
        feature_thresholds[5] = 0.0  # Ichimoku_threshold gelé
        feature_thresholds[6] = 0.5  # Bollinger_threshold fixé
        feature_thresholds[7] = np.clip(round(feature_thresholds[7], self.precision), 2.0, 6.0)    # Score_global_threshold
        feature_thresholds = tuple(feature_thresholds)
        
        # Contraintes sur les seuils globaux
        seuil_achat = np.clip(round(seuil_achat, self.precision), 2.0, 6.0)
        seuil_vente = np.clip(round(seuil_vente, self.precision), -6.0, -2.0)

        # Extract optional price extras
        price_extras = None
        if self.use_price_features and len(params) >= 24:
            # Round flags to 0/1
            use_ps = int(round(np.clip(params[18], 0.0, 1.0)))
            use_pa = int(round(np.clip(params[19], 0.0, 1.0)))
            a_ps = float(np.clip(round(params[20], self.precision), 0.0, 3.0))
            a_pa = float(np.clip(round(params[21], self.precision), 0.0, 3.0))
            th_ps = float(np.clip(round(params[22], self.precision), -0.05, 0.05))
            th_pa = float(np.clip(round(params[23], self.precision), -0.05, 0.05))
            price_extras = {
                'use_price_slope': use_ps,
                'use_price_acc': use_pa,
                'a_price_slope': a_ps,
                'a_price_acc': a_pa,
                'th_price_slope': th_ps,
                'th_price_acc': th_pa,
            }
        
        # Extract optional fundamentals extras
        fundamentals_extras = None
        fundamentals_index_offset = 24 if self.use_price_features else 18
        if self.use_fundamentals_features and len(params) >= (fundamentals_index_offset + 10):
            use_fund = int(round(np.clip(params[fundamentals_index_offset], 0.0, 1.0)))
            a_rev = float(np.clip(round(params[fundamentals_index_offset + 1], self.precision), 0.0, 3.0))
            a_eps = float(np.clip(round(params[fundamentals_index_offset + 2], self.precision), 0.0, 3.0))
            a_roe = float(np.clip(round(params[fundamentals_index_offset + 3], self.precision), 0.0, 3.0))
            a_fcf = float(np.clip(round(params[fundamentals_index_offset + 4], self.precision), 0.0, 3.0))
            a_de = float(np.clip(round(params[fundamentals_index_offset + 5], self.precision), 0.0, 3.0))
            th_rev = float(np.clip(round(params[fundamentals_index_offset + 6], self.precision), -30.0, 30.0))
            th_eps = float(np.clip(round(params[fundamentals_index_offset + 7], self.precision), -30.0, 30.0))
            th_roe = float(np.clip(round(params[fundamentals_index_offset + 8], self.precision), -30.0, 30.0))
            th_fcf = float(np.clip(round(params[fundamentals_index_offset + 9], self.precision), -10.0, 10.0))
            fundamentals_extras = {
                'use_fundamentals': use_fund,
                'a_rev_growth': a_rev,
                'a_eps_growth': a_eps,
                'a_roe': a_roe,
                'a_fcf_yield': a_fcf,
                'a_de_ratio': a_de,
                'th_rev_growth': th_rev,
                'th_eps_growth': th_eps,
                'th_roe': th_roe,
                'th_fcf_yield': th_fcf,
                'th_de_ratio': 0.0,  # Placeholder for 5th threshold (reserved)
            }

        # 🔧 Mise à jour: utiliser toujours le backtest "with_events" pour cohérence
        # Même lorsque les features étendues sont désactivées, on passe extras=None.
        have_active_price = bool(self.use_price_features and price_extras is not None and (
            int(price_extras.get('use_price_slope', 0) or 0) or int(price_extras.get('use_price_acc', 0) or 0)
        ))
        have_active_fund = bool(self.use_fundamentals_features and fundamentals_extras is not None and (
            int(fundamentals_extras.get('use_fundamentals', 0) or 0)
        ))

        total_gain = 0.0
        total_trades = 0
        try:
            # 🚀 PARALLÉLISATION PAR SYMBOLES
            if self.n_jobs != 1 and len(self.stock_data) > 1:
                # Utiliser joblib pour évaluation parallèle
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(_evaluate_single_symbol)(
                        symbol, data, self.domain, params,
                        self.montant, self.transaction_cost,
                        have_active_price, have_active_fund,
                        price_extras, fundamentals_extras
                    )
                    for symbol, data in self.stock_data.items()
                )
                gains = [r[0] for r in results]  # 🚀 Extraire les gains
                trades = [r[1] for r in results]  # 🚀 Extraire les trades
                total_gain = sum(gains)
                total_trades = sum(trades)  # 🚀 Accumuler les trades
            else:
                # Évaluation séquentielle (original)
                for symbol, data in self.stock_data.items():
                    # Utiliser systématiquement la version avec événements pour des règles identiques
                    result, _ = backtest_signals_with_events(
                        data['Close'], data['Volume'], self.domain,
                        montant=self.montant, transaction_cost=self.transaction_cost,
                        domain_coeffs={self.domain: coeffs},
                        domain_thresholds={self.domain: feature_thresholds},
                        seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                        extra_params=(price_extras if have_active_price else None),
                        fundamentals_extras=(fundamentals_extras if have_active_fund else None),
                        symbol_name=symbol
                    )
                    total_gain += result['gain_total']
                    total_trades += result['trades']

            avg_gain = total_gain / len(self.stock_data) if self.stock_data else 0.0
            self.evaluation_count += 1
            
            # 🔧 Tracker le nombre de trades de la meilleure config
            if param_key not in self.best_cache or avg_gain > self.best_cache[param_key]:
                self.meilleur_trades = total_trades

            # Cache le résultat
            self.best_cache[param_key] = avg_gain
            return avg_gain

        except Exception as e:
            print(f"⚠️ evaluate_config error: {e}")  # Debug: show exceptions
            return -1000.0  # Pénalité pour configurations invalides

    def genetic_algorithm(self, population_size=50, generations=30, mutation_rate=0.15, seed=None):
        """Algorithme génétique pour l'optimisation avec précision limitée"""
        print(f"🧬 Démarrage algorithme génétique (pop={population_size}, gen={generations}, précision={self.precision})")
        
        # Utiliser self.bounds (16 paramètres: 8 coefficients + 8 seuils individuels)
        bounds = self.bounds
        population = []
        for idx in range(population_size):
            individual = []
            for low, high in bounds:
                # 🔧 MODIFIÉ: Génération avec pas discret selon la précision
                if self.precision == 1:
                    step = 0.1
                elif self.precision == 2:
                    step = 0.05
                else:
                    step = 0.01
                
                # Génération par pas discrets
                n_steps = int((high - low) / step)
                random_step = np.random.randint(0, n_steps + 1)
                value = low + random_step * step
                individual.append(round(value, self.precision))
            candidate = np.array(individual)
            # 🔧 Injecter la graine historique comme premier individu si fournie
            if seed is not None and idx == 0:
                try:
                    seed_arr = np.array(seed, dtype=float)
                    seed_arr = np.clip(seed_arr, [b[0] for b in bounds], [b[1] for b in bounds])
                    candidate = self.round_params(seed_arr)
                except Exception:
                    pass
            population.append(candidate)

        best_fitness = -float('inf')
        best_individual = None
        best_trades = 0  # 🚀 Tracker les trades du meilleur score

        with tqdm(total=generations, desc="🧬 Évolution génétique", unit="gen") as pbar:
            for gen in range(generations):
                # Évaluation
                fitness_scores = [self.evaluate_config(ind) for ind in population]

                # Sélection des meilleurs
                fitness_indices = np.argsort(fitness_scores)[::-1]
                elite_size = population_size // 4
                elite = [population[i] for i in fitness_indices[:elite_size]]

                # Mise à jour du meilleur
                current_best = fitness_scores[fitness_indices[0]]
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_individual = population[fitness_indices[0]].copy()
                    best_trades = self.meilleur_trades  # 🚀 Capture les trades du nouveau meilleur

                # Nouvelle génération
                new_population = elite.copy()
                while len(new_population) < population_size:
                    # Sélection par tournoi
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Croisement
                    child1, child2 = self._crossover(parent1, parent2)

                    # Mutation
                    if np.random.random() < mutation_rate:
                        child1 = self._mutate(child1, bounds)
                    if np.random.random() < mutation_rate:
                        child2 = self._mutate(child2, bounds)

                    new_population.extend([child1, child2])

                population = new_population[:population_size]

                pbar.set_postfix({'Meilleur': f"{best_fitness:.3f}", 'Trades': best_trades})
                pbar.update(1)

        return self.round_params(best_individual), best_fitness

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Sélection par tournoi"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2, alpha=0.3):
        """Croisement BLX-α avec arrondi"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val

            low = min_val - alpha * range_val
            high = max_val + alpha * range_val

            # 🔧 MODIFIÉ: Arrondir les enfants
            child1[i] = round(np.random.uniform(low, high), self.precision)
            child2[i] = round(np.random.uniform(low, high), self.precision)

        return child1, child2

    def _mutate(self, individual, bounds, sigma=0.1):
        """Mutation gaussienne avec arrondi"""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < 0.1:  # Probabilité de mutation par gène
                noise = np.random.normal(0, sigma * (bounds[i][1] - bounds[i][0]))
                new_value = individual[i] + noise
                # 🔧 MODIFIÉ: Arrondir et contraindre
                mutated[i] = round(np.clip(new_value, bounds[i][0], bounds[i][1]), self.precision)
        return mutated

    def differential_evolution_opt(self, population_size=45, max_iterations=100, seed=None):
        """Optimisation par évolution différentielle avec arrondi
        
        Args:
            population_size: Taille de la population
            max_iterations: Nombre d'itérations
            seed: Vecteur de paramètres initial (warm-start) pour aider la convergence
        """
        print(f"🔄 Démarrage évolution différentielle (pop={population_size}, iter={max_iterations}, précision={self.precision})")
        
        bounds = self.bounds
        dim = len(bounds)

        # Cap initial population size to avoid huge first generation (popsize * dim)
        # Target around ~600 evaluations per generation
        target_evals_per_gen = 600
        max_popsize = max(8, target_evals_per_gen // max(1, dim))
        if population_size > max_popsize:
            print(f"   ⚠️ Pop trop grande pour {dim} dimensions → réduit {population_size} → {max_popsize}")
            population_size = max_popsize

        # 🌱 Préparer le seed comme point de départ (warm-start)
        init_candidates = None
        if seed is not None:
            try:
                seed_arr = np.array(seed, dtype=float)
                seed_arr = np.clip(seed_arr, [b[0] for b in bounds], [b[1] for b in bounds])
                seed_arr = self.round_params(seed_arr)
                init_candidates = seed_arr
                print(f"   🌱 Warm-start avec seed historique")
            except Exception as e:
                print(f"   ⚠️ Impossible d'utiliser seed: {e}")

        best_de_trades = [0]  # 🚀 Tracker les trades du meilleur score en DE (list pour closure)
        with tqdm(total=max_iterations, desc="🔄 Évolution différentielle", unit="iter") as pbar:
            def callback(xk, convergence):
                best_de_trades[0] = self.meilleur_trades  # 🚀 Capture les trades du meilleur
                pbar.set_postfix({'Convergence': f"{convergence:.6f}", 'Trades': best_de_trades[0]})
                pbar.update(1)

            de_kwargs = {
                'maxiter': max_iterations,
                'popsize': population_size,
                'mutation': (0.5, 1.5),
                'recombination': 0.7,
                'callback': callback,
                'polish': False,
                'seed': np.random.randint(0, 10000),
                'workers': -1,  # Use all cores; safe thanks to top-level objective
            }
            
            # 🌱 Ajouter le seed si disponible
            if init_candidates is not None:
                de_kwargs['init'] = 'latinhypercube'  # Utiliser LHC + ajouter seed
                # Note: on ajoute le seed via la stratégie, pas via init directement
                # car scipy DE n'a pas de paramètre x0 pour point de départ unique
                # Solution: on évalue le seed séparément et on le compare au meilleur trouvé
            
            result = differential_evolution(
                _de_objective,
                bounds,
                args=(self,),
                **de_kwargs
            )

        # 🌱 Si seed fourni, comparer le seed au résultat DE
        best_x = self.round_params(result.x)
        best_f = -result.fun
        
        if init_candidates is not None:
            seed_score = self.evaluate_config(init_candidates)
            if seed_score > best_f:
                print(f"   ℹ️ Seed meilleur que DE (seed={seed_score:.3f} vs DE={best_f:.3f}), en utilisant le seed")
                best_x = init_candidates
                best_f = seed_score
        
        return best_x, best_f

    def latin_hypercube_sampling(self, n_samples=500, seed=None):
        """Échantillonnage Latin Hypercube avec arrondi"""
        print(f"🎯 Latin Hypercube Sampling avec {n_samples} échantillons (précision={self.precision})")
        bounds = self.bounds

        # Gérer les paramètres figés (l==u) : on ne les fait pas varier dans le LHS
        var_idx = [i for i, (l, u) in enumerate(bounds) if u > l]
        fixed_idx = [i for i, (l, u) in enumerate(bounds) if u == l]

        if not var_idx:
            # Tout est figé : un seul point
            scaled_samples = np.array([[b[0] for b in bounds]] * n_samples, dtype=float)
        else:
            sampler = qmc.LatinHypercube(d=len(var_idx))
            samples = sampler.random(n=n_samples)
            l_bounds = [bounds[i][0] for i in var_idx]
            u_bounds = [bounds[i][1] for i in var_idx]
            var_scaled = qmc.scale(samples, l_bounds, u_bounds)

            # Reconstituer la dimension complète en réinjectant les valeurs figées
            scaled_samples = np.zeros((n_samples, len(bounds)), dtype=float)
            # Remplir par défaut avec les bornes inf (qui == sup pour les figées)
            for j, (l, _) in enumerate(bounds):
                scaled_samples[:, j] = l
            # Injecter les dimensions variables
            for col, idx in enumerate(var_idx):
                scaled_samples[:, idx] = var_scaled[:, col]

        # 🔧 Arrondir les échantillons
        scaled_samples = np.array([self.round_params(sample) for sample in scaled_samples])

        best_params = None
        best_score = -float('inf')
        best_trades = 0  # 🚀 Tracker les trades du meilleur score
        
        # 🔧 Évaluer d'abord une graine historique si fournie
        if seed is not None:
            try:
                seed_arr = np.array(seed, dtype=float)
                seed_arr = np.clip(seed_arr, [b[0] for b in bounds], [b[1] for b in bounds])
                seed_arr = self.round_params(seed_arr)
                seed_score = self.evaluate_config(seed_arr)
                best_params = seed_arr.copy()
                best_score = seed_score
                best_trades = self.meilleur_trades  # Capture les trades du seed
            except Exception:
                pass

        with tqdm(total=n_samples, desc="🎯 LHS Exploration", unit="sample") as pbar:
            for sample in scaled_samples:
                score = self.evaluate_config(sample)
                if score > best_score:
                    best_score = score
                    best_params = sample.copy()
                    best_trades = self.meilleur_trades  # 🚀 Capture les trades du nouveau meilleur

                pbar.set_postfix({'Meilleur': f"{best_score:.3f}", 'Trades': best_trades})
                pbar.update(1)

        return best_params, best_score

    def cma_es_optimization(self, lhs_samples=1000, top_k=5, max_generations=20, pop_size=None):
        """CMA-ES avec warm-start LHS. Nécessite le package 'cma'."""
        if not CMA_AVAILABLE:
            print("⚠️ CMA-ES indisponible (package 'cma' manquant). Fallback LHS.")
            return self.latin_hypercube_sampling(lhs_samples)

        dim = len(self.bounds)

        # 1) Warm-start via LHS en gérant les bornes figées
        lhs_samples = max(lhs_samples, top_k)
        print(f"🚀 CMA-ES warm-start: LHS {lhs_samples} échantillons, top-{top_k} pour initialiser")

        var_idx = [i for i, (l, u) in enumerate(self.bounds) if u > l]
        fixed_idx = [i for i, (l, u) in enumerate(self.bounds) if u == l]

        if not var_idx:
            scaled = np.array([[b[0] for b in self.bounds]] * lhs_samples, dtype=float)
        else:
            sampler = qmc.LatinHypercube(d=len(var_idx))
            samples = sampler.random(n=lhs_samples)
            l_bounds = [self.bounds[i][0] for i in var_idx]
            u_bounds = [self.bounds[i][1] for i in var_idx]
            var_scaled = qmc.scale(samples, l_bounds, u_bounds)

            scaled = np.zeros((lhs_samples, dim), dtype=float)
            for j, (l, _) in enumerate(self.bounds):
                scaled[:, j] = l
            for col, idx in enumerate(var_idx):
                scaled[:, idx] = var_scaled[:, col]

        scaled = np.array([self.round_params(s) for s in scaled])

        scores = []
        for s in scaled:
            scores.append(self.evaluate_config(s))
        scores = np.array(scores)
        best_indices = np.argsort(scores)[::-1][:top_k]
        best_candidates = scaled[best_indices]
        best_scores = scores[best_indices]

        x0 = best_candidates[0]
        spread = (u_bounds - l_bounds)
        sigma = max(1e-3, float(np.median(spread) * 0.15))
        pop_size = pop_size or int(4 + 3 * np.log(dim))

        print(f"   CMA-ES init score={best_scores[0]:.3f}, sigma={sigma:.4f}, pop_size={pop_size}")
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma, {
            'bounds': [l_bounds.tolist(), u_bounds.tolist()],
            'popsize': pop_size,
            'maxiter': max_generations,
            'verb_disp': 0,
        })

        best_params = x0
        best_score = best_scores[0]
        best_cma_trades = [0]  # 🚀 Tracker les trades du meilleur score en CMA-ES (list pour closure)

        with tqdm(total=max_generations, desc="🌌 CMA-ES", unit="gen") as pbar:
            for _ in range(max_generations):
                candidates = es.ask()
                fitness = []
                for c in candidates:
                    c_arr = np.array(c, dtype=float)
                    c_arr = np.clip(c_arr, l_bounds, u_bounds)
                    c_arr = self.round_params(c_arr)
                    score = self.evaluate_config(c_arr)
                    fitness.append(-score)  # cma minimise
                    if score > best_score:
                        best_score = score
                        best_params = c_arr.copy()
                        best_cma_trades[0] = self.meilleur_trades  # 🚀 Capture les trades du nouveau meilleur
                es.tell(candidates, fitness)
                pbar.set_postfix({'Meilleur': f"{best_score:.3f}", 'Trades': best_cma_trades[0]})
                pbar.update(1)
                if es.stop():
                    break

        return best_params, best_score

    def particle_swarm_optimization(self, n_particles=30, max_iterations=50, seed=None):
        """Optimisation par essaim particulaire (PSO) avec arrondi"""
        print(f"🐝 Particle Swarm Optimization (particles={n_particles}, iter={max_iterations}, précision={self.precision})")
        
        bounds = np.array(self.bounds)

        # Initialisation avec arrondi
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, len(self.bounds)))
        particles = np.array([self.round_params(p) for p in particles])  # 🔧 MODIFIÉ
        # 🔧 Injecter la graine historique si disponible
        if seed is not None:
            try:
                seed_arr = np.array(seed, dtype=float)
                seed_arr = np.clip(seed_arr, bounds[:, 0], bounds[:, 1])
                particles[0] = self.round_params(seed_arr)
            except Exception:
                pass
        
        velocities = np.random.uniform(-1, 1, (n_particles, len(self.bounds)))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([self.evaluate_config(p) for p in particles])

        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        global_best_trades = self.meilleur_trades  # 🚀 Tracker les trades du meilleur global

        w = 0.7  # Inertie
        c1 = 1.4  # Coefficient cognitif
        c2 = 1.4  # Coefficient social

        with tqdm(total=max_iterations, desc="🐝 PSO", unit="iter") as pbar:
            for iteration in range(max_iterations):
                for i in range(n_particles):
                    # Mise à jour vitesse
                    r1, r2 = np.random.random(2)
                    velocities[i] = (w * velocities[i] +
                                   c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                   c2 * r2 * (global_best_position - particles[i]))

                    # Mise à jour position avec arrondi
                    particles[i] += velocities[i]
                    particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                    particles[i] = self.round_params(particles[i])  # 🔧 MODIFIÉ

                    # Évaluation
                    score = self.evaluate_config(particles[i])

                    # Mise à jour personnel
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i].copy()

                    # Mise à jour global
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()
                        global_best_trades = self.meilleur_trades  # 🚀 Capture les trades du nouveau meilleur

                pbar.set_postfix({'Meilleur': f"{global_best_score:.3f}", 'Trades': global_best_trades})
                pbar.update(1)

        return global_best_position, global_best_score

    def local_search_refinement(self, initial_params, max_iterations=30):
        """Recherche locale pour affiner une solution avec pas adaptatif"""
        print(f"🔍 Affinement local (précision={self.precision}, iter={max_iterations})")
        
        # 🔧 MODIFIÉ: Pas adaptatif selon la précision
        if self.precision == 1:
            step_size = 0.1
        elif self.precision == 2:
            step_size = 0.05
        else:
            step_size = 0.01
            
        current_params = self.round_params(initial_params)
        current_score = self.evaluate_config(current_params)

        bounds = self.bounds

        improved = True
        iteration = 0

        with tqdm(total=max_iterations, desc="🔍 Recherche locale", unit="iter") as pbar:
            while improved and iteration < max_iterations:
                improved = False
                for i in range(len(current_params)):
                    # Essayer +/- step_size
                    for delta in [-step_size, step_size]:
                        test_params = current_params.copy()
                        test_params[i] += delta

                        # Respecter les contraintes et arrondir
                        test_params[i] = round(np.clip(test_params[i], bounds[i][0], bounds[i][1]), self.precision)

                        test_score = self.evaluate_config(test_params)

                        if test_score > current_score:
                            current_score = test_score
                            current_params = test_params.copy()
                            improved = True

                iteration += 1
                pbar.set_postfix({'Score': f"{current_score:.3f}", 'Amélioré': improved})
                pbar.update(1)

        return current_params, current_score

def optimize_sector_coefficients_hybrid(
    sector_symbols, domain,
    period='1y', strategy='hybrid',
    montant=50, transaction_cost=1.0,
    initial_thresholds=(4.20, -0.5),
    budget_evaluations=1000,
    precision=2,  # 🔧 NOUVEAU: Paramètre de précision
    cap_range='Unknown',
    use_price_features=False,
    use_fundamentals_features=False,
    enable_temporal_validation=True  # 📅 NOUVEAU: Activer validation temporelle train/val/holdout
):
    """
    Optimisation hybride des coefficients sectoriels avec limitation des décimales
    
    Strategies disponibles:
    - 'genetic': Algorithmes génétiques
    - 'differential': Évolution différentielle  
    - 'pso': Particle Swarm Optimization
    - 'lhs': Latin Hypercube Sampling
    - 'cma': CMA-ES avec warm-start LHS
    - 'hybrid': Combine plusieurs méthodes (LHS + CMA-ES + autres)
    
    precision: Nombre de décimales pour les paramètres (1, 2, ou 3)
    cap_range: Segment de capitalisation associé au secteur
    """
    if not sector_symbols:
        print(f"🚫 Secteur {domain} vide, ignoré")
        return None, 0.0, 0.0, initial_thresholds, None

    # Téléchargement des données
    # 📅 Si validation temporelle active, charger 24 mois (18 train + 4 val + 2 holdout)
    if enable_temporal_validation:
        total_months_needed = TRAIN_MONTHS + VAL_MONTHS + HOLDOUT_MONTHS
        # Clamp à max 24mo (compatible yfinance)
        period_extended = f"{min(total_months_needed, 24)}mo"
        stock_data = download_stock_data(sector_symbols, period=period_extended)
    else:
        stock_data = download_stock_data(sector_symbols, period=period)
    
    if not stock_data:
        print(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
        return None, 0.0, 0.0, initial_thresholds, None

    for symbol, data in stock_data.items():
        print(f"📊 {symbol}: {len(data['Close'])} points de données")
    
    # 📅 SPLIT TEMPOREL si validation activée
    train_data, val_data, holdout_data, dates_info = None, None, None, None
    if enable_temporal_validation:
        train_data, val_data, holdout_data, dates_info = split_data_temporal(stock_data)
        print(f"\n📅 Split temporel:")
        print(f"   Train: {dates_info['train_start'].strftime('%Y-%m-%d')} → {dates_info['val_start'].strftime('%Y-%m-%d')} ({len(train_data)} symbols)")
        print(f"   Val:   {dates_info['val_start'].strftime('%Y-%m-%d')} → {dates_info['holdout_start'].strftime('%Y-%m-%d')} ({len(val_data)} symbols)")
        print(f"   Holdout: {dates_info['holdout_start'].strftime('%Y-%m-%d')} → {dates_info['latest_date'].strftime('%Y-%m-%d')} ({len(holdout_data)} symbols)")
        # Utiliser train_data pour l'optimisation
        optimization_data = train_data
    else:
        # Mode classique: utiliser toutes les données
        optimization_data = stock_data

    # 🚀 PRÉ-CALCUL DES FEATURES TA POUR TOUS LES SYMBOLS
    # Cela peuple TA_CACHE et DERIV_CACHE dans qsi.py, évitant les recalculs pendant l'optimisation
    precalculate_features(optimization_data, domain)

    # Récupération des meilleurs paramètres historiques
    db_path = 'signaux/optimization_hist.db'
    best_params_per_sector = extract_best_parameters(db_path)

    if domain in best_params_per_sector:
        csv_coeffs, csv_thresholds, csv_globals, csv_gain = best_params_per_sector[domain]
        # 🔧 Sécuriser en float (déjà des floats depuis SQLite, mais par sécurité)
        csv_coeffs = tuple(float(x) for x in csv_coeffs)
        csv_thresholds = tuple(float(x) for x in csv_thresholds)
        csv_globals = tuple(float(x) for x in csv_globals)
        print(f"📋 Paramètres historiques trouvés: coeffs={csv_coeffs}, seuils={csv_thresholds}, globaux={csv_globals}, gain={csv_gain:.2f}")
    else:
        csv_coeffs, csv_thresholds, csv_globals, csv_gain = None, initial_thresholds, (4.2, -0.5), -float('inf')

    hist_avg_gain = None  # 🔧 Pour mesurer l'amélioration vs l'historique
    hist_total_trades = None
    hist_success_rate = None

    # ♻️ Réévaluer les paramètres historiques sur les données ACTUELLES
    historical_candidate = None
    if csv_coeffs is not None and len(csv_coeffs) >= 8 and len(csv_thresholds) >= 8 and len(csv_globals) == 2:
        try:
            hist_coeffs = tuple(csv_coeffs[:8])
            hist_feature_thresholds = tuple(csv_thresholds[:8])
            hist_seuil_achat = float(csv_globals[0])
            hist_seuil_vente = float(csv_globals[1])
            
            # 🔧 CORRIGÉ: Récupérer les extras depuis BEST_PARAM_EXTRAS (global)
            hist_extra_params = None
            hist_fundamentals_extras = None
            
            from qsi import BEST_PARAM_EXTRAS

            # print(f"   🔍 DEBUG: Toutes les clés dans BEST_PARAM_EXTRAS: {list(BEST_PARAM_EXTRAS.keys())}")
            # print(f"   🔍 DEBUG: Je cherche domain='{domain}'")
            # print(f"   🔍 DEBUG: domain in dict? {domain in BEST_PARAM_EXTRAS}")
            if domain in BEST_PARAM_EXTRAS:
                print(f"      Clés disponibles: {list(BEST_PARAM_EXTRAS[domain].keys())}")
                extras_dict = BEST_PARAM_EXTRAS[domain]
                # Vérifier si des extras de price existent dans le dictionnaire
                if 'use_price_slope' in extras_dict:
                    hist_extra_params = {
                        'use_price_slope': int(extras_dict.get('use_price_slope', 0)),
                        'use_price_acc': int(extras_dict.get('use_price_acc', 0)),
                        'a_price_slope': float(extras_dict.get('a_price_slope', 0.0)),
                        'a_price_acc': float(extras_dict.get('a_price_acc', 0.0)),
                        'th_price_slope': float(extras_dict.get('th_price_slope', 0.0)),
                        'th_price_acc': float(extras_dict.get('th_price_acc', 0.0)),
                    }
                
                # Vérifier si des extras de fundamentals existent dans le dictionnaire
                if 'use_fundamentals' in extras_dict:
                    hist_fundamentals_extras = {
                        'use_fundamentals': int(extras_dict.get('use_fundamentals', 0)),
                        'a_rev_growth': float(extras_dict.get('a_rev_growth', 0.0)),
                        'a_eps_growth': float(extras_dict.get('a_eps_growth', 0.0)),
                        'a_roe': float(extras_dict.get('a_roe', 0.0)),
                        'a_fcf_yield': float(extras_dict.get('a_fcf_yield', 0.0)),
                        'a_de_ratio': float(extras_dict.get('a_de_ratio', 0.0)),
                        'th_rev_growth': float(extras_dict.get('th_rev_growth', 0.0)),
                        'th_eps_growth': float(extras_dict.get('th_eps_growth', 0.0)),
                        'th_roe': float(extras_dict.get('th_roe', 0.0)),
                        'th_fcf_yield': float(extras_dict.get('th_fcf_yield', 0.0)),
                        'th_de_ratio': float(extras_dict.get('th_de_ratio', 0.0)),
                    }

            total_gain = 0.0
            total_trades = 0
            total_success = 0
            for symbol, data in stock_data.items():
                # 🔧 Utiliser backtest_with_events si des extras historiques existent
                if hist_extra_params is not None or hist_fundamentals_extras is not None:
                    result, _ = backtest_signals_with_events(
                        data['Close'], data['Volume'], domain,
                        domain_coeffs={domain: hist_coeffs},
                        domain_thresholds={domain: hist_feature_thresholds},
                        seuil_achat=hist_seuil_achat, seuil_vente=hist_seuil_vente,
                        montant=montant, transaction_cost=transaction_cost,
                        extra_params=hist_extra_params,
                        fundamentals_extras=hist_fundamentals_extras,
                        symbol_name=symbol
                    )
                else:
                    result = backtest_signals(
                        data['Close'], data['Volume'], domain,
                        domain_coeffs={domain: hist_coeffs},
                        domain_thresholds={domain: hist_feature_thresholds},
                        seuil_achat=hist_seuil_achat, seuil_vente=hist_seuil_vente,
                        montant=montant, transaction_cost=transaction_cost
                    )
                total_gain += result.get('gain_total', 0)
                total_trades += result.get('trades', 0)
                total_success += result.get('gagnants', 0)

            hist_avg_gain = total_gain / len(stock_data) if stock_data else 0.0
            hist_success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
            hist_total_trades = total_trades

            # Préparer un candidat pour la comparaison finale (18 paramètres de base)
            hist_params = list(hist_coeffs + hist_feature_thresholds + (hist_seuil_achat, hist_seuil_vente))
            
            # 🔧 Étendre les paramètres historiques si des features sont activées
            if use_price_features:
                # Ajouter 6 paramètres par défaut pour price features (désactivés par défaut)
                hist_params += [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]  # use_slope=0, use_acc=0, weights=0.5, thresholds=0
            
            if use_fundamentals_features:
                # Ajouter 10 paramètres par défaut pour fundamentals (désactivés par défaut)
                hist_params += [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
            
            historical_candidate = ('Historical (re-eval)', tuple(hist_params), hist_avg_gain)

            print(f"♻️ Paramètres historiques réévalués sur données actuelles: gain_moy={hist_avg_gain:.2f}, trades={total_trades}, success={hist_success_rate:.2f}%")
            if use_price_features or use_fundamentals_features:
                print(f"   ℹ️  Paramètres étendus de 18 → {len(hist_params)} (features désactivées par défaut)")
        except Exception as e:
            print(f"⚠️ Réévaluation des paramètres historiques impossible: {e}")

    # 🔧 MODIFIÉ: Initialisation de l'optimiseur avec précision et parallélisation
    # Utiliser n_jobs=2 ou plus pour activer la parallélisation par symboles
    # (n_jobs=1 reste séquentiel par défaut)
    # 🚀 Optimisation auto: ajuste n_jobs au nombre de stocks disponibles (max 4)
    n_jobs_param = min(4, max(1, len(optimization_data)))
    optimizer = HybridOptimizer(optimization_data, domain, montant, transaction_cost, precision, use_price_features, use_fundamentals_features, n_jobs=n_jobs_param)

    # 🔧 NOUVEAU: Ajuster le budget en fonction de la précision
    # Avec une plus grande précision, l'espace de recherche AUGMENTE
    # Plus de décimales = plus de points possibles à explorer
    # Donc on doit AUGMENTER le budget proportionnellement
    # Facteur: 1.0 pour précision=1, 1.5 pour précision=2, 2.0 pour précision=3
    precision_factor = 1.0 + (precision - 1) * 1.5
    adjusted_budget = int(budget_evaluations * precision_factor)
    
    # Stratégies d'optimisation
    results = []
    # Ajouter le candidat "historical" réévalué si disponible
    seed_vector = None
    if historical_candidate:
        results.append(historical_candidate)
        try:
            seed_vector = np.array(historical_candidate[1], dtype=float)
        except Exception:
            seed_vector = None

    # 🔍 Étape rapide: affinement local autour de la graine historique
    # ⚠️ SKIP si le score historique est déjà très bon (>100) car local search
    # n'améliorera probablement pas + coûte 2-3 minutes pour rien
    historical_score = historical_candidate[2] if historical_candidate else 0
    if seed_vector is not None and historical_score < 100.0:
        try:
            print("🔍 Quick local refinement around historical seed...")
            refined_params, refined_score = optimizer.local_search_refinement(seed_vector, max_iterations=5)
            # ⚠️ N'ajouter la version affinée que si elle AMÉLIORE le score historique
            if refined_score > historical_score:
                results.append(('Local Refinement (seed)', refined_params, refined_score))
                seed_vector = refined_params.copy()
            else:
                print(f"   ℹ️ Local refinement didn't improve (historical: {historical_score:.2f}, refined: {refined_score:.2f}), skipping")
                # Garder le seed historique original comme graine
                seed_vector = np.array(historical_candidate[1], dtype=float)
        except Exception as e:
            print(f"⚠️ Local refinement failed: {e}")
    elif seed_vector is not None:
        print(f"   ℹ️ Skipping local refinement (historical score {historical_score:.2f} already excellent)")
    print(f"🚀 Optimisation hybride pour {domain} avec stratégie '{strategy}' (précision: {precision} décimales)")
    print(f"📈 Budget d'évaluations initial: {budget_evaluations}")
    print(f"🔧 Budget ajusté selon la précision: {adjusted_budget} (facteur: {precision_factor:.2f}x)")

    if strategy == 'genetic': # if strategy == 'genetic' or strategy == 'hybrid':
        # Algorithmes génétiques - 🔧 Augmenter max pop_size selon budget
        pop_size = min(100, adjusted_budget // 20)  # Max 100 au lieu de 50
        generations = min(50, adjusted_budget // pop_size)  # Generations aussi augmentées
        params_ga, score_ga = optimizer.genetic_algorithm(pop_size, generations, seed=seed_vector)
        results.append(('Genetic Algorithm', params_ga, score_ga))

    if strategy == 'hybrid' or strategy == 'differential':
        # Évolution différentielle - 🔧 Augmenter max pop_size selon budget
        pop_size = min(60, adjusted_budget // 25)  # Max 60 au lieu de 45
        max_iter = min(100, adjusted_budget // pop_size)
        params_de, score_de = optimizer.differential_evolution_opt(pop_size, max_iter, seed=seed_vector)
        results.append(('Differential Evolution', params_de, score_de))

    if strategy == 'hybrid' or strategy == 'pso':
        # PSO - 🔧 Augmenter max particules selon budget
        n_particles = min(50, adjusted_budget // 30)  # Max 50 au lieu de 30
        max_iter = min(100, adjusted_budget // n_particles)
        params_pso, score_pso = optimizer.particle_swarm_optimization(n_particles, max_iter, seed=seed_vector)
        results.append(('PSO', params_pso, score_pso))

    # LHS toujours utilisé comme warm-start pour CMA ou en direct
    if strategy in ('hybrid', 'lhs', 'cma'):
        lhs_budget = int(max(200, min(3000, (3/2)*adjusted_budget)))
        params_lhs, score_lhs = optimizer.latin_hypercube_sampling(lhs_budget, seed=seed_vector)
        results.append(('Latin Hypercube', params_lhs, score_lhs))

    if strategy in ('hybrid', 'cma'):
        lhs_samples = int(max(500, min(3000, adjusted_budget)))
        top_k = 8
        max_gen = 25
        pop_size = None  # laisse CMA choisir
        params_cma, score_cma = optimizer.cma_es_optimization(lhs_samples=lhs_samples, top_k=top_k, max_generations=max_gen, pop_size=pop_size)
        results.append(('CMA-ES', params_cma, score_cma))

    # Sélection du meilleur résultat
    best_method, best_params, best_score = max(results, key=lambda x: x[2])
    print(f"🏆 Meilleure méthode: {best_method} avec score {best_score:.4f}")

    # Afficher aussi le meilleur candidat non-historique pour transparence
    non_hist_candidates = [r for r in results if r[0] != 'Historical (re-eval)']
    if non_hist_candidates:
        nh_method, nh_params, nh_score = max(non_hist_candidates, key=lambda x: x[2])
        nh_coeffs = tuple(float(x) for x in nh_params[:8])
        nh_th = tuple(float(nh_params[i]) for i in range(8, 16))
        nh_buy = float(nh_params[16])
        nh_sell = float(nh_params[17])
        print(f"🔎 Meilleur candidat Optimiseur: {nh_method} score={nh_score:.4f}")
        print(f"   coeffs={nh_coeffs}")
        print(f"   seuils: features={nh_th}, achat={nh_buy:.2f}, vente={nh_sell:.2f}")

    # Affinement local du meilleur résultat
    if strategy == 'hybrid':
        print(f"🔧 Affinement local du meilleur résultat...")
        refined_params, refined_score = optimizer.local_search_refinement(best_params)
        if refined_score > best_score:
            best_params = refined_params
            best_score = refined_score
            print(f"✨ Affinement réussi: nouveau score {best_score:.4f}")

    # 🔧 MODIFIÉ: Extraction des paramètres finaux avec conversion Python natif
    # V2.0: Extraire 8 coefficients + 8 seuils individuels + 2 seuils globaux
    best_coeffs = tuple(float(x) for x in best_params[:8])  # 8 coefficients
    best_feature_thresholds = tuple(float(best_params[i]) for i in range(8, 16))  # 8 seuils features
    best_seuil_achat = float(best_params[16])  # Seuil global achat
    best_seuil_vente = float(best_params[17])  # Seuil global vente

    # 🔧 DEBUG: Taille du vecteur de params
    print(f"   🔍 Taille du meilleur vecteur de params: {len(best_params)} (expected: {len(optimizer.bounds)})")

    # Extra params (price features) if present in vector
    extra_params = None
    if use_price_features and len(best_params) >= 24:
        use_ps = int(round(np.clip(best_params[18], 0.0, 1.0)))
        use_pa = int(round(np.clip(best_params[19], 0.0, 1.0)))
        a_ps = float(np.clip(round(best_params[20], precision), -1.0, 3.0))
        a_pa = float(np.clip(round(best_params[21], precision), -1.0, 3.0))
        th_ps = float(np.clip(round(best_params[22], precision), -0.15, 0.15))
        th_pa = float(np.clip(round(best_params[23], precision), -0.15, 0.15))
        extra_params = {
            'use_price_slope': use_ps,
            'use_price_acc': use_pa,
            'a_price_slope': a_ps,
            'a_price_acc': a_pa,
            'th_price_slope': th_ps,
            'th_price_acc': th_pa,
        }
    
    # Fundamentals extras if present in vector
    fundamentals_extras = None
    fundamentals_index_offset = 24 if use_price_features else 18
    if use_fundamentals_features and len(best_params) >= (fundamentals_index_offset + 10):
        use_fund = int(round(np.clip(best_params[fundamentals_index_offset], 0.0, 1.0)))
        a_rev = float(np.clip(round(best_params[fundamentals_index_offset + 1], precision), 0.0, 3.0))
        a_eps = float(np.clip(round(best_params[fundamentals_index_offset + 2], precision), 0.0, 3.0))
        a_roe = float(np.clip(round(best_params[fundamentals_index_offset + 3], precision), 0.0, 3.0))
        a_fcf = float(np.clip(round(best_params[fundamentals_index_offset + 4], precision), 0.0, 3.0))
        a_de = float(np.clip(round(best_params[fundamentals_index_offset + 5], precision), 0.0, 3.0))
        th_rev = float(np.clip(round(best_params[fundamentals_index_offset + 6], precision), -30.0, 30.0))
        th_eps = float(np.clip(round(best_params[fundamentals_index_offset + 7], precision), -30.0, 30.0))
        th_roe = float(np.clip(round(best_params[fundamentals_index_offset + 8], precision), -30.0, 30.0))
        th_fcf = float(np.clip(round(best_params[fundamentals_index_offset + 9], precision), -10.0, 10.0))
        fundamentals_extras = {
            'use_fundamentals': use_fund,
            'a_rev_growth': a_rev,
            'a_eps_growth': a_eps,
            'a_roe': a_roe,
            'a_fcf_yield': a_fcf,
            'a_de_ratio': a_de,
            'th_rev_growth': th_rev,
            'th_eps_growth': th_eps,
            'th_roe': th_roe,
            'th_fcf_yield': th_fcf,
            'th_de_ratio': 0.0,  # Reserved
        }

    # Calcul des statistiques finales
    total_success = 0
    total_trades = 0
    debug_trades_per_symbol = {}
    
    # 🔧 DEBUG: Afficher la configuration du recalcul
    print(f"   🔍 Recalcul avec: use_price_features={use_price_features}, use_fundamentals_features={use_fundamentals_features}")
    print(f"      extra_params={extra_params}")
    print(f"      fundamentals_extras={fundamentals_extras}")
    print(f"      Condition 1 (price): {use_price_features} and {extra_params is not None} = {use_price_features and extra_params is not None}")
    print(f"      Condition 2 (fund): {use_fundamentals_features} and {fundamentals_extras is not None} = {use_fundamentals_features and fundamentals_extras is not None}")
    
    for symbol, data in stock_data.items():
        if (use_price_features and extra_params is not None) or (use_fundamentals_features and fundamentals_extras is not None):
            # Utiliser backtest_with_events pour les features étendues
            result, _ = backtest_signals_with_events(
                data['Close'], data['Volume'], domain,
                domain_coeffs={domain: best_coeffs},
                domain_thresholds={domain: best_feature_thresholds},
                seuil_achat=best_seuil_achat, seuil_vente=best_seuil_vente,
                montant=montant, transaction_cost=transaction_cost,
                extra_params=extra_params,
                fundamentals_extras=fundamentals_extras,
                symbol_name=symbol
            )
        else:
            # Backtest classique sans extras
            result = backtest_signals(
                data['Close'], data['Volume'], domain,
                domain_coeffs={domain: best_coeffs},
                domain_thresholds={domain: best_feature_thresholds},
                seuil_achat=best_seuil_achat, seuil_vente=best_seuil_vente,
                montant=montant, transaction_cost=transaction_cost
            )
        
        symbol_trades = result.get('trades', 0)
        debug_trades_per_symbol[symbol] = symbol_trades
        total_success += result.get('gagnants', 0)
        total_trades += symbol_trades

    success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

    # � DEBUG: Afficher les trades par symbole si le résultat semble incohérent
    if total_trades == 0 and len(stock_data) > 0:
        print(f"   ⚠️  DEBUG: Aucun trade malgré {len(stock_data)} symboles")
        print(f"      Trades par symbole: {debug_trades_per_symbol}")

    # �📊 Rapport synthétique secteur
    if hist_avg_gain is not None:
        delta = best_score - hist_avg_gain
        delta_pct = (delta / abs(hist_avg_gain) * 100) if hist_avg_gain != 0 else None
        success_old_str = f"{hist_success_rate:.1f}%" if hist_success_rate is not None else "-"
        trades_old_str = hist_total_trades if hist_total_trades is not None else "-"
        delta_pct_str = f", {delta_pct:+.1f}%" if delta_pct is not None else ""
        print(f"✅ {domain}: gain {best_score:.2f} vs {hist_avg_gain:.2f} ({delta:+.2f}{delta_pct_str}); trades {total_trades} vs {trades_old_str}; success {success_rate:.1f}% vs {success_old_str} ; méthode {best_method}")
    else:
        print(f"✅ {domain}: gain {best_score:.2f}; trades {total_trades}; success {success_rate:.1f}% ; méthode {best_method}")

    print(f"   coeffs: {best_coeffs}")
    print(f"   seuils: features={best_feature_thresholds}, achat={best_seuil_achat:.2f}, vente={best_seuil_vente:.2f}")
    
    # 🔧 Afficher les extras si présents
    if extra_params:
        print(f"   📊 Price features: use_slope={extra_params['use_price_slope']}, use_acc={extra_params['use_price_acc']}")
        print(f"      Poids: slope={extra_params['a_price_slope']:.1f}, acc={extra_params['a_price_acc']:.1f}")
        print(f"      Seuils: slope={extra_params['th_price_slope']:.3f}, acc={extra_params['th_price_acc']:.3f}")
    
    if fundamentals_extras:
        print(f"   📊 Fundamentals: use={fundamentals_extras['use_fundamentals']}")
        print(f"      Poids: rev={fundamentals_extras['a_rev_growth']:.1f}, eps={fundamentals_extras['a_eps_growth']:.1f}, roe={fundamentals_extras['a_roe']:.1f}, fcf={fundamentals_extras['a_fcf_yield']:.1f}, de={fundamentals_extras['a_de_ratio']:.1f}")
        print(f"      Seuils: rev={fundamentals_extras['th_rev_growth']:.1f}%, eps={fundamentals_extras['th_eps_growth']:.1f}%, roe={fundamentals_extras['th_roe']:.1f}%, fcf={fundamentals_extras['th_fcf_yield']:.1f}%")

    # Sauvegarde des résultats - agrégé en un tuple unique
    all_thresholds = best_feature_thresholds + (best_seuil_achat, best_seuil_vente)
    
    # 🔧 Sauvegarder si le nouveau score surpasse le score historique RÉÉVALUÉ sur données actuelles
    # Comparaison avec hist_avg_gain (réévalué), pas avec le gain de la base de données
    save_epsilon = 0.01
    should_save = (hist_avg_gain is None) or (best_score > hist_avg_gain + save_epsilon)
    
    if should_save:
        save_optimization_results(domain, best_coeffs, best_score, success_rate, total_trades, all_thresholds, cap_range, extra_params=extra_params, fundamentals_extras=fundamentals_extras)
        hist_str = f"{hist_avg_gain:.2f}" if hist_avg_gain is not None else "N/A"
        print(f"💾 Sauvegarde: nouveau score {best_score:.2f} > historique réévalué {hist_str}")
    else:
        print(f"ℹ️ Pas de sauvegarde: nouveau {best_score:.2f} ≤ historique réévalué {hist_avg_gain:.2f} (epsilon={save_epsilon})")

    summary = {
        'sector': domain,
        'cap_range': cap_range,
        'gain_new': best_score,
        'gain_old': hist_avg_gain,
        'trades_new': total_trades,
        'trades_old': hist_total_trades,
        'success_new': success_rate,
        'success_old': hist_success_rate
    }

    return best_coeffs, best_score, success_rate, all_thresholds, summary

def save_optimization_results(domain, coeffs, gain_total, success_rate, total_trades, thresholds, cap_range=None, extra_params=None, fundamentals_extras=None):
    """
    Sauvegarde les résultats d'optimisation dans la base de données SQLite avec le format attendu:
    Timestamp, Sector, Gain_moy, Success_Rate, Trades, Seuil_Achat, Seuil_Vente, a1-a8, th1-th8
    
    Args:
        domain: Secteur ou clé composite secteur_capRange
        coeffs: 8 coefficients (a1-a8)
        gain_total: Gain moyen
        success_rate: Taux de réussite
        total_trades: Nombre total de trades
        thresholds: 8 feature thresholds + 2 global thresholds (buy, sell)
        cap_range: Segment de capitalisation (Small, Mid, Large, Mega, Unknown)
    """
    from datetime import datetime
    import sqlite3

    db_path = 'signaux/optimization_hist.db'

    def _ensure_opt_runs_schema(conn):
        try:
            cur = conn.cursor()
            # Check existing columns
            cur.execute("PRAGMA table_info(optimization_runs)")
            cols = {row[1] for row in cur.fetchall()}

            # Ensure market_cap_range exists (newer schema)
            if 'market_cap_range' not in cols:
                cur.execute("ALTER TABLE optimization_runs ADD COLUMN market_cap_range TEXT")

            # Prepare new columns for price derivatives params (future use)
            new_cols = [
                ('a9', 'REAL'), ('a10', 'REAL'),
                ('th9', 'REAL'), ('th10', 'REAL'),
                ('use_price_slope', 'INTEGER DEFAULT 0'),
                ('use_price_acc', 'INTEGER DEFAULT 0'),
                # Fundamentals columns
                ('a11', 'REAL'), ('a12', 'REAL'), ('a13', 'REAL'), ('a14', 'REAL'), ('a15', 'REAL'),
                ('th11', 'REAL'), ('th12', 'REAL'), ('th13', 'REAL'), ('th14', 'REAL'), ('th15', 'REAL'),
                ('use_fundamentals', 'INTEGER DEFAULT 0')
            ]
            for name, decl in new_cols:
                if name not in cols:
                    try:
                        cur.execute(f"ALTER TABLE optimization_runs ADD COLUMN {name} {decl}")
                    except Exception:
                        pass
            conn.commit()
        except Exception:
            # Non-fatal: keep retrocompat even if migration fails
            pass
    
    try:
        # La décision de sauvegarder est déjà prise par l'appelant
        # Pas de double vérification ici
        
        # Préparer le timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Inférer secteur / cap_range si non fourni
        normalized_cap = cap_range or 'Unknown'
        normalized_sector = domain

        allowed_caps = {'Small', 'Mid', 'Large', 'Mega', 'Unknown'}
        if '_' in domain:
            maybe_sector, maybe_cap = domain.rsplit('_', 1)
            if maybe_cap in allowed_caps:
                normalized_sector = maybe_sector
                if normalized_cap == 'Unknown':
                    normalized_cap = maybe_cap

        # Connexion à SQLite et insertion
        conn = sqlite3.connect(db_path)
        # Ensure schema is up-to-date (idempotent)
        _ensure_opt_runs_schema(conn)
        cursor = conn.cursor()
        
        # Construire l'INSERT OR REPLACE avec tous les paramètres
        # Prepare extras with defaults
        ex = extra_params or {}
        use_ps = int(ex.get('use_price_slope', 0) or 0)
        use_pa = int(ex.get('use_price_acc', 0) or 0)
        a9 = float(ex.get('a_price_slope', 0.0) or 0.0)
        a10 = float(ex.get('a_price_acc', 0.0) or 0.0)
        th9 = float(ex.get('th_price_slope', 0.0) or 0.0)
        th10 = float(ex.get('th_price_acc', 0.0) or 0.0)
        
        # Prepare fundamentals extras with defaults
        fx = fundamentals_extras or {}
        use_fund = int(fx.get('use_fundamentals', 0) or 0)
        a11 = float(fx.get('a_rev_growth', 0.0) or 0.0)
        a12 = float(fx.get('a_eps_growth', 0.0) or 0.0)
        a13 = float(fx.get('a_roe', 0.0) or 0.0)
        a14 = float(fx.get('a_fcf_yield', 0.0) or 0.0)
        a15 = float(fx.get('a_de_ratio', 0.0) or 0.0)
        th11 = float(fx.get('th_rev_growth', 0.0) or 0.0)
        th12 = float(fx.get('th_eps_growth', 0.0) or 0.0)
        th13 = float(fx.get('th_roe', 0.0) or 0.0)
        th14 = float(fx.get('th_fcf_yield', 0.0) or 0.0)
        th15 = float(fx.get('th_de_ratio', 0.0) or 0.0)

        cursor.execute('''
            INSERT OR REPLACE INTO optimization_runs
            (timestamp, sector, market_cap_range, gain_moy, success_rate, trades,
             a1, a2, a3, a4, a5, a6, a7, a8,
             th1, th2, th3, th4, th5, th6, th7, th8,
             seuil_achat, seuil_vente,
             a9, a10, th9, th10, use_price_slope, use_price_acc,
             a11, a12, a13, a14, a15, th11, th12, th13, th14, th15, use_fundamentals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, normalized_sector, normalized_cap, gain_total, success_rate, total_trades,
            coeffs[0], coeffs[1], coeffs[2], coeffs[3],
            coeffs[4], coeffs[5], coeffs[6], coeffs[7],
            thresholds[0] if len(thresholds) > 0 else 50.0,
            thresholds[1] if len(thresholds) > 1 else 0.0,
            thresholds[2] if len(thresholds) > 2 else 0.0,
            thresholds[3] if len(thresholds) > 3 else 1.2,
            thresholds[4] if len(thresholds) > 4 else 25.0,
            thresholds[5] if len(thresholds) > 5 else 0.0,
            thresholds[6] if len(thresholds) > 6 else 0.5,
            thresholds[7] if len(thresholds) > 7 else 4.20,
            thresholds[8] if len(thresholds) > 8 else 4.20,
            thresholds[9] if len(thresholds) > 9 else -0.5,
            a9, a10, th9, th10, use_ps, use_pa,
            a11, a12, a13, a14, a15, th11, th12, th13, th14, th15, use_fund
        ))
        
        conn.commit()
        conn.close()
        
        print(f"📝 Résultats sauvegardés dans SQLite pour {normalized_sector} ({normalized_cap})")

    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 OPTIMISATEUR HYBRIDE - Génération de coefficients par secteur × cap_range")
    print("="*80)
    
    # Chargement des symboles - priorité à SQLite
    list_type = "optimization"
    if SYMBOL_MANAGER_AVAILABLE:
        print("\n1️⃣  Chargement des symboles depuis SQLite...")
        init_symbols_table()
        try:
            count = sync_txt_to_sqlite("optimisation_symbols.txt", list_type)
            # Les messages sont maintenant affichés par sync_txt_to_sqlite
        except Exception as e:
            print(f"   ⚠️ Impossible de synchroniser optimisation_symbols.txt: {e}")

        symbols = get_symbols_by_list_type(list_type, active_only=True)
        print(f"   ✅ {len(symbols)} symboles actifs chargés ({list_type})")
        
        # Obtenir tous les secteurs et cap_ranges disponibles
        sectors_available = get_all_sectors(list_type=list_type)
        cap_ranges_available = get_all_cap_ranges(list_type=list_type)
        
        print(f"\n2️⃣  Organisation des symboles:")
        print(f"   - Secteurs: {len(sectors_available)}")
        print(f"   - Gammes de cap: {len(cap_ranges_available)}")
        
        # Construction de sector_cap_ranges depuis SQLite
        sector_cap_ranges = {}
        total_combos = 0
        for sector in sectors_available:
            sector_cap_ranges[sector] = {}
            for cap_range in cap_ranges_available:
                syms = get_symbols_by_sector_and_cap(sector, cap_range, list_type, active_only=True)
                if syms:
                    sector_cap_ranges[sector][cap_range] = syms
                    total_combos += 1
                    print(f"   ✅ {sector} × {cap_range}: {len(syms)} symboles")
        
        print(f"\n   📊 Total: {total_combos} combinaisons secteur×cap_range avec symboles")
        
        # 🧼 NETTOYAGE DES GROUPES: filtrer <5 symboles
        print(f"\n🧼 Nettoyage des groupes (minimum {MIN_SYMBOLS_PER_GROUP} symboles/groupe)...")
        sector_cap_ranges, ignored_log = clean_sector_cap_groups(sector_cap_ranges, MIN_SYMBOLS_PER_GROUP)
        
        if ignored_log:
            log_ignored_groups(ignored_log)
            print(f"   ⚠️ {len(ignored_log)} groupes ignorés (trop peu de symboles)")
        
        # Recompter les combinaisons après nettoyage
        total_combos_cleaned = sum(1 for s in sector_cap_ranges.values() for cap, syms in s.items() if syms)
        print(f"   ✅ {total_combos_cleaned} combinaisons valides après nettoyage")
    
    else:
        print("\n⚠️ SQLite non disponible, utilisation de la méthode classique...")
        # Fallback: méthode originale
        symbols = list(dict.fromkeys(load_symbols_from_txt("optimisation_symbols.txt")))
        
        sectors = {
            "Technology": [],
            "Healthcare": [],
            "Financial Services": [],
            "Consumer Cyclical": [],
            "Industrials": [],
            "Energy": [],
            "Basic Materials": [],
            "Communication Services": [],
            "Consumer Defensive": [],
            "Utilities": [],
            "Real Estate": [],
            "ℹ️Inconnu!!": []
        }
        
        cap_buckets = ["Small", "Mid", "Large", "Mega", "Unknown"]
        sector_cap_ranges = {sec: {cap: [] for cap in cap_buckets} for sec in sectors.keys()}
        
        print(f"📋 Assignation des secteurs (cache yfinance utilisé)...")
        for symbol in symbols:
            sector = get_sector(symbol, use_cache=True)
            if sector not in sectors:
                sector = "ℹ️Inconnu!!"
            cap_range = classify_cap_range(symbol)
            
            sectors[sector].append(symbol)
            sector_cap_ranges.setdefault(sector, {cap: [] for cap in cap_buckets})
            sector_cap_ranges[sector].setdefault(cap_range, []).append(symbol)
        
        print("\n📋 Assignation secteur × cap range:")
        for sector, buckets in sector_cap_ranges.items():
            for cap_range, syms in buckets.items():
                if syms:
                    print(f"{sector} [{cap_range}]: {len(syms)} symboles")

    # Paramètres d'optimisation
    print("\n3️⃣  Configuration de l'optimisation:")
    search_strategies = ['hybrid', 'differential', 'genetic', 'pso', 'lhs', 'cma']
    
    strategy = input("   Stratégie ('hybrid', 'differential', 'genetic', 'pso', 'lhs', 'cma') : ").strip().lower()
    i=0
    while (strategy not in search_strategies) and i<3:
        strategy = input("   Stratégie invalide. Choisir parmi ('hybrid', 'differential', 'genetic', 'pso', 'lhs') : ").strip().lower()
        i+=1
    if strategy not in search_strategies:
        strategy = random.choice(search_strategies)
        print(f"   Stratégie inconnue, utilisation aléatoire: {strategy}")

    # Choix de la précision
    try:
        precision = int(input("   Précision (décimales: 1, 2, ou 3) [défaut: 2] : ").strip() or "2")
        if precision not in [1, 2, 3]:
            precision = 2
    except ValueError:
        precision = 2

    # 🎯 ACTIVATION DES FEATURES ÉTENDUES
    print("\n   📊 Features d'optimisation:")
    use_price_features_input = input("   Activer price features (slope, acceleration) ? (o/N) : ").strip().lower()
    use_price_features = (use_price_features_input == 'o')
    
    use_fundamentals_features_input = input("   Activer fundamentals features (rev_growth, eps, roe, etc.) ? (o/N) : ").strip().lower()
    use_fundamentals_features = (use_fundamentals_features_input == 'o')
    
    # Calcul du nombre de paramètres
    param_count = 18  # Base
    if use_price_features:
        param_count += 6
    if use_fundamentals_features:
        param_count += 10
    
    print(f"\n   🔧 Configuration:")
    print(f"      - Stratégie: {strategy}")
    print(f"      - Précision: {precision} décimales")
    print(f"      - Price features: {'✅ Activé' if use_price_features else '❌ Désactivé'}")
    print(f"      - Fundamentals features: {'✅ Activé' if use_fundamentals_features else '❌ Désactivé'}")
    print(f"      - Nombre de paramètres: {param_count}")

    # Adapter le budget selon la précision
    budget_base = 700#1000
    if precision == 1:
        budget_evaluations = int(budget_base * 0.5)
    elif precision == 2:
        budget_evaluations = budget_base
    else:
        budget_evaluations = int(budget_base * 2)
    
    print(f"   💡 Budget d'évaluations: {budget_evaluations} éval/segment secteur×cap")
    
    # Confirmation avant lancement
    total_to_optimize = sum(1 for s in sector_cap_ranges.values() for cap, syms in s.items() if syms)
    print(f"\n4️⃣  Prêt à optimiser {total_to_optimize} combinaisons secteur×cap_range")
    confirm = input("   Lancer l'optimisation complète ? (o/N) : ").strip().lower()
    
    if confirm != 'o':
        print("\n❌ Optimisation annulée")
        sys.exit(0)

    optimized_coeffs = {}
    sector_summaries = []

    for sector, buckets in sector_cap_ranges.items():
        for cap_range, sector_symbols in buckets.items():
            if not sector_symbols:
                continue

            combo_key = f"{sector}_{cap_range}"
            print(f"\n" + "="*160)
            print(f"🎯 OPTIMISATION {strategy.upper()} - {sector} / {cap_range}")
            print(f"="*160)

            coeffs, gain_total, success_rate, thresholds, summary = optimize_sector_coefficients_hybrid(
                sector_symbols, combo_key,
                period='1y',
                strategy=strategy,
                montant=50,
                transaction_cost=0.02,
                budget_evaluations=budget_evaluations,
                precision=precision,  # 🔧 NOUVEAU: Paramètre de précision
                cap_range=cap_range,
                use_price_features=use_price_features,  # 🎯 Features étendues
                use_fundamentals_features=use_fundamentals_features  # 🎯 Features étendues
            )

            if coeffs:
                optimized_coeffs[combo_key] = coeffs
            if summary:
                sector_summaries.append(summary)

    print("\n" + "="*80)
    print("🏆 DICTIONNAIRE FINAL OPTIMISÉ")
    print("="*80)
    print("domain_coeffs = {")
    for sector, coeffs in optimized_coeffs.items():
        print(f"    '{sector}': {coeffs},")
    print("}")
    print("="*80)

    # 📊 Comparaison globale (sectors avec historique disponible)
    comparables = [s for s in sector_summaries if s.get('gain_old') is not None]
    if comparables:
        print("\n📊 Bilan global vs historique (réévalué aujourd'hui):")
        total_old_gain = 0.0
        total_new_gain = 0.0
        total_old_trades = 0
        total_new_trades = 0
        for s in comparables:
            delta = s['gain_new'] - s['gain_old']
            delta_pct = (delta / abs(s['gain_old']) * 100) if s['gain_old'] != 0 else None
            delta_pct_str = f", {delta_pct:+.1f}%" if delta_pct is not None else ""
            trades_old_str = s['trades_old'] if s['trades_old'] is not None else "-"
            success_old_str = f"{s['success_old']:.1f}%" if s['success_old'] is not None else "-"
            label = f"{s['sector']} ({s.get('cap_range', 'Unknown')})"
            print(f" - {label}: gain {s['gain_new']:.2f} vs {s['gain_old']:.2f} ({delta:+.2f}{delta_pct_str}); trades {s['trades_new']} vs {trades_old_str}; success {s['success_new']:.1f}% vs {success_old_str}")
            total_old_gain += s['gain_old']
            total_new_gain += s['gain_new']
            total_old_trades += s['trades_old'] or 0
            total_new_trades += s['trades_new']

        n = len(comparables)
        avg_old = total_old_gain / n if n else 0
        avg_new = total_new_gain / n if n else 0
        delta_tot = avg_new - avg_old
        delta_tot_pct = (delta_tot / abs(avg_old) * 100) if avg_old != 0 else None
        print(f"\nRésumé moyen (sur {n} secteurs): gain {avg_new:.2f} vs {avg_old:.2f} ({delta_tot:+.2f}{'' if delta_tot_pct is None else f', {delta_tot_pct:+.1f}%'}); trades totaux {total_new_trades} vs {total_old_trades}")