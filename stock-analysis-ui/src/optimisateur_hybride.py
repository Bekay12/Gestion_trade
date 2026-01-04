# optimisateur_hybride_fixed.py
# Version optimisée avec limitation des décimales pour réduire l'espace de recherche

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path
sys.path.append("C:\\Users\\berti\\Desktop\\Mes documents\\Gestion_trade\\stock-analysis-ui\\src\\trading_c_acceleration")
from qsi import download_stock_data, load_symbols_from_txt, extract_best_parameters
from qsi_optimized import backtest_signals, backtest_signals_with_events, backtest_signals_c_extended
from pathlib import Path
from tqdm import tqdm
# import yfinance as yf  # Import paresseux - chargé seulement si nécessaire
from collections import deque
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 🚀 PERFORMANCE: Détection automatique du nombre de workers
MAX_WORKERS = min(os.cpu_count() or 4, 12)  # Max 12 pour éviter surcharge

# Import du gestionnaire de symboles SQLite
try:
    from symbol_manager import (
        init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type, get_all_sectors,
        get_all_cap_ranges, get_symbols_by_sector_and_cap, get_symbol_count,
        get_cleaned_group_cache, save_cleaned_group_cache
    )
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ symbol_manager non disponible, utilisation de la méthode classique")
    SYMBOL_MANAGER_AVAILABLE = False

# 🔧 OPTIMISATION: Caching des secteurs (mémoire + disque)
SECTOR_CACHE_FILE = Path("cache_data/sector_cache.json")
SECTOR_CACHE_FILE.parent.mkdir(exist_ok=True)
SECTOR_TTL_DAYS = 30
SECTOR_TTL_UNKNOWN_DAYS = 7

def _load_sector_cache():
    try:
        if SECTOR_CACHE_FILE.exists():
            with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de charger le cache secteurs: {e}")
    return {}

def _save_sector_cache(cache):
    try:
        with open(SECTOR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ Impossible d'écrire le cache secteurs: {e}")

def _is_sector_expired(entry):
    try:
        ts = entry.get("ts")
        if not ts:
            return True
        dt = datetime.fromisoformat(ts)
        ttl_days = SECTOR_TTL_UNKNOWN_DAYS if entry.get("sector") == "ℹ️Inconnu!!" else SECTOR_TTL_DAYS
        return (datetime.utcnow() - dt).days >= ttl_days
    except Exception:
        return True

_sector_cache = _load_sector_cache()

def get_sector(symbol, use_cache=True):
    """Récupère le secteur d'une action avec cache mémoire + disque."""
    if use_cache:
        entry = _sector_cache.get(symbol)
        if entry and not _is_sector_expired(entry):
            return entry.get("sector", "ℹ️Inconnu!!")

    try:
        import yfinance as yf  # Import paresseux - chargé seulement si nécessaire
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'ℹ️Inconnu!!')
        print(f"📋 {symbol}: Secteur = {sector}")
        _sector_cache[symbol] = {"sector": sector, "ts": datetime.utcnow().isoformat()}
        _save_sector_cache(_sector_cache)
        return sector
    except Exception as e:
        print(f"⚠️ Erreur pour {symbol}: {e}")
        return 'ℹ️Inconnu!!'

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

# ----------------------------
# Nettoyage paresseux des groupes secteur × cap_range (complément + réduction)
# ----------------------------
def clean_sector_cap_groups(sector_cap_ranges: Dict[str, Dict[str, List[str]]],
                            ttl_days: int = 100,
                            min_symbols: int = 4,
                            max_symbols: int = 12) -> Dict[str, Dict[str, List[str]]]:
    """Nettoie et équilibre les groupes de symboles avec complément et réduction.

    Logique :
    - Utilise get_cleaned_group_cache pour éviter de recalculer si disponible et non expiré (100 jours).
    - Si groupe < 4 symboles : 
      1. Complète en puisant dans les symboles populaires du même secteur
      2. Si encore insuffisant, fusionne avec d'autres cap_range du même secteur
      3. Si toujours insuffisant, fallback sur tous les populaires (fusion transsectorielle)
    - Si groupe > 12 symboles : réduit à 12 (en gardant les meilleurs/premiers).
    - Sauvegarde le résultat nettoyé via save_cleaned_group_cache.
    """
    from symbol_manager import get_popular_symbols_by_sector, get_all_popular_symbols
    
    cleaned: Dict[str, Dict[str, List[str]]] = {}
    for sector, buckets in sector_cap_ranges.items():
        cleaned[sector] = {}
        for cap, syms in buckets.items():
            if not syms:
                cleaned[sector][cap] = []
                continue
            
            # Essayer le cache en premier mais toujours revalider les bornes min/max
            cached = None
            try:
                cached = get_cleaned_group_cache(sector, cap, ttl_days)
            except Exception:
                cached = None
            
            # Nettoyer et valider la liste de base (cache si dispo, sinon syms bruts)
            source_list = cached if cached is not None else syms
            base = [s for s in source_list if isinstance(s, str) and s.strip()]
            base = [s.strip().upper() for s in base]
            base = list(dict.fromkeys(base))  # Déduplicate
            
            # COMPLÉMENT (3 étapes) : si trop petit, remplir intelligemment
            if len(base) < min_symbols:
                needed = min_symbols - len(base)
                exclude_set = set(base)
                
                # Étape 1 : Populaires du même secteur
                try:
                    popular_same_sector = get_popular_symbols_by_sector(sector, max_count=needed * 2, exclude_symbols=exclude_set)
                    added = popular_same_sector[:needed]
                    base.extend(added)
                    exclude_set.update(added)
                    needed = min_symbols - len(base)
                except Exception:
                    pass
                
                # Étape 2 : Fusion avec d'autres cap_range du même secteur si encore insuffisant
                if needed > 0:
                    for other_cap, other_syms in buckets.items():
                        if other_cap == cap or needed <= 0:
                            continue
                        other_clean = [s.strip().upper() for s in other_syms if isinstance(s, str) and s.strip()]
                        candidates = [s for s in other_clean if s not in exclude_set]
                        added = candidates[:needed]
                        base.extend(added)
                        exclude_set.update(added)
                        needed = min_symbols - len(base)
                
                # Étape 3 : Fallback sur TOUS les populaires (transsectoriel)
                if needed > 0:
                    try:
                        all_popular = get_all_popular_symbols(max_count=needed * 2, exclude_symbols=exclude_set)
                        added = all_popular[:needed]
                        base.extend(added)
                        exclude_set.update(added)
                    except Exception:
                        pass
                
                base = list(dict.fromkeys(base))  # Rédéduplicate final
            
            # RÉDUCTION : si trop grand, réduire à max_symbols
            if len(base) > max_symbols:
                base = base[:max_symbols]
            
            cleaned[sector][cap] = base
            
            # Sauvegarder le groupe nettoyé
            try:
                save_cleaned_group_cache(sector, cap, base)
            except Exception:
                pass
    
    return cleaned


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

class HybridOptimizer:
    """Optimiseur hybride utilisant plusieurs stratégies d'optimisation avec limitation des décimales"""
    
    def __init__(self, stock_data, domain, montant=50, transaction_cost=1.0, precision=2, use_price_features: bool = False, use_fundamentals_features: bool = False):
        self.stock_data = stock_data
        self.domain = domain
        self.montant = montant
        self.transaction_cost = transaction_cost
        self.evaluation_count = 0
        self.best_cache = {}
        self.precision = precision  # 🔧 NUOVO: Précision des paramètres (nombre de décimales)
        self.use_price_features = use_price_features
        self.use_fundamentals_features = use_fundamentals_features
        
        # 🔧 Définir les bounds avec dimension réduite
        # - 8 coeffs (a1..a8)
        # - 4 seuils optimisés seulement: RSI, Volume, ADX, Score
        #   (les seuils MACD/EMA/Ichimoku/Bollinger sont fixés)
        # - 2 seuils globaux achat/vente
        base_bounds = (
            [(0.5, 3.0)] * 8 +  # coefficients a1-a8 (indices 0-7)
            [(30.0, 70.0)] +     # threshold RSI (index 8) - garder
            [(0.5, 2.5)] +       # threshold Volume (index 9) - garder
            [(15.0, 35.0)] +     # threshold ADX (index 10) - garder
            [(2.0, 6.0)] +       # threshold Score (index 11) - garder
            [(2.0, 6.0)] +       # seuil_achat global (index 12)
            [(-6.0, -2.0)]       # seuil_vente global (index 13)
        )
        self.bounds = base_bounds
        
        if self.use_price_features:
            # Extra 6 params: flags (rounded to 0/1), weights a9/a10, thresholds th9/th10 on relative values
            price_bounds = [
                (0.0, 1.0),  # use_price_slope
                (0.0, 1.0),  # use_price_acc
                (-0.5, 3.0),  # a9 weight
                (-0.5, 3.0),  # a10 weight
                (-0.15, 0.15),  # th9 on price_slope_rel
                (-0.15, 0.15),  # th10 on price_acc_rel
            ]
            self.bounds += price_bounds

        if self.use_fundamentals_features:
            # Extra 11 params for fundamentals: 1 flag, 5 weights, 5 thresholds
            fundamentals_bounds = [
                (0.0, 1.0),    # use_fundamentals flag
                (0.0, 3.0),    # a_rev_growth weight
                (0.0, 3.0),    # a_eps_growth weight
                (0.0, 3.0),    # a_roe weight
                (0.0, 3.0),    # a_fcf_yield weight
                (0.0, 3.0),    # a_de_ratio weight
                (-30.0, 30.0),  # th_rev_growth
                (-30.0, 30.0),  # th_eps_growth
                (-30.0, 30.0),  # th_roe
                (-10.0, 10.0),  # th_fcf_yield
                (-10.0, 10.0),  # th_de_ratio (MANQUANT AVANT!)
            ]
            self.bounds += fundamentals_bounds
        
        # ✨ V2.0: Charger les paramètres optimisés existants comme point de départ
        self.optimized_coeffs_loaded = False
        self.initial_coeffs = None
        self.initial_thresholds = None
        self.meilleur_score = -float('inf')  # 🔧 Meilleur score trouvé (global)
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

        # Extraire les paramètres : 8 coeffs + 4 seuils optimisés + 2 seuils globaux
        coeffs = tuple(params[:8])
        thr_rsi, thr_vol, thr_adx, thr_score = params[8:12]
        seuil_achat = float(params[12])  # Seuil global achat
        seuil_vente = float(params[13])  # Seuil global vente

        # Contraintes avec arrondi sur les coefficients
        coeffs = tuple(np.clip(self.round_params(coeffs), 0.5, 3.0))
        
        # Contraintes sur les seuils features (optimisés ou fixés)
        thr_rsi = np.clip(round(thr_rsi, self.precision), 30.0, 70.0)      # RSI (optimisé)
        thr_vol = np.clip(round(thr_vol, self.precision), 0.5, 2.5)        # Volume (optimisé)
        thr_adx = np.clip(round(thr_adx, self.precision), 15.0, 35.0)      # ADX (optimisé)
        thr_score = np.clip(round(thr_score, self.precision), 2.0, 6.0)    # Score (optimisé)

        # Seuils fixés/gélés non optimisés
        thr_macd = 0.0
        thr_ema = 0.0
        thr_ichimoku = 0.0
        thr_boll = 0.5

        # Reconstituer le vecteur complet de 8 seuils attendu par backtest
        feature_thresholds = [
            thr_rsi,
            thr_macd,
            thr_ema,
            thr_vol,
            thr_adx,
            thr_ichimoku,
            thr_boll,
            thr_score,
        ]
        feature_thresholds = tuple(feature_thresholds)
        
        # Contraintes sur les seuils globaux
        seuil_achat = np.clip(round(seuil_achat, self.precision), 2.0, 6.0)
        seuil_vente = np.clip(round(seuil_vente, self.precision), -6.0, -2.0)

        # Extract optional price extras
        price_extras = None
        if self.use_price_features and len(params) >= 20:
            # Round flags to 0/1
            use_ps = int(round(np.clip(params[14], 0.0, 1.0)))
            use_pa = int(round(np.clip(params[15], 0.0, 1.0)))
            a_ps = float(np.clip(round(params[16], self.precision), 0.0, 3.0))
            a_pa = float(np.clip(round(params[17], self.precision), 0.0, 3.0))
            th_ps = float(np.clip(round(params[18], self.precision), -0.05, 0.05))
            th_pa = float(np.clip(round(params[19], self.precision), -0.05, 0.05))
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
        # Nouveau décalage après réduction de dimension :
        # base = 14 (8 coeffs + 4 seuils + 2 globaux)
        # price features = +6
        fundamentals_index_offset = 20 if self.use_price_features else 14
        if self.use_fundamentals_features and len(params) >= (fundamentals_index_offset + 11):  # 🔧 Changé de 10 à 11
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
            th_de = float(np.clip(round(params[fundamentals_index_offset + 10], self.precision), -10.0, 10.0))  # 🔧 Nouveau: lire le 11ème param
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
                'th_de_ratio': th_de,  # 🔧 Utiliser la vraie valeur au lieu de 0.0
            }

        total_gain = 0.0
        total_trades = 0
        
        # 🚀 Helper function for parallel execution
        def evaluate_symbol(symbol, data):
            try:
                # 🚀 TOUJOURS utiliser l'accélération C (même avec features)
                result = backtest_signals_c_extended(
                    data['Close'], data['Volume'],
                    coeffs=coeffs,
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    montant=self.montant, transaction_cost=self.transaction_cost,
                    price_extras=price_extras if self.use_price_features else None,
                    fundamentals_extras=fundamentals_extras if self.use_fundamentals_features else None,
                    symbol_name=symbol
                )
                return result['gain_total'], result['trades']
            except Exception as e:
                return 0.0, 0
        
        try:
            # ⚡⚡ PARALLÉLISATION: Évaluer tous les symboles en parallèle
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(self.stock_data))) as executor:
                futures = {executor.submit(evaluate_symbol, symbol, data): symbol 
                          for symbol, data in self.stock_data.items()}
                
                for future in as_completed(futures):
                    gain, trades = future.result()
                    total_gain += gain
                    total_trades += trades

            avg_gain = total_gain / len(self.stock_data) if self.stock_data else 0.0
            self.evaluation_count += 1

            # 🚦 Gardes: rejeter les configs sans trades pour éviter des scores incohérents
            if total_trades == 0:
                penalized = -1e6
                self.best_cache[param_key] = penalized
                return penalized
            
            # 🔧 Tracker le nombre de trades de la meilleure config (global, pas par cache)
            if avg_gain > self.meilleur_score:
                self.meilleur_score = avg_gain
                self.meilleur_trades = total_trades

            # Cache le résultat
            self.best_cache[param_key] = avg_gain
            return avg_gain

        except Exception as e:
            print(f"⚠️ evaluate_config error: {e}")  # Debug: show exceptions
            return -1000.0  # Pénalité pour configurations invalides

    def genetic_algorithm(self, population_size=50, generations=30, mutation_rate=0.15):
        """Algorithme génétique pour l'optimisation avec précision limitée"""
        print(f"🧬 Démarrage algorithme génétique (pop={population_size}, gen={generations}, précision={self.precision})")
        
        # Utiliser self.bounds (16 paramètres: 8 coefficients + 8 seuils individuels)
        bounds = self.bounds
        population = []
        for _ in range(population_size):
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
            population.append(np.array(individual))

        best_fitness = -float('inf')
        best_individual = None

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

                pbar.set_postfix({'Meilleur': f"{best_fitness:.3f}", 'Trades': self.meilleur_trades})
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

    def differential_evolution_opt(self, population_size=45, max_iterations=100):
        """Optimisation par évolution différentielle avec arrondi"""
        print(f"🔄 Démarrage évolution différentielle (pop={population_size}, iter={max_iterations}, précision={self.precision})")
        
        bounds = self.bounds
        dim = len(bounds)

        with tqdm(total=max_iterations, desc="🔄 Évolution différentielle", unit="iter") as pbar:
            def callback(xk, convergence):
                pbar.set_postfix({'Convergence': f"{convergence:.6f}", 'Trades': self.meilleur_trades})
                pbar.update(1)

            result = differential_evolution(
                _de_objective,
                bounds,
                args=(self,),
                maxiter=max_iterations,
                popsize=population_size,
                mutation=(0.5, 1.5),
                recombination=0.7,
                callback=callback,
                polish=False,
                seed=np.random.randint(0, 10000),
                workers=-1,  # Use all cores; safe thanks to top-level objective
            )

        return self.round_params(result.x), -result.fun

    def latin_hypercube_sampling(self, n_samples=500):
        """Échantillonnage Latin Hypercube avec arrondi"""
        print(f"🎯 Latin Hypercube Sampling avec {n_samples} échantillons (précision={self.precision})")
        
        # 🔧 CORRIGÉ: Utiliser la dimension réelle des bounds au lieu de 18 hardcodé
        n_dimensions = len(self.bounds)
        sampler = qmc.LatinHypercube(d=n_dimensions)
        samples = sampler.random(n=n_samples)

        # Mise à l'échelle
        bounds = self.bounds
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        # 🔧 MODIFIÉ: Arrondir les échantillons
        scaled_samples = np.array([self.round_params(sample) for sample in scaled_samples])

        best_params = None
        best_score = -float('inf')

        with tqdm(total=n_samples, desc="🎯 LHS Exploration", unit="sample") as pbar:
            for sample in scaled_samples:
                score = self.evaluate_config(sample)
                if score > best_score:
                    best_score = score
                    best_params = sample.copy()

                pbar.set_postfix({'Meilleur': f"{best_score:.3f}", 'Trades': self.meilleur_trades})
                pbar.update(1)

        return best_params, best_score

    def particle_swarm_optimization(self, n_particles=30, max_iterations=50):
        """Optimisation par essaim particulaire (PSO) avec arrondi"""
        print(f"🐝 Particle Swarm Optimization (particles={n_particles}, iter={max_iterations}, précision={self.precision})")
        
        bounds = np.array(self.bounds)
        n_dims = len(self.bounds)  # 🔧 Dimension dynamique

        # Initialisation avec arrondi
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_dims))
        particles = np.array([self.round_params(p) for p in particles])  # 🔧 MODIFIÉ
        
        velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([self.evaluate_config(p) for p in particles])

        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

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

                pbar.set_postfix({'Meilleur': f"{global_best_score:.3f}", 'Trades': self.meilleur_trades})
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
    use_fundamentals_features=False
):
    """
    Optimisation hybride des coefficients sectoriels avec limitation des décimales
    
    Strategies disponibles:
    - 'genetic': Algorithmes génétiques
    - 'differential': Évolution différentielle  
    - 'pso': Particle Swarm Optimization
    - 'lhs': Latin Hypercube Sampling
    - 'hybrid': Combine plusieurs méthodes
    
    precision: Nombre de décimales pour les paramètres (1, 2, ou 3)
    cap_range: Segment de capitalisation associé au secteur
    """
    if not sector_symbols:
        print(f"🚫 Secteur {domain} vide, ignoré")
        return None, 0.0, 0.0, initial_thresholds, None

    # Téléchargement des données
    stock_data = download_stock_data(sector_symbols, period=period)
    if not stock_data:
        print(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
        return None, 0.0, 0.0, initial_thresholds, None

    # Pré-calcul léger des features pour chauffer les opérations pandas
    def precalculate_features(sd: Dict[str, Dict[str, pd.Series]]):
        try:
            for sym, dat in sd.items():
                close = dat.get('Close')
                volume = dat.get('Volume')
                if isinstance(close, pd.Series) and len(close) >= 50:
                    # Calculs légers pour chauffer les caches internes
                    _ema20 = close.ewm(span=20, adjust=False).mean()
                    _ema50 = close.ewm(span=50, adjust=False).mean()
                    _vol_mean = volume.rolling(window=30).mean() if isinstance(volume, pd.Series) else None
                    # Stockage facultatif pour analyse future (non utilisé par backtest)
                    dat['PRECALC_EMA20'] = _ema20
                    dat['PRECALC_EMA50'] = _ema50
                    if _vol_mean is not None:
                        dat['PRECALC_VOL30'] = _vol_mean
        except Exception:
            pass

    precalculate_features(stock_data)

    for symbol, data in stock_data.items():
        print(f"📊 {symbol}: {len(data['Close'])} points de données")

    # Récupération des meilleurs paramètres historiques
    db_path = 'signaux/optimization_hist.db'
    best_params_per_sector = extract_best_parameters(db_path)

    if domain in best_params_per_sector:
        csv_coeffs, csv_thresholds, csv_globals, csv_gain, _ = best_params_per_sector[domain]
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
                # 🔧 Vérifier si les features sont RÉELLEMENT activées (pas juste présentes)
                price_features_active = hist_extra_params is not None and (
                    hist_extra_params.get('use_price_slope', 0) or hist_extra_params.get('use_price_acc', 0)
                )
                fund_features_active = hist_fundamentals_extras is not None and (
                    hist_fundamentals_extras.get('use_fundamentals', 0)
                )
                
                if price_features_active or fund_features_active:
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
                    # Features désactivées → backtest classique C accéléré
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

            # 🚦 Gardes: ignorer les historiques sans trades
            if total_trades == 0:
                print(f"   ⚠️  Historique ignoré (0 trade)")
                historical_candidate = None
            else:
                # ✨ Appliquer la nouvelle structure réduite : 8 coeffs + 4 seuils optimisés + 2 globaux
                # Recomposer les 8 seuils attendus par le backtest avec les valeurs gelées
                hist_thr_rsi = hist_feature_thresholds[0] if len(hist_feature_thresholds) > 0 else 50.0
                hist_thr_vol = hist_feature_thresholds[3] if len(hist_feature_thresholds) > 3 else 1.0
                hist_thr_adx = hist_feature_thresholds[4] if len(hist_feature_thresholds) > 4 else 25.0
                hist_thr_score = hist_feature_thresholds[7] if len(hist_feature_thresholds) > 7 else 4.0
                hist_feature_thresholds = (
                    round(hist_thr_rsi, precision),  # RSI optimisé
                    0.0,                              # MACD gelé
                    0.0,                              # EMA gelé
                    round(hist_thr_vol, precision),   # Volume optimisé
                    round(hist_thr_adx, precision),   # ADX optimisé
                    0.0,                              # Ichimoku gelé
                    0.5,                              # Bollinger gelé
                    round(hist_thr_score, precision), # Score optimisé
                )

                # Paramètres de base pour comparaison finale : 14 (8 coeffs + 4 seuils + 2 globaux)
                hist_params = list(hist_coeffs[:8] + (hist_thr_rsi, hist_thr_vol, hist_thr_adx, hist_thr_score, hist_seuil_achat, hist_seuil_vente))
                
                # 🔧 Étendre les paramètres historiques si des features sont activées
                # IMPORTANT: Utiliser les vraies valeurs historiques depuis BEST_PARAM_EXTRAS, pas des 0 !
                if use_price_features:
                    # Récupérer les valeurs historiques ou utiliser des valeurs par défaut si absentes
                    if hist_extra_params:
                        hist_params += [
                            float(hist_extra_params.get('use_price_slope', 0)),
                            float(hist_extra_params.get('use_price_acc', 0)),
                            float(hist_extra_params.get('a_price_slope', 0.5)),
                            float(hist_extra_params.get('a_price_acc', 0.5)),
                            float(hist_extra_params.get('th_price_slope', 0.0)),
                            float(hist_extra_params.get('th_price_acc', 0.0))
                        ]
                    else:
                        # Pas de paramètres historiques → features désactivées
                        hist_params += [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
                
                if use_fundamentals_features:
                    # Récupérer les valeurs historiques ou utiliser des valeurs par défaut si absentes
                    if hist_fundamentals_extras:
                        hist_params += [
                            float(hist_fundamentals_extras.get('use_fundamentals', 0)),
                            float(hist_fundamentals_extras.get('a_rev_growth', 0.5)),
                            float(hist_fundamentals_extras.get('a_eps_growth', 0.5)),
                            float(hist_fundamentals_extras.get('a_roe', 0.5)),
                            float(hist_fundamentals_extras.get('a_fcf_yield', 0.5)),
                            float(hist_fundamentals_extras.get('a_de_ratio', 0.5)),
                            float(hist_fundamentals_extras.get('th_rev_growth', 0.0)),
                            float(hist_fundamentals_extras.get('th_eps_growth', 0.0)),
                            float(hist_fundamentals_extras.get('th_roe', 0.0)),
                            float(hist_fundamentals_extras.get('th_fcf_yield', 0.0)),
                            float(hist_fundamentals_extras.get('th_de_ratio', 0.0))
                        ]
                    else:
                        # Pas de paramètres historiques → features désactivées
                        hist_params += [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
                
                historical_candidate = ('Historical (re-eval)', tuple(hist_params), hist_avg_gain)

                print(f"♻️ Paramètres historiques réévalués sur données actuelles: gain_moy={hist_avg_gain:.2f}, trades={total_trades}, success={hist_success_rate:.2f}%")
                if use_price_features or use_fundamentals_features:
                    # Afficher l'état réel des features historiques
                    price_active = hist_extra_params and (hist_extra_params.get('use_price_slope', 0) or hist_extra_params.get('use_price_acc', 0))
                    fund_active = hist_fundamentals_extras and hist_fundamentals_extras.get('use_fundamentals', 0)
                    status_msg = f"price={'✅' if price_active else '❌'}, fundamentals={'✅' if fund_active else '❌'}"
                    print(f"   ℹ️  Paramètres étendus de 14 → {len(hist_params)} ({status_msg})")
        except Exception as e:
            print(f"⚠️ Réévaluation des paramètres historiques impossible: {e}")

    # 🔧 MODIFIÉ: Initialisation de l'optimiseur avec précision
    optimizer = HybridOptimizer(stock_data, domain, montant, transaction_cost, precision, use_price_features, use_fundamentals_features)

    # 🔧 NOUVEAU: Ajuster le budget en fonction de la précision
    # Avec une plus grande précision, l'espace de recherche AUGMENTE
    # Plus de décimales = plus de points possibles à explorer
    # Donc on doit AUGMENTER le budget proportionnellement
    # Facteur: 1.0 pour précision=1, 1.5 pour précision=2, 2.0 pour précision=3
    precision_factor = 1.0 + (precision - 1) * 1.5
    adjusted_budget = int(budget_evaluations * precision_factor)
    
    # Stratégies d'optimisation
    results = []
    # Ajouter le candidat "historical" réévalué si disponible et valide
    if historical_candidate:
        results.append(historical_candidate)
    print(f"🚀 Optimisation hybride pour {domain} avec stratégie '{strategy}' (précision: {precision} décimales)")
    print(f"📈 Budget d'évaluations initial: {budget_evaluations}")
    print(f"🔧 Budget ajusté selon la précision: {adjusted_budget} (facteur: {precision_factor:.2f}x)")

    if  strategy == 'genetic': # or strategy == 'hybrid':
        # Algorithmes génétiques - 🔧 Augmenter max pop_size selon budget
        pop_size = min(100, adjusted_budget // 20)  # Max 100 au lieu de 50
        generations = min(50, adjusted_budget // pop_size)  # Generations aussi augmentées
        params_ga, score_ga = optimizer.genetic_algorithm(pop_size, generations)
        results.append(('Genetic Algorithm', params_ga, score_ga))

    if strategy == 'hybrid' or strategy == 'differential':
        # Évolution différentielle - 🔧 Augmenter max pop_size selon budget
        pop_size = min(90, adjusted_budget // 25)  # Max 90 au lieu de 45
        max_iter = min(100, adjusted_budget // pop_size)
        params_de, score_de = optimizer.differential_evolution_opt(pop_size, max_iter)
        results.append(('Differential Evolution', params_de, score_de))

    if strategy == 'hybrid' or strategy == 'pso':
        # PSO - 🔧 Augmenter max particules selon budget
        n_particles = min(50, adjusted_budget // 30)  # Max 50 au lieu de 30
        max_iter = min(100, adjusted_budget // n_particles)
        params_pso, score_pso = optimizer.particle_swarm_optimization(n_particles, max_iter)
        results.append(('PSO', params_pso, score_pso))

    if strategy == 'hybrid' or strategy == 'lhs':
        # Latin Hypercube Sampling - 🔧 Utiliser pleinement le budget
        n_samples = min(500, adjusted_budget // 2)  # Max 500 au lieu de 200
        params_lhs, score_lhs = optimizer.latin_hypercube_sampling(n_samples)
        results.append(('Latin Hypercube', params_lhs, score_lhs))

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

    # 🔧 MODIFIÉ: Extraction conforme à la nouvelle structure (8 coeffs + 4 seuils optimisés + 2 globaux)
    best_coeffs = tuple(float(x) for x in best_params[:8])
    thr_rsi = float(best_params[8])
    thr_vol = float(best_params[9])
    thr_adx = float(best_params[10])
    thr_score = float(best_params[11])
    best_seuil_achat = float(best_params[12])
    best_seuil_vente = float(best_params[13])

    # Recomposer les 8 seuils attendus par le backtest avec valeurs gelées
    best_feature_thresholds = (
        thr_rsi,
        0.0,
        0.0,
        thr_vol,
        thr_adx,
        0.0,
        0.5,
        thr_score,
    )

    # 🔧 DEBUG: Taille du vecteur de params
    print(f"   🔍 Taille du meilleur vecteur de params: {len(best_params)} (expected: {len(optimizer.bounds)})")

    # Extra params (price features) if present in vector (nouvelle position: après 14 de base)
    extra_params = None
    price_index_offset = 14
    if use_price_features and len(best_params) >= (price_index_offset + 6):
        use_ps = int(round(np.clip(best_params[price_index_offset], 0.0, 1.0)))
        use_pa = int(round(np.clip(best_params[price_index_offset + 1], 0.0, 1.0)))
        a_ps = float(np.clip(round(best_params[price_index_offset + 2], precision), 0.0, 3.0))
        a_pa = float(np.clip(round(best_params[price_index_offset + 3], precision), 0.0, 3.0))
        th_ps = float(np.clip(round(best_params[price_index_offset + 4], precision), -0.05, 0.05))
        th_pa = float(np.clip(round(best_params[price_index_offset + 5], precision), -0.05, 0.05))
        extra_params = {
            'use_price_slope': use_ps,
            'use_price_acc': use_pa,
            'a_price_slope': a_ps,
            'a_price_acc': a_pa,
            'th_price_slope': th_ps,
            'th_price_acc': th_pa,
        }
    
    # Fundamentals extras if present in vector (nouvelle position: après prix ou base)
    fundamentals_extras = None
    fundamentals_index_offset = price_index_offset + 6 if use_price_features else 14
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
    
    # 🔧 Vérifier si les features sont RÉELLEMENT activées
    price_features_active = extra_params is not None and (
        extra_params.get('use_price_slope', 0) or extra_params.get('use_price_acc', 0)
    )
    fund_features_active = fundamentals_extras is not None and (
        fundamentals_extras.get('use_fundamentals', 0)
    )
    
    print(f"   🔍 Recalcul avec: use_price_features={use_price_features}, use_fundamentals_features={use_fundamentals_features}")
    print(f"      extra_params={extra_params}")
    print(f"      fundamentals_extras={fundamentals_extras}")
    print(f"      Features réellement actives: price={price_features_active}, fund={fund_features_active}")
    
    for symbol, data in stock_data.items():
        if price_features_active or fund_features_active:
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
    if total_trades == 0:
        # Ne rien sauvegarder si la config ne déclenche aucun trade
        should_save = False
    
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
        
        # Résumé par secteur
        print(f"\n   📈 Détail par secteur:")
        sector_symbol_counts = {}
        for sector, cap_dict in sector_cap_ranges.items():
            total_syms = sum(len(syms) for syms in cap_dict.values())
            if total_syms > 0:
                sector_symbol_counts[sector] = total_syms
                print(f"      • {sector}: {total_syms} symboles ({len(cap_dict)} cap_range(s))")
        
        print(f"      Total secteurs: {len(sector_symbol_counts)}")

        # Nettoyage paresseux avec cache
        print("\n   🧹 Nettoyage des groupes avec cache (complément + réduction)...")
        sector_cap_ranges = clean_sector_cap_groups(sector_cap_ranges, ttl_days=100, min_symbols=4, max_symbols=12)
    
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

        # Nettoyage paresseux avec cache sur la méthode fallback
        print("\n   🧹 Nettoyage des groupes avec cache (complément + réduction)...")
        sector_cap_ranges = clean_sector_cap_groups(sector_cap_ranges, ttl_days=100, min_symbols=4, max_symbols=12)

    # Paramètres d'optimisation
    print("\n3️⃣  Configuration de l'optimisation:")
    search_strategies = ['hybrid', 'differential', 'genetic', 'pso', 'lhs']
    
    strategy = input("   Stratégie ('hybrid', 'differential', 'genetic', 'pso', 'lhs') : ").strip().lower()
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
    
    # Calcul du nombre de paramètres (ajusté après réduction de dimension)
    param_count = 14  # Base réduite: 8 coeffs + 4 seuils optimisés + 2 globaux
    if use_price_features:
        param_count += 6
    if use_fundamentals_features:
        param_count += 10
    
    # Vérifier si accélération C disponible
    try:
        from trading_c_acceleration.qsi_optimized import C_ACCELERATION
        accel_status = "⚡ Module C activé" if C_ACCELERATION else "🐍 Python vectorisé"
    except:
        accel_status = "🐍 Python vectorisé"
    
    print(f"\n   🔧 Configuration:")
    print(f"      - Stratégie: {strategy}")
    print(f"      - Précision: {precision} décimales")
    print(f"      - Accélération: {accel_status}")
    print(f"      - Parallélisation: ⚡⚡ {min(MAX_WORKERS, len(symbols))} workers (ThreadPool)")
    print(f"      - Price features: {'✅ Activé (C accéléré)' if use_price_features else '❌ Désactivé'}")
    print(f"      - Fundamentals features: {'✅ Activé (C accéléré)' if use_fundamentals_features else '❌ Désactivé'}")
    print(f"      - Nombre de paramètres: {param_count}")
    print(f"      - Seuils gelés: MACD=0, EMA=0, Ichimoku=0, Bollinger=0.5")

    # Adapter le budget selon la précision
    budget_base = 1500#1000
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