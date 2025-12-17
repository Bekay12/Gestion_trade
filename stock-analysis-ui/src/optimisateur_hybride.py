# optimisateur_hybride_fixed.py
# Version optimisée avec limitation des décimales pour réduire l'espace de recherche

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path
sys.path.append("C:\\Users\\berti\\Desktop\\Mes documents\\Gestion_trade\\stock-analysis-ui\\src\\trading_c_acceleration")
from qsi import download_stock_data, load_symbols_from_txt, extract_best_parameters
from qsi_optimized import backtest_signals, extract_best_parameters
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
from collections import deque
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

def get_sector(symbol):
    """Récupère le secteur d'une action avec logs pour diagnostic"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'ℹ️Inconnu!!')
        print(f"📋 {symbol}: Secteur = {sector}")
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

class HybridOptimizer:
    """Optimiseur hybride utilisant plusieurs stratégies d'optimisation avec limitation des décimales"""
    
    def __init__(self, stock_data, domain, montant=50, transaction_cost=1.0, precision=2):
        self.stock_data = stock_data
        self.domain = domain
        self.montant = montant
        self.transaction_cost = transaction_cost
        self.evaluation_count = 0
        self.best_cache = {}
        self.precision = precision  # 🔧 NOUVEAU: Précision des paramètres (nombre de décimales)
        
        # 🔧 Définir les bounds une seule fois (18 paramètres: 8 coeffs + 8 seuils + 2 seuils globaux)
        self.bounds = (
            [(0.5, 3.0)] * 8 +  # coefficients a1-a8 (indices 0-7)
            [(30.0, 70.0)] +     # threshold 0: RSI_threshold (index 8)
            [(-1.0, 1.0)] +      # threshold 1: MACD_threshold (index 9)
            [(-1.0, 1.0)] +      # threshold 2: EMA_threshold (index 10)
            [(0.5, 2.5)] +       # threshold 3: Volume_threshold (index 11)
            [(15.0, 35.0)] +     # threshold 4: ADX_threshold (index 12)
            [(-1.0, 1.0)] +      # threshold 5: Ichimoku_threshold (index 13)
            [(0.3, 0.7)] +       # threshold 6: Bollinger_threshold (index 14)
            [(2.0, 6.0)] +       # threshold 7: Score_global_threshold (index 15)
            [(2.0, 6.0)] +       # seuil_achat global (index 16)
            [(-6.0, -2.0)]       # seuil_vente global (index 17)
        )
        
        # ✨ V2.0: Charger les paramètres optimisés existants comme point de départ
        self.optimized_coeffs_loaded = False
        self.initial_coeffs = None
        self.initial_thresholds = None
        
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
        coeffs = tuple(params[:8])
        feature_thresholds = tuple(params[8:16])  # 8 seuils individuels
        seuil_achat = float(params[16])  # Seuil global achat
        seuil_vente = float(params[17])  # Seuil global vente

        # Contraintes avec arrondi sur les coefficients
        coeffs = tuple(np.clip(self.round_params(coeffs), 0.5, 3.0))
        
        # Contraintes sur les seuils features
        feature_thresholds = list(feature_thresholds)
        feature_thresholds[0] = np.clip(round(feature_thresholds[0], self.precision), 30.0, 70.0)  # RSI_threshold
        feature_thresholds[1] = np.clip(round(feature_thresholds[1], self.precision), -1.0, 1.0)   # MACD_threshold
        feature_thresholds[2] = np.clip(round(feature_thresholds[2], self.precision), -1.0, 1.0)   # EMA_threshold
        feature_thresholds[3] = np.clip(round(feature_thresholds[3], self.precision), 0.5, 2.5)    # Volume_threshold
        feature_thresholds[4] = np.clip(round(feature_thresholds[4], self.precision), 15.0, 35.0)  # ADX_threshold
        feature_thresholds[5] = np.clip(round(feature_thresholds[5], self.precision), -1.0, 1.0)   # Ichimoku_threshold
        feature_thresholds[6] = np.clip(round(feature_thresholds[6], self.precision), 0.3, 0.7)    # Bollinger_threshold
        feature_thresholds[7] = np.clip(round(feature_thresholds[7], self.precision), 2.0, 6.0)    # Score_global_threshold
        feature_thresholds = tuple(feature_thresholds)
        
        # Contraintes sur les seuils globaux
        seuil_achat = np.clip(round(seuil_achat, self.precision), 2.0, 6.0)
        seuil_vente = np.clip(round(seuil_vente, self.precision), -6.0, -2.0)

        total_gain = 0.0
        total_trades = 0
        try:
            for data in self.stock_data.values():
                result = backtest_signals(
                    data['Close'], data['Volume'], self.domain,
                    domain_coeffs={self.domain: coeffs},
                    domain_thresholds={self.domain: feature_thresholds},
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    montant=self.montant, transaction_cost=self.transaction_cost
                )
                total_gain += result['gain_total']
                total_trades += result['trades']

            avg_gain = total_gain / len(self.stock_data) if self.stock_data else 0.0
            self.evaluation_count += 1

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

                pbar.set_postfix({'Meilleur': f"{best_fitness:.3f}", 'Eval': self.evaluation_count})
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

        def objective_function(params):
            return -self.evaluate_config(self.round_params(params))  # 🔧 MODIFIÉ: Arrondir avant évaluation

        with tqdm(total=max_iterations, desc="🔄 Évolution différentielle", unit="iter") as pbar:
            def callback(xk, convergence):
                pbar.set_postfix({'Convergence': f"{convergence:.6f}", 'Eval': self.evaluation_count})
                pbar.update(1)

            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iterations,
                popsize=population_size,
                mutation=(0.5, 1.5),
                recombination=0.7,
                callback=callback,
                polish=False,
                seed=np.random.randint(0, 10000)
            )

        return self.round_params(result.x), -result.fun

    def latin_hypercube_sampling(self, n_samples=500):
        """Échantillonnage Latin Hypercube avec arrondi"""
        print(f"🎯 Latin Hypercube Sampling avec {n_samples} échantillons (précision={self.precision})")
        
        sampler = qmc.LatinHypercube(d=18)
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

                pbar.set_postfix({'Meilleur': f"{best_score:.3f}", 'Eval': self.evaluation_count})
                pbar.update(1)

        return best_params, best_score

    def particle_swarm_optimization(self, n_particles=30, max_iterations=50):
        """Optimisation par essaim particulaire (PSO) avec arrondi"""
        print(f"🐝 Particle Swarm Optimization (particles={n_particles}, iter={max_iterations}, précision={self.precision})")
        
        bounds = np.array(self.bounds)

        # Initialisation avec arrondi
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, 18))
        particles = np.array([self.round_params(p) for p in particles])  # 🔧 MODIFIÉ
        
        velocities = np.random.uniform(-1, 1, (n_particles, 18))
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

                pbar.set_postfix({'Meilleur': f"{global_best_score:.3f}", 'Eval': self.evaluation_count})
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
    precision=2  # 🔧 NOUVEAU: Paramètre de précision
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
    """
    if not sector_symbols:
        print(f"🚫 Secteur {domain} vide, ignoré")
        return None, 0.0, 0.0, initial_thresholds

    # Téléchargement des données
    stock_data = download_stock_data(sector_symbols, period=period)
    if not stock_data:
        print(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
        return None, 0.0, 0.0, initial_thresholds

    for symbol, data in stock_data.items():
        print(f"📊 {symbol}: {len(data['Close'])} points de données")

    # Récupération des meilleurs paramètres historiques
    csv_path = 'signaux/optimization_hist_4stp.csv'
    best_params_per_sector = extract_best_parameters(csv_path)

    if domain in best_params_per_sector:
        csv_coeffs, csv_thresholds, csv_gain = best_params_per_sector[domain]
        print(f"📋 Paramètres historiques trouvés: coeffs={csv_coeffs}, seuils={csv_thresholds}, gain={csv_gain:.2f}")
    else:
        csv_coeffs, csv_thresholds, csv_gain = None, initial_thresholds, -float('inf')

    # 🔧 MODIFIÉ: Initialisation de l'optimiseur avec précision
    optimizer = HybridOptimizer(stock_data, domain, montant, transaction_cost, precision)

    # Stratégies d'optimisation
    results = []
    print(f"🚀 Optimisation hybride pour {domain} avec stratégie '{strategy}' (précision: {precision} décimales)")
    print(f"📈 Budget d'évaluations: {budget_evaluations}")

    if strategy == 'hybrid' or strategy == 'genetic':
        # Algorithmes génétiques
        pop_size = min(50, budget_evaluations // 20)
        generations = min(30, budget_evaluations // pop_size)
        params_ga, score_ga = optimizer.genetic_algorithm(pop_size, generations)
        results.append(('Genetic Algorithm', params_ga, score_ga))

    if strategy == 'hybrid' or strategy == 'differential':
        # Évolution différentielle
        pop_size = min(45, budget_evaluations // 25)
        max_iter = min(100, budget_evaluations // pop_size)
        params_de, score_de = optimizer.differential_evolution_opt(pop_size, max_iter)
        results.append(('Differential Evolution', params_de, score_de))

    if strategy == 'hybrid' or strategy == 'pso':
        # PSO
        n_particles = min(30, budget_evaluations // 30)
        max_iter = min(50, budget_evaluations // n_particles)
        params_pso, score_pso = optimizer.particle_swarm_optimization(n_particles, max_iter)
        results.append(('PSO', params_pso, score_pso))

    if strategy == 'hybrid' or strategy == 'lhs':
        # Latin Hypercube Sampling
        n_samples = min(200, budget_evaluations // 5)
        params_lhs, score_lhs = optimizer.latin_hypercube_sampling(n_samples)
        results.append(('Latin Hypercube', params_lhs, score_lhs))

    # Sélection du meilleur résultat
    best_method, best_params, best_score = max(results, key=lambda x: x[2])
    print(f"🏆 Meilleure méthode: {best_method} avec score {best_score:.4f}")

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

    # Calcul des statistiques finales
    total_success = 0
    total_trades = 0
    for data in stock_data.values():
        result = backtest_signals(
            data['Close'], data['Volume'], domain,
            domain_coeffs={domain: best_coeffs},
            domain_thresholds={domain: best_feature_thresholds},  # V2.0: Passer tous les 8 seuils features
            seuil_achat=best_seuil_achat, seuil_vente=best_seuil_vente,  # V2.0: Passer les 2 seuils globaux
            montant=montant, transaction_cost=transaction_cost
        )
        total_success += result['gagnants']
        total_trades += result['trades']

    success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

    print(f"✅ Optimisation terminée:")
    print(f" 📊 Évaluations effectuées: {optimizer.evaluation_count}")
    print(f" 🎯 Meilleurs coefficients: {best_coeffs}")
    print(f" 🎯 Meilleurs seuils features: {best_feature_thresholds}")
    print(f" 🎯 Seuil achat global: {best_seuil_achat:.2f}")
    print(f" 🎯 Seuil vente global: {best_seuil_vente:.2f}")
    print(f" 💰 Gain moyen: {best_score:.2f}")
    print(f" 📈 Taux de réussite: {success_rate:.2f}%")
    print(f" 🔄 Nombre de trades: {total_trades}")

    # Sauvegarde des résultats - agrégé en un tuple unique
    all_thresholds = best_feature_thresholds + (best_seuil_achat, best_seuil_vente)
    save_optimization_results(domain, best_coeffs, best_score, success_rate, total_trades, all_thresholds)

    return best_coeffs, best_score, success_rate, all_thresholds

def save_optimization_results(domain, coeffs, gain_total, success_rate, total_trades, thresholds):
    """Sauvegarde les résultats dans un CSV et dans le gestionnaire de paramètres V2.0"""
    from datetime import datetime
    import pandas as pd

    results = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Sector': domain,
        'Gain_moy': gain_total,
        'Success_Rate': success_rate,
        'Trades': total_trades,
        # V2.0: Sauvegarder les 8 seuils features + 2 seuils globaux
        'Seuil_Achat': thresholds[8] if len(thresholds) > 8 else 4.20,  # Seuil achat global
        'Seuil_Vente': thresholds[9] if len(thresholds) > 9 else -0.5,  # Seuil vente global
        # Les 8 seuils features
        'th1': thresholds[0] if len(thresholds) > 0 else 50.0,  # RSI_threshold
        'th2': thresholds[1] if len(thresholds) > 1 else 0.0,   # MACD_threshold
        'th3': thresholds[2] if len(thresholds) > 2 else 0.0,   # EMA_threshold
        'th4': thresholds[3] if len(thresholds) > 3 else 1.2,   # Volume_threshold
        'th5': thresholds[4] if len(thresholds) > 4 else 25.0,  # ADX_threshold
        'th6': thresholds[5] if len(thresholds) > 5 else 0.0,   # Ichimoku_threshold
        'th7': thresholds[6] if len(thresholds) > 6 else 0.5,   # Bollinger_threshold
        'th8': thresholds[7] if len(thresholds) > 7 else 4.20,  # Score_global_threshold
        # Les 2 seuils globaux
        'th_achat': thresholds[8] if len(thresholds) > 8 else 4.20,  # Global buy threshold
        'th_vente': thresholds[9] if len(thresholds) > 9 else -0.5,  # Global sell threshold
        # Les 8 coefficients
        'a1': coeffs[0], 'a2': coeffs[1], 'a3': coeffs[2], 'a4': coeffs[3],
        'a5': coeffs[4], 'a6': coeffs[5], 'a7': coeffs[6], 'a8': coeffs[7]
    }

    csv_path = 'signaux/optimization_hist_4stp.csv'

    try:
        # Vérifier si le fichier existe et charger les données existantes
        if pd.io.common.file_exists(csv_path):
            df_existing = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

            # Filtrer les données pour le secteur concerné
            sector_data = df_existing[df_existing['Sector'] == domain]

            if not sector_data.empty:
                # Trouver les meilleurs résultats existants pour ce secteur
                best_gain = sector_data['Gain_moy'].max()
                best_success_rate = sector_data['Success_Rate'].max()

                # Vérifier si les nouveaux résultats sont meilleurs
                is_best_gain = gain_total > best_gain
                is_best_success_rate = success_rate > best_success_rate

                # Ne sauvegarder que si au moins un des critères est meilleur
                if not (is_best_gain or is_best_success_rate):
                    print(f"⚠️ Résultats pour {domain} non sauvegardés:")
                    print(f"   Gain moyen actuel: {gain_total:.4f} (meilleur existant: {best_gain:.4f})")
                    print(f"   Taux de réussite actuel: {success_rate:.4f} (meilleur existant: {best_success_rate:.4f})")
                    print(f"   Les nouveaux paramètres ne sont pas meilleurs que ceux existants.")
                    return

                # Afficher quel critère s'est amélioré
                if is_best_gain:
                    print(f"🎯 Nouveau meilleur gain moyen pour {domain}: {gain_total:.4f} (ancien: {best_gain:.4f})")
                if is_best_success_rate:
                    print(f"🎯 Nouveau meilleur taux de réussite pour {domain}: {success_rate:.4f} (ancien: {best_success_rate:.4f})")

        # Sauvegarder les nouveaux résultats dans le CSV
        df_new = pd.DataFrame([results])
        df_new.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)
        print(f"📝 Résultats sauvegardés dans CSV pour {domain}")

    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des symboles
    symbols = list(dict.fromkeys(load_symbols_from_txt("optimisation_symbols.txt")))

    # Créer le dictionnaire des secteurs dynamiquement
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

    # Assigner les symboles aux secteurs
    for symbol in symbols:
        sector = get_sector(symbol)
        if sector in sectors:
            sectors[sector].append(symbol)
        else:
            sectors["ℹ️Inconnu!!"].append(symbol)

    print("\n📋 Assignation des secteurs:")
    for sector, syms in sectors.items():
        print(f"{sector}: {syms}")

    # Paramètres d'optimisation
    search_strategies = ['hybrid', 'differential', 'genetic', 'pso', 'lhs']
    
    strategy = input("Choisissez la stratégie d'optimisation ('hybrid', 'differential', 'genetic', 'pso', 'lhs') : ").strip().lower()
    i=0
    while (strategy not in search_strategies) and i<3:
        strategy = input("Stratégie invalide. Veuillez choisir parmi ('hybrid', 'differential', 'genetic', 'pso', 'lhs') : ").strip().lower()
        i+=1
    if strategy not in search_strategies:
        strategy = random.choice(search_strategies)
        print("Stratégie inconnue, utilisation de la stratégie aléatoire:", strategy)

    # 🔧 NOUVEAU: Choix de la précision
    try:
        precision = int(input("Choisissez la précision (nombre de décimales: 1, 2, ou 3) [défaut: 2] : ").strip() or "2")
        if precision not in [1, 2, 3]:
            precision = 2
    except ValueError:
        precision = 2

    print(f"🔧 Paramètres choisis: stratégie={strategy}, précision={precision} décimales")

    budget_evaluations = 1500  # Budget total d'évaluations par secteur

    optimized_coeffs = {}

    for sector, sector_symbols in sectors.items():
        if not sector_symbols:
            print(f"🚫 Secteur {sector} vide, ignoré")
            continue

        print(f"\n" + "="*160)
        print(f"🎯 OPTIMISATION {strategy.upper()} - {sector}")
        print(f"="*160)

        coeffs, gain_total, success_rate, thresholds = optimize_sector_coefficients_hybrid(
            sector_symbols, sector,
            period='1y',
            strategy=strategy,
            montant=50,
            transaction_cost=0.02,
            budget_evaluations=budget_evaluations,
            precision=precision  # 🔧 NOUVEAU: Paramètre de précision
        )

        if coeffs:
            optimized_coeffs[sector] = coeffs

            print(f"\n✅ RÉSULTATS FINAUX - {sector}")
            print(f"   🔬 Méthode: Optimisation hybride (précision: {precision} décimales)")
            print(f"   🧬 Meilleurs coefficients: {coeffs}")
            print(f"   ⚖️ Meilleurs seuils (achat, vente): {thresholds}")
            print(f"   💰 Gain total moyen: {gain_total:.2f}")
            print(f"   📊 Taux de réussite: {success_rate:.2f}%")

    print("\n" + "="*80)
    print("🏆 DICTIONNAIRE FINAL OPTIMISÉ")
    print("="*80)
    print("domain_coeffs = {")
    for sector, coeffs in optimized_coeffs.items():
        print(f"    '{sector}': {coeffs},")
    print("}")
    print("="*80)