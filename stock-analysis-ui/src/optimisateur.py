from qsi import analyse_et_affiche, analyse_signaux_populaires, popular_symbols, mes_symbols, period
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution
import pandas as pd

# 1. Préparation des données historiques
def prepare_training_data(symbols, period="24mo"):
    """Prépare un dataset pour l'optimisation"""
    data = download_stock_data(symbols, period)
    X, y = [], []
    
    for symbol, stock_data in data.items():
        prices = stock_data['Close']
        volumes = stock_data['Volume']
        domaine = get_sector(symbol)  # À implémenter
        
        for i in range(50, len(prices) - 5):  # On garde 5 jours pour le label
            # Features
            X.append({
                'prices': prices[:i],
                'volumes': volumes[:i],
                'domaine': domaine
            })
            
            # Label: rendement à 5 jours
            future_return = (prices.iloc[i+5] - prices.iloc[i]) / prices.iloc[i]
            y.append(1 if future_return > 0.02 else -1 if future_return < -0.02 else 0)
    
    return X, np.array(y)

# 2. Fonction objective pour l'optimisation
def objective_function(params, X, y):
    """Évalue la performance d'un ensemble de paramètres"""
    a1, a2, a3, a4, a5, a6, a7, a8, seuil_achat, seuil_vente = params
    
    predictions = []
    for sample in X:
        # Paramètres temporaires
        coeffs = (a1, a2, a3, a4, a5, a6, a7, a8)
        domain_coeffs = {k: coeffs for k in [
            "Technology", "Healthcare", "Financial Services", 
            "Consumer Cyclical", "Industrials", "Energy",
            "Basic Materials", "Communication Services", 
            "Utilities", "Real Estate"
        ]}
        
        # Générer le signal
        signal, *_ = get_trading_signal(
            sample['prices'],
            sample['volumes'],
            sample['domaine'],
            domain_coeffs=domain_coeffs,
            seuil_achat=seuil_achat,
            seuil_vente=seuil_vente
        )
        
        # Convertir en prédiction numérique
        pred = 1 if signal == "ACHAT" else -1 if signal == "VENTE" else 0
        predictions.append(pred)
    
    # Calculer la précision pondérée
    correct = np.sum((np.array(predictions) == y) & (y != 0))
    total_signals = np.sum(y != 0)
    
    if total_signals == 0:
        return 0  # Pénalité si aucun signal
    
    return -correct / total_signals  # À minimiser

# 3. Optimisation avec algorithme génétique
def optimize_parameters(X_train, y_train):
    """Trouve les meilleurs paramètres avec differential evolution"""
    bounds = [
        (0.5, 3.0), (0.5, 3.0), (0.5, 3.0), (0.5, 3.0),  # a1-a4
        (0.5, 3.0), (0.5, 3.0), (0.5, 3.0), (0.5, 3.0),  # a5-a8
        (3.0, 7.0),  # seuil_achat
        (-5.0, -0.1)  # seuil_vente
    ]
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(X_train, y_train),
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )
    
    return result.x

# 4. Validation des résultats
def evaluate_parameters(params, X_test, y_test):
    """Évalue les paramètres optimisés sur l'ensemble de test"""
    return -objective_function(params, X_test, y_test)

# 5. Pipeline complet
def optimize_trading_system(symbols):
    # Préparer les données
    X, y = prepare_training_data(symbols)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Optimisation
    best_params = optimize_parameters(X_train, y_train)
    
    # Évaluation
    accuracy = evaluate_parameters(best_params, X_test, y_test)
    
    print(f"🔍 Résultats de l'optimisation:")
    print(f"- Précision: {accuracy:.2%}")
    print(f"- Meilleurs paramètres:")
    param_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'seuil_achat', 'seuil_vente']
    for name, value in zip(param_names, best_params):
        print(f"  {name}: {value:.4f}")
    
    return best_params

# Utilisation
if __name__ == "__main__":
    # Utiliser un sous-ensemble représentatif de vos symboles
    training_symbols = popular_symbols[:100]  # 100 symboles pour l'optimisation
    best_params = optimize_trading_system(training_symbols)
    
    # Appliquer les paramètres optimisés dans votre système
    a1, a2, a3, a4, a5, a6, a7, a8, seuil_achat, seuil_vente = best_params