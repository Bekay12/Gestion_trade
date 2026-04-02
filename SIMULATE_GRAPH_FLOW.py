#!/usr/bin/env python3
"""
Diagnostic: Trace le flux exact du cap_range de on_download_complete jusqu'à _build_symbol_figure_with_score
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

from qsi import download_stock_data, get_trading_signal, get_cap_range_for_symbol, extract_best_parameters, period
import yfinance as yf
from sector_normalizer import normalize_sector

def simulate_on_download_complete_and_graph(symbol):
    """Simule exactement le code dans on_download_complete + _build_symbol_figure_with_score"""
    
    print(f"\n{'='*120}")
    print(f"SIMULATION COMPLÈTE: {symbol}")
    print('='*120)
    
    # === PHASE 1: On Download Complete ===
    print("\n[1] PHASE: on_download_complete()")
    print("─" * 120)
    
    data_dict = download_stock_data([symbol], period=period)
    if symbol not in data_dict:
        print(f"❌ Pas de données")
        return
    
    stock_data = data_dict[symbol]
    prices = stock_data['Close']
    volumes = stock_data['Volume']
    
    # Code exact de on_download_complete (lignes ~1139-1260)
    cap_range = get_cap_range_for_symbol(symbol)
    print(f"cap_range initial: {cap_range}")
    
    try:
        info = yf.Ticker(symbol).info
        domaine = info.get("sector", "Inconnu")
        from sector_normalizer import normalize_sector
        domaine_raw = domaine
        domaine = normalize_sector(domaine)
    except:
        domaine = "Inconnu"
    print(f"domaine: {domaine}")
    
    # Fallback cap_range (comme dans main_window.py lignes 1169-1198)
    from config import CAP_FALLBACK_ENABLED
    original_cap_range = cap_range
    
    if CAP_FALLBACK_ENABLED and (cap_range == "Unknown" or not cap_range):
        best_params_all = extract_best_parameters()
        for fallback_cap in ["Large", "Mid", "Small", "Mega"]:
            test_key = f"{domaine}_{fallback_cap}"
            if test_key in best_params_all:
                cap_range = fallback_cap
                print(f"Fallback cap_range applied: {cap_range}")
                break
    
    # Extraire seuils globaux optimisés: CODE EXACT de main_window.py (lignes 1210-1241)
    seuil_achat_opt = None
    seuil_vente_opt = None
    best_params_all = extract_best_parameters()
    param_key = None
    
    if cap_range and cap_range != "Unknown":
        test_key = f"{domaine}_{cap_range}"
        if test_key in best_params_all:
            param_key = test_key
    if not param_key and domaine in best_params_all:
        param_key = domaine
    
    print(f"param_key sélectionnée: {param_key}")
    
    if param_key and param_key in best_params_all:
        params = best_params_all[param_key]
        if len(params) > 2 and params[2]:
            globals_th = params[2]
            if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                seuil_achat_opt = float(globals_th[0])
                seuil_vente_opt = float(globals_th[1])
    
    print(f"seuils extraits: achat={seuil_achat_opt}, vente={seuil_vente_opt}")
    
    # Appel get_trading_signal: CODE EXACT lignes 1249-1253
    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
        prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol, cap_range=cap_range,
        seuil_achat=seuil_achat_opt, seuil_vente=seuil_vente_opt
    )
    
    print(f"\nCalcul du score initial:")
    print(f"  Signal: {sig}, Score: {score:.3f}")
    print(f"  Seuil utilisé: {derivatives.get('_seuil_achat_used', 4.2):.2f}")
    
    # STOCKER dans row_info (code exact des lignes 1328-1340)
    row_info = {
        'Symbole': symbol,
        'Signal': sig,
        'Score': score,
        'Prix': last_price,
        'Tendance': 'Hausse' if trend else 'Baisse',
        'RSI': last_rsi,
        'Domaine': domaine,
        'CapRange': cap_range,  # ✅ CIL CI est la valeur stockée
        'Volume moyen': volume_mean,
    }
    
    print(f"\n✅ row_info['CapRange'] STOCKÉE: {row_info['CapRange']}")
    
    # === PHASE 2: affichage du graphique via _build_symbol_figure_with_score ===
    print(f"\n[2] PHASE: _build_symbol_figure_with_score()")
    print("─" * 120)
    
    # Récupérer le precomp du row_info (code ligne ~1476)
    precomp = {
        'signal': row_info.get('Signal'),
        'last_price': row_info.get('Prix'),
        'trend': row_info.get('Tendance'),
        'last_rsi': row_info.get('RSI'),
        'volume_moyen': row_info.get('Volume moyen'),
        'score': row_info.get('Score'),
        'domaine': row_info.get('Domaine'),
        'cap_range': row_info.get('CapRange'),  # ✅ RÉCUPÉRER du row_info
    }
    
    print(f"precomp['cap_range']: {precomp['cap_range']}")
    print(f"precomp['domaine']: {precomp['domaine']}")
    
    # --- Sous-fonction 2a: _compute_score_series (lignes 2655-2675) ---
    print(f"\n  [2a] _compute_score_series():")
    print(f"      Paramètres: domaine={precomp.get('domaine')}, cap_range={precomp.get('cap_range')}")
    
    # Dernière barre du score historique
    last_i = len(prices) - 1
    try:
        _sig, _, _, _, _, score_hist_last, _ = get_trading_signal(
            prices,  # Toute l'historique
            volumes,
            domaine=precomp.get('domaine'),
            cap_range=precomp.get('cap_range'),
            symbol=symbol,
            return_derivatives=True,
        )
        print(f"      Score historique final: {score_hist_last:.3f}")
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        score_hist_last = score
    
    # --- Sous-fonction 2b: _get_global_thresholds_for_symbol (lignes 2681-2705) ---
    print(f"\n  [2b] _get_global_thresholds_for_symbol():")
    print(f"      Paramètres: domaine={precomp.get('domaine')}, cap_range={precomp.get('cap_range')}")
    
    best_params = extract_best_parameters()
    selected_key = None
    
    if precomp.get('cap_range'):
        comp_key = f"{precomp.get('domaine')}_{precomp.get('cap_range')}"
        if comp_key in best_params:
            selected_key = comp_key
    
    if not selected_key and precomp.get('domaine') in best_params:
        selected_key = precomp.get('domaine')
    
    buy_thr, sell_thr = 4.2, -0.5
    if selected_key:
        params = best_params[selected_key]
        if len(params) > 2 and params[2]:
            globals_thresholds = params[2]
            if isinstance(globals_thresholds, (tuple, list)) and len(globals_thresholds) >= 2:
                buy_thr = float(globals_thresholds[0])
                sell_thr = float(globals_thresholds[1])
    
    print(f"      selected_key: {selected_key}")
    print(f"      buy_thr: {buy_thr:.2f}, sell_thr: {sell_thr:.2f}")
    
    # === COMPARAISON FINALE ===
    print(f"\n{'='*120}")
    print("RÉSUMÉ COHÉRENCE")
    print('='*120)
    
    print(f"\n1️⃣  CALCUL INITIAL (on_download_complete):")
    print(f"    cap_range utilisé: {cap_range}")
    print(f"    param_key: {param_key}")
    print(f"    Score: {score:.3f}")
    print(f"    Seuil achat: {derivatives.get('_seuil_achat_used', 4.2):.2f}")
    print(f"    Stocké dans row['CapRange']: {row_info['CapRange']}")
    
    print(f"\n2️⃣  AFFICHAGE GRAPHIQUE (precomp + sous-fonctions):")
    print(f"    cap_range récupéré: {precomp['cap_range']}")
    print(f"    selected_key: {selected_key}")
    print(f"    Score historique final: {score_hist_last:.3f} vs initial: {score:.3f}")
    print(f"    Seuil achat graphique: {buy_thr:.2f}")
    
    # Vérifier cohérence
    cap_range_match = cap_range == precomp['cap_range']
    score_match = abs(score - score_hist_last) < 0.01
    threshold_match = abs(derivatives.get('_seuil_achat_used', 4.2) - buy_thr) < 0.01
    
    print(f"\n✅ cap_range cohérent: {cap_range_match} ({cap_range} vs {precomp['cap_range']})")
    print(f"✅ Score cohérent: {score_match} ({score:.3f} vs {score_hist_last:.3f})")
    print(f"✅ Seuil achat cohérent: {threshold_match} ({derivatives.get('_seuil_achat_used', 4.2):.2f} vs {buy_thr:.2f})")
    
    if cap_range_match and score_match and threshold_match:
        print(f"\n✅ COMPLÈTEMENT COHÉRENT!")
    else:
        print(f"\n❌ INCOHÉRENT - problème détecté:")
        if not cap_range_match:
            print(f"   - cap_range différent!")
        if not score_match:
            print(f"   - scores différents!")
        if not threshold_match:
            print(f"   - seuils différents!")

# Test
for sym in ['KIM', 'AEP']:
    simulate_on_download_complete_and_graph(sym)
