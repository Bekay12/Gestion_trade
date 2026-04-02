#!/usr/bin/env python3
"""
Diagnostic: Vérify la cohérence entre:
1. Score initial calculé avec cap_range
2. Seuils utilisé affichage (du tableau)
3. Historique recalculé avec les mêmes paramètres
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

from qsi import download_stock_data, get_trading_signal, extract_best_parameters, period, get_cap_range_for_symbol
import yfinance as yf
from sector_normalizer import normalize_sector
import pandas as pd

def full_diagnosis(symbol, show_history=False):
    """Simule exactement le flux main_window.py"""
    print(f"\n{'='*100}")
    print(f"DIAGNOSTIC COMPLET: {symbol}")
    print('='*100)
    
    # ===== STEP 1: Télécharger les données (comme dans AnalysisThread) =====
    data_dict = download_stock_data([symbol], period=period)
    if symbol not in data_dict:
        print(f"❌ Pas de données")
        return
    
    prices = data_dict[symbol]['Close']
    volumes = data_dict[symbol]['Volume']
    print(f"\n✅ Données: {len(prices)} bars, {prices.index[0]} à {prices.index[-1]}")
    
    # ===== STEP 2: Déterminer cap_range (comme dans on_download_complete) =====
    cap_range = get_cap_range_for_symbol(symbol)
    print(f"\n1. cap_range initial: {cap_range}")
    
    # ===== STEP 3: Déterminer secteur (comme dans on_download_complete) =====
    try:
        info = yf.Ticker(symbol).info
        domaine = info.get("sector", "Inconnu")
        domaine = normalize_sector(domaine)
    except Exception:
        domaine = "Inconnu"
    print(f"2. domaine: {domaine}")
    
    # ===== STEP 4: Extraire seuils optim (comme dans on_download_complete) =====
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
    
    if param_key and param_key in best_params_all:
        params = best_params_all[param_key]
        if len(params) > 2 and params[2]:
            globals_th = params[2]
            if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                seuil_achat_opt = float(globals_th[0])
                seuil_vente_opt = float(globals_th[1])
    
    print(f"3. param_key: {param_key}")
    print(f"4. seuils optimisés: achat={seuil_achat_opt}, vente={seuil_vente_opt}")
    
    # ===== STEP 5: Calculer le SCORE INITIAL (comme dans on_download_complete) =====
    print(f"\n{'─'*100}")
    print("CALCUL SCORE INITIAL (comme dans on_download_complete)")
    print('─'*100)
    
    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
        prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol, cap_range=cap_range,
        seuil_achat=seuil_achat_opt, seuil_vente=seuil_vente_opt
    )
    
    print(f"Signal: {sig}, Score: {score:.3f}, RSI: {last_rsi:.2f}, Prix: {last_price:.2f}")
    seuil_achat_used = derivatives.get('_seuil_achat_used', 4.2)
    seuil_vente_used = derivatives.get('_seuil_vente_used', -0.5)
    print(f"Seuils utilisés: achat={seuil_achat_used:.2f}, vente={seuil_vente_used:.2f}")
    
    # Simuler l'ajout au tableau row_info
    row_info = {
        'Symbole': symbol,
        'Signal': sig,
        'Score': score,
        'Prix': last_price,
        'RSI': last_rsi,
        'Domaine': domaine,
        'CapRange': cap_range,  # ✅ Ceci est stocké dans le tableau
    }
    
    # ===== STEP 6: Calculer l'HISTORIQUE (comme dans _build_symbol_figure_with_score) =====
    print(f"\n{'─'*100}")
    print("CALCUL HISTORIQUE (comme dans _build_symbol_figure_with_score)")
    print('─'*100)
    
    # Récupérer les seuils comme dans _get_global_thresholds_for_symbol
    best_params = extract_best_parameters()
    selected_key = None
    
    # On utilise le cap_range ET domaine du row_info (comme on l'a stocké)
    retrieved_cap_range = row_info.get('CapRange')
    retrieved_domaine = row_info.get('Domaine')
    
    if retrieved_cap_range:
        comp_key = f"{retrieved_domaine}_{retrieved_cap_range}"
        if comp_key in best_params:
            selected_key = comp_key
    
    if not selected_key and retrieved_domaine in best_params:
        selected_key = retrieved_domaine
    
    buy_thr, sell_thr = 4.2, -0.5
    if selected_key:
        params = best_params[selected_key]
        if len(params) > 2 and params[2]:
            globals_thresholds = params[2]
            if isinstance(globals_thresholds, (tuple, list)) and len(globals_thresholds) >= 2:
                buy_thr = float(globals_thresholds[0])
                sell_thr = float(globals_thresholds[1])
    
    print(f"retrieved_cap_range: {retrieved_cap_range}")
    print(f"retrieved_domaine: {retrieved_domaine}")
    print(f"selected_key: {selected_key}")
    print(f"Seuils pour historique: achat={buy_thr:.2f}, vente={sell_thr:.2f}")
    
    # Calculer scores historiques (simplifié)
    score_history = []
    start_idx = 50
    for i in range(start_idx, len(prices), 5):  # Step=5 pour accélérer
        try:
            _sig, _last_price, _trend, _last_rsi, _vol_mean, score_h, _ = get_trading_signal(
                prices.iloc[:i + 1],
                volumes.iloc[:i + 1],
                domaine=retrieved_domaine,
                cap_range=retrieved_cap_range,
                symbol=symbol,
                return_derivatives=True,
            )
            score_history.append({
                'date': prices.index[i],
                'score': score_h,
                'above_buy': score_h >= buy_thr,
                'below_sell': score_h <= sell_thr,
            })
        except Exception:
            pass
    
    print(f"\nHistorique calculé: {len(score_history)} points")
    if score_history:
        print(f"Scores min: {min(s['score'] for s in score_history):.3f}, max: {max(s['score'] for s in score_history):.3f}")
        print(f"Seuil achat ({buy_thr:.2f}): {sum(1 for s in score_history if s['above_buy'])} points au-dessus")
        print(f"Seuil vente ({sell_thr:.2f}): {sum(1 for s in score_history if s['below_sell'])} points au-dessous")
    
    # ===== STEP 7: VÉRIFICATION COHÉRENCE =====
    print(f"\n{'='*100}")
    print("VÉRIFICATION COHÉRENCE")
    print('='*100)
    
    is_match_score = abs(score - score_history[-1]['score']) < 0.01 if score_history else False
    is_match_seuil_achat = abs(seuil_achat_used - buy_thr) < 0.01
    
    print(f"\n✅ Score actuel ({score:.3f}) vs dernier historique ({score_history[-1]['score']:.3f} if exists): {'MATCH' if is_match_score else 'DIVERGENT'}")
    print(f"✅ Seuil achat utilisé ({seuil_achat_used:.2f}) vs seuil historique ({buy_thr:.2f}): {'MATCH' if is_match_seuil_achat else 'DIVERGENT'}")
    
    if is_match_score and is_match_seuil_achat:
        print(f"\n🎉 COHÉRENT: Tous les seuils et scores correspondent!")
    else:
        print(f"\n⚠️ INCOHÉRENT: Divergence détectée!")
        if not is_match_score:
            print(f"   - Score divergence: {score:.3f} vs {score_history[-1]['score']:.3f if score_history else 'N/A'}")
        if not is_match_seuil_achat:
            print(f"   - Seuil divergence: {seuil_achat_used:.2f} vs {buy_thr:.2f}")
    
    return {
        'symbol': symbol,
        'score_initial': score,
        'seuil_achat_initial': seuil_achat_used,
        'score_final_history': score_history[-1]['score'] if score_history else None,
        'seuil_achat_history': buy_thr,
        'coherent': is_match_score and is_match_seuil_achat,
    }


# Test
for symbol in ['KIM', 'AEP']:
    full_diagnosis(symbol)
