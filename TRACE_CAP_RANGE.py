#!/usr/bin/env python3
"""
Diagnostic HAUTE FIDÉLITÉ: trace exactement comment cap_range est déterminé et utilisé
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

import yfinance as yf
from qsi import download_stock_data, get_trading_signal, get_cap_range_for_symbol, period
import qsi

def trace_cap_range_flow(symbol):
    """Trace complètement le flux du cap_range"""
    print(f"\n{'='*80}")
    print(f"TRACE CAP_RANGE: {symbol}")
    print('='*80)
    
    # 1. Télécharger les données
    data_dict = download_stock_data([symbol], period=period)
    if symbol not in data_dict:
        print(f"❌ Pas de données")
        return
    
    prices = data_dict[symbol]['Close']
    volumes = data_dict[symbol]['Volume']
    print(f"✅ Données: {len(prices)} bars")
    
    # 2. Déterminer le cap_range selon le code de main_window.py
    cap_range = qsi.get_cap_range_for_symbol(symbol)
    print(f"\n1️⃣ Initial cap_range from qsi.get_cap_range_for_symbol(): {cap_range}")
    
    # 3. Déterminer le secteur
    ticket = yf.Ticker(symbol)
    info = ticket.info
    domaine = info.get("sector", "Inconnu")
    print(f"2️⃣ Secteur (raw): {domaine}")
    
    from sector_normalizer import normalize_sector
    domaine = normalize_sector(domaine)
    print(f"3️⃣ Secteur (normalized): {domaine}")
    
    # 4. Vérifier si le fallback cap_range est appliqué
    print(f"\n4️⃣ Vérification des paramètres optimisés disponibles:")
    best_params = qsi.extract_best_parameters()
    
    # Lister les clés disponibles pour ce secteur
    sector_params = {k: True for k in best_params.keys() if k.startswith(domaine)}
    print(f"   Clés pour '{domaine}': {list(sector_params.keys())}")
    
    # 5. Vérifier quelle clé sera utilisée (mise en place du fallback comme dans main_window.py)
    from config import CAP_FALLBACK_ENABLED
    original_cap_range = cap_range
    
    print(f"\n5️⃣ CAP_FALLBACK_ENABLED: {CAP_FALLBACK_ENABLED}")
    if CAP_FALLBACK_ENABLED and (cap_range == "Unknown" or not cap_range):
        print(f"   ⚠️  cap_range est 'Unknown' ou vide -> Recherche fallback")
        
        for fallback_cap in ["Large", "Mid", "Small", "Mega"]:
            test_key = f"{domaine}_{fallback_cap}"
            if test_key in best_params:
                cap_range = fallback_cap
                print(f"   ✅ Trouvé fallback: {test_key}")
                break
    
    print(f"\n6️⃣ cap_range final utilisé: {cap_range}")
    print(f"   (original: {original_cap_range} -> final: {cap_range})")
    
    # 6. Extraire les seuils globaux optimisés comme dans main_window.py
    seuil_achat_opt = None
    seuil_vente_opt = None
    param_key = None
    
    if cap_range and cap_range != "Unknown":
        test_key = f"{domaine}_{cap_range}"
        if test_key in best_params:
            param_key = test_key
    
    if not param_key and domaine in best_params:
        param_key = domaine
    
    print(f"\n7️⃣ Clé paramètres sélectionnée: {param_key}")
    
    if param_key and param_key in best_params:
        params = best_params[param_key]
        if len(params) > 2 and params[2]:
            globals_th = params[2]
            if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                seuil_achat_opt = float(globals_th[0])
                seuil_vente_opt = float(globals_th[1])
    
    print(f"8️⃣ Seuils optimisés extraits:")
    print(f"   Achat: {seuil_achat_opt}")
    print(f"   Vente: {seuil_vente_opt}")
    
    # 7. Enfin, appeler get_trading_signal comme dans main_window.py
    print(f"\n9️⃣ Appel get_trading_signal() avec:")
    print(f"   domaine={domaine}")
    print(f"   cap_range={cap_range}")
    print(f"   seuil_achat={seuil_achat_opt}")
    print(f"   seuil_vente={seuil_vente_opt}")
    
    sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
        prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol, cap_range=cap_range,
        seuil_achat=seuil_achat_opt, seuil_vente=seuil_vente_opt
    )
    
    print(f"\n🎯 RÉSULTAT:")
    print(f"   Signal: {sig}")
    print(f"   Score: {score:.3f}")
    print(f"   Seuil achat utilisé (dans derivatives): {derivatives.get('_seuil_achat_used', 'N/A')}")
    print(f"   Clé param utilisée: {derivatives.get('_selected_param_key', 'N/A')}")
    
    # 8. Vérifier la cohérence entre params optimisés et seuils réels
    expected_buy = seuil_achat_opt if seuil_achat_opt is not None else 4.2
    actual_buy = derivatives.get('_seuil_achat_used', 4.2)
    
    if abs(expected_buy - actual_buy) > 0.01:
        print(f"\n⚠️ INCOHERENCE: Seuil d'achat attendu={expected_buy}, réel={actual_buy}")
    else:
        print(f"\n✅ COHERENT: Seuil achat={actual_buy:.2f}")


# Test sur KIM et AEP
for symbol in ['KIM', 'AEP']:
    trace_cap_range_flow(symbol)
