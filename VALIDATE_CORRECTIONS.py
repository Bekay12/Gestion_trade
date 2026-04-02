#!/usr/bin/env python3
"""
✅ VALIDATION: Vérifier que les corrections apportées fonctionnent correctement
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

from qsi import download_stock_data, get_trading_signal, get_cap_range_for_symbol, extract_best_parameters, period
import yfinance as yf
from sector_normalizer import normalize_sector

def validate_corrections():
    """Tester que les corrections fonctionnent"""
    
    print("\n" + "="*100)
    print("VALIDATION: Les corrections apportées")
    print("="*100)
    
    # Test symboles
    symbols_to_test = ['KIM', 'AEP']
    
    all_pass = True
    
    for symbol in symbols_to_test:
        print(f"\n[TEST] {symbol}")
        print("-" * 100)
        
        # 1. Télécharger
        data_dict = download_stock_data([symbol], period=period)
        if symbol not in data_dict:
            print(f"❌ Pas de données")
            all_pass = False
            continue
        
        prices = data_dict[symbol]['Close']
        volumes = data_dict[symbol]['Volume']
        
        # 2. Obtenir cap_range et domaine
        cap_range = get_cap_range_for_symbol(symbol)
        info = yf.Ticker(symbol).info
        domaine = normalize_sector(info.get("sector", "Inconnu"))
        
        # 3. Appel get_trading_signal avec cap_range
        sig, _, _, _, _, score, derivatives = get_trading_signal(
            prices, volumes, domaine=domaine, cap_range=cap_range, symbol=symbol,
            return_derivatives=True
        )
        
        # 4. Valider les nouvelles clés dans derivatives
        checks = {
            '_seuil_achat_used': derivatives.get('_seuil_achat_used') is not None,
            '_seuil_vente_used': derivatives.get('_seuil_vente_used') is not None,
            '_selected_param_key': derivatives.get('_selected_param_key') is not None,
            '_cap_range_used': derivatives.get('_cap_range_used') is not None,  # ✅ NEW
            '_domaine_used': derivatives.get('_domaine_used') is not None,      # ✅ NEW
            '_score_value': derivatives.get('_score_value') is not None,        # ✅ NEW
        }
        
        print(f"Score: {score:.3f}")
        print(f"Cap_range utilisé: {derivatives.get('_cap_range_used', 'N/A')}")
        print(f"Domaine utilisé: {derivatives.get('_domaine_used', 'N/A')}")
        print(f"Seuil achat: {derivatives.get('_seuil_achat_used', 'N/A'):.2f}")
        
        print("\n✅ Vérifications:")
        for check_name, check_result in checks.items():
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        # 5. Vérifier cohérence cap_range fournisseur vs utilisé
        cap_consistency = cap_range == derivatives.get('_cap_range_used')
        status = "✅" if cap_consistency else "❌"
        print(f"\n{status} Cohérence cap_range: {cap_range} == {derivatives.get('_cap_range_used')}")
        if not cap_consistency:
            all_pass = False
    
    print(f"\n{'='*100}")
    if all_pass:
        print("✅ TOUTES LES VALIDATIONS PASSENT!")
    else:
        print("❌ CERTAINES VALIDATIONS ONT ÉCHOUÉ - Vérifier les logs ci-dessus")
    print('='*100)
    
    return all_pass


if __name__ == '__main__':
    success = validate_corrections()
    sys.exit(0 if success else 1)
