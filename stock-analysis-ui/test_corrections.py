#!/usr/bin/env python3
"""
test_corrections.py - Valide les corrections apportées

Teste que:
1. get_cap_range_for_symbol() récupère les cap_range depuis la DB
2. normalize_sector() convertit correctement les noms
3. ParamKeys sont maintenant correctes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cap_range():
    """Test la récupération du cap_range depuis la DB"""
    from qsi import get_cap_range_for_symbol
    
    print("\n" + "="*60)
    print("🧪 TEST 1: Cap_range récupération")
    print("="*60)
    
    test_symbols = ['IMNM', 'OCS', 'ARGX', 'HROW', 'PRCT']
    
    print("\n📊 Résultats:")
    for symbol in test_symbols:
        cap = get_cap_range_for_symbol(symbol)
        status = "✅" if cap != "Unknown" else "❌"
        print(f"  {status} {symbol}: cap_range = {cap}")
    
    print("\n💡 Attendu:")
    print("  - IMNM: Mid ou Small (pas Unknown)")
    print("  - OCS:  Small (pas Unknown)")
    print("  - ARGX: Large")
    print("  - HROW: Small")


def test_sector_normalization():
    """Test la normalisation des secteurs"""
    from sector_normalizer import normalize_sector
    
    print("\n" + "="*60)
    print("🧪 TEST 2: Normalisation secteurs")
    print("="*60)
    
    test_cases = [
        ('Health Care', 'Healthcare'),
        ('Information Technology', 'Technology'),
        ('Financials', 'Financial Services'),
        ('Healthcare', 'Healthcare'),
        ('Unknown', 'Unknown'),
        ('', 'Unknown'),
        (None, 'Unknown'),
    ]
    
    print("\n📊 Résultats:")
    all_pass = True
    for input_val, expected in test_cases:
        result = normalize_sector(input_val)
        match = result == expected
        status = "✅" if match else "❌"
        print(f"  {status} '{input_val}' → '{result}' (attendu: '{expected}')")
        all_pass = all_pass and match
    
    return all_pass


def test_param_keys():
    """Test que les clés de paramètres sont correctes"""
    import sqlite3
    from qsi import extract_best_parameters, get_cap_range_for_symbol
    from sector_normalizer import normalize_sector
    
    print("\n" + "="*60)
    print("🧪 TEST 3: ParamKeys construction")
    print("="*60)
    
    best_params = extract_best_parameters()
    
    print(f"\n📊 Paramètres optimisés disponibles: {len(best_params)} clés")
    print("\nExemples de clés:")
    for key in list(best_params.keys())[:10]:
        print(f"  - {key}")
    
    # Tester avec des symboles réels
    print("\n🔍 Construction ParamKeys pour symboles test:")
    
    db_path = 'symbols.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        for symbol in ['IMNM', 'OCS', 'ARGX']:
            cursor.execute(
                "SELECT sector, cap_range FROM symbols WHERE symbol = ? LIMIT 1",
                (symbol,)
            )
            row = cursor.fetchone()
            
            if row:
                sector = normalize_sector(row['sector'])
                cap = row['cap_range']
                
                # Construction de la clé
                param_key = f"{sector}_{cap}" if cap and cap != 'Unknown' else sector
                found = param_key in best_params
                status = "✅" if found else "⚠️"
                
                print(f"  {status} {symbol}:")
                print(f"     Secteur: {sector}, Cap: {cap}")
                print(f"     ParamKey: {param_key}")
                print(f"     Trouvée: {found}")
            else:
                print(f"  ❌ {symbol}: pas trouvé en DB")
        
        conn.close()
    else:
        print("  ⚠️ symbols.db non trouvée - impossible de tester")


def test_offline_mode():
    """Teste le fallback en mode offline"""
    print("\n" + "="*60)
    print("🧪 TEST 4: Mode offline / Cache")
    print("="*60)
    
    try:
        from qsi import OFFLINE_MODE
        print(f"\n📊 OFFLINE_MODE = {OFFLINE_MODE}")
        
        if OFFLINE_MODE:
            print("  ℹ️  Mode offline activé - utilisation du cache uniquement")
            print("  ⚠️  Assurez-vous que le cache est à jour!")
        else:
            print("  ✅ Mode online - yfinance utilisable")
    except Exception as e:
        print(f"  ⚠️ Erreur vérification mode: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 VALIDATION DES CORRECTIONS")
    print("="*60)
    print("Testes les fixes pour divergence Analyse vs Backtest")
    
    try:
        test_cap_range()
        test_sector_normalization()
        test_param_keys()
        test_offline_mode()
        
        print("\n" + "="*60)
        print("✅ VALIDATION COMPLÈTE")
        print("="*60)
        print("\n📝 Actions à vérifier:")
        print("  1. Tous les symboles test ont un cap_range ≠ Unknown")
        print("  2. Normalisation secteurs retourne les bons noms")
        print("  3. ParamKeys sont trouvées en best_params")
        print("  4. Logs montrent 'trouvé en DB' ou 'normalisé'")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
