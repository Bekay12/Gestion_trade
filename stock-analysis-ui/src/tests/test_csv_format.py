#!/usr/bin/env python
"""Test script pour vérifier que le format CSV et l'extraction fonctionnent correctement"""

import sys
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from qsi import extract_best_parameters

def test_extract_best_parameters():
    print("🧪 Test: extract_best_parameters avec le nouveau format CSV")
    print("=" * 60)
    
    try:
        params = extract_best_parameters('signaux/optimization_hist_4stp.csv')
        
        if not params:
            print("❌ ERREUR: Aucun paramètre extrait")
            return False
        
        print(f"✅ {len(params)} secteurs trouvés\n")
        
        # Vérifier chaque secteur
        for sector, data in list(params.items())[:5]:  # Afficher les 5 premiers
            if len(data) != 4:
                print(f"❌ ERREUR {sector}: tuple expected 4 elements, got {len(data)}")
                return False
            
            coeffs, thresholds, globals_thresholds, gain = data
            
            # Vérifier les tailles
            if len(coeffs) != 8:
                print(f"❌ ERREUR {sector}: coeffs expected 8, got {len(coeffs)}")
                return False
            
            if len(thresholds) != 8:
                print(f"❌ ERREUR {sector}: thresholds expected 8, got {len(thresholds)}")
                return False
            
            if len(globals_thresholds) != 2:
                print(f"❌ ERREUR {sector}: globals_thresholds expected 2, got {len(globals_thresholds)}")
                return False
            
            print(f"✅ {sector}:")
            print(f"   coeffs (a1-a8):      {coeffs}")
            print(f"   thresholds (th1-th8):{thresholds}")
            print(f"   globals (buy, sell): {globals_thresholds}")
            print(f"   gain_moy:            {gain:.4f}")
            print()
        
        print("=" * 60)
        print("✅ TOUS LES TESTS PASSENT!")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_extract_best_parameters()
    sys.exit(0 if success else 1)
