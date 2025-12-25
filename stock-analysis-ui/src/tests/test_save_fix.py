#!/usr/bin/env python
"""Test rapide pour vérifier les corrections de sauvegarde."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from optimisateur_hybride import save_optimization_results

# Test avec 18 params (base seulement)
print("Test 1: Sauvegarde avec 18 params (base seulement)")
coeffs = (1.5, 1.2, 1.8, 1.0, 1.3, 1.4, 1.1, 1.6)
thresholds = (50.0, 0.0, 0.0, 1.2, 25.0, 0.0, 0.5, 4.2, 4.2, -0.5)  # 8 features + 2 globaux
try:
    save_optimization_results(
        "TestSector", coeffs, 10.5, 75.0, 25, thresholds,
        cap_range="Large", extra_params=None, fundamentals_extras=None
    )
    print("✅ Sauvegarde 18 params réussie")
except Exception as e:
    print(f"❌ Erreur: {e}")

# Test avec 24 params (base + price)
print("\nTest 2: Sauvegarde avec 24 params (base + price)")
extra_params = {
    'use_price_slope': 1,
    'use_price_acc': 0,
    'a_price_slope': 0.5,
    'a_price_acc': 0.0,
    'th_price_slope': 0.01,
    'th_price_acc': 0.0,
}
try:
    save_optimization_results(
        "TestSector", coeffs, 11.2, 80.0, 30, thresholds,
        cap_range="Large", extra_params=extra_params, fundamentals_extras=None
    )
    print("✅ Sauvegarde 24 params réussie")
except Exception as e:
    print(f"❌ Erreur: {e}")

# Test avec 34 params (base + price + fundamentals)
print("\nTest 3: Sauvegarde avec 34 params (base + price + fundamentals)")
fundamentals_extras = {
    'use_fundamentals': 1,
    'a_rev_growth': 0.3,
    'a_eps_growth': 0.4,
    'a_roe': 0.2,
    'a_fcf_yield': 0.1,
    'a_de_ratio': 0.15,
    'th_rev_growth': 5.0,
    'th_eps_growth': 10.0,
    'th_roe': 15.0,
    'th_fcf_yield': 2.0,
    'th_de_ratio': 0.5,
}
try:
    save_optimization_results(
        "TestSector", coeffs, 12.8, 85.0, 35, thresholds,
        cap_range="Large", extra_params=extra_params, fundamentals_extras=fundamentals_extras
    )
    print("✅ Sauvegarde 34 params réussie")
except Exception as e:
    print(f"❌ Erreur: {e}")

print("\n" + "="*60)
print("Tests de sauvegarde terminés")
