#!/usr/bin/env python
"""Teste la cohérence des trades avec les features activées."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from optimisateur_hybride import optimize_sector_coefficients_hybrid

print("=" * 80)
print("TEST DE COHÉRENCE - Trade Count avec Features")
print("=" * 80)

# Symboles de test
test_symbols = ["AAPL", "MSFT", "GOOGL"]

print("\n1️⃣ Test avec 34 paramètres (price + fundamentals activés)")
print("-" * 80)
result = optimize_sector_coefficients_hybrid(
    test_symbols, "TestSector_Tech",
    period='6mo',
    strategy='lhs',
    montant=50,
    transaction_cost=0.02,
    budget_evaluations=30,
    precision=1,
    cap_range="Large",
    use_price_features=True,
    use_fundamentals_features=True
)
coeffs, gain, success, thresholds, summary = result
if summary:
    print(f"✅ Optimisation terminée")
    print(f"   Gain: {gain:.2f}")
    print(f"   Success rate: {success:.1f}%")
    print(f"   Trades count: {summary.get('trades_new', 'N/A')}")
    if summary.get('trades_new', 0) == 0:
        print("   ⚠️  WARNING: Aucun trade généré!")
    else:
        print("   ✅ Trades générés correctement")

print("\n" + "=" * 80)
print("✅ Test terminé")
print("=" * 80)
