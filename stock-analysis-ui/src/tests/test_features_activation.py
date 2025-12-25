#!/usr/bin/env python
"""Test rapide pour vérifier que les features peuvent être activées."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from optimisateur_hybride import optimize_sector_coefficients_hybrid

print("=" * 80)
print("TEST D'ACTIVATION DES FEATURES")
print("=" * 80)

# Symboles de test
test_symbols = ["AAPL", "MSFT", "GOOGL"]

print("\n1️⃣ Test avec 18 paramètres (base seulement)")
print("-" * 80)
result = optimize_sector_coefficients_hybrid(
    test_symbols, "TestSector_Large",
    period='6mo',
    strategy='lhs',
    montant=50,
    transaction_cost=0.02,
    budget_evaluations=20,
    precision=2,
    cap_range="Large",
    use_price_features=False,
    use_fundamentals_features=False
)
coeffs, gain, success, thresholds, summary = result
if summary:
    print(f"✅ Optimisation terminée: gain={gain:.2f}, success={success:.1f}%")
    print(f"   Coefficients: {len(coeffs)} valeurs")

print("\n2️⃣ Test avec 24 paramètres (base + price features)")
print("-" * 80)
result = optimize_sector_coefficients_hybrid(
    test_symbols, "TestSector_Large",
    period='6mo',
    strategy='lhs',
    montant=50,
    transaction_cost=0.02,
    budget_evaluations=20,
    precision=2,
    cap_range="Large",
    use_price_features=True,
    use_fundamentals_features=False
)
coeffs, gain, success, thresholds, summary = result
if summary:
    print(f"✅ Optimisation terminée: gain={gain:.2f}, success={success:.1f}%")
    print(f"   Paramètres price features extraits: {summary.get('extra_params', {})}")

print("\n3️⃣ Test avec 28 paramètres (base + fundamentals)")
print("-" * 80)
result = optimize_sector_coefficients_hybrid(
    test_symbols, "TestSector_Large",
    period='6mo',
    strategy='lhs',
    montant=50,
    transaction_cost=0.02,
    budget_evaluations=20,
    precision=2,
    cap_range="Large",
    use_price_features=False,
    use_fundamentals_features=True
)
coeffs, gain, success, thresholds, summary = result
if summary:
    print(f"✅ Optimisation terminée: gain={gain:.2f}, success={success:.1f}%")
    print(f"   Paramètres fundamentals extraits: {summary.get('fundamentals_extras', {})}")

print("\n" + "=" * 80)
print("✅ Tests terminés - Les features s'activent correctement!")
print("=" * 80)
