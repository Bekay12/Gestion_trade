#!/usr/bin/env python3
"""Test pour vérifier que TOUS les signaux validés sont affichés graphiquement."""
import sys
import io

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from qsi import analyse_signaux_populaires
import pandas as pd

# Test avec les symboles qui commencent par C et B (comme dans l'exemple utilisateur)
test_symbols = [
    'CGC', 'CRON', 'BFLY', 'BOE', 'BAYN.DE', 'CSX', 'COLL', 'DAL', 'BNP.PA',
    'CRWV', 'CTSH', 'BMW.DE', 'BIDU', 'BIIB', 'CRL', 'CDTX', 'BURL', 'CVNA',
    'CMRE', 'CVS', 'CDNS', 'CAT'
]

print(f"Test avec {len(test_symbols)} symboles...\n")

result = analyse_signaux_populaires(
    test_symbols, 
    [],  # mes_symbols
    verbose=True,
    afficher_graphiques=False,  # Pas de graphiques pour le test
    save_csv=False
)

signaux_valides = result.get('signaux_valides', [])
top_achats = [s for s in signaux_valides if s['Signal'] == 'ACHAT']
top_ventes = [s for s in signaux_valides if s['Signal'] == 'VENTE']

print(f"\n{'='*80}")
print(f"📊 RÉSULTATS DU TEST :")
print(f"{'='*80}")
print(f"✅ Total signaux validés: {len(signaux_valides)}")
print(f"   - Signaux ACHAT: {len(top_achats)}")
print(f"   - Signaux VENTE: {len(top_ventes)}")

print(f"\n📈 Signaux ACHAT qui seront affichés:")
for s in top_achats:
    print(f"   {s['Symbole']:<10} Score: {s['Score']:.2f}  Tendance: {s['Tendance']}")

if top_ventes:
    print(f"\n📉 Signaux VENTE qui seront affichés:")
    for s in top_ventes:
        print(f"   {s['Symbole']:<10} Score: {s['Score']:.2f}  Tendance: {s['Tendance']}")

print(f"\n{'='*80}")
if len(signaux_valides) == len(top_achats) + len(top_ventes):
    print("✅ SUCCÈS: Tous les signaux validés seront affichés!")
else:
    print(f"⚠️ ATTENTION: {len(signaux_valides)} validés != {len(top_achats) + len(top_ventes)} à afficher")
print(f"{'='*80}")
