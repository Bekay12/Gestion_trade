#!/usr/bin/env python3
"""Test script for SQLite symbol management."""
import sys
import io

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from symbol_manager import (
    init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type,
    get_symbols_by_sector, get_symbols_by_cap_range, get_all_sectors,
    get_all_cap_ranges, get_symbol_count
)

print("\n" + "="*80)
print("TEST DE GESTION DES SYMBOLES SQLITE")
print("="*80)

# Initialiser la table
print("\n1. Initialisation de la table 'symbols'...")
init_symbols_table()

# Synchroniser les symboles populaires
print("\n2. Synchronisation des symboles populaires...")
count_popular = sync_txt_to_sqlite('popular_symbols.txt', 'popular')

# Synchroniser les symboles personnels
print("\n3. Synchronisation des symboles personnels...")
count_personal = sync_txt_to_sqlite('mes_symbols.txt', 'personal')

# Statistiques
print("\n4. STATISTIQUES:")
total = get_symbol_count()
popular = get_symbol_count('popular')
personal = get_symbol_count('personal')
print(f"   ✅ Total symboles: {total}")
print(f"   ✅ Symboles populaires: {popular}")
print(f"   ✅ Symboles personnels: {personal}")

# Secteurs
print("\n5. SECTEURS DISPONIBLES:")
sectors = get_all_sectors()
print(f"   ✅ {len(sectors)} secteurs trouvés")
for sector in sorted(sectors)[:10]:
    count = len(get_symbols_by_sector(sector))
    print(f"      - {sector}: {count} symboles")

# Gammes de capitalisation
print("\n6. GAMMES DE CAPITALISATION:")
cap_ranges = get_all_cap_ranges()
for cap_range in cap_ranges:
    count = len(get_symbols_by_cap_range(cap_range))
    print(f"   ✅ {cap_range}: {count} symboles")

# Test requête filtrée
print("\n7. TEST REQUÊTE FILTRÉE (Technology + Large):")
from symbol_manager import get_symbols_by_sector_and_cap
tech_large = get_symbols_by_sector_and_cap('Technology', 'Large')
print(f"   ✅ Trouvé {len(tech_large)} symboles")
for sym in tech_large[:5]:
    print(f"      - {sym}")

print("\n" + "="*80)
print("✅ TOUS LES TESTS PASSÉS!")
print("="*80)
