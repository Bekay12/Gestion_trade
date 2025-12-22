#!/usr/bin/env python3
"""Initialize SQLite symbol management system."""
import sys

from symbol_manager import (
    init_symbols_table, sync_txt_to_sqlite, get_symbol_count,
    get_all_sectors, get_all_cap_ranges
)

print("\n" + "="*80)
print("INITIALISATION DU SYSTEME DE GESTION DES SYMBOLES")
print("="*80)

print("\n[1/3] Initialisation de la table SQLite...")
init_symbols_table()

print("\n[2/3] Synchronisation des fichiers txt -> SQLite...")
print("     - popular_symbols.txt...", end=" ", flush=True)
sync_txt_to_sqlite('popular_symbols.txt', 'popular')

print("     - mes_symbols.txt...", end=" ", flush=True)
sync_txt_to_sqlite('mes_symbols.txt', 'personal')

print("\n[3/3] Verification...")
total = get_symbol_count()
popular = get_symbol_count('popular')
personal = get_symbol_count('personal')
sectors = get_all_sectors()
cap_ranges = get_all_cap_ranges()

print(f"\n{'-'*80}")
print(f"STATISTIQUES:")
print(f"{'-'*80}")
print(f"  Total symboles:            {total}")
print(f"  - Symboles populaires:     {popular}")
print(f"  - Symboles personnels:     {personal}")
print(f"  Secteurs disponibles:      {len(sectors)}")
print(f"  Gammes de capitalisation:  {len(cap_ranges)}")
print(f"{'-'*80}")

if total > 0:
    print("\n✅ SYSTEME INITIALISE AVEC SUCCES!")
    print("\nVous pouvez maintenant utiliser symbol_manager pour:")
    print("  - get_symbols_by_list_type('popular') -> Liste populaires")
    print("  - get_symbols_by_sector('Technology') -> Par secteur")
    print("  - get_symbols_by_cap_range('Large') -> Par capitalisation")
    print("  - get_symbols_by_sector_and_cap('Technology', 'Large') -> Combo")
else:
    print("\n⚠️ ATTENTION: Aucun symbole trouve!")

print("="*80)
