#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mesure le temps de démarrage de optimisateur_hybride avec breakdown des étapes."""

import sys
import os
import io
import time
from pathlib import Path

# Forcer UTF-8 sur Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Timestamps
timestamps = {}

def checkpoint(label):
    """Enregistre le temps pour une étape."""
    timestamps[label] = time.time()
    print(f"[{label}] {len(timestamps)} etapes")

# Démarrer
checkpoint("START")

# Import des modules
print("\nPackages...")
checkpoint("IMPORTS_START")

try:
    import optimisateur_hybride as opt
    checkpoint("IMPORTS_DONE")
    print(f"   OK modules importes")
except Exception as e:
    print(f"   ERREUR imports: {e}")
    sys.exit(1)

# Chargement des symboles
print("\nSymboles...")
checkpoint("SYMBOLS_START")

try:
    # Initialiser symbol_manager
    if opt.SYMBOL_MANAGER_AVAILABLE:
        opt.init_symbols_table()
        try:
            opt.sync_txt_to_sqlite("optimisation_symbols.txt", "optimization")
        except:
            pass
        
        symbols = opt.get_symbols_by_list_type("optimization", active_only=True)
        checkpoint("SYMBOLS_LOADED")
        print(f"   OK {len(symbols)} symboles charges")
        
        # Récupérer secteurs et cap_ranges
        checkpoint("SECTORS_START")
        sectors = opt.get_all_sectors(list_type="optimization")
        cap_ranges = opt.get_all_cap_ranges(list_type="optimization")
        checkpoint("SECTORS_DONE")
        print(f"   OK {len(sectors)} secteurs, {len(cap_ranges)} cap_ranges")
        
        # Construction sector_cap_ranges
        checkpoint("BUILD_SECTOR_CAP_START")
        sector_cap_ranges = {}
        for sector in sectors:
            sector_cap_ranges[sector] = {}
            for cap_range in cap_ranges:
                syms = opt.get_symbols_by_sector_and_cap(sector, cap_range, "optimization", active_only=True)
                if syms:
                    sector_cap_ranges[sector][cap_range] = syms
        checkpoint("BUILD_SECTOR_CAP_DONE")
        total_combos = sum(1 for s in sector_cap_ranges.values() for cap, syms in s.items() if syms)
        print(f"   OK {total_combos} combinaisons secteur x cap")
        
        # Nettoyage des groupes
        checkpoint("CLEAN_GROUPS_START")
        sector_cap_ranges, ignored_log = opt.clean_sector_cap_groups(sector_cap_ranges, opt.MIN_SYMBOLS_PER_GROUP)
        checkpoint("CLEAN_GROUPS_DONE")
        print(f"   OK Nettoyage termine ({len(ignored_log)} ignores)")
        
    else:
        print("   Attention: Symbol_manager pas disponible")
        checkpoint("SYMBOLS_SKIPPED")

except Exception as e:
    print(f"   ERREUR symboles: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Afficher le résumé des temps
print("\n" + "="*80)
print("Temps de demarrage - BREAKDOWN")
print("="*80)

times_list = list(timestamps.items())
for i in range(len(times_list)):
    label = times_list[i][0]
    t = times_list[i][1]
    
    # Temps depuis le début
    elapsed_from_start = t - times_list[0][1]
    
    # Temps depuis l'étape précédente
    if i > 0:
        prev_t = times_list[i-1][1]
        elapsed_from_prev = t - prev_t
    else:
        elapsed_from_prev = 0
    
    print(f"{label:30s} | Depuis debut: {elapsed_from_start:7.2f}s | Depuis prec: {elapsed_from_prev:7.2f}s")

print("="*80)
print(f"\nDEMARRAGE COMPLET: {elapsed_from_start:.2f}s")
print("\nOK Script pret (avant download_stock_data)")
