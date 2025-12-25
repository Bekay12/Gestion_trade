#!/usr/bin/env python
"""Test de performance du chargement des symboles."""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from symbol_manager import sync_txt_to_sqlite, get_symbols_by_list_type

print("=" * 70)
print("TEST DE PERFORMANCE - Chargement des symboles")
print("=" * 70)

# Test 1: Première exécution (avec cache)
print("\n📊 Test 1: Chargement avec cache existant")
start = time.time()
count = sync_txt_to_sqlite("optimisation_symbols.txt", "optimization")
elapsed = time.time() - start
print(f"⏱️  Temps total: {elapsed:.2f}s")
print(f"📈 {count} symboles traités")

# Test 2: Récupération depuis SQLite
print("\n📊 Test 2: Récupération depuis SQLite")
start = time.time()
symbols = get_symbols_by_list_type("optimization", active_only=True)
elapsed = time.time() - start
print(f"⏱️  Temps total: {elapsed:.3f}s")
print(f"📈 {len(symbols)} symboles récupérés")

print("\n" + "=" * 70)
print("✅ Tests terminés")
