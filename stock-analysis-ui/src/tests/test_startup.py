#!/usr/bin/env python
# Test simple de démarrage du script

import sys
import os

print("Python version:", sys.version)
print("Current dir:", os.getcwd())
print("Python path:", sys.path[:3])

try:
    print("\n1. Importing optimisateur_hybride...")
    import optimisateur_hybride
    print("   ✅ Import OK")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Checking main code...")
    if hasattr(optimisateur_hybride, 'SYMBOL_MANAGER_AVAILABLE'):
        print(f"   ✅ SYMBOL_MANAGER_AVAILABLE = {optimisateur_hybride.SYMBOL_MANAGER_AVAILABLE}")
    else:
        print("   ❌ SYMBOL_MANAGER_AVAILABLE not found")
    
    if hasattr(optimisateur_hybride, 'split_data_temporal'):
        print("   ✅ split_data_temporal function exists")
    else:
        print("   ❌ split_data_temporal not found")
    
    print("\n3. All checks passed! Script can start.")
except Exception as e:
    print(f"   ❌ Check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
