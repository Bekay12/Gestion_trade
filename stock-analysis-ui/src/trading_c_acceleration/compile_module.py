#!/usr/bin/env python
"""Script pour compiler le module C trading_c avec les nouvelles features"""

import subprocess
import sys
import os

def main():
    # Se placer dans le bon répertoire
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print("🔨 Compiling C module with extended features support...")
    print("   (price_slope, price_acc, fundamentals)")
    print()
    
    # Compiler
    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        capture_output=True,
        text=True
    )
    
    print("=== STDOUT ===")
    print(result.stdout)
    
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr)
    
    print(f"\n✅ Return code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ Compilation successful!")
        # Tester le module
        try:
            import importlib
            if 'trading_c' in sys.modules:
                del sys.modules['trading_c']
            import trading_c
            print(f"✅ Module test: {trading_c.test_module()}")
        except Exception as e:
            print(f"⚠️ Module test failed: {e}")
    else:
        print("❌ Compilation failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
