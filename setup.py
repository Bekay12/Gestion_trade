#!/usr/bin/env python
"""
Setup pour compiler le module C de trading_c_acceleration
"""
import os
import sys
import subprocess
from pathlib import Path

def build_trading_c():
    """Compile le module C trading_c"""
    trading_c_dir = Path(__file__).parent / "stock-analysis-ui" / "src" / "trading_c_acceleration"
    
    if not trading_c_dir.exists():
        print(f"❌ Erreur: Dossier {trading_c_dir} introuvable")
        return False
    
    print(f"📦 Compilation du module C dans {trading_c_dir}...")
    
    # Sauvegarder le répertoire courant
    original_cwd = os.getcwd()
    
    try:
        # Changer vers le répertoire trading_c_acceleration
        os.chdir(trading_c_dir)
        
        # Exécuter le setup.py local
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        print(f"🔨 Commande: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("✅ Compilation réussie !")
            # Vérifier la présence du fichier compilé
            for ext in [".so", ".pyd"]:
                for f in trading_c_dir.glob(f"trading_c*{ext}"):
                    print(f"   ✓ Module compilé: {f.name}")
            return True
        else:
            print(f"❌ Compilation échouée (exit code: {result.returncode})")
            return False
    
    except Exception as e:
        print(f"❌ Erreur lors de la compilation: {e}")
        return False
    
    finally:
        # Restaurer le répertoire courant
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = build_trading_c()
    sys.exit(0 if success else 1)
