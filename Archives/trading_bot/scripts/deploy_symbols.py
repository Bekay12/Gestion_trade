#!/usr/bin/env python3
"""
Script pour déployer vos fichiers de symboles dans la nouvelle structure.
"""
import shutil
from pathlib import Path

def deploy_symbol_files():
    """Copie les fichiers de symboles vers le nouveau répertoire."""
    source_files = [
        "popular_symbols.txt",
        "test_symbols.txt", 
        "mes_symbols.txt",
        "optimisation_symbols.txt"
    ]
    
    target_dir = Path("trading_bot/data/symbols")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("📂 Déploiement des fichiers de symboles...")
    
    for file_name in source_files:
        source = Path(file_name)
        target = target_dir / file_name
        
        if source.exists():
            shutil.copy2(source, target)
            print(f"✅ Copié: {file_name} -> {target}")
        else:
            print(f"⚠️ Non trouvé: {file_name}")
            # Créer un fichier exemple
            if "test" in file_name:
                target.write_text("AAPL\\nMSFT\\nGOOGL\\nTSLA\\n")
                print(f"📝 Créé fichier exemple: {target}")
    
    print("✅ Déploiement terminé")

if __name__ == "__main__":
    deploy_symbol_files()