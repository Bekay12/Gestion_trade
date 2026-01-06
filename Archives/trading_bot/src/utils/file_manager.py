"""
Gestionnaire de fichiers de symboles.
Migration de vos fonctions de gestion de fichiers.
"""
from pathlib import Path
from typing import List
from config.settings import config
from .logger import get_logger

logger = get_logger(__name__)

class SymbolFileManager:
    """
    Gestionnaire pour les fichiers de symboles.
    Migration de vos fonctions load_symbols_from_txt, etc.
    """
    
    def __init__(self):
        self.symbols_dir = config.paths.symbols_dir
    
    def load_symbols_from_txt(self, filename: str) -> List[str]:
        """
        Charge les symboles depuis un fichier texte.
        Migration de votre fonction load_symbols_from_txt().
        
        Args:
            filename: Nom du fichier (ex: "mes_symbols.txt").
            
        Returns:
            Liste des symboles.
        """
        file_path = self.symbols_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"📋 Chargé {len(symbols)} symboles depuis {filename}")
                return symbols
        except FileNotFoundError:
            logger.warning(f"⚠️ Fichier {filename} non trouvé dans {self.symbols_dir}")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur lecture {filename}: {e}")
            return []
    
    def save_symbols_to_txt(self, symbols: List[str], filename: str) -> bool:
        """
        Sauvegarde les symboles dans un fichier texte.
        Migration de votre fonction save_symbols_to_txt().
        
        Args:
            symbols: Liste des symboles.
            filename: Nom du fichier.
            
        Returns:
            True si succès.
        """
        file_path = self.symbols_dir / filename
        
        try:
            # Créer le dossier si nécessaire
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for symbol in symbols:
                    f.write(f"{symbol}\\n")
            
            logger.info(f"💾 Sauvegardé {len(symbols)} symboles dans {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde {filename}: {e}")
            return False
    
    def modify_symbols_file(self, filename: str, symbols_to_change: List[str], action: str) -> bool:
        """
        Modifie un fichier de symboles (ajouter/supprimer).
        Migration de votre fonction modify_symbols_file().
        
        Args:
            filename: Nom du fichier.
            symbols_to_change: Symboles à modifier.
            action: "add" ou "remove".
            
        Returns:
            True si succès.
        """
        if action not in ["add", "remove"]:
            logger.error("⚠️ Action invalide. Utilisez 'add' ou 'remove'.")
            return False
        
        existing_symbols = set(self.load_symbols_from_txt(filename))
        initial_count = len(existing_symbols)
        
        added, removed = 0, 0
        
        if action == "add":
            for symbol in symbols_to_change:
                if symbol not in existing_symbols:
                    existing_symbols.add(symbol)
                    added += 1
        elif action == "remove":
            for symbol in symbols_to_change:
                if symbol in existing_symbols:
                    existing_symbols.remove(symbol)
                    removed += 1
        
        # Sauvegarder la nouvelle liste
        success = self.save_symbols_to_txt(sorted(existing_symbols), filename)
        
        if success:
            logger.info(f"✅ Fichier mis à jour: {filename}")
            logger.info(f"🔼 Symboles ajoutés: {added}")
            logger.info(f"🔽 Symboles retirés: {removed}")
            logger.info(f"📊 Total actuel: {len(existing_symbols)} symboles")
        
        return success

# Instance globale
symbol_file_manager = SymbolFileManager()

# Fonctions de compatibilité
def load_symbols_from_txt(filename: str) -> List[str]:
    """Fonction de compatibilité."""
    return symbol_file_manager.load_symbols_from_txt(filename)

def save_symbols_to_txt(symbols: List[str], filename: str) -> bool:
    """Fonction de compatibilité."""
    return symbol_file_manager.save_symbols_to_txt(symbols, filename)

def modify_symbols_file(filename: str, symbols_to_change: List[str], action: str) -> bool:
    """Fonction de compatibilité."""
    return symbol_file_manager.modify_symbols_file(filename, symbols_to_change, action)