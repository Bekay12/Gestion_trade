"""
Gestionnaire de fichiers pour les symboles.
Migration de vos fonctions load_symbols_from_txt, save_symbols_to_txt, modify_symbols_file.
"""
from pathlib import Path
from typing import List, Set
from config.settings import config
from .logger import get_logger

logger = get_logger(__name__)

class SymbolFileManager:
    """
    Gestionnaire des fichiers de symboles.
    Migration de vos fonctions de gestion de fichiers.
    """

    def __init__(self):
        self.symbols_dir = config.paths.symbols_dir

    def load_symbols_from_txt(self, filename: str) -> List[str]:
        """
        Charge les symboles depuis un fichier texte.
        Migration directe de votre fonction load_symbols_from_txt().

        Args:
            filename: Nom du fichier (ex: "popular_symbols.txt").

        Returns:
            Liste des symboles.
        """
        file_path = self.symbols_dir / filename

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                symbols = [line.strip() for line in f if line.strip()]

            logger.info(f"✅ Chargé {len(symbols)} symboles depuis {filename}")
            return symbols

        except FileNotFoundError:
            logger.error(f"❌ Fichier non trouvé: {filename}")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur lecture {filename}: {e}")
            return []

    def save_symbols_to_txt(self, symbols: List[str], filename: str) -> bool:
        """
        Sauvegarde les symboles dans un fichier texte.
        Migration directe de votre fonction save_symbols_to_txt().

        Args:
            symbols: Liste des symboles.
            filename: Nom du fichier.

        Returns:
            True si succès, False sinon.
        """
        file_path = self.symbols_dir / filename

        try:
            # Créer le répertoire si nécessaire
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                for symbol in symbols:
                    f.write(symbol + '\n')

            logger.info(f"✅ Sauvegardé {len(symbols)} symboles dans {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde {filename}: {e}")
            return False

    def modify_symbols_file(self, filename: str, symbols_to_change: List[str], 
                           action: str) -> bool:
        """
        Modifie un fichier de symboles (ajout/suppression).
        Migration directe de votre fonction modify_symbols_file().

        Args:
            filename: Nom du fichier.
            symbols_to_change: Symboles à ajouter/supprimer.
            action: "add" ou "remove".

        Returns:
            True si succès, False sinon.
        """
        file_path = self.symbols_dir / filename

        try:
            # Charger les symboles existants
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_symbols = set(line.strip() for line in f if line.strip())
            else:
                existing_symbols = set()

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
            else:
                logger.error("⚠️ Action invalide. Utilise 'add' ou 'remove'.")
                return False

            # Sauvegarder la nouvelle liste
            with open(file_path, 'w', encoding='utf-8') as f:
                for symbol in sorted(existing_symbols):
                    f.write(symbol + '\n')

            logger.info(f"✅ Fichier mis à jour: {filename}")
            logger.info(f"🔼 Symboles ajoutés: {added}")
            logger.info(f"🔽 Symboles retirés: {removed}")
            logger.info(f"📊 Total actuel: {len(existing_symbols)} symboles")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur modification {filename}: {e}")
            return False

    def get_all_symbol_files(self) -> List[str]:
        """Retourne la liste de tous les fichiers de symboles."""
        if not self.symbols_dir.exists():
            return []

        return [f.name for f in self.symbols_dir.glob("*.txt")]

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Valide et filtre les symboles selon les formats Yahoo Finance.
        Basé sur votre logique dans download_stock_data().
        """
        YAHOO_SUFFIXES = ('.HK', '.DE', '.PA', '.AS', '.SW', '.L', '.TO', '.V', 
                         '.MI', '.AX', '.SI', '.KQ', '.T', '.OL', '.HE', '.ST', 
                         '.CO', '.SA', '.MX', '.TW', '.JO', '.SZ', '.NZ', '.KS',
                         '.PL', '.IR', '.MC', '.VI', '.BK', '.SS', '.SG', '.F', 
                         '.BE', '.CN', '.TA', '-USD', '=F')

        valid_symbols = []
        invalid_symbols = []

        for symbol in symbols:
            if symbol and ('.' not in symbol or symbol.endswith(YAHOO_SUFFIXES)):
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)

        if invalid_symbols:
            logger.warning(f"🚨 Symboles invalides ignorés: {invalid_symbols}")

        return valid_symbols

# Instance globale
symbol_manager = SymbolFileManager()

# Fonctions de compatibilité (pour faciliter la migration)
def load_symbols_from_txt(filename: str) -> List[str]:
    """Fonction de compatibilité."""
    return symbol_manager.load_symbols_from_txt(filename)

def save_symbols_to_txt(symbols: List[str], filename: str) -> bool:
    """Fonction de compatibilité."""
    return symbol_manager.save_symbols_to_txt(symbols, filename)

def modify_symbols_file(filename: str, symbols_to_change: List[str], action: str) -> bool:
    """Fonction de compatibilité."""
    return symbol_manager.modify_symbols_file(filename, symbols_to_change, action)
