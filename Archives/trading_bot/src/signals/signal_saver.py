"""
Gestionnaire de sauvegarde des signaux.
Migration de votre fonction save_to_evolutive_csv().
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SignalSaver:
    """
    Gestionnaire de sauvegarde des signaux.
    Migration complète de votre fonction save_to_evolutive_csv().
    """
    
    def __init__(self):
        self.signals_dir = config.paths.results_dir / "signaux"
        self.signals_dir.mkdir(parents=True, exist_ok=True)
    
    def save_to_evolutive_csv(self, signals: List[Dict], filename: str = "signaux_trading.csv") -> bool:
        """
        Sauvegarde les signaux dans un CSV évolutif.
        Migration complète de votre fonction save_to_evolutive_csv().
        
        Args:
            signals: Liste des signaux à sauvegarder.
            filename: Nom du fichier CSV.
            
        Returns:
            True si succès, False sinon.
        """
        if not signals:
            logger.warning("Aucun signal à sauvegarder")
            return False
        
        try:
            # Préparer les données avec timestamp (votre logique exacte)
            df_new = pd.DataFrame(signals)
            if df_new.empty:
                logger.warning("DataFrame vide, aucune sauvegarde")
                return False
            
            # Ajout d'un timestamp pour le moment de détection
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new['detection_time'] = detection_time
            
            file_path = self.signals_dir / filename
            
            # Vérifier si le fichier existe déjà (votre logique)
            if file_path.exists():
                try:
                    # Lire l'historique existant
                    df_old = pd.read_csv(file_path)
                    
                    # Fusionner les nouveaux signaux avec l'historique
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    
                    # Supprimer les doublons en gardant la dernière version
                    df_combined = df_combined.sort_values(
                        by=['detection_time', 'Symbole', 'Fiabilite'],
                        ascending=[True, False, False]
                    )
                    
                    df_clean = df_combined.drop_duplicates(
                        subset=['Symbole', 'Signal', 'Prix', 'RSI'],
                        keep='first'
                    )
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture CSV: {e}")
                    df_clean = df_new
            else:
                df_clean = df_new
            
            # Sauvegarde avec vérification de la structure (votre logique exacte)
            # Sauvegarder avec date de mise à jour dans le nom
            timestamp = datetime.now().strftime("%Y%m%d")
            base_name = Path(filename).stem  # enlève l'extension
            archive_file = self.signals_dir / f"{base_name}_{timestamp}.csv"
            
            # Sauvegarde archive
            df_clean.to_csv(archive_file, index=False)
            
            # Sauvegarde principale
            df_clean.to_csv(file_path, index=False)
            
            logger.info(f"💾 Signaux sauvegardés: {filename} (archive: {archive_file.name})")
            return True
            
        except Exception as e:
            logger.error(f"🚨 Erreur sauvegarde CSV: {e}")
            return False
    
    def load_signals(self, filename: str = "signaux_trading.csv") -> pd.DataFrame:
        """Charge les signaux depuis un fichier CSV."""
        file_path = self.signals_dir / filename
        
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"📂 Signaux chargés: {len(df)} entrées depuis {filename}")
                return df
            else:
                logger.warning(f"Fichier non trouvé: {filename}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erreur chargement CSV: {e}")
            return pd.DataFrame()
    
    def get_latest_signals(self, filename: str = "signaux_trading.csv", 
                          hours: int = 24) -> pd.DataFrame:
        """Retourne les signaux des dernières heures."""
        df = self.load_signals(filename)
        
        if df.empty or 'detection_time' not in df.columns:
            return df
        
        try:
            # Filtrer par date
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
            df['detection_time'] = pd.to_datetime(df['detection_time'])
            recent_signals = df[df['detection_time'] >= cutoff_time]
            
            logger.info(f"🕐 Signaux récents ({hours}h): {len(recent_signals)} entrées")
            return recent_signals
            
        except Exception as e:
            logger.error(f"Erreur filtrage par date: {e}")
            return df

# Instance globale
signal_saver = SignalSaver()

# Fonction de compatibilité
def save_to_evolutive_csv(signals: List[Dict], filename: str = "signaux_trading.csv") -> bool:
    """Fonction de compatibilité avec votre code existant."""
    return signal_saver.save_to_evolutive_csv(signals, filename)