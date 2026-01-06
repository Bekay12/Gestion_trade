"""
Gestionnaire de paramètres d'optimisation.
Migration de votre fonction extract_best_parameters().
"""
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ParameterManager:
    """
    Gestionnaire des paramètres optimisés par secteur.
    Migration de votre fonction extract_best_parameters().
    """

    def __init__(self):
        self.config = config
        self.csv_path = config.paths.results_dir / "signaux" / config.paths.optimization_csv

    def extract_best_parameters(self, csv_path: str = None) -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, float]]]:
        """
        Extrait les meilleurs coefficients et seuils pour chaque secteur à partir du CSV.
        Migration exacte de votre fonction extract_best_parameters().

        Args:
            csv_path: Chemin vers le CSV contenant l'historique d'optimisation.

        Returns:
            Dict[str, Tuple[Tuple[float, ...], Tuple[float, float]]]: Dictionnaire avec pour chaque secteur
            un tuple (coefficients, seuils), où coefficients est (a1, a2, ..., a8) et seuils est (Seuil_Achat, Seuil_Vente).
        """
        if csv_path is None:
            csv_path = self.csv_path
        else:
            csv_path = Path(csv_path)

        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                logger.warning("🚫 CSV vide, aucun paramètre extrait")
                return self._get_default_parameters()

            # Colonnes requises (votre logique exacte)
            required_columns = ['Sector', 'Gain_moy', 'Success_Rate', 'Trades', 
                               'Seuil_Achat', 'Seuil_Vente', 
                               'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.warning(f"🚫 Colonnes manquantes dans le CSV : {missing}")
                return self._get_default_parameters()

            # Trier par Gain_moy (descendant), Success_Rate (descendant), Trades (descendant), Timestamp (descendant)
            # Votre logique exacte
            df_sorted = df.sort_values(
                by=['Sector', 'Gain_moy', 'Success_Rate', 'Trades', 'Timestamp'], 
                ascending=[True, False, False, False, False]
            )

            # Prendre la première entrée par secteur (la meilleure)
            best_params = df_sorted.groupby('Sector').first().reset_index()

            result = {}
            for _, row in best_params.iterrows():
                sector = row['Sector']
                coefficients = tuple(row[f'a{i+1}'] for i in range(8))
                thresholds = (row['Seuil_Achat'], row['Seuil_Vente'])
                gain_moy = row['Gain_moy']
                result[sector] = (coefficients, thresholds, gain_moy)

                logger.debug(f"📊 Meilleurs paramètres pour {sector}: "
                           f"Coefficients={coefficients}, Seuils={thresholds}, "
                           f"Gain_moy={gain_moy:.2f}")

            logger.info(f"✅ Paramètres extraits pour {len(result)} secteurs")
            return result

        except FileNotFoundError:
            logger.warning(f"🚫 Fichier CSV {csv_path} non trouvé")
            return self._get_default_parameters()
        except Exception as e:
            logger.error(f"⚠️ Erreur lors de l'extraction des paramètres: {e}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, float], float]]:
        """
        Retourne les paramètres par défaut.
        Reprend vos coefficients par défaut du code original.
        """
        default_coeffs = (1.75, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75)
        default_thresholds = (self.config.trading.default_buy_threshold, 
                            self.config.trading.default_sell_threshold)

        return {
            'default': (default_coeffs, default_thresholds, 0.0),
            'Technology': (default_coeffs, default_thresholds, 0.0),
            'Healthcare': (default_coeffs, default_thresholds, 0.0),
            'Financial Services': (default_coeffs, default_thresholds, 0.0),
            'Consumer Cyclical': (default_coeffs, default_thresholds, 0.0),
            'Industrials': (default_coeffs, default_thresholds, 0.0),
            'Energy': (default_coeffs, default_thresholds, 0.0),
            'Basic Materials': (default_coeffs, default_thresholds, 0.0),
            'Communication Services': (default_coeffs, default_thresholds, 0.0),
            'Consumer Defensive': (default_coeffs, default_thresholds, 0.0),
            'Utilities': (default_coeffs, default_thresholds, 0.0),
            'Real Estate': (default_coeffs, default_thresholds, 0.0),
            'ℹ️Inconnu!!': (default_coeffs, default_thresholds, 0.0)
        }

    def get_sector_parameters(self, sector: str) -> Tuple[Tuple[float, ...], Tuple[float, float], float]:
        """
        Retourne les paramètres optimisés pour un secteur spécifique.

        Args:
            sector: Nom du secteur.

        Returns:
            Tuple (coefficients, thresholds, gain_moyen).
        """
        all_params = self.extract_best_parameters()
        return all_params.get(sector, all_params.get('default', self._get_default_parameters()['default']))

    def save_parameters(self, sector: str, coefficients: Tuple[float, ...], 
                       thresholds: Tuple[float, float], gain_total: float,
                       success_rate: float, total_trades: int) -> bool:
        """
        Sauvegarde de nouveaux paramètres optimisés.

        Args:
            sector: Nom du secteur.
            coefficients: Tuple de 8 coefficients (a1 à a8).
            thresholds: Tuple (seuil_achat, seuil_vente).
            gain_total: Gain total moyen obtenu.
            success_rate: Taux de réussite.
            total_trades: Nombre total de trades.

        Returns:
            True si sauvegarde réussie.
        """
        try:
            # Préparer les données
            results = {
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Sector': sector,
                'Gain_moy': gain_total,
                'Success_Rate': success_rate,
                'Trades': total_trades,
                'Seuil_Achat': thresholds[0],
                'Seuil_Vente': thresholds[1],
                'a1': coefficients[0], 'a2': coefficients[1], 'a3': coefficients[2], 'a4': coefficients[3],
                'a5': coefficients[4], 'a6': coefficients[5], 'a7': coefficients[6], 'a8': coefficients[7]
            }

            df_new = pd.DataFrame([results])

            # Créer le répertoire si nécessaire
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Vérifier si le fichier existe et éviter les doublons
            if self.csv_path.exists():
                df_existing = pd.read_csv(self.csv_path)

                # Vérifier si la dernière entrée pour le secteur est identique
                last_entry = df_existing[df_existing['Sector'] == sector].tail(1)

                if not last_entry.empty:
                    same_params = (
                        abs(last_entry['Gain_moy'].iloc[0] - gain_total) < 0.01 and
                        abs(last_entry['Success_Rate'].iloc[0] - success_rate) < 0.1 and
                        last_entry['Trades'].iloc[0] == total_trades and
                        abs(last_entry['Seuil_Achat'].iloc[0] - thresholds[0]) < 0.01 and
                        abs(last_entry['Seuil_Vente'].iloc[0] - thresholds[1]) < 0.01 and
                        all(abs(last_entry[f'a{i+1}'].iloc[0] - coefficients[i]) < 0.01 for i in range(8))
                    )

                    if same_params:
                        logger.info(f"📝 Entrée identique pour {sector}, sauvegarde ignorée")
                        return True

                # Ajouter à l'existant
                df_new.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                # Créer nouveau fichier
                df_new.to_csv(self.csv_path, index=False)

            logger.info(f"📝 Résultats sauvegardés pour {sector}")
            return True

        except Exception as e:
            logger.error(f"⚠️ Erreur lors de la sauvegarde dans le CSV: {e}")
            return False

# Instance globale
parameter_manager = ParameterManager()

# Fonction de compatibilité
def extract_best_parameters(csv_path: str = 'signaux/optimization_hist_4stp.csv') -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, float]]]:
    """Fonction de compatibilité avec votre code existant."""
    return parameter_manager.extract_best_parameters(csv_path)
