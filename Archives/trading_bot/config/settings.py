"""
Configuration centralisée pour le trading bot.
Intègre tous les paramètres de vos fichiers originaux.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import pandas as pd

@dataclass
class PathsConfig:
    """Configuration des chemins."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    symbols_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "symbols")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    optimization_csv: str = "optimization_hist_4stp.csv"

@dataclass
class TradingConfig:
    """Configuration du trading."""
    # Paramètres de base (vos valeurs exactes)
    min_data_points: int = 50
    position_size: float = 50.0
    transaction_cost: float = 0.01
    default_period: str = "12mo"
    
    # Seuils par défaut (vos valeurs exactes)
    default_buy_threshold: float = 4.20
    default_sell_threshold: float = -0.5
    
    # Filtres (vos valeurs exactes)
    variation_threshold: float = -20.0
    volume_threshold: int = 100000
    
    # Coefficients par défaut (vos valeurs exactes)
    default_coefficients: Tuple[float, ...] = (1.75, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75)

@dataclass
class IndicatorsConfig:
    """Configuration des indicateurs techniques."""
    # MACD (vos valeurs exactes)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # RSI (votre valeur exacte)
    rsi_period: int = 17
    
    # EMAs (vos valeurs exactes)
    ema20_span: int = 20
    ema50_span: int = 50
    ema200_span: int = 200
    
    # Bollinger Bands (vos valeurs exactes)
    bb_window: int = 20
    bb_std_dev: int = 2
    
    # ADX (votre valeur exacte)
    adx_window: int = 14
    adx_strong_threshold: float = 25.0
    
    # Ichimoku (vos valeurs exactes)
    ichimoku_conversion: int = 9
    ichimoku_base: int = 26
    ichimoku_span: int = 52

@dataclass
class OptimizationConfig:
    """Configuration de l'optimisation."""
    # Paramètres d'optimisation (vos valeurs exactes)
    n_iterations: int = 10
    max_cycles: int = 255
    convergence_threshold: float = 0.1
    period: str = "1y"
    position_size: float = 50.0
    transaction_cost: float = 0.02
    
    # Plages de valeurs (vos plages exactes)
    coeff_min: float = 0.5
    coeff_max: float = 3.0
    buy_threshold_min: float = 2.0
    buy_threshold_max: float = 6.0
    sell_threshold_min: float = -3.0
    sell_threshold_max: float = 0.0
    
    # Cache pour éviter les doublons
    max_tested_configs: int = 60

@dataclass
class DataConfig:
    """Configuration des données."""
    # Yahoo Finance
    valid_periods: List[str] = field(default_factory=lambda: [
        '1d', '5d', '1mo', '3mo', '6mo', '12mo', '1y', '18mo', '24mo', 
        '2y', '5y', '10y', 'ytd', 'max'
    ])
    
    yahoo_suffixes: Tuple[str, ...] = (
        '.HK', '.DE', '.PA', '.AS', '.SW', '.L', '.TO', '.V', '.MI', '.AX', '.SI',
        '.KQ', '.T', '.OL', '.HE', '.ST', '.CO', '.SA', '.MX', '.TW', '.JO', '.SZ', 
        '.NZ', '.KS', '.PL', '.IR', '.MC', '.VI', '.BK', '.SS', '.SG', '.F', '.BE', 
        '.CN', '.TA', '-USD', '=F'
    )
    
    # Cache
    default_cache_age_hours: int = 6
    batch_size: int = 100
    max_workers: int = 8
    timeout: int = 30

@dataclass
class LoggingConfig:
    """Configuration du logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_name: str = "trading_bot.log"
    max_bytes: int = 10_000_000  # 10MB
    backup_count: int = 5

class TradingBotConfig:
    """Configuration principale du trading bot."""
    
    def __init__(self):
        self.paths = PathsConfig()
        self.trading = TradingConfig()
        self.indicators = IndicatorsConfig()
        self.optimization = OptimizationConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        
        # Créer les dossiers nécessaires
        self._create_directories()
        
        # Charger les paramètres optimisés
        self._sector_parameters = None
    
    def _create_directories(self):
        """Crée les dossiers nécessaires."""
        for directory in [self.paths.data_dir, self.paths.symbols_dir, 
                         self.paths.cache_dir, self.paths.results_dir, 
                         self.paths.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Créer le dossier signaux
        (self.paths.results_dir / "signaux").mkdir(exist_ok=True)
    
    def _load_sector_parameters(self) -> Dict:
        """Charge les paramètres optimisés depuis le CSV."""
        if self._sector_parameters is not None:
            return self._sector_parameters
        
        csv_path = self.paths.results_dir / "signaux" / self.paths.optimization_csv
        
        try:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    required_columns = ['Sector', 'Gain_moy', 'Success_Rate', 'Trades', 
                                      'Seuil_Achat', 'Seuil_Vente'] + [f'a{i+1}' for i in range(8)]
                    
                    if all(col in df.columns for col in required_columns):
                        # Trier et prendre les meilleurs paramètres par secteur
                        df_sorted = df.sort_values(
                            by=['Sector', 'Gain_moy', 'Success_Rate', 'Trades'], 
                            ascending=[True, False, False, False]
                        )
                        best_params = df_sorted.groupby('Sector').first().reset_index()
                        
                        result = {}
                        for _, row in best_params.iterrows():
                            sector = row['Sector']
                            coefficients = tuple(row[f'a{i+1}'] for i in range(8))
                            thresholds = (row['Seuil_Achat'], row['Seuil_Vente'])
                            gain_moy = row['Gain_moy']
                            result[sector] = (coefficients, thresholds, gain_moy)
                        
                        self._sector_parameters = result
                        return result
        except Exception as e:
            print(f"Erreur lors du chargement des paramètres sectoriels: {e}")
        
        # Paramètres par défaut si échec du chargement
        self._sector_parameters = self._get_default_sector_parameters()
        return self._sector_parameters
    
    def _get_default_sector_parameters(self) -> Dict:
        """Retourne les paramètres par défaut pour tous les secteurs."""
        default_coeffs = self.trading.default_coefficients
        default_thresholds = (self.trading.default_buy_threshold, 
                            self.trading.default_sell_threshold)
        
        sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
            'Industrials', 'Energy', 'Basic Materials', 'Communication Services',
            'Consumer Defensive', 'Utilities', 'Real Estate', 'ℹ️Inconnu!!', 'default'
        ]
        
        return {sector: (default_coeffs, default_thresholds, 0.0) for sector in sectors}
    
    def get_sector_parameters(self, sector: str) -> Tuple:
        """
        Retourne les paramètres optimisés pour un secteur.
        
        Returns:
            Tuple (coefficients, thresholds, gain_moyen)
        """
        params = self._load_sector_parameters()
        return params.get(sector, params.get('default', params.get('Technology')))
    
    def reload_sector_parameters(self):
        """Force le rechargement des paramètres sectoriels."""
        self._sector_parameters = None
        return self._load_sector_parameters()

# Instance globale
config = TradingBotConfig()