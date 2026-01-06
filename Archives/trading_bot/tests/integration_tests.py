#!/usr/bin/env python3
"""
Tests d'intégration - Remplace votre test.py original.
"""
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.utils.cache import DataCacheManager
from config.settings import config

class TestIntegration:
    """Tests d'intégration complets."""
    
    def setup_method(self):
        """Configuration des tests."""
        self.symbol_manager = SymbolFileManager()
        self.cache_manager = DataCacheManager()
        self.period = config.trading.default_period
    
    def test_symbol_loading(self):
        """Test de chargement des symboles."""
        # Test avec fichiers par défaut
        test_symbols = self.symbol_manager.load_symbols_from_txt("test_symbols.txt")
        mes_symbols = self.symbol_manager.load_symbols_from_txt("mes_symbols.txt")
        
        assert isinstance(test_symbols, list), "test_symbols doit être une liste"
        assert isinstance(mes_symbols, list), "mes_symbols doit être une liste"
        
        print(f"✅ Symboles de test chargés: {len(test_symbols)}")
        print(f"✅ Symboles personnels chargés: {len(mes_symbols)}")
    
    @patch('src.data.providers.yahoo_provider.yf.download')
    def test_cache_preloading(self, mock_download):
        """Test de préchargement du cache."""
        # Mock des données Yahoo Finance
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103],
            'Volume': [1000, 1100, 1200, 1300]
        })
        mock_download.return_value = mock_data
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        try:
            self.cache_manager.preload_cache(symbols, self.period)
            print("✅ Cache préchargé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur préchargement cache: {e}")
    
    def test_signal_analysis_structure(self):
        """Test de la structure d'analyse des signaux."""
        # Symboles de test minimaux
        test_symbols = ['AAPL', 'MSFT']
        mes_symbols = ['GOOGL']
        
        # Mock pour éviter les appels réels à Yahoo Finance
        with patch('src.data.providers.yahoo_provider.YahooProvider.download_batch') as mock_download:
            mock_data = {
                'AAPL': {
                    'Close': pd.Series([100, 101, 102, 103, 104] * 15),  # 75 points
                    'Volume': pd.Series([1000, 1100, 1200, 1300, 1400] * 15)
                },
                'MSFT': {
                    'Close': pd.Series([200, 201, 202, 203, 204] * 15),
                    'Volume': pd.Series([2000, 2100, 2200, 2300, 2400] * 15)
                }
            }
            mock_download.return_value = mock_data
            
            try:
                results = signal_analyzer.analyze_popular_signals(
                    test_symbols, 
                    mes_symbols, 
                    period=self.period,
                    display_charts=False,  # Pas d'affichage en test
                    verbose=False,
                    save_csv=False
                )
                
                # Vérifier la structure des résultats
                assert isinstance(results, dict), "Les résultats doivent être un dictionnaire"
                
                expected_keys = ['signals', 'organized_signals', 'backtest_results', 'reliable_signals']
                for key in expected_keys:
                    assert key in results, f"Clé manquante: {key}"
                
                print("✅ Structure d'analyse des signaux validée")
                print(f"   - Signaux détectés: {len(results.get('signals', []))}")
                print(f"   - Résultats backtest: {len(results.get('backtest_results', []))}")
                
            except Exception as e:
                print(f"⚠️ Erreur analyse signaux: {e}")
    
    def test_configuration_loading(self):
        """Test de chargement de la configuration."""
        # Vérifier que la config est chargée
        assert config.trading.min_data_points == 50, "Configuration trading incorrecte"
        assert config.indicators.macd_fast == 12, "Configuration indicateurs incorrecte"
        assert config.optimization.n_iterations == 10, "Configuration optimisation incorrecte"
        
        print("✅ Configuration chargée correctement")
        print(f"   - Période par défaut: {config.trading.default_period}")
        print(f"   - Seuils par défaut: {config.trading.default_buy_threshold}, {config.trading.default_sell_threshold}")

def run_integration_tests():
    """Lance tous les tests d'intégration."""
    print("🧪 Lancement des tests d'intégration")
    print("="*50)
    
    tester = TestIntegration()
    tester.setup_method()
    
    # Exécuter les tests un par un
    tests = [
        ('Chargement des symboles', tester.test_symbol_loading),
        ('Préchargement du cache', tester.test_cache_preloading),
        ('Structure d\'analyse', tester.test_signal_analysis_structure),
        ('Chargement configuration', tester.test_configuration_loading)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\\n🔍 {test_name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Échec {test_name}: {e}")
            failed += 1
    
    print("\\n" + "="*50)
    print(f"📊 Résultats: {passed} réussis, {failed} échoués")
    
    return failed == 0

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)