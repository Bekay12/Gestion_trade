"""
Integration tests for fundamentals features in trading signal and optimizer.
Tests fundamentals extraction, caching, signal scoring, and optimizer integration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from fundamentals_cache import get_fundamental_metrics, clear_fundamentals_cache

def test_fundamentals_extraction():
    """Test fundamentals extraction from yfinance."""
    print("📋 Test 1: Fundamentals extraction...")
    
    metrics = get_fundamental_metrics('AAPL', use_cache=False)
    
    # Check structure
    required_keys = ['rev_growth', 'eps_growth', 'gross_margin', 'fcf_yield', 'de_ratio', 'roe', 'ocf_yield', 'net_margin']
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
    
    # Check types
    for key, val in metrics.items():
        assert val is None or isinstance(val, (int, float)), f"{key} is not numeric: {type(val)}"
    
    print(f"✅ Fundamentals extracted for AAPL: {sum(1 for v in metrics.values() if v is not None)}/8 metrics available")
    return metrics

def test_cache_functionality():
    """Test caching and retrieval of fundamentals."""
    print("\n📋 Test 2: Cache functionality...")
    
    # Fetch fresh
    metrics1 = get_fundamental_metrics('MSFT', use_cache=False)
    
    # Fetch from cache (should be instant)
    metrics2 = get_fundamental_metrics('MSFT', use_cache=True)
    
    # Verify same values
    for key in metrics1:
        if metrics1[key] is not None:
            assert metrics2[key] == metrics1[key], f"Cache mismatch on {key}"
    
    print("✅ Cache retrieval works correctly")

def test_signal_with_fundamentals():
    """Test get_trading_signal with fundamentals_extras parameter."""
    print("\n📋 Test 3: Signal with fundamentals_extras...")
    
    try:
        from qsi import get_trading_signal
        
        # Create synthetic price/volume data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
        volumes = pd.Series(np.random.randint(100000, 500000, 100), index=dates)
        
        # Test 1: Signal without fundamentals
        signal1, price, trend, rsi, volume_mean, score1, deriv1 = get_trading_signal(
            prices, volumes, 'Technology',
            symbol='AAPL'
        )
        assert signal1 in ['ACHAT', 'VENTE', 'NEUTRE'], f"Invalid signal: {signal1}"
        print(f"   Signal without fundamentals: {signal1}, Score: {score1:.3f}")
        
        # Test 2: Signal with fundamentals_extras (disabled flag)
        fund_extras_disabled = {
            'use_fundamentals': 0,
            'a_rev_growth': 0.5,  # Won't be used
        }
        signal2, price, trend, rsi, volume_mean, score2, deriv2 = get_trading_signal(
            prices, volumes, 'Technology',
            symbol='AAPL',
            fundamentals_extras=fund_extras_disabled
        )
        assert signal2 in ['ACHAT', 'VENTE', 'NEUTRE'], f"Invalid signal: {signal2}"
        print(f"   Signal with disabled fundamentals: {signal2}, Score: {score2:.3f}")
        # Scores should be same when fundamentals disabled
        assert score1 == score2, "Score changed with disabled fundamentals"
        
        # Test 3: Signal with fundamentals_extras (enabled flag)
        fund_extras_enabled = {
            'use_fundamentals': 1,
            'a_rev_growth': 0.5,
            'a_eps_growth': 0.3,
            'a_roe': 0.4,
            'a_fcf_yield': 0.2,
            'a_de_ratio': 0.1,
            'th_rev_growth': 5.0,
            'th_eps_growth': 5.0,
            'th_roe': 15.0,
            'th_fcf_yield': 2.0,
            'th_de_ratio': 0.5,
        }
        signal3, price, trend, rsi, volume_mean, score3, deriv3 = get_trading_signal(
            prices, volumes, 'Technology',
            symbol='AAPL',
            fundamentals_extras=fund_extras_enabled
        )
        assert signal3 in ['ACHAT', 'VENTE', 'NEUTRE'], f"Invalid signal: {signal3}"
        print(f"   Signal with enabled fundamentals: {signal3}, Score: {score3:.3f}")
        # Score may be different when fundamentals enabled
        print(f"   Score delta: {score3 - score1:.3f}")
        
        print("✅ Fundamentals scoring in get_trading_signal works correctly")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_best_param_extras_loading():
    """Test that BEST_PARAM_EXTRAS includes fundamentals parameters."""
    print("\n📋 Test 4: BEST_PARAM_EXTRAS loading...")
    
    try:
        from qsi import extract_best_parameters, BEST_PARAM_EXTRAS
        
        # Trigger extraction
        best_params = extract_best_parameters()
        
        # Check if any extras were loaded
        print(f"   Loaded {len(BEST_PARAM_EXTRAS)} sector keys into BEST_PARAM_EXTRAS")
        
        # Check structure of first extra if available
        if BEST_PARAM_EXTRAS:
            first_key = next(iter(BEST_PARAM_EXTRAS))
            first_extras = BEST_PARAM_EXTRAS[first_key]
            
            # Should contain price params
            expected_keys = [
                'use_price_slope', 'use_price_acc', 
                'a_price_slope', 'a_price_acc',
                'th_price_slope', 'th_price_acc'
            ]
            for key in expected_keys:
                assert key in first_extras, f"Missing price param: {key}"
            
            # May contain fundamentals params (defaults)
            fund_keys = ['use_fundamentals', 'a_rev_growth', 'a_eps_growth', 'a_roe', 'a_fcf_yield', 'a_de_ratio']
            for key in fund_keys:
                if key in first_extras:
                    print(f"   Found fundamentals param: {key} = {first_extras[key]}")
            
            print(f"✅ BEST_PARAM_EXTRAS loaded with {len(first_extras)} parameters")
        else:
            print("⚠️  No optimization history in DB (expected for fresh installation)")
            print("✅ BEST_PARAM_EXTRAS structure is ready for fundamentals")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_optimizer_bounds():
    """Test that HybridOptimizer extends bounds for fundamentals."""
    print("\n📋 Test 5: HybridOptimizer bounds extension...")
    
    try:
        from optimisateur_hybride import HybridOptimizer
        import yfinance as yf
        
        # Create minimal stock data
        prices = yf.download('AAPL', start='2024-01-01', end='2024-01-31', progress=False)['Close']
        data = {'AAPL': {'Close': prices, 'Volume': pd.Series(np.random.randint(10000000, 50000000, len(prices)), index=prices.index)}}
        
        # Test 1: Base optimizer (18 params)
        opt_base = HybridOptimizer(data, 'Technology', use_price_features=False, use_fundamentals_features=False)
        assert len(opt_base.bounds) == 18, f"Expected 18 bounds, got {len(opt_base.bounds)}"
        print(f"   Base optimizer: {len(opt_base.bounds)} parameters")
        
        # Test 2: With price features (24 params)
        opt_price = HybridOptimizer(data, 'Technology', use_price_features=True, use_fundamentals_features=False)
        assert len(opt_price.bounds) == 24, f"Expected 24 bounds, got {len(opt_price.bounds)}"
        print(f"   With price features: {len(opt_price.bounds)} parameters")
        
        # Test 3: With fundamentals features (28 params)
        opt_fund = HybridOptimizer(data, 'Technology', use_price_features=False, use_fundamentals_features=True)
        assert len(opt_fund.bounds) == 28, f"Expected 28 bounds, got {len(opt_fund.bounds)}"
        print(f"   With fundamentals features: {len(opt_fund.bounds)} parameters")
        
        # Test 4: With both (34 params: 18 + 6 price + 10 fund)
        opt_both = HybridOptimizer(data, 'Technology', use_price_features=True, use_fundamentals_features=True)
        assert len(opt_both.bounds) == 34, f"Expected 34 bounds, got {len(opt_both.bounds)}"
        print(f"   With both features: {len(opt_both.bounds)} parameters")
        
        print("✅ HybridOptimizer bounds extend correctly for fundamentals")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    print("=" * 70)
    print("FUNDAMENTALS INTEGRATION TEST SUITE")
    print("=" * 70)
    
    try:
        test_fundamentals_extraction()
        test_cache_functionality()
        test_signal_with_fundamentals()
        test_best_param_extras_loading()
        test_optimizer_bounds()
        
        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
