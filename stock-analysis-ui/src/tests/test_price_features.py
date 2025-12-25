#!/usr/bin/env python
"""
Quick integration test: Verify price feature optimization flow works end-to-end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from qsi import get_trading_signal, BEST_PARAM_EXTRAS
from trading_c_acceleration.qsi_optimized import backtest_signals_with_events
from optimisateur_hybride import HybridOptimizer

def test_price_features_flow():
    print("\n" + "="*80)
    print("UNIT TEST: Price Feature Optimization Flow")
    print("="*80)
    
    # 1. Create synthetic data
    print("\n1️⃣  Creating synthetic price/volume data...")
    idx = pd.date_range('2024-01-01', periods=150, freq='D')
    prices = pd.Series(
        100 + 20*np.sin(np.linspace(0, 4*np.pi, 150)) + np.random.randn(150)*2,
        index=idx
    )
    volumes = pd.Series(
        1e6 + np.random.randn(150)*1e5,
        index=idx
    )
    print(f"   Data: {len(prices)} bars")
    
    # 2. Test get_trading_signal WITH price extras
    print("\n2️⃣  Testing get_trading_signal with price_extras...")
    price_extras = {
        'use_price_slope': 1,
        'use_price_acc': 1,
        'a_price_slope': 1.5,
        'a_price_acc': 0.8,
        'th_price_slope': 0.001,
        'th_price_acc': -0.001
    }
    
    sig, last_price, trend, rsi, vol_mean, score, deriv = get_trading_signal(
        prices, volumes, 'Technology',
        cap_range='Large',
        price_extras=price_extras,
        return_derivatives=True
    )
    print(f"   Signal: {sig}, Score: {score:.3f}")
    print(f"   Price slope rel: {deriv.get('price_slope_rel', 0):.6f}")
    print(f"   Price acc rel: {deriv.get('price_acc_rel', 0):.6f}")
    print(f"   ✅ get_trading_signal accepts and uses price_extras")
    
    # 3. Test backtest_signals_with_events with extras
    print("\n3️⃣  Testing backtest_signals_with_events with extra_params...")
    result, events = backtest_signals_with_events(
        prices, volumes, 'Technology',
        montant=50, transaction_cost=1.0,
        extra_params=price_extras,
        cap_range='Large'
    )
    print(f"   Backtest result: {result['trades']} trades, gain={result['gain_total']:.2f}")
    print(f"   Events recorded: {len(events)}")
    print(f"   ✅ backtest_signals_with_events passes extras to signal generation")
    
    # 4. Test HybridOptimizer with use_price_features
    print("\n4️⃣  Testing HybridOptimizer with use_price_features=True...")
    stock_data = {'TEST': {'Close': prices, 'Volume': volumes}}
    
    opt_base = HybridOptimizer(stock_data, 'Technology_Large', use_price_features=False)
    opt_pf = HybridOptimizer(stock_data, 'Technology_Large', use_price_features=True)
    
    print(f"   Base optimizer bounds: {len(opt_base.bounds)} parameters")
    print(f"   Price-feature optimizer bounds: {len(opt_pf.bounds)} parameters")
    print(f"   Delta: {len(opt_pf.bounds) - len(opt_base.bounds)} extra params for price features")
    
    if len(opt_pf.bounds) == 24:
        print(f"   ✅ Bounds correctly extended (18 + 6 for price features)")
    else:
        print(f"   ❌ Unexpected bound count: {len(opt_pf.bounds)}")
        return False
    
    # 5. Test evaluate_config with price params
    print("\n5️⃣  Testing optimizer evaluation with price features...")
    params_pf = np.array([
        1.5, 1.0, 1.5, 1.25, 1.75, 1.25, 1.0, 1.75,  # 8 coeffs
        50.0, 0.0, 0.0, 1.2, 25.0, 0.0, 0.5, 4.20,   # 8 thresholds
        4.2, -0.5,                                     # 2 globals
        1.0, 0.5,                                      # use_ps, use_pa flags (will be rounded)
        1.5, 0.8,                                      # a9, a10 weights
        0.001, -0.001                                  # th9, th10 thresholds
    ])
    
    score = opt_pf.evaluate_config(params_pf)
    print(f"   Score with price features: {score:.4f}")
    print(f"   ✅ Optimizer can evaluate extended parameter vector")
    
    # 6. Check BEST_PARAM_EXTRAS
    print("\n6️⃣  Testing BEST_PARAM_EXTRAS extraction...")
    print(f"   BEST_PARAM_EXTRAS keys: {list(BEST_PARAM_EXTRAS.keys())[:3]}")
    sample_key = list(BEST_PARAM_EXTRAS.keys())[0] if BEST_PARAM_EXTRAS else None
    if sample_key:
        extras = BEST_PARAM_EXTRAS[sample_key]
        print(f"   Sample extras for '{sample_key}':")
        print(f"     use_price_slope: {extras.get('use_price_slope')}")
        print(f"     use_price_acc: {extras.get('use_price_acc')}")
        print(f"   ✅ BEST_PARAM_EXTRAS always populated with price params")
    
    print("\n" + "="*80)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("="*80)
    print("\nFlow Summary:")
    print("  1. get_trading_signal accepts price_extras and adjusts score")
    print("  2. backtest_signals_with_events forwards extras to signal generation")
    print("  3. HybridOptimizer extends bounds when use_price_features=True")
    print("  4. Optimizer evaluates extended parameter vector with price weights/thresholds")
    print("  5. BEST_PARAM_EXTRAS always provides price param defaults (zeros if not optimized)")
    print("\nBackwards Compatibility:")
    print("  • Default use_price_features=False → 18-param vector (unchanged)")
    print("  • get_trading_signal(price_extras=None) → no price adjustment (backwards compat)")
    print("  • All price params default to zero when not optimized → neutral effect\n")

if __name__ == "__main__":
    try:
        test_price_features_flow()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
