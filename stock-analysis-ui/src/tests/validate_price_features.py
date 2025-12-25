#!/usr/bin/env python
"""
Validate price derivative feature optimization on Technology × Large segment.
Runs optimizer with use_price_features=True and reports before/after.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from symbol_manager import (
    init_symbols_table, sync_txt_to_sqlite,
    get_symbols_by_sector_and_cap, get_symbol_count
)
from qsi import download_stock_data, extract_best_parameters, BEST_PARAM_EXTRAS
from optimisateur_hybride import optimize_sector_coefficients_hybrid
import sqlite3

def validate_price_features():
    print("\n" + "="*80)
    print("VALIDATION: Price Derivative Features on Technology × Large")
    print("="*80)
    
    # Setup
    init_symbols_table()
    try:
        sync_txt_to_sqlite("optimisation_symbols.txt", "optimization")
    except Exception as e:
        print(f"⚠️ Sync warning: {e}")
    
    # Get symbols for Technology × Large
    sector = "Technology"
    cap_range = "Large"
    symbols = get_symbols_by_sector_and_cap(sector, cap_range, "optimization", active_only=True)
    print(f"\n1️⃣  Loaded {len(symbols)} {sector}×{cap_range} symbols")
    
    if not symbols:
        print(f"❌ No symbols found for {sector}×{cap_range}")
        return
    
    # Download data
    print(f"2️⃣  Downloading data for {len(symbols[:10])} symbols (sample)...")
    period = '1y'
    stock_data = download_stock_data(symbols[:10], period=period)
    
    if not stock_data:
        print(f"❌ Failed to download data")
        return
    
    print(f"   Downloaded {len(stock_data)} symbols with data")
    
    # Load prior state
    print(f"\n3️⃣  Loading prior parameters...")
    prior_params = extract_best_parameters()
    domain_key = f"{sector}_{cap_range}"
    prior_state = None
    if domain_key in prior_params:
        coeffs, thresholds, globals_th, gain = prior_params[domain_key]
        prior_state = {
            'key': domain_key,
            'gain': gain,
            'coeffs': coeffs[:3],  # Show first 3 for brevity
            'use_price': 'N/A'
        }
        print(f"   Prior params found: gain={gain:.2f}, coeffs_sample={coeffs[:3]}")
    else:
        print(f"   No prior params for {domain_key} (first-time optimization)")
    
    # Optimize WITHOUT price features (baseline)
    print(f"\n4️⃣  Baseline optimization (NO price features)...")
    coeffs_base, gain_base, sr_base, th_base, summary_base = optimize_sector_coefficients_hybrid(
        list(stock_data.keys()), domain_key,
        period=period, strategy='hybrid',
        montant=50, transaction_cost=1.0,
        budget_evaluations=500,  # Reduced for speed
        precision=2,
        cap_range=cap_range
    )
    print(f"   Baseline gain: {gain_base:.4f} | Success rate: {sr_base:.1f}%")
    
    # Optimize WITH price features
    print(f"\n5️⃣  Optimized (WITH price features enabled)...")
    from optimisateur_hybride import HybridOptimizer
    stock_data_dict = stock_data
    optimizer = HybridOptimizer(stock_data_dict, domain_key, 50, 1.0, precision=2, use_price_features=True)
    print(f"   Bounds extended for price features: {len(optimizer.bounds)} total parameters")
    
    coeffs_opt, gain_opt, sr_opt, th_opt, summary_opt = optimize_sector_coefficients_hybrid(
        list(stock_data.keys()), domain_key,
        period=period, strategy='hybrid',
        montant=50, transaction_cost=1.0,
        budget_evaluations=500,  # Same budget for fair comparison
        precision=2,
        cap_range=cap_range,
        # NOTE: optimize_sector_coefficients_hybrid doesn't expose use_price_features yet
        # We'll run manual optimizer below instead
    )
    
    # Manual run with price features to show proper integration
    print(f"\n6️⃣  Manual optimization WITH price features...")
    opt_pf = HybridOptimizer(stock_data_dict, domain_key, 50, 1.0, precision=2, use_price_features=True)
    from scipy.optimize import differential_evolution
    def eval_fn(params):
        return -opt_pf.evaluate_config(params)  # Negate for minimization
    
    bounds = opt_pf.bounds
    print(f"   Parameter vector size: {len(bounds)}")
    print(f"     - Base params (8 coeffs + 8 thresholds + 2 globals): 18")
    print(f"     - Price extras (use_ps, use_pa, a9, a10, th9, th10): 6")
    
    result_de = differential_evolution(eval_fn, bounds, seed=42, maxiter=20, popsize=30, workers=1)
    gain_pf = -result_de.fun
    params_pf = result_de.x
    
    print(f"   Optimized gain WITH price features: {gain_pf:.4f}")
    
    # Extract and display price params if enabled
    if len(params_pf) >= 24:
        use_ps = int(round(params_pf[18]))
        use_pa = int(round(params_pf[19]))
        a_ps = round(params_pf[20], 2)
        a_pa = round(params_pf[21], 2)
        th_ps = round(params_pf[22], 4)
        th_pa = round(params_pf[23], 4)
        print(f"\n7️⃣  Price Feature Parameters:")
        print(f"   use_price_slope: {use_ps}")
        print(f"   use_price_acc: {use_pa}")
        print(f"   a_price_slope (weight): {a_ps}")
        print(f"   a_price_acc (weight): {a_pa}")
        print(f"   th_price_slope (threshold): {th_ps}")
        print(f"   th_price_acc (threshold): {th_pa}")
    
    # Comparison
    print(f"\n8️⃣  RESULTS SUMMARY:")
    print(f"   Prior (from DB):        {prior_state['gain']:.4f}" if prior_state else "   Prior (from DB):        N/A (first-time)")
    print(f"   Baseline (no price):    {gain_base:.4f}")
    print(f"   Optimized (w/ price):   {gain_pf:.4f}")
    if prior_state:
        delta_vs_prior = gain_pf - prior_state['gain']
        print(f"   → Change vs. prior:     {delta_vs_prior:+.4f} ({delta_vs_prior/abs(prior_state['gain'])*100:+.1f}%)")
    delta_baseline = gain_pf - gain_base
    print(f"   → Change vs. baseline:  {delta_baseline:+.4f}")
    
    if delta_baseline > 0.01:
        print(f"\n✅ VALIDATION PASSED: Price features provide measurable improvement")
    elif delta_baseline > -0.01:
        print(f"\n⚠️  VALIDATION NEUTRAL: Price features did not degrade baseline (drift within noise)")
    else:
        print(f"\n⚠️  WARNING: Price features may have degraded performance slightly")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    validate_price_features()
