#!/usr/bin/env python
"""
Validate fundamentals feature optimization on Technology × Large segment.
Runs baseline (no fundamentals) vs fundamentals-enabled optimizer and reports deltas.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import differential_evolution

from symbol_manager import (
    init_symbols_table, sync_txt_to_sqlite,
    get_symbols_by_sector_and_cap
)
from qsi import download_stock_data, extract_best_parameters
from optimisateur_hybride import HybridOptimizer


def _run_de(optimizer: HybridOptimizer, maxiter: int = 4, popsize: int = 8):
    """Run a lightweight differential evolution search and return best gain and params."""
    def eval_fn(params):
        # Negate because DE minimizes
        return -optimizer.evaluate_config(params)

    result = differential_evolution(eval_fn, optimizer.bounds, seed=42, maxiter=maxiter, popsize=popsize, workers=1)
    best_gain = -result.fun
    best_params = result.x
    return best_gain, best_params


def validate_fundamentals_features():
    print("\n" + "="*80)
    print("VALIDATION: Fundamentals Features on Technology × Large")
    print("="*80)

    # Setup symbols
    init_symbols_table()
    try:
        sync_txt_to_sqlite("optimisation_symbols.txt", "optimization")
    except Exception as e:
        print(f"⚠️ Sync warning: {e}")

    sector = "Technology"
    cap_range = "Large"
    symbols = get_symbols_by_sector_and_cap(sector, cap_range, "optimization", active_only=True)
    print(f"\n1️⃣  Loaded {len(symbols)} {sector}×{cap_range} symbols")
    if not symbols:
        print(f"❌ No symbols found for {sector}×{cap_range}")
        return

    # Download data (sample 10 symbols for speed)
    sample_symbols = symbols[:5]
    print(f"2️⃣  Downloading data for {len(sample_symbols)} symbols (sample)...")
    period = '6mo'
    stock_data = download_stock_data(sample_symbols, period=period)
    if not stock_data:
        print("❌ Failed to download data")
        return
    print(f"   Downloaded {len(stock_data)} symbols with data")

    # Load prior parameters (for reference only)
    print(f"\n3️⃣  Loading prior parameters...")
    prior_params = extract_best_parameters()
    domain_key = f"{sector}_{cap_range}"
    if domain_key in prior_params:
        _, _, _, prior_gain = prior_params[domain_key]
        print(f"   Prior gain in DB: {prior_gain:.4f}")
    else:
        prior_gain = None
        print("   No prior params for this segment")

    # Baseline optimizer (no fundamentals)
    print(f"\n4️⃣  Baseline optimization (NO fundamentals)...")
    opt_base = HybridOptimizer(stock_data, domain_key, 50, 1.0, precision=2, use_price_features=False, use_fundamentals_features=False)
    print(f"   Bounds: {len(opt_base.bounds)} parameters (base only)")
    gain_base, params_base = _run_de(opt_base, maxiter=10, popsize=20)
    print(f"   Baseline gain: {gain_base:.4f}")

    # Fundamentals-enabled optimizer
    print(f"\n5️⃣  Optimization WITH fundamentals features...")
    opt_fund = HybridOptimizer(stock_data, domain_key, 50, 1.0, precision=2, use_price_features=False, use_fundamentals_features=True)
    print(f"   Bounds: {len(opt_fund.bounds)} parameters (base + fundamentals)")
    gain_fund, params_fund = _run_de(opt_fund, maxiter=10, popsize=20)
    print(f"   Gain with fundamentals: {gain_fund:.4f}")

    # Extract fundamentals params from best vector
    if len(params_fund) >= 28:
        off = 18  # fundamentals start index when price features disabled
        use_fund = int(round(params_fund[off]))
        a_rev = round(params_fund[off + 1], 3)
        a_eps = round(params_fund[off + 2], 3)
        a_roe = round(params_fund[off + 3], 3)
        a_fcf = round(params_fund[off + 4], 3)
        a_de = round(params_fund[off + 5], 3)
        th_rev = round(params_fund[off + 6], 3)
        th_eps = round(params_fund[off + 7], 3)
        th_roe = round(params_fund[off + 8], 3)
        th_fcf = round(params_fund[off + 9], 3)
        print("\n   Fundamentals parameters:")
        print(f"   use_fundamentals: {use_fund}")
        print(f"   a_rev_growth:     {a_rev}")
        print(f"   a_eps_growth:     {a_eps}")
        print(f"   a_roe:            {a_roe}")
        print(f"   a_fcf_yield:      {a_fcf}")
        print(f"   a_de_ratio:       {a_de}")
        print(f"   th_rev_growth:    {th_rev}")
        print(f"   th_eps_growth:    {th_eps}")
        print(f"   th_roe:           {th_roe}")
        print(f"   th_fcf_yield:     {th_fcf}")

    # Summary
    print(f"\n6️⃣  RESULTS SUMMARY:")
    print(f"   Baseline (no fundamentals):   {gain_base:.4f}")
    print(f"   With fundamentals:            {gain_fund:.4f}")
    delta = gain_fund - gain_base
    print(f"   → Change vs. baseline:        {delta:+.4f}")
    if prior_gain is not None:
        delta_prior = gain_fund - prior_gain
        print(f"   → Change vs. prior DB gain:   {delta_prior:+.4f}")

    if delta > 0.01:
        print("\n✅ VALIDATION PASSED: Fundamentals features improved or matched baseline")
    elif delta > -0.01:
        print("\n⚠️  VALIDATION NEUTRAL: Fundamentals neither improved nor degraded materially")
    else:
        print("\n⚠️  WARNING: Fundamentals may have degraded performance; review thresholds/weights")

    print("\n" + "="*80)


if __name__ == "__main__":
    validate_fundamentals_features()
