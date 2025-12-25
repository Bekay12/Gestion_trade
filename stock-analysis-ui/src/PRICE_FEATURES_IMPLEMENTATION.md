# Price Derivative Feature Implementation Summary

## Overview
Successfully implemented and validated **price momentum/acceleration derivatives** within the signal generation and optimization framework, with full backwards compatibility and opt-in feature flagging.

## Completed Work

### 1. Technical Derivatives in qsi.py ✅
- **compute_derivatives()**: Computes price/MACD/RSI/volume slopes over 8-bar windows
- **compute_accelerations()**: Computes second-order derivatives (slope differences across adjacent windows)
- **Return values**: Always returns pairs for each metric:
  - `*_slope` and `*_slope_rel` (relative to last value)
  - `*_acc` and `*_acc_rel` (acceleration)
- **Retrocompatibility**: Existing code unaffected; new values added to derivatives dict when `return_derivatives=True`

### 2. Database Schema Extension ✅
- **migration_csv_to_sqlite.py**: Idempotent schema upgrade
- **New columns in optimization_runs table**:
  - `market_cap_range` (TEXT): Ensures newer schema consistency
  - `a9`, `a10` (REAL): Weight coefficients for price_slope and price_acc
  - `th9`, `th10` (REAL): Threshold comparisons for price derivatives
  - `use_price_slope`, `use_price_acc` (INTEGER DEFAULT 0): Feature flags
- **Execution**: Columns added on-the-fly if missing; non-breaking for existing data

### 3. Parameter Extraction & Exposure ✅
- **extract_best_parameters()** in both `qsi.py` and `trading_c_acceleration/qsi_optimized.py`:
  - Always reads and populates **BEST_PARAM_EXTRAS** global map
  - Per sector and composite key (Sector_CapRange): stores price param dict
  - Safe defaults (zeros) when DB columns are missing
  - **Price extras always extracted** (not optional) for consistency
- **Return shape unchanged**: Still returns 4-tuple (coeffs8, thresholds8, globals2, gain); extras in separate module-level map

### 4. Signal Scoring Integration ✅
- **get_trading_signal() signature expanded**:
  - New param: `price_extras: Dict[str, Union[int, float]]`
  - Reads flags `use_price_slope`, `use_price_acc`
  - Applies weights `a_price_slope`, `a_price_acc` to score when enabled
  - Compares derivatives to thresholds `th_price_slope`, `th_price_acc`
- **Scoring logic**:
  ```python
  if use_price_slope:
      if price_slope_rel > th_price_slope:
          score += a_price_slope
      else:
          score -= a_price_slope
  # Similar for acceleration
  ```
- **Minimal computation**: Only calculates derivatives if flags are set; lazy evaluation prevents overhead

### 5. Optimizer Enhancement with Feature Flag ✅
- **HybridOptimizer**:
  - New param: `use_price_features: bool` (default False)
  - When False: Uses standard 18-parameter vector (8 coeffs + 8 thresholds + 2 globals)
  - When True: Extends to 24 parameters, adding:
    - Indices 18-19: `use_price_slope`, `use_price_acc` (rounded to 0/1)
    - Indices 20-21: `a_price_slope`, `a_price_acc` weights (bounds 0–3)
    - Indices 22-23: `th_price_slope`, `th_price_acc` thresholds (bounds ±0.05)
- **Evaluation**:
  - `evaluate_config()` extracts extra params and passes to backtest
  - Uses `backtest_signals_with_events()` when price features enabled
  - Forwards extras into `get_trading_signal()` during backtest loop

### 6. Persistence ✅
- **save_optimization_results()** now accepts `extra_params` dict
- Persists price feature weights/thresholds into new DB columns
- Defaults gracefully handle missing extras (all zeros)
- INSERT statement updated to include new columns

### 7. Integration Testing ✅
- **Test coverage**:
  - ✅ `get_trading_signal()` accepts and uses `price_extras`
  - ✅ `backtest_signals_with_events()` forwards extras to signal
  - ✅ `HybridOptimizer` extends bounds correctly when flagged
  - ✅ Optimizer evaluates extended parameter vector
  - ✅ `BEST_PARAM_EXTRAS` always populated with price param defaults
- **Backwards compatibility**:
  - Base optimizer (use_price_features=False) → 18-param vector unchanged
  - Signal without price_extras → no price adjustment
  - Price params default to zero → neutral effect

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    signal generation flow                        │
└─────────────────────────────────────────────────────────────────┘

User code (e.g., backtest_signals_with_events)
    │
    ├─ pass: domaine, cap_range, price_extras dict
    │
    v
get_trading_signal(prices, volumes, domaine, price_extras=...)
    │
    ├─ extract price_slope_rel, price_acc_rel from derivatives
    │
    ├─ lookup flags: use_price_slope, use_price_acc
    │ └─ from BEST_PARAM_EXTRAS[selected_key] (with DB fallback)
    │
    ├─ apply score adjustment when flags set:
    │ │  if use_price_slope and price_slope_rel > th_price_slope:
    │ │      score += a_price_slope
    │ │  (similar for acceleration)
    │
    └─ return: signal, price, trend, rsi, volume_mean, score, derivatives

┌─────────────────────────────────────────────────────────────────┐
│                    optimization flow                             │
└─────────────────────────────────────────────────────────────────┘

HybridOptimizer(stock_data, domain, use_price_features=True/False)
    │
    ├─ if use_price_features:
    │ │  bounds = [18 base] + [6 price feature bounds]
    │ │  parameter_vector_size = 24
    │ │
    │ └─ optimize over extended space
    │    └─ evaluate_config(params[24]):
    │       │
    │       ├─ extract use_ps, use_pa flags (params[18:20])
    │       ├─ extract a9, a10 weights (params[20:22])
    │       ├─ extract th9, th10 thresholds (params[22:24])
    │       │
    │       └─ backtest_signals_with_events(..., extra_params={...})
    │          └─ get_trading_signal(..., price_extras={...})
    │
    └─ save_optimization_results(..., extra_params={a9, a10, th9, th10, ...})
       └─ persist into DB columns a9, a10, th9, th10, use_price_slope, use_price_acc
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Always extract price params, not optional** | Ensures consistency across all optimizations; defaults (zeros) maintain neutrality |
| **Separate BEST_PARAM_EXTRAS global map** | Keeps original 4-tuple return shape stable for perfect backwards compatibility |
| **Feature flag in HybridOptimizer** | Allows gradual rollout; can test on single segments before full deployment |
| **Use_price_slope/acc as integer flags** | Cleaner than floats; easily enables/disables contribution without code branching |
| **Relative derivative values as thresholds** | Normalized (price_slope_rel, price_acc_rel) work across different price ranges |
| **Lazy evaluation in get_trading_signal** | Only compute price derivatives if flags set; minimal overhead for disabled features |

## Usage Examples

### Default (Backwards Compatible)
```python
# Optimization runs WITHOUT price features (original behavior)
coeffs, gain, sr, thresholds, summary = optimize_sector_coefficients_hybrid(
    symbols, 'Technology_Large',
    # use_price_features not specified → defaults to False
)
# Parameter vector: 18 (unchanged)
```

### With Price Features Enabled
```python
# To enable price feature optimization:
# 1. Pass use_price_features=True to HybridOptimizer
#    (Currently not exposed in optimize_sector_coefficients_hybrid; 
#     create optimizer directly for testing)

opt = HybridOptimizer(stock_data, 'Technology_Large', use_price_features=True)
# Parameter vector: 24 (18 base + 6 price)
# Optimizer now searches over price_slope and price_acc weights/thresholds
```

### Reading Optimized Price Params
```python
from qsi import extract_best_parameters, BEST_PARAM_EXTRAS

params = extract_best_parameters()
extras = BEST_PARAM_EXTRAS.get('Technology_Large', {})

if extras['use_price_slope']:
    print(f"Price slope enabled with weight {extras['a_price_slope']}")
else:
    print("Price slope disabled")
```

## Next Steps (Phase 2: Fundamentals)

When ready to add financial metrics (growth, margin, FCF yield, D/E, EPS growth):

1. **Add columns** to optimization_runs: `a11`, `a12`, ..., `th11`, `th12`, ... (fundamentals weights)
2. **Update extract_best_parameters()**: Add fundamentals to BEST_PARAM_EXTRAS (optional, with safe defaults)
3. **Extend get_trading_signal()**: Add `financial_extras` param (similar to price_extras)
4. **Update optimizer**: Option to include fundamentals in parameter vector (use_fundamental_features flag)
5. **Persist fundamentals**: Save via extra_params dict in save_optimization_results

This follows the exact same pattern as price features, so integration is straightforward.

## Testing Checklist

- [x] Synthetic data: get_trading_signal with price_extras → score includes price contribution
- [x] Bounds: HybridOptimizer(use_price_features=True) → 24-param vector
- [x] Extraction: extract_best_parameters() → BEST_PARAM_EXTRAS always has price defaults
- [x] Signal: get_trading_signal(price_extras=None) → no price effect (backwards compat)
- [x] Schema: save_optimization_results saves new columns without error
- [x] No regressions: Existing optimization code runs unchanged when feature flag is False

## Files Modified

1. **qsi.py**
   - Added `BEST_PARAM_EXTRAS` global map
   - Extended `extract_best_parameters()` to populate price params
   - Added `compute_accelerations()` helper
   - Updated `get_trading_signal()` to accept `price_extras` and integrate price scoring

2. **trading_c_acceleration/qsi_optimized.py**
   - Mirrored changes from qsi.py
   - Updated `extract_best_parameters()` for price extras
   - Updated `backtest_signals_with_events()` to accept and forward `extra_params`

3. **optimisateur_hybride.py**
   - Added schema migration for new DB columns
   - Extended `HybridOptimizer.__init__()` with `use_price_features` flag
   - Updated `evaluate_config()` to extract and pass price extras
   - Extended `optimize_sector_coefficients_hybrid()` result extraction for price params
   - Updated `save_optimization_results()` to persist price extras

---

**Status**: ✅ **COMPLETE** — Price derivative feature framework fully integrated, tested, and ready for production use or further enhancement with fundamentals.
