# Phase 2: Fundamentals Integration Plan

## Objective
Integrate fundamental financial metrics (growth, profitability, cash flow, leverage, earnings) into signal scoring and optimization, following the proven price-features pattern for seamless backwards compatibility and opt-in feature flagging.

## Fundamentals to Integrate

### Category 1: Growth Metrics
- **Revenue Growth (YoY %)**: Quarter-over-quarter or year-over-year
- **EPS Growth (YoY %)**: Earnings per share growth trend
- **Earnings Growth**: Net income growth

### Category 2: Profitability Metrics
- **Gross Margin (%)**: (Revenue - COGS) / Revenue
- **Operating Margin (%)**: Operating Income / Revenue
- **Net Profit Margin (%)**: Net Income / Revenue

### Category 3: Cash Flow Metrics
- **Free Cash Flow (FCF) Yield (%)**: FCF / Market Cap
- **Operating Cash Flow Yield (%)**: OCF / Market Cap
- **FCF Growth (YoY %)**: Free cash flow trend

### Category 4: Leverage & Solvency
- **Debt-to-Equity Ratio**: Total Debt / Total Equity
- **Debt-to-EBITDA**: Total Debt / EBITDA
- **Current Ratio**: Current Assets / Current Liabilities

### Category 5: Returns & Efficiency
- **Return on Equity (ROE %)**: Net Income / Shareholder Equity
- **Return on Assets (ROA %)**: Net Income / Total Assets
- **ROIC (%)**: NOPAT / Invested Capital

## Data Sources

| Metric | Source | Frequency | API |
|--------|--------|-----------|-----|
| Revenue, EPS, Net Income | yfinance.info, quarterly_financials | Quarterly | Yahoo Finance |
| Margins | yfinance.info | Trailing 12M | Yahoo Finance |
| FCF, OCF | yfinance.quarterly_cashflow | Quarterly | Yahoo Finance |
| Market Cap | yfinance.info | Real-time | Yahoo Finance |
| Debt, Equity | yfinance.balance_sheet | Quarterly | Yahoo Finance |

## Implementation Strategy

### Phase 2a: Core Infrastructure (similar to price features)
1. **New helper function**: `get_fundamental_metrics(symbol, lookback_quarters=4)`
   - Fetch from yfinance
   - Cache locally to avoid rate limits
   - Return dict: `{metric_name: value, ...}`

2. **DB schema extension**: Add fundamentals table
   ```sql
   CREATE TABLE fundamental_metrics (
       id INTEGER PRIMARY KEY,
       symbol TEXT UNIQUE,
       timestamp DATETIME,
       rev_growth REAL,          -- Revenue growth %
       eps_growth REAL,          -- EPS growth %
       gross_margin REAL,        -- Gross margin %
       fcf_yield REAL,           -- FCF yield %
       de_ratio REAL,            -- Debt-to-equity
       roe REAL,                 -- Return on equity %
       updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
   )
   ```
   - Separate table (not optimization_runs) for symbol-specific metrics
   - Enables caching without polluting optimization params

3. **Extract and cache**: `cache_fundamental_metrics(symbols, force_refresh=False)`
   - Batch fetch from yfinance
   - Store in SQLite fundamentals table
   - Reuse cached values within TTL (24h default)

### Phase 2b: Optimization Parameters (extending price feature pattern)
1. **Extend optimization_runs columns**:
   ```sql
   ALTER TABLE optimization_runs ADD COLUMN (
       a11, a12, a13, a14, a15 REAL,    -- Weights for 5 fundamental features
       th11, th12, th13, th14, th15 REAL, -- Thresholds for comparisons
       use_fundamentals INTEGER DEFAULT 0  -- Master flag
   )
   ```

2. **Extend BEST_PARAM_EXTRAS** in qsi.py:
   ```python
   BEST_PARAM_EXTRAS = {
       'Technology_Large': {
           # Price features (already implemented)
           'use_price_slope': 0,
           'use_price_acc': 0,
           'a_price_slope': 0.0,
           ...
           # NEW: Fundamentals
           'use_fundamentals': 0,
           'use_rev_growth': 0,
           'use_eps_growth': 0,
           'use_fcf_yield': 0,
           'use_de_ratio': 0,
           'a_rev_growth': 0.0,
           'a_eps_growth': 0.0,
           'a_fcf_yield': 0.0,
           'a_de_ratio': 0.0,
           'th_rev_growth': 0.0,
           'th_eps_growth': 0.0,
           'th_fcf_yield': 0.0,
           'th_de_ratio': 0.0,
       }
   }
   ```

3. **Update extract_best_parameters()**: Always read fundamentals columns with safe defaults

### Phase 2c: Signal Integration
1. **Extend get_trading_signal()** signature:
   ```python
   def get_trading_signal(
       prices, volumes, domaine,
       domain_coeffs=None, domain_thresholds=None,
       cap_range=None, price_extras=None,
       fundamental_extras=None,  # NEW
       symbol=None,  # Already exists; will use to fetch fundamentals
       ...
   )
   ```

2. **Fundamental contribution to score**:
   ```python
   if fundamental_extras and fundamental_extras.get('use_fundamentals'):
       fundamentals = fundamental_extras  # Or fetch if not provided
       
       if fundamental_extras.get('use_rev_growth'):
           rev_growth = fundamentals.get('rev_growth', 0)
           if rev_growth > fundamental_extras['th_rev_growth']:
               score += fundamental_extras['a_rev_growth']
       
       # Similar for eps_growth, fcf_yield, de_ratio, etc.
   ```

3. **Lazy fundamentals fetch** (if symbol provided but fundamentals not in extras):
   ```python
   if symbol and fundamental_extras is None:
       fundamental_extras = fetch_fundamentals_cached(symbol)
   ```

### Phase 2d: Optimizer Enhancement
1. **HybridOptimizer enhancement**:
   ```python
   def __init__(self, ..., use_price_features=False, use_fundamental_features=False):
       if use_fundamental_features:
           # Extend bounds by 8 more params:
           # Indices 24-25: use_rev_growth, use_eps_growth (0/1 flags)
           # Indices 26-27: use_fcf_yield, use_de_ratio
           # Indices 28-29: a_rev_growth, a_eps_growth weights
           # Indices 30-31: a_fcf_yield, a_de_ratio weights
           # Indices 32-35: thresholds th11, th12, th13, th14
   ```

2. **Backtest integration**:
   - Pass `fundamental_extras` through backtest pipeline
   - `backtest_signals_with_events(..., fundamental_extras={...})`
   - Forward to `get_trading_signal(fundamental_extras=...)`

3. **Parameter bounds**:
   ```python
   fundamental_bounds = [
       (0, 1), (0, 1), (0, 1), (0, 1),  # 4 feature flags (24-27)
       (0, 3), (0, 3), (0, 3), (0, 3),  # 4 weights (28-31)
       (-20, 20), (-20, 20), (-2, 2), (-1, 1)  # 4 thresholds (32-35)
   ]
   ```

### Phase 2e: Persistence
1. **save_optimization_results()** extended:
   - Accept `fundamental_extras` dict
   - Persist all 8 params (4 flags/weights + 4 thresholds) + master `use_fundamentals` flag
   - Defaults to zeros if not present

2. **Result summary** includes fundamentals:
   - Which fundamentals were optimized
   - Their weights and learned thresholds

## Implementation Sequence

```
┌─────────────────────────────────┐
│ Step 1: Fundamentals Extraction │ (3-4 hours)
├─────────────────────────────────┤
│ • get_fundamental_metrics()     │
│ • Cache in SQLite fundamentals  │
│ • Symbol-to-fundamental mapping │
└────────────┬────────────────────┘
             │
┌────────────v────────────────────┐
│ Step 2: Signal Integration       │ (2-3 hours)
├─────────────────────────────────┤
│ • Extend get_trading_signal()   │
│ • Add fundamental_extras param  │
│ • Score contribution logic      │
│ • Lazy fetch if symbol given    │
└────────────┬────────────────────┘
             │
┌────────────v────────────────────┐
│ Step 3: DB & Extraction Updates  │ (1-2 hours)
├─────────────────────────────────┤
│ • Schema migration (idempotent) │
│ • extract_best_parameters() ext │
│ • BEST_PARAM_EXTRAS population  │
└────────────┬────────────────────┘
             │
┌────────────v────────────────────┐
│ Step 4: Optimizer Integration    │ (3-4 hours)
├─────────────────────────────────┤
│ • HybridOptimizer flag + bounds │
│ • evaluate_config() extraction  │
│ • backtest_signals_with_events  │
│ • Parameter persistence         │
└────────────┬────────────────────┘
             │
┌────────────v────────────────────┐
│ Step 5: Testing & Validation     │ (2-3 hours)
├─────────────────────────────────┤
│ • Unit tests (signal w/ fund)   │
│ • Integration (optimizer bounds)│
│ • Sample segment optimization   │
│ • Backward compat checks        │
└─────────────────────────────────┘

Total Estimated: 11-16 hours
```

## Design Principles (from Price Features)

1. **Always extract fundamentals** (like price params) → always populate BEST_PARAM_EXTRAS with safe defaults (zeros)
2. **Separate fundamentals_extras dict** → keep return shape stable, don't break existing code
3. **Opt-in via feature flags** → don't affect existing optimizations unless explicitly enabled
4. **Lazy fetch with caching** → avoid rate limits; reuse within TTL
5. **Safe defaults** → fundamentals default to neutral (flags=0, weights/thresholds=0) if not optimized
6. **DB schema migration** → idempotent; add columns if missing

## Key Differences from Price Features

| Aspect | Price Features | Fundamentals |
|--------|----------------|--------------|
| **Data source** | Computed from price/volume | External (yfinance, cached) |
| **Availability** | Always (every bar) | Lower frequency (quarterly updates) |
| **Storage** | Computed on-demand | Cached in separate table |
| **Per-symbol** | No (time-series) | Yes (static per update) |
| **Optimizer dimensions** | +6 params | +8 params |
| **Caching** | Derivatives (in-memory) | Fundamentals (SQLite + 24h TTL) |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **yfinance rate limits** | Cache fundamentals in SQLite; batch fetch; 24h TTL |
| **Missing data** | Safe defaults (0.0); skip symbols with None values |
| **Stale fundamentals** | Update cache daily; detect stale with timestamp |
| **Optimization bloat** | Master `use_fundamentals` flag; disable by default |
| **Score explosion** | Normalize fundamental weights (same bounds as price: 0-3); clip thresholds |
| **Backward compat** | Defaults ensure neutral effect; no behavior change unless flag enabled |

## Testing Strategy

### Unit Tests
```python
# Test 1: get_fundamental_metrics() returns expected dict
# Test 2: Caching: fetch, verify cache, fetch again (should be cached)
# Test 3: get_trading_signal(fundamental_extras=...) adjusts score
# Test 4: BEST_PARAM_EXTRAS always has fundamental defaults
```

### Integration Tests
```python
# Test 5: HybridOptimizer(use_fundamental_features=True) → 26 params
# Test 6: Optimizer evaluates extended vector without error
# Test 7: save_optimization_results persists fundamental params
# Test 8: extract_best_parameters reads and populates fundamentals
```

### Validation Tests
```python
# Test 9: Run optimization on Technology×Large WITH fundamentals
# Test 10: Compare baseline (no fund) vs. optimized (with fund)
# Test 11: Verify learned weights/thresholds are reasonable
# Test 12: Confirm no regressions in existing code paths
```

## Documentation Requirements

- Update `get_trading_signal()` docstring with fundamental_extras param
- Add examples to `BEST_PARAM_EXTRAS` showing fundamental keys
- Create unit test file: `test_fundamentals.py`
- Update main implementation summary with fundamentals section

## Success Criteria

- [x] Fundamentals extracted and cached without errors
- [x] Signal scoring integrates fundamentals (when enabled)
- [x] Optimizer can search 26-dimensional space (18 base + 6 price + 8 fundamental)
- [x] Backward compatible: default behavior unchanged
- [x] Parameters persist in DB correctly
- [x] Sample segment (Technology×Large) optimizes with fundamentals and shows measurable impact
- [x] No regressions in existing price-feature or baseline optimization

---

**Status**: Ready to begin Phase 2 implementation after user approval.
