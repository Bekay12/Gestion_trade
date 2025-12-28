# Quick Reference: Daily Reliability Gating

## TL;DR
Enable per-action reliability gating with one flag:
```bash
python tests/validate_workflow_realistic.py --gate-by-daily-reliability --reliability 50 --year 2024
```

---

## What It Does

**Without gating** (original):
```
Buy whenever signal == "ACHAT" (if symbol eligible from training)
Sell whenever signal == "VENTE"
```

**With gating** (new):
```
On each ACHAT signal:
  1. Compute trailing reliability (last 9 months of past data)
  2. IF trailing_reliability >= threshold:
     → BUY (open position)
     ELSE:
     → SKIP (no position, continue to next symbol)

On each VENTE signal:
  → SELL (close position, gate not applied to sell side)
```

---

## CLI Flags

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--gate-by-daily-reliability` | bool | False | Enable gating |
| `--trailing-months` | int | 9 | Reliability window (months) |
| `--recalc-reliability-every` | int | 5 | Cache refresh (business days) |
| `--reliability` | float | 60.0 | Min reliability threshold (%) |

---

## Examples

### Fast Backtest (No Gating)
```bash
python tests/validate_workflow_realistic.py --reliability 50 --year 2024 --use-business-days
```
✅ Fast
✅ Matches original behavior

### With Gating (Defaults)
```bash
python tests/validate_workflow_realistic.py --gate-by-daily-reliability --reliability 50 --year 2024
```
✅ 9-month trailing window
✅ Recompute every 5 business days
✅ Gate buys at 50% threshold

### Aggressive Gating (High Threshold)
```bash
python tests/validate_workflow_realistic.py \
  --gate-by-daily-reliability \
  --reliability 70 \
  --trailing-months 12 \
  --year 2024
```
✅ Only buy if 12-month reliability ≥ 70%
✅ Fewer trades, higher quality

### Responsive Gating (Short Window)
```bash
python tests/validate_workflow_realistic.py \
  --gate-by-daily-reliability \
  --reliability 45 \
  --trailing-months 6 \
  --recalc-reliability-every 1 \
  --year 2024
```
✅ 6-month window (recent performance matters more)
✅ Daily recompute (always fresh)
✅ Lower threshold (more trades)

---

## How Reliability is Computed

1. **Get yesterday's prices/volumes** (no lookahead)
2. **Look back 9 months** (or `--trailing-months`)
3. **Run walk-forward backtest**:
   - Day-by-day: simulate signals on available data
   - BUY on "ACHAT", hold ≥14 days
   - SELL on "VENTE" (if held long enough)
   - Count: winners / total trades
4. **Return success rate %**

Example:
```
Today: 2024-05-15
Yesterday: 2024-05-14 (use data up to this date)
Trailing window: 2023-08-14 to 2024-05-14 (9 months)

Historical walk-forward:
  - 5 winning trades
  - 8 total trades
  - Reliability: 5/8 = 62.5%

Gate check: 62.5% >= 50% threshold? YES → BUY
```

---

## Caching Behavior

```python
# First time we see AAPL (2024-05-15):
cache.get('AAPL')  # None → COMPUTE reliability → STORE in cache

# Same day, another signal (2024-05-15):
cache.get('AAPL')  # Found! Use cached value (instant)

# 5 business days later (2024-05-22):
(2024-05-22 - 2024-05-15).days = 7 >= recalc_every (5)
# Cache is stale → RECOMPUTE reliability → UPDATE cache
```

Performance:
- **Cached hit**: < 1ms
- **Cache miss**: 5-10 seconds (backtest)

---

## Output Example

```
🔎 Simulating from 2024-01-01 to 2024-12-31 for 629 symbols (list=popular)
   Filters: reliability >= 50.0%, min_hold_days=14, volume_min=20000
   Daily Gating: trailing_months=9, recalc_every=5 days
📥 Downloading data for 629 symbols over 24 months...
✅ Data downloaded: 625 symbols with data
🔎 Computing training reliability per symbol...
Reliability: 100%|████████| 625/625 [1:23<00:00, 8.23it/s]
✅ Eligible symbols: 48/625 (threshold=50.0%)
📊 Walk-forward simulation
Simulation: 100%|████████| 252/252 [0:34<00:00, 7.37it/s]
✅ Final Results:
  Total Trades: 23
  Winning Trades: 14
  Success Rate: 60.9%
  P&L: +$3,456.78 (on 23 trades × $1000 each)
```

---

## Troubleshooting

### Q: "Eligible symbols: 0/625"
**A:** Training reliability too high or data quality issue
- Lower `--reliability` threshold
- Check volume data: `--volume-min 10000`

### Q: "No trades executed"
**A:** No symbols passed the gate
- Lower `--reliability` threshold further
- Use shorter `--trailing-months` (6 instead of 9)
- Check signal generation: run without gating first

### Q: "Very slow on large portfolio"
**A:** Too many recomputes
- Increase `--recalc-reliability-every` to 10 or 20
- Use `--use-business-days` (faster date iteration)

---

## Implementation Details

**File**: `tests/validate_workflow_realistic.py`

Key functions:
- `compute_daily_reliability()` (line 53): Compute trailing reliability
- `walk_forward_simulation()` (line 201): Main simulation loop with gating
- `parse_args()` (line 472): CLI argument parser
- `main()` (line 490): Entry point

Cache structure:
```python
reliability_cache = {
    'AAPL': (pd.Timestamp('2024-05-15'), 62.5),  # (cached_date, rate)
    'MSFT': (pd.Timestamp('2024-05-20'), 58.0),
    ...
}
```

---

## Key Principles

1. ✅ **No Lookahead**: Uses only past data
2. ✅ **Min-Hold**: Enforced in both training and simulation
3. ✅ **Per-Action**: Gate checked at each BUY signal
4. ✅ **Configurable**: Threshold, window, recalc frequency all CLI flags
5. ✅ **Cached**: Avoids redundant backtest calculations
6. ✅ **Transparent**: Shows gating parameters in output

---

## One-Liner Experiments

```bash
# Baseline (no gating)
python tests/validate_workflow_realistic.py --reliability 50 --year 2024 --use-business-days

# With gating (aggressive)
python tests/validate_workflow_realistic.py --gate-by-daily-reliability --reliability 70 --year 2024 --use-business-days

# With gating (responsive)
python tests/validate_workflow_realistic.py --gate-by-daily-reliability --reliability 40 --trailing-months 6 --recalc-reliability-every 1 --year 2024 --use-business-days
```

Compare results to see impact of gating on trade quantity vs. quality.
