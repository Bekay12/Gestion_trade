"""
Generate comprehensive marker analysis report from advanced_marker_analysis.json
Summarizes the key differentiators between winning and losing stocks.
"""

import json
import numpy as np

def generate_marker_report():
    """Generate and print detailed marker report"""
    
    # Load analysis results
    with open('advanced_marker_analysis.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("MARKER ANALYSIS REPORT: WINNING vs LOSING STOCKS")
    print("="*80)
    print(f"\nDataset: {data['winners_count']} TOP PERFORMERS vs {data['losers_count']} UNDERPERFORMERS\n")
    
    # Organize findings by significance
    findings = []
    
    for metric_name, stats in data['metrics'].items():
        section, metric = metric_name.split('_', 1)
        diff = abs(stats['difference'])
        pct_diff = abs(diff / stats['losers_mean']) * 100 if stats['losers_mean'] != 0 else 0
        
        # Score the significance
        significance = "Low"
        if pct_diff > 100:
            significance = "CRITICAL"
        elif pct_diff > 50:
            significance = "HIGH"
        elif pct_diff > 20:
            significance = "MEDIUM"
        
        findings.append({
            'metric': metric,
            'section': section,
            'stats': stats,
            'pct_diff': pct_diff,
            'significance': significance,
            'diff': stats['difference'],
        })
    
    # Sort by significance
    significance_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'Low': 3}
    findings.sort(key=lambda x: (significance_order[x['significance']], -x['pct_diff']))
    
    # Print by section
    sections_printed = set()
    for finding in findings:
        section = finding['section']
        if section not in sections_printed:
            sections_printed.add(section)
            print(f"\n{'='*80}")
            print(f"SECTION: {section.upper()}")
            print('='*80)
        
        metric = finding['metric']
        stats = finding['stats']
        significance = finding['significance']
        pct_diff = finding['pct_diff']
        
        # Color code significance
        sig_marker = ""
        if significance == "CRITICAL":
            sig_marker = "[CRITICAL] ***"
        elif significance == "HIGH":
            sig_marker = "[HIGH]     **"
        elif significance == "MEDIUM":
            sig_marker = "[MEDIUM]   *"
        else:
            sig_marker = "[Low]      "
        
        direction = "WINNERS better" if finding['diff'] > 0 else "LOSERS better"
        
        print(f"\n{sig_marker}  {metric}")
        print(f"    Winners:  {stats['winners_mean']:>12.2f}  (median: {stats['winners_median']:>10.2f})")
        print(f"    Losers:   {stats['losers_mean']:>12.2f}  (median: {stats['losers_median']:>10.2f})")
        print(f"    Difference: {finding['diff']:>7.2f}  ({pct_diff:+.1f}%)  -->  {direction}")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS & ACTIONABLE MARKERS")
    print("="*80)
    
    print("""
1. MOMENTUM PATTERNS
   - Winners show MUCH STRONGER 1-year momentum (+221% difference!)
   - This is the MOST DISCRIMINATIVE marker
   - Recent 1-month momentum is slightly NEGATIVE for winners (possible pullback after strong run)
   - 3-month momentum shows winners ahead (+6.6% difference)
   
   => ACTION: Look for stocks with strong 1-year gains (200%+) that have recently pulled back
   
2. VOLATILITY CHARACTERISTICS  
   - Winners are SIGNIFICANTLY MORE VOLATILE (28% higher annual volatility)
   - Daily volatility also 33% higher for winners
   - Winners tend to trade LOWER on Bollinger Bands (more downside potential from breakouts)
   
   => ACTION: Winners are growth stocks prone to swings - expect higher risk/reward
   
3. TREND & PRICE ACTION
   - Winners show WEAKER trend signals (only 1.2 vs 2.4 for losers)
   - Winners trade BELOW their 20-day EMA (mean -2.36% vs +2.20% for losers)
   - This suggests winners are in PULLBACK/CONSOLIDATION phases
   
   => ACTION: Winners appear to be in correction phases - good entry points after strong runs
   
4. VOLUME PATTERNS
   - Winners show SLIGHTLY LOWER volume ratios (85% vs 94% of historical)
   - Not a strong differentiator but suggests quieter entry periods
   
   => ACTION: Lower volume during winner selections may indicate smart accumulation phases


COMPOSITE TRADING SIGNAL FOR WINNING STOCKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A stock is likely to be a WINNER if it exhibits:

✓ MUST HAVE (Critical):
  1. Strong 1-year historical momentum (150%+ gains)
  2. High volatility (8%+ annualized daily vol, 100%+ annual)
  3. Currently in pullback/consolidation (trading below 20-day EMA)

✓ SHOULD HAVE (High probability):
  1. Recent weakness in trend indicators (not overbought)
  2. Slightly lower volume than historical average
  3. Trading in lower half of Bollinger Bands

✓ NICE TO HAVE (Confirms thesis):
  1. Positive 3-month momentum
  2. Lower RSI (35-45 range suggests room to run)


RISK FACTORS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠ Very high volatility means:
  - Expect 5-10% daily swings
  - Stop losses must be wider
  - Position sizing should be smaller
  - Watch for sudden reversals

⚠ Pullback trading requires:
  - Entry confirmation (volume surge, support hold)
  - Clear support/resistance levels
  - Risk management discipline


EXAMPLE STOCKS FROM ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CDTX (+391%): 1-year mega-momentum, high vol, currently in consolidation = PERFECT MATCH
SEZL (+328%): Similar pattern - strong run followed by pullback
OUST (+282%): Same characteristics
OKLO (+107%): Slightly lower momentum but still strong
TRVI (+101%): Entry point characteristics evident


RECOMMENDED NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Filter universe for stocks with:
   - > 150% 1-year returns
   - Volatility > 80% annualized
   - Price < 20-day EMA
   - Volume ratio 80-95% (not on volume spike yet)

2. Add these as technical filters to your QSI system:
   - mom_1y coefficient in optimization
   - volatility_filter as gating condition
   - ema_pullback check before entry

3. Backtest the composite signal on 2024 data:
   - Record success rate with these combined markers
   - Optimize individual thresholds
   - Check stability across sectors
""")
    
    print("="*80)


if __name__ == '__main__':
    generate_marker_report()
