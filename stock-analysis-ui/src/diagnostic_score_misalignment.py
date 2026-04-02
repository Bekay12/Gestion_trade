#!/usr/bin/env python3
"""
DIAGNOSTIC SCORE MISALIGNMENT
==============================

This script traces where score calculations diverge between:
1. Initial analysis (process_symbol) calculation
2. Display in table (update_results_table)  
3. Chart history (compute_score_series)

Critical issues to monitor:
- Threshold synchronization (seuil_achat/vente)
- Sector normalization (domaine raw vs normalized)
- Cap_range assignment divergence
- Analysis ID flow and stale results
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEBUG_LOG_FILE = Path(__file__).parent / "logs" / "score_debug.log"
DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

ANALYSIS_SNAPSHOTS = {}  # In-memory cache of analysis snapshots

# ============================================================================
# LOGGER
# ============================================================================

class ScoreDiagnosticLogger:
    """Central logger for score calculation diagnostics"""
    
    def __init__(self, log_file=DEBUG_LOG_FILE):
        self.log_file = log_file
        self.events = []
        
    def log_event(self, event_type, symbol, data):
        """Log a diagnostic event with full context"""
        timestamp = datetime.now().isoformat()
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'symbol': symbol,
            'data': data
        }
        self.events.append(event)
        self._write_to_file(event)
        
    def _write_to_file(self, event):
        """Append event to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"⚠️ Error writing log: {e}")
    
    def print_summary(self):
        """Print diagnostics summary"""
        print("\n" + "="*80)
        print("SCORE DIAGNOSTIC SUMMARY")
        print("="*80)
        
        # Group events by type
        by_type = {}
        for event in self.events:
            t = event['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(event)
        
        for event_type, events in sorted(by_type.items()):
            print(f"\n[{event_type}] - {len(events)} events")
            
            # Show first few and last few
            for event in events[:2] + (['...'] if len(events) > 4 else []) + events[-2:]:
                if isinstance(event, str):
                    print(f"  {event}")
                else:
                    sym = event.get('symbol', '?')
                    data = event.get('data', {})
                    print(f"  {sym}: {json.dumps(data, default=str)}")

# Global logger instance
logger = ScoreDiagnosticLogger()

# ============================================================================
# PATCHING FUNCTIONS
# ============================================================================

def patch_process_symbol(qsi_module):
    """Patch qsi.process_symbol() to log score calculation details"""
    original_process_symbol = qsi_module.process_symbol
    
    def traced_process_symbol(symbol, mes_symbols=None, **kwargs):
        result = original_process_symbol(symbol, mes_symbols, **kwargs)
        
        # Log calculation details
        logger.log_event('SCORE_CALCULATED_IN_ANALYSIS', symbol, {
            'score': result.get('Score'),
            'signal': result.get('Signal'),
            'domaine': result.get('Domaine'),
            'cap_range': result.get('CapRange'),
            'seuil_achat': result.get('_seuil_achat_used'),  # If available
            'seuil_vente': result.get('_seuil_vente_used'),  # If available
            'derivatives': {
                'dprice': result.get('dPrice'),
                'dmacd': result.get('dMACD'),
                'drsi': result.get('dRSI'),
                'dvol': result.get('dVolRel')
            }
        })
        
        return result
    
    qsi_module.process_symbol = traced_process_symbol

def patch_get_trading_signal(qsi_module):
    """Patch qsi.get_trading_signal() to log threshold and parameter selection"""
    original_get_trading_signal = qsi_module.get_trading_signal
    
    def traced_get_trading_signal(prices, volumes, domaine='', symbol='', cap_range='', **kwargs):
        result = original_get_trading_signal(prices, volumes, domaine=domaine, symbol=symbol, cap_range=cap_range, **kwargs)
        
        # Extract result components
        sig, last_price, trend, last_rsi, volume_mean, score = result[:6]
        derivatives = result[6] if len(result) > 6 else {}
        
        # Log calculation details
        seuil_achat_arg = kwargs.get('seuil_achat')
        seuil_vente_arg = kwargs.get('seuil_vente')
        
        logger.log_event('SCORE_CALCULATED_IN_SIGNAL', symbol, {
            'score': score,
            'signal': sig,
            'domaine_param': domaine,
            'cap_range_param': cap_range,
            'seuil_achat_arg': seuil_achat_arg,
            'seuil_vente_arg': seuil_vente_arg,
            'kwargs_keys': list(kwargs.keys())
        })
        
        return result
    
    qsi_module.get_trading_signal = traced_get_trading_signal

def patch_update_results_table(main_window_class):
    """Patch MainWindow.update_results_table() to log display calculations"""
    original_update_results_table = main_window_class.update_results_table
    
    def traced_update_results_table(self):
        """Modified update_results_table with diagnostics"""
        if not hasattr(self, 'current_results'):
            return
        
        # Log which results are being displayed
        logger.log_event('TABLE_UPDATE_START', 'BATCH', {
            'num_results': len(self.current_results),
            'has_backtest_map': hasattr(self, 'backtest_map'),
            'has_best_parameters': hasattr(self, 'best_parameters')
        })
        
        # Log details for each symbol
        for result in self.current_results[:5]:  # First 5 for brevity
            sym = result.get('Symbole')
            logger.log_event('TABLE_DISPLAY_SYMBOL', sym, {
                'score_from_results': result.get('Score'),
                'domaine': result.get('Domaine'),
                'cap_range': result.get('CapRange'),
                'signal': result.get('Signal'),
                'fiabilite': result.get('Fiabilite')
            })
        
        # Call original method
        original_update_results_table(self)
        
        logger.log_event('TABLE_UPDATE_END', 'BATCH', {
            'rows_displayed': self.merged_table.rowCount()
        })
    
    main_window_class.update_results_table = traced_update_results_table

def patch_compute_score_series(main_window_class):
    """Patch MainWindow._compute_score_series() to log historical calculations"""
    if hasattr(main_window_class, '_compute_score_series'):
        original_compute_score_series = main_window_class._compute_score_series
        
        def traced_compute_score_series(self, symbol, prices, volumes):
            """Modified _compute_score_series with diagnostics"""
            logger.log_event('CHART_SCORE_SERIES_START', symbol, {
                'num_candles': len(prices),
                'date_range': f"{prices.index[0]} to {prices.index[-1]}" if len(prices) > 0 else 'N/A'
            })
            
            result = original_compute_score_series(self, symbol, prices, volumes)
            
            # result should be a list of tuples (date, score, signal, ...)
            if result:
                logger.log_event('CHART_SCORE_SERIES_END', symbol, {
                    'num_points': len(result),
                    'first_score': result[0][1] if result else None,
                    'last_score': result[-1][1] if result else None,
                    'max_score': max((x[1] for x in result), default=None),
                    'min_score': min((x[1] for x in result), default=None)
                })
            
            return result
        
        main_window_class._compute_score_series = traced_compute_score_series

# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_thresholds(qsi_module, symbol, domaine, cap_range):
    """Compare thresholds used in analysis vs table display"""
    print(f"\n🔍 THRESHOLD COMPARISON for {symbol}")
    print(f"   Domain: {domaine}, Cap Range: {cap_range}")
    
    # Simulate threshold lookup as in update_results_table
    try:
        best_params = qsi_module.extract_best_parameters()
        
        # Method used in table display
        seuil_achat_table = 4.2  # Default
        seuil_vente_table = -0.5
        
        param_key = None
        if cap_range and cap_range != "Unknown":
            test_key = f"{domaine}_{cap_range}"
            if test_key in best_params:
                param_key = test_key
        if not param_key and domaine in best_params:
            param_key = domaine
        
        if param_key and param_key in best_params:
            params = best_params[param_key]
            if len(params) > 2 and params[2]:
                globals_th = params[2]
                if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                    seuil_achat_table = float(globals_th[0])
                    seuil_vente_table = float(globals_th[1])
        
        print(f"   ✅ Thresholds from table display: achat={seuil_achat_table:.2f}, vente={seuil_vente_table:.2f}")
        print(f"   ✅ Param key used: {param_key}")
        
        # Check what might have been used in analysis
        # (This assumes best_params are loaded at same time)
        
        return seuil_achat_table, seuil_vente_table
        
    except Exception as e:
        print(f"   ❌ Error loading thresholds: {e}")
        return None, None

def check_sector_normalization(symbol):
    """Check sector normalization consistency"""
    print(f"\n🔍 SECTOR NORMALIZATION CHECK for {symbol}")
    
    try:
        import yfinance as yf
        from sector_normalizer import normalize_sector
        
        # Get raw sector from yfinance
        info = yf.Ticker(symbol).info
        domaine_raw = info.get("sector", "Inconnu")
        
        # Normalize
        domaine_normalized = normalize_sector(domaine_raw)
        
        print(f"   Raw sector: '{domaine_raw}'")
        print(f"   Normalized: '{domaine_normalized}'")
        
        if domaine_raw != domaine_normalized:
            print(f"   ⚠️ DIVERGENCE DETECTED")
        else:
            print(f"   ✅ Consistent")
        
        return domaine_raw, domaine_normalized
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None

# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def run_diagnostic(symbols=['AAPL', 'MSFT', 'TSLA'], verbose=True):
    """Run full diagnostic on score calculation"""
    
    print("\n" + "="*80)
    print("STARTING SCORE MISALIGNMENT DIAGNOSTIC")
    print("="*80)
    print(f"Test symbols: {symbols}")
    print(f"Log file: {DEBUG_LOG_FILE}")
    
    try:
        # Import modules
        import qsi
        from stock_analysis_ui.src.ui.main_window import MainWindow
        
        # Apply patches
        print("\n✅ Patching modules...")
        patch_process_symbol(qsi)
        patch_get_trading_signal(qsi)
        patch_update_results_table(MainWindow)
        
        # Run analysis on test symbols
        print("\n📊 Running analysis...")
        for symbol in symbols:
            print(f"\n  Processing {symbol}...")
            
            # Compare thresholds
            try:
                from sector_normalizer import normalize_sector
                info = __import__('yfinance').Ticker(symbol).info
                domaine = normalize_sector(info.get("sector", "Technology"))
                cap_range = qsi.get_cap_range_for_symbol(symbol)
                
                compare_thresholds(qsi, symbol, domaine, cap_range)
                check_sector_normalization(symbol)
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
        
        # Print summary
        print("\n" + "="*80)
        logger.print_summary()
        
        # Save snapshots for later comparison
        with open(DEBUG_LOG_FILE.parent / 'diagnostic_summary.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'symbols_tested': symbols,
                'events': logger.events
            }, f, indent=2, default=str)
        
        print(f"\n✅ Diagnostic complete. Logs saved to: {DEBUG_LOG_FILE}")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Score Misalignment Diagnostic')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'TSLA'],
                        help='Symbols to test')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    run_diagnostic(symbols=args.symbols, verbose=args.verbose)
