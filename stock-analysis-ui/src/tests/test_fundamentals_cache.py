"""
Unit tests for fundamentals extraction and caching.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fundamentals_cache import (
    get_fundamental_metrics,
    clear_fundamentals_cache,
    _ensure_fundamentals_table
)

def test_table_creation():
    """Test that fundamentals table is created correctly."""
    print("📋 Test 1: Table creation...")
    result = _ensure_fundamentals_table()
    assert result, "Failed to create/ensure fundamentals table"
    print("✅ Fundamentals table created successfully")

def test_fetch_real_symbol():
    """Test fetching real fundamentals for AAPL."""
    print("\n📋 Test 2: Fetch real symbol (AAPL)...")
    metrics = get_fundamental_metrics('AAPL', use_cache=False)
    
    assert metrics, "No metrics returned"
    assert 'rev_growth' in metrics, "Missing rev_growth key"
    assert 'eps_growth' in metrics, "Missing eps_growth key"
    assert 'gross_margin' in metrics, "Missing gross_margin key"
    assert 'roe' in metrics, "Missing roe key"
    assert 'de_ratio' in metrics, "Missing de_ratio key"
    
    print(f"✅ Fetched metrics for AAPL:")
    for key, val in metrics.items():
        if val is not None:
            print(f"   {key}: {val:.2f}")
        else:
            print(f"   {key}: None")

def test_cache_persistence():
    """Test that metrics are cached and retrieved from cache."""
    print("\n📋 Test 3: Cache persistence...")
    
    # Fetch fresh
    metrics_fresh = get_fundamental_metrics('MSFT', use_cache=False)
    print(f"   Fresh fetch: {len(metrics_fresh)} metrics retrieved")
    
    # Fetch from cache (should be instant)
    metrics_cached = get_fundamental_metrics('MSFT', use_cache=True)
    print(f"   Cache fetch: {len(metrics_cached)} metrics retrieved")
    
    # Compare (should be same values)
    for key in metrics_fresh:
        if metrics_fresh[key] is not None:
            assert metrics_cached[key] == metrics_fresh[key], f"Cache mismatch on {key}"
    
    print("✅ Cache persistence verified")

def test_multiple_symbols():
    """Test batch fetching for multiple symbols."""
    print("\n📋 Test 4: Multiple symbols...")
    symbols = ['GOOG', 'TSLA', 'META']
    results = {}
    
    for sym in symbols:
        metrics = get_fundamental_metrics(sym, use_cache=False)
        results[sym] = metrics
        non_none_count = sum(1 for v in metrics.values() if v is not None)
        print(f"   {sym}: {non_none_count}/8 metrics available")
    
    assert len(results) == 3, "Not all symbols fetched"
    print("✅ Multiple symbol batch fetch successful")

def test_cache_clear():
    """Test cache clearing functionality."""
    print("\n📋 Test 5: Cache clearing...")
    
    # Add some data to cache
    get_fundamental_metrics('AMD', use_cache=False)
    
    # Clear only AMD
    cleared = clear_fundamentals_cache('AMD')
    assert cleared > 0, "Failed to clear cache for AMD"
    print(f"   Cleared {cleared} entries for AMD")
    
    print("✅ Cache clearing works correctly")

if __name__ == '__main__':
    print("=" * 60)
    print("FUNDAMENTALS EXTRACTION & CACHING TEST SUITE")
    print("=" * 60)
    
    try:
        test_table_creation()
        test_fetch_real_symbol()
        test_cache_persistence()
        test_multiple_symbols()
        test_cache_clear()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
