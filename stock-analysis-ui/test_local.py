#!/usr/bin/env python
"""
Test rapide de l'API en local pour debug
"""

import requests
import json

API_BASE = "http://localhost:5000"

def test_lists():
    """Test l'endpoint lists"""
    try:
        res = requests.get(f"{API_BASE}/api/lists", timeout=5)
        print(f"✅ /api/lists - Status: {res.status_code}")
        if res.ok:
            data = res.json()
            print(f"   Popular: {len(data.get('popular', []))} symboles")
            print(f"   Personal: {len(data.get('personal', []))} symboles")
            print(f"   Optimization: {len(data.get('optimization', []))} symboles")
        return res.ok
    except Exception as e:
        print(f"❌ /api/lists - Error: {e}")
        return False

def test_analyze():
    """Test l'analyse d'un symbole"""
    try:
        payload = {"symbol": "AAPL", "period": "1mo", "include_backtest": False}
        res = requests.post(
            f"{API_BASE}/api/analyze", 
            json=payload, 
            timeout=30
        )
        print(f"✅ /api/analyze - Status: {res.status_code}")
        if res.ok:
            data = res.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Signals: {len(data.get('signals', []))}")
            if data.get('signals'):
                sig = data['signals'][0]
                print(f"   Signal: {sig.get('signal')} - Fiabilité: {sig.get('fiabilite')}%")
        return res.ok
    except Exception as e:
        print(f"❌ /api/analyze - Error: {e}")
        return False

def test_stats():
    """Test les stats"""
    try:
        res = requests.get(f"{API_BASE}/api/stats", timeout=5)
        print(f"✅ /api/stats - Status: {res.status_code}")
        if res.ok:
            data = res.json()
            stats = data.get('stats', data)
            print(f"   Total signals: {stats.get('total_signals', 0)}")
        return res.ok
    except Exception as e:
        print(f"❌ /api/stats - Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🧪 TEST LOCAL DE L'API\n")
    print("=" * 50)
    
    test_lists()
    print()
    test_stats()
    print()
    test_analyze()
    
    print("\n" + "=" * 50)
    print("✅ Tests terminés")
