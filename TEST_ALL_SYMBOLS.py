#!/usr/bin/env python3
"""
Test TOUS les symboles: cherche les divergences cap_range/score
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

from qsi import download_stock_data, get_trading_signal, get_cap_range_for_symbol, extract_best_parameters, period
import yfinance as yf
from sector_normalizer import normalize_sector

# Charger les symboles à tester
def get_test_symbols():
    """Charger les symboles depuis les fichiers"""
    symbols = set()
    
    # Essayer plusieurs sources
    files = [
        'mes_symbols.txt',
        'popular_symbols.txt',
        'sp500_symbols.txt',
        'stock-analysis-ui/src/mes_symbols.txt',
        'stock-analysis-ui/src/popular_symbols.txt',
    ]
    
    for f in files:
        if os.path.exists(f):
            try:
                with open(f, 'r') as file:
                    for line in file:
                        sym = line.strip().upper()
                        if sym:
                            symbols.add(sym)
            except Exception:
                pass
    
    return sorted(list(symbols))[:20]  # Limiter à 20 pour tester


def check_symbol_cap_range(symbol):
    """Vérifie le cap_range pour un symbole"""
    try:
        # Télécharger
        data_dict = download_stock_data([symbol], period=period)
        if symbol not in data_dict:
            return None
        
        prices = data_dict[symbol]['Close']
        volumes = data_dict[symbol]['Volume']
        
        if len(prices) < 50:
            return None
        
        # Déterminer cap_range initial
        cap_range_initial = get_cap_range_for_symbol(symbol)
        
        # Déterminer secteur
        try:
            info = yf.Ticker(symbol).info
            domaine = normalize_sector(info.get("sector", "Inconnu"))
        except:
            domaine = "Inconnu"
        
        # Extraire seuils avec cap_range
        best_params = extract_best_parameters()
        
        # Avec cap_range
        key_with_cap = f"{domaine}_{cap_range_initial}" if cap_range_initial and cap_range_initial != "Unknown" else None
        score_with_cap = None
        seuil_with_cap = None
        
        if key_with_cap and key_with_cap in best_params:
            sig_c, _, _, _, _, score_c, deriv_c = get_trading_signal(
                prices, volumes, domaine=domaine, cap_range=cap_range_initial, symbol=symbol,
                return_derivatives=True
            )
            score_with_cap = score_c
            seuil_with_cap = deriv_c.get('_seuil_achat_used', 4.2)
        
        # Sans cap_range
        sig_nc, _, _, _, _, score_nc, deriv_nc = get_trading_signal(
            prices, volumes, domaine=domaine, cap_range=None, symbol=symbol,
            return_derivatives=True
        )
        score_without_cap = score_nc
        seuil_without_cap = deriv_nc.get('_seuil_achat_used', 4.2)
        
        return {
            'symbol': symbol,
            'domaine': domaine,
            'cap_range': cap_range_initial,
            'score_with_cap': score_with_cap,
            'score_without_cap': score_without_cap,
            'seuil_with_cap': seuil_with_cap,
            'seuil_without_cap': seuil_without_cap,
            'divergent': (score_with_cap is not None) and abs(score_with_cap - score_without_cap) > 0.5,
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


# Test
symbols = get_test_symbols()
if not symbols:
    print("❌ Aucun symbole trouvé à tester")
    sys.exit(1)

print(f"🔎 Test de {len(symbols)} symboles pour divergences cap_range...")
print(f"{'='*120}")

divergent_count = 0
results = []

for sym in symbols:
    result = check_symbol_cap_range(sym)
    if result is None:
        continue
    
    results.append(result)
    
    if 'error' in result:
        status = f"❌ {result.get('error', 'Unknown error')}"
    else:
        if result['divergent']:
            divergent_count += 1
            status = f"⚠️  DIVERGENT"
        else:
            status = f"✅"
    
    print(f"{status} {result['symbol']:6} | Cap={result.get('cap_range', 'N/A'):6} | "
          f"Domaine={result.get('domaine', 'N/A'):20} | "
          f"Score(+cap)={result.get('score_with_cap', 'N/A'):>7} | "
          f"Score(-cap)={result.get('score_without_cap', 'N/A'):>7}")

print(f"\n{'='*120}")
print(f"📊 Résumé: {len(results)} testés, {divergent_count} avec divergence cap_range")

# Afficher les divergences
if divergent_count > 0:
    print(f"\n{'='*120}")
    print("SYMBOLES AVEC DIVERGENCE:")
    for r in results:
        if r.get('divergent'):
            print(f"\n{r['symbol']} ({r['domaine']}):")
            print(f"  Cap_range initial: {r['cap_range']}")
            print(f"  Score AVEC cap_range: {r['score_with_cap']:.3f}")
            print(f"  Score SANS cap_range: {r['score_without_cap']:.3f}")
            print(f"  Différence: {abs(r['score_with_cap'] - r['score_without_cap']):.3f}")
            print(f"  Seuil avec cap: {r['seuil_with_cap']:.2f}")
            print(f"  Seuil sans cap: {r['seuil_without_cap']:.2f}")
