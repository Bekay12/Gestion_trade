"""Diagnostic pour comprendre pourquoi la simulation échoue"""
import os
import sys
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

import pandas as pd
from qsi import extract_best_parameters, get_trading_signal, download_stock_data, BEST_PARAM_EXTRAS
from symbol_manager import get_symbols_by_list_type
import sqlite3

print("=" * 80)
print("DIAGNOSTIC DE LA SIMULATION")
print("=" * 80)

# 1. Vérifier les paramètres chargés
print("\n📊 1. PARAMÈTRES CHARGÉS:")
best_params = extract_best_parameters()
print(f"   Nombre de secteurs: {len(best_params)}")

# Montrer quelques exemples
for domain, params in list(best_params.items())[:5]:
    if len(params) == 5:
        coeffs, thresholds, globals_th, gain, extras = params
    else:
        coeffs, thresholds, globals_th, gain = params
        extras = {}
    print(f"   {domain}:")
    print(f"      Coeffs: {coeffs[:4]}...")
    print(f"      Seuil achat: {globals_th[0]:.2f}, Seuil vente: {globals_th[1]:.2f}")
    print(f"      Gain historique: {gain:.2f}")

# 2. Télécharger quelques symboles de test
print("\n📊 2. TEST SUR QUELQUES SYMBOLES:")
symbols_to_test = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOG']
stock_data = download_stock_data(symbols_to_test, period='1y')

print(f"   Symboles téléchargés: {len(stock_data)}")

# 3. Tester get_trading_signal pour chaque symbole
print("\n📊 3. SIGNAUX GÉNÉRÉS:")
for symbol in symbols_to_test:
    if symbol not in stock_data:
        print(f"   {symbol}: Pas de données")
        continue
    
    data = stock_data[symbol]
    prices = data['Close']
    volumes = data['Volume']
    
    # Récupérer le secteur
    try:
        conn = sqlite3.connect('stock_analysis.db')
        cur = conn.cursor()
        cur.execute("SELECT sector, market_cap_range FROM symbols WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        conn.close()
        sector = row[0] if row else "Technology"
        cap_range = row[1] if row and row[1] else "Large"
    except:
        sector = "Technology"
        cap_range = "Large"
    
    # Trouver les paramètres appropriés
    param_key = f"{sector}_{cap_range}" if f"{sector}_{cap_range}" in best_params else sector
    if param_key not in best_params:
        print(f"   {symbol}: Pas de paramètres pour {param_key}")
        continue
    
    params = best_params[param_key]
    if len(params) == 5:
        coeffs, thresholds, globals_th, gain, extras = params
    else:
        coeffs, thresholds, globals_th, gain = params
        extras = {}
    
    seuil_achat = globals_th[0]
    seuil_vente = globals_th[1]
    
    # Appeler get_trading_signal
    try:
        signal, last_price, trend, rsi, vol_mean, score, derivatives = get_trading_signal(
            prices, volumes, sector,
            domain_coeffs={sector: coeffs},
            domain_thresholds={sector: thresholds},
            return_derivatives=True,
            symbol=symbol,
            cap_range=cap_range,
            seuil_achat=seuil_achat,
            seuil_vente=seuil_vente,
            price_extras=extras
        )
        
        print(f"\n   📈 {symbol} ({sector}/{cap_range}):")
        print(f"      Signal: {signal}")
        print(f"      Score: {score:.2f} (seuil_achat={seuil_achat:.2f}, seuil_vente={seuil_vente:.2f})")
        print(f"      Prix: {last_price:.2f}, RSI: {rsi:.1f}, Tendance: {'Hausse' if trend else 'Baisse'}")
        
        # Diagnostic: pourquoi ce signal?
        if signal == "NEUTRE":
            print(f"      ⚠️ NEUTRE car: Score {score:.2f} < seuil_achat {seuil_achat:.2f} ET > seuil_vente {seuil_vente:.2f}")
        elif signal == "ACHAT":
            print(f"      ✅ ACHAT car: Score {score:.2f} >= seuil_achat {seuil_achat:.2f}")
        else:
            print(f"      🔴 VENTE car: Score {score:.2f} <= seuil_vente {seuil_vente:.2f}")
            
    except Exception as e:
        print(f"   {symbol}: Erreur - {e}")
        import traceback
        traceback.print_exc()

# 4. Analyser la distribution des scores
print("\n📊 4. ANALYSE DES SEUILS:")
print("   Distribution des seuils d'achat optimisés:")
achat_thresholds = [params[2][0] for params in best_params.values() if len(params) >= 3]
vente_thresholds = [params[2][1] for params in best_params.values() if len(params) >= 3]
print(f"      Min: {min(achat_thresholds):.2f}, Max: {max(achat_thresholds):.2f}, Moy: {sum(achat_thresholds)/len(achat_thresholds):.2f}")
print("   Distribution des seuils de vente optimisés:")
print(f"      Min: {min(vente_thresholds):.2f}, Max: {max(vente_thresholds):.2f}, Moy: {sum(vente_thresholds)/len(vente_thresholds):.2f}")

# 5. Tester avec des seuils par défaut
print("\n📊 5. TEST AVEC SEUILS PAR DÉFAUT (4.2 / -0.5):")
for symbol in symbols_to_test[:3]:
    if symbol not in stock_data:
        continue
    
    data = stock_data[symbol]
    prices = data['Close']
    volumes = data['Volume']
    
    try:
        signal, last_price, trend, rsi, vol_mean, score, derivatives = get_trading_signal(
            prices, volumes, "Technology",
            return_derivatives=True,
            symbol=symbol,
            seuil_achat=4.2,
            seuil_vente=-0.5
        )
        print(f"   {symbol}: Signal={signal}, Score={score:.2f}")
    except Exception as e:
        print(f"   {symbol}: Erreur - {e}")

print("\n" + "=" * 80)
print("FIN DU DIAGNOSTIC")
print("=" * 80)
