#!/usr/bin/env python3
"""
Script diagnostic pour comparer exactement les scores de KIM vs AEP
et identifier où le décalage se produit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-analysis-ui/src'))

import numpy as np
import pandas as pd
import yfinance as yf
from qsi import download_stock_data, get_trading_signal, period
import ta
import json

def diagnose_symbol(symbol, sector="Technology"):
    """
    Diagnostic complet pour un symbole: télécharge, calcule, affiche tous les détails
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLET: {symbol}")
    print('='*80)
    
    try:
        # 1. Télécharger les données
        data_dict = download_stock_data([symbol], period=period)
        if symbol not in data_dict:
            print(f"❌ {symbol}: Pas de données trouvées")
            return None
        
        data = data_dict[symbol]
        prices = data['Close']
        volumes = data['Volume']
        
        if prices is None or prices.empty or len(prices) < 50:
            print(f"❌ {symbol}: Pas assez de données ({len(prices) if prices is not None else 0} rows)")
            return None
        
        print(f"\n✅ Données: {len(prices)} bars, dernière date: {prices.index[-1]}")
        print(f"   Prix (derniers 5): {prices.iloc[-5:].values}")
        print(f"   Volumes (derniers 5): {volumes.iloc[-5:].values}")
        
        # 2. Calculer les indicateurs
        last_close = float(prices.iloc[-1])
        print(f"\n   Dernier prix: {last_close}")
        
        # RSI
        rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
        last_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        print(f"   RSI (dernier): {last_rsi:.2f}, (prev): {prev_rsi:.2f}")
        
        # EMA
        ema20 = prices.ewm(span=20, adjust=False).mean()
        ema50 = prices.ewm(span=50, adjust=False).mean()
        ema200 = prices.ewm(span=200, adjust=False).mean()
        print(f"   EMA20: {float(ema20.iloc[-1]):.2f}, EMA50: {float(ema50.iloc[-1]):.2f}, EMA200: {float(ema200.iloc[-1]):.2f}")
        
        # MACD
        from qsi import calculate_macd
        macd, signal_line = calculate_macd(prices)
        last_macd = float(macd.iloc[-1])
        prev_macd = float(macd.iloc[-2])
        last_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])
        print(f"   MACD (dernier): {last_macd:.4f}, (prev): {prev_macd:.4f}")
        print(f"   Signal (dernier): {last_signal:.4f}, (prev): {prev_signal:.4f}")
        
        # Volume
        if len(volumes) >= 30:
            volume_mean = float(volumes.rolling(window=30).mean().iloc[-1])
        else:
            volume_mean = float(volumes.mean())
        current_volume = float(volumes.iloc[-1])
        print(f"   Volume moyen (30j): {volume_mean:.0f}, dernier: {current_volume:.0f}")
        
        # 3. PREMIER appel get_trading_signal
        print(f"\n{'─'*80}")
        print("APPEL 1: get_trading_signal() avec paramètres par défaut")
        signal1, price1, trend1, rsi1, vol1, score1, deriv1 = get_trading_signal(
            prices, volumes, domaine=sector, return_derivatives=True, symbol=symbol
        )
        print(f"   Signal: {signal1}, Score: {score1}, RSI: {rsi1}")
        print(f"   Seuil achat utilisé: {deriv1.get('_seuil_achat_used', 'N/A')}")
        print(f"   Seuil vente utilisé: {deriv1.get('_seuil_vente_used', 'N/A')}")
        print(f"   Clé paramètres: {deriv1.get('_selected_param_key', 'N/A')}")
        
        # 4. DEUXIÈME appel get_trading_signal (avec seuils explicites)
        print(f"\n{'─'*80}")
        print("APPEL 2: get_trading_signal() avec seuils explicites (4.2 / -0.5)")
        signal2, price2, trend2, rsi2, vol2, score2, deriv2 = get_trading_signal(
            prices, volumes, domaine=sector, return_derivatives=True, symbol=symbol,
            seuil_achat=4.2, seuil_vente=-0.5
        )
        print(f"   Signal: {signal2}, Score: {score2}, RSI: {rsi2}")
        
        # 5. TROISIÈME appel get_trading_signal (avec cap_range si pertinent)
        print(f"\n{'─'*80}")
        print("APPEL 3: get_trading_signal() avec cap_range à 'Large'")
        signal3, price3, trend3, rsi3, vol3, score3, deriv3 = get_trading_signal(
            prices, volumes, domaine=sector, cap_range="Large", return_derivatives=True, symbol=symbol
        )
        print(f"   Signal: {signal3}, Score: {score3}, RSI: {rsi3}")
        print(f"   Clé paramètres: {deriv3.get('_selected_param_key', 'N/A')}")
        
        # 6. Comparer
        print(f"\n{'='*80}")
        print("COMPARAISON DES 3 APPELS")
        print('='*80)
        signals_match = signal1 == signal2 == signal3
        scores_diff = abs(score1 - score3) > 0.01
        
        result = {
            'symbol': symbol,
            'calls': [
                {'signal': signal1, 'score': score1, 'rsi': rsi1, 'key': deriv1.get('_selected_param_key')},
                {'signal': signal2, 'score': score2, 'rsi': rsi2, 'key': deriv2.get('_selected_param_key')},
                {'signal': signal3, 'score': score3, 'rsi': rsi3, 'key': deriv3.get('_selected_param_key')},
            ],
            'signals_consistent': signals_match,
            'scores_different': scores_diff,
        }
        
        if signals_match and not scores_diff:
            print("✅ COHÉRENT: Tous les appels retournent le même signal et score similaire")
        else:
            print("❌ INCOHÉRENT:")
            if not signals_match:
                print(f"   ⚠️  Signaux différents: {signal1} vs {signal2} vs {signal3}")
            if scores_diff:
                print(f"   ⚠️  Scores différents: {score1:.3f} vs {score2:.3f} vs {score3:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur pour {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Comparer KIM vs AEP en détail"""
    results = {}
    
    # Diagnostiquer KIM
    results['KIM'] = diagnose_symbol('KIM', 'Real Estate')
    
    # Diagnostiquer AEP
    results['AEP'] = diagnose_symbol('AEP', 'Utilities')
    
    # Résumé comparatif
    print(f"\n{'='*80}")
    print("RÉSUMÉ COMPARATIF KIM vs AEP")
    print('='*80)
    
    for symbol in ['KIM', 'AEP']:
        if results[symbol]:
            r = results[symbol]
            print(f"\n{symbol}:")
            print(f"  Appel 1: Signal={r['calls'][0]['signal']}, Score={r['calls'][0]['score']:.3f}")
            print(f"  Appel 2: Signal={r['calls'][1]['signal']}, Score={r['calls'][1]['score']:.3f}")
            print(f"  Appel 3: Signal={r['calls'][2]['signal']}, Score={r['calls'][2]['score']:.3f}")
            print(f"  Cohérent: {r['signals_consistent']}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
