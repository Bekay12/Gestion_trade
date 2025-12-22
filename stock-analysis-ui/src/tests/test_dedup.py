#!/usr/bin/env python3
"""Test script to verify deduplication and synchronization of backtest results."""
import sys
import io

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from qsi import analyse_signaux_populaires
import pandas as pd

# Load all symbols
symbols = pd.read_csv('popular_symbols.txt', header=None)[0].tolist()

print(f"Testing with {len(symbols)} symbols...\n")

# Run analysis
result = analyse_signaux_populaires(symbols, [], verbose=False)

# Count at each stage
signals = result['signals']
signaux_tries = result['signaux_tries']
backtest_results = result['backtest_results']

signaux_affiches = (
    len(signaux_tries['ACHAT']['Hausse']) + 
    len(signaux_tries['ACHAT']['Baisse']) + 
    len(signaux_tries['VENTE']['Hausse']) + 
    len(signaux_tries['VENTE']['Baisse'])
)

print(f"✅ Signaux générés (bruts): {len(signals)}")
print(f"✅ Signaux affichés (filtrés): {signaux_affiches}")
print(f"✅ Résultats backtestés: {len(backtest_results)}")

# Check for duplicates
backtest_symbols = [r['Symbole'] for r in backtest_results]
print(f"\n✅ Symboles uniques backtestés: {len(set(backtest_symbols))}")

if len(set(backtest_symbols)) != len(backtest_symbols):
    print("⚠️ DOUBLONS DÉTECTÉS!")
    from collections import Counter
    dups = [sym for sym, count in Counter(backtest_symbols).items() if count > 1]
    print(f"   Symboles dupliqués: {dups}")
else:
    print("✅ Pas de doublons!")

# Check if all backtest results have signals
signal_symbols = set(s['Symbole'] for s in signals)
backtest_symbols_set = set(backtest_symbols)
orphan = backtest_symbols_set - signal_symbols
if orphan:
    print(f"\n⚠️ Symboles backtestés sans signal: {orphan}")
else:
    print(f"\n✅ Tous les symboles backtestés ont un signal")

# Verify counts match
if signaux_affiches == len(backtest_results):
    print(f"\n✅ SYNCHRONISÉ: {signaux_affiches} signaux affichés == {len(backtest_results)} backtestés")
else:
    print(f"\n⚠️ DÉSYNCHRONISÉ: {signaux_affiches} affichés != {len(backtest_results)} backtestés")
    diff = len(backtest_results) - signaux_affiches
    if diff > 0:
        extra = backtest_symbols_set - signal_symbols
        print(f"   {len(extra)} symboles backtestés sans signal: {extra}")
