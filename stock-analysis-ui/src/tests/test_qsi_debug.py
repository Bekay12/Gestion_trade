"""Test rapide pour diagnostiquer le problème des signaux"""
from qsi import analyse_signaux_populaires, extract_best_parameters
from symbol_manager import get_symbols_by_list_type

# Tester l'extraction
print("=" * 80)
print("TEST 1: Extraction des paramètres")
print("=" * 80)
best_params = extract_best_parameters()
print(f"✅ {len(best_params)} ensembles de paramètres chargés")

if best_params:
    key = list(best_params.keys())[0]
    params = best_params[key]
    print(f"\n📊 Exemple: {key}")
    print(f"   Longueur tuple: {len(params)}")
    print(f"   Coefficients: {params[0]}")
    print(f"   Thresholds: {params[1]}")
    print(f"   Globals: {params[2]}")
    print(f"   Gain moyen: {params[3]}")
    if len(params) > 4:
        print(f"   Extras type: {type(params[4])}")
        if isinstance(params[4], dict):
            print(f"   Extras keys: {list(params[4].keys())}")
        else:
            print(f"   ⚠️ Extras n'est pas un dict: {params[4]}")

# Tester l'analyse
print("\n" + "=" * 80)
print("TEST 2: Analyse de 3 symboles")
print("=" * 80)

# Récupérer quelques symboles
popular = get_symbols_by_list_type('popular')[:3]
mes_symbols = get_symbols_by_list_type('personal')[:3] if get_symbols_by_list_type('personal') else []

print(f"Popular symbols: {popular}")
print(f"Mes symbols: {mes_symbols}")

if popular:
    resultats = analyse_signaux_populaires(
        popular_symbols=popular,
        mes_symbols=mes_symbols,
        verbose=True,
        afficher_graphiques=False,
        save_csv=False,
        plot_all=False
    )
    
    print(f"\n✅ Résultats: {len(resultats.get('resultats', []))} signaux trouvés")
    if resultats and 'resultats' in resultats:
        for r in resultats['resultats'][:3]:
            print(f"   {r.get('Symbole')}: {r.get('Signal')} (Score: {r.get('Score')})")
else:
    print("❌ Aucun symbole populaire trouvé")
