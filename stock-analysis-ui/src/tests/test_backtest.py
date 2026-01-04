"""Test rapide pour vérifier que les backtests fonctionnent"""
from qsi import analyse_signaux_populaires
from symbol_manager import get_symbols_by_list_type

# Récupérer quelques symboles
popular = get_symbols_by_list_type('popular')[:5]
mes_symbols = get_symbols_by_list_type('personal')[:3] if get_symbols_by_list_type('personal') else []

print(f"Testing avec {len(popular)} symboles populaires")

if popular:
    resultats = analyse_signaux_populaires(
        popular_symbols=popular,
        mes_symbols=mes_symbols,
        verbose=True,
        afficher_graphiques=False,
        save_csv=False,
        plot_all=False
    )
    
    print(f"\n✅ Analyse terminée avec succès")
    print(f"   Signaux trouvés: {len(resultats) if resultats else 0}")
else:
    print("❌ Aucun symbole populaire trouvé")
