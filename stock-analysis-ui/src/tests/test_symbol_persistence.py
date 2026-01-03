"""
Test de la persistance des symboles ajoutés via l'UI.
Vérifie que les symboles ajoutés restent dans les bonnes listes.
"""
import sqlite3
from pathlib import Path
from symbol_manager import init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type, DB_PATH

def test_symbol_persistence():
    """
    Test: Ajouter un symbole à plusieurs listes et vérifier qu'il apparaît dans toutes.
    """
    print("=== Test de persistance des symboles ===\n")
    
    # Initialiser les tables
    init_symbols_table()
    
    # Simuler l'ajout de AAPL à popular_symbols.txt
    test_file_popular = Path("test_popular.txt")
    test_file_popular.write_text("AAPL\nMSFT\nGOOGL\n")
    
    print("1️⃣ Synchronisation de test_popular.txt (list_type='popular')...")
    sync_txt_to_sqlite(str(test_file_popular), list_type='popular')
    
    # Vérifier que AAPL est dans 'popular'
    popular_symbols = get_symbols_by_list_type('popular')
    print(f"   Symboles dans 'popular': {popular_symbols}")
    assert "AAPL" in popular_symbols, "AAPL devrait être dans 'popular'"
    print("   ✅ AAPL trouvé dans 'popular'\n")
    
    # Simuler l'ajout de AAPL à mes_symbols.txt (même symbole, liste différente)
    test_file_mes = Path("test_mes.txt")
    test_file_mes.write_text("AAPL\nTSLA\nNVDA\n")
    
    print("2️⃣ Synchronisation de test_mes.txt (list_type='personal')...")
    sync_txt_to_sqlite(str(test_file_mes), list_type='personal')
    
    # Vérifier que AAPL est TOUJOURS dans 'popular'
    popular_symbols_after = get_symbols_by_list_type('popular')
    print(f"   Symboles dans 'popular' (après): {popular_symbols_after}")
    assert "AAPL" in popular_symbols_after, "❌ AAPL a disparu de 'popular' !"
    print("   ✅ AAPL toujours dans 'popular'\n")
    
    # Vérifier que AAPL est AUSSI dans 'personal'
    personal_symbols = get_symbols_by_list_type('personal')
    print(f"   Symboles dans 'personal': {personal_symbols}")
    assert "AAPL" in personal_symbols, "AAPL devrait être dans 'personal'"
    print("   ✅ AAPL trouvé dans 'personal'\n")
    
    # Vérifier dans la base de données directement
    print("3️⃣ Vérification dans la base de données...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Compter combien de listes contiennent AAPL
    cursor.execute('''
        SELECT list_type FROM symbol_lists WHERE symbol = 'AAPL'
    ''')
    lists_for_aapl = [row[0] for row in cursor.fetchall()]
    print(f"   AAPL appartient à ces listes: {lists_for_aapl}")
    assert len(lists_for_aapl) >= 2, f"AAPL devrait être dans au moins 2 listes, trouvé: {len(lists_for_aapl)}"
    assert 'popular' in lists_for_aapl, "AAPL devrait être dans 'popular'"
    assert 'personal' in lists_for_aapl, "AAPL devrait être dans 'personal'"
    print(f"   ✅ AAPL correctement associé à {len(lists_for_aapl)} liste(s)\n")
    
    conn.close()
    
    # Nettoyage
    test_file_popular.unlink()
    test_file_mes.unlink()
    
    print("✅ Tous les tests passés !")
    print("   → Un symbole peut maintenant appartenir à plusieurs listes simultanément")
    print("   → Les symboles ajoutés via l'UI seront persistants au redémarrage")

if __name__ == "__main__":
    test_symbol_persistence()
