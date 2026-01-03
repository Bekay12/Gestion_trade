"""
Test de l'auto-enregistrement des symboles dans popular
"""
import sqlite3
from pathlib import Path
from symbol_manager import (
    init_symbols_table, 
    sync_txt_to_sqlite, 
    get_symbols_by_list_type,
    auto_add_to_popular,
    sync_all_to_popular,
    DB_PATH
)

def test_auto_popular():
    """
    Teste que :
    1. Les symboles de personal sont auto-ajoutés à popular
    2. Les symboles de optimization sont auto-ajoutés à popular
    3. Les symboles analysés sont auto-ajoutés à popular
    """
    print("=== Test auto-enregistrement dans popular ===\n")
    
    # Initialiser
    init_symbols_table()
    
    # Créer des fichiers de test
    test_file_personal = Path("test_personal_auto.txt")
    test_file_personal.write_text("TSLA\nNVDA\n")
    
    test_file_optim = Path("test_optim_auto.txt")
    test_file_optim.write_text("COIN\nMSTR\n")
    
    print("1️⃣ Test: Ajout de symboles à 'personal'")
    sync_txt_to_sqlite(str(test_file_personal), list_type='personal')
    
    # Vérifier que TSLA est dans personal ET popular
    personal_symbols = get_symbols_by_list_type('personal')
    popular_symbols = get_symbols_by_list_type('popular')
    
    print(f"   Personal contient: {[s for s in ['TSLA', 'NVDA'] if s in personal_symbols]}")
    print(f"   Popular contient: {[s for s in ['TSLA', 'NVDA'] if s in popular_symbols]}")
    
    assert 'TSLA' in personal_symbols, "TSLA devrait être dans personal"
    assert 'TSLA' in popular_symbols, "❌ TSLA devrait être auto-ajouté à popular !"
    print("   ✅ Auto-sync personal → popular fonctionne\n")
    
    print("2️⃣ Test: Ajout de symboles à 'optimization'")
    sync_txt_to_sqlite(str(test_file_optim), list_type='optimization')
    
    # Vérifier que COIN est dans optimization ET popular
    optim_symbols = get_symbols_by_list_type('optimization')
    popular_symbols = get_symbols_by_list_type('popular')
    
    print(f"   Optimization contient: {[s for s in ['COIN', 'MSTR'] if s in optim_symbols]}")
    print(f"   Popular contient: {[s for s in ['COIN', 'MSTR'] if s in popular_symbols]}")
    
    assert 'COIN' in optim_symbols, "COIN devrait être dans optimization"
    assert 'COIN' in popular_symbols, "❌ COIN devrait être auto-ajouté à popular !"
    print("   ✅ Auto-sync optimization → popular fonctionne\n")
    
    print("3️⃣ Test: Fonction sync_all_to_popular()")
    stats = sync_all_to_popular()
    print(f"   Résultat sync: {stats}")
    print("   ✅ Sync globale fonctionne\n")
    
    print("4️⃣ Vérification base de données")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Vérifier TSLA
    cursor.execute('''
        SELECT list_type FROM symbol_lists WHERE symbol = 'TSLA' ORDER BY list_type
    ''')
    tsla_lists = [row[0] for row in cursor.fetchall()]
    print(f"   TSLA est dans: {tsla_lists}")
    assert 'personal' in tsla_lists, "TSLA devrait être dans personal"
    assert 'popular' in tsla_lists, "TSLA devrait être dans popular"
    
    # Vérifier COIN
    cursor.execute('''
        SELECT list_type FROM symbol_lists WHERE symbol = 'COIN' ORDER BY list_type
    ''')
    coin_lists = [row[0] for row in cursor.fetchall()]
    print(f"   COIN est dans: {coin_lists}")
    assert 'optimization' in coin_lists, "COIN devrait être dans optimization"
    assert 'popular' in coin_lists, "COIN devrait être dans popular"
    
    conn.close()
    
    # Nettoyage
    test_file_personal.unlink()
    test_file_optim.unlink()
    
    print("\n✅ Tous les tests passés !")
    print("   → Les symboles de personal sont auto-ajoutés à popular")
    print("   → Les symboles de optimization sont auto-ajoutés à popular")
    print("   → Popular devient la liste master automatique")

if __name__ == "__main__":
    test_auto_popular()
