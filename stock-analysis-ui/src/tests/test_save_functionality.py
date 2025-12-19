#!/usr/bin/env python
"""
Test script: Vérifie que la sauvegarde SQLite fonctionne correctement
"""

import sqlite3
from pathlib import Path
import sys

def test_save_functionality():
    """Test la fonction de sauvegarde SQLite."""
    
    db_path = 'signaux/optimization_hist.db'
    
    print("🧪 Test de la sauvegarde SQLite")
    print("=" * 70)
    
    # Vérifier que la BD existe
    if not Path(db_path).exists():
        print(f"❌ Base de données non trouvée: {db_path}")
        return False
    
    print(f"✅ Base de données trouvée: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier que la table existe
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='optimization_runs'
        ''')
        if not cursor.fetchone():
            print("❌ Table 'optimization_runs' non trouvée")
            conn.close()
            return False
        
        print("✅ Table 'optimization_runs' existe")
        
        # Test 1: Vérifier que les indices existent
        print("\n📋 Vérification des indices:")
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='index' AND tbl_name='optimization_runs'
        ''')
        indices = [row[0] for row in cursor.fetchall()]
        
        required_indices = ['idx_sector_timestamp', 'idx_sector_gain']
        for idx in required_indices:
            if idx in indices:
                print(f"  ✅ Index '{idx}' existe")
            else:
                print(f"  ⚠️  Index '{idx}' manquant")
        
        # Test 2: Vérifier la contrainte UNIQUE
        print("\n🔒 Vérification de la contrainte UNIQUE:")
        cursor.execute('''
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='optimization_runs'
        ''')
        schema = cursor.fetchone()[0]
        if 'UNIQUE(sector, timestamp)' in schema:
            print("  ✅ Contrainte UNIQUE(sector, timestamp) active")
        else:
            print("  ⚠️  Contrainte UNIQUE manquante")
        
        # Test 3: Vérifier que les colonnes requises existent
        print("\n🏛️  Vérification des colonnes requises:")
        cursor.execute('PRAGMA table_info(optimization_runs)')
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_columns = {
            'timestamp': 'DATETIME', 'sector': 'TEXT', 'gain_moy': 'REAL',
            'success_rate': 'REAL', 'trades': 'INTEGER',
            'a1': 'REAL', 'a2': 'REAL', 'a3': 'REAL', 'a4': 'REAL',
            'a5': 'REAL', 'a6': 'REAL', 'a7': 'REAL', 'a8': 'REAL',
            'th1': 'REAL', 'th2': 'REAL', 'th3': 'REAL', 'th4': 'REAL',
            'th5': 'REAL', 'th6': 'REAL', 'th7': 'REAL', 'th8': 'REAL',
            'seuil_achat': 'REAL', 'seuil_vente': 'REAL'
        }
        
        all_present = True
        for col, expected_type in required_columns.items():
            if col in columns:
                print(f"  ✅ Colonne '{col}' ({columns[col]})")
            else:
                print(f"  ❌ Colonne '{col}' MANQUANTE")
                all_present = False
        
        # Test 4: Vérifier l'intégrité des données
        print("\n📊 Vérification de l'intégrité des données:")
        
        cursor.execute('SELECT COUNT(*) FROM optimization_runs')
        count = cursor.fetchone()[0]
        print(f"  ✅ {count} enregistrements")
        
        cursor.execute('SELECT COUNT(DISTINCT sector) FROM optimization_runs')
        sector_count = cursor.fetchone()[0]
        print(f"  ✅ {sector_count} secteurs distincts")
        
        # Vérifier qu'il n'y a pas de doublons (sector, timestamp)
        cursor.execute('''
            SELECT sector, timestamp, COUNT(*) as cnt
            FROM optimization_runs
            GROUP BY sector, timestamp
            HAVING cnt > 1
        ''')
        duplicates = cursor.fetchall()
        if not duplicates:
            print("  ✅ Aucun doublon (sector, timestamp)")
        else:
            print(f"  ❌ {len(duplicates)} doublon(s) trouvé(s)!")
            for sector, ts, cnt in duplicates:
                print(f"     {sector} @ {ts}: {cnt} doublons")
            all_present = False
        
        # Test 5: Vérifier que la requête d'extraction fonctionne
        print("\n🔍 Test de la requête d'extraction:")
        cursor.execute('''
            SELECT sector, timestamp, gain_moy
            FROM optimization_runs
            WHERE (sector, timestamp) IN (
                SELECT sector, MAX(timestamp)
                FROM optimization_runs
                GROUP BY sector
            )
            ORDER BY sector
        ''')
        results = cursor.fetchall()
        if results:
            print(f"  ✅ Requête d'extraction fonctionne ({len(results)} secteurs)")
        else:
            print(f"  ❌ Requête d'extraction échouée")
            all_present = False
        
        conn.close()
        
        print("\n" + "=" * 70)
        if all_present:
            print("✅ Tous les tests réussis! La base SQLite est prête pour la sauvegarde.")
            return True
        else:
            print("⚠️  Certains tests ont échoué. Voir les détails ci-dessus.")
            return False
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_save_functionality()
    sys.exit(0 if success else 1)
