#!/usr/bin/env python
"""
Validation script: Vérifie que la migration SQLite fonctionne correctement
"""

import sqlite3
from pathlib import Path

def validate_migration():
    """Valide la configuration SQLite."""
    
    from config import OPTIMIZATION_DB_PATH
    db_path = OPTIMIZATION_DB_PATH
    
    print("🔍 Validation de la migration SQLite")
    print("=" * 70)
    
    # Vérifier que la BD existe
    if not Path(db_path).exists():
        print(f"❌ Base de données non trouvée: {db_path}")
        print("   Veuillez exécuter: python migration_csv_to_sqlite.py")
        return False
    
    print(f"✅ Base de données trouvée: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier la table existe
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='optimization_runs'
        ''')
        if not cursor.fetchone():
            print("❌ Table 'optimization_runs' non trouvée")
            conn.close()
            return False
        print("✅ Table 'optimization_runs' existe")
        
        # Compter les lignes
        cursor.execute('SELECT COUNT(*) FROM optimization_runs')
        count = cursor.fetchone()[0]
        print(f"✅ {count} lignes de données")
        
        # Vérifier les secteurs
        cursor.execute('SELECT DISTINCT sector FROM optimization_runs ORDER BY sector')
        sectors = [row[0] for row in cursor.fetchall()]
        print(f"✅ {len(sectors)} secteurs: {sectors}")
        
        # Vérifier les indices
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='index' AND tbl_name='optimization_runs'
        ''')
        indices = [row[0] for row in cursor.fetchall()]
        print(f"✅ {len(indices)} indices créés: {indices}")
        
        # Tester la query pour derniers résultats par secteur
        print("\n📋 Derniers résultats par secteur:")
        print("-" * 70)
        cursor.execute('''
            SELECT sector, timestamp, gain_moy, success_rate
            FROM optimization_runs
            WHERE (sector, timestamp) IN (
                SELECT sector, MAX(timestamp) 
                FROM optimization_runs 
                GROUP BY sector
            )
            ORDER BY sector
        ''')
        
        for sector, ts, gain, sr in cursor.fetchall():
            print(f"  {sector:20s} | TS: {ts} | Gain: {gain:8.4f} | Success: {sr:7.4f}")
        
        conn.close()
        
        print("\n✅ Migration SQLite validée avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de validation: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = validate_migration()
    sys.exit(0 if success else 1)
