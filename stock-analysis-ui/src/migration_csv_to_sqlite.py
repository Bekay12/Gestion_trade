#!/usr/bin/env python
"""
Migration: CSV → SQLite pour l'historique d'optimisation
Convertit signaux/optimization_hist_4stpV2.csv → signaux/optimization_hist.db
"""

import sqlite3
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

def migrate_csv_to_sqlite():
    """Convertit le CSV existant en SQLite et crée les indices optimaux."""
    
    csv_path = 'signaux/optimization_hist_4stpV2.csv'
    db_path = 'signaux/optimization_hist.db'
    backup_path = 'signaux/optimization_hist.db.backup'
    
    print("🔄 Migration: CSV → SQLite")
    print("=" * 70)
    
    # Vérifier que le CSV existe
    if not Path(csv_path).exists():
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        return False
    
    try:
        # Lire le CSV
        print(f"📖 Lecture du CSV: {csv_path}")
        df = pd.read_csv(csv_path, skipinitialspace=True, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        print(f"   ✅ {len(df)} lignes chargées")
        
        # Créer une sauvegarde si la DB existe déjà
        if Path(db_path).exists():
            print(f"⚠️  Sauvegarde de {db_path} → {backup_path}")
            shutil.copy(db_path, backup_path)
        
        # Créer/ouvrir la BD SQLite
        print(f"💾 Création de la base SQLite: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Créer la table avec schéma optimisé
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                sector TEXT NOT NULL,
                gain_moy REAL,
                success_rate REAL,
                trades INTEGER,
                seuil_achat REAL,
                seuil_vente REAL,
                a1 REAL, a2 REAL, a3 REAL, a4 REAL, a5 REAL, a6 REAL, a7 REAL, a8 REAL,
                th1 REAL, th2 REAL, th3 REAL, th4 REAL, th5 REAL, th6 REAL, th7 REAL, th8 REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sector, timestamp)
            )
        ''')
        
        # Créer les indices pour requêtes rapides
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector_timestamp ON optimization_runs(sector, timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector_gain ON optimization_runs(sector, gain_moy DESC)')
        
        # Insérer les données du CSV
        print(f"📝 Insertion des {len(df)} lignes dans SQLite...")
        for idx, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO optimization_runs
                    (timestamp, sector, gain_moy, success_rate, trades, 
                     seuil_achat, seuil_vente, 
                     a1, a2, a3, a4, a5, a6, a7, a8,
                     th1, th2, th3, th4, th5, th6, th7, th8)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(row.get('Timestamp', '')),
                    str(row.get('Sector', '')).strip(),
                    float(row.get('Gain_moy', 0)),
                    float(row.get('Success_Rate', 0)),
                    int(row.get('Trades', 0)),
                    float(row.get('Seuil_Achat', 4.2)),
                    float(row.get('Seuil_Vente', -0.5)),
                    float(row.get('a1', 1.0)), float(row.get('a2', 1.0)), float(row.get('a3', 1.0)), float(row.get('a4', 1.0)),
                    float(row.get('a5', 1.0)), float(row.get('a6', 1.0)), float(row.get('a7', 1.0)), float(row.get('a8', 1.0)),
                    float(row.get('th1', 50.0)), float(row.get('th2', 0.0)), float(row.get('th3', 0.0)), float(row.get('th4', 1.2)),
                    float(row.get('th5', 25.0)), float(row.get('th6', 0.0)), float(row.get('th7', 0.5)), float(row.get('th8', 4.2))
                ))
            except Exception as e:
                print(f"   ⚠️  Erreur ligne {idx}: {e}")
                continue
        
        conn.commit()
        
        # Vérifier les données
        cursor.execute('SELECT COUNT(*) FROM optimization_runs')
        total_rows = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT sector) FROM optimization_runs')
        total_sectors = cursor.fetchone()[0]
        
        print(f"✅ {total_rows} lignes insérées dans {total_sectors} secteurs")
        
        # Afficher un échantillon
        print("\n📊 Échantillon des données:")
        cursor.execute('''
            SELECT sector, timestamp, gain_moy, success_rate
            FROM optimization_runs
            ORDER BY timestamp DESC
            LIMIT 5
        ''')
        for row in cursor.fetchall():
            print(f"   {row[0]:20s} | {row[1]:19s} | Gain: {row[2]:7.2f} | Success: {row[3]:6.1f}%")
        
        conn.close()
        
        print("\n" + "=" * 70)
        print(f"✅ Migration réussie!")
        print(f"   BD SQLite créée: {db_path}")
        print(f"   CSV original conservé: {csv_path}")
        print(f"   Sauvegarde backup: {backup_path if Path(backup_path).exists() else '(aucune)'}")
        print("\n💡 Pour revenir à CSV, restaurer depuis backup ou utiliser le CSV original.")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la migration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = migrate_csv_to_sqlite()
    exit(0 if success else 1)
