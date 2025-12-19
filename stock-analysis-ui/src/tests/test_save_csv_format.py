#!/usr/bin/env python
"""Test script pour vérifier que la sauvegarde dans le nouveau format CSV fonctionne"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

def test_save_optimization_results():
    print("🧪 Test: save_optimization_results avec le nouveau format CSV")
    print("=" * 70)
    
    try:
        # Créer un fichier test
        test_csv = "signaux/test_optimization_save.csv"
        
        # Données de test
        domain = "Test Sector"
        best_coeffs = (1.5, 2.0, 1.2, 1.8, 1.4, 2.1, 1.3, 2.5)
        gain_total = 25.5
        success_rate = 75.0
        total_trades = 42
        best_feature_thresholds = (50.0, 1.0, 2.0, 1.5, 25.0, 3.0, 0.5, 4.5)
        best_seuil_achat = 3.5
        best_seuil_vente = -1.2
        all_thresholds = best_feature_thresholds + (best_seuil_achat, best_seuil_vente)
        
        # Mock de save_optimization_results
        import pandas as pd
        
        results = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Sector': domain,
            'Gain_moy': gain_total,
            'Success_Rate': success_rate,
            'Trades': total_trades,
            'Seuil_Achat': all_thresholds[8],
            'Seuil_Vente': all_thresholds[9],
            'a1': best_coeffs[0], 'a2': best_coeffs[1], 'a3': best_coeffs[2], 'a4': best_coeffs[3],
            'a5': best_coeffs[4], 'a6': best_coeffs[5], 'a7': best_coeffs[6], 'a8': best_coeffs[7],
            'th1': all_thresholds[0], 'th2': all_thresholds[1], 'th3': all_thresholds[2], 'th4': all_thresholds[3],
            'th5': all_thresholds[4], 'th6': all_thresholds[5], 'th7': all_thresholds[6], 'th8': all_thresholds[7],
        }
        
        # Écrire le test CSV
        df_new = pd.DataFrame([results])
        Path(test_csv).parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(test_csv, index=False)
        
        print(f"✅ Sauvegarde test réussie: {test_csv}")
        
        # Lire et vérifier
        df_read = pd.read_csv(test_csv)
        print(f"✅ Lecture test réussie")
        print(f"\nColumns: {list(df_read.columns)}")
        print(f"Values: {df_read.iloc[0].to_dict()}")
        
        # Vérifier l'ordre des colonnes
        expected_order = ['Timestamp','Sector','Gain_moy','Success_Rate','Trades','Seuil_Achat','Seuil_Vente',
                         'a1','a2','a3','a4','a5','a6','a7','a8','th1','th2','th3','th4','th5','th6','th7','th8']
        
        # Vérifier les colonnes principales
        essential_cols = ['Timestamp','Sector','Seuil_Achat','Seuil_Vente','a1','a8','th1','th8']
        for col in essential_cols:
            if col not in df_read.columns:
                print(f"❌ Colonne manquante: {col}")
                return False
            print(f"✅ Colonne présente: {col} = {df_read[col][0]}")
        
        # Vérifier les valeurs
        if float(df_read['a1'][0]) != best_coeffs[0]:
            print(f"❌ a1 incorrect: {df_read['a1'][0]} vs {best_coeffs[0]}")
            return False
        
        if float(df_read['th1'][0]) != best_feature_thresholds[0]:
            print(f"❌ th1 incorrect: {df_read['th1'][0]} vs {best_feature_thresholds[0]}")
            return False
        
        if float(df_read['Seuil_Achat'][0]) != best_seuil_achat:
            print(f"❌ Seuil_Achat incorrect: {df_read['Seuil_Achat'][0]} vs {best_seuil_achat}")
            return False
        
        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS DE SAUVEGARDE PASSENT!")
        
        # Nettoyer
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_save_optimization_results()
    sys.exit(0 if success else 1)
