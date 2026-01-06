"""
benchmark_validation.py - Mesure les performances des différentes configurations

Ce script teste plusieurs configurations et mesure leur temps d'exécution.
"""

import subprocess
import time
from datetime import datetime
import json

CONFIGURATIONS = {
    "baseline": {
        "name": "Configuration baseline (défaut)",
        "args": "--year 2024",
        "estimated": "20-30 min"
    },
    "business_days": {
        "name": "Avec jours ouvrables",
        "args": "--year 2024 --use-business-days",
        "estimated": "15-20 min"
    },
    "recalc_20": {
        "name": "Recalc tous les 20 jours",
        "args": "--year 2024 --use-business-days --recalc-reliability-every 20",
        "estimated": "8-12 min"
    },
    "fast": {
        "name": "Configuration rapide",
        "args": "--year 2024 --use-business-days --recalc-reliability-every 20 --reliability 60 --train-months 6",
        "estimated": "3-5 min"
    },
    "balanced": {
        "name": "Configuration équilibrée",
        "args": "--year 2024 --use-business-days --recalc-reliability-every 10 --reliability 40 --train-months 9",
        "estimated": "5-10 min"
    }
}


def run_benchmark(config_name, config):
    """Exécute un benchmark pour une configuration donnée"""
    print(f"\n{'='*80}")
    print(f"🔍 Test: {config['name']}")
    print(f"   Commande: python validate_workflow_realistic.py {config['args']}")
    print(f"   Temps estimé: {config['estimated']}")
    print(f"{'='*80}\n")
    
    cmd = f"python validate_workflow_realistic.py {config['args']}"
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    return {
        "config": config_name,
        "name": config["name"],
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "estimated": config["estimated"],
        "success": result.returncode == 0,
        "timestamp": datetime.now().isoformat()
    }


def main():
    print("="*80)
    print("📊 BENCHMARK DE VALIDATION")
    print("="*80)
    print("\n⚠️  ATTENTION: Ce benchmark peut prendre plusieurs heures!")
    print("    Recommandation: Tester seulement les configurations rapides d'abord\n")
    
    # Demander quelles configs tester
    print("Configurations disponibles:")
    for i, (key, cfg) in enumerate(CONFIGURATIONS.items(), 1):
        print(f"  {i}. {cfg['name']} - Estimé: {cfg['estimated']}")
    
    print("\nOptions:")
    print("  - 'all' pour tout tester (plusieurs heures)")
    print("  - '4,5' pour tester seulement les configs 4 et 5")
    print("  - 'fast' pour tester seulement les configs rapides (4,5)")
    print("  - 'q' pour quitter")
    
    choice = input("\nVotre choix: ").strip().lower()
    
    if choice == 'q':
        print("Benchmark annulé.")
        return
    
    # Déterminer quelles configs tester
    if choice == 'all':
        configs_to_test = list(CONFIGURATIONS.keys())
    elif choice == 'fast':
        configs_to_test = ['fast', 'balanced']
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            configs_to_test = [list(CONFIGURATIONS.keys())[i-1] for i in indices]
        except:
            print("❌ Choix invalide. Benchmark annulé.")
            return
    
    print(f"\n✅ Configurations à tester: {', '.join(configs_to_test)}")
    input("\nAppuyez sur ENTRÉE pour commencer...")
    
    # Exécuter les benchmarks
    results = []
    for config_name in configs_to_test:
        config = CONFIGURATIONS[config_name]
        result = run_benchmark(config_name, config)
        results.append(result)
        
        print(f"\n{'='*80}")
        print(f"✅ Terminé: {result['name']}")
        print(f"   Durée: {result['duration_minutes']:.2f} minutes")
        print(f"   Statut: {'Succès' if result['success'] else 'Échec'}")
        print(f"{'='*80}")
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Afficher le résumé
    print(f"\n\n{'='*80}")
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print(f"{'='*80}\n")
    
    if results:
        baseline_time = results[0]['duration_minutes']
        
        print(f"{'Configuration':<40} {'Durée':<15} {'vs Baseline':<15}")
        print("-" * 80)
        
        for r in results:
            speedup = (baseline_time / r['duration_minutes']) if r['duration_minutes'] > 0 else 0
            pct = (r['duration_minutes'] / baseline_time * 100) if baseline_time > 0 else 100
            
            print(f"{r['name']:<40} {r['duration_minutes']:>6.2f} min    "
                  f"{speedup:>4.2f}x ({pct:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"💾 Résultats sauvegardés dans: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
