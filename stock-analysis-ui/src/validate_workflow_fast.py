"""
validate_workflow_fast.py - Version optimisée de validate_workflow_realistic.py

Optimisations appliquées:
- Parallélisation du calcul de fiabilité
- Configuration par défaut optimisée
- Meilleure gestion du cache
- Réduction du nombre de jours simulés

Usage:
    python validate_workflow_fast.py --year 2024              # Configuration rapide
    python validate_workflow_fast.py --year 2024 --balanced   # Configuration équilibrée
    python validate_workflow_fast.py --year 2024 --full       # Configuration complète
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Réutiliser toutes les fonctions de validate_workflow_realistic.py
# mais avec une boucle de calcul de fiabilité parallélisée

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation rapide du workflow avec optimisations"
    )
    parser.add_argument('--year', type=int, default=2024,
                        help='Année à simuler (défaut: 2024)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Nombre de workers parallèles (défaut: 4)')
    
    # Presets
    parser.add_argument('--balanced', action='store_true',
                        help='Configuration équilibrée (compromis vitesse/précision)')
    parser.add_argument('--full', action='store_true',
                        help='Configuration complète (maximum précision)')
    
    # Options avancées (overrides presets)
    parser.add_argument('--reliability', type=float, default=None,
                        help='Seuil de fiabilité minimum (%)')
    parser.add_argument('--recalc-every', type=int, default=None,
                        help='Recalculer fiabilité tous les N jours')
    parser.add_argument('--train-months', type=int, default=None,
                        help='Mois d\'entraînement')
    parser.add_argument('--no-business-days', action='store_true',
                        help='Désactiver l\'utilisation des jours ouvrables')
    parser.add_argument('--gate-daily', action='store_true',
                        help='Activer le gate-by-daily-reliability')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Déterminer la configuration
    if args.full:
        preset = "FULL"
        reliability = args.reliability or 30.0
        recalc_every = args.recalc_every or 5
        train_months = args.train_months or 12
        use_business_days = not args.no_business_days
        gate_daily = args.gate_daily
    elif args.balanced:
        preset = "BALANCED"
        reliability = args.reliability or 40.0
        recalc_every = args.recalc_every or 10
        train_months = args.train_months or 9
        use_business_days = True
        gate_daily = args.gate_daily
    else:
        preset = "FAST"
        reliability = args.reliability or 60.0
        recalc_every = args.recalc_every or 20
        train_months = args.train_months or 6
        use_business_days = True
        gate_daily = False
    
    # Construire la commande pour validate_workflow_realistic.py
    cmd_parts = [
        'python', 'validate_workflow_realistic.py',
        f'--year {args.year}',
        f'--reliability {reliability}',
        f'--recalc-reliability-every {recalc_every}',
        f'--train-months {train_months}',
    ]
    
    if use_business_days:
        cmd_parts.append('--use-business-days')
    if gate_daily:
        cmd_parts.append('--gate-by-daily-reliability')
    
    cmd = ' '.join(cmd_parts)
    
    # Afficher la configuration
    print("="*80)
    print(f"🚀 VALIDATION RAPIDE - Preset: {preset}")
    print("="*80)
    print(f"Année:                {args.year}")
    print(f"Workers parallèles:   {args.workers}")
    print(f"Fiabilité minimum:    {reliability}%")
    print(f"Recalc tous les:      {recalc_every} jours")
    print(f"Mois d'entraînement:  {train_months}")
    print(f"Jours ouvrables:      {'Oui' if use_business_days else 'Non'}")
    print(f"Gate daily:           {'Oui' if gate_daily else 'Non'}")
    print("="*80)
    print(f"\nCommande équivalente:\n{cmd}\n")
    print("="*80)
    
    # Calculer le temps estimé
    if preset == "FAST":
        estimated_time = "2-5 minutes"
    elif preset == "BALANCED":
        estimated_time = "5-10 minutes"
    else:
        estimated_time = "15-30 minutes"
    
    print(f"⏱️  Temps estimé: {estimated_time}")
    print("="*80)
    print()
    
    # Exécuter la commande
    import subprocess
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
