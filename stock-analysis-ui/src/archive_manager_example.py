#!/usr/bin/env python3
"""
Exemple : Système d'Archivage Automatisé des Analyses Boursières

Cas d'usage:
- Génère des rapports PDF quotidiens
- Archive les résultats en JSON
- Gère un historique décentralisé
- Envoie des alertes sur signaux excellents
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Importer nos modules
sys.path.insert(0, str(Path(__file__).parent))
from batch_report_generator import BatchReportGenerator


class AnalysisArchiveManager:
    """Gestionnaire d'archives pour analyses boursières"""
    
    def __init__(self):
        self.archive_dir = Path("Results") / "archives"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.batch_gen = BatchReportGenerator(Path("Results"))
    
    def save_daily_analysis(self, current_results, clean_columns, tag=""):
        """Archiver l'analyse du jour"""
        
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M%S')
        
        # Nom du fichier: archive_YYYYMMDD_HHMMSS_[tag].json
        filename = f"analysis_{date_str}_{time_str}"
        if tag:
            filename += f"_{tag}"
        filename += ".json"
        
        filepath = self.archive_dir / filename
        
        # Préparer les données
        archive_data = {
            'timestamp': timestamp.isoformat(),
            'date': date_str,
            'results': current_results,
            'columns': clean_columns,
            'total_symbols': len(current_results),
            'signal_stats': self._compute_signal_stats(current_results),
            'top_scores': self._get_top_scores(current_results),
        }
        
        # Sauvegarder
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Archive sauvegardée: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
    
    def load_daily_analysis(self, date_str):
        """Charger une analyse archivée"""
        # Trouve le fichier le plus récent pour une date donnée
        pattern = f"analysis_{date_str}_*.json"
        files = sorted(self.archive_dir.glob(pattern))
        
        if not files:
            print(f"❌ Pas d'archive trouvée pour {date_str}")
            return None
        
        latest = files[-1]
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Archive chargée: {latest.name}")
            return data
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None
    
    def _compute_signal_stats(self, results):
        """Compter les signaux par type"""
        stats = {
            'ACHAT': 0,
            'VENTE': 0,
            'NEUTRE': 0,
            'TOTAL': len(results)
        }
        
        for result in results:
            signal = result.get('Signal', 'NEUTRE')
            if signal in stats:
                stats[signal] += 1
            else:
                stats['NEUTRE'] += 1
        
        return stats
    
    def _get_top_scores(self, results, limit=5):
        """Récupérer les top N scores"""
        # Trier par Score (si disponible)
        valid = [r for r in results if 'Score' in r]
        sorted_results = sorted(valid, key=lambda x: x['Score'], reverse=True)
        
        top = []
        for result in sorted_results[:limit]:
            top.append({
                'Symbol': result.get('Symbol'),
                'Score': result.get('Score'),
                'Signal': result.get('Signal'),
            })
        
        return top
    
    def compare_analyses(self, date1, date2):
        """Comparer deux analyses"""
        data1 = self.load_daily_analysis(date1)
        data2 = self.load_daily_analysis(date2)
        
        if not data1 or not data2:
            print("❌ Impossible de charger les deux analyses")
            return
        
        print(f"\n📊 Comparaison {date1} vs {date2}")
        print("=" * 50)
        
        # Extraire les listes de symboles
        symbols1 = {r['Symbol'] for r in data1['results']}
        symbols2 = {r['Symbol'] for r in data2['results']}
        
        nouveaux = symbols2 - symbols1
        disparus = symbols1 - symbols2
        communs = symbols1 & symbols2
        
        print(f"\n📈 Symboles nouveaux ({len(nouveaux)}):")
        for sym in sorted(nouveaux)[:5]:
            print(f"   • {sym}")
        if len(nouveaux) > 5:
            print(f"   ... et {len(nouveaux)-5} autres")
        
        print(f"\n📉 Symboles disparus ({len(disparus)}):")
        for sym in sorted(disparus)[:5]:
            print(f"   • {sym}")
        if len(disparus) > 5:
            print(f"   ... et {len(disparus)-5} autres")
        
        print(f"\n🔄 Symboles communs: {len(communs)}")
        
        # Comparaison des signaux
        stats1 = data1['signal_stats']
        stats2 = data2['signal_stats']
        
        print(f"\n🎯 Évolution des signaux:")
        for signal in ['ACHAT', 'VENTE', 'NEUTRE']:
            diff = stats2[signal] - stats1[signal]
            symbol = "📈" if diff > 0 else "📉" if diff < 0 else "→"
            print(f"   {signal}: {stats1[signal]} → {stats2[signal]} ({symbol} {diff:+d})")
    
    def get_excellent_opportunities(self, min_score=8.5):
        """Récupérer les excellentes opportunités du jour"""
        # Charger la dernière analyse
        today = datetime.now().strftime('%Y%m%d')
        data = self.load_daily_analysis(today)
        
        if not data:
            print("❌ Pas d'analyse disponible pour aujourd'hui")
            return []
        
        # Filtrer les excellents signaux
        excellent = [
            r for r in data['results']
            if r.get('Score', 0) >= min_score and r.get('Signal') == 'ACHAT'
        ]
        
        if excellent:
            print(f"\n🚀 EXCELLENTES OPPORTUNITÉS (score >= {min_score}):")
            for item in sorted(excellent, key=lambda x: x.get('Score', 0), reverse=True):
                symbol = item.get('Symbol', 'N/A')
                score = item.get('Score', 'N/A')
                roe = item.get('ROE', 'N/A')
                peg = item.get('PEG', 'N/A')
                print(f"   🎯 {symbol}: Score {score}, ROE {roe}, PEG {peg}")
        
        return excellent
    
    def list_archives(self, days=7):
        """Lister les archives des N derniers jours"""
        archives = list(self.archive_dir.glob("analysis_*.json"))
        archives.sort(reverse=True)
        
        if not archives:
            print("❌ Pas d'archives trouvées")
            return
        
        print(f"\n📂 Archives ({len(archives)} fichiers):")
        print("=" * 60)
        
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = 0
        for archive in archives:
            try:
                with open(archive, 'r') as f:
                    data = json.load(f)
                    ts = datetime.fromisoformat(data['timestamp'])
                    
                    if ts > cutoff:
                        total = data['total_symbols']
                        signals = data['signal_stats']
                        print(f"✅ {archive.name}")
                        print(f"   {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Symboles: {total} | ACHAT: {signals['ACHAT']} | VENTE: {signals['VENTE']}")
                        recent += 1
            except:
                pass
        
        print(f"\n📊 Total des {{recent}} derniers jours: {len(archives)}")
    
    def export_period_summary(self, days=7):
        """Générer un résumé sur une période"""
        archives = list(self.archive_dir.glob("analysis_*.json"))
        
        total_results = []
        dates = []
        
        cutoff = datetime.now() - timedelta(days=days)
        
        for archive in archives:
            try:
                with open(archive, 'r') as f:
                    data = json.load(f)
                    ts = datetime.fromisoformat(data['timestamp'])
                    
                    if ts > cutoff:
                        total_results.extend(data['results'])
                        dates.append(data['date'])
            except:
                pass
        
        if not total_results:
            print(f"❌ Pas de données pour les {days} derniers jours")
            return None
        
        # Générer un résumé
        summary = {
            'period': f"derniers_{days}_jours",
            'start_date': min(dates),
            'end_date': max(dates),
            'total_analyses': len(dates),
            'unique_dates': len(set(dates)),
            'total_results_collected': len(total_results),
            'unique_symbols': len(set(r['Symbol'] for r in total_results)),
            'signal_distribution': self._compute_signal_stats(total_results),
            'top_performers': self._get_top_scores(total_results, limit=10),
        }
        
        # Sauvegarder le résumé
        summary_file = self.archive_dir / f"summary_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 RÉSUMÉ ({days} jours)")
        print("=" * 50)
        print(f"Fichier: {summary_file.name}")
        print(f"Analyses: {summary['total_analyses']}")
        print(f"Dates uniques: {summary['unique_dates']}")
        print(f"Symboles collectés: {summary['total_results_collected']}")
        print(f"Symboles uniques: {summary['unique_symbols']}")
        print(f"\nSignaux:")
        for sig, count in summary['signal_distribution'].items():
            print(f"  {sig}: {count}")
        
        return summary


def demo():
    """Démonstration du système"""
    print("🚀 Démonstration du Gestionnaire d'Archives")
    print("=" * 60)
    
    manager = AnalysisArchiveManager()
    
    # Exemple de données d'analyse
    sample_results = [
        {
            'Symbol': 'AAPL',
            'Signal': 'ACHAT',
            'Score': 8.7,
            'ROE': 95.2,
            'PEG': 1.12,
        },
        {
            'Symbol': 'MSFT',
            'Signal': 'ACHAT',
            'Score': 9.1,
            'ROE': 87.5,
            'PEG': 0.98,
        },
        {
            'Symbol': 'GOOGL',
            'Signal': 'VENTE',
            'Score': 5.2,
            'ROE': 12.3,
            'PEG': 2.45,
        },
        {
            'Symbol': 'AMZN',
            'Signal': 'NEUTRE',
            'Score': 6.8,
            'ROE': 45.1,
            'PEG': 1.55,
        },
    ]
    
    sample_columns = ['Symbol', 'Signal', 'Score', 'ROE', 'PEG']
    
    # Exemple 1: Sauvegarder une analyse
    print("\n1️⃣  SAUVEGARDE D'UNE ANALYSE")
    archive_file = manager.save_daily_analysis(sample_results, sample_columns, tag="demo")
    
    # Exemple 2: Récupérer les excellentes opportunités
    print("\n2️⃣  EXCELLENTES OPPORTUNITÉS")
    manager.get_excellent_opportunities(min_score=8.5)
    
    # Exemple 3: Lister les archives
    print("\n3️⃣  ARCHIVES DISPONIBLES")
    manager.list_archives(days=7)
    
    # Exemple 4: Résumé de période
    print("\n4️⃣  RÉSUMÉ PÉRIODIQUE")
    manager.export_period_summary(days=7)
    
    print("\n✅ Démonstration complétée !")


if __name__ == "__main__":
    demo()
