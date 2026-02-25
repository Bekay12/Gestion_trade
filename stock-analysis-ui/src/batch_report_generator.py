#!/usr/bin/env python3
"""
Script de génération automatisée de rapports PDF
Permet de générer des rapports PDF en batch depuis les données d'analyse

Usage:
    python3 batch_report_generator.py --results-file results.json
    python3 batch_report_generator.py --schedule daily  # Génération quotidienne
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_reports.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchReportGenerator:
    """Générateur de rapports en batch"""
    
    def __init__(self, results_dir=None):
        """Initialiser le générateur batch"""
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).parent / "Results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Importer le générateur de PDF
        try:
            from pdf_generator import PDFReportGenerator
            self.pdf_generator = PDFReportGenerator(self.results_dir)
            logger.info("✅ Générateur PDF chargé avec succès")
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer le générateur PDF: {e}")
            raise
    
    def load_results_from_json(self, json_file):
        """Charger les résultats depuis un fichier JSON"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Résultats chargés depuis {json_file}")
            return data.get('current_results', []), data.get('clean_columns', [])
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du JSON: {e}")
            return None, None
    
    def save_results_to_json(self, current_results, clean_columns, output_file=None):
        """Sauvegarder les résultats en JSON pour batch processing"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.results_dir / f"results_export_{timestamp}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'current_results': current_results,
            'clean_columns': clean_columns,
            'total_results': len(current_results)
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Résultats sauvegardés dans {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde JSON: {e}")
            return None
    
    def generate_report_from_json(self, json_file, dry_run=False):
        """Générer un rapport PDF depuis un fichier JSON avec résultats"""
        logger.info(f"📊 Génération de rapport depuis {json_file}")
        
        current_results, clean_columns = self.load_results_from_json(json_file)
        if not current_results or not clean_columns:
            logger.error("❌ Impossible de charger les résultats du JSON")
            return False
        
        if dry_run:
            logger.info("🔍 Mode DRY-RUN: pas de génération de PDF")
            logger.info(f"   - Résultats trouvés: {len(current_results)}")
            logger.info(f"   - Colonnes: {len(clean_columns)}")
            return True
        
        # Note: La génération PDF nécessite des figures matplotlib
        # Cette fonction est surtout un template pour montrer comment utiliser le module
        logger.warning("⚠️ La génération PDF nécessite les figures matplotlib (plots_layout)")
        logger.info("   Pour utiliser cette fonction complètement, passez les figures depuis l'interface")
        return False
    
    def list_available_reports(self):
        """Lister les rapports disponibles"""
        pdfs = list(self.results_dir.glob("*.pdf"))
        jsons = list(self.results_dir.glob("results_export_*.json"))
        
        logger.info(f"\n📋 Rapports disponibles dans {self.results_dir}")
        logger.info(f"   PDFs: {len(pdfs)}")
        for pdf in sorted(pdfs)[-5:]:
            size_mb = pdf.stat().st_size / (1024*1024)
            logger.info(f"      • {pdf.name} ({size_mb:.2f} MB)")
        
        logger.info(f"   JSONs (exports): {len(jsons)}")
        for js in sorted(jsons)[-5:]:
            logger.info(f"      • {js.name}")
        
        return len(pdfs), len(jsons)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Générateur automatisé de rapports PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Lister les rapports disponibles
  python3 batch_report_generator.py --list
  
  # Charger et afficher un fichier JSON de résultats
  python3 batch_report_generator.py --load results.json --dry-run
  
  # Afficher les statistiques des résultats
  python3 batch_report_generator.py --load results.json --stats
        """
    )
    
    parser.add_argument('--results-dir', default=None, help='Dossier des résultats')
    parser.add_argument('--load', help='Charger les résultats depuis un fichier JSON')
    parser.add_argument('--list', action='store_true', help='Lister les rapports disponibles')
    parser.add_argument('--dry-run', action='store_true', help='Mode simulation')
    parser.add_argument('--stats', action='store_true', help='Afficher les statistiques')
    
    args = parser.parse_args()
    
    try:
        generator = BatchReportGenerator(args.results_dir)
        
        if args.list:
            pdfs, jsons = generator.list_available_reports()
            logger.info(f"\n✅ Total: {pdfs} PDFs + {jsons} JSONs")
            return 0
        
        if args.load:
            results, columns = generator.load_results_from_json(args.load)
            if results is None:
                return 1
            
            if args.dry_run:
                logger.info(f"✅ Chargement réussi (mode DRY-RUN)")
                logger.info(f"   - Symboles: {len(results)}")
                logger.info(f"   - Colonnes: {len(columns)}")
                
                # Afficher quelques statistiques
                achats = sum(1 for r in results if r.get('Signal') == 'ACHAT')
                ventes = sum(1 for r in results if r.get('Signal') == 'VENTE')
                logger.info(f"   - Signaux ACHAT: {achats}")
                logger.info(f"   - Signaux VENTE: {ventes}")
            
            if args.stats:
                logger.info("\n📊 Statistiques des résultats:")
                for col in columns[:10]:  # Afficher les 10 premières colonnes
                    non_empty = sum(1 for r in results if r.get(col))
                    logger.info(f"   • {col}: {non_empty}/{len(results)} remplis")
            
            return 0
        
        logger.info("✅ Générateur de rapports initialisé avec succès")
        logger.info(f"   Dossier: {generator.results_dir}")
        logger.info("\nUtilisez --help pour voir les options disponibles")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
