"""
Fusionne tous les PDFs de la formation Academy-Germain dans l'ordre correct :
Partie0 → Partie1 → ... → Partie12 → PARTIE 13
puis par numéro de chapitre à l'intérieur de chaque partie.
"""

import os
import re
from pathlib import Path

try:
    from pypdf import PdfWriter
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    from pypdf import PdfWriter


def get_partie_order(folder_name: str) -> int:
    """Retourne l'index numérique de la partie pour le tri."""
    name = folder_name.upper()
    match = re.search(r'PARTIE\s*(\d+)', name)
    if match:
        return int(match.group(1))
    return 999


def get_chapter_order(filename: str) -> int:
    """Retourne le numéro de chapitre extrait du nom de fichier."""
    match = re.search(r'_Ch(\d+)_', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 999


BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "Academy_Germain_COMPLET.pdf"

# Trouver tous les sous-dossiers Partie*/PARTIE*/
partie_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]

# Trier par numéro de partie
partie_dirs.sort(key=lambda d: get_partie_order(d.name))

pdf_files = []

for partie_dir in partie_dirs:
    pdf_subdir = partie_dir / "PDF"
    if not pdf_subdir.exists():
        continue
    # Lister les PDFs dans ce dossier, triés par numéro de chapitre
    pdfs = sorted(pdf_subdir.glob("*.pdf"), key=lambda f: get_chapter_order(f.name))
    pdf_files.extend(pdfs)

print(f"Nombre de PDFs trouvés : {len(pdf_files)}")
print("\nOrdre de fusion :")
for i, f in enumerate(pdf_files, 1):
    print(f"  {i:3}. {f.parent.parent.name} / {f.name}")

print(f"\nFusion en cours → {OUTPUT_FILE.name} ...")

writer = PdfWriter()

for pdf_path in pdf_files:
    try:
        writer.append(str(pdf_path))
        print(f"  ✓ {pdf_path.name}")
    except Exception as e:
        print(f"  ✗ ERREUR sur {pdf_path.name} : {e}")

with open(OUTPUT_FILE, "wb") as f_out:
    writer.write(f_out)

print(f"\nFusion terminée ! Fichier créé : {OUTPUT_FILE}")
print(f"Taille : {OUTPUT_FILE.stat().st_size / 1_000_000:.1f} Mo")
