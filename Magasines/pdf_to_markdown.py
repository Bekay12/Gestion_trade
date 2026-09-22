"""
pdf_to_markdown - convertit les magazines PDF de PDFs/ en Markdown dans MARKDOWNS/,
nommés <Magazine>_<Titre>_<Date>.md.

Titre : table titres.json (titre de couverture), sinon titre lisible dans le nom du fichier.
Date  : date de parution lue sur la couverture, sinon date AAAAMMJJ du nom de fichier,
        sinon date de modification du fichier (téléchargement).

Usage :
    .venv_new/bin/python Magasines/pdf_to_markdown.py [--force] [--jobs N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from markitdown import MarkItDown

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "PDFs"
OUT_DIR = BASE_DIR / "MARKDOWNS"
TITRES_FILE = BASE_DIR / "titres.json"

# Préfixe du nom de fichier -> nom de magazine utilisé dans la sortie.
MAGAZINES = {
    "boerseonline": "Boerse_Online",
    "capital": "Capital",
    "cash": "Cash",
    "euro": "Euro",
    "fortuneeu": "Fortune_Europe",
    "teleskop": "Teleskop",
}

MOIS = {
    "januar": 1, "january": 1, "februar": 2, "february": 2, "märz": 3, "maerz": 3,
    "march": 3, "april": 4, "mai": 5, "may": 5, "juni": 6, "june": 6, "juli": 7,
    "july": 7, "august": 8, "september": 9, "oktober": 10, "october": 10,
    "november": 11, "dezember": 12, "december": 12,
}

HASH_RE = re.compile(r"[0-9a-f]{32}", re.I)
FILE_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


def _cover_text(pdf: Path) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Texte de la première page (couche texte, sans OCR).

    Inputs:
        pdf (Path): fichier PDF

    Outputs:
        text (str): texte de la couverture, "" si illisible
    --------------------------------------------------------------------------
    """
    res = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", str(pdf), "-"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout


def detect_magazine(pdf: Path, cover: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Nom du magazine, d'après le préfixe du fichier puis la couverture.

    Inputs:
        pdf (Path): fichier PDF
        cover (str): texte de couverture

    Outputs:
        magazine (str): nom normalisé ("Magazine" si inconnu)
    --------------------------------------------------------------------------
    """
    prefix = pdf.name.split("_", 1)[0].lower()
    if prefix in MAGAZINES:
        return MAGAZINES[prefix]
    if "DAS MAGAZIN FÜR WIRTSCHAFT UND GELD" in cover:
        return "Euro"
    if "Börsenmagazin" in cover:
        return "Boerse_Online"
    return "Magazine"


def detect_title(pdf: Path, titres: dict[str, str]) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Titre : table titres.json, sinon partie lisible du nom de fichier
        (sans préfixe magazine, date AAAAMMJJ ni hash).

    Inputs:
        pdf (Path): fichier PDF
        titres (dict): nom de fichier -> titre de couverture

    Outputs:
        title (str): titre brut (non encore nettoyé pour le système de fichiers)
    --------------------------------------------------------------------------
    """
    # Clés comparées à espaces normalisés : certains noms contiennent \r\n.
    norm = {re.sub(r"\s+", " ", k): v for k, v in titres.items()}
    key = re.sub(r"\s+", " ", pdf.name)
    if key in norm:
        return norm[key]
    stem = pdf.stem
    parts = stem.split("_", 1)
    if parts[0].lower() in MAGAZINES and len(parts) == 2:
        stem = parts[1]
    stem = FILE_DATE_RE.sub("", HASH_RE.sub("", stem))
    stem = re.sub(r"[_\s]+", " ", stem).strip()
    return stem or "Ausgabe"


def detect_date(pdf: Path, cover: str) -> tuple[str, str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Date de parution lue sur la couverture ; repli sur la date du nom de
        fichier puis sur la date de modification (téléchargement).

    Inputs:
        pdf (Path): fichier PDF
        cover (str): texte de couverture

    Outputs:
        (date, source) (tuple[str, str]): date ISO (AAAA-MM-JJ ou AAAA-MM) et
            son origine ("couverture", "nom_fichier", "telechargement")
    --------------------------------------------------------------------------
    """
    # Börse Online : "13.11.–19.11.2025" -> premier jour de la semaine de parution.
    m = re.search(r"(\d{2})\.(\d{2})\.\s*[–-]\s*\d{2}\.\d{2}\.(\d{4})", cover)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}", "couverture"
    # Capital / Cash / Fortune : "DEZ E M B E R 2 02 5" (lettres espacées) -> mois + année.
    for line in cover.splitlines():
        flat = re.sub(r"\s+", "", line).lower()
        m = re.search(r"(" + "|".join(MOIS) + r")(20\d{2})", flat)
        if m:
            return f"{m.group(2)}-{MOIS[m.group(1)]:02d}", "couverture"
    # Euro : "Ausgabe 6/2026" ; Teleskop : "Nr. 11 06/2025".
    m = re.search(r"(?:Ausgabe|Nr\.\s*\d+)\s+(\d{1,2})/(20\d{2})", cover)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}", "couverture"
    m = FILE_DATE_RE.search(pdf.stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "nom_fichier"
    mtime = dt.date.fromtimestamp(pdf.stat().st_mtime)
    return mtime.isoformat(), "telechargement"


def _slug(text: str) -> str:
    """Nettoie un fragment pour un nom de fichier (garde les umlauts)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_-")


def convert_one(pdf: Path, out_path: Path, meta: dict[str, str]) -> tuple[str, int]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Convertit un PDF en Markdown avec MarkItDown et ajoute un en-tête YAML.
        MarkItDown ne conserve pas les sauts de page : le nombre de pages vient
        de pdfinfo.

    Inputs:
        pdf (Path): fichier source
        out_path (Path): fichier Markdown à écrire
        meta (dict): magazine, titre, date, source de la date

    Outputs:
        (nom, pages) (tuple[str, int]): nom du fichier écrit, nombre de pages
    --------------------------------------------------------------------------
    """
    body = MarkItDown().convert(str(pdf)).text_content.strip()
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False)
    m = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
    n_pages = int(m.group(1)) if m else 0
    src_name = re.sub(r"\s+", " ", pdf.name)
    header = (
        "---\n"
        f"magazine: {meta['magazine']}\n"
        f"titre: \"{meta['titre']}\"\n"
        f"date: {meta['date']}\n"
        f"date_source: {meta['date_source']}\n"
        f"source_pdf: \"{src_name}\"\n"
        f"pages_pdf: {n_pages}\n"
        f"converti_le: {dt.date.today().isoformat()}\n"
        "---\n\n"
        f"# {meta['magazine'].replace('_', ' ')}: {meta['titre']} ({meta['date']})\n\n"
    )
    out_path.write_text(header + body + "\n", encoding="utf-8")
    return out_path.name, n_pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true", help="réécrire les .md existants")
    parser.add_argument("--jobs", type=int, default=4, help="conversions en parallèle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    titres = json.loads(TITRES_FILE.read_text(encoding="utf-8")) if TITRES_FILE.exists() else {}
    OUT_DIR.mkdir(exist_ok=True)
    pdfs = sorted(p for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf")

    jobs: list[tuple[Path, Path, dict[str, str]]] = []
    used: set[str] = set()
    for pdf in pdfs:
        cover = _cover_text(pdf)
        magazine = detect_magazine(pdf, cover)
        titre = re.sub(r"\s+", " ", detect_title(pdf, titres)).strip()
        date, source = detect_date(pdf, cover)
        name = f"{magazine}_{_slug(titre)}_{date}"
        n, candidate = 2, name
        while candidate in used:  # deux PDF donnant le même nom
            candidate, n = f"{name}_{n}", n + 1
        used.add(candidate)
        out = OUT_DIR / f"{candidate}.md"
        if out.exists() and not args.force:
            logger.info("[SKIP] %s existe déjà", out.name)
            continue
        jobs.append((pdf, out, {"magazine": magazine, "titre": titre,
                                "date": date, "date_source": source}))

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(convert_one, *j): j[0] for j in jobs}
        for fut in as_completed(futures):
            pdf = futures[fut]
            try:
                name, n = fut.result()
                logger.info("[OK] %s -> %s (%d pages)", re.sub(r"\s+", " ", pdf.name), name, n)
            except Exception as exc:  # un PDF défectueux ne bloque pas les autres
                logger.error("[ERREUR] %s : %s", re.sub(r"\s+", " ", pdf.name), exc)


if __name__ == "__main__":
    main()
