"""
split_course - Split the Academy Germain course PDF into per-chapter text files.

Working tool, not a deliverable: it produces a local extraction cache under
docu/_extraction/ so each chapter can be read and synthesised individually.
The course carries its table of contents as plain text on its first pages
rather than as a PDF outline, so the chapter boundaries are parsed from there.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE = REPO_ROOT / "docu" / "Livres" / "Academy_Germain_COMPLET.pdf"
OUT_DIR = REPO_ROOT / "docu" / "_extraction"
TOC_PAGES = 3

# Repeated on every page; stripped so the chapter text reads as prose.
_BOILERPLATE = (
    "ACADEMY GERMAIN",
    "Ecole de Formation aux Marches Financiers",
    "École de Formation aux Marchés Financiers",
    "Academy Germain - Formation aux Marches Financiers",
    "Academy Germain – Formation aux Marchés Financiers",
    "Fait par Germain Lionel",
)


def parse_toc(doc: pymupdf.Document) -> list[dict]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Read the printed table of contents and turn it into chapter records
        with their start page.

    Inputs:
        doc (pymupdf.Document): the opened course PDF

    Outputs:
        chapters (list[dict]): part, chapter, title and start page, in order
    --------------------------------------------------------------------------
    """
    text = "\n".join(doc[i].get_text() for i in range(TOC_PAGES))
    lines = text.splitlines()
    chapters: list[dict] = []
    part = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        part_match = re.match(r"^Partie\s+(\d+)", stripped)
        if part_match:
            part = int(part_match.group(1))
            continue

        chapter_match = re.match(r"^Ch\.(\d+)\s+(.*)", stripped)
        if not chapter_match:
            continue
        title = re.sub(r"\.{3,}.*$", "", chapter_match.group(2)).strip()
        # The page number lands on one of the next lines, alone.
        for candidate in lines[index + 1 : index + 4]:
            if candidate.strip().isdigit():
                chapters.append({
                    "part": part,
                    "chapter": int(chapter_match.group(1)),
                    "title": title,
                    "start": int(candidate.strip()),
                })
                break
    return chapters


def clean(text: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Remove the page furniture repeated on every page of the course.

    Inputs:
        text (str): raw page text

    Outputs:
        text (str): the same text without header and footer lines
    --------------------------------------------------------------------------
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in _BOILERPLATE:
            continue
        # Page markers such as "- 12 -" and bare "PARTIE 7  -  Ch.02" headers.
        if re.fullmatch(r"[-–]\s*\d+\s*[-–]", stripped):
            continue
        if re.fullmatch(r"(PARTIE|Partie)\s+\d+\s*[-–]\s*Ch\.\d+", stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def export(parts: set[int] | None = None) -> list[Path]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Write one text file per chapter, bounded by the next chapter's start.

    Inputs:
        parts (set[int] | None): restrict to these part numbers, None for all

    Outputs:
        written (list[Path]): the files created
    --------------------------------------------------------------------------
    """
    doc = pymupdf.open(COURSE)
    chapters = parse_toc(doc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, chapter in enumerate(chapters):
        if parts is not None and chapter["part"] not in parts:
            continue
        start = chapter["start"] - 1  # printed numbering is 1-indexed
        end = chapters[index + 1]["start"] - 1 if index + 1 < len(chapters) else len(doc)
        body = "\n".join(clean(doc[page].get_text()) for page in range(start, min(end, len(doc))))

        slug = re.sub(r"[^a-z0-9]+", "-", chapter["title"].lower()).strip("-")[:50]
        path = OUT_DIR / f"p{chapter['part']:02d}-ch{chapter['chapter']:02d}-{slug}.md"
        header = (
            f"# Partie {chapter['part']} / Ch.{chapter['chapter']:02d} — {chapter['title']}\n"
            f"<!-- source: Academy_Germain_COMPLET.pdf p.{chapter['start']}-{end} -->\n\n"
        )
        path.write_text(header + body, encoding="utf-8")
        written.append(path)

    doc.close()
    return written


if __name__ == "__main__":
    selected = {int(a) for a in sys.argv[1:]} or None
    files = export(selected)
    total = sum(len(f.read_text(encoding="utf-8").split()) for f in files)
    print(f"{len(files)} chapitres extraits, {total} mots -> {OUT_DIR}")
