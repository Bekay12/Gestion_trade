"""Seitenweiser Textindex der Primaerquellen.

Jede Seite wird einzeln extrahiert, damit die GEDRUCKTE Seitenzahl aus der
Fusszeile gelesen und nicht aus einer Zeilennummer abgeleitet wird (Regel 1 der
Skill: abgeleitete Seitenzahlen erzeugten im Referenzfall fuenf Fehlzitate).
"""
import re
import subprocess
from pathlib import Path

QUELLEN = Path(__file__).parent / 'quellen'
CACHE = Path(__file__).parent / 'seiten'


def seiten_zahl(pdf: Path) -> int:
    aus = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    return int(re.search(r'Pages:\s+(\d+)', aus).group(1))


def dump(pdf: Path) -> int:
    ziel = CACHE / pdf.stem
    ziel.mkdir(parents=True, exist_ok=True)
    n = seiten_zahl(pdf)
    for p in range(1, n + 1):
        texte = subprocess.run(
            ['pdftotext', '-layout', '-f', str(p), '-l', str(p), str(pdf), '-'],
            capture_output=True, text=True).stdout
        (ziel / f'{p:04d}.txt').write_text(texte)
    return n


if __name__ == '__main__':
    for pdf in sorted(QUELLEN.glob('*.pdf')):
        print(f'{pdf.name}: {dump(pdf)} Seiten extrahiert')
