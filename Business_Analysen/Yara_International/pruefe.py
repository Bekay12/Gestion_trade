"""Guard: jeder Wert MUSS auf der zitierten gedruckten Seite vorkommen.

Regel 1 der Skill verlangt einen Automatismus statt Sorgfalt: Urteilsvermoegen
versagt bei dieser Menge, ein Skript nicht. spellings() erzeugt die Schreib-
weisen, die Geschaeftsberichte tatsaechlich verwenden (Tausendertrennung,
Klammern fuer negative Werte), sonst meldet die Pruefung bei jedem gruppierten
Betrag einen Fehlalarm.
"""
from pathlib import Path

from werte import WERTE

SEITEN = Path(__file__).parent / 'seiten'
DATEI = {'GB2025': 'yara-annual-report-2025', 'Q2-2026': 'yara-2q-2026-report'}


def schreibweisen(wert):
    """Alle Formen, in denen ein Bericht diesen Wert setzen kann."""
    formen = set()
    if isinstance(wert, int) or (isinstance(wert, float) and wert.is_integer()):
        n = int(wert)
        formen |= {str(n), f"{n:,}", f"({n})", f"({n:,})", f"({n:,}", f"({n}"}
    else:
        formen |= {f"{wert}", f"{wert:,}", f"({wert})"}
        if abs(wert - round(wert, 1)) < 1e-9:
            formen.add(f"{wert:.1f}")
        if abs(wert - round(wert, 2)) < 1e-9:
            formen.add(f"{wert:.2f}")
    return formen


def seitentext(quelle: str, seite: int) -> str:
    # Gedruckte Seite = PDF-Seite, geprueft an der Kopfzeile beider Berichte.
    return (SEITEN / DATEI[quelle] / f'{seite:04d}.txt').read_text()


def main() -> int:
    fehler = []
    for name, (wert, quelle, seite) in WERTE.items():
        text = seitentext(quelle, seite)
        if not any(f in text for f in schreibweisen(wert)):
            fehler.append(f"  {name} = {wert}: nicht auf {quelle}, S. {seite}")
    print(f"{len(WERTE) - len(fehler)} von {len(WERTE)} Werten auf der zitierten Seite belegt")
    if fehler:
        print("NICHT BELEGT:")
        print("\n".join(fehler))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
