#!/usr/bin/env python3
"""
reihe.py - baut die Mehrjahresreihe 2016-2025 aus den Kennzahlenseiten.

Warum ein eigenes Skript: Die Kennzahlenseite jedes MD&A setzt VIER Spalten -
Q4 des Berichtsjahres, Q4 des Vorjahres, Geschaeftsjahr, Vorjahr. Die dritte
ist die gesuchte. Zeilen mit einem Fussnotenzeichen tragen dieses Zeichen
jedoch als zusaetzliche kleine Zahl vor den Spalten und verschieben alles um
eine Stelle: "Gold production (ounces)" lieferte im MD&A 2016 die Folge
[1, 105676, 104734, 392000, 380000], und die dritte Stelle war damit die
Quartalsproduktion statt der Jahresproduktion - ein um den Faktor vier
falscher Wert, den kein Seitenbeleg auffaengt, weil er auf der Seite steht.

Die Abwehr ist die Kette: Der Vorjahreswert eines Berichts muss dem
Berichtsjahreswert des Vorberichts gleichen. Zehn Berichte ergeben neun
solche Gleichungen je Kennzahl; wo sie nicht aufgeht, ist die Spalte falsch
gewaehlt und nicht der Wert falsch abgedruckt.

Aufruf: python3 scripts/reihe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten, werte  # noqa: E402

JAHRE = list(range(2016, 2026))
# Kennzahl -> (Etikett auf der Kennzahlenseite, Toleranz der Kettenpruefung)
FELDER = {
    "umsatz": ("Operating revenues", 0.05),
    "opcf": ("Cash provided by operating activities", 0.05),
    "produktion": ("Gold production (ounces)", 1.0),
    "aisc": ("All-in sustaining costs per ounce of gold sold", 1.0),
    "cashkosten": ("Total cash costs per ounce of gold sold", 1.0),
}


def spalten(text: str, etikett: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt [Jahr, Vorjahr] der Kennzahlenzeile zurueck. Fuehrt die Zeile ein
        Fussnotenzeichen, wird es verworfen: Es ist als einstellige Zahl vor
        einer Folge betragsmaessig viel groesserer Werte erkennbar.

    Inputs:
        text (str): Text der Kennzahlenseite
        etikett (str): Zeilenbeschriftung

    Outputs:
        paar (list): [Berichtsjahr, Vorjahr] oder None
    --------------------------------------------------------------------------
    """
    v = werte(text, etikett, 5)
    if not v or len(v) < 4:
        return None
    if len(v) == 5 and v[0] < 10 and v[0] == int(v[0]) and abs(v[1]) > 10 * max(v[0], 1):
        v = v[1:]
    return [v[2], v[3]]


# Anerkannte Kettenbrueche. Ein Bruch ist nicht immer ein Fehler: Er kann eine
# Neuberechnung des Unternehmens sein. Solche Faelle werden hier mit Grund
# eingetragen, damit die Pruefung nicht dauerhaft rot steht - und damit ein
# NEUER Bruch weiterhin auffaellt. Ein Wachhund, den man wegen eines bekannten
# Befundes abschaltet, schuetzt danach vor gar nichts mehr.
BEKANNT = {
    ("aisc", 2025): (
        "Neuberechnung: Der Bericht 2025 nimmt die Marktbewertungseffekte der "
        "aktienbasierten Verguetung aus der Kennzahl heraus und stellt die "
        "Vergleichszahlen entsprechend um (MD&A 2025, S. 41). Im Dokument "
        "benannt in Abschnitt 2.2."),
}


def main() -> int:
    roh = {}
    for jahr in JAHRE:
        seite = seiten(f"refs/agi-mda-{jahr}.txt").get("4", "")
        roh[jahr] = {k: spalten(seite, et) for k, (et, _) in FELDER.items()}

    fehler, anerkannt = 0, 0
    for feld, (_, tol) in FELDER.items():
        kopf = f"{feld:12s}"
        werte_j = []
        for jahr in JAHRE:
            paar = roh[jahr][feld]
            werte_j.append(paar[0] if paar else None)
        print(f"{kopf} " + " ".join(
            f"{w:>10.1f}" if isinstance(w, float) else f"{'--':>10}" for w in werte_j))
        # Kettenpruefung: Vorjahr des Berichts gegen Berichtsjahr des Vorberichts
        for i, jahr in enumerate(JAHRE[1:], start=1):
            paar = roh[jahr][feld]
            vorher = werte_j[i - 1]
            if not paar or vorher is None:
                continue
            if abs(paar[1] - vorher) > tol:
                grund = BEKANNT.get((feld, jahr))
                marke = "ANERKANNT" if grund else "KETTE    "
                print(f"   {marke} {feld} {jahr}: Vorjahresspalte {paar[1]} "
                      f"gegen Bericht {JAHRE[i-1]} {vorher}")
                if grund:
                    print(f"             {grund}")
                    anerkannt += 1
                else:
                    fehler += 1
    print(f"\n[REIHE] {fehler} unerklaerte Kettenbrueche, "
          f"{anerkannt} anerkannt")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
