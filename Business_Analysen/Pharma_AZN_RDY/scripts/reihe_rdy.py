#!/usr/bin/env python3
"""
reihe_rdy.py - baut die Mehrjahresreihe von Dr. Reddy's aus fuenf Form 20-F
und prueft sie ueber Kreuz.

Wie bei AstraZeneca druckt jede Einreichung drei Jahre, und die Ueberlappung
ergibt die Kettengleichungen. Zwei Eigenheiten dieses Emittenten:

  - Vor den drei Rupienspalten steht eine Umrechnung in US-Dollar. Sie wird
    ueber ueberspringen=1 verworfen, und zwar unabhaengig von ihrer
    Groessenordnung; eine Regel, die sich an der Stellenzahl orientiert,
    verschiebt frueher oder spaeter stillschweigend eine Spalte.
  - Das Geschaeftsjahr endet am 31. Maerz. FY2026 meint das Jahr zum
    31.03.2026. Die Jahreszahl im Dateinamen ist immer das Endjahr.

Aufruf: python3 scripts/reihe_rdy.py
Rueckgabewert 1, wenn eine Kettengleichung bricht.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeile import lies_werte                                 # noqa: E402

# Einreichung -> (Endjahr der ersten Spalte, Seite GuV, Seite Kapitalfluss)
BERICHTE = {
    2020: (2020, "128", "132"),
    2022: (2022, "120", "124"),
    2023: (2023, "122", "126"),
    2025: (2025, "121", "126"),
    2026: (2026, "112", "118"),
}
# Die Etiketten wechseln zwischen den Jahrgaengen: was 2026 "Purchase of
# property, plant and equipment" heisst, hiess 2022 "Expenditures on ..." und
# 2020 "Expenditure on ...". Deshalb je Posten eine Liste von Schreibweisen,
# die der Reihe nach versucht werden.
POSTEN = {
    "Umsatz":            (["Revenues"], "guv"),
    "Herstellkosten":    (["Cost of revenues"], "guv"),
    "Rohertrag":         (["Gross profit"], "guv"),
    "Vertriebskosten":   (["Selling, general and administrative expenses"], "guv"),
    "Forschung":         (["Research and development expenses"], "guv"),
    "Jahresueberschuss": (["Profit for the year"], "guv"),
    "OperativerCF":      (["Net cash from operating activities"], "cf"),
    "SachInvestition":   (["Purchase of property, plant and equipment",
                           "Expenditures on property, plant and equipment",
                           "Expenditure on property, plant and equipment"], "cf"),
    "ImmatInvestition":  (["Purchase of other intangible assets",
                           "Expenditures on other intangible assets"], "cf"),
    "SachAbgang":        (["Proceeds from sale of property, plant and equipment"], "cf"),
}


def sammle() -> dict:
    aus = {}
    for bericht, (erstes, s_guv, s_cf) in BERICHTE.items():
        datei = f"refs/rdy-20f-{bericht}.txt"
        for name, (etiketten, wo) in POSTEN.items():
            seite = s_guv if wo == "guv" else s_cf
            werte = None
            for etikett in etiketten:
                werte = lies_werte(datei, seite, etikett, 3, von_hinten=True)
                if werte:
                    break
            if not werte:
                print(f"[REIHE] {name}: Etikett nicht gefunden in 20-F {bericht}, S. {seite}")
                continue
            for versatz, wert in enumerate(werte):
                aus.setdefault((name, erstes - versatz), []).append(
                    (wert, f"Form 20-F {bericht}", seite))
    return aus


def main() -> int:
    alle = sammle()
    jahre = sorted({j for _, j in alle})
    brueche = 0
    print(f"{'Kennzahl (Mio. INR)':22s} " + " ".join(f"{j:>9d}" for j in jahre))
    for name in POSTEN:
        zeile, konflikt = [], []
        for j in jahre:
            lesungen = alle.get((name, j), [])
            werte = {w for w, _, _ in lesungen}
            if not werte:
                zeile.append("        -")
            elif len(werte) > 1:
                konflikt.append(f"{j}: {sorted(werte)} aus {[b for _, b, _ in lesungen]}")
                zeile.append("  KONFLIKT")
            else:
                zeile.append(f"{werte.pop():9,.0f}")
        print(f"{name:22s} " + " ".join(zeile))
        for k in konflikt:
            print(f"    BRUCH {name} {k}")
            brueche += 1
    gleichungen = sum(len(v) - 1 for v in alle.values() if len(v) > 1)
    print(f"\n[REIHE] {len(alle)} Jahreswerte, {gleichungen} Kettengleichungen, "
          f"{brueche} Brueche")
    return 1 if brueche else 0


if __name__ == "__main__":
    sys.exit(main())
