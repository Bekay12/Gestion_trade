#!/usr/bin/env python3
"""
reihe_azn.py - baut die Mehrjahresreihe von AstraZeneca aus fuenf Berichten
und prueft sie ueber Kreuz.

Jeder Geschaeftsbericht druckt drei Jahre. Fuenf Berichte ergeben deshalb
fuer die meisten Jahre zwei oder drei unabhaengige Lesungen, und die
Kettenpruefung verlangt, dass sie uebereinstimmen. Sie faengt genau den
Fehler, den der Seitenwachhund nicht sehen kann: eine Spalte zu weit rechts
gegriffen, der Wert steht auf der zitierten Seite und ist trotzdem der
eines anderen Jahres.

Aufruf: python3 scripts/reihe_azn.py
Rueckgabewert 1, wenn eine Kettengleichung bricht.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeile import lies_werte                                 # noqa: E402

# Bericht -> (Jahr der ersten Spalte, Seite GuV, Seite Kapitalfluss)
BERICHTE = {
    2021: (2021, "134", "137"),
    2022: (2022, "138", "141"),
    2023: (2023, "148", "151"),
    2024: (2024, "148", "151"),
    2025: (2025, "125", "128"),
}
# Kennzahl -> (Etikett, "guv"|"cf", zu ueberspringende Vorspalten)
POSTEN = {
    "Umsatz":            ("Total Revenue", "guv", 0),
    "Betriebsergebnis":  ("Operating profit", "guv", 0),
    "Jahresueberschuss": ("Profit for the period", "guv", 0),
    "ErgebnisJeAktie":   ("Basic earnings per $0.25 Ordinary Share", "guv", 1),
    "AktienGewichtet":   ("Weighted average number of Ordinary Shares in issue (millions)",
                          "guv", 1),
    "OperativerCF":      ("Net cash inflow from operating activities", "cf", 0),
    "SachInvestition":   ("Purchase of property, plant and equipment", "cf", 0),
    "ImmatInvestition":  ("Purchase of intangible assets", "cf", 0),
    "ImmatAbgang":       ("Disposal of intangible assets", "cf", 0),
    "LeasingTilgung":    ("Repayment of obligations under leases", "cf", 0),
    "DividendeGezahlt":  ("Dividends paid", "cf", 0),
}


def sammle() -> dict:
    """{(Posten, Jahr): [(Wert, Bericht, Seite), ...]} - alle Lesungen."""
    aus = {}
    for bericht, (erstes, s_guv, s_cf) in BERICHTE.items():
        datei = f"refs/azn-ar-{bericht}.txt"
        for name, (etikett, wo, ueber) in POSTEN.items():
            seite = s_guv if wo == "guv" else s_cf
            werte = lies_werte(datei, seite, etikett, 3, von_hinten=True)
            if not werte:
                print(f"[REIHE] {name}: Etikett nicht gefunden in AR{bericht}, S. {seite}")
                continue
            for versatz, wert in enumerate(werte):
                aus.setdefault((name, erstes - versatz), []).append(
                    (wert, f"Geschaeftsbericht {bericht}", seite))
    return aus


def main() -> int:
    alle = sammle()
    jahre = sorted({j for _, j in alle})
    brueche = 0
    print(f"{'Kennzahl':20s} " + " ".join(f"{j:>9d}" for j in jahre))
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
                w = werte.pop()
                zeile.append(f"{w:9,.2f}" if abs(w) < 100 else f"{w:9,.0f}")
        print(f"{name:20s} " + " ".join(zeile))
        for k in konflikt:
            print(f"    BRUCH {name} {k}")
            brueche += 1
    gleichungen = sum(len(v) - 1 for v in alle.values() if len(v) > 1)
    print(f"\n[REIHE] {len(alle)} Jahreswerte, {gleichungen} Kettengleichungen, "
          f"{brueche} Brueche")
    return 1 if brueche else 0


if __name__ == "__main__":
    sys.exit(main())
