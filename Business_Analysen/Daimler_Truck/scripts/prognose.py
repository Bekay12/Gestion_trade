#!/usr/bin/env python3
"""
prognose.py - Prognosetreue von GEA ueber fuenf Jahre (report-template.md, Teil 3).

Jeder Geschaeftsbericht enthaelt den Abschnitt "Forecast/actual comparison" mit drei
Groessen: organisches Umsatzwachstum, EBITDA-Marge vor Restrukturierung und ROCE, je als
urspruengliche Prognose, unterjaehrig angepasste Prognose und Ergebnis. Gelesen wird die
Prognose WIE ZUERST GEGEBEN und die angepasste getrennt - sonst misst die Tabelle nichts
(report-template.md: "Record the guidance as first given, not as later revised").

Die Zeilen des Vergleichs stehen im zweispaltigen Satz verstreut; gelesen werden deshalb
alle Prozentwerte der Zeile, die das Etikett traegt, in der gedruckten Reihenfolge:
urspruengliche Spanne, angepasste Spanne, Ergebnis.

Aufruf: python3 scripts/prognose.py
Ausgabe: data/prognose.json, data/prognose.tex (Tabellenkoerper)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAHRE = (2021, 2022, 2023, 2024, 2025)
# Die Prognosegroessen wechseln mit dem Jahrgang: bis 2023 ein EBITDA-Betrag in EUR und
# qualitative Angaben ("significantly rising"), ab 2024 die EBITDA-Marge in Prozent.
# Gelesen wird die Zelle, wie sie gedruckt ist; die Einordnung rechnet rechnung_gea.py
# gegen die eigene Reihe, nicht gegen die Ergebnisspalte der Tabelle.
ETIKETTEN = {
    "umsatz": r"Revenue development \(organic",
    "ebitda": r"EBITDA before restructuring expenses",
    "marge": r"EBITDA margin before restructuring expenses",
    "roce": r"ROCE",
}
_PROZENT = re.compile(r"[+-]?\d{1,3}\.\d(?=\s*%)|[+-]?\d{1,3}(?=\s*%)")


def seite_mit_vergleich(text: dict) -> str:
    """Seite des Prognose-Ist-Vergleichs; die Ueberschrift wechselt je Jahrgang."""
    for seite, t in text.items():
        if re.search(r"Revenue development \(organic", t) and re.search(r"(?i)forecast", t):
            return seite
    return None


ZELLE = r"(?:^|(?<=\s{2}))"


def zellen(zeile: str) -> list:
    """
    Tabellenzellen hinter dem Etikett. Die Seite setzt links Prosa und rechts die Tabelle
    in dieselbe Textzeile; gezaehlt werden nur Zellen, die eine Prozentangabe oder
    \"unchanged\" tragen - Prosa faellt damit heraus.
    """
    teile = [z.strip() for z in re.split(r"\s{2,}", zeile.strip()) if z.strip()]
    return [z for z in teile if re.search(r"(?i)%|unchanged|eur|million|rising|declining|stable", z)]


def werte(zeile: str) -> list:
    return [float(x) for x in _PROZENT.findall(zeile)]


def lies_bericht(jahr: int) -> dict:
    text = seiten(os.path.join(ROOT, "refs", f"gea-ar-{jahr}.txt"))
    seite = seite_mit_vergleich(text)
    if not seite:
        return {"jahr": jahr, "seite": None, "posten": {}}
    aus = {}
    for name, muster in ETIKETTEN.items():
        for zeile in text[seite].splitlines():
            treffer = list(re.finditer(ZELLE + muster, zeile))
            if not treffer:
                continue
            z = max((zellen(zeile[m.end():]) for m in treffer), key=len)
            # Erste Zelle: Prognose wie zuerst gegeben. Letzte: Ergebnis. Dazwischen die
            # Anpassungen (2024 wurde zweimal angepasst, 2025 einmal, 2022 gar nicht).
            if len(z) >= 2:
                aus[name] = {"erst": z[0], "zuletzt": z[-2] if len(z) > 2 else "--",
                             "ergebnis": z[-1], "spalten": z}
                break
    return {"jahr": jahr, "seite": seite, "posten": aus}


def main() -> int:
    alle = [lies_bericht(j) for j in JAHRE]
    json.dump(alle, open(os.path.join(ROOT, "data", "prognose.json"), "w"), indent=1,
              ensure_ascii=False)
    for b in alle:
        print(f"[PROGNOSE] {b['jahr']} S. {b['seite']}: " + " | ".join(
            f"{k}: erst {v['erst']}, zuletzt {v['zuletzt']}, Ergebnis {v['ergebnis']}"
            for k, v in b["posten"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
