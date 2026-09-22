#!/usr/bin/env python3
"""
kennzahlen_seite.py - liest die Kennzahlenseite (Seite 2) jedes Geschaeftsberichts.

Dort stehen die Groessen, die der Abschluss nicht in einer Zeile fuehrt: Auftragseingang,
Book-to-bill, Serviceanteil, EBITDA vor Restrukturierung und seine Marge, EBIT, ROCE,
freier Cashflow, Nettoliquiditaet, Ergebnis je Aktie, Mitarbeiter.

Zwei Fallen der Seite:
  * Der Jahrgang 2025 setzt zuerst das Quartal, dann das Gesamtjahr (sechs Zahlen je
    Zeile); aeltere Jahrgaenge nur das Gesamtjahr (drei Zahlen). Genommen werden die
    beiden Jahreswerte, nie die Quartalswerte.
  * "as % of revenue" steht zweimal auf der Seite (EBITDA-Marge und Net Working Capital).
    Die Marge wird deshalb als FOLGEZEILE der EBITDA-Zeile gelesen, nicht per Etikett.

Jeder Bericht druckt zwei Jahre; reihe-artige Kreuzpruefung in pruefe.py.

Aufruf: python3 scripts/kennzahlen_seite.py
Ausgabe: data/kennzahlen_seite.json {jahr: {posten: {"wert":, "bericht":, "seite":}}}
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lies import seiten                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERICHTE = (2020, 2021, 2022, 2023, 2024, 2025)
# Etikett am Anfang einer Tabellenzelle: Zeilenanfang oder zwei Leerzeichen davor. Die
# Seite setzt links einen Kommentarspalt und rechts die Tabelle in DIESELBE Textzeile;
# ohne die Zellengrenze las "Revenue" den Prozentwert aus dem Kommentar ("up 1.4 percent").
ZELLE = r"(?:^|(?<=\s{2}))"
ETIKETTEN = {
    "auftragseingang": r"Order intake\b",
    "booktobill": r"Book-to-bill ratio\b",
    "umsatz": r"Revenue\b",
    "organisches_wachstum": r"Organic revenue growth in %\d?",
    "serviceanteil": r"Share of service revenue in %\d?",
    "ebitda_vr": r"EBITDA before restructuring expenses\b",
    "ebitda": r"EBITDA\b(?! before)",
    "ebit": r"EBIT\b(?! before)",
    "roce": r"ROCE in %\d?",
    "freier_cashflow": r"Free cash flow\b",
    "nettoliquiditaet": r"Net liquidity[^\d]*",  # Etikett enthaelt "(+)/Net debt (-)"
    "eigenkapitalquote": r"Equity ratio in %\d?",
    "eps": r"Earnings per share \(EUR\)",
    "mitarbeiter": r"Employees[^\d]*",
    "capital_employed": r"Capital employed[^\d]*",
}
_ZAHL = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?")


def zahlen(rest: str) -> list:
    """
    Alle Zahlen der Zeile, auch die "bps"-Spalten: Die Stellung entscheidet, welche Zahl
    das Geschaeftsjahr traegt (bei sechs Zahlen die vierte und fuenfte). Wer die
    bps-Spalte vorher entfernt, verschiebt genau diese Stellung - dann stand die
    Quartalsmarge (16,7) statt der Jahresmarge (16,5) in der Reihe.
    """
    return [float(x.replace(",", "")) for x in _ZAHL.findall(rest)]


def _naechstes_wort(rest: str):
    """Erstes Wort, das eine neue Spalte beginnt; "bps" gehoert zur Zahl davor."""
    for m in re.finditer(r"[A-Za-z]{2,}", rest):
        if m.group(0).lower() != "bps":
            return m
    return None


def jahreswerte(zeile: str, muster: str) -> list:
    """
    Die zwei Jahreswerte. Bei sechs Zahlen stehen sie an vierter und fuenfter Stelle
    (der Jahrgang 2025 setzt das Quartal zuerst). Das Etikett kann mehrfach in der Zeile
    stehen - links im Kommentarspalt, rechts in der Tabelle -, deshalb werden alle
    Fundstellen geprueft und die erste genommen, die zwei Jahreswerte traegt.
    """
    for m in re.finditer(ZELLE + muster, zeile):
        w = _werte_ab(zeile, m.end())
        if w:
            return w
    return []


def _werte_ab(zeile: str, ab: int) -> list:
    rest = zeile[ab:]
    # Fussnotenzeichen direkt hinter dem Etikett ("Net debt (-)2   378.9"): einzelne
    # Ziffer, der der Spaltenabstand folgt - kein Wert.
    rest = re.sub(r"^\s*\d(?=\s{2})", "", rest)
    wort = _naechstes_wort(rest)                 # nicht in die Nachbarspalte lesen
    z = zahlen(rest[:wort.start()] if wort else rest)
    if len(z) >= 6:
        return [z[3], z[4]]
    if len(z) >= 2:
        return [z[0], z[1]]
    return []


def lies(jahr: int) -> dict:
    pfad = os.path.join(ROOT, "refs", f"gea-ar-{jahr}.txt")
    text = seiten(pfad)
    seite = next((s for s, t in text.items() if re.search(r"(?i)financial key figures", t[:3000])), "2")
    zeilen = [z.rstrip() for z in text[seite].splitlines()]   # Spaltenabstaende erhalten
    aus = {}
    for i, z in enumerate(zeilen):
        for name, muster in ETIKETTEN.items():
            if name in aus or not re.search(ZELLE + muster, z):
                continue
            w = jahreswerte(z, muster)
            if len(w) == 2:
                aus[name] = w
            if name == "ebitda_vr" and len(w) == 2:
                # Marge = Folgezeile "as % of revenue"
                for folge in zeilen[i + 1:i + 3]:
                    if re.search(ZELLE + r"as % of revenue", folge):
                        m = jahreswerte(folge, r"as % of revenue")
                        if len(m) == 2:
                            aus["ebitda_marge"] = m
                        break
    return {"seite": seite, "posten": aus}


def main() -> int:
    reihen = {}
    for jahr in BERICHTE:
        b = lies(jahr)
        for posten, (aktuell, vorjahr) in b["posten"].items():
            for j, w in ((jahr, aktuell), (jahr - 1, vorjahr)):
                eintrag = {"wert": w, "bericht": f"gea-ar-{jahr}", "seite": b["seite"]}
                # juengster Bericht gewinnt; abweichende Lesung wird gemeldet
                alt = reihen.setdefault(str(j), {}).get(posten)
                if alt and abs(alt["wert"] - w) > max(0.005 * abs(w), 0.01) and alt["bericht"] > eintrag["bericht"]:
                    alt.setdefault("abweichend", []).append([w, eintrag["bericht"]])
                elif not alt or eintrag["bericht"] >= alt["bericht"]:
                    if alt and abs(alt["wert"] - w) > max(0.005 * abs(w), 0.01):
                        eintrag["abweichend"] = [[alt["wert"], alt["bericht"]]]
                    reihen[str(j)][posten] = eintrag
        print(f"[KENNZAHLEN] gea-ar-{jahr} S. {b['seite']}: {len(b['posten'])} Posten")
    json.dump(reihen, open(os.path.join(ROOT, "data", "kennzahlen_seite.json"), "w"),
              indent=1, ensure_ascii=False)
    for j in sorted(reihen):
        p = reihen[j]
        print(f"  {j}: Auftrag {p.get('auftragseingang', {}).get('wert')}, "
              f"Umsatz {p.get('umsatz', {}).get('wert')}, EBITDA vR {p.get('ebitda_vr', {}).get('wert')}, "
              f"Marge {p.get('ebitda_marge', {}).get('wert')}, ROCE {p.get('roce', {}).get('wert')}, "
              f"FCF {p.get('freier_cashflow', {}).get('wert')}, Netto {p.get('nettoliquiditaet', {}).get('wert')}")
    abw = [(j, p, v["abweichend"]) for j, ps in reihen.items() for p, v in ps.items() if v.get("abweichend")]
    for j, p, a in abw:
        print(f"[ABWEICHUNG] {j} {p}: {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
