#!/usr/bin/env python3
"""
reihe.py - Mehrjahresreihe 2017-2025 aus den Kennzahlenseiten und Kapitalflussrechnungen.

Quelle je Jahr ist der JUENGSTE Bericht, der das Jahr druckt. Jede Kennzahlenseite
druckt drei Jahre, jede Kapitalflussrechnung zwei; die Ueberlappung ergibt die
Kettenpruefung: derselbe Wert muss in jedem Bericht gleich stehen, der ihn druckt.
Eine Abweichung ist entweder ein Lesefehler oder eine Neudarstellung des Unternehmens.
Neudarstellungen, die an der Seite geprueft sind, stehen in BEKANNT mit Grund und
Seite; jede weitere Abweichung laesst das Skript fehlschlagen.

Aufruf: python3 scripts/reihe.py      Ausgabe: data/reihen.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import seite, werte, werte_spalten                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kennzahlenseite: (Dokument, gedruckte Seite, Jahre in Spaltenfolge)
KF = [("neste-ar-2019", "123", (2019, 2018, 2017)), ("neste-ar-2020", "129", (2020, 2019, 2018)),
      ("neste-ar-2021", "162", (2021, 2020, 2019)), ("neste-ar-2022", "172", (2022, 2021, 2020)),
      ("neste-ar-2023", "171", (2023, 2022, 2021)), ("neste-ar-2024", "150", (2024, 2023, 2022)),
      ("neste-ar-2025", "146", (2025, 2024, 2023))]
KF_POSTEN = {
    "umsatz": r"Revenue(?=\s{2})",
    "ebitda": r"EBITDA(?=\s{2})",
    "ebit": r"Operating profit(?=\s{2})",
    "vergl_ebitda": r"Comparable EBITDA(?=\s{2})",
    "vergl_ebit": r"Comparable operating profit(?=\s{2})",
    "ergebnis": r"Profit for the period(?=\s{2})",
    "nettoschulden": r"Interest-bearing net debt(?=\s{2})",
    "verschuldungsgrad": r"Leverage ratio(?=\s{2})",
    "roace": r"(?:employed, after tax )?\((?:Comparable )?ROACE\)",
    "roe": r"Return on equity \(ROE\)",
    "capital_employed": r"Capital employed(?=\s{2})",
    "capex_kf": r"Capital expenditure and investments in shares",
    "eps": r"Earnings per share \(EPS\)",
    "ek_je_aktie": r"Equity per share",
    "dps": r"Dividend per share",
    "kurs_ende": r"Closing (?:share )?price",
    "kurs_hoch": r"Highest (?:share )?price",
    "kurs_tief": r"Lowest (?:share )?price",
    "boersenwert": r"Market capitalization",
    "aktien_ende": r"(?:Number of shares outstanding )?at the end of the period",
}
# Kapitalflussrechnung: (Dokument, gedruckte Seite, Jahre)
CF = [("neste-ar-2019", "131", (2019, 2018)), ("neste-ar-2020", "137", (2020, 2019)),
      ("neste-ar-2021", "170", (2021, 2020)), ("neste-ar-2022", "180", (2022, 2021)),
      ("neste-ar-2023", "180", (2023, 2022)), ("neste-ar-2024", "158", (2024, 2023)),
      ("neste-ar-2025", "154", (2025, 2024))]
CF_POSTEN = {
    "operativer_cf": r"Net cash (?:generated )?from operating activities",
    "sachanlagen": r"Purchases of property, plant and equipment",
    "immateriell": r"Purchases of intangible assets",
    "leasing": r"Repayments? of lease liabilities",
    "dividenden": r"Dividends paid to the owners of the parent",
    "akquisitionen": r"Acquisitions of subsidiaries",
}
# Werte, die keine Tabellenzeile tragen, einzeln an ihrer Seite gelesen:
# (Posten, Jahr, Wert, Dokument, Seite, Anmerkung)
EINZELN = [
    ("operativer_cf", 2017, 1094, "neste-fsr-2018", "26", ""),
    ("investitionen_gesamt", 2017, -475, "neste-fsr-2018", "26",
     "Capital expenditure als eine Zeile; ab 2018 nach Sach- und immateriellen Anlagen getrennt"),
    ("dividenden", 2017, -332, "neste-fsr-2018", "26", ""),
    # Vor 2021 keine eigene Leasingzeile: Tilgung in "Repayments of non-current
    # interest-bearing liabilities" enthalten; der Anhang nennt den Gesamtabfluss.
    ("leasing", 2019, -68, "neste-ar-2020", "193", "Anhang: total cash outflow for leases"),
    ("leasing", 2020, -115, "neste-ar-2020", "193", "Anhang: total cash outflow for leases"),
    ("ergebnis", 2017, 914, "neste-fsr-2018", "24", ""),
    ("ergebnis", 2018, 779, "neste-fsr-2018", "24", ""),
    ("ergebnis", 2018, 775, "neste-ar-2019", "129", ""),
    ("ergebnis", 2019, 1789, "neste-ar-2019", "129", ""),
]
# Gepruefte Neudarstellungen (Posten, Jahr): Grund mit Seite
BEKANNT = {
    ("operativer_cf", 2024): "GB 2025 S. 154: 2024 neu dargestellt (Wechselkurseffekte aus "
                             "Finance cost and income taxes paid herausgenommen), 1.183 -> 1.154",
    ("roace", 2020): "Definitionswechsel ROACE -> Comparable ROACE (GB 2022 S. 172)",
    ("roace", 2021): "Definitionswechsel ROACE -> Comparable ROACE (GB 2022 S. 172)",
    ("ergebnis", 2018): "GB 2019 S. 129 fuehrt 2018 als 'Restated' (779 -> 775)",
    ("boersenwert", 2022): "GB 2022 druckt 33.063, GB 2023 und 2024 drucken 33.091",
}


def main() -> int:
    reihen, bruch = {}, []

    def eintragen(posten, jahr, wert, dok, s, anm=""):
        e = {"wert": wert, "bericht": dok, "seite": s}
        if anm:
            e["anmerkung"] = anm
        alt = reihen.setdefault(posten, {}).get(str(jahr))
        if alt is not None and abs(alt["wert"] - wert) > 0.0051 * max(1, abs(wert)):
            if (posten, jahr) in BEKANNT:
                e["neu_dargestellt"] = BEKANNT[(posten, jahr)]
            else:
                bruch.append(f"{posten} {jahr}: {alt['wert']} ({alt['bericht']}) vs {wert} ({dok})")
        # juengerer Bericht gewinnt (Liste ist aufsteigend nach Berichtsjahr)
        # Rang nach Berichtsjahr, nicht nach Dateiname ("fsr" sortiert hinter "ar")
        if alt is None or int(dok[-4:]) >= int(alt["bericht"][-4:]):
            if alt is not None and abs(alt["wert"] - wert) > 0.0051 * max(1, abs(wert)):
                e["frueher"] = [alt["wert"], alt["bericht"]]
            reihen[posten][str(jahr)] = e

    fehlend = []
    for dok, s, jahre in KF:
        t = seite(dok, s)
        for posten, muster in KF_POSTEN.items():
            w = werte(t, muster, len(jahre))
            if w is None:
                fehlend.append(f"{dok} S.{s} {posten}")
                continue
            for j, v in zip(jahre, w):
                eintragen(posten, j, v, dok, s)
    for dok, s, jahre in CF:
        t = seite(dok, s)
        for posten, muster in CF_POSTEN.items():
            w = werte_spalten(t, muster, r"Dec 20\d\d", len(jahre))
            if w is None:
                fehlend.append(f"{dok} S.{s} {posten}")
                continue
            for j, v in zip(jahre, w):
                eintragen(posten, j, v, dok, s)
    for posten, j, v, dok, s, anm in EINZELN:
        eintragen(posten, j, v, dok, s, anm)
        if anm.startswith("Anhang"):
            # Der Anhang druckt den Abfluss als positiven Betrag; die Reihe fuehrt ihn
            # negativ. Die gedruckte Form wird mitgespeichert, damit der Seitenwachhund
            # die Zahl findet, ohne dass er Vorzeichen allgemein ignorieren muss.
            reihen[posten][str(j)]["gedruckt"] = abs(v)

    json.dump(reihen, open(os.path.join(ROOT, "data", "reihen.json"), "w"), indent=1,
              ensure_ascii=False)
    jahre = range(2017, 2026)
    for posten in list(KF_POSTEN) + list(CF_POSTEN) + ["investitionen_gesamt"]:
        r = reihen.get(posten, {})
        print(f"{posten:18s}" + "".join(f"{r[str(j)]['wert']:>10g}" if str(j) in r else f"{'.':>10s}"
                                         for j in jahre))
    print("[NICHT GELESEN]", "; ".join(fehlend) if fehlend else "keine")
    for b in bruch:
        print("[KETTE]", b)
    return 1 if bruch else 0


if __name__ == "__main__":
    sys.exit(main())
