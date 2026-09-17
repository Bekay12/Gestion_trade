#!/usr/bin/env python3
"""
vergleich_kurs.py - stellt den Kurs von Alamos dem Goldpreis und dem
Minensektor gegenueber.

Warum das noetig ist: Die Aktie ist 2026 von ihrem Hoechstkurs am 02.03. bis
zum Tief am 31.07. um die Haelfte gefallen. Ein Bericht, der nach einem
solchen Einbruch geschrieben wird, muss sagen, WAS gefallen ist - der
Fruehindikator des Unternehmens oder seine Bewertung. Ein Rueckgang, der sich
im Goldpreis und im Sektor gleichermassen findet, ist ein Bewertungsereignis;
ihn als operative Verschlechterung in die Szenarien zu nehmen, zaehlt ihn
doppelt.

Massstaebe:
    GLD   SPDR Gold Shares - der Goldpreis selbst
    GDX   VanEck Gold Miners ETF - der Sektor

Ausgabe: data/vergleich.csv (indexiert auf den 01.01.2026 = 100)
Quelle: Nasdaq Historical Quotes, Abrufdatum im Kopf der Datei.

Aufruf: python3 scripts/vergleich_kurs.py
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_kurs import UA, _usd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://api.nasdaq.com/api/quote/{}/historical?assetclass={}"
       "&fromdate=2025-12-01&todate={}&limit=1000")
REIHEN = [("AGI", "stocks", "Alamos Gold"), ("GLD", "etf", "Goldpreis (SPDR Gold Shares)"),
          ("GDX", "etf", "Goldminen (VanEck Gold Miners)")]


def reihe(symbol: str, klasse: str, heute: datetime.date) -> dict:
    """Gibt {Datum: Schlusskurs} zurueck; Datum als ISO-Zeichenkette."""
    req = urllib.request.Request(URL.format(symbol, klasse, heute.isoformat()),
                                 headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        roh = json.load(resp)
    aus = {}
    for z in roh["data"]["tradesTable"]["rows"]:
        m, t, j = z["date"].split("/")
        aus[f"{j}-{m}-{t}"] = _usd(z["close"])
    return aus


def main() -> int:
    heute = datetime.date.today()
    try:
        daten = {s: reihe(s, k, heute) for s, k, _ in REIHEN}
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"[VERGLEICH] Abruf fehlgeschlagen: {exc}")
        return 1
    # Nur Handelstage, an denen alle drei notieren - sonst verschoebe ein
    # Feiertag einer Boerse die Indexierung.
    tage = sorted(set.intersection(*(set(d) for d in daten.values())))
    tage = [t for t in tage if t >= "2026-01-02"]
    basis = {s: daten[s][tage[0]] for s, _, _ in REIHEN}
    pfad = os.path.join(ROOT, "data", "vergleich.csv")
    with open(pfad, "w", encoding="utf-8") as fh:
        fh.write(f"# Quelle: Nasdaq Historical Quotes, abgerufen am "
                 f"{heute.strftime('%d.%m.%Y')}\n")
        fh.write(f"# Indexiert auf {tage[0]} = 100\n")
        fh.write("datum," + ",".join(s for s, _, _ in REIHEN) + "\n")
        for t in tage:
            fh.write(t + "," + ",".join(
                f"{daten[s][t] / basis[s] * 100:.2f}" for s, _, _ in REIHEN) + "\n")
    hoch = max(tage, key=lambda t: daten["AGI"][t])
    tief = min(tage, key=lambda t: daten["AGI"][t])
    print(f"[VERGLEICH] {len(tage)} gemeinsame Handelstage ab {tage[0]}")
    for s, _, name in REIHEN:
        v_h, v_t, v_e = daten[s][hoch], daten[s][tief], daten[s][tage[-1]]
        print(f"  {name:32s} Hoch {hoch} {v_h:8.2f} -> Tief {tief} {v_t:8.2f} "
              f"({(v_t / v_h - 1) * 100:+6.1f} %), heute {v_e:8.2f} "
              f"({(v_e / v_h - 1) * 100:+6.1f} % vom Hoch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
