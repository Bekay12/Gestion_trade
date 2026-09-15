#!/usr/bin/env python3
"""
fetch_kurs.py - Monatliche Schlusskurse der Aumann-Aktie (ISIN DE000A2DAM03,
Xetra-Kuerzel AAG) fuer den Zeitraum 2022-01 bis 2025-12.

Schreibt data/aktienkurs.csv mit Quellenangabe und Abrufdatum im Kopf.
Bricht mit Rueckgabewert 1 ab, wenn keine Quelle antwortet - in diesem Fall
ist auf die in den Geschaeftsberichten veroeffentlichten Kursdaten
zurueckzufallen. Reine Standardbibliothek.
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

SYMBOL = "AAG.DE"
# 2021-12-01 bis 2026-01-05, damit der Dezember 2025 vollstaendig enthalten ist
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    f"{SYMBOL}?period1=1638316800&period2=1767571200&interval=1mo"
)
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def fetch() -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Laedt die monatlichen Schlusskurse und gibt sie als Liste zurueck.

    Inputs:
        keine

    Outputs:
        rows (list[tuple[str, float]]): (Datum ISO, Schlusskurs in EUR)
    --------------------------------------------------------------------------
    """
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    rows = []
    for ts, close in zip(stamps, closes):
        if close is None:
            continue
        day = datetime.datetime.utcfromtimestamp(ts).date()
        if not (2022 <= day.year <= 2025):
            continue
        rows.append((day.isoformat(), round(float(close), 2)))
    return rows


def main() -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        Holt die Kursreihe und schreibt data/aktienkurs.csv.

    Inputs:
        keine (Kommandozeile ohne Argumente)

    Outputs:
        code (int): 0 bei Erfolg, 1 wenn die Quelle nicht nutzbar ist.
    --------------------------------------------------------------------------
    """
    try:
        rows = fetch()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
            ValueError, TimeoutError, OSError) as exc:
        print(f"FEHLER: Kursquelle nicht erreichbar oder unlesbar: {exc}", file=sys.stderr)
        print("Rueckfall: Kursdaten aus den Geschaeftsberichten 2022-2025 verwenden.",
              file=sys.stderr)
        return 1
    if len(rows) < 40:
        print(f"FEHLER: nur {len(rows)} Datenpunkte erhalten, erwartet werden rund 48.",
              file=sys.stderr)
        return 1

    abruf = datetime.date.today().strftime("%d.%m.%Y")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "aktienkurs.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Aumann AG (ISIN DE000A2DAM03), monatliche Schlusskurse in EUR\n")
        fh.write(f"# Quelle: Yahoo Finance, Symbol {SYMBOL}\n")
        fh.write(f"# Abrufdatum: {abruf}\n")
        fh.write("datum,schluss\n")
        for day, close in rows:
            fh.write(f"{day},{close}\n")
    print(f"OK -> data/aktienkurs.csv ({len(rows)} Datenpunkte, Abruf {abruf})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
