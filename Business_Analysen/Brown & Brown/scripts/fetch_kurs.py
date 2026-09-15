#!/usr/bin/env python3
"""
fetch_kurs.py - Kursreihe der Brown-&-Brown-Aktie (NYSE: BRO).

Laedt die Tagesschlusskurse ab dem 01.12.2020 von Nasdaq und schreibt:
data/kurs_monat.csv   Monatsschlusskurse fuer die Abbildung
data/kurs_jahr.csv    Jahresschluss, Jahreshoechst- und Jahrestiefstkurs
data/kurs_extrema.csv Handelstag des Jahreshoechst- und Jahrestiefstkurses
data/kursverlauf.csv  Stuetzpunkte der Abbildung, streng nach Datum geordnet

Yahoo Finance ist als zweite Quelle hinterlegt; von dort kam am 28.08.2026
nur HTTP 429. Alle Dateien tragen Quelle und Abrufdatum im Kopf. Reine
Standardbibliothek. Rueckgabewert 1, wenn keine Quelle antwortet.
"""
import datetime
import json
import sys
import urllib.error
import urllib.request

SYMBOL = "BRO"
NASDAQ = (
    "https://api.nasdaq.com/api/quote/{}/historical"
    "?assetclass=stocks&fromdate=2020-12-01&todate={}&limit=3000"
)
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nasdaq.com/",
}
HEAD = "# Quelle: Nasdaq, Historical Quotes BRO (NYSE), abgerufen am {}\n"


def _usd(text: str) -> float:
    """Wandelt '$110.57' oder '1,234.5' in eine Gleitkommazahl."""
    return float(text.replace("$", "").replace(",", "").strip())


def fetch(today: datetime.date) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Laedt die Tageskurse und gibt sie aufsteigend sortiert zurueck.

    Inputs:
        today (datetime.date): letzter abzurufender Handelstag

    Outputs:
        rows (list[tuple[datetime.date, float, float, float]]):
            (Handelstag, Schlusskurs, Tageshoch, Tagestief) in USD
    --------------------------------------------------------------------------
    """
    url = NASDAQ.format(SYMBOL, today.isoformat())
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    table = payload["data"]["tradesTable"]["rows"]
    rows = []
    for r in table:
        month, day, year = r["date"].split("/")
        rows.append((
            datetime.date(int(year), int(month), int(day)),
            _usd(r["close"]), _usd(r["high"]), _usd(r["low"]),
        ))
    rows.sort()
    return rows


def main() -> int:
    heute = datetime.date.today()
    try:
        rows = fetch(heute)
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"[KURS] Abruf fehlgeschlagen: {exc}", file=sys.stderr)
        return 1
    stamp = heute.strftime("%d.%m.%Y")

    monat = {}
    for day, c, _, _ in rows:
        monat[(day.year, day.month)] = (day, c)
    with open("data/kurs_monat.csv", "w", encoding="utf-8") as fh:
        fh.write(HEAD.format(stamp))
        fh.write("datum,schluss_usd\n")
        for key in sorted(monat):
            day, c = monat[key]
            fh.write(f"{day.isoformat()},{c:.2f}\n")

    with open("data/kurs_jahr.csv", "w", encoding="utf-8") as fh, \
         open("data/kurs_extrema.csv", "w", encoding="utf-8") as fx:
        fh.write(HEAD.format(stamp))
        fh.write("jahr,schluss_usd,hoch_usd,tief_usd\n")
        fx.write(HEAD.format(stamp))
        fx.write("jahr,art,datum,kurs_usd\n")
        for jahr in range(2021, heute.year + 1):
            js = [r for r in rows if r[0].year == jahr]
            if len(js) < 5:
                continue
            hi = max(js, key=lambda r: r[2])
            lo = min(js, key=lambda r: r[3])
            fh.write(f"{jahr},{js[-1][1]:.2f},{hi[2]:.2f},{lo[3]:.2f}\n")
            fx.write(f"{jahr},hoch,{hi[0].isoformat()},{hi[2]:.2f}\n")
            fx.write(f"{jahr},tief,{lo[0].isoformat()},{lo[3]:.2f}\n")
    # Stuetzpunkte der Abbildung: Ausgangspunkt ist der letzte Schlusskurs
    # vor dem Betrachtungszeitraum, danach je Jahr Hoch, Tief und Schluss in
    # der Reihenfolge, in der sie EINGETRETEN sind. Eine nach Kategorien
    # getrennte Reihe wuerde die Linie zerreissen; die Spalte "art" (S/H/T)
    # steuert allein die Markenform.
    punkte = []
    vorlauf = [r for r in rows if r[0].year == 2020]
    if vorlauf:
        punkte.append((vorlauf[-1][0], vorlauf[-1][1], "S"))
    for jahr in range(2021, heute.year + 1):
        js = [r for r in rows if r[0].year == jahr]
        if len(js) < 5:
            continue
        hi = max(js, key=lambda r: r[2])
        lo = min(js, key=lambda r: r[3])
        drei = [(hi[0], hi[2], "H"), (lo[0], lo[3], "T"),
                (js[-1][0], js[-1][1], "S")]
        # Faellt ein Extremum auf den letzten Handelstag des Jahres, stuenden
        # zwei Punkte auf derselben Abszisse; der Schlusskurs bleibt dann als
        # der aussagekraeftigere stehen.
        gesehen = set()
        for tag, kurs, art in sorted(drei, key=lambda x: (x[0], x[2] != "S")):
            if tag in gesehen and art != "S":
                continue
            gesehen.add(tag)
            punkte.append((tag, kurs, art))
    punkte.sort(key=lambda x: x[0])
    with open("data/kursverlauf.csv", "w", encoding="utf-8") as fh:
        fh.write(HEAD.format(stamp))
        fh.write("# S = Jahresschluss, H = Jahreshoechstkurs, T = Jahrestiefstkurs\n")
        fh.write("datum,kurs,art\n")
        for tag, kurs, art in punkte:
            fh.write(f"{tag.isoformat()},{kurs:.2f},{art}\n")

    print(f"[KURS] {len(rows)} Handelstage, {len(monat)} Monatswerte, "
          f"{len(punkte)} Stuetzpunkte geschrieben")
    return 0


if __name__ == "__main__":
    sys.exit(main())
