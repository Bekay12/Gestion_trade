#!/usr/bin/env python3
"""
kurse.py - Kursreihen beider Titel und die Huerde des Branchenindex.

Erzeugt:
    data/kurs_<x>_jahr.csv     Jahresschluss, Jahreshoch, Jahrestief je Titel
    data/kurs_<x>_extrema.csv  Handelstag des Jahreshochs und Jahrestiefs
    data/kurs_<x>_monat.csv    Monatsschlusskurse fuer die Abbildung
    data/kurs_makros.tex       Kursgroessen als Makros fuer die Zahlenschicht

Zwei Bereinigungen, die dieser Reihe nicht anzusehen sind und ohne die jede
Aussage ueber den Kursverlauf falsch waere:

  AZN  Bis zum 02.02.2026 wurde an der NYSE ein ADR gehandelt, das eine halbe
       Stammaktie verbriefte; seither ist die Stammaktie selbst notiert. Die
       Reihe von Nasdaq ist auf die Stammaktie zurueckgerechnet: der Schluss
       2016 steht mit 53,87 USD, waehrend das ADR damals rund 27 USD kostete.
  RDY  Aktiensplit 1:5 zum 28.10.2024. Die Reihe ist zurueckgerechnet; um den
       Splittermin herum findet sich kein Sprung.

Beide Reihen sind KURSreihen, nicht Gesamtrenditen: Dividenden sind nicht
eingerechnet. Die Huerde dagegen ist eine Gesamtrendite, und das ist der
strengere Vergleich.

Aufruf: python3 scripts/kurse.py
"""
import datetime
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://www.nasdaq.com/",
}
NASDAQ = ("https://api.nasdaq.com/api/quote/{}/historical"
          "?assetclass=stocks&fromdate={}&todate={}&limit=4000")
TITEL = {"azn": "AZN", "rdy": "RDY"}
VON = "2016-09-01"


def hole(symbol: str, bis: datetime.date) -> dict:
    url = NASDAQ.format(symbol, VON, bis.isoformat())
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        nutz = json.load(r)
    zeilen = nutz["data"]["tradesTable"]["rows"]
    aus = {}
    for z in zeilen:
        tag = datetime.datetime.strptime(z["date"], "%m/%d/%Y").date()
        aus[tag] = float(z["close"].replace("$", "").replace(",", ""))
    if len(aus) < 2000:
        raise SystemExit(f"[KURS] {symbol}: nur {len(aus)} Handelstage erhalten")
    return aus


def schreibe(kuerzel: str, reihe: dict, abruf: str) -> dict:
    kopf = (f"# Quelle: Nasdaq, Historical Quotes {TITEL[kuerzel]}, "
            f"abgerufen am {abruf}\n"
            "# Kursreihe ohne Dividenden; um Split bzw. Umstellung der "
            "Notierung zurueckgerechnet.\n")
    jahre, extrema, monate = [], [], []
    for j in sorted({t.year for t in reihe}):
        tage = [t for t in reihe if t.year == j]
        schluss = max(tage)
        hoch = max(tage, key=lambda t: reihe[t])
        tief = min(tage, key=lambda t: reihe[t])
        jahre.append((j, reihe[schluss], reihe[hoch], reihe[tief]))
        extrema.append((j, hoch.isoformat(), reihe[hoch], tief.isoformat(), reihe[tief]))
    for m in sorted({(t.year, t.month) for t in reihe}):
        tage = [t for t in reihe if (t.year, t.month) == m]
        monate.append((max(tage).isoformat(), reihe[max(tage)]))
    def _schreib(name, kopfzeile, zeilen):
        with open(os.path.join(ROOT, "data", name), "w") as f:
            f.write(kopf + kopfzeile + "\n")
            for z in zeilen:
                f.write(";".join(str(x) for x in z) + "\n")
    _schreib(f"kurs_{kuerzel}_jahr.csv", "Jahr;Schluss;Hoch;Tief", jahre)
    _schreib(f"kurs_{kuerzel}_extrema.csv", "Jahr;Tag_Hoch;Hoch;Tag_Tief;Tief", extrema)
    _schreib(f"kurs_{kuerzel}_monat.csv", "Monatsende;Schluss", monate)
    # Stuetzpunkte der Abbildung: Jahresschluss, Jahreshoch und Jahrestief,
    # jeder an seinem TATSAECHLICHEN Handelstag. Ein Jahreshoch ohne seinen
    # Tag ist eine Zahl; mit seinem Tag ist es ein Argument, weil sich
    # nachsehen laesst, was an diesem Tag geschah.
    punkte = []
    for j, tag_h, hoch, tag_t, tief in extrema:
        schluss = max(t for t in reihe if t.year == j)
        punkte += [(tag_h, f"{hoch:.2f}", "H"), (tag_t, f"{tief:.2f}", "T"),
                   (schluss.isoformat(), f"{reihe[schluss]:.2f}", "S")]
    punkte.sort()
    with open(os.path.join(ROOT, "data", f"kursverlauf_{kuerzel}.csv"), "w") as f:
        f.write(kopf + "# S = Jahresschluss, H = Jahreshoechstkurs, T = Jahrestiefstkurs\n")
        f.write("datum,kurs,art\n")
        for z in punkte:
            f.write(",".join(z) + "\n")
    # Tabellenkoerper fuer den LaTeX-Satz der Jahrestabelle
    with open(os.path.join(ROOT, "data", f"kurs_{kuerzel}_tabelle.tex"), "w") as f:
        for j, schluss, hoch, tief in jahre:
            f.write(f"{j} & \\num{{{schluss:.2f}}} & \\num{{{hoch:.2f}}} "
                    f"& \\num{{{tief:.2f}}} \\\\\n")
    letzt = max(reihe)
    return {"letzter_tag": letzt, "letzter_kurs": reihe[letzt], "reihe": reihe,
            "jahre": {j: s for j, s, _, _ in jahre},
            "hoch": {j: h for j, _, h, _ in jahre},
            "tief": {j: t for j, _, _, t in jahre}}


def main() -> int:
    heute = datetime.date.today()
    abruf = heute.strftime("%d.%m.%Y")
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    stand = {}
    for kuerzel, symbol in TITEL.items():
        stand[kuerzel] = schreibe(kuerzel, hole(symbol, heute), abruf)
        print(f"[KURS] {symbol}: letzter Schluss {stand[kuerzel]['letzter_kurs']:.2f} USD "
              f"am {stand[kuerzel]['letzter_tag']}")
    # Handelstage der Extrema des laufenden Jahres als Makros. Ein Jahreshoch
    # ohne seinen Tag ist eine Zahl; mit seinem Tag laesst sich nachsehen, was
    # an diesem Tag geschah - und genau davon lebt die Erklaerung des
    # Kursverlaufs.
    with open(os.path.join(ROOT, "data", "kurs_tage.tex"), "w") as f:
        f.write("% erzeugt von scripts/kurse.py - NICHT von Hand aendern\n")
        for kuerzel, vorsatz in (("azn", "Azn"), ("rdy", "Rdy")):
            reihe = stand[kuerzel]["reihe"]
            tage = [t for t in reihe if t.year == heute.year]
            hoch = max(tage, key=lambda t: reihe[t])
            tief = min(tage, key=lambda t: reihe[t])
            for name, tag in (("KurstagHoch", hoch), ("KurstagTief", tief)):
                f.write(f"\\newcommand{{\\{vorsatz}{name}}}"
                        f"{{{tag.strftime('%d.%m.%Y')}}}\n")
    with open(os.path.join(ROOT, "data", "kurs_stand.json"), "w") as f:
        json.dump({k: {"letzter_tag": str(v["letzter_tag"]),
                       "letzter_kurs": v["letzter_kurs"],
                       "jahre": v["jahre"], "hoch": v["hoch"], "tief": v["tief"]}
                   for k, v in stand.items()}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
