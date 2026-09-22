#!/usr/bin/env python3
"""
seiten.py - zerlegt alle Quellen unter refs/ in gedruckte Seiten.

HTML (SEC-Einreichungen) laeuft ueber zerlege_html() aus seiten_zerlegen.py.
PDF (GEA, BESI) wird in EINEM pdftotext-Lauf gelesen und am Seitenvorschub
getrennt; die gedruckte Seitenzahl liest je Emittent ein eigenes Muster, weil
genau dort jede uebernommene Zerlegung scheitert:

    GEA   Fusszeile gesperrt gesetzt: "G E A G E S C H Ä F T S B E R I C H T
          2 02 5 1 51" - nach Entfernen der Leerzeichen steht die Zahl hinter
          dem Jahrgang.
    BESI  Zahl am Anfang der Navigationszeile im Kopf:
          "39 OF MANAGEMENT REPORT ..."

Ausgabe: refs/<name>.txt mit Trennzeilen "=== SEITE <n> (Block <i>) ===".
Aufruf: python3 scripts/seiten.py [refs/datei ...]
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgar_seiten import folgefilter, luecken                 # noqa: E402
from edgar_seiten import text_aus                            # noqa: E402
from seiten_zerlegen import _TRENNER, _bericht, _schreibe, fuelle_luecken  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dateipraefix -> (Zeilenbereich, Muster auf der leerzeichenfreien Zeile)
PDF_MUSTER = {
    # Deutscher Jahrgang gesperrt ("G E A G E S C H Ä F T S B E R I C H T 2 02 5 41"),
    # englischer Jahrgang 2020 schlicht ("GEA Annual Report 2020 11").
    "gea": ("letzte", re.compile(r"(?i)GEA(?:GESCH[ÄA]FTSBERICHT|ANNUALREPORT)\d{4}(\d{1,3})$")),
    "besi": ("erste", re.compile(r"^(\d{1,3})OFMANAGEMENT")),
    # Jahrgang 2019 anders gesetzt: blanke Zahl, mal im Kopf, mal im Fuss. Welche Zahl
    # die Seite ist, entscheidet die Folgekette (folgefilter), nicht die Stellung.
    "besi-ar-2019": ("beide", re.compile(r"^(\d{1,3})$")),
    # TTE-Registrierungsdokument: Seitenzahl vor dem Titel (gerade Seiten) oder dahinter
    "tte": ("letzte", re.compile(r"^(\d{1,3})TotalEnergies.UniversalRegistrationDocument\d{4}$|"
                                 r"TotalEnergies.UniversalRegistrationDocument\d{4}(\d{1,3})$")),
    # Daimler Truck: Kopfzeile "Daimler Truck | 2025 Annual Report ... 6" (2023-2025) bzw.
    # "Annual Report 2022 | Daimler Truck<BS> ... 6" (2022, mit Steuerzeichen \x08 vor der
    # Zahl) bzw. "Daimler Truck | Interim Report Q2 2026 ... 9" (Halbjahresbericht). Seiten
    # mit Breadcrumb-Navigation drucken die Zahl nicht im Kopf; fuelle_luecken schliesst sie.
    "daimler-ar-2022": ("erste", re.compile(r"(?i)AnnualReport2022.DaimlerTruck.?(\d{1,3})$")),
    "daimler": ("erste", re.compile(r"(?i)DaimlerTruck.\d{4}AnnualReport.*?(\d{1,3})$")),
    # 2023 allein: im Abschlussteil steht der Kopf im Randbereich gedreht gesetzt und
    # erscheint deshalb ueber ein Dutzend Ausgabezeilen kaskadiert; die Seitenzahl verliert
    # dabei jeden Nachbartext und steht auf einer eigenen, sonst leeren Zeile. Das zweite
    # Glied der Wechselform faengt genau diesen Fall auf, ohne die gewoehnlichen Seiten
    # (erstes Glied, Zeile 1) zu gefaehrden: Zeilen werden der Reihe nach geprueft, die
    # erste passende gewinnt.
    "daimler-ar-2023": ("kaskade", re.compile(r"(?i)DaimlerTruck.\d{4}AnnualReport.*?(\d{1,3})$|^(\d{1,3})$")),
    "daimler-hj-2026": ("erste", re.compile(r"(?i)DaimlerTruck.InterimReportQ2 ?2026.*?(\d{1,3})$")),
}


def zerlege_pdf(quelle: str, ziel: str):
    name = os.path.splitext(os.path.basename(quelle))[0]
    bereich, muster = PDF_MUSTER.get(name) or PDF_MUSTER[name.split("-")[0]]
    t = subprocess.run(["pdftotext", "-layout", quelle, "-"], capture_output=True,
                       text=True).stdout
    seiten_roh, texte = [], []
    for block in t.split("\f"):
        zeilen = [z.rstrip() for z in block.splitlines() if z.strip()]
        kandidaten = {"letzte": zeilen[-3:][::-1], "erste": zeilen[:3],
                      "beide": zeilen[:4] + zeilen[-4:][::-1],
                      # Daimler Truck, Abschlussteil 2023: der Kopf steht gedreht im
                      # Rand und erscheint deshalb ueber ein Dutzend Ausgabezeilen
                      # kaskadiert, je eine Silbe weiter - die Zahl steht ganz unten.
                      "kaskade": zeilen[:14]}[bereich]
        treffer = "?"
        for z in kandidaten:
            m = muster.search(re.sub(r"\s+", "", z))
            if m:
                treffer = next(g for g in m.groups() if g)
                break
        seiten_roh.append(treffer)
        texte.append("\n".join(zeilen))
    roh = folgefilter(seiten_roh)
    seiten = fuelle_luecken(roh)
    _schreibe(ziel, seiten, texte)
    return seiten, roh


_F = re.compile(r"^F-?(\d{1,3})$")
_N = re.compile(r"^(\d{1,3})$")


def _fuss(text: str) -> tuple:
    """(Segment, Zahl) aus den letzten fuenf Zeilen: ("F", n) fuer "F-12", ("", n) fuer "12"."""
    for z in reversed([re.sub(r"[\s\u200b]+", "", z) for z in text.splitlines() if z.strip()][-5:]):
        m = _F.match(z)
        if m:
            return "F", m.group(1)
        m = _N.match(z)
        if m:
            return "", m.group(1)
    return "", "?"


def _segment(zahlen: list) -> tuple:
    roh = folgefilter(zahlen)
    return fuelle_luecken(roh), roh


def zerlege_html(quelle: str, ziel: str):
    """
    SEC-Einreichung in gedruckte Seiten. Der Abschlussteil eines 20-F zaehlt
    "F-1", "F-2", ... und bildet eine EIGENE Folge: Wuerde die Hauptfolge ueber
    ihr letztes gelesenes Blatt hinaus verlaengert, bekaeme F-1 die Nummer der
    naechsten gewoehnlichen Seite - jeder Beleg aus Bilanz, GuV und
    Kapitalflussrechnung zeigte dann auf eine falsche Seite, ohne Fehlermeldung.
    """
    roh = open(quelle, encoding="utf-8", errors="replace").read()
    bloecke = max((re.split(m, roh) for m in _TRENNER), key=len)
    texte = [text_aus(b) for b in bloecke]
    fuss = [_fuss(t) if t else ("", "?") for t in texte]
    erster_f = next((i for i, (seg, n) in enumerate(fuss) if seg == "F" and n != "?"), len(fuss))
    haupt, haupt_roh = _segment([n if seg == "" else "?" for seg, n in fuss[:erster_f]])
    f_teil, f_roh = _segment([n if seg == "F" else "?" for seg, n in fuss[erster_f:]])
    seiten = haupt + [f"F-{n}" if n != "?" else "?" for n in f_teil]
    roh_alle = haupt_roh + [f"F-{n}" if n != "?" else "?" for n in f_roh]
    _schreibe(ziel, seiten, texte)
    return seiten, roh_alle


def zerlege_workiva(quelle: str, ziel: str):
    """
    Einberufungsschreiben (DEF 14A) im Workiva-Satz: absolut positionierte <div>, kein
    Seitenumbruch im Markup. Jede Seite endet aber mit ihrer Zahl, gefolgt von der
    Kopfzeile "Table of Contents" der naechsten - daran wird getrennt.
    """
    import html as _html
    roh = open(quelle, encoding="utf-8", errors="replace").read()
    text = _html.unescape(re.sub(r"<[^>]+>", "\n", roh))
    text = re.sub(r"[ \t\u00a0]+", " ", re.sub(r"\n\s*\n+", "\n", text))
    teile = re.split(r"\n\s*(\d{1,3})\s*\n\s*Table of Contents\s*\n", text)
    texte, seiten = [], []
    for i in range(0, len(teile) - 1, 2):
        texte.append(teile[i].strip())
        seiten.append(teile[i + 1])
    texte.append(teile[-1].strip())
    seiten.append(str(int(seiten[-1]) + 1) if seiten else "?")
    roh_seiten = folgefilter(seiten)
    seiten = fuelle_luecken(roh_seiten)
    _schreibe(ziel, seiten, texte)
    return seiten, roh_seiten


def main(argv: list) -> int:
    refs = os.path.join(ROOT, "refs")
    dateien = argv or sorted(os.path.join(refs, f) for f in os.listdir(refs)
                             if f.endswith((".htm", ".pdf")) and not f.startswith("_"))
    mangel = []
    for quelle in dateien:
        ziel = os.path.splitext(quelle)[0] + ".txt"
        zerleger = (zerlege_pdf if quelle.endswith(".pdf") else
                    zerlege_workiva if "def14a" in quelle else zerlege_html)
        alle, roh = zerleger(quelle, ziel)
        seiten = [s for s in alle if s != "?" and not s.startswith("F-")]
        f_seiten = [s[2:] for s in alle if s.startswith("F-")]
        zahlen = sorted({int(s) for s in seiten if s.isdigit()})
        fehlend = luecken(seiten) + [f"F-{x}" for x in luecken(f_seiten)]
        if not zahlen:
            befund = "ohne Paginierung"
            mangel.append(os.path.basename(quelle))
        elif fehlend:
            befund = f"Seiten {zahlen[0]}-{zahlen[-1]}, {len(fehlend)} fehlend: {fehlend[:8]}"
            mangel.append(os.path.basename(quelle))
        else:
            befund = f"Seiten {zahlen[0]}-{zahlen[-1]} lueckenlos, {_bericht(roh, alle)}"
            if f_seiten:
                befund += f"; F-1 bis F-{max(map(int, f_seiten))} lueckenlos"
        print(f"{os.path.basename(ziel):24s} {len(alle):4d} Bloecke, {befund}")
    if mangel:
        print(f"\n[SEITEN] zu pruefen: {', '.join(mangel)}")
    return 1 if mangel else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
