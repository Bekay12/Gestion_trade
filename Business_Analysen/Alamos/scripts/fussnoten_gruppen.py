#!/usr/bin/env python3
"""
fussnoten_gruppen.py - loest die Befunde von check_footnote_pages.py und
check_footnote_groups.py auf.

Die Zitierweise erlaubt, eine bereits gesetzte Fussnote wiederzuverwenden
(\\quelleMerken / \\quelleErneut in preamble.tex). Beides gilt aber nur
innerhalb EINER gedruckten Seite: Ein \\quelleErneut auf der Folgeseite
verweist auf eine Nummer, die dort nicht mehr steht.

Das Skript ordnet jeden Zitationsaufruf \"uber SyncTeX seiner gedruckten
Seite zu und sorgt daf\"ur, dass je Quelle und Seite genau eine
Vollzitation mit eigenem \\quelleMerken steht und alle weiteren Aufrufe
derselben Quelle auf derselben Seite \\quelleErneut darauf verweisen.

Weil jede eingef\"ugte Fussnote den Seitenumbruch verschieben kann, ist der
Lauf zu wiederholen, bis beide Pr\"ufskripte schweigen; das \"ubernimmt
--schleife.

Aufruf: python3 scripts/fussnoten_gruppen.py [--schleife]
Voraussetzung: ./build.sh wurde mit -synctex=1 ausgefuehrt.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(ROOT, "sections")
PDF = os.path.join(ROOT, "out", "analyse.pdf")

_VOLL = re.compile(r"\\quelle(MDA|FS|Web|ZBA|ZB|Circ)\{([^}]*)\}(?:\{([^}]*)\})?"
                   r"(?:\{([^}]*)\})?")
_ERNEUT = re.compile(r"\\quelleErneut\{([^}]*)\}")
_MERKEN = re.compile(r"\\quelleMerken\{([^}]*)\}")


def seite(name: str, lineno: int) -> int:
    """Gibt die gedruckte PDF-Seite einer Quelltextzeile zurueck."""
    rel = os.path.join("sections", name)
    out = subprocess.run(["synctex", "view", "-i", f"{lineno}:1:{rel}",
                          "-o", PDF], capture_output=True, text=True,
                         cwd=ROOT).stdout
    treffer = re.findall(r"^Page:(\d+)$", out, re.M)
    return int(treffer[0]) if treffer else -1


# Makros, die nur die Seite tragen und kein Jahr: Das Dokument steht in
# ihrem Namen fest. Ohne diese Unterscheidung baute das Skript aus dem
# fehlenden zweiten Argument den Schluessel "...zbnone" und schrieb ein
# \quelleErneut darauf, zu dem es kein \quelleMerken gab - LaTeX brach
# dann mit "Missing number, treated as zero" ab.
_EINSTELLIG = {"ZB", "ZBA", "Circ"}


def _buchstaben(quelle: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Wandelt einen Quellenschluessel wie "MDA|2025|7" in eine Folge aus
        reinen Buchstaben, die LaTeX als Makronamen zulaesst.

        Die Ziffern werden UMGESETZT und nicht entfernt. Sie zu entfernen war
        der urspruengliche Weg, und er war falsch: Aus "MDA|2025|7" und
        "MDA|2025|13" wurde beide Male "mda", beide bekamen auf derselben
        gedruckten Seite denselben Merkschluessel, und das zweite
        \quelleMerken ueberschrieb das erste. Jedes \quelleErneut zeigte
        danach auf die zuletzt gesetzte Fussnote - also auf die falsche
        Seitenangabe. Der Fehler war unsichtbar: Er erzeugt keine Warnung,
        nur eine Fussnote, die auf die falsche Seite verweist.

    Inputs:
        quelle (str): Quellenschluessel, z. B. "MDA|2025|7"

    Outputs:
        name (str): Buchstabenfolge, je Quelle verschieden
    --------------------------------------------------------------------------
    """
    ziffern = "abcdefghij"
    aus = []
    for z in quelle.lower():
        if z.isdigit():
            aus.append(ziffern[int(z)])
        elif z.isalpha():
            aus.append(z)
        else:
            aus.append("x")  # Trennzeichen bleibt als Trennzeichen erkennbar
    return "".join(aus)


def zitat_text(m: re.Match) -> tuple:
    """Gibt (Volltext des Aufrufs, Schluessel der Quelle) zurueck."""
    art, a, b, c = m.groups()
    if art == "Web":
        return f"\\quelleWeb{{{a}}}{{{b}}}{{{c}}}", f"Web|{a}|{b}"
    if art in _EINSTELLIG:
        # a ist hier die Seite, b ist None.
        return f"\\quelle{art}{{{a}}}", f"{art}|{a}"
    return f"\\quelle{art}{{{a}}}{{{b}}}", f"{art}|{a}|{b}"


def main() -> int:
    # 1. Alle Aufrufe einsammeln: Vollzitationen und Erneut-Verweise.
    aufrufe = []          # (datei, zeile, spalte, art, text, quelle, merkkey)
    merk = {}             # Merkschluessel -> Quellenschluessel
    for datei in sorted(os.listdir(SECTIONS)):
        if not datei.endswith(".tex"):
            continue
        pfad = os.path.join(SECTIONS, datei)
        for nr, roh in enumerate(open(pfad, encoding="utf-8"), start=1):
            code = roh.split("%")[0]
            for m in _VOLL.finditer(code):
                text, quelle = zitat_text(m)
                rest = code[m.end():]
                mk = _MERKEN.match(rest)
                schluessel = mk.group(1) if mk else None
                if schluessel:
                    merk[schluessel] = quelle
                aufrufe.append([datei, nr, m.start(), "voll", text, quelle,
                                schluessel])
            for m in _ERNEUT.finditer(code):
                aufrufe.append([datei, nr, m.start(), "erneut",
                                m.group(0), None, m.group(1)])
    for a in aufrufe:
        if a[3] == "erneut":
            a[5] = merk.get(a[6])

    # 2. Seiten bestimmen und je (Seite, Quelle) die erste Zitation waehlen.
    seiten = {}
    for a in aufrufe:
        schl = (a[0], a[1])
        if schl not in seiten:
            seiten[schl] = seite(a[0], a[1])
        a.append(seiten[schl])

    volltext = {a[5]: a[4] for a in aufrufe if a[3] == "voll" and a[5]}
    erste = {}
    for a in sorted(aufrufe, key=lambda x: (x[7], x[0], x[1], x[2])):
        if a[5] is None or a[7] < 0:
            continue
        erste.setdefault((a[7], a[5]), a)

    # 3. Ersetzungen je Datei aufbauen, von hinten nach vorn angewandt.
    aenderungen = {}
    for a in aufrufe:
        if a[5] is None or a[7] < 0:
            continue
        ist_erste = erste.get((a[7], a[5])) is a
        key = f"s{a[7]}" + _buchstaben(a[5])
        if ist_erste:
            neu = volltext.get(a[5], a[4]) + f"\\quelleMerken{{{key}}}"
        else:
            neu = f"\\quelleErneut{{{key}}}"
        alt = a[4]
        if a[3] == "voll":
            # Ein bereits angehaengtes \quelleMerken mit ersetzen.
            alt = alt + (f"\\quelleMerken{{{a[6]}}}" if a[6] else "")
        if alt != neu:
            aenderungen.setdefault(a[0], []).append((a[1], a[2], alt, neu))

    for datei, liste in aenderungen.items():
        pfad = os.path.join(SECTIONS, datei)
        zeilen = open(pfad, encoding="utf-8").read().split("\n")
        for nr, spalte, alt, neu in sorted(liste, key=lambda x: (-x[0], -x[1])):
            z = zeilen[nr - 1]
            if z[spalte:spalte + len(alt)] == alt:
                zeilen[nr - 1] = z[:spalte] + neu + z[spalte + len(alt):]
            else:
                print(f"UEBERSPRUNGEN {datei}:{nr} - Text nicht deckungsgleich")
        open(pfad, "w", encoding="utf-8").write("\n".join(zeilen))
    gesamt = sum(len(v) for v in aenderungen.values())
    print(f"[FUSSNOTEN] {gesamt} Aufrufe angepasst in "
          f"{len(aenderungen)} Dateien")
    return gesamt


if __name__ == "__main__":
    if "--schleife" in sys.argv:
        for runde in range(1, 9):
            n = main()
            subprocess.run(["./build.sh"], cwd=ROOT, capture_output=True)
            p = subprocess.run([sys.executable, "scripts/check_footnote_pages.py"],
                               cwd=ROOT, capture_output=True, text=True)
            g = subprocess.run([sys.executable, "scripts/check_footnote_groups.py"],
                               cwd=ROOT, capture_output=True, text=True)
            print(f"  Runde {runde}: Seitenpruefung {p.returncode}, "
                  f"Gruppenpruefung {g.returncode}")
            if p.returncode == 0 and g.returncode == 0:
                sys.exit(0)
        sys.exit(1)
    sys.exit(0 if main() == 0 else 0)
