#!/usr/bin/env python3
"""
check_literals.py - Wachhund gegen hartkodierte Zahlen in sections/*.tex.

Regel des Projekts: jeder Zahlenwert des Dokuments stammt aus einem Makro
in data/kennzahlen.tex oder aus einer generierten Tabelle in data/.
Erlaubt bleiben Jahreszahlen (19xx/20xx), Gliederungsnummern, kleine
Ganzzahlen (< 100 ohne Nachkommastelle) sowie alles in Kommentaren
und in \\label{}/\\ref{}-Argumenten. Erkannt (und damit angeschlagen)
werden auch deutsch formatierte Tausendertrennzeichen-Zahlen wie
"12.345", "246.800" oder "1.234,5".

Bekannter Zielkonflikt: die Jahreszahl-Ausnahme (19xx/20xx) blendet
bewusst auch eine vierstellige Finanzkennzahl aus, die zufaellig mit
19 oder 20 beginnt (z.B. ein Umsatz von 2024 TEUR). Das ist eine
akzeptierte Abwaegung zugunsten weniger Fehlalarme bei echten
Jahreszahlen, kein Versehen.

Opt-out pro Datei: enthaelt der Text eine Zeile, deren Inhalt (nach dem
Entfernen von fuehrendem Leerraum) genau der Marker "% check-literals: skip"
ist, liefert find_literals() fuer diesen Text keine Treffer. Damit lassen
sich einzelne, bewusst legitime Zahlen (z.B. eine Matrikelnummer auf dem
Deckblatt) freigeben, ohne den Dateinamen im Skript hart zu verdrahten.

Aufruf: python3 scripts/check_literals.py sections/*.tex
Rueckgabewert 1, wenn Treffer gefunden wurden.
"""
import re
import sys

# \label{...} und \ref{...} samt Argument entfernen
_REF = re.compile(
    r"\\(?:label|ref|eqref|cite|input|include|url|tabellenkoerper)\{[^}]*\}")
# Zitiermakros samt Argumenten entfernen. Eine Seitenzahl ist der ORT einer
# Angabe und nicht die Angabe selbst; sie gehoert nicht in die Zahlenschicht,
# weil sie dort keinen Wert haette, den man belegen koennte - sie IST der
# Beleg. Ohne diese Ausnahme meldet der Wachhund jede dreistellige Seitenzahl
# eines Belegs, und die Reaktion darauf waere, ihn abzuschalten.
_ZITAT = re.compile(
    r"\\quelle(?:GB|ZF|Web)\{[^}]*\}\{[^}]*\}(?:\{[^}]*\})?"
    r"|\\quelle(?:HJ|QB|Merken|Erneut)\{[^}]*\}"
    # Dual_Champions: \Q{Dokument}{Seite}, \QW{Titel}{Schluessel}{Datum}
    r"|\\Q\{[^}]*\}\{[^}]*\}|\\QW\{[^}]*\}\{[^}]*\}\{[^}]*\}")
# Jedes Klammerargument, das eine Adresse enthaelt. Noetig fuer \quelleWeb,
# dessen zweites Argument eine URL ist und nicht in \url{} steht: In
# Archiv-Adressen (web.archive.org) und Nachrichten-IDs stehen lange
# Ziffernfolgen, die sonst als hartkodierte Zahlen gemeldet wuerden.
# Bewusst eng gefasst - nur "{http://..." bzw. "{https://...".
_URL_ARG = re.compile(r"\{https?://[^}]*\}")
# Gliederungsnummern wie "5.4" oder "6.1.2"
_GLIEDERUNG = re.compile(r"(?<![\d,.])\d{1,2}(?:\.\d{1,2}){1,2}(?![\d,])")
# Jahreszahlen
_JAHR = re.compile(r"(?<![\d,.])(?:19|20)\d{2}(?![\d,.])")
# Verdaechtig, in Pruefreihenfolge:
#  1) deutsch gruppierte Zahl mit Tausenderpunkt(en), optional mit
#     Dezimalkomma: "12.345", "246.800", "1.234,5". Jede Dreiergruppe muss
#     genau 3 Ziffern haben, sonst waere es eine Gliederungsnummer (siehe
#     _GLIEDERUNG oben, die max. 2 Ziffern pro Gruppe zulaesst).
#  2) Dezimalzahl mit Komma ohne Tausenderpunkt: "246,8"
#  3) Ganzzahl mit >= 3 Stellen ohne Trennzeichen: "2135"
_LITERAL = re.compile(
    r"(?<![\d,.])\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![\d,.])"
    r"|(?<![\d,.])\d+,\d+(?![\d,.])"
    r"|(?<![\d,.])\d{3,}(?![\d,.])"
)
# Opt-out-Marker fuer eine ganze Datei (eigene Kommentarzeile)
_SKIP_MARKER = re.compile(r"^%\s*check-literals:\s*skip\s*$")


def _strip_comment(line: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Entfernt den Kommentarteil einer LaTeX-Zeile ab einem unescapten
        "%"; ein escaptes "\\%" bleibt als Literalzeichen erhalten.

    Inputs:
        line (str): eine einzelne Zeile aus einer .tex-Datei.

    Outputs:
        result (str): die Zeile ohne Kommentarteil.
    --------------------------------------------------------------------------
    """
    out = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def find_literals(text: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Findet unerlaubte Zahlenliterale in einem LaTeX-Quelltext.

    Inputs:
        text (str): Inhalt einer .tex-Datei.

    Outputs:
        hits (list[tuple[int, str]]): (Zeilennummer 1-basiert, Treffertext)
    --------------------------------------------------------------------------
    """
    lines = text.splitlines()
    if any(_SKIP_MARKER.match(raw.strip()) for raw in lines):
        return []

    hits = []
    for lineno, raw in enumerate(lines, start=1):
        line = _strip_comment(raw)
        line = _ZITAT.sub(" ", line)
        line = _REF.sub(" ", line)
        line = _URL_ARG.sub(" ", line)
        line = _GLIEDERUNG.sub(" ", line)
        line = _JAHR.sub(" ", line)
        for m in _LITERAL.finditer(line):
            hits.append((lineno, m.group(0)))
    return hits


def main(argv: list) -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        CLI-Einstiegspunkt: prueft jede uebergebene .tex-Datei mit
        find_literals() und meldet alle Treffer auf stdout.

    Inputs:
        argv (list[str]): sys.argv-artige Liste; argv[0] ist der
            Skriptname, argv[1:] sind die zu pruefenden Dateipfade.

    Outputs:
        exit_code (int): 1, wenn mindestens ein Treffer gefunden wurde,
            sonst 0.
    --------------------------------------------------------------------------
    """
    total = 0
    for path in argv[1:]:
        with open(path, encoding="utf-8") as fh:
            hits = find_literals(fh.read())
        for lineno, frag in hits:
            print(f"{path}:{lineno}: hartkodierte Zahl {frag!r}")
            total += 1
    if total:
        print(f"\n{total} Treffer - Werte gehoeren nach data/kennzahlen.tex.")
        return 1
    print("OK - keine hartkodierten Zahlen in den geprueften Dateien.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
