#!/usr/bin/env python3
"""
projekt.py - die einzige Datei des Geruests, die je Unternehmen angepasst wird.

Alles andere in scripts/ ist firmenneutral. Wer eine neue Analyse beginnt,
aendert hier vier Dinge und sonst nichts:

    FIRMA         Name, wie er in der Fusszeile der Einreichungen steht
    FUSSZEILEN    die Formen, in denen die Seitenzahl gedruckt wird
    ZITATE        die Zitiermakros, die preamble.tex definiert
    DOKUMENTE     Zuordnung Quellenbezeichnung -> Datei unter refs/

Warum die Fusszeilen hier stehen und nicht im Zerleger: Sie sind der einzige
Teil der Seitenzerlegung, der sich von Emittent zu Emittent unterscheidet,
und sie sind zugleich der Teil, an dem sie am haeufigsten scheitert. Ein
Emittent setzt "12", der naechste "12 | FIRMA", der dritte "FIRMA | 12", und
der vierte schreibt "12 |" in eine eigene Zeile. Wird eine Form nicht
erkannt, ist jede Angabe aus diesen Seiten unbelegbar - und das faellt ohne
die Lueckenpruefung in edgar_seiten.py nicht auf.
"""

# Name des Emittenten, wie er in der Fusszeile erscheint. Gross- und
# Kleinschreibung ist unerheblich; Sonderzeichen bitte regulaer maskieren.
FIRMA = r"MUSTER GOLD INC\.?"

# Weitere feste Bestandteile der Fusszeile neben dem Firmennamen, z. B. der
# Titel eines Einberufungsschreibens. Leere Liste ist zulaessig.
WEITERE_FUSSZEILEN = [
    r"\d{4} Management Information Circular",
]

# Zitiermakros aus preamble.tex, nach Zahl ihrer Argumente getrennt.
#   ZWEISTELLIG  \quelleXYZ{jahr}{seite}
#   EINSTELLIG   \quelleXYZ{seite}          - Dokument steht im Makronamen
#   DREISTELLIG  \quelleWeb{titel}{key}{datum}
ZITATE_ZWEISTELLIG = ("MDA", "FS")
ZITATE_EINSTELLIG = ("ZBA", "ZB", "Circ")
ZITATE_DREISTELLIG = ("Web",)

# Quellenbezeichnung im Kommentar von kennzahlen.py -> Datei unter refs/.
# Die Bezeichnung ist frei waehlbar, muss aber mit dem Text vor dem ", S. n"
# uebereinstimmen.
DOKUMENTE = {
    "MD&A 2025": "refs/mda-2025.txt",
    "Abschluss 2025": "refs/fs-2025.txt",
}
