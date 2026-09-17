#!/usr/bin/env python3
"""
test_fussnoten_gruppen.py - Regressionstests fuer die Schluesselbildung des
Fussnoten-Aufloesers.

Anlass ist ein Fehler, der keine Warnung erzeugt und trotzdem eine falsche
Seitenangabe druckt: Die urspruengliche Schluesselbildung entfernte alle
Nicht-Buchstaben, sodass aus "MDA|2025|7" und "MDA|2025|13" beide Male "mda"
wurde. Beide Zitationen bekamen auf derselben gedruckten Seite denselben
Merkschluessel, das zweite \\quelleMerken ueberschrieb das erste, und jedes
\\quelleErneut zeigte danach auf die zuletzt gesetzte Fussnote.

Aufruf: python3 -m unittest test_fussnoten_gruppen
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fussnoten_gruppen import _buchstaben, _VOLL, zitat_text  # noqa: E402


class TestBuchstaben(unittest.TestCase):

    def test_verschiedene_seiten_ergeben_verschiedene_schluessel(self):
        # Der eigentliche Regressionsfall.
        self.assertNotEqual(_buchstaben("MDA|2025|7"), _buchstaben("MDA|2025|13"))

    def test_verschiedene_jahre_ergeben_verschiedene_schluessel(self):
        self.assertNotEqual(_buchstaben("MDA|2024|4"), _buchstaben("MDA|2025|4"))

    def test_verschiedene_dokumentarten_ergeben_verschiedene_schluessel(self):
        self.assertNotEqual(_buchstaben("MDA|2025|4"), _buchstaben("FS|2025|4"))

    def test_schluessel_ist_ein_gueltiger_latex_makroname(self):
        # LaTeX-Makronamen duerfen nur Buchstaben enthalten; eine Ziffer im
        # Schluessel bricht \csname ab.
        for quelle in ("MDA|2025|7", "FS|2016|41", "ZB|4", "Web|Titel|https://a.b"):
            self.assertRegex(_buchstaben(quelle), r"^[a-z]+$",
                             f"{quelle} ergibt keinen reinen Buchstabennamen")

    def test_gleiche_quelle_ergibt_gleichen_schluessel(self):
        self.assertEqual(_buchstaben("MDA|2025|7"), _buchstaben("MDA|2025|7"))


class TestZitatText(unittest.TestCase):

    def test_einstelliges_makro_erzeugt_keinen_none_schluessel(self):
        # Vor der Korrektur wurde aus dem fehlenden zweiten Argument der
        # Schluessel "...zbnone", und LaTeX brach mit "Missing number" ab.
        m = _VOLL.search(r"Text.\quelleZB{4} weiter")
        self.assertIsNotNone(m)
        text, quelle = zitat_text(m)
        self.assertEqual(text, r"\quelleZB{4}")
        self.assertNotIn("none", quelle.lower())

    def test_zweistelliges_makro_bleibt_unveraendert(self):
        m = _VOLL.search(r"Text.\quelleMDA{2025}{7} weiter")
        text, quelle = zitat_text(m)
        self.assertEqual(text, r"\quelleMDA{2025}{7}")
        self.assertEqual(quelle, "MDA|2025|7")

    def test_webquelle_behaelt_drei_argumente(self):
        m = _VOLL.search(r"T.\quelleWeb{Titel}{quelle:x}{16.09.2026} w")
        text, _ = zitat_text(m)
        self.assertTrue(text.startswith(r"\quelleWeb{Titel}"))


if __name__ == "__main__":
    unittest.main()
