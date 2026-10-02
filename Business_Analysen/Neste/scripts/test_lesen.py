#!/usr/bin/env python3
"""
Tests fuer lesen.py - Zellenleser der Neste-Berichte. Offline, nur Zeichenketten.

Die Faelle stammen aus echten Seiten: die Kapitalflussrechnung des GB 2025 (S. 154)
setzt hinter die rechte Tabelle eine Randspalte ("3"), vor die Werte eine Notenspalte
("13"); die Kennzahlenseite des GB 2021 (S. 162) setzt neben den Verschuldungsgrad die
Zelle "- of weighted average number of shares", deren Werte vor der Korrektur in die
Reihe gerieten und von der Kettenpruefung gefangen wurden.
"""
import unittest

from lesen import werte, werte_spalten

KOPF = " " * 60 + "1 Jan-31 Dec 2025    1 Jan-31 Dec 2024"


class TestWerteSpalten(unittest.TestCase):
    def _seite(self, zeile):
        return KOPF + "\n" + zeile

    def test_randspalte_hinter_der_tabelle_wird_ignoriert(self):
        # Werte enden unter den Koepfen; die "3" steht elf Zeichen dahinter.
        k1 = KOPF.index("2025") + 4
        k2 = KOPF.index("2024") + 4
        zeile = "Purchases of property, plant and equipment".ljust(k1 - 4) + "-910"
        zeile = zeile.ljust(k2 - 6) + "-1,525" + " " * 11 + "3"
        self.assertEqual(werte_spalten(self._seite(zeile), "Purchases of property, plant and equipment",
                                       r"Dec 20\d\d", 2), [-910.0, -1525.0])

    def test_notenspalte_vor_den_werten_wird_ignoriert(self):
        k1 = KOPF.index("2025") + 4
        k2 = KOPF.index("2024") + 4
        zeile = "Purchases of intangible assets".ljust(40) + "13"
        zeile = zeile.ljust(k1 - 3) + "-12"
        zeile = zeile.ljust(k2 - 3) + "-27"
        self.assertEqual(werte_spalten(self._seite(zeile), "Purchases of intangible assets",
                                       r"Dec 20\d\d", 2), [-12.0, -27.0])

    def test_fehlendes_etikett_gibt_none(self):
        self.assertIsNone(werte_spalten(self._seite("Other line  1  2"), "Missing", r"Dec 20\d\d", 2))


class TestWerte(unittest.TestCase):
    def test_nachbarzelle_mit_bindestrich_beendet_die_zelle(self):
        zeile = ("Leverage ratio        %      0.6     -4.7     -3.3     "
                 "- of weighted average number of shares   %   32   44   40")
        self.assertEqual(werte(zeile, r"Leverage ratio(?=\s{2})", 3), [0.6, -4.7, -3.3])

    def test_halbgeviertstrich_ist_minus(self):
        zeile = "Interest-bearing net debt   EUR million   –191   –70   412"
        self.assertEqual(werte(zeile, r"Interest-bearing net debt(?=\s{2})", 3), [-191.0, -70.0, 412.0])

    def test_fussnote_hinter_wert_zaehlt_nicht(self):
        zeile = "Dividend per share   EUR   0.20 1)   0.20   1.20"
        self.assertEqual(werte(zeile, "Dividend per share", 3), [0.20, 0.20, 1.20])


if __name__ == "__main__":
    unittest.main()
