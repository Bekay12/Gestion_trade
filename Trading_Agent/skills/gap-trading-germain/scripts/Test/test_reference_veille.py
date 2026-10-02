#!/usr/bin/env python3
"""
Tests hors ligne de gap_scan.reference_veille(). Regression du 29.09.2026: en
pre-marche la cloture de la veille etait lue sur l'avant-veille (MSGY annonce a
-53,7 % "sous la veille" alors qu'il montait de 19,5 %). Stdlib + pandas.
"""
import os
import sys
import unittest
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gap_scan import reference_veille   # noqa: E402

C = pd.Series([8.07, 3.13])     # avant-veille, veille (derniere barre)
O = pd.Series([8.00, 3.50])


class TestReferenceVeille(unittest.TestCase):
    def test_pre_marche_derniere_barre_est_la_veille(self):
        veille, ouv = reference_veille(date(2026, 9, 28), C, O, jour=date(2026, 9, 29))
        self.assertEqual(veille, 3.13)
        self.assertIsNone(ouv)

    def test_en_seance_avant_derniere_barre(self):
        veille, ouv = reference_veille(date(2026, 9, 29), C, O, jour=date(2026, 9, 29))
        self.assertEqual(veille, 8.07)
        self.assertEqual(ouv, 3.50)

    def test_une_seule_barre_en_seance(self):
        veille, ouv = reference_veille(date(2026, 9, 29), C[-1:], O[-1:], jour=date(2026, 9, 29))
        self.assertIsNone(veille)


if __name__ == "__main__":
    unittest.main()
