#!/usr/bin/env python3
"""
Tests hors ligne de gap_scan.perimetre(): plafonds de prix et de capitalisation
appliqués aux valeurs AVANT le gap, pas au cours gonflé par le gap.

Régression: KOD, 28.09.2026. Clôture de la veille 32,35 $, pré-marché ~61 $ (+89 %),
62,85 M actions. Finviz l'écartait sur « Price Under $50 » et « Market Cap. under
$2bln »; il était le titre de la revue Academy Germain du jour et a clôturé
+46,7 % au-dessus de son ouverture.

Aucun réseau, stdlib seule.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gap_scan import perimetre, PRIX_MAX_VEILLE, CAP_SMALL_MAX   # noqa: E402


def ligne(ticker, prix, change_pct, market_cap):
    return {"ticker": ticker, "prix": prix, "change_pct": change_pct, "market_cap": market_cap}


class TestPerimetre(unittest.TestCase):
    def test_kod_est_garde_et_marque_hors_small_cap(self):
        # 61,28 $ au scan, +89,4 %; capitalisation Finviz au cours du gap ~3,85 Md$
        g = perimetre([ligne("KOD", 61.28, 89.43, 3.851e9)])
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(g[0]["prix_veille"], 32.35, places=1)
        self.assertAlmostEqual(g[0]["cap_avant_gap"] / 1e9, 2.03, places=2)
        self.assertTrue(g[0]["hors_small_cap"])

    def test_prix_veille_au_dessus_du_plafond_est_ecarte(self):
        # 72 $ la veille, +10 %: exclu par le plafond de 50 $ appliqué à la veille
        self.assertEqual(perimetre([ligne("XXX", 79.2, 10.0, 1.0e9)]), [])

    def test_small_cap_reste_dans_le_perimetre(self):
        g = perimetre([ligne("SMAL", 3.0, 50.0, 150e6)])
        self.assertAlmostEqual(g[0]["prix_veille"], 2.0)
        self.assertFalse(g[0]["hors_small_cap"])
        self.assertLess(g[0]["cap_avant_gap"], CAP_SMALL_MAX)

    def test_au_dela_de_dix_milliards_avant_gap_est_ecarte(self):
        self.assertEqual(perimetre([ligne("BIG", 40.0, 10.0, 12e9)]), [])

    def test_variation_illisible_garde_la_ligne_et_la_marque(self):
        g = perimetre([ligne("NA", 60.0, None, 5e9)])
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0]["avant_gap_inconnu"])
        self.assertIsNone(g[0]["prix_veille"])

    def test_plafond_de_prix_inchange(self):
        self.assertEqual(PRIX_MAX_VEILLE, 50.0)


if __name__ == "__main__":
    unittest.main()
