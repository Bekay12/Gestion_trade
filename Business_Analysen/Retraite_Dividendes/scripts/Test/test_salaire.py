"""
Bareme §32a: valeurs calculees avec les coefficients 2025 (connus et publies), pour que
le test ne depende pas de la valeur 2026 trouvee dans la nuit.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import salaire as s   # noqa: E402

TARIF_2025 = {"zonen": [[12096, "null", 0, 0, 0],
                        [17443, "y", 932.30, 1400, 0],
                        [68480, "z", 176.64, 2397, 1015.13],
                        [277825, "lin", 0.42, -10911.92, 0],
                        [None, "lin", 0.45, -19246.67, 0]]}


class TestEst(unittest.TestCase):
    def test_werte_2025(self):
        for zve, erwartet in ((10000, 0), (15000, 485), (30000, 4303), (60000, 14415),
                              (100000, 31088), (300000, 115753)):
            self.assertEqual(s.est_32a(zve, TARIF_2025), erwartet, zve)

    def test_netto_kleiner_als_brutto_und_positiv(self):
        sv = {"rv": 0.093, "av": 0.013, "kv_allgemein": 0.146, "kv_zusatz": 0.025, "pv": 0.036,
              "pv_kinderlos_zuschlag": 0.006, "bbg_rv_jahr": 96600, "bbg_kv_jahr": 66150}
        n = s.netto_jahr(60000, TARIF_2025, sv, 19950)
        self.assertTrue(0.55 * 60000 < n < 0.70 * 60000, n)


if __name__ == "__main__":
    unittest.main()
