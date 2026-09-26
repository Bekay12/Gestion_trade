#!/usr/bin/env python3
"""
test_lecture_finviz.py - tests hors ligne de la lecture des colonnes Finviz.

Stdlib seule, aucun appel reseau. Ces tests verrouillent le defaut du
23.09.2026: finvizfinance a renomme "Change" en "Change %" et a change le format
de la valeur, une fraction devenant une chaine deja en pourcentage. Le code
lisait l'ancien nom et multipliait par 100, donc gap_pct valait None pour chaque
candidat et la normalisation du gap par l'ATR n'avait plus lieu.

Le defaut etait silencieux: aucune exception, aucun verdict change, seulement
une mesure centrale absente du rapport. D'ou ces tests.

    python3 scripts/Test/test_lecture_finviz.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gap_scan import lire_variation_pct, _num, COLONNES_VARIATION   # noqa: E402


class TestFormatCourant(unittest.TestCase):
    """Format du 23.09.2026: colonne 'Change %', chaine deja en pourcentage."""

    def test_whlr_cas_reel(self):
        """WHLR, 23.09.2026 pre-marche. Recoupe sur IBKR: cloture 1,87 la
        veille, 5,31 au scan, 6,59 en direct. La valeur est bien le mouvement
        du jour."""
        self.assertAlmostEqual(lire_variation_pct({"Change %": "180.75%"}), 180.75)

    def test_valeur_negative(self):
        self.assertAlmostEqual(lire_variation_pct({"Change %": "-5.57%"}), -5.57)

    def test_separateur_de_milliers(self):
        self.assertAlmostEqual(lire_variation_pct({"Change %": "1,234.5%"}), 1234.5)

    def test_espaces_parasites(self):
        self.assertAlmostEqual(lire_variation_pct({"Change %": "  12.3%  "}), 12.3)


class TestAncienFormat(unittest.TestCase):
    """Format anterieur: colonne 'Change', nombre nu exprimant une fraction."""

    def test_fraction_est_convertie(self):
        self.assertAlmostEqual(lire_variation_pct({"Change": 0.0477}), 4.77)

    def test_fraction_negative(self):
        self.assertAlmostEqual(lire_variation_pct({"Change": -0.0557}), -5.57)

    def test_le_nom_recent_prime_sur_l_ancien(self):
        """Si les deux colonnes existent, la plus recente fait foi."""
        self.assertAlmostEqual(
            lire_variation_pct({"Change %": "10.0%", "Change": 0.99}), 10.0)


class TestDonneeAbsente(unittest.TestCase):
    """Convention du depot: une valeur illisible rend None, jamais 0.

    Confondre les deux ferait passer un titre pour immobile alors qu'on ignore
    simplement son mouvement, exactement l'erreur deja corrigee sur le RVOL.
    """

    def test_aucune_colonne(self):
        self.assertIsNone(lire_variation_pct({}))

    def test_valeur_nulle(self):
        self.assertIsNone(lire_variation_pct({"Change %": None}))

    def test_marqueurs_de_vide(self):
        for vide in ("", "  ", "nan", "NaN", "None", "-"):
            self.assertIsNone(lire_variation_pct({"Change %": vide}), vide)

    def test_texte_illisible(self):
        self.assertIsNone(lire_variation_pct({"Change %": "indisponible"}))

    def test_zero_reste_zero(self):
        """Un vrai zero est une mesure et doit survivre."""
        self.assertAlmostEqual(lire_variation_pct({"Change %": "0.00%"}), 0.0)


class TestNum(unittest.TestCase):
    """Le convertisseur partage, utilise aussi pour volume et capitalisation."""

    def test_suffixes(self):
        self.assertAlmostEqual(_num("1.2M"), 1_200_000)
        self.assertAlmostEqual(_num("3.5K"), 3_500)
        self.assertAlmostEqual(_num("2B"), 2_000_000_000)

    def test_separateurs_et_pourcent(self):
        self.assertAlmostEqual(_num("17,415,569"), 17_415_569)
        self.assertAlmostEqual(_num("45.3%"), 45.3)

    def test_illisible_rend_none(self):
        self.assertIsNone(_num("abc"))


class TestContrat(unittest.TestCase):

    def test_ordre_des_colonnes_documente(self):
        """La plus recente d'abord: l'ordre porte la logique de repli."""
        self.assertEqual(COLONNES_VARIATION[0], "Change %")
        self.assertIn("Change", COLONNES_VARIATION)

    def test_les_deux_screeners_partagent_l_implementation(self):
        """Un seul point a corriger au prochain changement de format."""
        import gap_scan
        import cassure_scan
        self.assertIs(cassure_scan.lire_variation_pct, gap_scan.lire_variation_pct)
        self.assertIs(cassure_scan._num, gap_scan._num)


if __name__ == "__main__":
    unittest.main(verbosity=2)
