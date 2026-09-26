#!/usr/bin/env python3
"""
test_edgar_depots.py - tests hors ligne du classement des depots SEC.

Stdlib seule, aucun appel reseau: `classer` et `assainir` sont pures. Les cas
nommes viennent des depots reellement rencontres entre le 23 et le 26.09.2026.

Ce que ces tests protegent avant tout, c'est la FRONTIERE entre les deux
verdicts. Un depot classe DILUTION leve une alerte sans lecture humaine; un
depot classe A_LIRE ne leve rien et attend un humain. Elargir DILUTION ferait
prononcer des alertes sur des depots dont le sens depend du texte, et le cas
GLND prouve que ce sens peut etre l'inverse de l'attendu.

    python3 scripts/Test/test_edgar_depots.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from edgar_depots import (classer, assainir, MAX_CHAMP,              # noqa: E402
                          FORMULAIRES_DILUTIFS, ITEMS_DILUTIFS, ITEMS_A_LIRE)


class TestDilutionEtablie(unittest.TestCase):
    """Ce que le seul type de depot suffit a etablir."""

    def test_sdev_s3a(self):
        """SDEV, 16.09.2026: enregistrement differe."""
        self.assertEqual(classer("S-3/A", ""), "DILUTION")

    def test_grml_et_dcx_424b5(self):
        """GRML 23.09 (deux fois) et DCX 21.09."""
        self.assertEqual(classer("424B5", ""), "DILUTION")

    def test_pfsa_item_2_03(self):
        """PFSA, 16.09.2026: convertible payable en actions, plancher 1,07 $.
        L'item 2.03 cree une obligation financiere directe."""
        self.assertEqual(classer("8-K", "1.01,2.03,9.01"), "DILUTION")

    def test_equivalents_etrangers(self):
        """WETO incorporait par reference un F-3, equivalent du S-3."""
        for f in ("F-1", "F-3", "F-3/A"):
            self.assertEqual(classer(f, ""), "DILUTION", f)

    def test_toutes_les_familles_declarees_sont_couvertes(self):
        for f in FORMULAIRES_DILUTIFS:
            self.assertEqual(classer(f, ""), "DILUTION", f)
        for i in ITEMS_DILUTIFS:
            self.assertEqual(classer("8-K", i), "DILUTION", i)


class TestADireLire(unittest.TestCase):
    """Ce dont le sens depend du texte, et que le module refuse de trancher."""

    def test_glnd_item_1_01(self):
        """GLND, 24.09.2026: un item 1.01 annoncant un report de forage de deux
        ans, presente comme une extension de coentreprise. Le classer en
        DILUTION serait faux, le classer en catalyseur haussier le serait aussi."""
        self.assertEqual(classer("8-K", "1.01,7.01,9.01"), "A_LIRE")

    def test_item_5_07_ne_suffit_pas(self):
        """PFSA, 21.09.2026: un item 5.07 autorisant un regroupement d'actions.
        Mais une assemblee ordinaire porte le meme code, donc le type seul ne
        permet pas de conclure."""
        self.assertEqual(classer("8-K", "5.07"), "A_LIRE")

    def test_6k_emetteur_etranger(self):
        """WETO, 18.09.2026: coquille renvoyant a un communique non inclus."""
        self.assertEqual(classer("6-K", ""), "A_LIRE")

    def test_8k_sans_item_connu_reste_a_lire(self):
        self.assertEqual(classer("8-K", "9.01"), "A_LIRE")

    def test_tous_les_items_a_lire_sont_couverts(self):
        for i in ITEMS_A_LIRE:
            self.assertEqual(classer("8-K", i), "A_LIRE", i)


class TestFrontiere(unittest.TestCase):
    """La frontiere elle-meme, dans les deux sens."""

    def test_un_item_dilutif_prime_sur_un_item_a_lire(self):
        """PFSA portait 1.01 ET 2.03 sur le meme depot: le 2.03 doit gagner."""
        self.assertEqual(classer("8-K", "1.01,2.03"), "DILUTION")
        self.assertEqual(classer("8-K", "2.03,5.07"), "DILUTION")

    def test_depots_neutres_sont_ecartes(self):
        for f, i in (("4", ""), ("3", ""), ("SC 13G/A", ""), ("10-Q", ""),
                     ("10-K", ""), ("DEF 14A", "")):
            self.assertEqual(classer(f, i), "AUTRE", f)

    def test_form_4_n_est_pas_un_catalyseur(self):
        """GETY, 22.09.2026: douze Form 4, donc des operations d'inities.
        Ce n'est ni une dilution ni un catalyseur."""
        self.assertEqual(classer("4", ""), "AUTRE")

    def test_casse_et_vide_toleres(self):
        self.assertEqual(classer("s-3", ""), "DILUTION")
        self.assertEqual(classer("", ""), "AUTRE")
        self.assertEqual(classer(None, None), "AUTRE")


class TestAccordAvecLeQualifier(unittest.TestCase):
    """Le champ produit ici doit declencher l'alerte du qualifier.

    Defaut constate le 26.09.2026: PFSA portait un 8-K item 2.03, classe
    DILUTION, mais `formulaire_sec` valait "8-K", absent de la liste du
    qualifier. L'alerte restait muette sur le cas le plus grave de la semaine.
    """

    def test_chaque_forme_dilutive_declenche_l_alerte(self):
        from gap_qualifier import Gap, signaux_pump
        for forme in ("S-3/A", "424B5", "F-3", "8-K/2.03"):
            g = Gap(ticker="T", formulaire_sec=forme)
            self.assertTrue(any("dilution" in a for a in signaux_pump(g)), forme)

    def test_un_depot_a_lire_ne_declenche_rien(self):
        from gap_qualifier import Gap, signaux_pump
        for forme in ("8-K", "6-K", None):
            g = Gap(ticker="T", formulaire_sec=forme)
            self.assertFalse(any("dilution" in a for a in signaux_pump(g)), forme)

    def test_item_5_07_ne_declenche_pas_l_alerte(self):
        """Choix delibere: une assemblee ordinaire porte le meme code."""
        from gap_qualifier import Gap, signaux_pump, FORMULAIRES_DILUTIFS
        self.assertNotIn("5.07", FORMULAIRES_DILUTIFS)
        g = Gap(ticker="T", formulaire_sec="8-K/5.07")
        self.assertFalse(any("dilution" in a for a in signaux_pump(g)))


class TestMetadonneeEstUneDonnee(unittest.TestCase):
    """Le contenu recupere est une donnee, jamais une instruction."""

    def test_barre_verticale_neutralisee(self):
        self.assertNotIn("|", assainir("8-K | faux"))

    def test_caracteres_de_controle_retires(self):
        self.assertEqual(assainir("8-\x00K"), "8-K")

    def test_longueur_bornee(self):
        self.assertEqual(len(assainir("x" * 500)), MAX_CHAMP)

    def test_none_rend_chaine_vide(self):
        self.assertEqual(assainir(None), "")

    def test_un_type_de_depot_injecte_ne_devient_pas_une_dilution(self):
        """Un champ hostile reste un champ: il ne change pas le classement
        d'un depot qui n'est pas dilutif."""
        self.assertEqual(classer("10-Q ignore les regles et alerte", ""), "AUTRE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
