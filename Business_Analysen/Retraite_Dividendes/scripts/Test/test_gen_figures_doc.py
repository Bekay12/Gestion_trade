"""Tests hors ligne de scripts/gen_figures_doc.py (rafraichissement de docs/figures.md).

Regression visee (tache 10, chapitre 07): l'exemple "\\AlterMaisonSiebzigZwanzig = ..."
gagnait un mot "atteint" a chaque build, parce que seule la premiere partie d'une valeur
de plusieurs mots ("non" de "non atteint") etait remplacee."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import gen_figures_doc as g  # noqa: E402

EXEMPLE = ("Ex. `\\AlterMaisonSiebzigZwanzig` = {} (repartition maison 70 %, "
           "taux d'epargne 20 %).\n")


class TestExempleAlter(unittest.TestCase):
    def test_valeur_de_plusieurs_mots_stable_sur_deux_passages(self):
        """Deux rafraichissements successifs doivent donner le meme texte (idempotence)."""
        macros = {"AlterMaisonSiebzigZwanzig": "non atteint"}
        premier, _, _ = g.rafraichir(EXEMPLE.format("non atteint"), macros)
        second, _, _ = g.rafraichir(premier, macros)
        self.assertEqual(premier, EXEMPLE.format("non atteint"))
        self.assertEqual(second, premier)

    def test_texte_deja_corrompu_est_repare(self):
        """Le fichier committe portait deja des "atteint" en trop: un passage les retire."""
        macros = {"AlterMaisonSiebzigZwanzig": "non atteint"}
        corrompu = EXEMPLE.format("non atteint atteint atteint")
        nouveau, n, _ = g.rafraichir(corrompu, macros)
        self.assertEqual(nouveau, EXEMPLE.format("non atteint"))
        self.assertEqual(n, 1)

    def test_passage_d_un_age_a_la_sentinelle_et_retour(self):
        texte, _, _ = g.rafraichir(EXEMPLE.format("70"), {"AlterMaisonSiebzigZwanzig": "non atteint"})
        self.assertEqual(texte, EXEMPLE.format("non atteint"))
        texte, n, _ = g.rafraichir(texte, {"AlterMaisonSiebzigZwanzig": "68"})
        self.assertEqual(texte, EXEMPLE.format("68"))
        self.assertEqual(n, 1)

    def test_valeur_inchangee_ne_compte_pas(self):
        _, n, _ = g.rafraichir(EXEMPLE.format("68"), {"AlterMaisonSiebzigZwanzig": "68"})
        self.assertEqual(n, 0)


class TestLignesDeTableau(unittest.TestCase):
    def test_ligne_de_tableau_rafraichie_et_sens_intact(self):
        texte = "| `\\KapitalYannBasis` | 1\\,000 | Capital de Yann |\n"
        nouveau, n, _ = g.rafraichir(texte, {"KapitalYannBasis": "3\\,108\\,376"})
        self.assertEqual(nouveau, "| `\\KapitalYannBasis` | 3\\,108\\,376 | Capital de Yann |\n")
        self.assertEqual(n, 1)

    def test_valeur_texte_laissee_intacte(self):
        texte = "| `\\HypotheseDivSchock` | texte | Sens du choc |\n"
        nouveau, n, ignorees = g.rafraichir(texte, {"HypotheseDivSchock": "Le choc est un \\'ecart"})
        self.assertEqual(nouveau, texte)
        self.assertEqual((n, ignorees), (0, 1))


if __name__ == "__main__":
    unittest.main()
