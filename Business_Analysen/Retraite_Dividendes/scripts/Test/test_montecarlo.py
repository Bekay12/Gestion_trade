"""
test_montecarlo.py - Tests pour montecarlo.py (bootstrap par blocs, succes/echec retraite).

Trois tests viennent du brief de tache (forme/graine, monde constant -> succes garanti).
Trois tests supplementaires couvrent les decisions du controleur (2026-09-29):
- test_block_start_beide_moeglich : borne superieure inclusive de rng.integers (decision 1) -
  avec un pool de 6 lignes et block=5, les deux departs possibles (0 et 1) doivent survenir.
- test_erste_zeile_nicht_tiree : la premiere ligne de jahres (placeholder de jahresreihe,
  div_wachstum_real force a 0.0) ne doit jamais etre tiree (decision 2).
- test_abgeschnitten_gezaehlt : une trajectoire qui atteint l'objectif trop tard pour derouler
  tout rentenjahre dans la fenetre simulee doit etre comptee dans "abgeschnitten", pas
  seulement ignoree en silence (decision 3).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import montecarlo as mc   # noqa: E402

J = pd.DataFrame({"jahr": range(1900, 2000), "rendite_real": [0.05] * 100,
                  "div_wachstum_real": [0.01] * 100})


class TestMC(unittest.TestCase):
    def test_form_und_seed(self):
        a = mc.pfade(J, n=50, jahre=30, seed=1)
        b = mc.pfade(J, n=50, jahre=30, seed=1)
        self.assertEqual(a.shape, (50, 30, 2))
        self.assertTrue(np.array_equal(a, b))

    def test_konstante_welt_erfolg_eins(self):
        p = mc.pfade(J, n=20, jahre=80, seed=1)
        r = mc.erfolg(p, [20000.0] * 40, ziel_real_jahr=20000.0, rentenjahre=30)
        self.assertEqual(r["erfolgsquote"], 1.0)
        self.assertEqual(r["abgeschnitten"], 0)

    def test_block_start_beide_moeglich(self):
        # 7 lignes -> apres retrait de la 1ere (placeholder), pool de 6 lignes ; avec
        # block=5, len(daten) - block + 1 = 2 departs possibles (0 et 1). L'ancienne borne
        # exclusive (len(daten) - block) n'aurait jamais tire le depart 1.
        j = pd.DataFrame({"jahr": range(2000, 2007),
                          "rendite_real": [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66],
                          "div_wachstum_real": [0.0] * 7})
        p = mc.pfade(j, n=500, jahre=5, block=5, seed=42)
        premiers = {round(x, 2) for x in p[:, 0, 0]}
        self.assertIn(0.11, premiers)   # depart 0 (indices 1..5 de j) -> premiere valeur 0.11
        self.assertIn(0.22, premiers)   # depart 1 (indices 2..6 de j) -> premiere valeur 0.22

    def test_erste_zeile_nicht_tiree(self):
        # Valeur sentinelle -0.99 sur la premiere ligne uniquement : si elle apparait dans
        # une trajectoire, la premiere ligne a ete tiree a tort.
        j = pd.DataFrame({"jahr": range(1900, 2000),
                          "rendite_real": [-0.99] + [0.05] * 99,
                          "div_wachstum_real": [-0.99] + [0.01] * 99})
        p = mc.pfade(j, n=30, jahre=50, seed=7)
        self.assertFalse(np.any(p == -0.99))

    def test_abgeschnitten_gezaehlt(self):
        # Monde constant : l'objectif est atteint a t=18 (calcul a la main, croissance
        # composee 5 % + versements de 20000). Avec jahre=40 et rentenjahre=30,
        # erreicht + rentenjahre = 48 >= 40 : toutes les trajectoires sont tronquees.
        p = mc.pfade(J, n=15, jahre=40, seed=3)
        r = mc.erfolg(p, [20000.0] * 40, ziel_real_jahr=20000.0, rentenjahre=30)
        self.assertEqual(r["abgeschnitten"], 15)
        self.assertEqual(r["erfolgsquote"], 0.0)

    def test_div_schock_defaut_neutre_bit_a_bit(self):
        """Revue de code (tache 9, fix round 1): div_schock_erstes_rentenjahr est
        additif et vaut 0.0 par defaut, donc erfolg() doit rendre EXACTEMENT le meme
        resultat avec et sans l'argument (aucune regression sur les appelants
        existants qui ne le passent pas)."""
        j = pd.DataFrame({"jahr": range(1950, 2023),
                          "rendite_real": np.linspace(-0.3, 0.4, 73),
                          "div_wachstum_real": np.linspace(-0.2, 0.15, 73)})
        p = mc.pfade(j, n=200, jahre=60, seed=11)
        sans = mc.erfolg(p, [15000.0] * 30, ziel_real_jahr=42000.0, rentenjahre=20)
        avec = mc.erfolg(p, [15000.0] * 30, ziel_real_jahr=42000.0, rentenjahre=20,
                         div_schock_erstes_rentenjahr=0.0)
        self.assertEqual(sans["erfolgsquote"], avec["erfolgsquote"])
        self.assertEqual(sans["abgeschnitten"], avec["abgeschnitten"])
        self.assertTrue(np.array_equal(sans["wert_perzentile"], avec["wert_perzentile"]))
        self.assertEqual(sans["ziel_jahr_perzentile"], avec["ziel_jahr_perzentile"])

    def test_div_schock_negatif_baisse_le_taux_de_reussite(self):
        """Un choc negatif sur la croissance du dividende de la premiere annee de
        retraite ne peut jamais ameliorer, et doit ici degrader, le taux de reussite
        par rapport au meme scenario sans choc (monde volatil, marge de securite non
        nulle a l'entree en retraite)."""
        j = pd.DataFrame({"jahr": range(1950, 2023),
                          "rendite_real": np.linspace(-0.3, 0.4, 73),
                          "div_wachstum_real": np.linspace(-0.2, 0.15, 73)})
        p = mc.pfade(j, n=200, jahre=60, seed=11)
        base = mc.erfolg(p, [15000.0] * 30, ziel_real_jahr=42000.0, rentenjahre=20)
        choque = mc.erfolg(p, [15000.0] * 30, ziel_real_jahr=42000.0, rentenjahre=20,
                           div_schock_erstes_rentenjahr=-0.30)
        self.assertLess(choque["erfolgsquote"], base["erfolgsquote"])


if __name__ == "__main__":
    unittest.main()
