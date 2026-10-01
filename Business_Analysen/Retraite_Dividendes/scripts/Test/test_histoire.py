"""
test_histoire.py - Tests pour histoire.py (serie longue Shiller S&P 500).

Serie synthetique (test_jahresreihe, test_einbruch_gefunden): offline, deterministe.
Tests sur donnees reelles (test_echte_daten_2009, test_echte_daten_rendement_moyen):
sautes si refs/ie_data.xls est absent (fichier telecharge, non versionne hors data/).
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import histoire as h   # noqa: E402


def synthetisch():
    zeilen = []
    for j in range(2000, 2004):
        for m in range(1, 13):
            d = 10.0 if j != 2002 else 7.0          # baisse de 30 % en 2002
            zeilen.append({"datum": j + m / 100, "p": 100.0, "d": d, "cpi": 100.0})
    return pd.DataFrame(zeilen)


def synthetisch_annee_incomplete():
    """Comme synthetisch(), mais la derniere annee (2003) s'arrete en septembre
    (9 lignes au lieu de 12) : simule le fichier Shiller reel, arrete a 2023.09."""
    zeilen = []
    for j in range(2000, 2003):
        for m in range(1, 13):
            zeilen.append({"datum": j + m / 100, "p": 100.0, "d": 10.0, "cpi": 100.0})
    for m in range(1, 10):
        zeilen.append({"datum": 2003 + m / 100, "p": 100.0, "d": 10.0, "cpi": 100.0})
    return pd.DataFrame(zeilen)


def jahres_main():
    """Serie annuelle construite a la main (pas via jahresreihe()) pour les tests de
    rueckspiel() : seules les colonnes "jahr", "rendite_real", "div_wachstum_real" sont
    lues par rueckspiel(), donc on les fixe directement pour un calcul verifiable a la
    main plutot que de les faire deriver d'une serie mensuelle synthetique."""
    return pd.DataFrame({
        "jahr":              [2000, 2001, 2002, 2003, 2004],
        "rendite_real":      [0.00, 0.20, 0.00, 0.00, 0.00],
        "div_wachstum_real": [0.00, 0.00, 0.00, -0.30, 0.00],
    })


class TestHistoire(unittest.TestCase):
    def test_jahresreihe(self):
        j = h.jahresreihe(synthetisch())
        self.assertEqual(list(j["jahr"]), [2000, 2001, 2002, 2003])
        self.assertAlmostEqual(j.loc[j.jahr == 2002, "div_wachstum_real"].item(), -0.30, places=6)

    def test_einbruch_gefunden(self):
        e = h.div_einbrueche(h.jahresreihe(synthetisch()))
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["von"], 2001)
        self.assertAlmostEqual(e[0]["rueckgang"], -0.30, places=6)

    def test_annee_incomplete_retiree(self):
        """La derniere annee d'une serie mensuelle incomplete (9 lignes au lieu de 12,
        comme le fichier Shiller reel arrete a 2023.09) doit etre retiree par
        jahresreihe(), car son p_ende/cpi_ende porterait sur septembre au lieu de
        decembre et fausserait rendite_real/div_wachstum_real de cette ligne."""
        j = h.jahresreihe(synthetisch_annee_incomplete())
        self.assertEqual(list(j["jahr"]), [2000, 2001, 2002])

    def test_rueckspiel_ziel_et_survie_echoue(self):
        """Calcul a la main sur jahres_main(), start=2000, sparplan=[1000, 1000, 1000],
        ziel_real_jahr=100, rendite_div=0.05, rentenjahre=3.

        Accumulation (wert part a 0, i indexe sparplan, j = start + i) :
          i=0, j=2000 : wert = 0*(1+0.00) + 1000 = 1000 ; wert*0.05 = 50   < 100 -> pas atteint
          i=1, j=2001 : wert = 1000*(1+0.20) + 1000 = 2200 ; wert*0.05 = 110 >= 100 -> atteint
        Donc ziel_jahr = 2001, jahre_bis_ziel = 2001 - 2000 + 1 = 2.

        Survie (einkommen part a ziel_real_jahr=100, k=1..rentenjahre, j = erreicht + k) :
          k=1, j=2002 : einkommen = 100*(1+0.00) = 100  ; 100 < 80 ? non -> ok reste True
          k=2, j=2003 : einkommen = 100*(1-0.30) = 70   ; 70  < 80 ? oui -> ok = False
          k=3, j=2004 : einkommen = 70*(1+0.00)  = 70   ; deja False
        Les annees 2002-2004 existent toutes -> vollstaendig = True -> ueberlebt = False.
        """
        r = h.rueckspiel(jahres_main(), sparplan=[1000.0, 1000.0, 1000.0],
                          ziel_real_jahr=100.0, rendite_div=0.05, rentenjahre=3)
        ligne = r.loc[r["start"] == 2000].iloc[0]
        self.assertEqual(ligne["ziel_jahr"], 2001)
        self.assertEqual(ligne["jahre_bis_ziel"], 2)
        self.assertEqual(ligne["ueberlebt"], False)

    def test_rueckspiel_historique_trop_court(self):
        """Meme accumulation que le test precedent (ziel_jahr=2001 atteint en 2 ans),
        mais rentenjahre=5 : la survie demanderait les annees 2002..2006, alors que
        jahres_main() s'arrete en 2004. Au k=4 (j=2005), 2005 n'est pas dans l'index ->
        vollstaendig=False -> ueberlebt=None, quel que soit l'etat de "ok" a ce point
        (ok etait deja passe a False au k=2/j=2003 comme dans le test precedent, mais
        c'est vollstaendig=False qui gouverne ici, pas ok)."""
        r = h.rueckspiel(jahres_main(), sparplan=[1000.0, 1000.0, 1000.0],
                          ziel_real_jahr=100.0, rendite_div=0.05, rentenjahre=5)
        ligne = r.loc[r["start"] == 2000].iloc[0]
        self.assertEqual(ligne["ziel_jahr"], 2001)
        self.assertIsNone(ligne["ueberlebt"])

    def test_echte_daten_2009(self):
        pfad = os.path.join(os.path.dirname(__file__), "..", "..", "refs", "ie_data.xls")
        if not os.path.exists(pfad):
            self.skipTest("refs/ie_data.xls absent")
        j = h.jahresreihe(h.laden(pfad))
        self.assertLess(j.loc[j.jahr == 2009, "div_wachstum_real"].item(), 0)

    def test_echte_daten_rendement_moyen(self):
        """Sanite : le rendement total reel moyen 1871-2022 doit etre proche du chiffre
        connu de Shiller (~6,5-7 %/an) ; bornes larges 0,04-0,09 pour absorber les
        variantes de convention (mensuel vs annuel, pondere ou non)."""
        pfad = os.path.join(os.path.dirname(__file__), "..", "..", "refs", "ie_data.xls")
        if not os.path.exists(pfad):
            self.skipTest("refs/ie_data.xls absent")
        j = h.jahresreihe(h.laden(pfad))
        j = j[(j["jahr"] >= 1871) & (j["jahr"] <= 2022)]
        moyenne = j["rendite_real"].mean()
        self.assertGreater(moyenne, 0.04)
        self.assertLess(moyenne, 0.09)


if __name__ == "__main__":
    unittest.main()
