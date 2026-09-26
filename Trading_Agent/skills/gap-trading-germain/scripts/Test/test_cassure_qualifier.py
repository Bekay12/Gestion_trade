#!/usr/bin/env python3
"""
test_cassure_qualifier.py - tests hors ligne du classement des cassures.

Stdlib seule, aucun appel reseau, aucune cle. Chaque test porte le nom de la
regle qu'il verifie, pour qu'un echec dise laquelle a ete cassee.

Les cas nommes viennent tous de la seance du 22.09.2026, la mesure qui a motive
ce module (voir docs/methode-gaps-et-cassures.md).

    python3 scripts/Test/test_cassure_qualifier.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cassure_qualifier import (Cassure, qualifier_cassure,          # noqa: E402
                               position_cloture, rvol_effectif,
                               tendance_etablie, signaux_pump,
                               CLOTURE_HAUTE, CLOTURE_VENDUE,
                               gap_efface)


def _base(**kw) -> Cassure:
    """Cassure saine et liquide par defaut; chaque test ne change que son objet."""
    defauts = dict(ticker="TEST", gap_pct=0.3, variation_seance_pct=15.0,
                   rvol=4.0, rvol_pre_tendance=4.2, prix=9.5,
                   vwap=9.0, haut_jour=10.0, bas_jour=7.0,
                   volume_jour=3_000_000, volume_moyen=800_000,
                   float_actions=50_000_000, market_cap=500_000_000,
                   short_interest_pct=5.0,
                   catalyseur="8-K: resultats publies", formulaire_sec="8-K",
                   seances_de_hausse=3)
    defauts.update(kw)
    return Cassure(**defauts)


class TestPositionCloture(unittest.TestCase):
    """L'arbitre propre a la cassure: ou la cloture tombe dans le range du jour."""

    def test_cloture_au_plus_haut_donne_un(self):
        self.assertAlmostEqual(position_cloture(_base(prix=10.0)), 1.0)

    def test_cloture_au_plus_bas_donne_zero(self):
        self.assertAlmostEqual(position_cloture(_base(prix=7.0)), 0.0)

    def test_indp_cloture_haut_dans_son_range(self):
        """INDP, 22.09.2026: cloture 3,99, haut 4,06, bas 2,95."""
        pos = position_cloture(_base(prix=3.99, haut_jour=4.06, bas_jour=2.95))
        self.assertAlmostEqual(pos, 0.937, places=2)
        self.assertGreater(pos, CLOTURE_HAUTE)

    def test_maze_aussi(self):
        """MAZE, 22.09.2026: cloture 28,38, haut 29,40, bas 24,50."""
        pos = position_cloture(_base(prix=28.38, haut_jour=29.40, bas_jour=24.50))
        self.assertAlmostEqual(pos, 0.792, places=2)
        self.assertGreater(pos, CLOTURE_HAUTE)

    def test_range_inconnu_ou_nul_rend_none(self):
        self.assertIsNone(position_cloture(_base(haut_jour=None)))
        self.assertIsNone(position_cloture(_base(haut_jour=9.0, bas_jour=9.0)))


class TestLiquidite(unittest.TestCase):
    """Meme regle que pour un gap: le volume du jour decide de l'execution."""

    def test_volume_du_jour_insuffisant_ecarte_le_titre(self):
        v = qualifier_cassure(_base(volume_jour=45_000))
        self.assertEqual(v.classe, "INSUFFISANT")

    def test_plancher_structurel_sur_le_volume_moyen(self):
        v = qualifier_cassure(_base(volume_moyen=40_000))
        self.assertEqual(v.classe, "INSUFFISANT")

    def test_adv_faible_mais_seance_liquide_passe(self):
        v = qualifier_cassure(_base(volume_jour=2_726_280, volume_moyen=290_127))
        self.assertNotEqual(v.classe, "INSUFFISANT")


class TestCassureConfirmee(unittest.TestCase):
    """Catalyseur + RVOL fort + au-dessus du VWAP + cloture haute dans le range."""

    def test_indp_est_une_cassure_confirmee(self):
        """Le cas qui a motive le module: aucune des deux grilles ne le voyait."""
        v = qualifier_cassure(_base(ticker="INDP", gap_pct=0.32,
                                    variation_seance_pct=28.71,
                                    rvol=1.99, rvol_pre_tendance=11.34,
                                    prix=3.99, vwap=3.70,
                                    haut_jour=4.06, bas_jour=2.95,
                                    volume_jour=1_072_003, volume_moyen=539_473,
                                    short_interest_pct=4.0, seances_de_hausse=10))
        self.assertEqual(v.classe, "CASSURE")
        self.assertTrue(any("base pre-tendance" in m for m in v.motifs))

    def test_le_short_interest_n_est_pas_requis(self):
        """Contrairement au SQUEEZE, la cassure ne demande aucun short interest."""
        v = qualifier_cassure(_base(short_interest_pct=1.0))
        self.assertEqual(v.classe, "CASSURE")

    def test_sous_le_vwap_ce_n_est_pas_une_cassure(self):
        v = qualifier_cassure(_base(prix=8.0, vwap=9.5, haut_jour=10.0, bas_jour=7.0))
        self.assertNotEqual(v.classe, "CASSURE")

    def test_rvol_faible_ne_confirme_pas(self):
        v = qualifier_cassure(_base(rvol=1.2, rvol_pre_tendance=1.3))
        self.assertNotEqual(v.classe, "CASSURE")

    def test_tendance_longue_raccourcit_l_horizon(self):
        court = qualifier_cassure(_base(seances_de_hausse=12))
        normal = qualifier_cassure(_base(seances_de_hausse=3))
        self.assertIn("tendance deja longue", court.horizon)
        self.assertIn("semaine", normal.horizon)

    def test_aucun_horizon_ne_depasse_la_semaine(self):
        for c in (_base(), _base(seances_de_hausse=12), _base(rvol=1.0)):
            self.assertNotIn("mois", qualifier_cassure(c).horizon)


class TestEpuisement(unittest.TestCase):
    """Une avance rendue en fin de seance invalide la cassure.

    Miroir mesure le 22.09.2026: LXEO a gappe de +5,6 % puis rendu -5,6 % de
    l'ouverture a la cloture, le schema inverse d'INDP le meme jour.
    """

    def test_cloture_dans_le_bas_du_range_donne_epuisement(self):
        v = qualifier_cassure(_base(prix=7.3, haut_jour=10.0, bas_jour=7.0,
                                    vwap=7.0))
        self.assertEqual(v.classe, "EPUISEMENT")
        self.assertLess(v.position_cloture, CLOTURE_VENDUE)

    def test_epuisement_prime_sur_la_confirmation(self):
        """Meme avec catalyseur et RVOL fort, une cloture basse invalide."""
        v = qualifier_cassure(_base(prix=7.2, haut_jour=10.0, bas_jour=7.0,
                                    vwap=7.0, rvol=9.0, rvol_pre_tendance=9.0))
        self.assertEqual(v.classe, "EPUISEMENT")

    def test_epuisement_n_annonce_aucun_horizon(self):
        v = qualifier_cassure(_base(prix=7.3, haut_jour=10.0, bas_jour=7.0, vwap=7.0))
        self.assertIn("aucun", v.horizon)


class TestGapEfface(unittest.TestCase):
    """Meme signal que cote gaps, cas VBIO du 24.09.2026."""

    def test_sous_la_cloture_veille_apres_etre_monte(self):
        self.assertTrue(gap_efface(_base(prix=2.67, cloture_veille=2.72,
                                         haut_jour=4.84, bas_jour=2.60,
                                         variation_seance_pct=-1.8)))

    def test_prime_sur_la_confirmation(self):
        """Meme avec catalyseur, RVOL fort et cloture haute dans le range."""
        v = qualifier_cassure(_base(prix=2.67, cloture_veille=2.72,
                                    haut_jour=2.70, bas_jour=2.60, vwap=2.62,
                                    rvol=9.0, rvol_pre_tendance=9.0))
        self.assertEqual(v.classe, "EPUISEMENT")
        self.assertIn("disparu", v.horizon)

    def test_au_dessus_de_la_veille_reste_une_cassure(self):
        v = qualifier_cassure(_base(cloture_veille=7.0))
        self.assertEqual(v.classe, "CASSURE")

    def test_cloture_veille_inconnue_ne_bloque_rien(self):
        self.assertIsNone(gap_efface(_base(cloture_veille=None)))
        self.assertEqual(qualifier_cassure(_base(cloture_veille=None)).classe,
                         "CASSURE")


class TestPump(unittest.TestCase):
    """Le risque de manipulation prime sur toute lecture haussiere."""

    def test_volume_explosif_sans_catalyseur_est_ecarte(self):
        v = qualifier_cassure(_base(catalyseur=None, formulaire_sec=None,
                                    rvol=12.0, rvol_pre_tendance=12.0))
        self.assertEqual(v.classe, "PUMP_RISK")

    def test_trois_signaux_suffisent(self):
        v = qualifier_cassure(_base(market_cap=30_000_000, float_actions=9_000_000,
                                    prix=7.5, vwap=9.0, haut_jour=10.0, bas_jour=7.0))
        self.assertEqual(v.classe, "PUMP_RISK")

    def test_s3_n_est_pas_un_catalyseur_haussier(self):
        v = qualifier_cassure(_base(catalyseur="S-3 depose", formulaire_sec="S-3"))
        self.assertNotEqual(v.classe, "CASSURE")
        self.assertTrue(any("dilution" in a for a in v.alertes))

    def test_avance_rendue_est_un_signal(self):
        alertes = signaux_pump(_base(prix=7.2, haut_jour=10.0, bas_jour=7.0))
        self.assertTrue(any("avance rendue" in a for a in alertes))


class TestDonneeAbsente(unittest.TestCase):
    """Convention du depot: une donnee absente ne vaut jamais feu vert."""

    def test_rvol_absent_ne_donne_jamais_cassure(self):
        v = qualifier_cassure(_base(rvol=None, rvol_pre_tendance=None))
        self.assertNotEqual(v.classe, "CASSURE")
        self.assertTrue(any("RVOL non mesurable" in i for i in v.inconnues))

    def test_range_absent_ne_donne_jamais_cassure(self):
        v = qualifier_cassure(_base(haut_jour=None, bas_jour=None))
        self.assertNotEqual(v.classe, "CASSURE")
        self.assertTrue(any("range de la seance" in i for i in v.inconnues))

    def test_volume_du_jour_inconnu_n_est_pas_un_volume_insuffisant(self):
        v = qualifier_cassure(_base(volume_jour=None))
        self.assertNotEqual(v.classe, "INSUFFISANT")
        self.assertTrue(any("volume du jour non mesure" in i for i in v.inconnues))

    def test_mesure_manquante_donne_a_surveiller(self):
        v = qualifier_cassure(_base(rvol=None, rvol_pre_tendance=None, vwap=None))
        self.assertEqual(v.classe, "A_SURVEILLER")


class TestRvolPreTendance(unittest.TestCase):
    """Meme correction que pour les gaps, et c'est ici qu'elle compte le plus:
    une cassure survient par definition au bout d'une tendance."""

    def test_le_plus_grand_des_deux_est_retenu(self):
        self.assertAlmostEqual(
            rvol_effectif(_base(rvol=1.99, rvol_pre_tendance=11.34)), 11.34)

    def test_tendance_detectee_par_l_ecart(self):
        self.assertTrue(tendance_etablie(_base(rvol=1.99, rvol_pre_tendance=11.34)))
        self.assertFalse(tendance_etablie(_base(rvol=13.45, rvol_pre_tendance=14.38)))

    def test_tendance_inconnue_sans_les_deux_mesures(self):
        self.assertIsNone(tendance_etablie(_base(rvol_pre_tendance=None)))


class TestVerdictComplet(unittest.TestCase):
    """Aucun verdict ne sort sans horizon ni declencheur de sortie."""

    def test_tout_verdict_porte_un_horizon_et_une_sortie(self):
        cas = [_base(),
               _base(volume_jour=45_000),
               _base(catalyseur=None, formulaire_sec=None, rvol=15.0,
                     rvol_pre_tendance=15.0),
               _base(prix=7.2, haut_jour=10.0, bas_jour=7.0, vwap=7.0),
               _base(rvol=None, rvol_pre_tendance=None, vwap=None),
               _base(rvol=1.0, rvol_pre_tendance=1.0, catalyseur=None,
                     formulaire_sec=None)]
        for c in cas:
            v = qualifier_cassure(c)
            self.assertTrue(v.horizon, f"{v.classe} sans horizon")
            self.assertTrue(v.sortie, f"{v.classe} sans declencheur de sortie")


if __name__ == "__main__":
    unittest.main(verbosity=2)
