#!/usr/bin/env python3
"""
test_gap_qualifier.py - tests hors ligne du classement des gaps.

Stdlib seule, aucun appel reseau, aucune cle. Chaque test porte le nom de la
regle de la methode qu'il verifie, pour qu'un echec dise laquelle a ete cassee.

    python3 scripts/Test/test_gap_qualifier.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gap_qualifier import (Gap, qualifier, gap_en_atr,              # noqa: E402
                           probabilite_comblement, signaux_pump)


def _base(**kw) -> Gap:
    """Gap liquide et sain par defaut; chaque test ne change que ce qu'il teste."""
    defauts = dict(ticker="TEST", gap_pct=10.0, atr_pct=5.0, rvol=4.0, prix=10.0,
                   vwap=9.5, volume_moyen=1_000_000, float_actions=50_000_000,
                   market_cap=500_000_000, short_interest_pct=5.0,
                   catalyseur="8-K: contrat signe", formulaire_sec="8-K",
                   volume_pic_au_sommet=False)
    defauts.update(kw)
    return Gap(**defauts)


class TestNormalisationATR(unittest.TestCase):
    """Un gap de 5 % ne vaut pas la meme chose selon la volatilite du titre."""

    def test_gap_normalise_par_atr(self):
        self.assertAlmostEqual(gap_en_atr(10.0, 5.0), 2.0)
        self.assertAlmostEqual(gap_en_atr(3.0, 10.0), 0.3)

    def test_atr_absent_rend_none(self):
        self.assertIsNone(gap_en_atr(10.0, None))
        self.assertIsNone(gap_en_atr(10.0, 0))

    def test_probabilite_decroit_avec_la_taille(self):
        # Tableau de references/methode-germain.md
        self.assertEqual(probabilite_comblement(0.2), 78)
        self.assertEqual(probabilite_comblement(0.5), 42)
        self.assertEqual(probabilite_comblement(1.0), 25)
        self.assertEqual(probabilite_comblement(2.0), 8)

    def test_probabilite_inconnue_sans_atr(self):
        self.assertIsNone(probabilite_comblement(None))


class TestLiquidite(unittest.TestCase):
    """P7 Ch.01: ADV < 500K = risque pour le trading actif."""

    def test_volume_moyen_insuffisant_ecarte_le_titre(self):
        v = qualifier(_base(volume_moyen=200_000))
        self.assertEqual(v.classe, "INSUFFISANT")
        self.assertEqual(v.horizon, "aucun")

    def test_volume_suffisant_ne_bloque_pas(self):
        self.assertNotEqual(qualifier(_base(volume_moyen=600_000)).classe, "INSUFFISANT")


class TestPump(unittest.TestCase):
    """P7 Ch.03: les sept signaux d'alarme, et la primaute du risque."""

    def test_volume_explosif_sans_catalyseur_est_ecarte(self):
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertEqual(v.classe, "PUMP_RISK")
        self.assertEqual(v.horizon, "aucun")

    def test_trois_signaux_suffisent(self):
        v = qualifier(_base(market_cap=30_000_000,        # nano cap
                            float_actions=9_000_000,      # low float
                            prix=8.0, vwap=10.0))         # sous le VWAP
        self.assertEqual(v.classe, "PUMP_RISK")
        self.assertGreaterEqual(len(v.alertes), 3)

    def test_pic_de_volume_au_sommet_est_signale(self):
        alertes = signaux_pump(_base(volume_pic_au_sommet=True))
        self.assertTrue(any("distribution" in a for a in alertes))

    def test_chute_depuis_le_sommet_est_signalee(self):
        alertes = signaux_pump(_base(chute_depuis_sommet_pct=-32.0))
        self.assertTrue(any("chute" in a for a in alertes))

    def test_un_champ_inconnu_n_est_pas_un_signal_absent(self):
        """Ne pas savoir n'est pas savoir que non: le champ muet sort en inconnue."""
        v = qualifier(_base(vwap=None))
        self.assertTrue(any("VWAP" in i for i in v.inconnues))


class TestDilution(unittest.TestCase):
    """P9 Ch.03: S-3 et 424B annoncent une dilution, pas une bonne nouvelle."""

    def test_s3_n_est_pas_un_catalyseur_haussier(self):
        v = qualifier(_base(catalyseur="S-3 depose", formulaire_sec="S-3",
                            short_interest_pct=25.0))
        self.assertNotEqual(v.classe, "SQUEEZE")
        self.assertTrue(any("dilution" in a for a in v.alertes))

    def test_424b_aussi(self):
        self.assertTrue(any("dilution" in a for a in
                            signaux_pump(_base(formulaire_sec="424B5"))))


class TestSqueeze(unittest.TestCase):
    """P12: Short Interest eleve + Low Float + Catalyseur = configuration squeeze."""

    def test_configuration_complete_donne_squeeze_sur_plusieurs_jours(self):
        v = qualifier(_base(short_interest_pct=18.3, float_actions=9_370_000,
                            rvol=6.2, market_cap=310_000_000))
        self.assertEqual(v.classe, "SQUEEZE")
        self.assertIn("jours", v.horizon)

    def test_short_interest_faible_n_est_pas_un_squeeze(self):
        v = qualifier(_base(short_interest_pct=3.0))
        self.assertNotEqual(v.classe, "SQUEEZE")

    def test_pic_de_volume_au_sommet_disqualifie_le_squeeze(self):
        """Volume explosif au sommet = distribution, pas couverture de shorts."""
        v = qualifier(_base(short_interest_pct=20.0, volume_pic_au_sommet=True))
        self.assertNotEqual(v.classe, "SQUEEZE")


class TestContinuationEtFade(unittest.TestCase):

    def test_catalyseur_rvol_fort_au_dessus_vwap_donne_continuation(self):
        v = qualifier(_base(short_interest_pct=4.0, rvol=5.0, prix=10.0, vwap=9.0))
        self.assertEqual(v.classe, "CONTINUATION")
        self.assertIn("jour", v.horizon)

    def test_sous_le_vwap_ce_n_est_pas_une_continuation(self):
        """P7 Ch.01: sous le VWAP, la pression vendeuse domine."""
        v = qualifier(_base(short_interest_pct=4.0, prix=9.0, vwap=10.0))
        self.assertEqual(v.classe, "FADE")

    def test_rvol_faible_sans_catalyseur_donne_fade(self):
        v = qualifier(_base(rvol=1.2, catalyseur=None, formulaire_sec=None,
                            short_interest_pct=2.0))
        self.assertEqual(v.classe, "FADE")
        self.assertIn("premiere heure", v.horizon)

    def test_tenue_a_30min_prolonge_l_horizon(self):
        """La regle des 30 minutes: un gap qui survit bascule en continuation."""
        sans = qualifier(_base(short_interest_pct=4.0, gap_tenu_30min=None))
        avec = qualifier(_base(short_interest_pct=4.0, gap_tenu_30min=True))
        self.assertEqual(sans.horizon, "jour")
        self.assertIn("semaine", avec.horizon)


class TestDonneeAbsente(unittest.TestCase):
    """Mesure du 22.09.2026 (LXEO): yfinance ne rend aucun volume pre-marche.
    Un RVOL absent ne doit pas se lire comme un RVOL nul."""

    def test_rvol_absent_n_est_pas_rvol_nul(self):
        inconnu = qualifier(_base(rvol=None, vwap=None))
        mesure_bas = qualifier(_base(rvol=0.0))
        self.assertNotEqual(inconnu.classe, mesure_bas.classe)
        self.assertTrue(any("RVOL non mesurable" in i for i in inconnu.inconnues))

    def test_catalyseur_sans_volume_mesurable_donne_a_confirmer(self):
        v = qualifier(_base(rvol=None, vwap=None, short_interest_pct=29.1))
        self.assertEqual(v.classe, "A_CONFIRMER")
        self.assertIn("30 min", v.sortie)

    def test_a_confirmer_n_annonce_aucun_horizon_ferme(self):
        v = qualifier(_base(rvol=None, vwap=None))
        self.assertIn("indetermine", v.horizon)

    def test_sans_catalyseur_et_sans_rvol_reste_un_fade(self):
        """A_CONFIRMER exige un catalyseur: sans lui, rien a confirmer."""
        v = qualifier(_base(rvol=None, vwap=None, catalyseur=None, formulaire_sec=None))
        self.assertEqual(v.classe, "FADE")
        self.assertTrue(any("RVOL non mesure" in m for m in v.motifs))

    def test_rvol_absent_ne_donne_jamais_continuation_ni_squeeze(self):
        """Une donnee absente ne vaut pas feu vert (convention du depot)."""
        for si in (5.0, 29.0):
            v = qualifier(_base(rvol=None, vwap=None, short_interest_pct=si))
            self.assertNotIn(v.classe, ("CONTINUATION", "SQUEEZE"))


class TestHorizon(unittest.TestCase):
    """Aucun verdict ne sort sans horizon ni declencheur de sortie."""

    def test_tout_verdict_porte_un_horizon_et_une_sortie(self):
        cas = [_base(),
               _base(volume_moyen=100_000),
               _base(catalyseur=None, formulaire_sec=None, rvol=15.0),
               _base(short_interest_pct=20.0),
               _base(rvol=1.0, catalyseur=None, formulaire_sec=None)]
        for g in cas:
            v = qualifier(g)
            self.assertTrue(v.horizon, f"{v.classe} sans horizon")
            self.assertTrue(v.sortie, f"{v.classe} sans declencheur de sortie")

    def test_aucun_horizon_ne_depasse_la_semaine(self):
        """rapport-template.md: au-dela, ce n'est plus un trade de gap."""
        for g in (_base(), _base(short_interest_pct=20.0), _base(rvol=1.0)):
            self.assertNotIn("mois", qualifier(g).horizon)


if __name__ == "__main__":
    unittest.main(verbosity=2)
