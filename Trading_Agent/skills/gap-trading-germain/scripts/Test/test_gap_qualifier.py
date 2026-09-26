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
                           probabilite_comblement, signaux_pump,
                           rvol_effectif, tendance_etablie,
                           retour_sous_cloture_veille,
                           DIRECTION_PAR_CLASSE, STOP_VENDEUR_PCT, SSR_SEUIL)


def _base(**kw) -> Gap:
    """Gap liquide et sain par defaut; chaque test ne change que ce qu'il teste."""
    defauts = dict(ticker="TEST", gap_pct=10.0, atr_pct=5.0, rvol=4.0, prix=10.0,
                   vwap=9.5, volume_jour=4_000_000, volume_moyen=1_000_000,
                   float_actions=50_000_000,
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
    """Le seuil de 500 K de P7 Ch.01 porte sur le volume DU JOUR.

    Revision du 22.09.2026. Applique a la moyenne, il ecartait MAZE, seul
    gap-and-go complet de la seance; applique au jour, il ecarte toujours STFS,
    qui est bien inexploitable. Les deux cas sont testes nommement.
    """

    def test_stfs_volume_du_jour_insuffisant_ecarte_le_titre(self):
        """STFS, 22.09.2026: +26,2 % au total mais 45 K titres echanges."""
        v = qualifier(_base(ticker="STFS", volume_jour=45_000, volume_moyen=120_000))
        self.assertEqual(v.classe, "INSUFFISANT")
        self.assertEqual(v.horizon, "aucun")
        self.assertTrue(any("volume du jour" in m for m in v.motifs))

    def test_maze_adv_faible_mais_seance_liquide_passe(self):
        """MAZE, 22.09.2026: ADV 290 127, mais 2 726 280 titres le jour du gap."""
        v = qualifier(_base(ticker="MAZE", volume_jour=2_726_280, volume_moyen=290_127))
        self.assertNotEqual(v.classe, "INSUFFISANT")

    def test_plancher_structurel_ecarte_un_titre_habituellement_mort(self):
        """Seance active mais activite ordinaire quasi nulle: sortie incertaine."""
        v = qualifier(_base(volume_jour=900_000, volume_moyen=40_000))
        self.assertEqual(v.classe, "INSUFFISANT")
        self.assertTrue(any("plancher structurel" in m for m in v.motifs))

    def test_volume_du_jour_inconnu_n_est_pas_un_volume_insuffisant(self):
        """Convention du depot: ne pas savoir n'est pas savoir que non."""
        v = qualifier(_base(volume_jour=None, volume_moyen=290_127))
        self.assertNotEqual(v.classe, "INSUFFISANT")
        self.assertTrue(any("volume du jour non mesure" in i for i in v.inconnues))

    def test_volume_suffisant_ne_bloque_pas(self):
        self.assertNotEqual(
            qualifier(_base(volume_jour=600_000, volume_moyen=600_000)).classe,
            "INSUFFISANT")


class TestRvolPreTendance(unittest.TestCase):
    """Le RVOL sur fenetre glissante est gonfle par la tendance qu'il doit voir.

    Cas mesure le 22.09.2026 sur INDP, plus forte hausse de la seance
    (+28,7 % de l'ouverture a la cloture, sans gap): RVOL 1,99 sur les
    20 dernieres seances, 11,34 sur la base [-60:-20].
    """

    def test_le_plus_grand_des_deux_rvol_est_retenu(self):
        self.assertAlmostEqual(rvol_effectif(_base(rvol=1.99, rvol_pre_tendance=11.34)),
                               11.34)

    def test_une_seule_mesure_suffit(self):
        self.assertAlmostEqual(rvol_effectif(_base(rvol=2.5, rvol_pre_tendance=None)), 2.5)
        self.assertAlmostEqual(rvol_effectif(_base(rvol=None, rvol_pre_tendance=7.0)), 7.0)

    def test_aucune_mesure_rend_none(self):
        self.assertIsNone(rvol_effectif(_base(rvol=None, rvol_pre_tendance=None)))

    def test_indp_devient_classable_sur_la_base_pre_tendance(self):
        """Avec le seul RVOL glissant, INDP tombait sous le seuil de continuation."""
        glissant = qualifier(_base(ticker="INDP", rvol=1.99, rvol_pre_tendance=None,
                                   short_interest_pct=4.0))
        corrige = qualifier(_base(ticker="INDP", rvol=1.99, rvol_pre_tendance=11.34,
                                  short_interest_pct=4.0))
        self.assertEqual(glissant.classe, "FADE")
        self.assertEqual(corrige.classe, "CONTINUATION")
        self.assertTrue(any("base pre-tendance" in m for m in corrige.motifs))

    def test_tendance_detectee_par_l_ecart_des_deux_rvol(self):
        self.assertTrue(tendance_etablie(_base(rvol=1.99, rvol_pre_tendance=11.34)))
        # MAZE le meme jour: aucune tendance prealable, les deux mesures coincident.
        self.assertFalse(tendance_etablie(_base(rvol=13.45, rvol_pre_tendance=14.38)))

    def test_tendance_inconnue_tant_que_les_deux_mesures_manquent(self):
        self.assertIsNone(tendance_etablie(_base(rvol_pre_tendance=None)))
        self.assertIsNone(tendance_etablie(_base(rvol=None)))

    def test_la_base_pre_tendance_ne_peut_pas_abaisser_un_verdict(self):
        """Prendre le maximum est sans risque: un verdict ne se degrade jamais."""
        for base_rvol in (None, 0.5, 4.0, 20.0):
            sans = qualifier(_base(rvol=4.0, rvol_pre_tendance=None))
            avec = qualifier(_base(rvol=4.0, rvol_pre_tendance=base_rvol))
            if sans.classe == "CONTINUATION":
                self.assertIn(avec.classe, ("CONTINUATION", "SQUEEZE", "PUMP_RISK"))


class TestPump(unittest.TestCase):
    """P7 Ch.03: les sept signaux d'alarme, et la primaute du risque."""

    def test_volume_explosif_sans_catalyseur_est_ecarte(self):
        """PUMP_RISK ne signifie plus 'ne pas jouer' depuis le 26.09.2026: la
        classe porte une direction vendeuse et son horizon est celui mesure par
        le backtest, trois seances. Elle interdit toujours l'achat."""
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertEqual(v.classe, "PUMP_RISK")
        self.assertEqual(v.direction, "vendeuse")
        # La sortie decrit le rachat d'un short, jamais une entree acheteuse.
        self.assertIn("racheter", v.sortie)
        self.assertIn("vendeuse", v.horizon)

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


class TestSqueezeNeSuffitPas(unittest.TestCase):
    """Cas reel LXEO, 22.09.2026: SI 29,1 %, catalyseur verifie, RVOL 5,0 - et
    pourtant gap comble, cloture sous le VWAP et sous la cloture de la veille.
    Le short interest ne rachete pas une seance de distribution."""

    def test_sous_vwap_interdit_le_squeeze(self):
        v = qualifier(_base(short_interest_pct=29.1, rvol=5.0, prix=4.07, vwap=4.145))
        self.assertNotEqual(v.classe, "SQUEEZE")
        self.assertTrue(any("VWAP" in a for a in v.alertes))

    def test_gap_non_tenu_interdit_le_squeeze(self):
        v = qualifier(_base(short_interest_pct=29.1, rvol=5.0, gap_tenu_30min=False))
        self.assertNotEqual(v.classe, "SQUEEZE")

    def test_squeeze_reste_possible_au_dessus_du_vwap_et_gap_tenu(self):
        """La garde ne doit pas tuer le vrai cas."""
        v = qualifier(_base(short_interest_pct=29.1, rvol=5.0, prix=4.40,
                            vwap=4.20, gap_tenu_30min=True))
        self.assertEqual(v.classe, "SQUEEZE")

    def test_gap_tenu_inconnu_n_interdit_pas_le_squeeze(self):
        """Inconnu n'est pas faux: avant l'ouverture, la tenue n'est pas mesurable."""
        v = qualifier(_base(short_interest_pct=29.1, rvol=5.0, prix=4.40,
                            vwap=4.20, gap_tenu_30min=None))
        self.assertEqual(v.classe, "SQUEEZE")


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


class TestGapEfface(unittest.TestCase):
    """Signal repris de la revue Academy Germain du 24.09.2026 (VBIO).

    Le titre bondit de 2,72 a 4,84 $ (+78 %) en deux minutes, puis repasse sous
    2,72 $ en un quart d'heure: les acheteurs du spike perdent jusqu'a 45 %.
    Sa formulation: "quand un titre qui vient de bondir repasse sous la cloture
    de la veille, tout le gain de la journee a disparu".
    """

    # VBIO reel: cloture veille 2,72, ouverture en hausse, plus haut 4,84,
    # cours final 2,67. Les trois mesures sont coherentes entre elles.
    VBIO = dict(ouverture=2.90, cloture_veille=2.72, prix=2.67)

    def test_cas_vbio(self):
        self.assertTrue(retour_sous_cloture_veille(_base(**self.VBIO)))

    def test_au_dessus_de_la_veille_n_est_pas_un_gap_efface(self):
        self.assertFalse(retour_sous_cloture_veille(
            _base(ouverture=2.90, cloture_veille=2.72, prix=3.10)))

    def test_ouverture_en_baisse_le_signal_ne_s_applique_pas(self):
        """Un titre qui n'a pas ouvert en hausse n'a pas de gain a rendre."""
        self.assertIsNone(retour_sous_cloture_veille(
            _base(ouverture=2.60, cloture_veille=2.72, prix=2.50)))

    def test_la_condition_ne_se_lit_pas_sur_gap_pct(self):
        """Defaut du 25.09.2026: gap_pct porte la variation COURANTE de Finviz,
        pas l'ecart d'ouverture. S'y fier rendait le signal indeclenchable."""
        self.assertIsNone(retour_sous_cloture_veille(
            _base(gap_pct=12.0, ouverture=None, cloture_veille=2.72, prix=2.67)))

    def test_mesures_manquantes_rendent_none(self):
        for manque in ("ouverture", "cloture_veille", "prix"):
            kw = dict(self.VBIO); kw[manque] = None
            self.assertIsNone(retour_sous_cloture_veille(_base(**kw)), manque)

    def test_l_alerte_est_levee(self):
        v = qualifier(_base(**self.VBIO))
        self.assertTrue(any("cloture de la veille" in a for a in v.alertes))

    def test_un_gap_efface_interdit_la_continuation(self):
        v = qualifier(_base(vwap=2.50, rvol=8.0, short_interest_pct=3.0,
                            **self.VBIO))
        self.assertNotEqual(v.classe, "CONTINUATION")

    def test_un_gap_efface_interdit_le_squeeze(self):
        v = qualifier(_base(vwap=2.50, rvol=6.0, short_interest_pct=29.0,
                            gap_tenu_30min=True, **self.VBIO))
        self.assertNotEqual(v.classe, "SQUEEZE")

    def test_cloture_veille_absente_ressort_en_inconnue(self):
        v = qualifier(_base(cloture_veille=None))
        self.assertTrue(any("cloture de la veille" in i for i in v.inconnues))


class TestVwapReel(unittest.TestCase):
    """Revision du 25.09.2026: le VWAP approxime est supprime.

    Il utilisait le prix typique de la derniere barre journaliere. En mode
    premarket cette barre est celle de la VEILLE, donc l'alerte "cours sous le
    VWAP, pression vendeuse" enoncait une affirmation sur le flux du jour a
    partir des donnees de la veille. Mesure sur DCX le 25.09 a 07h19 ET:
    derniere barre journaliere du 24.09, alerte annoncee a -16,2 %.

    Sur deux journees notees, cette alerte portait 3 des 5 erreurs de
    classement: 50 % d'erreur avec, 14 % sans.
    """

    def test_sans_vwap_aucune_alerte_de_pression_vendeuse(self):
        """En pre-marche le VWAP vaut None: l'alerte ne peut plus se declencher."""
        v = qualifier(_base(vwap=None, prix=5.0))
        self.assertFalse(any("VWAP" in a for a in v.alertes))

    def test_sans_vwap_l_ignorance_est_declaree(self):
        v = qualifier(_base(vwap=None))
        self.assertTrue(any("VWAP" in i for i in v.inconnues))

    def test_vwap_mesure_declenche_toujours_l_alerte(self):
        """La garde ne doit pas tuer le vrai signal quand la mesure existe."""
        v = qualifier(_base(prix=8.0, vwap=10.0))
        self.assertTrue(any("VWAP" in a for a in v.alertes))

    def test_sans_vwap_pas_de_continuation(self):
        """CONTINUATION exige le cours au-dessus du VWAP: sans mesure, pas de feu vert."""
        v = qualifier(_base(vwap=None, rvol=8.0, short_interest_pct=3.0))
        self.assertNotEqual(v.classe, "CONTINUATION")

    def test_le_retrait_de_l_alerte_peut_sauver_un_titre(self):
        """Effet recherche: un titre a 3 alertes dont le VWAP repasse sous le seuil."""
        avec = qualifier(_base(market_cap=30_000_000, float_actions=9_000_000,
                               prix=8.0, vwap=10.0))
        sans = qualifier(_base(market_cap=30_000_000, float_actions=9_000_000,
                               prix=8.0, vwap=None))
        self.assertEqual(avec.classe, "PUMP_RISK")
        self.assertNotEqual(sans.classe, "PUMP_RISK")


class TestDirection(unittest.TestCase):
    """Chantier du 26.09.2026: le verdict porte une direction et un stop.

    Les valeurs viennent du backtest sur 226 verdicts. FADE et PUMP_RISK sont
    deux configurations vendeuses de profils opposes: FADE rend +4,65 % avec stop
    contre +4,75 % sans, donc le stop ne change presque rien; PUMP_RISK rend
    +11,83 % avec stop contre -9,57 % sans, donc tout en depend.
    """

    def test_toute_classe_a_une_direction_declaree(self):
        """Une classe sans entree dans la table sortirait sans direction."""
        for classe in ("FADE", "PUMP_RISK", "CONTINUATION", "SQUEEZE",
                       "A_CONFIRMER", "INSUFFISANT"):
            self.assertIn(classe, DIRECTION_PAR_CLASSE, classe)

    def test_fade_est_vendeuse_avec_stop(self):
        v = qualifier(_base(rvol=1.2, catalyseur=None, formulaire_sec=None,
                            short_interest_pct=2.0))
        self.assertEqual(v.classe, "FADE")
        self.assertEqual(v.direction, "vendeuse")
        self.assertEqual(v.stop_pct, STOP_VENDEUR_PCT)

    def test_pump_risk_est_vendeuse_avec_stop(self):
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertEqual(v.classe, "PUMP_RISK")
        self.assertEqual(v.direction, "vendeuse")
        self.assertEqual(v.stop_pct, STOP_VENDEUR_PCT)

    def test_continuation_est_acheteuse_sans_stop_vendeur(self):
        v = qualifier(_base(short_interest_pct=4.0, rvol=5.0, prix=10.0, vwap=9.0))
        self.assertEqual(v.classe, "CONTINUATION")
        self.assertEqual(v.direction, "acheteuse")
        self.assertIsNone(v.stop_pct)

    def test_les_classes_sans_promesse_n_ont_pas_de_direction(self):
        insuffisant = qualifier(_base(volume_jour=45_000))
        self.assertEqual(insuffisant.classe, "INSUFFISANT")
        self.assertIsNone(insuffisant.direction)
        confirmer = qualifier(_base(rvol=None, vwap=None))
        self.assertEqual(confirmer.classe, "A_CONFIRMER")
        self.assertIsNone(confirmer.direction)

    def test_un_verdict_vendeur_dit_qu_il_depend_du_borrow(self):
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertIn("borrow", v.sortie)

    def test_une_sortie_vendeuse_ne_dit_pas_de_ne_pas_entrer(self):
        """Defaut du 26.09.2026 sur PFSA: le libelle d'origine de PUMP_RISK
        disait 'ne pas entrer' tout en portant une direction vendeuse."""
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertNotIn("ne pas entrer", v.sortie)
        self.assertIn("racheter", v.sortie)

    def test_un_horizon_vendeur_n_est_pas_aucun(self):
        v = qualifier(_base(catalyseur=None, formulaire_sec=None, rvol=12.0))
        self.assertNotEqual(v.horizon, "aucun")
        self.assertIn("vendeuse", v.horizon)

    def test_le_borrow_est_toujours_une_inconnue(self):
        """Aucune source publique ne le donne: chaque verdict doit le declarer."""
        for g in (_base(), _base(volume_jour=45_000), _base(rvol=12.0, catalyseur=None,
                                                            formulaire_sec=None)):
            self.assertTrue(any("borrow" in i for i in qualifier(g).inconnues))


class TestRule201(unittest.TestCase):
    """R7 du depot (Trading_Agent/agent/rules.py): la restriction se declenche a
    10 % sous la cloture de la veille. Mesure du 26.09.2026: 65 des 226 candidats
    du backtest cloturent au moins 10 % sous leur ouverture."""

    def test_alerte_levee_sous_le_seuil(self):
        v = qualifier(_base(prix=8.9, cloture_veille=10.0))
        self.assertTrue(any("Rule 201" in a for a in v.alertes))

    def test_pas_d_alerte_au_dessus_du_seuil(self):
        v = qualifier(_base(prix=9.5, cloture_veille=10.0))
        self.assertFalse(any("Rule 201" in a for a in v.alertes))

    def test_seuil_conforme_a_r7(self):
        self.assertAlmostEqual(SSR_SEUIL, 0.90)

    def test_sans_cloture_veille_aucune_alerte(self):
        v = qualifier(_base(prix=8.9, cloture_veille=None))
        self.assertFalse(any("Rule 201" in a for a in v.alertes))


class TestHorizon(unittest.TestCase):
    """Aucun verdict ne sort sans horizon ni declencheur de sortie."""

    def test_tout_verdict_porte_un_horizon_et_une_sortie(self):
        cas = [_base(),
               _base(volume_jour=100_000),
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
