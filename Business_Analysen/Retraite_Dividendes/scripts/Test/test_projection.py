"""
Valeurs de reference calculees a part (plan, 29.09.2026): 200 EUR/mois pendant 10 ans a
6 % l'an (taux mensuel equivalent), versement en fin de mois, sans impot -> 32 494,69;
avec 5 000 de depart -> 41 448,93.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import projection as p   # noqa: E402
import fiscalite as f    # noqa: E402
import hypotheses as h   # noqa: E402


class TestKern(unittest.TestCase):
    def test_zinseszins_ohne_steuer(self):
        w = p.endwert_ohne_steuer(monatlich=200.0, jahre=10, rendite=0.06, start=0.0)
        self.assertAlmostEqual(w, 32494.69, places=1)

    def test_zinseszins_mit_startkapital(self):
        w = p.endwert_ohne_steuer(monatlich=200.0, jahre=10, rendite=0.06, start=5000.0)
        self.assertAlmostEqual(w, 41448.93, places=1)


class TestSimulation(unittest.TestCase):
    MARKT = p.Markt(rendite_div_maison=0.035, wachstum_maison=0.03, rendite_div_etf=0.018,
                    wachstum_etf=0.05, inflation=0.02, basiszins=0.0253)

    def test_steuer_senkt_endwert(self):
        plan = [2400.0] * 20
        s100 = p.simulieren(p.Strategie("maison100", 1.0, False, "dividende"), self.MARKT, plan, 5000, 3500, 20)
        self.assertLess(s100["wert"][-1], p.endwert_ohne_steuer(200.0, 20, 0.065, 5000))

    def test_etf_kapitalisierend_weniger_steuer_als_maison(self):
        plan = [6000.0] * 20
        m = p.simulieren(p.Strategie("maison100", 1.0, False, "dividende"), self.MARKT, plan, 5000, 3500, 20)
        e = p.simulieren(p.Strategie("etf_ref", 0.0, False, "entnahme4"), self.MARKT, plan, 5000, 3500, 20)
        self.assertLess(sum(e["steuern"]), sum(m["steuern"]))

    def test_ziel_jahr_monoton_in_sparquote(self):
        s = p.Strategie("maison70", 0.7, False, "dividende")
        wenig = p.simulieren(s, self.MARKT, [3000.0] * 45, 5000, 3500)["ziel_jahr"]
        viel = p.simulieren(s, self.MARKT, [15000.0] * 45, 5000, 3500)["ziel_jahr"]
        self.assertIsNotNone(viel)
        self.assertTrue(wenig is None or viel < wenig)

    def test_sparplan_minimum(self):
        plan = p.sparplan_aus_quote(0.0, minimum=200.0, jahre=3, netto_monat=[2000, 2000, 2000])
        self.assertEqual(plan, [2400.0, 2400.0, 2400.0])


class TestBuchhaltung(unittest.TestCase):
    """Decisions du controleur (29.09.2026 puis tour de revue 1/5), verifiees une par une:
    Vorabpauschale non gonflee par les versements, impot retire une seule fois par annee
    (poche maison ET poche ETF), erosion reelle de basis_etf par l'inflation, convention
    GKV coherente entre les deux modes de revenu.
    """
    MARKT_DIV = p.Markt(rendite_div_maison=0.035, wachstum_maison=0.0, rendite_div_etf=0.0,
                        wachstum_etf=0.0, inflation=0.0, basiszins=0.0253)

    def test_vorabpauschale_ignore_les_versements(self):
        """Marche plat (croissance et rendement de dividende ETF nuls), strategie ETF pure
        capitalisante, capital de depart 1000, versement annuel 2400, forfait mis a 0 (sans
        cela, le forfait de 1000 EUR absorbe integralement la petite base imposable et le
        test ne discrimine plus rien: avec OU sans la correction, l'impot resultant est nul).
        basiszins = 0,0253.

        Sans la correction (wert_ende = valeur de fin d'annee brute, versements inclus):
        Zuwachs = wert_ende - wert_anfang = 2400 (le versement, pas un gain); base VP =
        min(basisertrag, Zuwachs) = min(1000 x 0,0253 x 0,7 ; 2400) = min(17,71 ; 2400)
        = 17,71. Apres Teilfreistellung (30 %), base imposable = 17,71 x 0,7 = 12,397;
        sans forfait, impot = 12,397 x 25 % x 1,055 (Soli) ~= 3,27 EUR -> impot non nul,
        provoque par un versement.

        Avec la correction (versements de l'annee retires de wert_ende avant l'appel):
        Zuwachs = 0 -> base VP = min(17,71 ; 0) = 0 -> impot de l'annee nul et
        wert = start + versement exactement.
        """
        vp_corrigee = f.vorabpauschale(1000.0, 1000.0, 0.0, 0.0253)
        vp_sans_correction = f.vorabpauschale(1000.0, 3400.0, 0.0, 0.0253)
        self.assertAlmostEqual(vp_corrigee, 0.0, places=6)
        self.assertAlmostEqual(vp_sans_correction, 17.71, places=6)

        strat = p.Strategie("etf_pur", 0.0, False, "dividende")
        markt = p.Markt(rendite_div_maison=0.0, wachstum_maison=0.0, rendite_div_etf=0.0,
                        wachstum_etf=0.0, inflation=0.0, basiszins=0.0253)
        s = p.simulieren(strat, markt, [2400.0] * 3, 1000.0, 3500.0, 3,
                         pauschbetrag_nominal=0.0)
        self.assertAlmostEqual(s["wert"][0], 1000.0 + 2400.0, places=6)
        self.assertAlmostEqual(s["steuern"][0], 0.0, places=6)

    def test_steuer_egale_a_jahres_netto(self):
        """Dividende maison connu (5000 x 3,5 % = 175, sans croissance ni versement),
        strategie 100 % maison distribuante: l'impot retenu par simulieren doit egaler
        exactement brut - fiscalite.jahres_netto sur les memes postes et le meme forfait,
        ET la richesse resultante doit egaler exactement start + dividende - impot (identite
        qui echouerait si l'impot etait retire deux fois de la poche maison)."""
        strat = p.Strategie("maison_pur", 1.0, True, "dividende")
        s = p.simulieren(strat, self.MARKT_DIV, [0.0], 5000.0, 3500.0, 1)
        div_maison = 5000.0 * 0.035
        self.assertAlmostEqual(div_maison, 175.0, places=6)
        posten = [(div_maison * a, k) for k, a in p.LAENDER_MIX.items()] + [(0.0, "ETF")]
        pausch = h.wert("sparerpauschbetrag")
        attendu = sum(b for b, _ in posten) - f.jahres_netto(posten, pausch, sq=f._sq_standard())
        self.assertAlmostEqual(s["steuern"][0], attendu, places=6)
        self.assertAlmostEqual(s["wert"][0], 5000.0 + div_maison - attendu, places=6)

    def test_impot_retire_une_seule_fois_poche_etf(self):
        """Strategie ETF pure capitalisante, croissance reelle 5 %, pas de rendement de
        dividende ETF, pas de versement, forfait a 0 (pas d'absorption): la Vorabpauschale
        est plafonnee par le Basisertrag (10000 x 0,0253 x 0,7 = 177,10 EUR, inferieur au
        Zuwachs reel de 500 EUR). L'identite de richesse wert = valeur brute de fin d'annee
        - impot verifie que l'impot n'est retire qu'une seule fois de la poche ETF (celle
        qui paie quand anteil_maison = 0); une double soustraction romprait cette identite
        alors que steuern[0] resterait, lui, egal a la meme valeur calculee une seule fois."""
        strat = p.Strategie("etf_pur_vp", 0.0, False, "dividende")
        markt = p.Markt(rendite_div_maison=0.0, wachstum_maison=0.0, rendite_div_etf=0.0,
                        wachstum_etf=0.05, inflation=0.0, basiszins=0.0253)
        s = p.simulieren(strat, markt, [0.0], 10000.0, 3500.0, 1, pauschbetrag_nominal=0.0)

        etf_pretax = 10000.0 * (1 + markt.wachstum_etf)
        vp = f.vorabpauschale(10000.0, etf_pretax, 0.0, markt.basiszins)
        self.assertAlmostEqual(vp, 177.1, places=2)
        posten = [(0.0, k) for k in p.LAENDER_MIX] + [(vp, "ETF")]
        attendu = sum(b for b, _ in posten) - f.jahres_netto(posten, 0.0, sq=f._sq_standard())
        self.assertAlmostEqual(s["steuern"][0], attendu, places=6)
        self.assertAlmostEqual(s["wert"][0], etf_pretax - attendu, places=4)

    def test_basis_etf_erosion_inflation_cree_un_gain_taxable(self):
        """Croissance et rendement reels nuls, ETF pur capitalisant, mode entnahme4,
        inflation 3 %: sans deflation de basis_etf, la base de cout resterait strictement
        egale a la valeur ETF (aucune croissance, aucune Vorabpauschale puisque Zuwachs
        nul chaque annee, aucune distribution) et gewinnanteil = 0 pour toujours, quelle
        que soit l'inflation. En deflatant basis_etf d'une annee d'inflation avant chaque
        versement (annee 0 exclue, deja au niveau de prix de l'annee 0), le cout
        d'acquisition nominal perd du pouvoir d'achat reel chaque annee suivante: apres 5
        ans, basis_etf < etf et un retrait de 4 % porte une part imposable non nulle bien
        que la valeur reelle du portefeuille soit exactement egale aux versements cumules
        (aucune croissance reelle).
        """
        strat = p.Strategie("etf_entnahme4", 0.0, False, "entnahme4")
        markt = p.Markt(rendite_div_maison=0.0, wachstum_maison=0.0, rendite_div_etf=0.0,
                        wachstum_etf=0.0, inflation=0.03, basiszins=0.0253)
        jahre = 5
        plan = [2400.0] * jahre
        s = p.simulieren(strat, markt, plan, 5000.0, 3500.0, jahre,
                         pauschbetrag_nominal=0.0, kv_teilfreistellung_etf=True)

        # Reproduction independante de la recursion basis_etf/etf (aucune croissance,
        # aucun rendement, aucune VP: Zuwachs nul chaque annee -> VP = 0 chaque annee).
        basis, etf = 5000.0, 5000.0
        for i in range(jahre):
            if i > 0:
                basis /= 1 + markt.inflation
            basis += 2400.0
            etf += 2400.0
        self.assertLess(basis, etf)   # le cout nominal a perdu du pouvoir d'achat reel
        self.assertAlmostEqual(s["wert"][-1], etf, places=4)   # valeur reelle = versements cumules

        gewinnanteil = 1 - basis / etf
        self.assertGreater(gewinnanteil, 0.0)
        entnahme = 0.04 * etf
        steuerpfl_brutto = entnahme * gewinnanteil
        saetze = f.saetze_2026()
        deflator = (1 + markt.inflation) ** (jahre - 1)
        s_real = p._saetze_real(saetze, deflator)
        netto_part = f.posten_netto(steuerpfl_brutto, "ETF", 0.0, sq=f._sq_standard())[0]
        netto = entnahme - steuerpfl_brutto + netto_part
        steuerpfl_kv = steuerpfl_brutto * (1 - f.TEILFREI)
        attendu = (netto - f.kv_beitrag_jahr(steuerpfl_kv, s_real)) / 12
        self.assertAlmostEqual(s["einkommen_netto_monat_real"][-1], attendu, places=4)

    def test_kv_base_teilfreistellung_etf_seulement(self):
        """Source: GKV-Spitzenverband, 'Katalog von Einnahmen und deren beitragsrechtliche
        Bewertung nach § 240 SGB V', ligne 'Investmenterträge': 'ja, unter Beruecksichtigung
        der §§ 20 und 56 Abs. 6 InvStG' (data/quellen.json, cle 'kv_teilfreistellung_etf').
        La Teilfreistellung InvStG (30 %) doit reduire l'assiette GKV du dividende ETF
        (fonds), jamais celle d'un dividende d'action individuelle. Isole via _einkommen
        directement (poche maison a 0), gros montant ETF pour eviter le plancher/plafond
        GKV (Mindestbemessung/BBG), assiette annuelle comparee, pas seulement le signe."""
        markt = p.Markt(rendite_div_maison=0.0, wachstum_maison=0.0, rendite_div_etf=0.02,
                        wachstum_etf=0.0, inflation=0.0, basiszins=0.0253)
        strat = p.Strategie("etf_gros", 0.0, True, "dividende")
        saetze = f.saetze_2026()
        sq = f._sq_standard()
        etf = 2_000_000.0
        div_etf = etf * 0.02
        posten = [(0.0, k) for k in p.LAENDER_MIX] + [(div_etf, "ETF")]
        netto_impot = f.jahres_netto(posten, 0.0, sq=sq)

        eink_true = p._einkommen(strat, markt, 0.0, etf, etf, 0.0, saetze, sq, True)
        eink_false = p._einkommen(strat, markt, 0.0, etf, etf, 0.0, saetze, sq, False)

        attendu_true = netto_impot - f.kv_beitrag_jahr(div_etf * (1 - f.TEILFREI), saetze)
        attendu_false = netto_impot - f.kv_beitrag_jahr(div_etf, saetze)
        self.assertAlmostEqual(eink_true, attendu_true, places=4)
        self.assertAlmostEqual(eink_false, attendu_false, places=4)
        self.assertGreater(eink_true, eink_false)

    def test_kv_seuils_constants_en_reel_malgre_inflation(self):
        """Correctif tache 10 (revue de code nocturne): le plancher (Mindestbemessung) et le
        plafond (BBG) de la cotisation maladie/dependance volontaire sont revalorises CHAQUE
        ANNEE avec les salaires en droit (§6 Abs. 6-7 et §223 Abs. 4 SGB V pour la BBG; §18
        SGB IV pour la Bezugsgroesse dont depend la Mindestbemessung; sources et citations
        completes dans data/quellen.json, cle 'kv_schwellen_dynamisierung'). Comme le moteur
        travaille en euros de 2026 (reel) et que les salaires croissent au moins autant que
        les prix, la convention retenue est de tenir ces deux seuils CONSTANTS en reel d'une
        annee a l'autre - jamais deflates comme un montant fixe en nominal.

        Avant le correctif, _saetze_real deflatait aussi min_monat/bbg_monat (meme traitement
        que le Sparerpauschbetrag, qui LUI est bien fixe en nominal par la loi, §20 Abs. 9
        EStG): pour un revenu reel identique d'une annee a l'autre, la cotisation KV/PV
        plafonnee baissait avec l'inflation cumulee (le plafond nominal 2026 divise par un
        deflateur croissant), et le revenu net reel simule augmentait a tort avec le temps
        (jusqu'a faire chuter le plafond mensuel simule de ~1226 a ~513 EUR/mois en 2071,
        mesure independamment au moment du correctif). Ce test isole ce mecanisme via
        _einkommen/_saetze_real directement (comme test_kv_base_teilfreistellung_etf_seulement
        ci-dessus), avec un revenu ETF fixe et enorme (tres au-dessus du plafond BBG, quelle
        que soit l'annee) pour que la cotisation soit bien celle du PLAFOND dans les deux cas.
        """
        markt = p.Markt(rendite_div_maison=0.0, wachstum_maison=0.0, rendite_div_etf=0.10,
                        wachstum_etf=0.0, inflation=0.03, basiszins=0.0253)
        strat = p.Strategie("etf_gros", 0.0, True, "dividende")
        saetze = f.saetze_2026()
        sq = f._sq_standard()
        etf = 2_000_000.0                    # div_etf = 200 000 EUR/an, tres au-dessus de la BBG
        basis_etf = etf

        deflator_annee0 = 1.0
        deflator_annee20 = (1 + markt.inflation) ** 20
        s_real_annee0 = p._saetze_real(saetze, deflator_annee0)
        s_real_annee20 = p._saetze_real(saetze, deflator_annee20)

        # Les seuils eux-memes doivent rester inchanges, peu importe le deflateur (droit:
        # revalorises avec les salaires, donc constants en euros de 2026 dans ce modele).
        self.assertEqual(s_real_annee0["min_monat"], saetze["min_monat"])
        self.assertEqual(s_real_annee20["min_monat"], saetze["min_monat"])
        self.assertEqual(s_real_annee0["bbg_monat"], saetze["bbg_monat"])
        self.assertEqual(s_real_annee20["bbg_monat"], saetze["bbg_monat"])

        eink_annee0 = p._einkommen(strat, markt, 0.0, etf, basis_etf, 0.0, s_real_annee0, sq, False)
        eink_annee20 = p._einkommen(strat, markt, 0.0, etf, basis_etf, 0.0, s_real_annee20, sq, False)

        # Meme revenu reel injecte les deux fois -> meme cotisation au plafond -> meme
        # revenu net reel, quelle que soit l'annee simulee (avant le correctif, ce test
        # echouait: eink_annee20 > eink_annee0 de plusieurs milliers d'euros/an).
        self.assertAlmostEqual(eink_annee0, eink_annee20, places=6)

        # Le Sparerpauschbetrag, lui, reste fixe en NOMINAL par la loi (§20 Abs. 9 EStG):
        # comportement INCHANGE par ce correctif, il continue de s'eroder en euros de 2026
        # quand on le deflate (meme mecanique que celle retiree des seuils KV ci-dessus).
        pausch_nominal = 1000.0
        pausch_annee0 = pausch_nominal / deflator_annee0
        pausch_annee20 = pausch_nominal / deflator_annee20
        self.assertLess(pausch_annee20, pausch_annee0)


if __name__ == "__main__":
    unittest.main()
