"""Le resume et les chapitres lisent LE MEME calcul: l'age du tableau de bord est celui
de la projection, et chaque figure annoncee a son CSV."""
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIGUREN = ["fig_ziel_nominal", "fig_kaskade", "fig_steuer_100", "fig_steuer_kumuliert",
           "fig_kapital_rendite", "fig_div_vs_4", "fig_gehalt", "fig_jahre_sparquote",
           "fig_kapital_zeit", "fig_zinseszins", "fig_sparbedarf", "fig_start_verzoegerung",
           "fig_aufteilungen", "fig_anzahl_titel", "fig_sektoren", "fig_laender",
           "fig_rendite_wachstum", "fig_div_beispiele", "fig_shiller_div", "fig_einbrueche",
           "fig_rueckspiel", "fig_mc_faecher", "fig_mc_erfolg", "fig_inflation_2022",
           "fig_bruecke", "fig_rente_alter", "fig_puffer", "fig_zeitplan",
           "fig_bruecke_anticipee", "fig_regle_quatre_histoire"]

# Figures produites par la partie A de la tache 9 (chapitres 2 a 5); les autres sont
# encore des stubs (partie B, second sous-agent) et n'ont pas encore de CSV.
FIGUREN_PARTIE_A = ["fig_ziel_nominal", "fig_kaskade", "fig_steuer_100", "fig_steuer_kumuliert",
                     "fig_kapital_rendite", "fig_div_vs_4", "fig_gehalt", "fig_jahre_sparquote",
                     "fig_kapital_zeit", "fig_zinseszins", "fig_sparbedarf",
                     "fig_start_verzoegerung"]


def setUpModule():
    """Revue finale, constat 2: refs/ est gitignore (2,4 Mo, cite BZSt/GKV-Spitzenverband/
    Shiller). rechnung_retraite.kapitel_7/historique lit refs/ie_data.xls sans repli; sur une
    machine qui n'a pas ce fichier localement, tout r.main() leverait un FileNotFoundError
    opaque au lieu d'un skip explicite. skipTest module-entier plutot qu'un essaim de
    if-manquant repete dans chaque setUpClass."""
    if not os.path.exists(os.path.join(ROOT, "refs", "ie_data.xls")):
        raise unittest.SkipTest("refs/ie_data.xls absent (refs/ est gitignore): "
                                "fournir le classeur Shiller localement pour executer ce module")


class TestRechnung(unittest.TestCase):
    def test_alle_figuren_haben_daten(self):
        for f in FIGUREN:
            self.assertTrue(os.path.exists(os.path.join(ROOT, "data", f + ".csv")), f)

    def test_variante_ziel_niedrig_a_ses_fichiers(self):
        """Tache 11 (variante 2500 EUR/mois, demande explicite de l'utilisateur): le CSV de
        figure (fig_alter_ziel_vergleich) et le corps de tableau (tab_ziel_niedrig) de
        kapitel_5_ziel_niedrig() existent sur le disque apres un r.main()."""
        self.assertTrue(os.path.exists(os.path.join(ROOT, "data", "fig_alter_ziel_vergleich.csv")))
        self.assertTrue(os.path.exists(os.path.join(ROOT, "data", "tab_ziel_niedrig.tex")))

    def test_figures_partie_a_ont_des_donnees(self):
        """Sous-ensemble garanti par la partie A de la tache 9 (voir rapport):
        echoue seul si un bloc du perimetre de cette partie manque, independamment
        de l'etat des stubs de la partie B."""
        for f in FIGUREN_PARTIE_A:
            self.assertTrue(os.path.exists(os.path.join(ROOT, "data", f + ".csv")), f)

    def test_zusammenfassung_konsistent(self):
        z = json.load(open(os.path.join(ROOT, "data", "zusammenfassung.json")))
        makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        for strat, werte in z["alter"].items():
            for quote, alter in werte.items():
                if alter is not None:
                    self.assertIn(str(alter), makros, f"{strat} {quote}")

    def test_keine_ziffer_in_makronamen(self):
        import re
        for zeile in open(os.path.join(ROOT, "data", "kennzahlen.tex")):
            m = re.match(r"\\newcommand\{\\([^}]*)\}", zeile)
            if m:
                self.assertFalse(re.search(r"\d", m.group(1)), m.group(1))


class TestFmtEscape(unittest.TestCase):
    """Format francais du fmt() de rechnung_retraite.py et echappement LaTeX (decisions
    du controleur 2 et 3, tache 9)."""

    def setUp(self):
        import rechnung_retraite as r
        self.r = r

    def test_fmt_milliers_virgule(self):
        self.assertEqual(self.r.fmt(3500), "3\\,500")

    def test_fmt_decimale(self):
        self.assertEqual(self.r.fmt(4.5, 1), "4{,}5")

    def test_fmt_negatif(self):
        self.assertEqual(self.r.fmt(-1234.5, 1), "$-$1\\,234{,}5")

    def test_fmt_petit_nombre_sans_separateur(self):
        self.assertEqual(self.r.fmt(44), "44")

    def test_escapieren_caracteres_speciaux(self):
        brut = "S&P 4% #1 _x {y} ~z ^w"
        echappe = self.r.escapieren(brut)
        for interdit in "&%#_{}~^":
            # chaque caractere interdit n'apparait plus tel quel (seul, sans backslash
            # ni commande de substitution devant)
            self.assertNotIn(interdit, echappe.replace("\\&", "").replace("\\%", "")
                              .replace("\\#", "").replace("\\_", "").replace("\\{", "")
                              .replace("\\}", "").replace(r"\textasciitilde{}", "")
                              .replace(r"\textasciicircum{}", ""))

    def test_escapieren_neutralise_macro_input(self):
        """Un champ scrape contenant une commande LaTeX (input/write) ne doit pas
        pouvoir s'executer: le backslash doit disparaitre en tant que backslash actif."""
        brut = r"\input{/etc/passwd}\write18{rm -rf}"
        echappe = self.r.escapieren(brut)
        self.assertNotIn("\\input", echappe)
        self.assertNotIn("\\write", echappe)
        self.assertIn(r"\textbackslash{}", echappe)

    def test_escapieren_nettoie_espaces_et_dechets(self):
        self.assertEqual(self.r.escapieren("  Allianz SE                    v  "),
                          "Allianz SE                    v")


class TestMacroNamesHelper(unittest.TestCase):
    def setUp(self):
        import rechnung_retraite as r
        self.r = r

    def test_roemisch_annee(self):
        self.assertEqual(self.r.annee_romaine(2044), "XLIV")

    def test_zahlwort_dix_huit(self):
        self.assertEqual(self.r.zahlwort(18), "Achtzehn")


class TestDeterminisme(unittest.TestCase):
    """Deux executions successives du calcul produisent des fichiers identiques
    (decision 8 du controleur): verifie ici sur une figure de la partie A."""

    def test_fig_gehalt_stable(self):
        import rechnung_retraite as r
        r.main()
        a = open(os.path.join(ROOT, "data", "fig_gehalt.csv")).read()
        r.main()
        b = open(os.path.join(ROOT, "data", "fig_gehalt.csv")).read()
        self.assertEqual(a, b)

    def test_tab_hypotheses_sources_stable(self):
        """Revue finale, constat 1: generalise la regression au-dela du seul cas
        pop()/reinsertion de soli_satz. Un troisieme r.main() (deux classes de test
        suffisaient a reproduire le bug d'origine, un run isole n'en appelle qu'un
        ou deux) doit reproduire EXACTEMENT le meme fichier, ordre des lignes
        compris, quel que soit ce qui s'est execute entre les deux appels dans ce
        process (ANNEXE_HYPOTHESES est un dict module-level partage)."""
        import rechnung_retraite as r
        r.main()
        a = open(os.path.join(ROOT, "data", "tab_hypotheses_sources.tex")).read()
        r.main()
        b = open(os.path.join(ROOT, "data", "tab_hypotheses_sources.tex")).read()
        self.assertEqual(a, b)


class TestPartieB(unittest.TestCase):
    """Garde-fous specifiques a la partie B (chapitres 6, 7, 8, 10)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r

    def test_tab_portefeuille_23_lignes_pas_30(self):
        """Decision 1 du controleur: le portefeuille reel compte 23 titres, pas 30;
        data/tab_portefeuille.tex doit avoir une ligne de donnees par titre."""
        import pandas as pd
        n = len(pd.read_csv(os.path.join(ROOT, "data", "portefeuille.csv")))
        self.assertEqual(n, 23)
        corps = open(os.path.join(ROOT, "data", "tab_portefeuille.tex"), encoding="utf-8").read()
        lignes = [l for l in corps.splitlines() if l.strip() and not l.startswith("%")]
        self.assertEqual(len(lignes), n)

    def test_fig_anzahl_titel_va_jusquau_nombre_reel(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_anzahl_titel.csv"))
        self.assertEqual(df["n"].max(), 23)
        self.assertEqual(df["n"].min(), 1)

    def test_fig_mc_erfolg_varie_avec_le_choc(self):
        """Garde-fou contre le defaut de conception decouvert en tache 9 partie B: une
        premiere version forcait le rendement de l'annee 0 calendaire (capital nul a cet
        instant, donc choc inerte) et produisait un taux de reussite identique sur toute
        la grille. La version corrigee choque la croissance du dividende (colonne
        "div_schock", montecarlo.erfolg(div_schock_erstes_rentenjahr=...)) de la
        premiere annee de RETRAITE et doit donc varier avec ce choc."""
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_mc_erfolg.csv"))
        self.assertIn("div_schock", df.columns)
        for col in ("erfolg_30", "erfolg_40", "erfolg_50"):
            self.assertGreater(df[col].nunique(), 1, col)
            # Un choc de dividende plus favorable ne doit jamais reduire la reussite.
            self.assertTrue((df[col].diff().dropna() >= 0).all(), col)

    def test_puffer_meme_calcul_que_erfolgsquote(self):
        """Spec section 8, 'meme macro, jamais deux calculs' (revue de code, tache 9
        fix round 1): \\ErfolgsquoteBasis/\\ErfolgsquoteMitPuffer et fig_puffer.csv
        doivent venir du MEME calcul Monte Carlo, pas de deux recherches independantes
        (qui donnaient auparavant 88,7 % contre 90,0 % pour le meme scenario nominal)."""
        import pandas as pd
        makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_puffer.csv"))
        basis = df.loc[df["puffer_jahre"] == 0, "erfolgsquote"].iloc[0]
        avec_puffer = df.loc[df["puffer_jahre"] == 2, "erfolgsquote"].iloc[0]
        self.assertIn(f"\\newcommand{{\\ErfolgsquoteBasis}}{{{self.r.fmt(basis * 100, 1)}}}", makros)
        self.assertIn(f"\\newcommand{{\\ErfolgsquoteMitPuffer}}{{{self.r.fmt(avec_puffer * 100, 1)}}}",
                      makros)

    def test_macros_hypotheses_declarees(self):
        """Decision 6 du controleur (EtfUmschichtung) et sens de l'axe div_schock
        (tache 9 fix round 1) doivent etre des macros texte, pas seulement des
        commentaires de code."""
        makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        self.assertIn("\\HypotheseUmschichtung", makros)
        self.assertIn("\\HypotheseDivSchock", makros)

    def test_aucun_abgeschnitten_dans_les_figures_mc(self):
        """Decision du brief tache 9: 'jahre sized so abgeschnitten == 0'; verifie ici
        directement sur les trajectoires de base utilisees pour fig_mc_faecher/fig_puffer,
        independamment des assert internes de kapitel_7()/kapitel_8(). Quote du scenario
        de reference (self.r.REFERENCE_QUOTE_NOM, pas "Zwanzig" en dur: correctif tache 10
        du 30.09.2026, meme scenario que celui reellement calcule par kapitel_7())."""
        import histoire
        import montecarlo as mc
        jahres_s = self.r._charger_shiller_komplett()
        markt = self.r.markt_basis()
        plan_ref = self.r._reference_plan()
        jahre_mc, pf = self.r._mc_jahre_sans_troncature(
            jahres_s, plan_ref, self.r.ZIEL_REAL_JAHR, markt.rendite_div_maison,
            [{"rentenjahre": max(self.r.RENTENJAHRE_GRID)}])
        for rj in self.r.RENTENJAHRE_GRID:
            e = mc.erfolg(pf, plan_ref, self.r.ZIEL_REAL_JAHR, rendite_div=markt.rendite_div_maison,
                          rentenjahre=rj)
            self.assertEqual(e["abgeschnitten"], 0, rj)


class TestTabPresse(unittest.TestCase):
    """Controller note 4 (tache 9): data/tab_presse.tex n'avait pas de test dedie
    (revue de code, tache 9 fix round 1)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        cls.r = r

    def _assert_ligne_bien_formee(self, ligne):
        """Verifie chaque CELLULE du tableau separement (le " & " entre cellules est
        le separateur voulu de la ligne, pas un caractere a echapper; le decouper
        d'abord evite de le confondre avec un "&" brut non echappe dans le texte)."""
        contenu = ligne.rstrip("\n")
        self.assertTrue(contenu.endswith("\\\\"), ligne)
        champs = contenu[:-2].rstrip().split(" & ")
        self.assertEqual(len(champs), 4, ligne)
        for champ in champs:
            nettoye = (champ.replace("\\&", "").replace("\\%", "").replace("\\#", "")
                      .replace("\\_", "").replace("\\{", "").replace("\\}", "")
                      .replace(r"\textasciitilde{}", "").replace(r"\textasciicircum{}", "")
                      .replace(r"\textbackslash{}", ""))
            for interdit in "&%#_{}~^\\":
                self.assertNotIn(interdit, nettoye, ligne)

    def test_fichier_existe_une_ligne_par_entree(self):
        presse = json.load(open(os.path.join(ROOT, "data", "presse.json")))
        self.r.tab_presse()
        chemin = os.path.join(ROOT, "data", "tab_presse.tex")
        self.assertTrue(os.path.exists(chemin))
        corps = open(chemin, encoding="utf-8").read()
        lignes = [l for l in corps.splitlines() if l.strip() and not l.startswith("%")]
        self.assertEqual(len(lignes), len(presse))

    def test_entree_reelle_avec_esperluette(self):
        """boerseonline_fire_traum contient 'S&P 500' dans une citation de 213
        caracteres (tronquee a 140 dans _ligne_presse): verifie que le passage REEL
        par le chemin tab_presse() (troncature puis echappement) ne produit pas de
        sortie mal formee."""
        presse = json.load(open(os.path.join(ROOT, "data", "presse.json")))
        entree = next(p for p in presse if p["id"] == "boerseonline_fire_traum")
        self.assertIn("&", entree["zitat"])
        ligne = self.r._ligne_presse(entree)
        self._assert_ligne_bien_formee(ligne)

    def test_troncature_a_cheval_sur_esperluette_et_pourcent(self):
        """Entree synthetique ou '&' (position 135) et '%' (position 136) tombent
        juste avant la frontiere de troncature (caractere 137 du texte BRUT): garantit
        que la troncature opere avant l'echappement, donc qu'aucune sequence "\\&"/"\\%"
        n'est jamais coupee en deux par la troncature."""
        prefixe = "x" * 135
        zitat = prefixe + "&%" + "y" * 50
        entree = {"quelle": "Test & Co", "position": "pro", "thema": "risque % eleve",
                 "zitat": zitat}
        ligne = self.r._ligne_presse(entree)
        self._assert_ligne_bien_formee(ligne)
        self.assertIn(prefixe + "\\&\\%", ligne)
        self.assertNotIn("y", ligne)   # les 50 "y" sont au-dela de la coupe a 137 caracteres


class TestQuellenTex(unittest.TestCase):
    """Tache 10: \\QH{cle} (preamble.tex) est alimentee par data/quellen.tex, genere
    par quellen_tex() a partir de data/quellen.json."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        cls.r = r
        cls.r.quellen_tex()
        cls.quellen = json.load(open(os.path.join(ROOT, "data", "quellen.json")))
        cls.corps = open(os.path.join(ROOT, "data", "quellen.tex"), encoding="utf-8").read()
        cls.annexe = open(os.path.join(ROOT, "data", "quellen_annexe.tex"), encoding="utf-8").read()

    def test_une_macro_par_cle(self):
        for cle in self.quellen:
            self.assertIn(f"\\csname QHfn@{cle}\\endcsname", self.corps, cle)

    def test_annexe_a_un_label_par_cle(self):
        for cle in self.quellen:
            self.assertIn(f"\\label{{src:{cle}}}", self.annexe, cle)

    def test_url_href_courte_dans_la_note(self):
        # Correctif infrastructure (tache 10): la note (footmisc[para]) affiche un
        # lien court (domaine) vers l'URL complete via \href, pas \url{URL complete}
        # (footmisc compose chaque note dans un \hbox non brisable ou aucun point de
        # coupure de \url n'est insere - Overfull \hbox 152 a 461 pt constate sur les
        # notes a URL longue avant ce correctif). L'URL complete reste la cible du lien.
        cle = "abgeltungsteuer_satz"
        url = self.quellen[cle]["url"]
        self.assertIn(f"\\href{{{url}}}{{gesetze-im-internet.de}}", self.corps)
        self.assertNotIn(f"\\url{{{url}}}", self.corps)

    def test_url_complete_dans_l_annexe(self):
        # L'annexe "Sources" n'est pas composee dans le hbox restreint des notes:
        # l'URL complete y reste affichee en clair via \url (coupure normale).
        cle = "abgeltungsteuer_satz"
        url = self.quellen[cle]["url"]
        self.assertIn(f"\\url{{{url}}}", self.annexe)

    def test_seite_reprise_quand_presente(self):
        # basiszins_2026 porte un champ "seite" (reference du BMF-Schreiben, pas un
        # numero de page): doit apparaitre dans la note.
        seite = self.quellen["basiszins_2026"]["seite"]
        self.assertIn(seite, self.corps)

    def test_date_reformatee_en_francais(self):
        # abgeltungsteuer_satz: abgerufen = "2026-09-29" -> "29.09.2026" dans la note.
        self.assertIn("29.09.2026", self.corps)
        self.assertNotIn("2026-09-29", self.corps)

    def test_url_dangereuse_leve_une_erreur(self):
        with self.assertRaises(ValueError):
            self.r._verifier_url("https://x.example/a%b", "cle_test")
        with self.assertRaises(ValueError):
            self.r._verifier_url("https://x.example/a#b", "cle_test")
        with self.assertRaises(ValueError):
            self.r._verifier_url("https://x.example/a&b", "cle_test")

    def test_url_normale_ne_leve_pas(self):
        self.r._verifier_url("https://x.example/a_b~c", "cle_test")  # ne doit pas lever

    def test_cle_hors_motif_leve_une_erreur(self):
        # Revue finale, constat 7: "cle" sert de suffixe a \csname ... \endcsname et
        # a \label{src:<cle>} sans validation. Une cle hors ^[a-z0-9_]+$ (espace,
        # accolade, backslash) casserait la compilation LaTeX plutot que d'etre
        # refusee au moment ou elle est ecrite. json.load() est remplace pour ne
        # jamais toucher le vrai data/quellen.json.
        faux = {"cle invalide}": {"quelle": "x", "url": "https://x.example",
                                  "abgerufen": "2026-01-01", "primaer": True}}
        with mock.patch.object(self.r.json, "load", return_value=faux):
            with self.assertRaises(ValueError):
                self.r.quellen_tex()

    def _ligne_note(self, cle: str) -> str:
        """Ligne de data/quellen.tex qui definit le corps de note de `cle`."""
        marque = f"\\csname QHfn@{cle}\\endcsname"
        lignes = [l for l in self.corps.splitlines() if marque in l]
        self.assertEqual(len(lignes), 1, cle)
        return lignes[0]

    def test_note_sans_champ_hinweis(self):
        # Tache 10 (passage resume/annexes): le champ interne "hinweis" (remarques
        # de collecte en allemand, p. ex. "ATTENTION ecart de revision..." pour
        # inflation_destatis_2019_2025) s'imprimait en bas de page. La note ne
        # garde que titre, page, lien court et date de consultation.
        avec_hinweis = [c for c, q in self.quellen.items() if q.get("hinweis")]
        self.assertTrue(avec_hinweis)  # sinon le test ne verifie rien
        for cle in avec_hinweis:
            ligne = self._ligne_note(cle)
            self.assertNotIn(self.r.escapieren(self.quellen[cle]["hinweis"]), ligne, cle)
            self.assertNotIn("Remarque", ligne, cle)
        self.assertNotIn("ATTENTION", self.corps)

    def test_note_garde_titre_page_lien_date(self):
        cle = "basiszins_2026"
        q = self.quellen[cle]
        ligne = self._ligne_note(cle)
        self.assertIn(self.r.escapieren(q["quelle"]), ligne)
        self.assertIn(self.r.escapieren(q["seite"]), ligne)
        self.assertIn(f"\\href{{{q['url']}}}", ligne)
        self.assertIn(self.r._date_fr(q["abgerufen"]), ligne)

    def test_hinweis_garde_dans_l_annexe(self):
        # La remarque de collecte reste consultable en annexe, jamais en note.
        for cle, q in self.quellen.items():
            if q.get("hinweis"):
                self.assertIn(self.r.escapieren(q["hinweis"]), self.annexe, cle)


class TestPresseNotesTex(unittest.TestCase):
    """Tache 10: \\QP{id} (preamble.tex) est alimentee par data/presse_notes.tex,
    genere par presse_notes_tex() a partir de data/presse.json, deduplique par id
    (plusieurs citations peuvent partager le meme id de source)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        cls.r = r
        cls.r.presse_notes_tex()
        cls.presse = json.load(open(os.path.join(ROOT, "data", "presse.json")))
        cls.ids = sorted({p["id"] for p in cls.presse})
        cls.corps = open(os.path.join(ROOT, "data", "presse_notes.tex"), encoding="utf-8").read()
        cls.annexe = open(os.path.join(ROOT, "data", "presse_annexe.tex"), encoding="utf-8").read()

    def test_une_macro_par_id_unique_pas_par_ligne(self):
        # presse.json a plus de lignes (citations) que d'id (sources); une seule
        # macro par id, pas une par ligne.
        self.assertLess(len(self.ids), len(self.presse))
        for id_ in self.ids:
            self.assertEqual(self.corps.count(f"\\csname QPfn@{id_}\\endcsname"), 1, id_)

    def test_annexe_a_un_label_par_id(self):
        for id_ in self.ids:
            self.assertIn(f"\\label{{presse:{id_}}}", self.annexe, id_)

    def test_ne_reprend_pas_le_zitat(self):
        # Le corps de note ne doit contenir aucune citation individuelle (celles-ci
        # restent dans data/tab_presse.tex, pas dans la note generique par source).
        premiere = next(p for p in self.presse if p["zitat"])
        self.assertNotIn(premiere["zitat"][:40], self.corps)

    def test_id_hors_motif_leve_une_erreur(self):
        # Meme constat 7 que TestQuellenTex.test_cle_hors_motif_leve_une_erreur,
        # pour presse_notes_tex()/data/presse.json (\csname QPfn@<id>,
        # \label{presse:<id>}).
        faux = [{"id": "id invalide}", "quelle": "x", "titel": "t",
                "url": "https://x.example", "abgerufen": "2026-01-01"}]
        with mock.patch.object(self.r.json, "load", return_value=faux):
            with self.assertRaises(ValueError):
                self.r.presse_notes_tex()


class TestChapitresDeuxTrois(unittest.TestCase):
    """Macros ajoutees pour la redaction des chapitres 2 et 3 (tache 10): la cascade se
    referme, la Guenstigerpruefung suit le bareme sur un cas calcule a la main, l'impot
    d'Eglise augmente le brut necessaire."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", "."))

    def test_cascade_se_referme(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_kaskade.csv")).set_index("posten")["betrag"]
        somme = df["brut_necessaire"] + df["retenues_etrangeres"] + df["impot_allemand"] + df["kv_pv"]
        # Les parts pays de data/portefeuille_meta.json sont arrondies a 4 decimales et
        # sommaient a 1,0001 avant le correctif infrastructure (tache 10):
        # _charger_laender_mix() renormalise desormais le mix a somme exactement 1, donc
        # cet ecart doit etre nul (aux arrondis flottants pres, tolerance residuelle).
        self.r._charger_laender_mix()
        ecart_mix = abs(sum(self.r.p.LAENDER_MIX.values()) - 1) * df["brut_necessaire"]
        self.assertAlmostEqual(ecart_mix, 0.0, places=6)
        self.assertAlmostEqual(somme, df["disponible"], delta=ecart_mix + 0.01)
        self.assertAlmostEqual(df["disponible"], self.r.ZIEL_REAL_JAHR, places=0)

    def test_guenstiger_bareme_cas_a_la_main(self):
        # KV 10 % sans plancher ni plafond; brut 30 000, forfait 1 000:
        # zve = 30 000 - 1 000 - 3 000 = 26 000; zone "z" du §32a 2026:
        # t = (26 000 - 17 799)/10 000 = 0,8201; ESt = floor((173,10 t + 2 397) t + 1 034,87)
        # = floor(3 117,07) = 3 117; aucune retenue etrangere (DE), sous la Freigrenze.
        saetze = {"kv": 0.10, "zusatz": 0.0, "pv": 0.0, "min_monat": 0.0, "bbg_monat": 1e9}
        sq = self.r._sq_de_base()
        g = self.r._guenstigerpruefung(30000.0, {"DE": 1.0}, saetze, 1000.0, sq)
        self.assertAlmostEqual(g["zve"], 26000.0, places=6)
        self.assertEqual(g["est"], 3117)
        self.assertEqual(g["steuer_de"], 3117)

    def test_guenstiger_credit_plafonne_a_l_impot(self):
        # Meme cas, 100 % americain: retenue imputable 15 % x 30 000 = 4 500 > ESt 3 117,
        # le credit est plafonne a l'impot allemand, qui tombe a zero (jamais negatif).
        saetze = {"kv": 0.10, "zusatz": 0.0, "pv": 0.0, "min_monat": 0.0, "bbg_monat": 1e9}
        sq = self.r._sq_de_base()
        g = self.r._guenstigerpruefung(30000.0, {"US": 1.0}, saetze, 1000.0, sq)
        self.assertEqual(g["credit"], 3117)
        self.assertEqual(g["steuer_de"], 0.0)

    def test_kirchensteuer_augmente_le_brut(self):
        base = self._wert("BruttoNoetig")
        huit = self._wert("BruttoNoetigKistAcht")
        neuf = self._wert("BruttoNoetigKistNeun")
        self.assertLess(base, huit)
        self.assertLess(huit, neuf)
        self.assertLess(self._wert("NettoDeKistNeun"), self._wert("NettoDeKistAcht"))
        self.assertLess(self._wert("NettoDeKistAcht"), self._wert("NettoDe"))

    def test_steuer_kumuliert_coherent(self):
        self.assertAlmostEqual(self._wert("SteuerKumMaison") - self._wert("SteuerKumEtfThes"),
                               self._wert("SteuerDifferenzAchtzehn"), delta=1.0)

    def test_ziel_nominal_basis_suit_l_annee_objectif(self):
        # Correctif tache 10 (seuils KV constants en reel, cf. projection._saetze_real):
        # le cas de base (70/30, 20 %, "Mitte") n'atteint plus l'objectif dans l'horizon,
        # donc AnneeYannBasis/ZielNominalBasis sont desormais "non atteint" (decision 7 du
        # controleur). Verifie que les deux macros restent coherentes ENSEMBLE dans ce
        # cas plutot que d'affaiblir le test: jamais l'une numerique et l'autre pas.
        annee_texte = self._texte("AnneeYannBasis")
        if annee_texte == "non atteint":
            self.assertEqual(self._texte("ZielNominalBasis"), "non atteint")
            return
        annee = int(annee_texte)
        attendu = self.r.ZIEL_REAL_MONAT * 1.02 ** (annee - 2026)
        self.assertAlmostEqual(self._wert("ZielNominalBasis"), attendu, delta=1.0)


class TestChapitresQuatreCinq(unittest.TestCase):
    """Macros ajoutees pour la redaction des chapitres 4 et 5 (tache 10): elles relisent
    les memes calculs que les figures et que le resume (specification, section 8)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", "."))

    def test_capitaux_chapitre_quatre_suivent_fig_div_vs_4(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_div_vs_4.csv")).set_index("strategie")["kapital"]
        self.assertAlmostEqual(self._wert("KapitalMaisonFuenfzig"), df["MaisonFuenfzig"], delta=1.0)
        self.assertAlmostEqual(self._wert("KapitalMaisonSiebzig"), df["MaisonSiebzig"], delta=1.0)
        self.assertAlmostEqual(self._wert("KapitalMaisonHundert"), df["MaisonHundert"], delta=1.0)
        self.assertAlmostEqual(self._wert("KapitalDifferenzSiebzigEuro"),
                               df["MaisonSiebzig"] - df["EtfReferenz"], delta=1.0)

    def test_variante_prudente_demande_plus_de_capital(self):
        self.assertGreater(self._wert("KapitalEtfPrudent"), self._wert("KapitalEtfReferenz"))
        self.assertLess(self._wert("KapitalDifferenzPrudentProzent"),
                        self._wert("KapitalDifferenzProzent"))

    def test_rendement_d_equilibre_dans_la_grille(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_kapital_rendite.csv"))
        r_eq = self._wert("RenditeEquilibreEtf") / 100
        k_etf = self._wert("KapitalEtfReferenz")
        # Sous le rendement d'equilibre, la strategie tout-maison demande plus que l'ETF.
        for _, ligne in df.iterrows():
            if ligne["rendite"] < r_eq - 0.001:
                self.assertGreater(ligne["maison100"], k_etf)
            if ligne["rendite"] > r_eq + 0.001:
                self.assertLess(ligne["maison100"], k_etf)

    def test_zinseszins_se_referme_sur_le_capital_de_yann(self):
        # Correctif tache 10 (seuils KV constants en reel): SI le scenario de reference
        # n'atteignait pas l'objectif dans l'horizon, KapitalYannBasis (kapitel_8, defini
        # seulement quand l'objectif EST atteint, decision 7 du controleur) deviendrait
        # "non atteint" alors que ZinsVersements/ZinsGains (kapitel_5, fig_zinseszins)
        # retombent sur le dernier index de l'horizon (idx=-1 quand ziel_jahr est None,
        # meme scenario de reference). Dans ce cas, verifie la coherence contre
        # KapitalHorizon<REFERENCE_QUOTE_NOM> (kapitel_5, meme s_ref["wert"][-1] lu par un
        # autre chemin) plutot que d'affaiblir le test en comparant un nombre a une
        # chaine. Depuis le correctif du 30.09.2026 (reference relevee a 30 %), la
        # branche numerique est celle exercee.
        if self._texte("KapitalYannBasis") == "non atteint":
            total = self._wert("ZinsVersements") + self._wert("ZinsGains")
            self.assertAlmostEqual(total, self._wert(f"KapitalHorizon{self.r.REFERENCE_QUOTE_NOM}"),
                                   delta=2.0)
            return
        total = self._wert("ZinsVersements") + self._wert("ZinsGains")
        self.assertAlmostEqual(total, self._wert("KapitalYannBasis"), delta=2.0)

    def test_age_depart_meme_calcul_que_le_resume(self):
        # Chapitre 5 et encadre Yann: l'age de base est l'annee d'objectif moins l'annee
        # de naissance, jamais un second calcul. Compare contre le macro Alter<Strategie>
        # <Quote> du SCENARIO DE REFERENCE (self.r.REFERENCE_QUOTE_NOM, pas "Zwanzig" en
        # dur: correctif tache 10 du 30.09.2026, la reference est passee a 30 %
        # d'epargne). Si un jour la reference cesse d'atteindre l'objectif, les deux
        # macros doivent rester coherentes ENSEMBLE en "non atteint" (jamais l'une
        # numerique et l'autre pas).
        macro_alter_ref = f"Alter{self.r.REFERENCE_STRATEGIE_NOM}{self.r.REFERENCE_QUOTE_NOM}"
        annee_texte = self._texte("AnneeYannBasis")
        if annee_texte == "non atteint":
            self.assertEqual(self._texte("ZielJahrBasis"), "non atteint")
            self.assertEqual(self._texte(macro_alter_ref), "non atteint")
            return
        self.assertEqual(int(annee_texte) - self.r.GEBURTSJAHR,
                         int(self._wert(macro_alter_ref)))
        self.assertEqual(self._texte("ZielJahrBasis"), annee_texte)

    def test_capital_de_depart_plus_eleve_jamais_plus_tard(self):
        self.assertGreater(self._wert("KapitalHorizonDepartHoch"),
                           self._wert("KapitalHorizonDepartNiedrig"))
        self.assertGreater(self._wert("RevenuHorizonDepartHoch"),
                           self._wert("RevenuHorizonDepartNiedrig"))

    def test_epargne_necessaire_decroit_avec_l_age(self):
        self.assertGreater(self._wert("SparbedarfQuarante"), self._wert("SparbedarfQuaranteCinq"))
        self.assertGreater(self._wert("SparbedarfQuaranteCinq"), self._wert("SparbedarfCinquante"))

    def test_controle_salaire_sous_le_seuil(self):
        for nom in ("Bas", "Haut"):
            self.assertLess(abs(self._wert(f"SalaireControleEcart{nom}")),
                            self._wert("SalaireControleSeuil"))

    def test_epargne_mensuelle_suit_le_taux(self):
        net = self._wert("GehaltNetEntreeMois")
        self.assertAlmostEqual(self._wert("EpargneMoisZwanzig"), 0.2 * net, delta=1.0)


class TestScenarioDeReference(unittest.TestCase):
    """Tache 10 (correctif du 30.09.2026): le scenario de reference (auparavant 70/30 a
    20 % d'epargne, qui n'atteint plus jamais l'objectif depuis le correctif des seuils
    KV constants en euros de 2026) est releve a 30 % et n'existe plus qu'a UN SEUL
    endroit du module (REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM/
    REFERENCE_STARTKAPITAL_NOM). Ces tests verifient que les macros *Basis en derivent
    reellement (pas une redefinition independante quelque part d'autre dans le module)
    et que l'ancien resultat a 20 % reste publie tel quel, sans etre supprime."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        cls.zus = json.load(open(os.path.join(ROOT, "data", "zusammenfassung.json")))

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def test_reference_est_bien_70_30_a_trente_pourcent(self):
        # Verrouille la decision du controleur (tache 10, 30.09.2026): si quelqu'un
        # change la definition module-level sans mettre a jour ce test, ce test echoue
        # plutot que de laisser passer silencieusement un autre scenario de reference.
        self.assertEqual(self.r.REFERENCE_STRATEGIE_NOM, "MaisonSiebzig")
        self.assertEqual(self.r.REFERENCE_QUOTE_NOM, "Dreissig")
        self.assertEqual(self.r.REFERENCE_STARTKAPITAL_NOM, "Mitte")
        self.assertAlmostEqual(self.r.QUOTEN[self.r.REFERENCE_QUOTE_NOM], 0.30, places=6)

    def test_macros_referenz_publiees_et_coherentes(self):
        # \ReferenzStrategie/\ReferenzSparquote (redaction ch. 2-5, prochaine passe):
        # doivent refleter EXACTEMENT la definition module-level, pas une valeur retapee.
        self.assertEqual(self._texte("ReferenzStrategie"), "70/30")
        self.assertEqual(self._texte("ReferenzSparquote"),
                         self.r.fmt(self.r.QUOTEN[self.r.REFERENCE_QUOTE_NOM] * 100))

    def test_zieljahr_basis_vient_du_scenario_de_reference(self):
        # Coherence "une seule definition, jamais deux calculs": l'age tire de
        # \ZielJahrBasis (kapitel_10) doit etre EXACTEMENT celui deja calcule dans
        # ZUS["alter"][REFERENCE_STRATEGIE_NOM][REFERENCE_QUOTE_NOM] par kapitel_5(), pas
        # un second calcul independant qui pourrait diverger.
        alter_zus = self.zus["alter"][self.r.REFERENCE_STRATEGIE_NOM][self.r.REFERENCE_QUOTE_NOM]
        zieljahr_texte = self._texte("ZielJahrBasis")
        if alter_zus is None:
            self.assertEqual(zieljahr_texte, "non atteint")
            return
        self.assertEqual(int(zieljahr_texte) - self.r.GEBURTSJAHR, alter_zus)

    def test_ancien_resultat_vingt_pourcent_conserve(self):
        # Decision 3 du controleur: l'ancien resultat a 20 % (avant ce correctif) reste
        # publie tel quel sous ses propres macros, sans etre supprime ni recalcule vers
        # la nouvelle reference.
        self.assertIn("\\AlterMaisonSiebzigZwanzig", self.makros)
        self.assertIn("\\EcartAgeEtfMaisonVingt", self.makros)
        alter_vingt = self.zus["alter"]["MaisonSiebzig"]["Zwanzig"]
        texte_vingt = self._texte("AlterMaisonSiebzigZwanzig")
        if alter_vingt is None:
            self.assertEqual(texte_vingt, "non atteint")
        else:
            self.assertEqual(int(texte_vingt), alter_vingt)


class TestMacrosRevisionChapitres(unittest.TestCase):
    """Tache 10, revision des chapitres 02-05 (30.09.2026): macros qui suivent le
    scenario de reference (REFERENCE_*) pour que la prose ne cite plus une quote en dur
    (\\AlterMaisonSiebzigDreissig, \\EpargneMoisDreissig...), et macros du depart
    anticipe (taux minimal de fig_jahre_sparquote donnant un depart avant l'age legal).
    Chaque macro est relue contre un calcul deja publie, jamais contre un chiffre fige."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        cls.zus = json.load(open(os.path.join(ROOT, "data", "zusammenfassung.json")))

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", ".").replace("$-$", "-"))

    def test_alter_yann_basis_suit_l_annee_d_objectif(self):
        annee = self._texte("AnneeYannBasis")
        if annee == "non atteint":
            self.assertEqual(self._texte("AlterYannBasis"), "non atteint")
        else:
            self.assertEqual(int(self._texte("AlterYannBasis")), int(annee) - self.r.GEBURTSJAHR)

    def test_ecart_etf_basis_au_taux_de_reference(self):
        q = self.r.REFERENCE_QUOTE_NOM
        a_ref = self.zus["alter"][self.r.REFERENCE_STRATEGIE_NOM][q]
        a_etf = self.zus["alter"]["EtfReferenz"][q]
        self.assertEqual(int(self._texte("AlterEtfBasis")), a_etf)
        self.assertEqual(int(self._texte("EcartAgeEtfMaisonBasis")), a_ref - a_etf)

    def test_epargne_mois_basis_est_celle_du_taux_de_reference(self):
        self.assertEqual(self._texte("EpargneMoisBasis"),
                         self._texte("EpargneMois" + self.r.REFERENCE_QUOTE_NOM))

    def test_quote_anticipee_premier_taux_avant_age_legal(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_jahre_sparquote.csv"))
        age_legal = int(self._texte("AgeLegal"))
        for col, nom in (("maison70", "MaisonSiebzig"), ("etf_ref", "EtfReferenz")):
            ages = [(q, self.r.START_JAHR + n - 1 - self.r.GEBURTSJAHR)
                    for q, n in zip(df["quote"], df[col]) if n == n]
            avant = [(q, a) for q, a in ages if a < age_legal]
            self.assertTrue(avant, nom)
            self.assertAlmostEqual(self._wert("QuoteAnticipee" + nom), avant[0][0] * 100, places=6)
            self.assertEqual(int(self._texte("AlterAnticipee" + nom)), avant[0][1])

    def test_ecart_cible_valeur_absolue_et_sens(self):
        brut = self._wert("KapitalYannSousCibleProzent")
        self.assertAlmostEqual(self._wert("KapitalYannEcartCibleProzent"), abs(brut), places=6)
        self.assertEqual(self._texte("KapitalYannSensCible"),
                         "au-dessus du" if brut < 0 else "au-dessous du")

    def test_revenu_horizon_vingt_sous_la_cible_si_non_atteint(self):
        if self._texte("AlterMaisonSiebzigZwanzig") == "non atteint":
            self.assertLess(self._wert("RevenuHorizonZwanzig"), self.r.ZIEL_REAL_MONAT)
            self.assertLess(self._wert("RevenuHorizonZwanzigPartCible"), 100)


class TestChapitreSix(unittest.TestCase):
    """Tache 10, chapitre 6: macros de redaction lues dans le script de selection, dans
    portefeuille_meta.json et dans les CSV du chapitre, jamais contre un chiffre fige."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        cls.meta = json.load(open(os.path.join(ROOT, "data", "portefeuille_meta.json")))

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", ".").replace("$-$", "-"))

    def test_regles_lues_dans_le_script_de_selection(self):
        import inspect
        import portefeuille_exemple as pe
        sig = inspect.signature(pe.auswahl).parameters
        self.assertEqual(self._wert("RegleCroissanceMin"), sig["cagr_min"].default * 100)
        self.assertEqual(self._wert("ReglePayoutMax"), sig["payout_max"].default * 100)
        self.assertEqual(self._wert("RegleSecteurMax"), sig["sektor_max"].default)
        regles = self.r._regles_selection()
        self.assertLess(regles["cagr_relache"], regles["cagr_min"])
        self.assertGreater(regles["payout_relache"], regles["payout_max"])
        self.assertLess(regles["rendement_min"], regles["rendement_max"])

    def test_nombre_manquant_et_paliers(self):
        vise, n = self._wert("NombreTitresVise"), self._wert("NombreTitres")
        self.assertEqual(self._wert("NombreTitresManquants"), vise - n)
        self.assertEqual([self._wert(k) for k in ("PalierStrictTitres", "PalierCroissanceTitres",
                                                  "PalierPayoutTitres")],
                         [e[1] for e in self.meta["paliers_essayes"]])

    def test_entonnoir_concorde_avec_meta(self):
        """L'entonnoir reconstitue hors ligne retrouve les compteurs de la selection."""
        self.assertEqual(self._wert("TitresEcartesFin"),
                         self.meta["mesures_n"] - self.meta["choisis_n"])
        self.assertLessEqual(self._wert("TitresEcartesCroissanceOk")
                             + self._wert("TitresEcartesSansRecul"), self._wert("TitresEcartesFin"))
        self.assertEqual(self._wert("UniversAristocrates") + self._wert("UniversDax")
                         + self._wert("UniversEurostoxx"), self._wert("UniversTitres"))

    def test_aufteilungen_soixante_dix_memes_macros_que_ch5(self):
        """Le 70/30 a 20 % de fig_aufteilungen est le meme calcul que le ch. 5."""
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_aufteilungen.csv")).set_index("strategie")
        self.assertEqual(self.r.fmt(df.loc["MaisonSiebzig", "kapital_xlv"]),
                         self._texte("KapitalHorizonZwanzig"))
        self.assertEqual(self.r.fmt(df.loc["MaisonSiebzig", "netto_monat_xlv"]),
                         self._texte("RevenuHorizonZwanzig"))
        self.assertEqual(self.r.fmt(df.loc["MaisonHundert", "netto_monat_xlv"]),
                         self._texte("RevenuAufteilungMaisonHundert"))

    def test_neuf_dixiemes_de_la_baisse_du_risque(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_anzahl_titel.csv"))
        r1, rn = df["risiko"].iloc[0], df["risiko"].iloc[-1]
        n9 = int(df.loc[df["risiko"] <= rn + 0.1 * (r1 - rn), "n"].iloc[0])
        self.assertEqual(self._wert("TitresNeufDixiemes"), n9)
        self.assertLess(self._wert("RisqueTousTitres"), self._wert("RisqueUnTitre"))

    def test_bornes_plus_longue_hausse_cas_a_la_main(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0], index=range(2000, 2007))
        self.assertEqual(self.r._bornes_plus_longue_hausse(s), (2003, 2006, 3))

    def test_nom_court_coupe_a_la_virgule(self):
        self.assertEqual(self.r._nom_court("McCormick & Company, Incorporat"), "McCormick & Company")
        self.assertEqual(self.r._nom_court("Allianz SE"), "Allianz SE")

    def test_tableau_net_avec_formulaire_w8ben(self):
        """Le net du tableau suit le ch. 3 (W-8BEN suppose): un titre americain garde
        \\NettoUs pour cent de son rendement brut."""
        import pandas as pd
        pf = pd.read_csv(os.path.join(ROOT, "data", "portefeuille.csv"))
        corps = open(os.path.join(ROOT, "data", "tab_portefeuille.tex"), encoding="utf-8").read()
        ligne_us = pf[pf["land"] == "US"].iloc[0]
        attendu = self.r.fmt(ligne_us["rendite_ttm"] * self._wert("NettoUs"), 1)
        ligne = [l for l in corps.splitlines() if l.startswith(ligne_us["ticker"] + " ")][0]
        self.assertEqual(ligne.split("&")[4].strip(), attendu + "\\%")


class TestChapitreSept(unittest.TestCase):
    """Tache 10, chapitre 7: macros de redaction relues contre les CSV du chapitre, la
    serie Destatis de quellen.json ou d'autres macros, jamais contre un chiffre fige."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", ".").replace("$-$", "-"))

    def test_seuil_de_revenu_lu_dans_les_deux_modeles(self):
        """Le seuil publie est celui du code des deux modeles (rejeu et Monte Carlo)."""
        seuil = self.r._seuil_revenu_modeles()
        self.assertGreater(seuil, 0.0)
        self.assertLess(seuil, 1.0)
        self.assertEqual(self._wert("SeuilReussite"), round(seuil * 100))

    def test_rejeu_compte_coherent_avec_le_csv(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_rueckspiel.csv"))
        tenus, echecs = int(df["ans_tenu"].notna().sum()), int(df["ans_echec"].notna().sum())
        self.assertEqual(self._wert("RueckspielTenus"), tenus)
        self.assertEqual(self._wert("RueckspielEchecs"), echecs)
        self.assertEqual(self._wert("RueckspielJuges"), tenus + echecs)
        self.assertEqual(self._wert("RueckspielAtteints"), int(df["jahre_bis_ziel"].notna().sum()))
        self.assertEqual(self._wert("RueckspielDebuts"), len(df))
        self.assertAlmostEqual(self._wert("RueckspielTenusProzent"), tenus / (tenus + echecs) * 100,
                               places=1)
        # Chaque annee atteinte tombe dans exactement une des trois classes du nuage.
        classes = df[["ans_tenu", "ans_echec", "ans_non_juge"]].notna().sum(axis=1)
        self.assertTrue((classes[df["jahre_bis_ziel"].notna()] == 1).all())
        self.assertTrue((classes[df["jahre_bis_ziel"].isna()] == 0).all())

    def test_echecs_du_rejeu_dans_les_episodes_de_baisse(self):
        import pandas as pd
        episodes = set(pd.read_csv(os.path.join(ROOT, "data", "fig_einbrueche.csv"))["episode"])
        texte = self._texte("RueckspielEchecsEpisodes").replace(" et ", ", ")
        noms = [e.replace("--", "-") for e in texte.split(", ")]
        self.assertEqual(len(noms), self._wert("RueckspielEchecsNombreEpisodes"))
        for nom in noms:
            self.assertIn(nom, episodes)

    def test_annees_sous_le_sommet(self):
        self.assertEqual(self._wert("GroessterEinbruchSousSommetAns"),
                         self._wert("GroessterEinbruchRetour") - self._wert("GroessterEinbruchDebut"))
        self.assertEqual(self._wert("BaisseLongueAns"),
                         self._wert("BaisseLongueRetour") - self._wert("BaisseLongueDebut"))
        self.assertGreaterEqual(self._wert("BaisseLongueAns"), self._wert("GroessterEinbruchSousSommetAns"))

    def test_seuil_monte_carlo_sans_impot_et_capital_du_chapitre_quatre(self):
        """Le seuil du Monte Carlo est l'objectif brut divise par le rendement maison
        (aucun impot), et la ligne du capital requis est celle du chapitre 4."""
        import pandas as pd
        markt = self.r.markt_basis()
        seuil = self.r.ZIEL_REAL_JAHR / markt.rendite_div_maison
        self.assertEqual(self._texte("McKapitalSeuil"), self.r.fmt(seuil))
        self.assertEqual(self._texte("KapitalRequisReference"), self._texte("KapitalMaisonSiebzig"))
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_mc_faecher.csv"))
        self.assertAlmostEqual(df["seuil_mc"].iloc[0], seuil, places=2)
        self.assertEqual(self.r.fmt(df["capital_requis"].iloc[0]), self._texte("KapitalMaisonSiebzig"))
        self.assertLess(seuil, df["capital_requis"].iloc[0])

    def test_reussite_dans_l_horizon(self):
        """Exiger l'objectif dans l'horizon ne peut qu'abaisser la reussite, et l'ecart
        publie est la difference des deux macros."""
        base, horizon = self._wert("ErfolgsquoteBasis"), self._wert("McErfolgDansHorizon")
        self.assertLessEqual(horizon, base)
        self.assertAlmostEqual(self._wert("McErfolgEcartHorizon"), base - horizon, delta=0.11)

    def test_choc_et_reserve_bornes(self):
        self.assertLess(self._wert("McErfolgChocMin"), self._wert("McErfolgChocMoinsVingt"))
        self.assertLess(self._wert("McErfolgChocMoinsVingt"), self._wert("McErfolgChocMoinsDix"))
        self.assertLess(self._wert("McErfolgChocMoinsDix"), self._wert("ErfolgsquoteBasis"))
        self.assertLess(self._wert("ErfolgsquoteBasis"), self._wert("McErfolgChocMax"))
        self.assertLess(self._wert("ErfolgsquoteBasis"), self._wert("ErfolgsquoteMitPuffer"))
        self.assertEqual(self._wert("PufferMontantMit"),
                         self._wert("PufferJahreMit") * self._wert("ZielJahr"))

    def test_ages_d_objectif_monte_carlo(self):
        for nom in ("Bas", "Median", "Haut"):
            self.assertEqual(self._wert(f"McAgeObjectif{nom}"),
                             self._wert(f"McAnneeObjectif{nom}") - self.r.GEBURTSJAHR)
        self.assertLessEqual(self._wert("McAgeObjectifBas"), self._wert("McAgeObjectifMedian"))
        self.assertLessEqual(self._wert("McAgeObjectifMedian"), self._wert("McAgeObjectifHaut"))

    def test_inflation_cumulee_contre_la_serie_destatis(self):
        """Hausse des prix recalculee directement sur la serie sourcee (quellen.json),
        independamment de fig_inflation_2022."""
        q = json.load(open(os.path.join(ROOT, "data", "quellen.json"), encoding="utf-8"))
        serie = q["inflation_destatis_2019_2025"]["wert"]
        annees = sorted(int(a) for a in serie)
        cum = 1.0
        for a in annees[1:]:
            cum *= 1 + serie[str(a)]
        self.assertAlmostEqual(self._wert("PrixHausseCumulee"), (cum - 1) * 100, places=1)
        self.assertAlmostEqual(self._wert("RenteNonIndexeeFin"), 3500 / cum, delta=0.5)
        self.assertEqual(self._wert("InflationPicRevisee"), round(max(serie.values()) * 100, 1))

    def test_yann_brut_apres_pire_baisse(self):
        brut = self._wert("BruttoNoetigMois")
        attendu = brut * (1 - self._wert("GroessterEinbruch") / 100)
        # Tolerance = arrondis des deux macros (euro entier, pourcentage a 0,1 point).
        self.assertAlmostEqual(self._wert("YannBrutApresPireBaisse"), attendu, delta=brut * 0.0005 + 1.0)

    def test_hypothese_div_schock_accentuee(self):
        texte = self._texte("HypotheseDivSchock")
        self.assertNotIn("div\\_schock", texte)
        self.assertIn("premi\\`ere ann\\'ee de retraite", texte)

    def test_rapport_de_croissance(self):
        """Rapport entre la croissance du modele et celle de l'indice, relu contre les deux
        macros publiees (tolerance = arrondis a 0,1 point)."""
        attendu = self._wert("CroissanceModeleMaison") / self._wert("ShillerDivCroissanceFenetre")
        self.assertAlmostEqual(self._wert("CroissanceModeleRapport"), attendu, delta=0.15)
        self.assertGreater(self._wert("CroissanceModeleRapport"), 1.0)

    def test_liste_francaise(self):
        self.assertEqual(self.r._liste_fr(["a"]), "a")
        self.assertEqual(self.r._liste_fr(["a", "b", "c"]), "a, b et c")


def _serie_constante(rendement, premiere=1900, annees=61, premiere_ligne=0.0):
    """Serie annuelle synthetique au format de histoire.jahresreihe: rendement reel constant,
    la premiere ligne portant l'artefact de bord (rendement non defini, rempli par 0.0 dans
    jahresreihe) qu'on peut rendre absurde pour verifier qu'elle n'est jamais lue."""
    import pandas as pd
    jahre = list(range(premiere, premiere + annees))
    r = [premiere_ligne] + [rendement] * (annees - 1)
    return pd.DataFrame({"jahr": jahre, "rendite_real": r, "div_real": 1.0,
                         "div_wachstum_real": 0.0, "inflation": 0.0})


class TestRetraitHistorique(unittest.TestCase):
    """Tache 10, chapitre 8: retrait constant en termes reels (regle des 4 %) rejoue sur
    une serie annuelle. Cas ecrits a la main sur des rendements constants."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        cls.r = r

    def test_rendement_nul_epuise_en_vingt_cinq_ans(self):
        """Sans rendement, 4 % du capital initial par an vident le capital en 25 retraits."""
        df = self.r._retrait_constant_historique(_serie_constante(0.0), 0.04, 30)
        # 60 annees de rendement (la premiere ligne est ecartee), fenetres de 30 ans: 31 debuts.
        self.assertEqual(len(df), 31)
        self.assertEqual(int(df["start"].min()), 1901)
        self.assertEqual(int(df["start"].max()), 1931)
        self.assertFalse(df["tenu"].any())
        self.assertTrue((df["annees_tenues"] == 25).all())
        self.assertTrue((df["capital_fin"] == 0.0).all())

    def test_rendement_nul_duree_egale_tient_juste(self):
        """25 ans a 4 % sans rendement: les 25 retraits passent (tolerance flottante), rien ne reste."""
        df = self.r._retrait_constant_historique(_serie_constante(0.0), 0.04, 25)
        self.assertTrue(df["tenu"].all())
        self.assertTrue((df["annees_tenues"] == 25).all())
        for v in df["capital_fin"]:
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_rendement_constant_formule_fermee(self):
        """Retrait en debut d'annee puis rendement r: W_n = (1+r)^n - w (1+r) ((1+r)^n - 1) / r."""
        r_, w, n = 0.05, 0.04, 30
        attendu = (1 + r_) ** n - w * (1 + r_) * ((1 + r_) ** n - 1) / r_
        df = self.r._retrait_constant_historique(_serie_constante(r_), w, n)
        self.assertTrue(df["tenu"].all())
        for v in df["capital_fin"]:
            self.assertAlmostEqual(v, attendu, places=9)

    def test_premiere_ligne_jamais_lue(self):
        """La premiere ligne de jahresreihe n'a pas de rendement (artefact de bord): meme une
        valeur absurde n'y change rien."""
        a = self.r._retrait_constant_historique(_serie_constante(0.05), 0.04, 30)
        b = self.r._retrait_constant_historique(_serie_constante(0.05, premiere_ligne=5.0), 0.04, 30)
        self.assertEqual(list(a["start"]), list(b["start"]))
        for x, y in zip(a["capital_fin"], b["capital_fin"]):
            self.assertAlmostEqual(x, y, places=12)

    def test_krach_fait_echouer_les_seuls_departs_touches(self):
        """Un krach de -100 % en 1930 vide tout portefeuille en cours: echouent les departs qui
        doivent encore retirer apres 1930; celui dont 1930 est la derniere annee a fait tous ses
        retraits (il tient, capital final nul); les autres tiennent."""
        serie = _serie_constante(0.05)
        serie.loc[serie["jahr"] == 1930, "rendite_real"] = -1.0
        df = self.r._retrait_constant_historique(serie, 0.04, 30)
        for _, l in df.iterrows():
            start = int(l["start"])
            touche = start <= 1930 <= start + 28
            self.assertEqual(bool(l["tenu"]), not touche, start)
            if touche:
                # Retraits faits jusqu'a 1930 compris, puis plus rien.
                self.assertEqual(int(l["annees_tenues"]), 1930 - start + 1)
            if start + 29 == 1930:
                self.assertEqual(l["capital_fin"], 0.0)

    def test_resume_part_et_pire_depart(self):
        import pandas as pd
        df = pd.DataFrame({"start": [1901, 1902, 1903], "tenu": [True, False, True],
                           "annees_tenues": [30, 12, 30], "capital_fin": [1.2, 0.0, 0.8]})
        res = self.r._resume_retrait_historique(df)
        self.assertEqual(res["debuts"], 3)
        self.assertEqual(res["tenus"], 2)
        self.assertAlmostEqual(res["part"], 2 / 3)
        self.assertEqual(res["pire_start"], 1902)
        self.assertEqual(res["pire_annees"], 12)
        self.assertEqual(res["echecs"], [1902])

    def test_resume_sans_echec_pire_depart_au_plus_petit_capital(self):
        import pandas as pd
        df = pd.DataFrame({"start": [1901, 1902], "tenu": [True, True],
                           "annees_tenues": [30, 30], "capital_fin": [1.2, 0.8]})
        res = self.r._resume_retrait_historique(df)
        self.assertEqual(res["part"], 1.0)
        self.assertEqual(res["pire_start"], 1902)
        self.assertEqual(res["echecs"], [])


class TestRenteLegale(unittest.TestCase):
    """Tache 10, chapitre 8: points de rente plafonnes au plafond de cotisation de
    l'assurance pension (un salaire au-dessus du plafond ne cotise pas au-dela)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        cls.r = r

    def test_points_plafonnes(self):
        de, rw, bbg = 50000.0, 40.0, 75000.0
        traj = [{"brutto": 2 * de}] * 10
        # Arret a 26 + 10 ans: 10 annees cotisees, chacune plafonnee a 1,5 point.
        self.assertAlmostEqual(self.r._rente_mensuelle(36, traj, de, rw, bbg), 10 * 1.5 * rw)

    def test_salaire_sous_le_plafond_inchange(self):
        de, rw, bbg = 50000.0, 40.0, 75000.0
        traj = [{"brutto": de}] * 10
        self.assertAlmostEqual(self.r._rente_mensuelle(31, traj, de, rw, bbg), 5 * rw)
        self.assertEqual(self.r._rente_mensuelle(26, traj, de, rw, bbg), 0.0)


class TestChapitreHuit(unittest.TestCase):
    """Tache 10, chapitre 8: macros relues contre les CSV du chapitre et contre les macros
    des chapitres 5 et 7 (meme calcul, jamais un second)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", ".").replace("$-$", "-"))

    def test_regle_historique_macros_et_csv(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_regle_quatre_histoire.csv"))
        for nom, duree in self.r.RETRAIT_HIST_DUREES.items():
            col = df[f"fin_{nom.lower()}"].dropna()
            ans = df[f"ans_{nom.lower()}"].dropna()
            self.assertEqual(self._wert(f"RetraitHistDebuts{nom}"), len(col))
            tenus = int((col > 0).sum())
            self.assertEqual(self._wert(f"RetraitHistTenus{nom}"), tenus)
            self.assertAlmostEqual(self._wert(f"RetraitHistReussite{nom}"), tenus / len(col) * 100,
                                   delta=0.05)
            self.assertLessEqual(ans.max(), duree)
            pire = int(self._wert(f"RetraitHistPireDebut{nom}"))
            self.assertEqual(ans[df["start"] == pire].iloc[0], ans.min())
            self.assertEqual(self._wert(f"RetraitHistAns{nom}"), duree)

    def test_memes_departs_que_le_rejeu(self):
        """Comparaison a armes egales: memes annees de depart en retraite que le rejeu du ch. 7."""
        self.assertEqual(self._wert("RetraitHistMemesDepartsJuges"), self._wert("RueckspielJuges"))
        cases = [self._wert(n) for n in ("RetraitHistLesDeuxTiennent", "RetraitHistSeulRetraitTient",
                                         "RetraitHistSeulDividendeTient", "RetraitHistAucunNeTient")]
        self.assertEqual(sum(cases), self._wert("RueckspielJuges"))
        self.assertEqual(cases[0] + cases[2], self._wert("RueckspielTenus"))
        self.assertEqual(cases[0] + cases[1], self._wert("RetraitHistMemesDepartsTenus"))
        self.assertLessEqual(self._wert("RetraitHistMemesDepartsAnnees"), self._wert("RueckspielJuges"))
        self.assertEqual(cases[2] + cases[3], self._wert("RetraitHistMemesDepartsEchecs"))
        self.assertIn(self._texte("RetraitHistPireDebutVierzig"), self._texte("RetraitHistEchecsListeVierzig"))

    def test_rente_legale_plafonnee_et_coherente(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_rente_alter.csv"))
        self.assertEqual(int(df["ausstiegsalter"].max()), int(self._wert("AgeLegal")))
        self.assertEqual(self.r.fmt(df["rente_monat"].iloc[-1]), self._texte("RenteBeiZielalter"))
        # Aucune annee ne peut valoir plus que le plafond en points.
        pts_max = self._wert("PointsPlafond")
        self.assertLess(pts_max, 2.5)
        self.assertGreater(self._wert("PointsEntree"), 0.0)
        self.assertLess(self._wert("PointsEntree"), pts_max)

    def test_pont_anticipe(self):
        import pandas as pd
        age_legal = int(self._wert("AgeLegal"))
        age = int(self._wert("AlterMaisonSiebzigFuenfzig"))
        self.assertEqual(self._wert("BrueckeJahreFuenfzig"), age_legal - age)
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_bruecke_anticipee.csv"))
        self.assertEqual(int(df["alter"].min()), age)
        self.assertTrue((df.loc[df["alter"] < age_legal, "rente"] == 0).all())
        apres = df.loc[df["alter"] >= age_legal, "rente"]
        self.assertTrue((apres == apres.iloc[0]).all())
        self.assertEqual(self.r.fmt(apres.iloc[0]), self._texte("RenteBrueckeFuenfzig"))
        self.assertLess(self._wert("RenteBrueckeFuenfzig"), self._wert("RenteBeiZielalter"))
        self.assertGreaterEqual(self._wert("RevenuNetBrueckeFuenfzig"), self._wert("ZielMonatNombre"))
        self.assertEqual(self._wert("BrueckeJahreAnticipee"),
                         age_legal - self._wert("AlterAnticipeeMaisonSiebzig"))

    def test_reserve_lue_dans_fig_puffer(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_puffer.csv")).set_index("puffer_jahre")
        self.assertEqual(self.r.fmt(df.loc[0, "erfolgsquote"] * 100, 1), self._texte("ErfolgsquoteBasis"))
        self.assertEqual(self.r.fmt(df.loc[1, "erfolgsquote"] * 100, 1), self._texte("ErfolgsquotePufferEins"))
        self.assertEqual(self.r.fmt(df.index.max()), self._texte("PufferJahreMax"))
        self.assertEqual(self.r.fmt(df.loc[df.index.max(), "erfolgsquote"] * 100, 1),
                         self._texte("ErfolgsquotePufferMax"))


class TestChapitresNeufDix(unittest.TestCase):
    """Tache 10, chapitres 9 et 10: comptes de la presse relus dans data/presse.json,
    jalons relus dans fig_zeitplan.csv, ages et epargnes relus contre les macros des
    chapitres 5 et 8 (meme calcul, jamais un second)."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex")).read()
        cls.presse = json.load(open(os.path.join(ROOT, "data", "presse.json")))

    def _texte(self, name):
        import re
        m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}$", self.makros, re.M)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _wert(self, name):
        return float(self._texte(name).replace("\\,", "").replace("{,}", ".").replace("$-$", "-"))

    def test_comptes_presse(self):
        n = len(self.presse)
        self.assertEqual(self._wert("PresseCitations"), n)
        self.assertEqual(self._wert("PresseArticles"), len({e["id"] for e in self.presse}))
        self.assertEqual(self._wert("PressePublications"), len({e["quelle"] for e in self.presse}))
        pos = [self._wert(k) for k in ("PressePro", "PresseContra", "PresseNeutre")]
        self.assertEqual(sum(pos), n)
        self.assertEqual(pos[0], sum(e["position"] == "pro" for e in self.presse))

    def test_fiscalite_toutes_contra(self):
        """La prose dit que toutes les citations sur la fiscalite sont defavorables: le
        script leve une erreur sinon; ici, le compte et la date sont relus."""
        fisc = [e for e in self.presse if self.r._theme_normalise(e["thema"]) == "fiscalite"]
        self.assertEqual(self._wert("PresseCitationsFiscalite"), len(fisc))
        self.assertTrue(all(e["position"] == "contra" for e in fisc))
        self.assertGreater(len(fisc), 0)
        self.assertEqual(self._texte("PresseDateConsultation"),
                         self.r._date_fr(max(e["abgerufen"] for e in self.presse)))

    def test_jalons_relus_dans_fig_zeitplan(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fig_zeitplan.csv"))
        self.assertIn("libelle", df.columns)
        for nom, seuil in self.r.JALONS_CAPITAL.items():
            ligne = df[df["ereignis"] == f"capital_{int(seuil)}"]
            self.assertEqual(len(ligne), 1, nom)
            self.assertEqual(self._wert(f"JalonAnnee{nom}"), ligne["jahr"].iloc[0])
            self.assertEqual(self._texte(f"JalonSeuil{nom}"), self.r.fmt(seuil))
            self.assertGreaterEqual(ligne["kapital"].iloc[0], seuil)
        obj = df[df["ereignis"] == "objectif"]
        self.assertEqual(str(obj["jahr"].iloc[0]), self._texte("ZielJahrBasis"))
        self.assertEqual(self._texte("ZielJahrBasis"), self._texte("AnneeYannBasis"))

    def test_annees_des_choix(self):
        for age, annee in (("AlterEtfBasis", "AnneeEtfBasis"),
                           ("AlterAnticipeeMaisonSiebzig", "AnneeAnticipeeMaisonSiebzig"),
                           ("AlterAnticipeeEtfReferenz", "AnneeAnticipeeEtfReferenz")):
            self.assertEqual(self._wert(annee), self.r.GEBURTSJAHR + self._wert(age), annee)
        self.assertEqual(self._wert("AnneeRevueAvantObjectif"),
                         self._wert("ZielJahrBasis") - self._wert("RevueAvantObjectifAns"))

    def test_epargnes_des_choix(self):
        import projection as p
        for nom in ("MaisonSiebzig", "EtfReferenz"):
            q = self._wert(f"QuoteAnticipee{nom}") / 100
            attendu = p.sparplan_aus_quote(q, jahre=self.r.JAHRE_HORIZONT)[0] / 12
            self.assertEqual(self._texte(f"EpargneMoisAnticipee{nom}"), self.r.fmt(attendu))
        base = p.sparplan_aus_quote(self.r.QUOTEN[self.r.REFERENCE_QUOTE_NOM],
                                    jahre=self.r.JAHRE_HORIZONT)[0] / 12
        part = self.r._reference_strategie().anteil_maison
        self.assertEqual(self._texte("EpargneMoisMaisonBasis"), self.r.fmt(base * part))
        self.assertEqual(self._texte("EpargneMoisParTitre"),
                         self.r.fmt(base * part / self._wert("NombreTitres")))

    def test_reserve_urgence(self):
        import math
        import salaire
        net = salaire.trajektorie(self.r.START_JAHR, 1)[0]["netto_monat"]
        reserve = self.r.RESERVE_URGENCE_MOIS * net
        self.assertEqual(self._wert("ReserveUrgenceMois"), self.r.RESERVE_URGENCE_MOIS)
        self.assertEqual(self._texte("ReserveUrgence"), self.r.fmt(reserve))
        complement = max(0.0, reserve - self.r.STARTKAPITAL[self.r.REFERENCE_STARTKAPITAL_NOM])
        self.assertEqual(self._texte("ReserveUrgenceComplement"), self.r.fmt(complement))
        base = self.r.p.sparplan_aus_quote(self.r.QUOTEN[self.r.REFERENCE_QUOTE_NOM],
                                           jahre=self.r.JAHRE_HORIZONT)[0] / 12
        self.assertEqual(self._wert("ReserveUrgenceMoisComplement"), math.ceil(complement / base))


class TestAnnexeHypotheses(unittest.TestCase):
    """Tache 10, annexes: tableaux d'hypotheses generes depuis data/quellen.json et les
    macros de kennzahlen.tex, jamais tapes; bibliographie vide detectee par macro."""

    @classmethod
    def setUpClass(cls):
        import rechnung_retraite as r
        r.main()
        cls.r = r
        cls.quellen = json.load(open(os.path.join(ROOT, "data", "quellen.json"), encoding="utf-8"))
        cls.sources = open(os.path.join(ROOT, "data", "tab_hypotheses_sources.tex"), encoding="utf-8").read()
        cls.modele = open(os.path.join(ROOT, "data", "tab_hypotheses_modele.tex"), encoding="utf-8").read()
        cls.makros = open(os.path.join(ROOT, "data", "kennzahlen.tex"), encoding="utf-8").read()

    def test_une_ligne_par_cle_avec_lien_vers_la_source(self):
        lignes = [l for l in self.sources.splitlines() if not l.startswith("%")]
        self.assertEqual(len(lignes), len(self.quellen))
        for cle in self.quellen:
            self.assertIn(f"\\hyperref[src:{cle}]", self.sources, cle)

    def test_valeurs_formatees_depuis_le_json(self):
        # 0.25 -> "25~\\%" ; 1000 -> "1\\,000~€" : la valeur vient du JSON, format francais.
        self.assertIn(self.r._pct(self.quellen["abgeltungsteuer_satz"]["wert"], 0), self.sources)
        self.assertIn("25~\\%", self.sources)
        self.assertIn("1\\,000~€ par an", self.sources)
        self.assertIn("(15~\\% avec W-8BEN)", self.sources)

    def test_pas_de_hinweis_dans_le_tableau(self):
        for cle, q in self.quellen.items():
            if q.get("hinweis"):
                self.assertNotIn(self.r.escapieren(q["hinweis"]), self.sources, cle)

    def test_cle_absente_leve_une_erreur(self):
        # Revue finale, constat 1: un simple pop()/reassignation remet la cle a la FIN
        # du dict (un dict Python garde l'ordre d'insertion), ce qui deplacait
        # durablement "soli_satz" dans data/tab_hypotheses_sources.tex pour tout
        # r.main() ulterieur dans le meme processus (repere avec
        # `python scripts/Test/test_rechnung.py`, ou l'ordre alphabetique des noms de
        # classe fait tourner ce test avant d'autres qui rappellent main()).
        # mock.patch.dict fait un clear()+update() sur le dict ORIGINAL a la sortie,
        # ce qui restaure aussi bien le contenu que l'ordre d'insertion.
        ordre_avant = list(self.r.ANNEXE_HYPOTHESES)
        with mock.patch.dict(self.r.ANNEXE_HYPOTHESES, clear=False):
            del self.r.ANNEXE_HYPOTHESES["soli_satz"]
            with self.assertRaises(ValueError):
                self.r.annexe_hypotheses()
        self.assertEqual(list(self.r.ANNEXE_HYPOTHESES), ordre_avant,
                          "l'ordre de ANNEXE_HYPOTHESES doit survivre au test: "
                          "data/tab_hypotheses_sources.tex en depend (revue finale, constat 1)")

    def test_conventions_citent_des_macros_existantes(self):
        import re
        noms = set(re.findall(r"^\\newcommand\{\\(\w+)\}", self.makros, re.M))
        for nom in re.findall(r"\\([A-Za-z]+)", self.modele):
            if nom in ("ref",):
                continue
            self.assertIn(nom, noms, nom)

    def test_compte_des_references_bibtex(self):
        texte = "% commentaire @article{x,\n@comment{y}\n@Article{a2020b,\n title={t}}\n@book{c,}\n"
        self.assertEqual(len(self.r._BIB_ENTREE.findall(texte)), 2)
        bib = open(os.path.join(ROOT, "data", "literatur.bib"), encoding="utf-8").read()
        self.assertIn(f"\\newcommand{{\\NombreReferencesAcademiques}}{{{len(self.r._BIB_ENTREE.findall(bib))}}}",
                      self.makros)


if __name__ == "__main__":
    unittest.main()
