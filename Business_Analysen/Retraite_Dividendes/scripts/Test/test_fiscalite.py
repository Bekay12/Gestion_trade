"""
Cas calcules a la main. Retenue a la source: le taux imputable en Allemagne reduit
l'impot allemand (25 %), le Soli porte sur l'impot residuel. ETF actions: 30 % exoneres.
Les taux de pays sont passes explicitement (sq) pour que le test ne depende pas de
quellen.json, sauf test_deutschland_table_par_defaut qui verifie precisement la table
par defaut (garde-fou contre une double imposition des dividendes allemands).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fiscalite as f   # noqa: E402

SQ = {"DE": {"einbehalt": 0.0, "anrechenbar": 0.0, "mit_antrag": 0.0},
      "GB": {"einbehalt": 0.0, "anrechenbar": 0.0, "mit_antrag": 0.0},
      "US": {"einbehalt": 0.15, "anrechenbar": 0.15, "mit_antrag": 0.15},
      "CH": {"einbehalt": 0.35, "anrechenbar": 0.15, "mit_antrag": 0.15},
      "FR": {"einbehalt": 0.25, "anrechenbar": 0.128, "mit_antrag": 0.128}}


class TestPosten(unittest.TestCase):
    def netto(self, art, antrag=False):
        return f.posten_netto(100.0, art, 0.0, antrag=antrag, sq=SQ)[0]

    def test_deutschland(self):
        self.assertAlmostEqual(self.netto("DE"), 73.625, places=3)

    def test_usa(self):            # 15 retenus, 25-15=10 d'impot allemand, +5,5 % Soli
        self.assertAlmostEqual(self.netto("US"), 74.45, places=3)

    def test_schweiz_ohne_erstattung(self):
        self.assertAlmostEqual(self.netto("CH"), 54.45, places=3)

    def test_schweiz_mit_erstattung(self):
        self.assertAlmostEqual(self.netto("CH", antrag=True), 74.45, places=3)

    def test_frankreich(self):     # 25 retenus, 12,8 imputables: 12,2 x 1,055 = 12,871
        self.assertAlmostEqual(self.netto("FR"), 62.129, places=3)

    def test_etf_teilfreistellung(self):   # 70 imposables x 26,375 % = 18,4625
        self.assertAlmostEqual(self.netto("ETF"), 81.5375, places=4)

    def test_pauschbetrag_frisst_anrechnung(self):
        # 1 000 US-dividende sous le forfait: impot allemand nul, retenue perdue
        netto, rest = f.posten_netto(1000.0, "US", 1000.0, sq=SQ)
        self.assertAlmostEqual(netto, 850.0)
        self.assertAlmostEqual(rest, 0.0)

    def test_kirchensteuer_8(self):
        # (e)/(4+k): 100/(4.08) = 24,5098 ESt ; Soli 1,348 ; KiSt 1,9608
        self.assertAlmostEqual(f.steuer_kap(100.0, kist=0.08), 27.8186, places=3)

    def test_deutschland_table_par_defaut(self):
        # Garde-fou: quellen.json "DE" doit rester einbehalt=anrechenbar=0 (posten_netto
        # calcule deja l'impot allemand via SATZ/steuer_kap; une retenue "etrangere" non
        # nulle sur "DE" doublerait l'impot). sq=None -> charge la table de hypotheses.py.
        netto, _ = f.posten_netto(100.0, "DE", 0.0)
        self.assertAlmostEqual(netto, 73.625, places=3)

    def test_kirchensteuer_avec_credit_etranger(self):
        # US, kist=0,08, pas de forfait. Formule statutaire (§32d al. 1 EStG):
        # ESt = (e - 4q)/(4+k), q = credit plafonne a l'impot allemand du poste.
        #   e = 100, k = 0,08
        #   est_sans_credit = 100 / 4,08 = 24,509803921568627
        #   q = min(0,15 x 100, est_sans_credit) = min(15, 24,5098) = 15
        #   ESt = (100 - 4 x 15) / 4,08 = 40 / 4,08 = 9,803921568627452
        #   Soli+KiSt: ESt x (1 + 0,055 + 0,08) = 9,803921568627452 x 1,135
        #            = 11,127450980392158
        #   netto = 100 - 0,15 x 100 - 11,127450980392158 = 73,872549019607847
        # (la formule naive ESt = e/(4+k) - q donnerait 74,206..., un montant different
        # des lors que k > 0 ; c'est pourquoi ce cas est verifie separement.)
        netto, _ = f.posten_netto(100.0, "US", 0.0, kist=0.08, sq=SQ)
        self.assertAlmostEqual(netto, 73.873, places=3)


class TestKV(unittest.TestCase):
    SAETZE = {"kv": 0.14, "zusatz": 0.025, "pv": 0.042, "min_monat": 1250.0, "bbg_monat": 5800.0}

    def test_unter_mindestbemessung(self):
        self.assertAlmostEqual(f.kv_beitrag_jahr(6000.0, self.SAETZE), 1250.0 * 12 * 0.207, places=2)

    def test_ueber_bbg(self):
        self.assertAlmostEqual(f.kv_beitrag_jahr(200000.0, self.SAETZE), 5800.0 * 12 * 0.207, places=2)

    def test_brutto_fuer_netto_ist_umkehrung(self):
        mix = {"ETF": 0.3, "US": 0.4, "DE": 0.3}
        b = f.brutto_fuer_netto(42000.0, mix, self.SAETZE, 1000.0, sq=SQ)
        posten = [(b * a, k) for k, a in mix.items()]
        netto = f.jahres_netto(posten, 1000.0, sq=SQ) - f.kv_beitrag_jahr(b, self.SAETZE)
        self.assertAlmostEqual(netto, 42000.0, delta=1.0)


class TestVorabpauschale(unittest.TestCase):
    def test_basisertrag_begrenzt(self):
        # 10 000 x 2,53 % x 0,7 = 177,10 ; hausse 1 000 -> pas limitant
        self.assertAlmostEqual(f.vorabpauschale(10000, 11000, 0.0, 0.0253), 177.10, places=2)

    def test_verlustjahr_null(self):
        self.assertEqual(f.vorabpauschale(10000, 9000, 0.0, 0.0253), 0.0)

    def test_ausschuettung_wird_abgezogen(self):
        self.assertAlmostEqual(f.vorabpauschale(10000, 11000, 100.0, 0.0253), 77.10, places=2)


if __name__ == "__main__":
    unittest.main()
