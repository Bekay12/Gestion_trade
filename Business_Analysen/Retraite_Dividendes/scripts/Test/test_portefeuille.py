import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import portefeuille_exemple as pe   # noqa: E402


def serie(jahresbetraege, start=2015):
    idx = pd.to_datetime([f"{start + i}-06-15" for i in range(len(jahresbetraege))])
    return pd.Series(jahresbetraege, index=idx)


class TestKennzahlen(unittest.TestCase):
    def test_cagr_und_keine_kuerzung(self):
        d = serie([1.0 * 1.05 ** i for i in range(11)])
        k = pe.kennzahlen(d, pd.Series([100.0] * 11, index=d.index))
        self.assertAlmostEqual(k["div_cagr_10j"], 0.05, places=4)
        self.assertEqual(k["kuerzungen_10j"], 0)

    def test_kuerzung_gezaehlt(self):
        d = serie([1, 1.1, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        k = pe.kennzahlen(d, pd.Series([100.0] * 11, index=d.index))
        self.assertEqual(k["kuerzungen_10j"], 1)


class TestAuswahl(unittest.TestCase):
    def test_sektorgrenze_und_anzahl(self):
        df = pd.DataFrame({"ticker": [f"T{i}" for i in range(60)],
                           "sektor": ["A"] * 20 + [f"S{i}" for i in range(40)],
                           "rendite_netto_de": [0.03] * 60, "div_cagr_10j": [0.06] * 60,
                           "kuerzungen_10j": [0] * 60, "payout": [0.5] * 60,
                           "fcf_deckung": [1.5] * 60, "rendite_ttm": [0.035] * 60})
        a = pe.auswahl(df, n=30, sektor_max=5)
        self.assertEqual(len(a), 30)
        self.assertLessEqual((a["sektor"] == "A").sum(), 5)

    def test_filter_schliesst_kuerzer_aus(self):
        df = pd.DataFrame({"ticker": ["OK", "CUT"], "sektor": ["A", "B"],
                           "rendite_netto_de": [0.03, 0.05], "div_cagr_10j": [0.05, 0.05],
                           "kuerzungen_10j": [0, 2], "payout": [0.5, 0.5],
                           "fcf_deckung": [1.5, 1.5], "rendite_ttm": [0.035, 0.06]})
        self.assertEqual(list(pe.auswahl(df, n=30)["ticker"]), ["OK"])


class TestDaxZeilen(unittest.TestCase):
    """Tour de revue 1: le tableau DAX de Wikipedia porte deja le suffixe de cotation dans
    "Ticker" (ex. SAP.DE, ou AIR.PA pour Airbus). _dax_zeilen ne doit jamais accoler un
    second suffixe et doit deriver le pays du suffixe reel, pas le fixer a "DE"."""

    def test_suffixe_deja_present_et_bareback(self):
        table = pd.DataFrame({"Ticker": ["SAP.DE", "AIR.PA", "BMW"]})
        with patch.object(pe, "_lire_tabellen", return_value=[table]):
            luecken, diag = [], {}
            zeilen = pe._dax_zeilen(luecken, diag)
        par_ticker = {z["ticker"]: z["land"] for z in zeilen}
        self.assertEqual(par_ticker["SAP.DE"], "DE")
        self.assertEqual(par_ticker["AIR.PA"], "FR")   # suffixe reel, pas "DE" en dur
        self.assertEqual(par_ticker["BMW.DE"], "DE")   # ticker nu -> suffixe ajoute
        self.assertNotIn("SAP.DE.DE", par_ticker)      # jamais de double suffixe


class TestEurostoxxZeilen(unittest.TestCase):
    """Tour de revue 1: le pays vient de "Registered office" (domiciliation), pas d'un
    substring match contre le libelle abrege de la place de cotation; toute ligne au pays
    non reconnu doit etre comptee, jamais silencieusement ecartee."""

    def test_pays_reconnu_et_ligne_non_reconnue_comptee(self):
        table = pd.DataFrame({"Ticker": ["RACE.MI", "XYZ.ZZ"],
                              "Registered office": ["Netherlands", "Ruritania"]})
        with patch.object(pe, "_lire_tabellen", return_value=[table]):
            luecken, diag = [], {}
            zeilen = pe._eurostoxx_zeilen(luecken, diag)
        self.assertEqual(len(zeilen), 1)
        self.assertEqual(zeilen[0]["ticker"], "RACE.MI")
        self.assertEqual(zeilen[0]["land"], "NL")      # domiciliation, pas la place de cotation (.MI -> IT)
        self.assertEqual(diag.get("eurostoxx_pays_non_reconnu"), 1)
        self.assertTrue(any("pays non" in l for l in luecken))


class TestTickersValides(unittest.TestCase):
    def test_rejet_compte_et_journalise(self):
        luecken, diag = [], {}
        ok = pe._tickers_valides(["AAPL", "not a ticker!", "BRK-B"], luecken, diag,
                                 "contexte-test", "cle-test")
        self.assertEqual(ok, ["AAPL", "BRK-B"])
        self.assertEqual(diag["cle-test"], 1)
        self.assertTrue(any("rejetes" in l for l in luecken))


class TestPortefeuilleCsv(unittest.TestCase):
    """Contrat d'interface (tache 9): colonnes exactes, score arrondi a 4 decimales,
    aucune colonne de travail (index, kurs_cagr_10j, max_drawdown) ne doit fuiter."""

    def test_colonnes_et_arrondi(self):
        port = pd.DataFrame({"ticker": ["T1"], "name": ["Nom"], "land": ["DE"],
                             "sektor": ["Industrials"], "rendite_ttm": [0.04],
                             "rendite_netto_de": [0.03], "div_cagr_10j": [0.05],
                             "kuerzungen_10j": [0], "payout": [0.5], "fcf_deckung": [1.5],
                             "score": [0.123456789], "index": ["dax"],
                             "kurs_cagr_10j": [0.1], "max_drawdown": [-0.3]})
        out = pe._portefeuille_csv(port)
        self.assertEqual(list(out.columns), pe.COLONNES_PORTEFEUILLE)
        self.assertEqual(out["score"].iloc[0], 0.1235)

    def test_vide(self):
        out = pe._portefeuille_csv(pd.DataFrame())
        self.assertEqual(list(out.columns), pe.COLONNES_PORTEFEUILLE)
        self.assertEqual(len(out), 0)


def _ligne(ticker, sektor, cagr, payout):
    return {"ticker": ticker, "sektor": sektor, "rendite_ttm": 0.04, "rendite_netto_de": cagr * 0.6,
           "div_cagr_10j": cagr, "kuerzungen_10j": 0, "payout": payout, "fcf_deckung": 1.5}


class TestWaehleMitRelaxation(unittest.TestCase):
    def test_premier_palier_qui_atteint_n_arrete_la(self):
        # 2 lignes passent le seuil strict (croissance >= 3 %); 4 de plus passent des que la
        # croissance est relachee a 2 % (payout reste <= 80 %) -> le palier "croissance" a
        # lui seul suffit pour n=5, le palier payout ne doit meme pas etre essaye.
        lignes = [_ligne(f"S{i}", f"Sec{i}", 0.035, 0.5) for i in range(2)]
        lignes += [_ligne(f"G{i}", f"Sec{i + 2}", 0.025, 0.5) for i in range(4)]
        df = pd.DataFrame(lignes)
        port, relache, essais = pe._waehle_mit_relaxation(df, n=5)
        self.assertEqual(relache, "croissance >= 2 %")
        self.assertEqual(len(port), 5)
        self.assertEqual([nom for nom, _ in essais], ["strict", "croissance >= 2 %"])

    def test_aucun_palier_ne_suffit_garde_le_meilleur(self):
        # Aucun palier n'atteint n=30: strict=10, croissance relachee seule=15 (10+5),
        # payout relache seul=18 (10+8) -> le palier payout doit etre retenu (18 > 15 > 10).
        lignes = [_ligne(f"A{i}", f"SecA{i}", 0.035, 0.5) for i in range(10)]
        lignes += [_ligne(f"B{i}", f"SecB{i}", 0.025, 0.5) for i in range(5)]
        lignes += [_ligne(f"C{i}", f"SecC{i}", 0.035, 0.85) for i in range(8)]
        df = pd.DataFrame(lignes)
        port, relache, essais = pe._waehle_mit_relaxation(df, n=30)
        self.assertEqual(relache, "payout <= 90 %")
        self.assertEqual(len(port), 18)
        self.assertEqual(essais, [("strict", 10), ("croissance >= 2 %", 15), ("payout <= 90 %", 18)])


if __name__ == "__main__":
    unittest.main()
