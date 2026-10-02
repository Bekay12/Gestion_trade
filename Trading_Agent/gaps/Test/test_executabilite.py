#!/usr/bin/env python3
"""
Tests hors ligne de executabilite: un short "juste" n'est compte gagnant que si
le titre etait negociable dans les 30 premieres minutes. Motif, 29.09.2026: la
revue Germain sur EGG (pic a 13,11 $ dans un carnet vide a 04:00 ET) et nos
shorts SLXN / CYAB notes justes alors que le borrow etait douteux.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from executabilite import bilan_shorts, juger   # noqa: E402


def barres(prix, volumes):
    return [{"cloture": prix, "volume": v} for v in volumes]


class TestJuger(unittest.TestCase):
    def test_titre_liquide_est_executable(self):
        # FFAI 29.09: ~1,5 $, plusieurs millions de titres a l'ouverture
        e = juger(barres(1.50, [3_000_000, 800_000, 500_000, 400_000, 300_000, 300_000]))
        self.assertEqual(e["executable"], "oui")
        self.assertGreater(e["volume_post_ouverture_usd"], 200_000)

    def test_volume_en_dollars_trop_faible(self):
        # 0,20 $ x 500 000 titres = 100 000 $ apres l'ouverture: une position de 10 000 $
        # pese plus de 5 % du flux
        e = juger(barres(0.20, [100_000] * 6))
        self.assertEqual(e["executable"], "non")
        self.assertIn("volume", e["motif"])

    def test_barres_vides_rendent_le_titre_inexecutable(self):
        # Beaucoup de dollars sur la premiere barre (enchere), puis carnet vide
        e = juger(barres(5.0, [1_000_000, 0, 0, 2_000, 0, 1_000]))
        self.assertEqual(e["executable"], "non")
        self.assertIn("sans echange", e["motif"])

    def test_la_barre_d_ouverture_ne_compte_pas(self):
        # EGG 29.09, barres reelles 09:30-09:55: la barre de 09:30 porte l'enchere
        # d'ouverture et le pre-marche (1,56 M titres), puis le carnet se vide.
        egg = [{"cloture": c, "volume": v} for c, v in
               [(3.66, 1559615), (3.363, 19802), (3.46, 14302),
                (3.415, 5971), (3.31, 7954), (3.33, 10058)]]
        e = juger(egg)
        self.assertEqual(e["executable"], "non")
        self.assertLess(e["volume_post_ouverture_usd"], 200_000)

    def test_sans_barres_inconnu(self):
        self.assertEqual(juger(None)["executable"], "inconnu")
        self.assertEqual(juger([])["executable"], "inconnu")


class TestExtraire(unittest.TestCase):
    def test_une_barre_absente_compte_comme_barre_vide(self):
        # yfinance omet les barres de 5 min sans echange au lieu de les mettre a 0
        import pandas as pd
        from executabilite import _extraire
        heures = ["09:30", "09:35", "09:45", "09:55"]      # 09:40 et 09:50 absentes
        idx = pd.DatetimeIndex([f"2026-09-29 {h}" for h in heures]).tz_localize("America/New_York")
        cols = pd.MultiIndex.from_product([["EGG"], ["Close", "Volume"]])
        lot = pd.DataFrame([[4.0, 1e6], [3.7, 2e4], [3.4, 1e3], [3.5, 5e2]], index=idx, columns=cols)
        b = _extraire(lot, ["EGG"])["EGG"]
        self.assertEqual(len(b), 6)
        self.assertEqual(sum(1 for x in b if not x["volume"]), 2)


class TestBilanShorts(unittest.TestCase):
    def test_seuls_les_shorts_justes_et_executables_comptent(self):
        notes = [
            {"classe": "FADE", "resultat": "juste", "executabilite": {"executable": "oui"}},
            {"classe": "PUMP_RISK", "resultat": "juste", "executabilite": {"executable": "non"}},
            {"classe": "FADE", "resultat": "juste", "executabilite": {"executable": "inconnu"}},
            {"classe": "PUMP_RISK", "resultat": "faux", "executabilite": {"executable": "oui"}},
            {"classe": "INSUFFISANT", "resultat": "non juge"},
        ]
        b = bilan_shorts(notes)
        self.assertEqual(b["justes"], 3)
        # Une donnee absente vaut refus: "inconnu" ne compte pas comme executable
        self.assertEqual(b["justes_executables"], 1)


if __name__ == "__main__":
    unittest.main()
