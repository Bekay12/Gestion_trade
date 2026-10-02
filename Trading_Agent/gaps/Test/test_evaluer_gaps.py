#!/usr/bin/env python3
"""
Tests hors ligne de evaluer_gaps.noter(): chaque verdict est juge contre sa propre
promesse. Cas reels du 28.09.2026 pour la promesse FADE (retour a la cloture de la
veille), corrigee ce jour-la: l'ancien test (bas <= ouverture x 0,995) jugeait
"juste" ARAY, clos +36,3 % au-dessus de la veille.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluer_gaps import noter   # noqa: E402


def serie(veille, o, h, b, c):
    return {"veille": veille, "dates": ["2026-09-28"], "ouverture": [o], "haut": [h],
            "bas": [b], "cloture": [c]}


class TestFade(unittest.TestCase):
    def test_aray_gap_non_comble_est_faux(self):
        # veille 0,220 ; ouverture 0,288 ; bas 0,284 ; cloture 0,300
        n = noter({"ticker": "ARAY", "classe": "FADE"}, serie(0.220, 0.288, 0.382, 0.284, 0.300))
        self.assertEqual(n["resultat"], "faux")
        self.assertLess(n["vente_ouverture_cloture_pct"], 0)

    def test_srfm_gap_comble_mais_vente_perdante(self):
        # veille 1,110 ; bas 1,075 <= veille : promesse tenue ; cloture au-dessus de l'ouverture
        n = noter({"ticker": "SRFM", "classe": "FADE"}, serie(1.110, 1.125, 1.200, 1.075, 1.185))
        self.assertEqual(n["resultat"], "juste")
        self.assertAlmostEqual(n["vente_ouverture_cloture_pct"], -5.33, places=1)

    def test_tolerance_d_un_demi_pour_cent(self):
        n = noter({"ticker": "T", "classe": "FADE"}, serie(1.000, 1.100, 1.120, 1.004, 1.010))
        self.assertEqual(n["resultat"], "juste")

    def test_sans_veille_le_verdict_est_incomplet(self):
        n = noter({"ticker": "T", "classe": "FADE"}, serie(None, 1.1, 1.2, 1.0, 1.05))
        self.assertEqual(n["resultat"], "incomplet")


class TestAutresClasses(unittest.TestCase):
    def test_pump_risk_retombe_est_juste(self):
        n = noter({"ticker": "GYGY", "classe": "PUMP_RISK"}, serie(0.740, 1.215, 1.270, 0.780, 0.873))
        self.assertEqual(n["resultat"], "juste")

    def test_insuffisant_n_est_pas_juge(self):
        n = noter({"ticker": "MTEK", "classe": "INSUFFISANT"}, serie(1.095, 1.0, 1.1, 0.93, 0.971))
        self.assertEqual(n["resultat"], "non juge")

    def test_sans_cours_incomplet(self):
        self.assertEqual(noter({"ticker": "X", "classe": "FADE"}, None)["resultat"], "incomplet")


class TestExecutabiliteDansLaNotation(unittest.TestCase):
    def test_les_shorts_recoivent_une_mesure_d_executabilite(self):
        import json
        import tempfile
        from unittest import mock

        import evaluer_gaps
        detection = {"date": "2026-09-29", "mode": "premarket", "verdicts": [
            {"ticker": "FFAI", "classe": "PUMP_RISK"},
            {"ticker": "CYAB", "classe": "FADE"},
            {"ticker": "MSGY", "classe": "INSUFFISANT"}]}
        cours = {"FFAI": serie(1.41, 1.53, 1.56, 1.25, 1.30),
                 "CYAB": serie(0.190, 0.197, 0.216, 0.186, 0.211),
                 "MSGY": serie(3.13, 3.94, 4.77, 3.20, 4.44)}
        barres = {"FFAI": [{"cloture": 1.5, "volume": 1e6}] * 6}   # CYAB sans mesure
        with tempfile.TemporaryDirectory() as tmp:
            chemin = os.path.join(tmp, "2026-09-29_premarket.json")
            json.dump(detection, open(chemin, "w"))
            with mock.patch.object(evaluer_gaps, "cours_depuis", return_value=cours), \
                 mock.patch.object(evaluer_gaps, "barres_ouverture", return_value=barres) as b:
                evaluer_gaps.main(["--fichier", chemin, "--jours", "1"])
            notes = {n["ticker"]: n for n in
                     json.load(open(chemin.replace(".json", "_evalue_J1.json")))["notes"]}
        self.assertEqual(sorted(b.call_args[0][0]), ["CYAB", "FFAI"])   # shorts seulement
        self.assertEqual(notes["FFAI"]["executabilite"]["executable"], "oui")
        self.assertEqual(notes["CYAB"]["executabilite"]["executable"], "inconnu")
        self.assertNotIn("executabilite", notes["MSGY"])


if __name__ == "__main__":
    unittest.main()
