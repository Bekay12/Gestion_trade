"""Chaque hypothese a une valeur plausible ET une source avec date de consultation."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hypotheses as h   # noqa: E402
import salaire as sal    # noqa: E402

PLAGES = {
    "abgeltungsteuer_satz": (0.25, 0.25), "soli_satz": (0.055, 0.055),
    "sparerpauschbetrag": (1000, 2000), "teilfreistellung_aktienfonds": (0.30, 0.30),
    "basiszins_2026": (0.0, 0.05), "kv_satz_ermaessigt": (0.13, 0.15),
    "kv_zusatzbeitrag_2026": (0.015, 0.04), "pv_satz_kinderlos_2026": (0.035, 0.05),
    "kv_mindestbemessung_monat_2026": (1000, 1500), "kv_bbg_monat_2026": (5000, 6500),
    "inflation_ziel": (0.02, 0.02), "inflation_2022_de": (0.05, 0.09),
    "einstiegsgehalt_brutto": (45000, 75000), "gehaltssteigerung_real": (0.0, 0.04),
    "rentenwert_2026": (35, 50), "regelaltersgrenze": (67, 67),
    "durchschnittsentgelt_2026": (45000, 60000),
}


class TestHypothesen(unittest.TestCase):
    def test_werte_in_plausiblen_bereichen(self):
        for k, (lo, hi) in PLAGES.items():
            w = h.wert(k)
            self.assertIsNotNone(w, k)
            self.assertTrue(lo <= w <= hi, f"{k}={w} hors de [{lo}, {hi}]")

    def test_jede_quelle_hat_url_und_datum(self):
        for k, e in h.alle().items():
            self.assertTrue(e.get("quelle"), k)
            self.assertTrue(e.get("abgerufen"), k)

    def test_quellensteuer_tabelle(self):
        q = h.wert("quellensteuer")
        for land in ("US", "CH", "FR", "NL", "GB", "DE"):
            self.assertIn(land, q)
            self.assertTrue(0 <= q[land]["anrechenbar"] <= q[land]["einbehalt"] or q[land]["einbehalt"] == 0)

    def test_est_tarif_hat_fuenf_zonen(self):
        t = h.wert("est_tarif_2026")
        self.assertEqual(len(t["zonen"]), 5)

    def test_est_32a_gegen_veroeffentlichte_2026_werte(self):
        """
        Kreuzprobe der echten est_tarif_2026-Tabelle aus quellen.json gegen die am
        30.09.2026 per WebFetch abgerufenen §32a-Formeln fuer den VZ 2026
        (https://www.finanz-tools.de/einkommensteuer/berechnung-formeln/2026): identische
        Koeffizienten (Grundfreibetrag 12.348 EUR; Zone 2 y=(zvE-12348)/10000,
        ESt=(914,51*y+1400)*y; Zone 3 z=(zvE-17799)/10000, ESt=(173,10*z+2397)*z+1034,87;
        Zone 4 ESt=0,42*zvE-11135,63; Zone 5 ESt=0,45*zvE-19470,38). Erwartete Werte hier
        unabhaengig von salaire.est_32a von Hand aus genau diesen Formeln nachgerechnet
        (siehe docs/salaire_controle.md), nicht aus salaire.py uebernommen.
        """
        t = h.wert("est_tarif_2026")
        erwartet = {15000: 435, 30000: 4217, 60000: 14233}
        for zve, est in erwartet.items():
            self.assertEqual(sal.est_32a(zve, t), est, zve)

    def test_pruefen_leer(self):
        self.assertEqual(h.pruefen(), [])


if __name__ == "__main__":
    unittest.main()
