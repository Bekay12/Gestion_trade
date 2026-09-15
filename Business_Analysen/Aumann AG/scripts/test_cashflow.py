"""
Tests fuer cashflow.py - grobe Cash-Flow-Analyse der Optionen A, B, C.
Die Erwartungswerte folgen unmittelbar aus den Annahmen der
Aufgabenstellung und dem Investitionsprofil des Spezifikationsdokuments.
Reine Standardbibliothek.
"""
import unittest

import cashflow as cf


class TestAnnahmen(unittest.TestCase):
    def test_investitionssummen(self):
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["A"]["investment"].values()), 35.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["B"]["investment"].values()), 30.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["C"]["investment"].values()), 60.0, places=2)

    def test_kapitalbindungssummen(self):
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["A"]["working_capital"].values()), 15.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["B"]["working_capital"].values()), 15.0, places=2)
        self.assertAlmostEqual(sum(cf.ASSUMPTIONS["C"]["working_capital"].values()), 20.0, places=2)

    def test_ebitda_startjahre(self):
        self.assertEqual(cf.ASSUMPTIONS["A"]["ebitda_start"], 2028)
        self.assertEqual(cf.ASSUMPTIONS["B"]["ebitda_start"], 2027)
        self.assertEqual(cf.ASSUMPTIONS["C"]["ebitda_start"], 2027)

    def test_ebitda_zuwaechse(self):
        self.assertAlmostEqual(cf.ASSUMPTIONS["A"]["ebitda_delta"], 16.0, places=2)
        self.assertAlmostEqual(cf.ASSUMPTIONS["B"]["ebitda_delta"], 14.0, places=2)
        self.assertAlmostEqual(cf.ASSUMPTIONS["C"]["ebitda_delta"], 10.0, places=2)

    def test_akquisition_zahlt_kaufpreis_bei_vollzug(self):
        # Option C: Kaufpreis 45 Mio. EUR im Vollzugsjahr 2026,
        # danach je 7,5 Mio. EUR Integration.
        self.assertAlmostEqual(cf.ASSUMPTIONS["C"]["investment"][2026], 45.0, places=2)


class TestSchedule(unittest.TestCase):
    def _liq(self, key):
        return [round(r["liquiditaet"], 1) for r in cf.schedule(key)]

    def test_liquiditaet_option_a(self):
        # 138,2 -11,7 = 126,5 | -11,7 -7,5 = 107,3 | -11,6 +16 -7,5 = 104,2
        self.assertEqual(self._liq("A"), [126.5, 107.3, 104.2])

    def test_liquiditaet_option_b(self):
        # 138,2 -10 -7,5 = 120,7 | -10 +14 -7,5 = 117,2 | -10 +14 = 121,2
        self.assertEqual(self._liq("B"), [120.7, 117.2, 121.2])

    def test_liquiditaet_option_c(self):
        # 138,2 -45 -20 = 73,2 | -7,5 +10 = 75,7 | -7,5 +10 = 78,2
        self.assertEqual(self._liq("C"), [73.2, 75.7, 78.2])

    def test_schedule_hat_drei_jahre(self):
        for key in ("A", "B", "C"):
            rows = cf.schedule(key)
            self.assertEqual([r["jahr"] for r in rows], [2026, 2027, 2028])

    def test_jede_zeile_hat_bemerkung(self):
        for key in ("A", "B", "C"):
            for row in cf.schedule(key):
                self.assertTrue(row["bemerkung"].strip())

    def test_startliquiditaet_ist_vorgabe(self):
        self.assertAlmostEqual(cf.OPENING_LIQUIDITY, 138.2, places=2)

    def test_option_c_hat_tiefsten_liquiditaetspunkt(self):
        # Der Kaufpreis im Vollzugsjahr ist der tiefste Punkt aller Optionen.
        tiefstwerte = {k: min(r["liquiditaet"] for r in cf.schedule(k)) for k in ("A", "B", "C")}
        self.assertLess(tiefstwerte["C"], tiefstwerte["A"])
        self.assertLess(tiefstwerte["C"], tiefstwerte["B"])


class TestIRR(unittest.TestCase):
    def test_irr_einfacher_fall(self):
        self.assertAlmostEqual(cf.irr([-100.0, 110.0]), 0.10, places=4)

    def test_irr_zweiperiodig(self):
        # -100 heute, +60 und +60 -> IRR ca. 13,07 %
        self.assertAlmostEqual(cf.irr([-100.0, 60.0, 60.0]), 0.13066, places=4)

    def test_npv_am_irr_ist_null(self):
        for key in ("A", "B", "C"):
            flows = cf.full_cashflows(key)
            r = cf.irr(flows)
            self.assertAlmostEqual(cf.npv(flows, r), 0.0, places=6)

    def test_npv_bei_null_prozent_ist_summe(self):
        flows = [-10.0, 5.0, 8.0]
        self.assertAlmostEqual(cf.npv(flows, 0.0), 3.0, places=6)

    def test_irr_ohne_vorzeichenwechsel_wirft_fehler(self):
        with self.assertRaises(ValueError):
            cf.irr([10.0, 10.0, 10.0])


class TestFullCashflows(unittest.TestCase):
    def test_laenge_entspricht_horizont(self):
        flows = cf.full_cashflows("A", horizon_end=2035)
        self.assertEqual(len(flows), 10)  # 2026..2035

    def test_kapitalbindung_wird_am_ende_freigesetzt(self):
        flows_a = cf.full_cashflows("A", horizon_end=2035)
        # Letztes Jahr: laufendes EBITDA 16 + freigesetzte Kapitalbindung 15
        self.assertAlmostEqual(flows_a[-1], 31.0, places=2)

    def test_erstes_jahr_entspricht_investition_option_b(self):
        # B 2026: -10 Investition -7,5 Kapitalbindung, kein EBITDA
        self.assertAlmostEqual(cf.full_cashflows("B")[0], -17.5, places=2)

    def test_alle_optionen_haben_positiven_irr_auf_zehn_jahren(self):
        for key in ("A", "B", "C"):
            self.assertGreater(cf.irr(cf.full_cashflows(key)), 0.0)


class TestEbitdaMarge(unittest.TestCase):
    def test_marge_formel(self):
        # (Basis-EBITDA 20 + Zuwachs 16) / Umsatz 300 = 12 %
        got = cf.ebitda_marge_2028("A", umsatz=300.0, ebitda_basis=20.0)
        self.assertAlmostEqual(got, 12.0, places=4)


class TestMargenSensitivitaet(unittest.TestCase):
    """
    Die Marge Ende 2028 haengt vollstaendig davon ab, welcher Umsatz im
    Nenner steht. Der Umsatz 2025 war gegenueber 2024 stark eingebrochen;
    ihn konstant fortzuschreiben schmeichelt der Marge. Diese Tests sichern,
    dass die Rechnung den Wachstumspfad tatsaechlich beruecksichtigt.
    """

    def test_wachstum_null_entspricht_basisfall(self):
        ohne = cf.ebitda_marge_2028("A", umsatz=300.0, ebitda_basis=20.0)
        mit = cf.ebitda_marge_2028_wachstum(
            "A", umsatz=300.0, ebitda_basis=20.0, wachstum=0.0)
        self.assertAlmostEqual(ohne, mit, places=6)

    def test_umsatzwachstum_senkt_die_marge(self):
        # Groesserer Nenner, unveraenderter Zaehler -> kleinere Marge.
        basis = cf.ebitda_marge_2028("B", umsatz=200.0, ebitda_basis=27.0)
        gewachsen = cf.ebitda_marge_2028_wachstum(
            "B", umsatz=200.0, ebitda_basis=27.0, wachstum=0.03)
        self.assertLess(gewachsen, basis)

    def test_wachstum_wirkt_ueber_drei_jahre(self):
        # 100 * 1.1^3 = 133.1 -> (20 + 16) / 133.1 = 27.05 %
        got = cf.ebitda_marge_2028_wachstum(
            "A", umsatz=100.0, ebitda_basis=20.0, wachstum=0.10)
        self.assertAlmostEqual(got, 36.0 / 133.1 * 100.0, places=6)

    def test_referenzumsatz_ergibt_niedrigste_marge(self):
        # Mit dem Umsatzniveau 2024 im Nenner faellt die Marge deutlich.
        got = cf.ebitda_marge_2028_referenz("A", umsatz_referenz=312.346,
                                            ebitda_basis=27.278)
        self.assertAlmostEqual(got, 43.278 / 312.346 * 100.0, places=6)


class TestNetDebt(unittest.TestCase):
    def test_nettoguthaben_ergibt_negativen_verschuldungsgrad(self):
        # Option C endet 2028 mit positiver Nettoliquiditaet -> negativer Wert.
        self.assertLess(cf.net_debt_to_ebitda("C", ebitda_basis=20.0), 0.0)


class TestLatexAusgabe(unittest.TestCase):
    def test_tabellenkoerper_hat_drei_zeilen(self):
        body = cf.emit_schedule_table("A")
        self.assertEqual(body.count(r"\\"), 3)

    def test_tabellenkoerper_nutzt_dezimalpunkt_fuer_siunitx(self):
        # siunitx formatiert selbst; im Quelltext steht der englische Punkt.
        body = cf.emit_schedule_table("A")
        self.assertIn("126.5", body)
        self.assertNotIn("126,5", body)

    def test_vergleichstabelle_nennt_net_debt_nur_fuer_c(self):
        body = cf.emit_comparison_table(umsatz=300.0, ebitda_basis=20.0)
        self.assertEqual(body.count("nicht relevant"), 2)

    def test_sensitivitaetstabelle_hat_drei_horizonte(self):
        body = cf.emit_irr_sensitivity()
        self.assertEqual(body.count(r"\\"), 3)

    def test_makrodatei_definiert_je_option_die_kernwerte(self):
        # Die Wertungsabschnitte duerfen keine Zahl hartkodieren; sie lesen
        # die Ergebnisse der Rechnung ueber diese Makros.
        src = cf.emit_macros(umsatz=203.985, ebitda_basis=27.278)
        for key in ("A", "B", "C"):
            for name in ("Investitionssumme", "Kapitalbindungssumme",
                         "Kapitalbedarf", "EbitdaZuwachs", "EbitdaStart",
                         "Irr", "IrrFuenf", "LiquiditaetEnde",
                         "MargeZweitausendachtundzwanzig"):
                self.assertIn(f"\\newcommand{{\\{name}{key}}}", src)

    def test_makrodatei_kennt_die_besonderheiten_der_option_c(self):
        src = cf.emit_macros(umsatz=203.985, ebitda_basis=27.278)
        self.assertIn(r"\newcommand{\KaufpreisC}", src)
        self.assertIn(r"\newcommand{\LiquiditaetTiefC}", src)
        self.assertIn(r"\newcommand{\NetDebtEbitdaC}", src)

    def test_makrowerte_stimmen_mit_der_rechnung_ueberein(self):
        src = cf.emit_macros(umsatz=203.985, ebitda_basis=27.278)
        erwartet = f"{cf.schedule('C')[-1]['liquiditaet']:.1f}"
        self.assertIn(r"\newcommand{\LiquiditaetEndeC}{" + erwartet + "}", src)

    def test_kapitalbedarf_ist_investition_plus_kapitalbindung(self):
        src = cf.emit_macros(umsatz=203.985, ebitda_basis=27.278)
        self.assertIn(r"\newcommand{\KapitalbedarfC}{80.0}", src)

    def test_margentabelle_hat_drei_szenarien(self):
        body = cf.emit_marge_sensitivity(umsatz=203.985, ebitda_basis=27.278,
                                         umsatz_referenz=312.346)
        self.assertEqual(body.count(r"\\"), 3)

    def test_margentabelle_faellt_von_szenario_zu_szenario(self):
        # Zeile 1 Basisfall, Zeile 2 Umsatz +3 % p.a., Zeile 3 Niveau 2024:
        # der Nenner waechst, die ausgewiesene Marge muss sinken.
        body = cf.emit_marge_sensitivity(umsatz=203.985, ebitda_basis=27.278,
                                         umsatz_referenz=312.346)
        werte = [float(z.split("&")[1].split("{")[1].split("}")[0])
                 for z in body.strip().splitlines()]
        self.assertEqual(werte, sorted(werte, reverse=True))


if __name__ == "__main__":
    unittest.main()
