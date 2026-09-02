"""
test_scanner - Suite hors ligne de la decouverte de candidats (F1).

    .venv\\Scripts\\python.exe agent\\Test\\test_scanner.py

Ni TWS, ni compte, ni reseau. Le test le plus important de ce fichier n'est pas
fonctionnel mais architectural : il verifie que les bornes du balayage SONT
celles de gate.py, pas des copies. Un scanner qui reciterait ses propres
chiffres divergerait du crible en silence, et le seul symptome serait une liste
de candidats que le crible refuse tous.
"""

from __future__ import annotations

import sys
import unittest
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.gate import (
    LOW_FLOAT_CEILING,
    MAX_MARKET_CAP,
    MAX_PRICE,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    R9_FLOAT_CEILING,
    check_filters,
)
from agent.models import Snapshot, Verdict
from agent.sources.ibkr import IbkrConnector, IbkrUnavailable
from agent.sources.scanner import (
    DEFAULT_ROWS,
    IB_MAX_ACTIVE_SCANS,
    IB_MAX_ROWS,
    SCAN_CODES,
    R9_TRAP,
    MarketScanner,
    ScanFilters,
    ScanUnavailable,
    build_filters,
    build_subscription,
    symbols_from_scan,
)


class FakeContract:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeDetails:
    def __init__(self, symbol):
        self.contract = FakeContract(symbol)


class FakeScanData:
    def __init__(self, rank, symbol):
        self.rank = rank
        self.contractDetails = FakeDetails(symbol)


def scan_rows(*symbols):
    return [FakeScanData(i, s) for i, s in enumerate(symbols)]


class FakeIB:
    """Client ib_async simule, cote scanner."""

    def __init__(self, results=None, fail_scan=False, fail_connect=False):
        # {scanCode: [ScanData]} ; une seule liste vaut pour tous les codes.
        self.results = results if results is not None else scan_rows("AAAA", "BBBB")
        self.fail_scan = fail_scan
        self.fail_connect = fail_connect
        self.connected = False
        self.subscriptions = []

    def isConnected(self):
        return self.connected

    def connect(self, host, port, clientId, timeout):
        if self.fail_connect:
            raise ConnectionRefusedError("port ferme")
        self.connected = True

    def disconnect(self):
        self.connected = False

    def reqMarketDataType(self, kind):
        pass

    def reqScannerData(self, subscription, *args, **kwargs):
        self.subscriptions.append(subscription)
        if self.fail_scan:
            raise RuntimeError("droits de donnees de marche absents")
        if isinstance(self.results, dict):
            return self.results.get(subscription.scanCode, [])
        return self.results


def scanner(ib):
    return MarketScanner(IbkrConnector(ib=ib))


class TestBornesImportees(unittest.TestCase):
    """Le test qui protege l'architecture : R6 est la seule source des seuils."""

    def test_les_bornes_sont_celles_de_gate(self):
        sub = build_subscription("spike")
        self.assertEqual(sub.abovePrice, MIN_PRICE)
        self.assertEqual(sub.belowPrice, MAX_PRICE)
        self.assertEqual(sub.aboveVolume, int(MIN_AVG_VOLUME))
        self.assertEqual(sub.marketCapBelow, float(MAX_MARKET_CAP))

    def test_aucune_borne_n_est_ecrite_en_dur_dans_le_module(self):
        # Si quelqu'un recopie 0.50 ou 20.0 dans scanner.py, ce test tombe.
        source = (Path(__file__).resolve().parents[1] / "sources" / "scanner.py")
        texte = source.read_text(encoding="utf-8")
        for interdit in ("0.50", "20.0", "500_000", "300_000_000", "20_000_000"):
            self.assertNotIn(interdit, texte, f"seuil R6 recopie : {interdit}")


class TestConstructionAbonnement(unittest.TestCase):

    def test_cle_traduite_en_code_tws(self):
        self.assertEqual(build_subscription("spike").scanCode, SCAN_CODES["spike"])
        self.assertEqual(build_subscription("volume").scanCode, "HOT_BY_VOLUME")

    def test_code_brut_passe_tel_quel(self):
        # Un compte peut connaitre des codes absents de SCAN_CODES.
        self.assertEqual(build_subscription("TOP_PERC_LOSE").scanCode, "TOP_PERC_LOSE")

    def test_lignes_plafonnees_par_ib(self):
        self.assertEqual(build_subscription("spike", rows=999).numberOfRows, IB_MAX_ROWS)

    def test_defaut_modeste_conforme_a_la_reserve_du_cours(self):
        # Le cours recommande trois a cinq alertes : un defaut a 50 produirait
        # l'effet inverse de celui recherche.
        self.assertLessEqual(DEFAULT_ROWS, 15)
        self.assertEqual(build_subscription("spike").numberOfRows, DEFAULT_ROWS)

    def test_actions_americaines(self):
        sub = build_subscription("spike")
        self.assertEqual(sub.instrument, "STK")
        self.assertEqual(sub.locationCode, "STK.US.MAJOR")


class TestFiltresAvances(unittest.TestCase):
    """Les valeurs viennent du XML du compte, pas d'une supposition."""

    def _tags(self, filters=None):
        return {t.tag: t.value for t in build_filters(filters)}

    def test_suspendus_ecartes_par_defaut(self):
        # R6 refuse un titre suspendu : lui laisser une des 50 lignes gaspille
        # la place d'un candidat que le crible aurait accepte.
        self.assertEqual(self._tags().get("haltedIs"), "false")

    def test_valeurs_booleennes_exactes(self):
        # Un libelle errone est accepte sans erreur par TWS et rend un balayage
        # vide, ce qui ressemble a un marche calme. La faute la plus dure a voir.
        tags = self._tags(ScanFilters(ssr_only=True, shortable_only=True))
        self.assertEqual(tags["shortSaleRestrictionIs"], "true")
        self.assertEqual(tags["unshortableIs"], "false")

    def test_plafond_de_flottant_importe_de_r6_et_r9(self):
        tags = self._tags(ScanFilters(low_float=True))
        attendu = str(int(min(LOW_FLOAT_CEILING, R9_FLOAT_CEILING)))
        self.assertEqual(tags["floatSharesBelow"], attendu)

    def test_piege_r9_filtre_la_restriction(self):
        # La configuration que R9 refuse : on la balaie pour savoir quels
        # titres NE PAS vendre a decouvert, pas pour les negocier.
        self.assertEqual(self._tags(R9_TRAP)["shortSaleRestrictionIs"], "true")

    def test_piege_r9_ne_demande_pas_le_flottant(self):
        # TWS a repondu « erreur 10360, floatSharesBelow is not allowed » sur le
        # compte de test, et un filtre refuse rend une liste VIDE, pas une
        # erreur. Le flottant est donc evalue en aval, par le crible.
        self.assertNotIn("floatSharesBelow", self._tags(R9_TRAP))

    def test_aucun_filtre_superflu(self):
        # Un filtre non demande ne doit pas apparaitre : chacun retranche des
        # candidats, et un filtre fantome vide le balayage sans le dire.
        self.assertNotIn("unshortableIs", self._tags())
        self.assertNotIn("shortSaleRestrictionIs", self._tags())
        self.assertNotIn("floatSharesBelow", self._tags())

    def test_variation_minimale(self):
        self.assertEqual(
            self._tags(ScanFilters(change_pct_above=20.0))["changePercAbove"], "20.0")

    def test_filtres_transmis_au_balayage(self):
        ib = FakeIB(scan_rows("CNET"))
        recu = {}

        def capture(subscription, options=None, filter_options=None):
            recu["tags"] = {t.tag: t.value for t in (filter_options or [])}
            return scan_rows("CNET")

        ib.reqScannerData = capture
        scanner(ib).scan("spike", filters=R9_TRAP)
        self.assertEqual(recu["tags"]["shortSaleRestrictionIs"], "true")


class TestCodesDeBalayage(unittest.TestCase):

    def test_les_gaps_d_ouverture_sont_disponibles(self):
        # Le moment ou la methode travaille reellement.
        self.assertEqual(build_subscription("gap").scanCode, "TOP_OPEN_PERC_GAIN")
        self.assertEqual(build_subscription("gap_haut").scanCode, "HIGH_OPEN_GAP")

    def test_le_classement_par_volume_relatif_existe(self):
        self.assertEqual(
            build_subscription("rvol").scanCode, "SCAN_stVolumeVsAvg5min_DESC")


class TestExtractionSymboles(unittest.TestCase):

    def test_ordre_du_rang_preserve(self):
        self.assertEqual(symbols_from_scan(scan_rows("CNET", "CDTG")), ["CNET", "CDTG"])

    def test_doublons_ecartes(self):
        self.assertEqual(symbols_from_scan(scan_rows("CNET", "CNET")), ["CNET"])

    def test_ligne_illisible_ecartee_pas_completee(self):
        rows = scan_rows("CNET")
        rows.append(FakeScanData(9, ""))
        self.assertEqual(symbols_from_scan(rows), ["CNET"])

    def test_reponse_vide_ne_leve_pas(self):
        self.assertEqual(symbols_from_scan(None), [])
        self.assertEqual(symbols_from_scan([]), [])


class TestBalayage(unittest.TestCase):

    def test_symboles_remontes(self):
        ib = FakeIB(scan_rows("CNET", "CDTG"))
        self.assertEqual(scanner(ib).scan("spike"), ["CNET", "CDTG"])

    def test_refus_de_tws_est_actionnable(self):
        ib = FakeIB(fail_scan=True)
        with self.assertRaises(ScanUnavailable) as capture:
            scanner(ib).scan("spike")
        self.assertIn("droits de donnees", str(capture.exception))

    def test_tws_ferme_remonte_en_scan_unavailable(self):
        ib = FakeIB(fail_connect=True)
        with self.assertRaises(ScanUnavailable):
            scanner(ib).scan("spike")


class TestSweep(unittest.TestCase):

    def test_origine_de_chaque_symbole_conservee(self):
        # Un titre remonte par deux balayages est plus interessant qu'un titre
        # remonte par un seul : l'information ne doit pas etre aplatie.
        ib = FakeIB({
            "TOP_PERC_GAIN": scan_rows("CNET", "CDTG"),
            "HOT_BY_VOLUME": scan_rows("CDTG", "UCL"),
        })
        origins = scanner(ib).sweep(("spike", "volume"))
        self.assertEqual(origins["CDTG"], ["spike", "volume"])
        self.assertEqual(origins["CNET"], ["spike"])
        self.assertEqual(origins["UCL"], ["volume"])

    def test_un_balayage_refuse_n_annule_pas_les_autres(self):
        class PartialIB(FakeIB):
            def reqScannerData(self, subscription, *args, **kwargs):
                if subscription.scanCode == "HOT_BY_VOLUME":
                    raise RuntimeError("code inconnu de ce compte")
                return scan_rows("CNET")

        origins = scanner(PartialIB()).sweep(("spike", "volume"))
        self.assertEqual(list(origins), ["CNET"])

    def test_tous_les_balayages_en_echec_ne_passe_pas_pour_un_marche_calme(self):
        # Distinction vitale : zero symbole parce que le marche est calme n'est
        # pas zero symbole parce que TWS est ferme. Confondre les deux, c'est
        # presenter une panne comme une seance sans candidat.
        ib = FakeIB(fail_connect=True)
        with self.assertRaises(ScanUnavailable) as capture:
            scanner(ib).sweep(("spike", "volume"))
        self.assertIn("aucun balayage n'a abouti", str(capture.exception))

    def test_marche_calme_rend_une_liste_vide_sans_lever(self):
        ib = FakeIB(results=[])
        self.assertEqual(scanner(ib).sweep(("spike", "volume")), {})

    def test_plafond_de_balayages_simultanes(self):
        trop = tuple(f"scan{i}" for i in range(IB_MAX_ACTIVE_SCANS + 1))
        with self.assertRaises(ScanUnavailable):
            scanner(FakeIB()).sweep(trop)


class TestScannerNeQualifiePas(unittest.TestCase):
    """Un symbole remonte n'est pas un candidat : le crible reste le juge."""

    def test_un_titre_remonte_peut_etre_refuse_par_r6(self):
        # Volume relatif insuffisant : le scanner filtre sur un volume ABSOLU,
        # le rapport au volume habituel ne se mesure qu'ici.
        snap = Snapshot(
            symbol="CNET", price=1.39, previous_close=1.40,
            volume=600_000, average_volume=1_000_000,
        )
        verdict = Verdict()
        check_filters(snap, verdict)
        self.assertTrue(verdict.blocks)
        self.assertTrue(any("volume relatif" in b for b in verdict.blocks))

    def test_un_titre_conforme_passe_le_filtre(self):
        snap = Snapshot(
            symbol="CDTG", price=1.83, previous_close=3.00,
            volume=8_000_000, average_volume=1_000_000,
        )
        verdict = Verdict()
        check_filters(snap, verdict)
        self.assertEqual(verdict.blocks, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
