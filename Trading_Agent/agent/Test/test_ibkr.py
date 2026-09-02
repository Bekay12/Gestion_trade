"""
test_ibkr - Suite hors ligne du producteur TWS/Gateway.

    .venv\\Scripts\\python.exe agent\\Test\\test_ibkr.py

Ni TWS, ni Gateway, ni compte, ni reseau : le client ib_async est remplace par
un faux qui rend les memes objets. Ce qui est verifie ici n'est pas la
plomberie du protocole mais les trois traductions qui, fausses, passeraient
inapercues : le multiplicateur 100 du volume americain, les seuils du tick 46,
et le fait qu'une donnee absente reste None au lieu de devenir zero.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import PaperPortfolio, RiskProfile, Snapshot, TradePlan, evaluate
from agent.sources.broker import BrokerBridge
from agent.sources.ib_ticks import MAX_PLAUSIBLE_VOLUME, session_volume
from agent.sources.ibkr import detect_port, probe_port
from agent.sources.ibkr import (
    IbkrConnector,
    IbkrSettings,
    IbkrUnavailable,
    borrow_from_shortable,
    halted_from_tick,
)

NAN = float("nan")
NOW = datetime(2026, 8, 21, 10, 30)
PRIME = time(10, 30)


class FakeTicker:
    """Ticker ib_async minimal : les champs reellement lus par le connecteur."""

    def __init__(self, **fields):
        defaults = dict(
            last=NAN, close=NAN, bid=NAN, ask=NAN, bidSize=NAN, askSize=NAN,
            volume=NAN, halted=NAN, shortable=NAN, shortableShares=NAN,
            marketDataType=1,
        )
        defaults.update(fields)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeBar:
    def __init__(self, close):
        self.close = close


class FakeIB:
    """Client ib_async simule. Enregistre ce qu'on lui demande."""

    def __init__(self, ticker=None, bars=None, fail_connect=False, fail_hist=False):
        self.ticker = ticker or FakeTicker()
        self.bars = bars if bars is not None else [FakeBar(85.0)]
        self.fail_connect = fail_connect
        self.fail_hist = fail_hist
        self.connected = False
        self.market_data_type = None
        self.generic_ticks = None
        self.cancelled = []
        self.hist_calls = []

    def isConnected(self):
        return self.connected

    def qualifyContracts(self, *contracts):
        # TWS renseigne le conId ; sans lui ib_async refuse le contrat.
        for c in contracts:
            c.conId = 1234
        return list(contracts)

    def connect(self, host, port, clientId, timeout):
        if self.fail_connect:
            raise ConnectionRefusedError("port ferme")
        self.connected = True

    def disconnect(self):
        self.connected = False

    def reqMarketDataType(self, kind):
        self.market_data_type = kind

    def reqMktData(self, contract, generic, snapshot, regulatory):
        self.generic_ticks = generic
        return self.ticker

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)

    def reqHistoricalData(self, contract, **kwargs):
        self.hist_calls.append(kwargs)
        if self.fail_hist:
            raise RuntimeError("pas de SLB pour ce titre")
        return self.bars

    def sleep(self, seconds):
        return None


def connector(ib):
    return IbkrConnector(IbkrSettings(client_id=99), ib=ib)


class TestStatutEmprunt(unittest.TestCase):
    """Tick 46 : les seuils viennent de la documentation TWS, pas d'un choix."""

    def test_au_dessus_de_2_5_est_facile(self):
        self.assertEqual(borrow_from_shortable(3.0), "easy")

    def test_entre_1_5_et_2_5_est_difficile(self):
        # Le locate est requis : c'est exactement le "hard to borrow" de R9.
        self.assertEqual(borrow_from_shortable(2.0), "hard")

    def test_seuil_2_5_exclu_reste_difficile(self):
        self.assertEqual(borrow_from_shortable(2.5), "hard")

    def test_sous_1_5_est_non_empruntable(self):
        self.assertEqual(borrow_from_shortable(1.0), "none")

    def test_tick_absent_reste_inconnu(self):
        # Le point le plus important du module : ni "easy", ni "none".
        self.assertIsNone(borrow_from_shortable(NAN))
        self.assertIsNone(borrow_from_shortable(None))


class TestSuspension(unittest.TestCase):
    """Tick 49 : -1 signifie 'je ne sais pas', pas 'non suspendu'."""

    def test_suspension_reglementaire(self):
        self.assertEqual(halted_from_tick(1.0), (True, True))

    def test_suspension_de_volatilite(self):
        self.assertEqual(halted_from_tick(2.0), (True, True))

    def test_non_suspendu_est_une_information(self):
        self.assertEqual(halted_from_tick(0.0), (False, True))

    def test_moins_un_est_une_ignorance_signalee(self):
        self.assertEqual(halted_from_tick(-1.0), (False, False))

    def test_tick_absent_est_une_ignorance_signalee(self):
        self.assertEqual(halted_from_tick(NAN), (False, False))
        self.assertEqual(halted_from_tick(None), (False, False))


class TestTraductionCotation(unittest.TestCase):

    def test_volume_en_actions_n_est_pas_remultiplie(self):
        # Defaut moderne : TWS 985+ envoie des ACTIONS quand « Send volumes in
        # lots » est decoche. Remultiplier par cent rendrait le RVOL de R6 cent
        # fois trop eleve, et le crible accepterait ce qu'il doit refuser.
        ib = FakeIB(FakeTicker(last=1.71, volume=71500.0))
        row = connector(ib).quote_row(ib.ticker, "CDTG")
        self.assertEqual(row["volume"], 71500.0)

    def test_mode_lots_historique_reste_possible(self):
        # TWS configure a l'ancienne : le multiplicateur se declare, il ne se
        # devine pas — l'API ne rapporte pas ce reglage.
        from agent.sources.ibkr import LOT_VOLUME_MULTIPLIER
        ib = FakeIB(FakeTicker(last=1.71, volume=715.0))
        c = IbkrConnector(
            IbkrSettings(client_id=99, us_volume_multiplier=LOT_VOLUME_MULTIPLIER),
            ib=ib,
        )
        self.assertEqual(c.quote_row(ib.ticker, "CDTG")["volume"], 71500.0)

    def test_le_defaut_se_trompe_du_cote_du_refus(self):
        # Invariant de conception : le defaut ne doit jamais GONFLER le volume.
        from agent.sources.ibkr import US_VOLUME_MULTIPLIER
        self.assertEqual(US_VOLUME_MULTIPLIER, 1)

    def test_sans_transaction_le_prix_retombe_sur_la_cloture(self):
        ib = FakeIB(FakeTicker(last=NAN, close=1.39))
        row = connector(ib).quote_row(ib.ticker, "CNET")
        self.assertEqual(row["price"], 1.39)
        self.assertTrue(row["quote_stale"])

    def test_transaction_reelle_n_est_pas_perimee(self):
        ib = FakeIB(FakeTicker(last=1.71, close=1.80))
        row = connector(ib).quote_row(ib.ticker, "CDTG")
        self.assertEqual(row["price"], 1.71)
        self.assertFalse(row["quote_stale"])
        self.assertEqual(row["prior_close"], 1.80)

    def test_flux_differe_ne_se_declare_pas_temps_reel(self):
        ib = FakeIB(FakeTicker(last=1.71, marketDataType=3))
        row = connector(ib).quote_row(ib.ticker, "CDTG")
        self.assertFalse(row["realtime"])

    def test_champs_absents_restent_none_jamais_zero(self):
        ib = FakeIB(FakeTicker(last=1.71))
        row = connector(ib).quote_row(ib.ticker, "CDTG")
        for champ in ("bid", "ask", "bid_size", "ask_size", "volume", "borrow"):
            self.assertIsNone(row[champ], champ)

    def test_carnet_complet_traduit(self):
        ib = FakeIB(FakeTicker(
            last=1.71, bid=1.66, ask=1.71, bidSize=680.0, askSize=413.0,
            halted=0.0, shortable=2.0, shortableShares=12000.0,
        ))
        row = connector(ib).quote_row(ib.ticker, "CDTG", rate=0.85)
        self.assertEqual((row["bid"], row["ask"]), (1.66, 1.71))
        self.assertEqual(row["borrow"], "hard")
        self.assertEqual(row["borrow_rate"], 0.85)
        self.assertEqual(row["shortable_shares"], 12000.0)
        self.assertFalse(row["halted"])


class TestTauxEmprunt(unittest.TestCase):
    """R7 : le taux est la donnee que la documentation croyait hors de portee."""

    def test_pourcentage_converti_en_fraction(self):
        # IB rend 85.0 pour 85 %/an ; le moteur raisonne en fraction.
        ib = FakeIB(bars=[FakeBar(85.0)])
        self.assertAlmostEqual(connector(ib).borrow_rate("CDTG"), 0.85, places=6)

    def test_serie_demandee_en_fee_rate(self):
        ib = FakeIB()
        connector(ib).borrow_rate("CDTG")
        self.assertEqual(ib.hist_calls[0]["whatToShow"], "FEE_RATE")

    def test_serie_vide_reste_inconnue(self):
        ib = FakeIB(bars=[])
        self.assertIsNone(connector(ib).borrow_rate("CDTG"))

    def test_titre_sans_pret_ne_leve_pas(self):
        ib = FakeIB(fail_hist=True)
        self.assertIsNone(connector(ib).borrow_rate("CDTG"))


class TestConnexion(unittest.TestCase):

    def test_port_ferme_donne_un_message_actionnable(self):
        ib = FakeIB(fail_connect=True)
        with self.assertRaises(IbkrUnavailable) as capture:
            connector(ib).connect()
        self.assertIn("API activee", str(capture.exception))

    def test_le_direct_est_demande_explicitement(self):
        ib = FakeIB()
        connector(ib).connect()
        self.assertEqual(ib.market_data_type, 1)

    def test_tick_generique_236_demande(self):
        # Sans lui, ni statut d'emprunt ni actions empruntables.
        ib = FakeIB(FakeTicker(last=1.71))
        connector(ib).quotes(["CDTG"], with_rate=False)
        self.assertEqual(ib.generic_ticks, "236")

    def test_abonnement_annule_apres_lecture(self):
        ib = FakeIB(FakeTicker(last=1.71))
        connector(ib).quotes(["CDTG"], with_rate=False)
        self.assertEqual(ib.cancelled, ["CDTG"])


class TestPontEtMoteur(unittest.TestCase):
    """Le producteur change, le moteur ne bouge pas."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.bridge = BrokerBridge(path=Path(self.dir.name) / "snap.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_emprunt_traverse_le_pont(self):
        ib = FakeIB(FakeTicker(last=1.71, shortable=2.0), bars=[FakeBar(85.0)])
        connector(ib).publish(["CDTG"], bridge=self.bridge, now=NOW)

        quote = self.bridge.quote("CDTG", now=NOW)
        self.assertEqual(quote.borrow, "hard")
        self.assertAlmostEqual(quote.borrow_rate, 0.85, places=6)

    def test_le_crible_arme_un_short_que_l_emprunt_debloque(self):
        # Sans statut d'emprunt, R7 refuse. Avec celui du courtier, le plan
        # passe : c'est le verrou que ce module leve.
        plan = TradePlan(
            symbol="CDTG", direction="short", catalyst="Chute de 39 %",
            entry=1.83, stop=2.05, targets=[1.30],
        )
        commun = dict(
            symbol="CDTG", price=1.83, previous_close=3.00,
            volume=8_000_000, average_volume=1_000_000,
            free_float=30_000_000, ssr_active=False,
        )
        profile = RiskProfile(capital=10_000)

        sans = evaluate(plan, Snapshot(borrow=None, **commun), profile,
                        PaperPortfolio(10_000), now=PRIME)
        self.assertFalse(sans.allowed)
        self.assertTrue(any("R7" in motif for motif in sans.blocks))

        avec = evaluate(plan, Snapshot(borrow="hard", borrow_rate=0.85, **commun),
                        profile, PaperPortfolio(10_000), now=PRIME)
        self.assertTrue(avec.allowed, avec.blocks)
        self.assertGreater(avec.size, 0)


class TestVolumeAberrant(unittest.TestCase):
    """Un Gateway reel a renvoye 48 591 578 764 254 pour AAPL le 2026-08-23."""

    def test_volume_en_virgule_fixe_est_decode(self):
        # Mesure du 2026-08-23 : TWS envoie lui-meme cette valeur sur le tick 74.
        # Divisee par 10^6 elle donne 48,6 M, du meme ordre que les 42,2 M de la
        # source publique. Corrobore sur cinq titres, trois ordres de grandeur.
        self.assertAlmostEqual(
            session_volume(48_591_578_764_254.0, 1, "AAPL"), 48_591_578.764254,
            places=3)

    def test_le_decodage_ne_s_applique_pas_a_une_valeur_saine(self):
        # Une valeur deja plausible ne doit jamais etre mise a l'echelle : ce
        # serait une inference silencieuse, dans le sens permissif.
        self.assertEqual(session_volume(42_216_056.0, 1, "AAPL"), 42_216_056.0)

    def test_valeur_absurde_meme_decodee_reste_absente(self):
        # Le decodage n'est pas un blanc-seing : hors fenetre, on refuse.
        self.assertIsNone(session_volume(1e20, 1, "X"))

    def test_volume_plausible_passe(self):
        self.assertEqual(session_volume(48_591_578.0, 1, "AAPL"), 48_591_578.0)

    def test_le_plafond_est_une_absurdite_pas_un_filtre(self):
        # Un penny stock tres actif doit passer : le garde-fou vise la donnee
        # corrompue, pas le titre extreme.
        self.assertEqual(session_volume(3_000_000_000.0, 1), 3_000_000_000.0)
        self.assertLess(3_000_000_000.0, MAX_PLAUSIBLE_VOLUME)

    def test_mode_lots_franchit_le_plafond_et_est_decode(self):
        # 1e9 lots = 1e11 actions, impossible ; le decodage rend 100 000.
        self.assertEqual(session_volume(1e9, 100, "X"), 100_000.0)

    def test_la_ligne_de_cotation_porte_le_decodage(self):
        ib = FakeIB(FakeTicker(last=1.71, volume=48_591_578_764_254.0))
        volume = connector(ib).quote_row(ib.ticker, "AAPL")["volume"]
        self.assertAlmostEqual(volume, 48_591_578.764254, places=3)


class TestDetectionPort(unittest.TestCase):
    """Une erreur de port produit le meme message qu'un logiciel ferme."""

    @staticmethod
    def _fausse_sonde(predicat):
        # Patcher ib_ports, ou la fonction est DEFINIE : la remplacer dans
        # ibkr.py ne toucherait que la re-export, et le vrai socket repondrait.
        import agent.sources.ib_ports as mod
        return mod, mod.probe_port, predicat

    def test_aucun_port_ouvert_rend_none(self):
        import agent.sources.ib_ports as mod
        vrai = mod.probe_port
        mod.probe_port = lambda host, port, timeout=0.6: False
        try:
            self.assertIsNone(detect_port())
        finally:
            mod.probe_port = vrai

    def test_le_premier_port_en_ecoute_est_choisi(self):
        import agent.sources.ib_ports as mod
        vrai = mod.probe_port
        mod.probe_port = lambda host, port, timeout=0.6: port == 4002
        try:
            self.assertEqual(detect_port(), 4002)
        finally:
            mod.probe_port = vrai

    def test_la_sonde_ne_touche_pas_le_reseau_reel(self):
        # Garde-fou d'isolation : un test qui sonde vraiment la machine passe
        # ou echoue selon que Gateway tourne. C'est arrive le 2026-08-23.
        import agent.sources.ib_ports as mod
        appels = []
        vrai = mod.probe_port
        mod.probe_port = lambda host, port, timeout=0.6: appels.append(port) or False
        try:
            detect_port()
        finally:
            mod.probe_port = vrai
        self.assertEqual(appels, [7497, 4002, 7496, 4001])


class TestRepliDiffere(unittest.TestCase):
    """Sans abonnement, le direct ne rend RIEN — pas une version degradee."""

    def test_repli_automatique_quand_aucun_prix_en_direct(self):
        class SansDirect(FakeIB):
            def __init__(self):
                super().__init__(FakeTicker())
                self.types = []

            def reqMarketDataType(self, kind):
                self.types.append(kind)
                # Le differe, lui, repond.
                if kind == 3:
                    self.ticker = FakeTicker(last=309.69, marketDataType=3)

        ib = SansDirect()
        rows = connector(ib).quotes(["AAPL"], with_rate=False)
        self.assertIn(3, ib.types)
        self.assertEqual(rows["AAPL"]["price"], 309.69)
        self.assertFalse(rows["AAPL"]["realtime"])

    def test_pas_de_repli_si_le_direct_repond(self):
        ib = FakeIB(FakeTicker(last=1.71, marketDataType=1))
        c = connector(ib)
        c.quotes(["CDTG"], with_rate=False)
        self.assertNotIn(3, [t for t in getattr(ib, "types", [])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
