"""
test_sources - Suite hors ligne des sources SSR et marche.

Aucun reseau : les appels HTTP et yfinance sont remplaces par des faux.

    .venv\\Scripts\\python.exe agent\\Test\\test_sources.py

Le cas central est celui du repli : quand le fichier du jour n'est pas encore
publie, le calendrier remonte d'un jour sans perdre de validite, parce qu'une
restriction declenchee la veille court encore. Et quand rien n'est trouvable,
il rend None — jamais False, qui ferait passer un titre pour non restreint.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.sources.broker import BrokerBridge, from_ibkr_snapshot
from agent.sources.market import MarketSource, MarketUnavailable
from agent.sources.ssr import SsrCalendar, SsrUnavailable, parse_file

DAY = date(2026, 8, 20)

# Extrait fidele au format publie : nom entre guillemets contenant une virgule,
# et derniere ligne d'horodatage de generation sans virgule.
SAMPLE = (
    "Symbol,Security Name,Market Category,Trigger Time\n"
    "CNET,ZW Data Action Techno,R,8/19/2026 9:30:00 AM\n"
    'BZAIW,"Blaize Holdings, Inc. W",R,8/20/2026 9:34:26 AM\n'
    "UCL,uCloudlink Group Inc. ADS,Q,8/20/2026 9:30:01 AM\n"
    "20260820163011\n"
)


class FakeResponse:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status


class TestParsing(unittest.TestCase):
    def test_extracts_symbols_and_skips_furniture(self):
        symbols = parse_file(SAMPLE)
        self.assertEqual(symbols, {"CNET", "BZAIW", "UCL"})

    def test_quoted_name_with_comma_does_not_shift_columns(self):
        self.assertIn("BZAIW", parse_file(SAMPLE))

    def test_generation_timestamp_line_is_not_a_symbol(self):
        self.assertNotIn("20260820163011", parse_file(SAMPLE))

    def test_header_is_not_a_symbol(self):
        self.assertNotIn("SYMBOL", parse_file(SAMPLE))

    def test_empty_file_yields_empty_set(self):
        self.assertEqual(parse_file(""), set())


class TestCalendar(unittest.TestCase):
    def _calendar(self, tmp):
        return SsrCalendar(cache_dir=Path(tmp))

    def test_active_symbol_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get", return_value=FakeResponse(SAMPLE)):
                self.assertTrue(cal.is_active("CNET", DAY))
                self.assertTrue(cal.is_active("cnet", DAY))     # casse indifferente

    def test_absent_symbol_is_false_not_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get", return_value=FakeResponse(SAMPLE)):
                self.assertIs(cal.is_active("AAPL", DAY), False)

    def test_falls_back_to_previous_day_when_file_missing(self):
        calls: list[str] = []

        def route(url, **kwargs):
            calls.append(url)
            # Le fichier du jour n'existe pas encore ; celui de la veille, si.
            return FakeResponse(SAMPLE) if "20260819" in url else FakeResponse("", 404)

        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get", side_effect=route):
                symbols, source = cal.active_symbols(DAY)
        self.assertEqual(source, date(2026, 8, 19))
        self.assertIn("CNET", symbols)
        self.assertGreaterEqual(len(calls), 2)

    def test_unavailable_raises_after_lookback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get", return_value=FakeResponse("", 404)):
                with self.assertRaises(SsrUnavailable):
                    cal.active_symbols(DAY)

    def test_is_active_returns_none_when_undeterminable(self):
        # None et False ne veulent pas dire la meme chose : None doit bloquer.
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get", return_value=FakeResponse("", 404)):
                self.assertIsNone(cal.is_active("CNET", DAY))

    def test_network_error_returns_none_not_false(self):
        import requests
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get",
                              side_effect=requests.RequestException("coupure")):
                self.assertIsNone(cal.is_active("CNET", DAY))

    def test_past_day_is_cached_and_not_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = self._calendar(tmp)
            with patch.object(cal.session, "get",
                              return_value=FakeResponse(SAMPLE)) as mock:
                cal.for_day(DAY)
                cal.for_day(DAY)
                self.assertEqual(mock.call_count, 1)


class FakeTicker:
    def __init__(self, info):
        self.info = info


def absent_bridge() -> BrokerBridge:
    """
    Pont pointant sur un fichier inexistant. Sans cette isolation, les tests
    liraient l'instantane courtier reel du projet et leurs fixtures seraient
    silencieusement ecrasees par des cours de marche.
    """
    return BrokerBridge(path=Path(tempfile.gettempdir()) / "_agent_absent" / "b.json")


def market_with(info, ssr_value=False, broker: BrokerBridge | None = None):
    """MarketSource dont yfinance, le calendrier SSR et le pont sont remplaces."""
    source = MarketSource(ssr=object(), broker=broker or absent_bridge())
    source.ssr = type("S", (), {"is_active": lambda self, s, d=None: ssr_value})()
    source._ticker_factory = lambda symbol: FakeTicker(info)
    return source


BASE_INFO = {
    "currentPrice": 1.39, "previousClose": 1.46,
    "regularMarketVolume": 2_500_000, "averageVolume": 457_987,
    "marketCap": 5_099_116, "floatShares": 3_026_160, "sharesShort": 144_612,
}


class TestMarketSource(unittest.TestCase):
    def test_snapshot_maps_fields(self):
        snap = market_with(BASE_INFO).snapshot("cnet")
        self.assertEqual(snap.symbol, "CNET")
        self.assertAlmostEqual(snap.price, 1.39)
        self.assertAlmostEqual(snap.previous_close, 1.46)
        self.assertAlmostEqual(snap.free_float, 3_026_160)
        self.assertAlmostEqual(snap.shares_short, 144_612)
        self.assertIs(snap.ssr_active, False)

    def test_borrow_stays_none_unless_supplied(self):
        # Aucune source publique : le champ doit rester vide par defaut.
        self.assertIsNone(market_with(BASE_INFO).snapshot("CNET").borrow)

    def test_borrow_is_passed_through(self):
        snap = market_with(BASE_INFO).snapshot("CNET", borrow="hard", borrow_rate=0.85)
        self.assertEqual(snap.borrow, "hard")
        self.assertAlmostEqual(snap.borrow_rate, 0.85)

    def test_missing_float_is_none_not_shares_outstanding(self):
        # La substitution fausserait R6, R8 et R9 : elle est interdite.
        info = dict(BASE_INFO, sharesOutstanding=3_668_429)
        info.pop("floatShares")
        self.assertIsNone(market_with(info).snapshot("CNET").free_float)

    def test_missing_average_volume_raises(self):
        info = {k: v for k, v in BASE_INFO.items() if k != "averageVolume"}
        with self.assertRaises(MarketUnavailable):
            market_with(info).snapshot("CNET")

    def test_missing_price_raises(self):
        with self.assertRaises(MarketUnavailable):
            market_with({"averageVolume": 100_000}).snapshot("CNET")

    def test_ssr_none_is_propagated(self):
        source = market_with(BASE_INFO)
        source.ssr = type("S", (), {"is_active": lambda self, s, d=None: None})()
        self.assertIsNone(source.snapshot("CNET").ssr_active)

    def test_broker_quote_overrides_public_data(self):
        # Le courtier prime : prix temps reel, volume comptant le pre-marche,
        # drapeau de suspension et haut du carnet, qu'aucune source publique
        # ne donne. C'est ce qui a corrige un RVOL faux calcule sur le volume
        # de la seance precedente.
        with tempfile.TemporaryDirectory() as tmp:
            bridge = BrokerBridge(path=Path(tmp) / "b.json")
            bridge.write({"CNET": from_ibkr_snapshot("CNET", {
                "volume": {"volume": 0.0},
                "last": {"price": 1.39, "is_close": True, "halted": False},
                "top-status": {"status": "REALTIME"},
                "prior-close": {"priorClose": 1.39},
                "bid-ask": {"bid": 1.31, "bid_size": 10.0,
                            "ask": 1.45, "ask_size": 300.0},
            })})
            snap = market_with(BASE_INFO, broker=bridge).snapshot("CNET")

        self.assertAlmostEqual(snap.previous_close, 1.39)   # et non 1.46
        self.assertEqual(snap.volume, 0.0)                  # vrai zero de pre-marche
        self.assertTrue(snap.realtime)
        self.assertTrue(snap.quote_stale)
        self.assertAlmostEqual(snap.spread_pct, 0.1014, places=3)
        # Le flottant reste public : le courtier ne le fournit pas.
        self.assertAlmostEqual(snap.free_float, 3_026_160)

    def test_screen_skips_failures_without_aborting(self):
        source = MarketSource(ssr=object(), broker=absent_bridge())
        source.ssr = type("S", (), {"is_active": lambda self, s, d=None: False})()

        def factory(symbol):
            if symbol == "BAD":
                raise RuntimeError("indisponible")
            return FakeTicker(BASE_INFO)

        source._ticker_factory = factory
        snaps = source.screen(["GOOD", "BAD", "ALSO"])
        self.assertEqual([s.symbol for s in snaps], ["GOOD", "ALSO"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
