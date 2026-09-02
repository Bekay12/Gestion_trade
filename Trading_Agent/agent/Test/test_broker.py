"""
test_broker - Suite hors ligne du pont courtier et des controles qu'il alimente.

    .venv\\Scripts\\python.exe agent\\Test\\test_broker.py

Le cas central est celui de la peremption : un instantane trop vieux doit etre
ignore plutot que servi. Un prix perime est plus dangereux qu'un prix absent,
parce qu'il a l'air exploitable.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import PaperPortfolio, RiskProfile, Snapshot, TradePlan, evaluate
from agent.sources.broker import BrokerBridge, from_ibkr_snapshot

NOW = datetime(2026, 8, 21, 10, 30)
PRIME = time(10, 30)

# Reponse fidele au format observe sur le connecteur.
IBKR_LIVE = {
    "volume": {"volume": 71513.0},
    "high": {"high": 0.0},
    "last": {"price": 1.71, "ts": 1787306780, "halted": False, "is_close": False},
    "low": {"low": 0.0},
    "top-status": {"status": "REALTIME"},
    "prior-close": {},
    "bid-ask": {"bid": 1.66, "bid_size": 680.0, "ask": 1.71, "ask_size": 413.0},
}
IBKR_STALE = {
    "volume": {"volume": 0.0},
    "last": {"price": 1.39, "is_close": True},
    "top-status": {"status": "REALTIME"},
    "prior-close": {"priorClose": 1.39},
    "bid-ask": {"bid": 1.31, "bid_size": 10.0, "ask": 1.45, "ask_size": 300.0},
}


def bridge(tmp) -> BrokerBridge:
    return BrokerBridge(path=Path(tmp) / "broker.json")


class TestIbkrTranslation(unittest.TestCase):
    """Traduction du format du connecteur vers le pont."""

    def test_live_quote_fields(self):
        row = from_ibkr_snapshot("CDTG", IBKR_LIVE)
        self.assertAlmostEqual(row["price"], 1.71)
        self.assertAlmostEqual(row["bid"], 1.66)
        self.assertAlmostEqual(row["ask"], 1.71)
        self.assertAlmostEqual(row["volume"], 71513.0)
        self.assertTrue(row["realtime"])
        self.assertFalse(row["quote_stale"])
        self.assertFalse(row["halted"])

    def test_close_price_is_flagged_stale(self):
        # is_close signale une cloture reprise, pas une transaction du jour.
        row = from_ibkr_snapshot("CNET", IBKR_STALE)
        self.assertTrue(row["quote_stale"])
        self.assertAlmostEqual(row["prior_close"], 1.39)
        self.assertEqual(row["volume"], 0.0)

    def test_hyphenated_keys_are_read(self):
        # Le connecteur repond en tirets la ou les champs demandes ont des
        # soulignes : lire la mauvaise forme rendrait tout None en silence.
        self.assertTrue(from_ibkr_snapshot("X", IBKR_LIVE)["realtime"])
        self.assertIsNotNone(from_ibkr_snapshot("X", IBKR_LIVE)["bid"])

    def test_missing_sections_do_not_raise(self):
        row = from_ibkr_snapshot("X", {})
        self.assertIsNone(row["price"])
        self.assertFalse(row["realtime"])


class TestBridgeIO(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = bridge(tmp)
            b.write({"CDTG": from_ibkr_snapshot("CDTG", IBKR_LIVE)}, now=NOW)
            quote = b.quote("cdtg", now=NOW)
        self.assertIsNotNone(quote)
        self.assertAlmostEqual(quote.price, 1.71)
        self.assertTrue(quote.realtime)

    def test_absent_symbol_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = bridge(tmp)
            b.write({"CDTG": from_ibkr_snapshot("CDTG", IBKR_LIVE)}, now=NOW)
            self.assertIsNone(b.quote("AAPL", now=NOW))

    def test_stale_snapshot_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = BrokerBridge(path=Path(tmp) / "b.json", max_age=timedelta(minutes=15))
            b.write({"CDTG": from_ibkr_snapshot("CDTG", IBKR_LIVE)}, now=NOW)
            self.assertIsNone(b.quote("CDTG", now=NOW + timedelta(minutes=45)))

    def test_fresh_within_window_is_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = BrokerBridge(path=Path(tmp) / "b.json", max_age=timedelta(minutes=15))
            b.write({"CDTG": from_ibkr_snapshot("CDTG", IBKR_LIVE)}, now=NOW)
            self.assertIsNotNone(b.quote("CDTG", now=NOW + timedelta(minutes=5)))

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(bridge(tmp).quote("CDTG"))

    def test_corrupt_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broker.json"
            path.write_text("{ pas du json", encoding="utf-8")
            self.assertIsNone(BrokerBridge(path=path).quote("CDTG"))

    def test_second_write_preserves_other_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = bridge(tmp)
            b.write({"CDTG": from_ibkr_snapshot("CDTG", IBKR_LIVE)}, now=NOW)
            b.write({"CNET": from_ibkr_snapshot("CNET", IBKR_STALE)}, now=NOW)
            self.assertIsNotNone(b.quote("CDTG", now=NOW))
            self.assertIsNotNone(b.quote("CNET", now=NOW))


class TestSpreadAndHalt(unittest.TestCase):
    """Les controles R6 que seule la donnee courtier rend possibles."""

    def _snap(self, **over) -> Snapshot:
        base = dict(
            symbol="ABCD", price=5.00, previous_close=4.80,
            volume=6_000_000, average_volume=1_000_000,
            market_cap=90_000_000, free_float=8_000_000,
            borrow="easy", ssr_active=False,
        )
        base.update(over)
        return Snapshot(**base)

    def _plan(self) -> TradePlan:
        return TradePlan(
            symbol="ABCD", direction="long", catalyst="8-K",
            entry=5.00, stop=4.80, targets=[5.60],
        )

    def _account(self):
        pf = PaperPortfolio(10_000)
        pf.start_day(NOW.date())
        return pf

    def test_spread_property(self):
        self.assertIsNone(self._snap().spread_pct)
        snap = self._snap(bid=4.90, ask=5.10)
        self.assertAlmostEqual(snap.spread_pct, 0.04, places=3)

    def test_wide_spread_blocks(self):
        # 1.31 / 1.45 releve en pre-marche : environ 10 %.
        snap = self._snap(bid=1.31, ask=1.45, price=1.39,
                          previous_close=1.39)
        v = evaluate(self._plan(), snap, RiskProfile(10_000), self._account(), PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("ecart" in b for b in v.blocks))

    def test_moderate_spread_only_warns(self):
        snap = self._snap(bid=4.95, ask=5.08)
        v = evaluate(self._plan(), snap, RiskProfile(10_000), self._account(), PRIME)
        self.assertTrue(v.allowed, v.blocks)
        self.assertTrue(any("ecart" in w for w in v.warnings))

    def test_tight_spread_is_silent(self):
        snap = self._snap(bid=4.995, ask=5.005)
        v = evaluate(self._plan(), snap, RiskProfile(10_000), self._account(), PRIME)
        self.assertTrue(v.allowed, v.blocks)
        self.assertFalse(any("ecart" in w for w in v.warnings))

    def test_halted_from_broker_blocks(self):
        # Champ auparavant code en dur a False : R6 ne rejetait jamais rien.
        v = evaluate(self._plan(), self._snap(halted=True), RiskProfile(10_000),
                     self._account(), PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("suspendu" in b for b in v.blocks))

    def test_stale_quote_warns_without_blocking(self):
        v = evaluate(self._plan(), self._snap(quote_stale=True), RiskProfile(10_000),
                     self._account(), PRIME)
        self.assertTrue(v.allowed, v.blocks)
        self.assertTrue(any("cloture" in w for w in v.warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
