"""
test_agent - Suite hors ligne du moteur de decision et du portefeuille papier.

Aucun reseau, aucune cle, aucun chargement de modele. Lancer avec le Python du
projet :

    .venv\\Scripts\\python.exe agent\\Test\\test_agent.py

Les cas les plus importants sont ceux de R9 et des donnees manquantes : ils
verifient que le moteur REFUSE au lieu d'avertir, ce qui est la raison d'etre
de tout le module (voir docu/methode/05-specification-agent.md).
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import PaperPortfolio, RiskProfile, Snapshot, TradePlan, evaluate
from agent import rules

NOW = datetime(2026, 8, 21, 10, 30)
PRIME = time(10, 30)


def snapshot(**over) -> Snapshot:
    """Etat marche par defaut qui passe tous les filtres R6."""
    base = dict(
        symbol="ABCD", price=5.00, previous_close=4.80,
        volume=6_000_000, average_volume=1_000_000,
        market_cap=90_000_000, free_float=8_000_000,
        shares_short=400_000, borrow="easy", borrow_rate=0.05,
        ssr_active=False, vwap=4.90, halted=False,
    )
    base.update(over)
    return Snapshot(**base)


def plan(**over) -> TradePlan:
    """Plan acheteur par defaut, rapport gain/risque 3:1."""
    base = dict(
        symbol="ABCD", direction="long", catalyst="Contrat annonce en 8-K",
        entry=5.00, stop=4.80, targets=[5.60], invalidation="424B depose",
    )
    base.update(over)
    return TradePlan(**base)


def account(capital=10_000.0) -> PaperPortfolio:
    pf = PaperPortfolio(capital)
    pf.start_day(NOW.date())
    return pf


class TestRules(unittest.TestCase):
    """Calculs purs : R1, R2, R4, R7, R8."""

    def test_position_size_derives_from_risk_and_stop(self):
        # R1 : 1 % de 10 000 = 100 risques ; 0,20 par action -> 500 actions.
        risk = rules.risk_amount(10_000, 0.01)
        self.assertEqual(risk, 100.0)
        self.assertEqual(rules.position_size(risk, 5.00, 4.80), 500)

    def test_position_size_rejects_zero_distance(self):
        with self.assertRaises(ValueError):
            rules.position_size(100, 5.0, 5.0)

    def test_risk_amount_rejects_absurd_fraction(self):
        with self.assertRaises(ValueError):
            rules.risk_amount(10_000, 0.5)

    def test_risk_reward_symmetric_for_short(self):
        # Vendeuse : entree 5, stop 5.20, cible 4.40 -> 0.60 / 0.20 = 3.
        self.assertAlmostEqual(rules.risk_reward(5.0, 5.2, 4.4), 3.0, places=2)

    def test_breakeven_win_rate(self):
        # R2 : un rapport 1:3 est a l'equilibre a 25 % de reussite.
        self.assertAlmostEqual(rules.breakeven_win_rate(3.0), 0.25, places=3)
        self.assertAlmostEqual(rules.breakeven_win_rate(1.0), 0.50, places=3)

    def test_sizing_ladder(self):
        self.assertEqual(rules.sizing_pct(3), 0.005)
        self.assertEqual(rules.sizing_pct(12), 0.01)
        self.assertEqual(rules.sizing_pct(24), 0.02)
        # Le palier professionnel exige un journal tenu.
        self.assertEqual(rules.sizing_pct(48, has_journal=True), 0.03)
        self.assertEqual(rules.sizing_pct(48, has_journal=False), 0.02)

    def test_drawdown_ladder(self):
        self.assertEqual(rules.drawdown_scale(0.03), 1.00)
        self.assertEqual(rules.drawdown_scale(0.07), 0.75)
        self.assertEqual(rules.drawdown_scale(0.12), 0.50)
        self.assertEqual(rules.drawdown_scale(0.18), 0.00)
        self.assertEqual(rules.drawdown_scale(0.30), 0.00)

    def test_borrow_cost_accrues_per_day(self):
        # R7 : 10 000 a 50 %/an ~ 13,70 par jour.
        self.assertAlmostEqual(rules.borrow_cost(10_000, 0.50, 1), 13.70, places=2)
        self.assertAlmostEqual(rules.borrow_cost(10_000, 0.50, 30), 410.96, places=2)

    def test_ssr_threshold_is_ten_percent(self):
        self.assertTrue(rules.ssr_triggered(0.90, 1.00))
        self.assertFalse(rules.ssr_triggered(0.91, 1.00))

    def test_short_interest_uses_float_not_outstanding(self):
        self.assertAlmostEqual(rules.short_interest_pct(2_000_000, 10_000_000), 0.20)

    def test_float_rotation_and_rvol(self):
        self.assertAlmostEqual(rules.float_rotation(23_649_207, 9_370_000), 2.52, places=2)
        self.assertAlmostEqual(rules.relative_volume(6_000_000, 1_000_000), 6.0)


class TestGateHappyPath(unittest.TestCase):
    """Un plan conforme doit passer et porter une taille calculee."""

    def test_valid_long_is_armed(self):
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), account(), PRIME)
        self.assertTrue(v.allowed, v.blocks)
        self.assertEqual(v.size, 500)
        self.assertAlmostEqual(v.risk_reward, 3.0, places=2)

    def test_valid_short_is_armed_when_data_present(self):
        v = evaluate(
            plan(direction="short", entry=5.00, stop=5.20, targets=[4.40]),
            snapshot(borrow="easy", ssr_active=False, shares_short=100_000),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertTrue(v.allowed, v.blocks)


class TestGateRefusals(unittest.TestCase):
    """Chaque regle bloquante doit produire un refus, pas un avertissement."""

    def test_missing_borrow_status_blocks_short(self):
        # Donnee non diffusee gratuitement : son absence doit bloquer.
        v = evaluate(
            plan(direction="short", entry=5.0, stop=5.2, targets=[4.4]),
            snapshot(borrow=None),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("R7" in b for b in v.blocks))

    def test_missing_ssr_status_blocks_short(self):
        v = evaluate(
            plan(direction="short", entry=5.0, stop=5.2, targets=[4.4]),
            snapshot(ssr_active=None),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("R9" in b for b in v.blocks))

    def test_missing_data_does_not_block_long(self):
        # Les contraintes d'emprunt ne concernent que la vente a decouvert.
        v = evaluate(
            plan(), snapshot(borrow=None, ssr_active=None),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertTrue(v.allowed, v.blocks)

    def test_r9_trap_configuration_blocks_short(self):
        # Rule 201 + difficile a emprunter + flottant reduit + interet eleve.
        v = evaluate(
            plan(direction="short", entry=5.0, stop=5.2, targets=[4.4]),
            snapshot(ssr_active=True, borrow="hard", free_float=6_000_000,
                     shares_short=1_500_000),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("R9" in b and "piege" in b for b in v.blocks))

    def test_r9_two_conditions_warns_but_allows(self):
        v = evaluate(
            plan(direction="short", entry=5.0, stop=5.2, targets=[4.4]),
            snapshot(ssr_active=True, borrow="hard", free_float=50_000_000,
                     shares_short=100_000),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertTrue(v.allowed, v.blocks)
        self.assertTrue(any("R9" in w for w in v.warnings))

    def test_no_borrow_available_blocks(self):
        v = evaluate(
            plan(direction="short", entry=5.0, stop=5.2, targets=[4.4]),
            snapshot(borrow="none"),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)

    def test_daily_loss_limit_blocks(self):
        pf = account()
        pf.cash -= 400          # -4 % sur 10 000, au-dela des 3 %
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), pf, PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("R3" in b for b in v.blocks))

    def test_consecutive_losses_block(self):
        pf = account()
        for i in range(3):
            pf.open_position(f"X{i}", "long", 10, 5.0, 4.8, NOW)
            pf.close_position(f"X{i}", 4.5, NOW, reason="stop")
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), pf, PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("R12" in b for b in v.blocks))

    def test_poor_risk_reward_blocks(self):
        v = evaluate(
            plan(targets=[5.20]),      # 0.20 / 0.20 = 1:1
            snapshot(), RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("R2" in b for b in v.blocks))

    def test_missing_catalyst_blocks(self):
        v = evaluate(plan(catalyst="   "), snapshot(), RiskProfile(10_000), account(), PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("R11" in b for b in v.blocks))

    def test_low_relative_volume_blocks(self):
        v = evaluate(
            plan(), snapshot(volume=1_200_000, average_volume=1_000_000),
            RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("volume relatif" in b for b in v.blocks))

    def test_price_out_of_band_blocks(self):
        v = evaluate(
            plan(entry=45.0, stop=44.0, targets=[48.0]),
            snapshot(price=45.0), RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)

    def test_halted_blocks(self):
        v = evaluate(plan(), snapshot(halted=True), RiskProfile(10_000), account(), PRIME)
        self.assertFalse(v.allowed)

    def test_premarket_blocks(self):
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), account(), time(8, 0))
        self.assertFalse(v.allowed)
        self.assertTrue(any("R10" in b for b in v.blocks))

    def test_first_five_minutes_block(self):
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), account(), time(9, 32))
        self.assertFalse(v.allowed)

    def test_stop_on_wrong_side_blocks(self):
        v = evaluate(
            plan(stop=5.20, targets=[5.60]),   # acheteuse, stop au-dessus
            snapshot(), RiskProfile(10_000), account(), PRIME,
        )
        self.assertFalse(v.allowed)
        self.assertTrue(any("R5" in b for b in v.blocks))

    def test_symbol_mismatch_blocks(self):
        v = evaluate(plan(symbol="ZZZZ"), snapshot(), RiskProfile(10_000), account(), PRIME)
        self.assertFalse(v.allowed)

    def test_drawdown_beyond_twenty_percent_blocks(self):
        pf = account()
        pf.peak_equity = 10_000
        pf.cash = 7_500                     # -25 %
        pf._day_start_equity = 7_500        # neutralise R3 pour isoler R4
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), pf, PRIME)
        self.assertFalse(v.allowed)
        self.assertTrue(any("R4" in b for b in v.blocks))

    def test_drawdown_scales_size_down(self):
        pf = account()
        pf.peak_equity = 10_000
        pf.cash = 9_200                     # -8 % -> coefficient 0,75
        pf._day_start_equity = 9_200
        v = evaluate(plan(), snapshot(), RiskProfile(10_000), pf, PRIME)
        self.assertTrue(v.allowed, v.blocks)
        self.assertEqual(v.size, 375)       # 500 * 0,75


class TestPortfolio(unittest.TestCase):
    """Executions simulees, journal, grandeurs de suivi."""

    def test_long_profit_after_slippage_and_fees(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=1.0)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "long", 100, 5.00, 4.80, NOW)
        trade = pf.close_position("ABCD", 5.50, NOW)
        self.assertAlmostEqual(trade.pnl, 100 * 0.50 - 1.0, places=2)
        self.assertAlmostEqual(pf.cash, 10_000 + 49.0 - 1.0, places=2)

    def test_slippage_is_always_adverse(self):
        pf = PaperPortfolio(10_000, slippage=0.01, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pos = pf.open_position("ABCD", "long", 100, 5.00, 4.50, NOW)
        self.assertGreater(pos.entry, 5.00)          # achat plus cher
        trade = pf.close_position("ABCD", 5.00, NOW)
        self.assertLess(trade.exit_price, 5.00)      # vente moins chere
        self.assertLess(trade.pnl, 0)                # aller-retour perdant

    def test_short_profit_and_borrow_cost(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "short", 1000, 5.00, 5.20, NOW, borrow_rate=0.73)
        trade = pf.close_position("ABCD", 4.50, datetime(2026, 8, 22, 10, 0))
        # Gain brut 500 ; emprunt 5000 a 73 %/an sur 1 jour ~ 10.
        self.assertAlmostEqual(trade.fees, 5000 * 0.73 / 365, places=2)
        self.assertAlmostEqual(trade.pnl, 500 - trade.fees, places=2)

    def test_stop_is_applied_when_touched(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "long", 100, 5.00, 4.80, NOW)
        closed = pf.apply_stops({"ABCD": 4.70}, NOW)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].reason, "stop")
        # Execute au stop, pas au dernier cours.
        self.assertAlmostEqual(closed[0].exit_price, 4.80, places=2)

    def test_stop_not_applied_when_untouched(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "long", 100, 5.00, 4.80, NOW)
        self.assertEqual(pf.apply_stops({"ABCD": 4.90}, NOW), [])

    def test_short_stop_triggers_above(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "short", 100, 5.00, 5.20, NOW)
        self.assertEqual(len(pf.apply_stops({"ABCD": 5.30}, NOW)), 1)

    def test_duplicate_position_refused(self):
        pf = PaperPortfolio(10_000)
        pf.start_day(NOW.date())
        pf.open_position("ABCD", "long", 100, 5.0, 4.8, NOW)
        with self.assertRaises(ValueError):
            pf.open_position("ABCD", "long", 50, 5.0, 4.8, NOW)

    def test_drawdown_tracks_peak(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("A", "long", 100, 5.0, 4.0, NOW)
        pf.close_position("A", 15.0, NOW)          # +1000 -> sommet 11 000
        self.assertAlmostEqual(pf.peak_equity, 11_000, places=2)
        pf.open_position("B", "long", 100, 5.0, 4.0, NOW)
        pf.close_position("B", 0.5, NOW)           # -450
        self.assertAlmostEqual(pf.drawdown, 450 / 11_000, places=4)

    def test_consecutive_losses_reset_on_win(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        for i in range(2):
            pf.open_position(f"L{i}", "long", 10, 5.0, 4.0, NOW)
            pf.close_position(f"L{i}", 4.0, NOW)
        self.assertEqual(pf.consecutive_losses, 2)
        pf.open_position("W", "long", 10, 5.0, 4.0, NOW)
        pf.close_position("W", 6.0, NOW)
        self.assertEqual(pf.consecutive_losses, 0)

    def test_daily_pnl_resets_on_new_day(self):
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("A", "long", 100, 5.0, 4.0, NOW)
        pf.close_position("A", 4.0, NOW)
        self.assertAlmostEqual(pf.daily_pnl, -100, places=2)
        pf.start_day(datetime(2026, 8, 22).date())
        self.assertAlmostEqual(pf.daily_pnl, 0.0, places=2)

    def test_stats_and_save_roundtrip(self):
        import json, tempfile
        pf = PaperPortfolio(10_000, slippage=0.0, fee_per_trade=0.0)
        pf.start_day(NOW.date())
        pf.open_position("A", "long", 100, 5.0, 4.0, NOW)
        pf.close_position("A", 6.0, NOW)
        s = pf.stats()
        self.assertEqual(s["trades"], 1)
        self.assertEqual(s["wins"], 1)
        self.assertAlmostEqual(s["win_rate"], 1.0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pf.json"
            pf.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["closed"]), 1)
        self.assertEqual(data["closed"][0]["symbol"], "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
