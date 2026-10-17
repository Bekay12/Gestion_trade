"""
test_autonomy - Suite hors ligne des garde-fous d'execution autonome (bot eToro).

Aucun reseau, aucune cle, aucun compte courtier. Lancer avec le Python du
projet :

    python3 agent/Test/test_autonomy.py

Les cas qui comptent le plus sont ceux qui REFUSENT ou qui redescendent le
levier - le module existe pour borner, pas pour calculer une performance.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import autonomy


class TestLeverageTier(unittest.TestCase):
    def test_une_semaine_perdante_redescend_immediatement(self):
        tier, streak = autonomy.leverage_tier_after_week(1, 3, -100.0)
        self.assertEqual(tier, 0)
        self.assertEqual(streak, 0)

    def test_perte_au_palier_zero_reste_au_palier_zero(self):
        tier, streak = autonomy.leverage_tier_after_week(0, 2, -1.0)
        self.assertEqual(tier, 0)
        self.assertEqual(streak, 0)

    def test_quatre_semaines_beneficiaires_font_monter_un_palier(self):
        tier, streak = 0, 0
        for _ in range(4):
            tier, streak = autonomy.leverage_tier_after_week(tier, streak, 10.0)
        self.assertEqual(tier, 1)
        self.assertEqual(streak, 0)

    def test_trois_semaines_beneficiaires_ne_suffisent_pas(self):
        tier, streak = 0, 0
        for _ in range(3):
            tier, streak = autonomy.leverage_tier_after_week(tier, streak, 10.0)
        self.assertEqual(tier, 0)
        self.assertEqual(streak, 3)

    def test_plafond_du_dernier_palier_ne_monte_plus(self):
        dernier = len(autonomy.LEVERAGE_LADDER) - 1
        tier, streak = dernier, 0
        for _ in range(4):
            tier, streak = autonomy.leverage_tier_after_week(tier, streak, 10.0)
        self.assertEqual(tier, dernier)

    def test_palier_hors_bornes_refuse(self):
        with self.assertRaises(ValueError):
            autonomy.leverage_tier_after_week(len(autonomy.LEVERAGE_LADDER), 0, 1.0)

    def test_current_leverage_lit_la_table(self):
        self.assertEqual(autonomy.current_leverage(0), 1)
        self.assertEqual(autonomy.current_leverage(len(autonomy.LEVERAGE_LADDER) - 1),
                         autonomy.LEVERAGE_LADDER[-1])


class TestDailyBreaker(unittest.TestCase):
    def test_perte_sous_le_seuil_ne_declenche_pas(self):
        self.assertFalse(autonomy.daily_breaker_triggered(-999.0, capital_total=100_000.0))

    def test_perte_au_seuil_exact_declenche(self):
        self.assertTrue(autonomy.daily_breaker_triggered(-10_000.0, capital_total=100_000.0))

    def test_perte_au_dela_du_seuil_declenche(self):
        self.assertTrue(autonomy.daily_breaker_triggered(-15_000.0, capital_total=100_000.0))

    def test_gain_ne_declenche_jamais(self):
        self.assertFalse(autonomy.daily_breaker_triggered(50_000.0, capital_total=100_000.0))

    def test_capital_nul_ou_negatif_refuse(self):
        with self.assertRaises(ValueError):
            autonomy.daily_breaker_triggered(-1.0, capital_total=0.0)


class TestPositionCap(unittest.TestCase):
    def test_sous_le_plafond_autorise(self):
        self.assertTrue(autonomy.can_open_new_position(2, max_positions=3))

    def test_au_plafond_refuse(self):
        self.assertFalse(autonomy.can_open_new_position(3, max_positions=3))

    def test_au_dela_du_plafond_refuse(self):
        self.assertFalse(autonomy.can_open_new_position(4, max_positions=3))


class TestSessionsEtHorizon(unittest.TestCase):
    def test_meme_jour_zero_seance_ecoulee(self):
        d = date(2026, 10, 8)
        self.assertEqual(autonomy.sessions_elapsed(d, d), 0)

    def test_lendemain_ouvrable_une_seance(self):
        self.assertEqual(
            autonomy.sessions_elapsed(date(2026, 10, 8), date(2026, 10, 9)), 1
        )

    def test_weekend_ne_compte_pas(self):
        # Jeudi 08.10.2026 -> lundi 12.10.2026 : vendredi puis lundi, le
        # week-end ne s'ajoute pas (2 seances, pas 4 jours calendaires).
        self.assertEqual(
            autonomy.sessions_elapsed(date(2026, 10, 8), date(2026, 10, 12)), 2
        )

    def test_today_avant_opened_on_refuse(self):
        with self.assertRaises(ValueError):
            autonomy.sessions_elapsed(date(2026, 10, 9), date(2026, 10, 8))

    def test_horizon_pump_risk_trois_seances_pas_encore_du(self):
        # Ouvert jeudi, horizon 3 seances : pas encore du le vendredi (1 seance).
        self.assertFalse(
            autonomy.horizon_exit_due(date(2026, 10, 8), 3, date(2026, 10, 9))
        )

    def test_horizon_pump_risk_trois_seances_du_le_bon_jour(self):
        # Ouvert jeudi (seance 1) ; vendredi = seance 2 ; lundi = seance 3 ->
        # du a la cloture du lundi, le jour d'ouverture comptant comme la
        # premiere seance.
        self.assertTrue(
            autonomy.horizon_exit_due(date(2026, 10, 8), 3, date(2026, 10, 12))
        )

    def test_horizon_fade_une_seance_du_dans_la_journee(self):
        d = date(2026, 10, 8)
        self.assertTrue(autonomy.horizon_exit_due(d, 1, d))

    def test_horizon_nul_refuse(self):
        with self.assertRaises(ValueError):
            autonomy.horizon_exit_due(date(2026, 10, 8), 0, date(2026, 10, 8))


class TestKillSwitch(unittest.TestCase):
    def test_absent_par_defaut(self, tmp_name="agent_test_killswitch_absent.flag"):
        p = Path(__file__).resolve().parent / tmp_name
        if p.exists():
            p.unlink()
        self.assertFalse(autonomy.kill_switch_active(p))

    def test_actif_si_fichier_present(self, tmp_name="agent_test_killswitch_present.flag"):
        p = Path(__file__).resolve().parent / tmp_name
        p.write_text("stop")
        try:
            self.assertTrue(autonomy.kill_switch_active(p))
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
