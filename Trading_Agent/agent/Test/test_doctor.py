"""
test_doctor - Suite hors ligne du preflight TWS.

    .venv\Scripts\python.exe agent\Test\test_doctor.py

Le preflight mesure ce qu'un compte sait faire au lieu de le supposer. Ce qui
est verifie ici est ce qui, faux, ferait payer un abonnement pour rien : la
distinction entre une capacite absente et un marche ferme.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.ibkr_doctor import KO, MEH, OK, marche_ouvert, quote_capabilities

NAN = float("nan")


class FakeTicker:
    """Ticker ib_async minimal : les champs reellement lus par le preflight."""

    def __init__(self, **fields):
        defaults = dict(
            last=NAN, close=NAN, bid=NAN, ask=NAN, bidSize=NAN, askSize=NAN,
            volume=NAN, halted=NAN, shortable=NAN, shortableShares=NAN,
            marketDataType=1,
        )
        defaults.update(fields)
        for key, value in defaults.items():
            setattr(self, key, value)


class TestHeuresDeMarche(unittest.TestCase):
    """Hors seance, l'absence de carnet n'est pas un droit manquant."""

    def test_dimanche_est_ferme(self):
        from agent.ibkr_doctor import marche_ouvert
        ouvert, texte = marche_ouvert(datetime(2026, 8, 23, 5, 46))
        self.assertFalse(ouvert)
        self.assertIn("week-end", texte)

    def test_mardi_en_seance_est_ouvert(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertTrue(marche_ouvert(datetime(2026, 8, 25, 10, 30))[0])

    def test_pre_marche_n_est_pas_la_seance(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertFalse(marche_ouvert(datetime(2026, 8, 25, 5, 0))[0])

    def test_apres_cloture(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertFalse(marche_ouvert(datetime(2026, 8, 25, 16, 30))[0])

    def test_les_bornes_viennent_de_gate(self):
        # R10 est la seule source des horaires : pas de copie locale.
        import agent.ibkr_doctor as doc
        from agent.gate import CLOSE_TIME, OPEN_TIME
        self.assertIs(doc.OPEN_TIME, OPEN_TIME)
        self.assertIs(doc.CLOSE_TIME, CLOSE_TIME)


class TestPreflight(unittest.TestCase):
    """Le preflight mesure les capacites au lieu de les supposer."""

    def test_port_ferme_est_detecte(self):
        from agent.sources.ibkr import probe_port
        # Port improbable : rien n'ecoute, la sonde doit le dire sans lever.
        self.assertFalse(probe_port("127.0.0.1", 59999, timeout=0.3))

    def test_port_ouvert_est_detecte(self):
        from agent.sources.ibkr import probe_port
        import socket
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        try:
            self.assertTrue(probe_port("127.0.0.1", server.getsockname()[1]))
        finally:
            server.close()

    def test_cotation_complete_ne_signale_aucun_manque(self):
        ticker = FakeTicker(
            last=1.71, bid=1.66, ask=1.71, volume=715.0,
            halted=0.0, shortable=2.0, shortableShares=12000.0,
        )
        caps = quote_capabilities(ticker)
        self.assertEqual([n for n, (m, _) in caps.items() if m == KO], [])

    def test_cotation_vide_pointe_les_droits_manquants(self):
        caps = quote_capabilities(FakeTicker())
        manquants = [n for n, (m, _) in caps.items() if m == KO]
        self.assertIn("Statut d'emprunt (236)", manquants)
        self.assertIn("Carnet (bid/ask)", manquants)

    def test_flux_differe_est_une_reserve_pas_un_echec(self):
        # Differe n'est pas une panne : R6 reste calculable, l'execution non.
        caps = quote_capabilities(FakeTicker(last=1.71, marketDataType=3))
        self.assertEqual(caps["Flux temps reel"][0], MEH)

    def test_flux_direct_est_un_succes(self):
        caps = quote_capabilities(FakeTicker(last=1.71, marketDataType=1))
        self.assertEqual(caps["Flux temps reel"][0], OK)


class TestHeuresDeMarche(unittest.TestCase):
    """Hors seance, l'absence de carnet n'est pas un droit manquant."""

    def test_dimanche_est_ferme(self):
        from agent.ibkr_doctor import marche_ouvert
        ouvert, texte = marche_ouvert(datetime(2026, 8, 23, 5, 46))
        self.assertFalse(ouvert)
        self.assertIn("week-end", texte)

    def test_mardi_en_seance_est_ouvert(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertTrue(marche_ouvert(datetime(2026, 8, 25, 10, 30))[0])

    def test_pre_marche_n_est_pas_la_seance(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertFalse(marche_ouvert(datetime(2026, 8, 25, 5, 0))[0])

    def test_apres_cloture(self):
        from agent.ibkr_doctor import marche_ouvert
        self.assertFalse(marche_ouvert(datetime(2026, 8, 25, 16, 30))[0])

    def test_les_bornes_viennent_de_gate(self):
        # R10 est la seule source des horaires : pas de copie locale.
        import agent.ibkr_doctor as doc
        from agent.gate import CLOSE_TIME, OPEN_TIME
        self.assertIs(doc.OPEN_TIME, OPEN_TIME)
        self.assertIs(doc.CLOSE_TIME, CLOSE_TIME)


if __name__ == "__main__":
    unittest.main(verbosity=2)
