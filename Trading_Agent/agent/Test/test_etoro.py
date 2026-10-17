"""
test_etoro - Suite hors ligne du client REST eToro.

Aucun reseau, aucune cle reelle, aucun compte. `requests.request` est
remplace par un faux via unittest.mock.patch - jamais appele pour de vrai.
Patcher LA OU LA FONCTION EST DEFINIE (agent.sources.etoro.requests.request),
pas un autre module qui la reexporterait (piege deja tombe une fois ailleurs,
voir Test/CLAUDE.md).

Lancer avec le Python du projet :

    python3 agent/Test/test_etoro.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.sources import etoro


class FakeResponse:
    """Reponse requests minimale : .ok, .status_code, .json(), .text."""

    def __init__(self, payload: dict | None = None, status: int = 200):
        self._payload = payload or {}
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = str(self._payload)

    def json(self):
        return self._payload


def _with_credentials():
    """Contexte qui pose des cles d'environnement factices pour la duree du bloc."""
    return patch.dict(os.environ, {"ETORO_API_KEY": "fake-api", "ETORO_USER_KEY": "fake-user"})


class TestCredentials(unittest.TestCase):
    def test_cle_absente_refuse(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(etoro.EtoroUnavailable):
                etoro._credentials()

    def test_une_seule_cle_presente_refuse(self):
        with patch.dict(os.environ, {"ETORO_API_KEY": "x"}, clear=True):
            with self.assertRaises(etoro.EtoroUnavailable):
                etoro._credentials()

    def test_les_deux_cles_presentes_passent(self):
        with _with_credentials():
            api, user = etoro._credentials()
            self.assertEqual((api, user), ("fake-api", "fake-user"))

    def test_entetes_portent_x_request_id_uuid(self):
        with _with_credentials():
            headers = etoro._headers()
            self.assertEqual(headers["x-api-key"], "fake-api")
            self.assertEqual(headers["x-user-key"], "fake-user")
            self.assertEqual(len(headers["x-request-id"]), 36)  # format UUID


class TestRequestTranslatesErrors(unittest.TestCase):
    def test_reponse_ok_rend_le_json(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({"ok": True})):
                self.assertEqual(etoro._request("GET", "/x"), {"ok": True})

    def test_429_devient_etoro_unavailable(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({}, status=429)):
                with self.assertRaises(etoro.EtoroUnavailable):
                    etoro._request("GET", "/x")

    def test_erreur_http_devient_etoro_unavailable(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({}, status=500)):
                with self.assertRaises(etoro.EtoroUnavailable):
                    etoro._request("GET", "/x")

    def test_exception_reseau_devient_etoro_unavailable(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             side_effect=etoro.requests.ConnectionError("coupe")):
                with self.assertRaises(etoro.EtoroUnavailable):
                    etoro._request("GET", "/x")


class TestCreateOrder(unittest.TestCase):
    def test_transaction_invalide_refuse_avant_tout_appel_reseau(self):
        order = etoro.OrderRequest(instrument_id=1, transaction="sell",
                                   amount=100.0, stop_loss_rate=1.0)
        with patch.object(etoro.requests, "request") as mock_req:
            with self.assertRaises(ValueError):
                etoro.create_order(order, sandbox=True)
            mock_req.assert_not_called()

    def test_sandbox_appelle_le_chemin_demo(self):
        order = etoro.OrderRequest(instrument_id=101, transaction="buy",
                                   amount=1000.0, stop_loss_rate=1.2)
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({"orderId": 1})) as mock_req:
                etoro.create_order(order, sandbox=True)
                called_path = mock_req.call_args.args[1]
                self.assertIn("/demo/orders", called_path)

    def test_reel_appelle_le_chemin_reel_sans_demo(self):
        order = etoro.OrderRequest(instrument_id=101, transaction="sellShort",
                                   amount=1000.0, stop_loss_rate=1.2, leverage=1,
                                   settlement_type="cfd")
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({"orderId": 2})) as mock_req:
                etoro.create_order(order, sandbox=False)
                called_path = mock_req.call_args.args[1]
                self.assertNotIn("demo", called_path)
                body = mock_req.call_args.kwargs["json"]
                self.assertEqual(body["transaction"], "sellShort")
                self.assertEqual(body["settlementType"], "cfd")
                self.assertEqual(body["stopLossRate"], 1.2)

    def test_stop_loss_absent_du_type_refuse_a_la_construction(self):
        # OrderRequest n'a pas de defaut pour stop_loss_rate : un appel sans
        # le fournir leve TypeError avant meme d'atteindre create_order.
        with self.assertRaises(TypeError):
            etoro.OrderRequest(instrument_id=1, transaction="buy", amount=100.0)


class TestLacunesNonDevinees(unittest.TestCase):
    """Les chemins demo non verifies doivent refuser, jamais deviner une URL."""

    def test_get_portfolio_sandbox_refuse_sans_deviner(self):
        with self.assertRaises(NotImplementedError):
            etoro.get_portfolio(sandbox=True)

    def test_close_position_sandbox_refuse_sans_deviner(self):
        with self.assertRaises(NotImplementedError):
            etoro.close_position(position_id=1, instrument_id=2, sandbox=True)

    def test_get_portfolio_reel_appelle_le_bon_chemin(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({"instrumentAggregates": []})) as mock_req:
                etoro.get_portfolio(sandbox=False)
                called_path = mock_req.call_args.args[1]
                self.assertTrue(called_path.endswith("/api/v1/trading/info/aggregate-portfolio"))


class TestLookupOrder(unittest.TestCase):
    def test_exige_exactement_un_identifiant(self):
        with self.assertRaises(ValueError):
            etoro.lookup_order()
        with self.assertRaises(ValueError):
            etoro.lookup_order(order_id=1, reference_id="x")

    def test_sandbox_utilise_le_chemin_demo_lookup(self):
        with _with_credentials():
            with patch.object(etoro.requests, "request",
                             return_value=FakeResponse({})) as mock_req:
                etoro.lookup_order(order_id=42, sandbox=True)
                called_path = mock_req.call_args.args[1]
                self.assertIn("demo/orders:lookup", called_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
