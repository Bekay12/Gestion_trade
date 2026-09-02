"""
test_edgar - Suite hors ligne du client SEC EDGAR.

Aucun reseau : requests.Session.get est remplace par un faux. Lancer avec le
Python du projet :

    .venv\\Scripts\\python.exe agent\\Test\\test_edgar.py

Les cas qui comptent le plus sont ceux de assess_dilution : ils verifient la
distinction entre capacite d'emettre (S-3) et emission constatee (424B), qui
est ce que le corpus place au centre de la selection.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.sources import edgar
from agent.sources.edgar import EdgarClient, EdgarError, Filing, assess_dilution

TODAY = date(2026, 8, 21)


def filing(form: str, days_ago: int, accession="0001234567-26-000001", doc="d.htm") -> Filing:
    return Filing(
        form=form, filed=TODAY - timedelta(days=days_ago),
        accession=accession, document=doc, cik="0000320193",
    )


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class TestDilutionAssessment(unittest.TestCase):
    """La lecture des depots : capacite d'emettre contre emission constatee."""

    def test_no_signal_when_nothing_relevant(self):
        report = assess_dilution("ABCD", [filing("10-Q", 20), filing("4", 5)], TODAY)
        self.assertFalse(report.shelf_active)
        self.assertEqual(report.recent_offerings, [])
        self.assertEqual(report.risk, "faible")

    def test_recent_shelf_is_capacity_not_issuance(self):
        # Un S-3 seul : la dilution devient possible, elle n'a pas eu lieu.
        report = assess_dilution("ABCD", [filing("S-3", 200)], TODAY)
        self.assertTrue(report.shelf_active)
        self.assertEqual(report.recent_offerings, [])
        self.assertEqual(report.risk, "modere")
        self.assertIn("dormante", report.summary())

    def test_expired_shelf_is_not_active(self):
        # Au-dela de trois ans, la capacite ne porte plus.
        report = assess_dilution("ABCD", [filing("S-3", 365 * 3 + 30)], TODAY)
        self.assertFalse(report.shelf_active)
        self.assertEqual(report.risk, "faible")

    def test_shelf_at_exact_boundary_still_active(self):
        report = assess_dilution("ABCD", [filing("S-3", 365 * 3)], TODAY)
        self.assertTrue(report.shelf_active)

    def test_recent_offering_raises_risk_above_shelf(self):
        # Un 424B constate l'emission : il prime sur la simple capacite.
        report = assess_dilution("ABCD", [filing("S-3", 400), filing("424B5", 3)], TODAY)
        self.assertEqual(report.risk, "eleve")
        self.assertEqual(len(report.recent_offerings), 1)
        self.assertIn("424B5", report.summary())

    def test_old_offering_is_ignored(self):
        report = assess_dilution("ABCD", [filing("424B5", 90)], TODAY)
        self.assertEqual(report.recent_offerings, [])
        self.assertEqual(report.risk, "faible")

    def test_offerings_sorted_most_recent_first(self):
        report = assess_dilution(
            "ABCD", [filing("424B5", 20), filing("424B3", 2), filing("424B2", 10)], TODAY
        )
        self.assertEqual([f.form for f in report.recent_offerings], ["424B3", "424B2", "424B5"])

    def test_recent_events_window_is_one_week(self):
        report = assess_dilution("ABCD", [filing("8-K", 2), filing("8-K", 30)], TODAY)
        self.assertEqual(len(report.recent_events), 1)

    def test_latest_shelf_is_the_most_recent(self):
        report = assess_dilution(
            "ABCD", [filing("S-3", 500), filing("S-3ASR", 100)], TODAY
        )
        self.assertEqual(report.latest_shelf.form, "S-3ASR")

    def test_symbol_is_normalised(self):
        self.assertEqual(assess_dilution("abcd", [], TODAY).symbol, "ABCD")


class TestFilingUrl(unittest.TestCase):
    def test_url_strips_padding_and_dashes(self):
        f = Filing("8-K", TODAY, "0000320193-26-000042", "form8k.htm", "0000320193")
        url = f.url
        self.assertIn("/data/320193/", url)          # zeros de tete retires
        self.assertIn("000032019326000042", url)     # tirets retires
        self.assertTrue(url.endswith("form8k.htm"))

    def test_age_in_days(self):
        self.assertEqual(filing("8-K", 5).age_days(TODAY), 5)


class TestClientContract(unittest.TestCase):
    """En-tete obligatoire, cache, et remontee d'erreur explicite."""

    def test_missing_user_agent_refuses_to_start(self):
        # La SEC exige un en-tete nominatif : on ne le devine pas.
        with patch.dict("os.environ", {"SEC_USER_AGENT": ""}, clear=False):
            with self.assertRaises(EdgarError) as ctx:
                EdgarClient(cache_dir=Path(tempfile.gettempdir()))
        self.assertIn("SEC_USER_AGENT", str(ctx.exception))

    def test_explicit_user_agent_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            self.assertIn("User-Agent", client.session.headers)

    def test_resolve_cik_pads_to_ten_digits(self):
        payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", return_value=FakeResponse(payload)):
                self.assertEqual(client.resolve_cik("aapl"), "0000320193")

    def test_unknown_ticker_raises(self):
        payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", return_value=FakeResponse(payload)):
                with self.assertRaises(EdgarError):
                    client.resolve_cik("ZZZZ")

    def test_http_error_is_wrapped(self):
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", return_value=FakeResponse({}, status=403)):
                with self.assertRaises(EdgarError) as ctx:
                    client.resolve_cik("AAPL")
        self.assertIn("403", str(ctx.exception))

    def test_cache_is_served_without_second_call(self):
        payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", return_value=FakeResponse(payload)) as mock:
                client.resolve_cik("AAPL")
                client.resolve_cik("AAPL")
                self.assertEqual(mock.call_count, 1)

    def test_filings_parsed_from_submissions(self):
        tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        subs = {"filings": {"recent": {
            "form": ["8-K", "424B5", "10-Q"],
            "filingDate": ["2026-08-20", "2026-08-18", "2026-07-30"],
            "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002",
                                "0000320193-26-000003"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            "primaryDocDescription": ["EVENT", "PROSPECTUS", "QUARTERLY"],
        }}}

        def route(url, **kwargs):
            return FakeResponse(tickers if "company_tickers" in url else subs)

        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", side_effect=route):
                filings = client.filings("AAPL")

        self.assertEqual(len(filings), 3)
        self.assertEqual(filings[0].form, "8-K")
        self.assertEqual(filings[0].filed, date(2026, 8, 20))
        self.assertEqual(filings[1].description, "PROSPECTUS")

    def test_malformed_date_row_is_skipped_not_fatal(self):
        tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        subs = {"filings": {"recent": {
            "form": ["8-K", "10-Q"],
            "filingDate": ["pas-une-date", "2026-07-30"],
            "accessionNumber": ["a-1", "a-2"],
            "primaryDocument": ["a.htm", "c.htm"],
            "primaryDocDescription": ["", ""],
        }}}

        def route(url, **kwargs):
            return FakeResponse(tickers if "company_tickers" in url else subs)

        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", side_effect=route):
                filings = client.filings("AAPL")

        self.assertEqual(len(filings), 1)
        self.assertEqual(filings[0].form, "10-Q")

    def test_empty_submissions_returns_empty_list(self):
        tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

        def route(url, **kwargs):
            return FakeResponse(tickers if "company_tickers" in url else {})

        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get", side_effect=route):
                self.assertEqual(client.filings("AAPL"), [])

    def test_network_failure_is_wrapped(self):
        import requests
        with tempfile.TemporaryDirectory() as d:
            client = EdgarClient(cache_dir=Path(d), user_agent="Test test@example.org")
            with patch.object(client.session, "get",
                              side_effect=requests.RequestException("coupure")):
                with self.assertRaises(EdgarError):
                    client.resolve_cik("AAPL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
