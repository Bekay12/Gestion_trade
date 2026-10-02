"""Tests for persistent snapshots of displayed screener results."""
import csv
from datetime import datetime, timedelta

import ui.mixins.screeners as screeners
from core import store_screeners
from ui.mixins.screeners import ScreenersMixin


def test_archive_enregistre_entetes_metadonnees_et_resultats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(screeners, "SCREENER_RESULTS_DIR", tmp_path)

    archive = screeners._archive_screener_results(
        "Combined pur — Dual Champion*",
        ["Symbole", "Profil", "Score"],
        [("AAA", "Dual Champion*", 7), ("BBB", "Pure Safe", None)],
    )

    with archive.open(newline="", encoding="utf-8-sig") as csvfile:
        rows = list(csv.reader(csvfile))

    assert archive.parent == tmp_path
    assert archive.suffix == ".csv"
    assert rows[0] == ["run_at", "screener", "Symbole", "Profil", "Score"]
    assert rows[1][1:] == ["Combined pur — Dual Champion*", "AAA", "Dual Champion*", "7"]
    assert rows[2][1:] == ["Combined pur — Dual Champion*", "BBB", "Pure Safe", ""]


def test_nouveaux_tickers_comparés_au_dernier_snapshot_du_meme_screener(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(screeners, "SCREENER_RESULTS_DIR", tmp_path)
    headers = ["Symbole", "Domaine"]

    real_datetime = screeners.datetime

    class Yesterday(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return super().now(tz) - timedelta(days=1)

    monkeypatch.setattr(screeners, "datetime", Yesterday)
    screeners._archive_screener_results(
        "Combined pur (catalogue) — 1 ⭐ Dual Champion* — 10 résultat(s)",
        headers,
        [("AAA", "Industrials"), ("BBB", "Technology")],
    )
    monkeypatch.setattr(screeners, "datetime", real_datetime)
    screeners._archive_screener_results(
        "Yahoo Screener — New High (max 50)",
        headers,
        [("ZZZ", "Energy")],
    )

    new_symbols = screeners._new_screener_symbols(
        "Combined pur (catalogue) — 3 ⭐ Dual Champion* — 12 résultat(s)",
        headers,
        [("AAA", "Industrials"), ("CCC", "Healthcare")],
    )

    assert new_symbols == {"CCC"}


def test_archive_remplace_le_fichier_du_meme_screener_dans_la_journee(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(screeners, "SCREENER_RESULTS_DIR", tmp_path)
    title = "Combined pur (catalogue) — 1 ⭐ Dual Champion* — 10 résultat(s)"

    first_path = screeners._archive_screener_results(
        title, ["Symbole"], [("AAA",)],
    )
    second_path = screeners._archive_screener_results(
        "Combined pur (catalogue) — 2 ⭐ Dual Champion* — 11 résultat(s)",
        ["Symbole"], [("BBB",)],
    )

    assert first_path == second_path
    assert len(list(tmp_path.glob("*.csv"))) == 1
    with second_path.open(newline="", encoding="utf-8-sig") as csvfile:
        archived_rows = list(csv.DictReader(csvfile))
    assert [row["Symbole"] for row in archived_rows] == ["BBB"]


def test_premiere_ouverture_marque_les_tickers_comme_nouveaux(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(screeners, "SCREENER_RESULTS_DIR", tmp_path)

    assert screeners._new_screener_symbols(
        "Finviz — test", ["Symbole"], [("AAA",)],
    ) == {"AAA"}


def test_secteur_finviz_est_normalise_en_domaine() -> None:
    headers, rows = screeners._ensure_domain_column(
        ["Symbole", "Nom", "Secteur", "Pays"],
        [("AAA", "Acme", "Industrials", "US")],
        {},
    )

    assert headers == ["Symbole", "Nom", "Domaine", "Pays"]
    assert rows == [["AAA", "Acme", "Industrials", "US"]]


def test_presenter_archive_meme_si_le_dialogue_est_annule(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(screeners, "SCREENER_RESULTS_DIR", tmp_path)
    captured = {}

    class CancelledDialog:
        Accepted = 1

        def __init__(self, *args, **kwargs) -> None:
            captured["headers"] = args[1]
            captured["rows"] = args[2]
            captured["new_symbols"] = kwargs["new_symbols"]

        def exec_(self) -> int:
            return 0

    monkeypatch.setattr(ScreenersMixin, "_domain_map", lambda self, symbols: {"AAA": "Industrials"})
    monkeypatch.setattr(ScreenersMixin, "_completer_profils_en_arriere_plan", lambda self, rows: None)
    monkeypatch.setattr("ui.dialogs.ScreenerResultsDialog", CancelledDialog)
    selected = ScreenersMixin()._present_screener_results(
        "Test screener", ["Symbole", "Profil"], [("AAA", "Dual Champion*")]
    )

    assert selected == []
    assert captured["headers"] == ["Symbole", "Domaine", "Profil"]
    assert captured["rows"] == [["AAA", "Industrials", "Dual Champion*"]]
    assert captured["new_symbols"] == {"AAA"}
    assert len(list(tmp_path.glob("*.csv"))) == 1


def test_combined_catalogue_ne_garde_pas_de_colonne_etoile(monkeypatch) -> None:
    import pandas as pd

    frame = pd.DataFrame([{
        "symbol": "AAA",
        "name": "Acme",
        "sector": "Technologie",
        "c1_ok": True,
        "c2_ok": True,
        "c3_ok": True,
        "c4_ok": True,
        "c5_ok": False,
        "s4_ok": True,
        "secure_score": 5,
    }])
    monkeypatch.setattr(store_screeners, "get_latest_features", lambda symbols=None: frame)
    monkeypatch.setattr(store_screeners, "get_country_map", lambda symbols: {"AAA": "US"})
    monkeypatch.setattr(store_screeners, "get_name_map", lambda symbols: {"AAA": "Acme"})

    result = store_screeners.screen_combined()

    assert "⭐" not in result["headers"]
    assert result["rows"] == [["AAA", "Acme", "Technologie", "US", "Dual Champion*", 4, 5]]
    assert "1 ⭐ Dual Champion*" in result["title"]


def test_dual_star_et_golden_cross_catalogue_exposent_le_domaine(monkeypatch) -> None:
    import pandas as pd

    frame = pd.DataFrame([{
        "symbol": "AAA",
        "name": "Acme",
        "sector": "Industrials",
        "c1_ok": True,
        "c2_ok": True,
        "c3_ok": True,
        "c4_ok": True,
        "c5_ok": False,
        "s4_ok": True,
        "secure_score": 5,
        "feature_date": "2026-10-02",
        "price": 110,
        "sma50": 104.9,
        "sma200": 100,
    }])
    monkeypatch.setattr(store_screeners, "get_latest_features", lambda symbols=None: frame)
    monkeypatch.setattr(store_screeners, "get_country_map", lambda symbols: {"AAA": "US"})
    monkeypatch.setattr(store_screeners, "get_name_map", lambda symbols: {"AAA": "Acme"})

    for result in (store_screeners.screen_dual_star(), store_screeners.screen_golden_cross()):
        assert "Domaine" in result["headers"]
        assert result["rows"][0][result["headers"].index("Domaine")] == "Industrials", result["title"]