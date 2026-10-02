"""Forme « Finviz + Combined » : etoile signalee, ordre, interruption (hors reseau)."""
import core.combined_finviz as cf
import core.finviz_screeners as fs
import core.scan_fondamentaux as sf
import Combined_scan as cs


def _r(t, profil, total):
    return {"ticker": t, "nom": t, "pays": "US", "profil": profil, "score_total": total,
            "score_growth": 3, "score_safe": 5, "G3_Underval": 2, "G4_Momentum_%": 12.0,
            "S4_DivYield_%": 1.5}


def _brancher(monkeypatch, resultats):
    monkeypatch.setattr(fs, "run_preset", lambda key, limit=500: {"rows": [[t] for t in resultats]})
    monkeypatch.setattr(sf, "EUR_RATES", {"EUR": 1.0, "USD": 1.15})   # pas de telechargement de taux
    monkeypatch.setattr(cs, "analyze_safe", lambda t: resultats[t])


def test_etoile_signalee_et_en_tete(monkeypatch):
    _brancher(monkeypatch, {"AAA": _r("AAA", "⚪ Below", 2), "BBB": _r("BBB", "💎 Dual Champion", 9),
                            "CCC": _r("CCC", "💎 Dual Champion*", 8)})
    res = cf.run_finviz_combined()
    assert [r[0] for r in res["rows"]] == ["CCC", "BBB", "AAA"]
    assert [r[4] for r in res["rows"]] == ["⭐", "", ""]
    assert "1 ⭐ Dual Champion*" in res["title"] and "1 Dual Champion" in res["title"]


def test_annuler_garde_ce_qui_est_analyse(monkeypatch):
    _brancher(monkeypatch, {"AAA": _r("AAA", "💎 Dual Champion*", 9), "BBB": _r("BBB", "⚪ Below", 1)})
    res = cf.run_finviz_combined(progress=lambda i, n, t: i == 2)   # annule avant le 2e titre
    assert [r[0] for r in res["rows"]] == ["AAA"] and "interrompu" in res["title"]


def test_titre_ecarte_par_le_combined_ignore(monkeypatch):
    _brancher(monkeypatch, {"AAA": None, "BBB": _r("BBB", "🛡️  Pure Safe", 6)})
    assert [r[0] for r in cf.run_finviz_combined()["rows"]] == ["BBB"]
