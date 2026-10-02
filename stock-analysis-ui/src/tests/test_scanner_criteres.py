"""Verrouille quatre defauts du Combined_scan constates le 02.10.2026 sur les
recommandations de Boerse Online 40/2026 (champs yfinance bruts verifies).

D3 — Dividende suspendu. Takkt (TTK.DE) : dividendYield=22.14,
     trailingAnnualDividendYield=0.0, dernier detachement le 22.05.2025. Le
     zero explicite du champ annuel etait ignore (seul un etalon > 0 servait),
     le champ brut passait le seuil de plausibilite et S4 affichait 22 %.

D4 — S6 « croissance du FCF » ne lisait aucun FCF : elle prenait
     earningsQuarterlyGrowth (trimestre sur trimestre), sinon earningsGrowth,
     sinon revenueGrowth. Resultat : ASR +542 %, Stabilus +423 %, et pour
     Takkt, PWO, Fossil la meme valeur que G1.

D5 — S7 « croissance CA+BPA » prenait la croissance trimestrielle des
     benefices (ASR 647,8 % -> moyenne 332,8 %) ou, faute de champ, recopiait
     revenueGrowth (= G1).

D6 — S5 « marge FCF » lisait info["freeCashflow"], un FCF « levered » TTM qui
     depasse parfois le flux operationnel (Takkt 46,3 M contre 28,3 M ; Fossil
     +24,7 M contre -37,9 M), ce qui est impossible pour FCF = OCF - capex.

Correction : S5, S6 et S7 lisent les etats annuels (cashflow, income_stmt),
la croissance est un taux annuel compose entre le premier et le dernier
exercice disponibles.
"""
import datetime as dt

import pandas as pd

from core import scan_fondamentaux as cs


def _etat(valeurs: dict[str, list]) -> pd.DataFrame:
    """Etat yfinance : lignes = postes, colonnes = exercices, plus recent a gauche."""
    dates = [pd.Timestamp(f"{2025 - i}-12-31") for i in range(len(next(iter(valeurs.values()))))]
    return pd.DataFrame(valeurs, index=dates).T


# ── D3 : dividende suspendu ────────────────────────────────────────────────
def test_dividende_suspendu_vaut_zero():
    vieux = int((dt.datetime.now() - dt.timedelta(days=500)).timestamp())
    info = {"dividendYield": 22.14, "trailingAnnualDividendYield": 0.0,
            "trailingAnnualDividendRate": 0.0, "exDividendDate": vieux}
    ok, pct = cs.s4_dividend(info)
    assert pct == 0.0 and ok is False


def test_dividende_annuel_recent_reste_compte():
    """Un payeur annuel juste avant son detachement garde son rendement."""
    recent = int((dt.datetime.now() - dt.timedelta(days=200)).timestamp())
    info = {"dividendYield": 3.1, "trailingAnnualDividendYield": 0.0,
            "trailingAnnualDividendRate": 0.0, "exDividendDate": recent}
    assert cs.s4_dividend(info) == (True, 3.1)


# ── D4 : S6 lit le FCF annuel ──────────────────────────────────────────────
def test_s6_utilise_le_fcf_annuel_et_pas_les_benefices():
    # Stabilus : FCF 80,7 -> 108,0 M sur 3 ans, earningsQuarterlyGrowth 4.23 (423 %)
    etats = cs.etats_annuels(_etat({"Free Cash Flow": [108.0e6, 114.0e6, 104.4e6, 80.7e6]}), None)
    ok, pct = cs.s6_fcf_growth({"earningsQuarterlyGrowth": 4.23}, "STM.DE", etats)
    assert ok is True and 9.0 < pct < 11.0


def test_s6_non_calculable_si_fcf_negatif_aux_bornes():
    etats = cs.etats_annuels(_etat({"Free Cash Flow": [-5060e6, -2757e6, -923e6, -2078e6]}), None)
    assert cs.s6_fcf_growth({"revenueGrowth": -0.09}, "RWE.DE", etats) == (False, None)


def test_s6_sans_etats_ne_recopie_pas_le_chiffre_d_affaires():
    assert cs.s6_fcf_growth({"revenueGrowth": -0.05}, "TTK.DE", None) == (False, None)


# ── D5 : S7 lit CA et BPA annuels ──────────────────────────────────────────
def test_s7_bpa_en_baisse_fait_echouer_le_critere():
    # Stabilus : CA 1116 -> 1296 M (+5 %/an), BPA 4,17 -> 0,93 (effondrement)
    inc = _etat({"Total Revenue": [1296.1e6, 1305.9e6, 1215.3e6, 1116.3e6],
                 "Diluted EPS": [0.93, 2.84, 4.12, 4.17]})
    etats = cs.etats_annuels(None, inc)
    ok, pct = cs.s7_rev_eps_growth({"earningsGrowth": 4.175, "revenueGrowth": -0.05}, "STM.DE", etats)
    assert ok is False and pct is not None and pct < 0


def test_s7_croissance_reguliere_passe():
    inc = _etat({"Total Revenue": [130.0, 120.0, 110.0, 100.0], "Diluted EPS": [1.4, 1.25, 1.1, 1.0]})
    ok, pct = cs.s7_rev_eps_growth({}, "X", cs.etats_annuels(None, inc))
    assert ok is True and 9.0 < pct < 12.0


# ── D6 : S5 lit le FCF annuel ──────────────────────────────────────────────
def test_s5_prefere_le_fcf_de_l_etat_annuel():
    # Takkt : FCF 2025 21,9 M / CA 964,3 M = 2,3 % ; info["freeCashflow"] donnait 5,0 %
    etats = cs.etats_annuels(_etat({"Free Cash Flow": [21.9e6, 82.0e6]}),
                              _etat({"Total Revenue": [964.3e6, 1052.9e6], "Diluted EPS": [-1.88, -0.64]}))
    ok, pct = cs.s5_fcf_margin({"freeCashflow": 46.3e6, "totalRevenue": 927.5e6}, etats)
    assert ok is False and pct == 2.3


# ── Regroupement : un seul jeu de criteres pour tous les scanners ─────────
def test_news_monitor_dividende_n_est_plus_multiplie_par_cent():
    """L'ancienne copie faisait dividendYield * 100 : 4,37 devenait 437 %."""
    from AI_Implement import news_monitor_combined as nm
    ok, txt = nm.s4_dividend({"dividendYield": 4.37, "trailingAnnualDividendYield": 0.0429})
    assert ok is True and txt == "4.37%"


def test_les_scanners_partagent_les_criteres():
    import Combined_scan, Sichere_Unternehmen_scan, Big_Growth_scan
    assert Combined_scan.s6_fcf_growth is cs.s6_fcf_growth
    assert Combined_scan._record_skip is cs.record_skip
    assert Sichere_Unternehmen_scan._skip_reasons is cs.SKIP_REASONS
    assert Big_Growth_scan._throttle is cs.throttle


# ── Dual Champion* ─────────────────────────────────────────────────────────
def test_dual_etoile_exige_g3_g4_s4():
    assert cs.get_profile(3, 5, cs.est_etoile(True, True, True)) == "💎 Dual Champion*"
    assert cs.get_profile(3, 5, cs.est_etoile(True, False, True)) == "💎 Dual Champion"
    # L'etoile ne fabrique pas un Dual : sous les seuils elle est sans effet.
    assert cs.get_profile(2, 5, True) == "🛡️  Pure Safe"
    assert cs.est_dual("💎 Dual Champion*") and cs.est_dual("💎 Dual Champion")
    assert not cs.est_dual("⚖️  Balanced")
