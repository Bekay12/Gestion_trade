#!/usr/bin/env python3
"""
Combined_backtest - backtest point-in-time du Combined scan : a chaque date T,
note l'univers du store avec les 12 criteres (core/scan_fondamentaux.py) en
n'utilisant que ce qui etait connu avant T, puis mesure ce que chaque profil
(Dual Champion, Pure Safe, ...) a rapporte a 6 et 12 mois.

Sources (0 requete yfinance, sauf un telechargement groupe des indices) :
    - cours journaliers : market_parquet/prices (depuis 2023-01)
    - comptes annuels / trimestriels : stock_analysis.db (2021+ / mi-2024+)
    - devise, nombre d'actions, PER actuel : market_parquet/fundamentals
    - change historique : market_parquet/fx_rates_daily

Ce qui differe du scanner en direct (impossible a reconstituer a la date T) :
    - comptes publies : exercice + 90 j, trimestre + 45 j (aucun look-ahead)
    - G1 : CA du dernier trimestre sur un an si 5 trimestres publies, sinon
      dernier exercice sur le precedent
    - G3 : PEG = PER / taux compose du BPA passe (proxy retrospectif ; le
      PEG Yahoo utilise les previsions des analystes, introuvables a T)
    - S1 : nombre d'actions ACTUEL x cours a T
    - S3 : beta hebdomadaire sur 52 semaines contre l'indice local (Yahoo :
      5 ans mensuels contre le S&P 500)
    - S4 : dividendes deduits de l'ecart cours ajuste / cours brut sur 12 mois
Biais connus : univers = titres encore presents dans le store (survivants),
comptes retraites par Yahoo.

Usage :
    python Combined_backtest.py                       # dates trimestrielles par defaut
    python Combined_backtest.py --dates 2024-10-01 2025-04-01
"""
from __future__ import annotations

import argparse
import glob
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from core import scan_fondamentaux as sf

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent
STORE = SRC_DIR / "market_parquet"
DB_PATH = SRC_DIR / "stock_analysis.db"
OUT_DIR = SRC_DIR / "backtests" / "combined"

DATES_DEFAUT = ["2024-04-01", "2024-07-01", "2024-10-01", "2025-01-02",
                "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-02"]
HORIZONS_MOIS = (6, 12)
DELAI_ANNUEL_J = 90       # publication d'un exercice apres sa cloture
DELAI_TRIM_J = 45         # publication d'un trimestre apres sa cloture
MIN_JOURS_COURS = 130     # comme le scanner (G4)
PROFILS = ["💎 Dual Champion*", "💎 Dual Champion", "🛡️  Pure Safe", "🚀 Pure Growth", "⚖️  Balanced", "⚪ Below"]

# Indice local par suffixe Yahoo (meme table que Magasines/evaluer_recommandations.py).
BENCHMARKS = {
    ".DE": "^GDAXI", ".PA": "^FCHI", ".BR": "^BFX", ".AS": "^AEX", ".MI": "FTSEMIB.MI",
    ".MC": "^IBEX", ".VI": "^ATX", ".IR": "^ISEQ", ".L": "^FTSE", ".SW": "^SSMI",
    ".ST": "^OMX", ".HE": "^OMXH25", ".OL": "OBX.OL", ".CO": "^OMXC25", ".T": "^N225",
    ".SI": "^STI", ".NZ": "^NZ50", ".TO": "^GSPTSE", ".HK": "^HSI", ".KS": "^KS11",
    ".TW": "^TWII", ".AX": "^AXJO", "": "^GSPC",
}


def benchmark_for(ticker: str) -> str:
    """Indice de la place de cotation ; S&P 500 a defaut."""
    suffixe = "." + ticker.rsplit(".", 1)[1] if "." in ticker else ""
    return BENCHMARKS.get(suffixe, "^GSPC")


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════

def charger_prix() -> dict[str, pd.DataFrame]:
    """Cours du store : symbole -> DataFrame(close, adj_close, high, volume) indexe par date."""
    out = {}
    for d in glob.glob(str(STORE / "prices" / "symbol=*")):
        sym = d.split("symbol=", 1)[1]
        try:
            df = pd.concat(pd.read_parquet(f, columns=["trade_date", "close", "adj_close", "high", "volume"])
                           for f in glob.glob(d + "/*.parquet"))
        except Exception as exc:
            logger.warning("[BT] cours illisibles %s : %s", sym, exc)
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.dropna(subset=["close"]).drop_duplicates("trade_date").set_index("trade_date").sort_index()
        if len(df):
            out[sym] = df
    return out


def charger_profils() -> pd.DataFrame:
    """Dernier instantane par symbole : devise, actions, PER et cours actuels (calibrage)."""
    rows = []
    for f in glob.glob(str(STORE / "fundamentals" / "symbol=*" / "*.parquet")):
        try:
            rows.append(pd.read_parquet(f, columns=["symbol", "as_of_date", "name", "sector", "currency",
                                                    "shares_outstanding", "trailing_pe", "current_price"]))
        except Exception:
            continue
    df = pd.concat(rows).sort_values("as_of_date")
    return df.groupby("symbol").last()


def charger_comptes() -> tuple[dict, dict]:
    """Comptes annuels et trimestriels par symbole, tries par date de cloture."""
    con = sqlite3.connect(DB_PATH)
    cols = "symbol, {d} AS d, total_revenue, gross_profit, diluted_eps, free_cash_flow, total_debt, stockholders_equity"
    a = pd.read_sql(f"SELECT {cols.format(d='fiscal_date')} FROM annual_financials", con)
    q = pd.read_sql(f"SELECT {cols.format(d='quarter_date')} FROM quarterly_financials", con)
    con.close()
    for df in (a, q):
        df["d"] = pd.to_datetime(df["d"])
    a = a.sort_values("d")
    q = q.sort_values("d").drop_duplicates(["symbol", "d"], keep="last")
    return {s: g for s, g in a.groupby("symbol")}, {s: g for s, g in q.groupby("symbol")}


def charger_fx() -> pd.DataFrame:
    """Taux USD par unite de devise, une colonne par devise, jours ouvres remplis vers l'avant."""
    rows = [pd.read_parquet(f) for f in glob.glob(str(STORE / "fx_rates_daily" / "**" / "*.parquet"), recursive=True)]
    fx = pd.concat(rows)
    fx["trade_date"] = pd.to_datetime(fx["trade_date"])
    fx = fx.pivot_table(index="trade_date", columns="currency", values="rate_to_usd", aggfunc="last")
    fx["USD"] = 1.0
    return fx.sort_index().ffill()


def charger_indices(tickers: list[str], debut: str, cache: Path) -> dict[str, pd.DataFrame]:
    """Indices de reference : un seul telechargement groupe, mis en cache."""
    if cache.exists():
        data = pd.read_parquet(cache)
    else:
        import yfinance as yf
        raw = yf.download(sorted(set(tickers)), start=debut, progress=False, auto_adjust=False, group_by="ticker")
        parts = []
        for t in sorted(set(tickers)):
            try:
                d = raw[t][["Close", "Adj Close"]].dropna().rename(columns={"Close": "close", "Adj Close": "adj_close"})
            except KeyError:
                continue
            d["ticker"] = t
            parts.append(d)
        data = pd.concat(parts)
        data.index.name = "trade_date"
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return {t: g.drop(columns="ticker").sort_index() for t, g in data.groupby("ticker")}


# ═══════════════════════════════════════════════════════════════
# POINT-IN-TIME
# ═══════════════════════════════════════════════════════════════

COLS_COMPTES = ["d", "total_revenue", "gross_profit", "diluted_eps", "free_cash_flow",
                "total_debt", "stockholders_equity"]


def publies(comptes: pd.DataFrame | None, T: pd.Timestamp, delai_j: int) -> pd.DataFrame:
    """Lignes de comptes publiees avant T (cloture + delai <= T) ; jamais de look-ahead."""
    if comptes is None or comptes.empty:
        return pd.DataFrame(columns=COLS_COMPTES)
    return comptes[comptes["d"] + pd.Timedelta(days=delai_j) <= T]


def _serie(df: pd.DataFrame, col: str) -> list:
    s = df[["d", col]].dropna()
    return [(pd.Timestamp(d), float(v)) for d, v in zip(s["d"], s[col])]


def eps_ttm(annuels: pd.DataFrame, trims: pd.DataFrame):
    """BPA sur 12 mois : 4 derniers trimestres consecutifs, sinon dernier exercice."""
    t = trims.dropna(subset=["diluted_eps"]).tail(4)
    if len(t) == 4 and (t["d"].iloc[-1] - t["d"].iloc[0]).days < 300:
        return float(t["diluted_eps"].sum())
    a = annuels.dropna(subset=["diluted_eps"])
    return float(a["diluted_eps"].iloc[-1]) if len(a) else None


def croissance_ca(annuels: pd.DataFrame, trims: pd.DataFrame):
    """(fraction, source) : dernier trimestre sur un an, sinon dernier exercice sur le precedent."""
    t = trims.dropna(subset=["total_revenue"])
    if len(t):
        der = t.iloc[-1]
        cible = der["d"] - pd.DateOffset(years=1)
        prec = t[(t["d"] - cible).abs() <= pd.Timedelta(days=20)]
        if len(prec) and prec["total_revenue"].iloc[0] > 0:
            return der["total_revenue"] / prec["total_revenue"].iloc[0] - 1, "trimestre"
    a = annuels.dropna(subset=["total_revenue"])
    if len(a) >= 2 and a["total_revenue"].iloc[-2] > 0:
        return a["total_revenue"].iloc[-1] / a["total_revenue"].iloc[-2] - 1, "exercice"
    return None, None


def rendement_dividende(px: pd.DataFrame) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rendement du dividende sur la fenetre fournie, deduit des cours : le
        cours ajuste Yahoo applique au jour precedant un detachement le facteur
        f = 1 - D / cours, donc le rapport ajuste/brut change de f ce jour-la
        et nulle part ailleurs (les divisions d'actions touchent les deux).

    Inputs:
        px (DataFrame): close, adj_close sur la fenetre (12 mois avant T)

    Outputs:
        rendement (float): somme des dividendes / dernier cours, en fraction
    --------------------------------------------------------------------------
    """
    if len(px) < 2:
        return 0.0
    r = (px["adj_close"] / px["close"]).to_numpy()
    close = px["close"].to_numpy()
    f = r[:-1] / r[1:]                        # facteur applique au jour j avant le detachement j+1
    evenements = (f < 0.9999) & (f > 0.5)     # > 0.5 : ecarte une donnee aberrante
    dividendes = (close[:-1] * (1 - f))[evenements].sum()
    return float(dividendes / close[-1]) if close[-1] > 0 else 0.0


def beta_hebdo(px: pd.Series, idx: pd.Series | None) -> float | None:
    """Beta des rendements hebdomadaires sur 52 semaines contre l'indice ; None si < 40 semaines."""
    if idx is None:
        return None
    a = px.resample("W-FRI").last().pct_change(fill_method=None)
    b = idx.resample("W-FRI").last().pct_change(fill_method=None)
    j = pd.concat([a, b], axis=1).dropna().tail(52)
    if len(j) < 40 or j.iloc[:, 1].var() == 0:
        return None
    return float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())


def facteur_devise(profil: pd.Series, eps_actuel) -> float | None:
    """
    Unites de cours par unite de BPA des comptes : 1 en general, 100 pour une
    cotation en pence. Calibre sur le PER actuel (cours / PER = BPA dans la
    devise du cours) ; None si les comptes sont dans une autre devise que le
    cours (ADR, titre etranger) : le PER a T n'est alors pas calculable.
    """
    defaut = 100.0 if profil.get("currency") in ("GBp", "GBX") else 1.0
    pe, prix = sf.safe_float(profil.get("trailing_pe")), sf.safe_float(profil.get("current_price"))
    if not pe or pe <= 0 or not prix or not eps_actuel or eps_actuel <= 0:
        return defaut
    k = (prix / pe) / eps_actuel
    for cible in (1.0, 100.0):
        if 0.6 * cible < k < 1.6 * cible:
            return cible
    return None


def taux_eur(fx: pd.DataFrame, T: pd.Timestamp) -> dict:
    """EUR_RATES du module commun a la date T : unites de devise par euro."""
    ligne = fx[fx.index < T].iloc[-1]
    taux = {"EUR": 1.0}
    for dev, usd in ligne.items():
        if pd.notna(usd) and usd > 0 and pd.notna(ligne.get("EUR")):
            taux[dev] = float(ligne["EUR"] / usd)
    if "GBP" in taux:
        taux["GBP_PENCE"] = taux["GBP"] * 100.0
    return taux


def noter(sym: str, T: pd.Timestamp, px: pd.DataFrame, annuels, trims, profil, idx) -> dict | None:
    """Les 12 criteres du Combined a la date T, avec les seules donnees connues avant T."""
    hist = px[px.index < T]
    if len(hist) < MIN_JOURS_COURS:
        return None
    a = publies(annuels, T, DELAI_ANNUEL_J)
    q = publies(trims, T, DELAI_TRIM_J)
    cours = float(hist["close"].iloc[-1])
    etats = {"fcf": _serie(a, "free_cash_flow"), "rev": _serie(a, "total_revenue"), "eps": _serie(a, "diluted_eps")}

    eps_actuel = eps_ttm(publies(annuels, pd.Timestamp.max, 0), publies(trims, pd.Timestamp.max, 0))
    k = facteur_devise(profil, eps_actuel)
    eps_t = eps_ttm(a, q)
    pe = cours / (eps_t * k) if (k and eps_t and eps_t > 0) else None
    t_bpa = sf.taux_compose(etats["eps"])
    peg = pe / t_bpa if (pe and t_bpa and t_bpa > 0) else None
    rg, src_rg = croissance_ca(a, q)
    der = a.dropna(subset=["total_revenue"]).tail(1)
    gm = (der["gross_profit"].iloc[0] / der["total_revenue"].iloc[0]) if len(der) and der["total_revenue"].iloc[0] > 0 \
        and pd.notna(der["gross_profit"].iloc[0]) else None
    bilan = a.dropna(subset=["stockholders_equity"]).tail(1)
    de = (bilan["total_debt"].iloc[0] / bilan["stockholders_equity"].iloc[0] * 100) \
        if len(bilan) and bilan["stockholders_equity"].iloc[0] > 0 and pd.notna(bilan["total_debt"].iloc[0]) else None
    actions = sf.safe_float(profil.get("shares_outstanding"))
    info = {
        "revenueGrowth": rg, "grossMargins": gm,
        "trailingPE": pe, "pegRatio": peg, "currentPrice": cours,
        "fiftyTwoWeekHigh": float(hist["high"].iloc[-252:].max()),
        "marketCap": actions * cours if actions else None, "currency": profil.get("currency"),
        "debtToEquity": de, "beta": beta_hebdo(hist["close"], idx),
        "trailingAnnualDividendYield": rendement_dividende(hist[hist.index >= T - pd.Timedelta(days=365)]),
    }
    h = hist.rename(columns={"close": "Close", "volume": "Volume"})
    g = [sf.g1_revenue_growth(info), sf.g2_gross_margin(info), sf.g3_undervaluation(info),
         sf.g4_momentum(h), sf.g5_volume(h)]
    s = [sf.s1_market_cap(info), sf.s2_debt_equity(info), sf.s3_beta(info), sf.s4_dividend(info),
         sf.s5_fcf_margin({}, etats), sf.s6_fcf_growth({}, sym, etats), sf.s7_rev_eps_growth({}, sym, etats)]
    gs, ss = sum(r[0] for r in g), sum(r[0] for r in s)
    out = {"date_T": T.date().isoformat(), "ticker": sym, "nom": profil.get("name"), "secteur": profil.get("sector"),
           "score_total": gs + ss, "score_growth": gs, "score_safe": ss, "profil": sf.get_profile(gs, ss, sf.est_etoile(g[2][0], g[3][0], s[3][0])),
           "G1_source": src_rg, "PER_calculable": pe is not None,
           # Champs bruts du preset Finviz « dual_star », pour l'emuler a T.
           "PER": pe, "PEG": peg,
           "au_dessus_sma50": bool(hist["close"].iloc[-1] > hist["close"].iloc[-50:].mean()),
           "vol_moy_3m": float(hist["volume"].iloc[-63:].mean())}
    for i, r in enumerate(g, 1):
        out[f"G{i}"], out[f"G{i}_ok"] = r[1], r[0]
    for i, r in enumerate(s, 1):
        out[f"S{i}"], out[f"S{i}_ok"] = r[1], r[0]
    return out


def rendement(px: pd.DataFrame | None, T: pd.Timestamp, mois: int, fin_donnees: pd.Timestamp):
    """Rendement total (cours ajuste) de la 1re seance >= T a la derniere seance <= T + mois ; None si inachevé."""
    if px is None:
        return None
    cible = T + pd.DateOffset(months=mois)
    if cible > fin_donnees:
        return None
    apres = px[px.index >= T]
    avant = px[px.index <= cible]
    if apres.empty or avant.empty or avant.index[-1] < cible - pd.Timedelta(days=10):
        return None   # serie interrompue avant l'echeance (radiation, trou du store)
    a0, a1 = float(apres["adj_close"].iloc[0]), float(avant["adj_close"].iloc[-1])
    return a1 / a0 - 1 if a0 > 0 else None


def saut_suspect(px: pd.DataFrame | None, T: pd.Timestamp, mois: int) -> bool:
    """
    Vrai si le cours ajuste fait, entre T et l'echeance, un saut journalier
    superieur a x4 ou inferieur a /4. Mesure du 02.10.2026 : PPCB passe de
    0,01 a 625 en une seance (regroupement d'actions non ajuste dans le store)
    et affiche +916 567 % sur 6 mois ; une poignee de cas pareils suffit a
    rendre toute moyenne d'univers absurde.
    """
    if px is None:
        return False
    w = px.loc[(px.index >= T) & (px.index <= T + pd.DateOffset(months=mois)), "adj_close"].dropna()
    j = (w / w.shift(1)).dropna()
    return bool(((j > 4) | (j < 0.25)).any())


# ═══════════════════════════════════════════════════════════════
# SYNTHESE
# ═══════════════════════════════════════════════════════════════

PREMIERE_DATE_FIABLE = "2025-04-01"
# Avant cette date, le store n'a que 2 exercices publies avec un FCF (le plus
# ancien, 2021, est vide chez yfinance) : S6 et S7 sont calculables pour 3 a
# 20 % des titres contre 45 a 87 % ensuite, et un Dual Champion (S >= 5/7)
# devient presque impossible. Ces dates restent en annexe, non comparables.


def ecreter(df: pd.DataFrame) -> pd.DataFrame:
    """Rendements et ecarts ecretes aux 1er et 99e centiles de l'univers, date par date."""
    out = df.copy()
    for col in [f"{p}{m}" for p in ("r", "x") for m in HORIZONS_MOIS]:
        bornes = out.groupby("date_T")[col].quantile([0.01, 0.99]).unstack()
        lo, hi = out["date_T"].map(bornes[0.01]), out["date_T"].map(bornes[0.99])
        out[col] = out[col].clip(lo, hi)
    return out


def synthese(df: pd.DataFrame, cle: str, ordre: list | None = None) -> pd.DataFrame:
    rows = []
    for m in HORIZONS_MOIS:
        for k, sub in df.dropna(subset=[f"r{m}"]).groupby(cle, observed=True):
            rows.append({cle: k, "horizon": f"{m} mois", "titres": len(sub), "dates": sub["date_T"].nunique(),
                         "perf_moy_%": 100 * sub[f"r{m}"].mean(), "perf_med_%": 100 * sub[f"r{m}"].median(),
                         "ecart_med_univers_pts": 100 * sub[f"u{m}"].mean(),
                         "ecart_indice_pts": 100 * sub[f"x{m}"].mean(),
                         "battent_indice_%": 100 * (sub[f"x{m}"] > 0).mean(),
                         "gagnants_%": 100 * (sub[f"r{m}"] > 0).mean()})
    out = pd.DataFrame(rows)
    if ordre and not out.empty:
        out["_o"] = out[cle].map({k: i for i, k in enumerate(ordre)})
        out = out.sort_values(["horizon", "_o"]).drop(columns="_o")
    return out


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, v in r.items():
            if isinstance(v, float):
                cells.append("n.d." if pd.isna(v) else (f"{v:.3f}" if c.startswith("p_") else
                             f"{v:+.1f}" if ("%" in c or "pts" in c) else f"{v:.0f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def par_date_dual(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date_T"):
        dg = g[g["profil"].map(sf.est_dual)]
        for m in HORIZONS_MOIS:
            if g[f"r{m}"].notna().any():
                rows.append({"date_T": d, "horizon": f"{m} mois", "dual_champions": len(dg),
                             "dual_perf_med_%": 100 * dg[f"r{m}"].median(),
                             "univers_perf_med_%": 100 * g[f"r{m}"].median(),
                             "dual_ecart_med_univers_pts": 100 * dg[f"u{m}"].mean(),
                             "dual_ecart_indice_pts": 100 * dg[f"x{m}"].mean(),
                             "dual_battent_indice_%": 100 * (dg[f"x{m}"] > 0).mean()})
    return pd.DataFrame(rows).sort_values(["horizon", "date_T"])


SELECTIONS = [  # (libelle, filtre sur le detail) : seuils des scanners en direct et variantes
    ("Big Growth G>=3 (seuil Big_Growth_scan)", lambda d: d["score_growth"] >= 3),
    ("Big Growth G>=4", lambda d: d["score_growth"] >= 4),
    ("Big Growth G=5", lambda d: d["score_growth"] == 5),
    ("Sichere S>=5 (seuil Pure Safe / Dual)", lambda d: d["score_safe"] >= 5),
    ("Sichere S>=6", lambda d: d["score_safe"] >= 6),
    ("Sichere S=7", lambda d: d["score_safe"] == 7),
    ("Dual Champion (G>=3 et S>=5)", lambda d: (d["score_growth"] >= 3) & (d["score_safe"] >= 5)),
    ("Dual Champion* (Dual + G3 + G4 + S4)", lambda d: d["profil"] == sf.DUAL_ETOILE),
    ("Dual sans etoile", lambda d: d["profil"] == sf.DUAL),
    ("Univers", lambda d: d["score_total"] >= 0),
]


def p_aleatoire(df: pd.DataFrame, masque: pd.Series, m: int, tirages: int = 2000, graine: int = 0) -> float:
    """
    Probabilite qu'un tirage au hasard de meme taille, date par date, fasse au
    moins aussi bien (moyenne des medianes d'ecart a l'indice) que la selection.
    """
    rng = np.random.default_rng(graine)
    obs, sims = [], []
    for _, g in df.dropna(subset=[f"x{m}"]).assign(_sel=masque).groupby("date_T"):
        sel = g.loc[g["_sel"], f"x{m}"].to_numpy()
        if len(sel) == 0:
            continue
        x = g[f"x{m}"].to_numpy()
        obs.append(np.median(sel))
        sims.append([np.median(rng.choice(x, len(sel), replace=False)) for _ in range(tirages)])
    if not obs:
        return float("nan")
    return float(np.mean(np.mean(np.array(sims), axis=0) >= np.mean(obs)))


def tableau_selections(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in HORIZONS_MOIS:
        base = df.dropna(subset=[f"x{m}"])
        for nom, filtre in SELECTIONS:
            masque = filtre(base)
            sub = base[masque]
            hors_mat = sub[sub["secteur"] != "Basic Materials"]
            par_date = sub.groupby("date_T")[f"x{m}"].median()
            rows.append({"selection": nom, "horizon": f"{m} mois", "titres": len(sub),
                         "par_date": round(len(sub) / max(1, sub["date_T"].nunique())),
                         "perf_med_%": 100 * sub[f"r{m}"].median(),
                         "ecart_indice_moy_pts": 100 * sub[f"x{m}"].mean(),
                         "ecart_indice_med_pts": 100 * sub[f"x{m}"].median(),
                         "battent_indice_%": 100 * (sub[f"x{m}"] > 0).mean(),
                         "dates_positives": f"{int((par_date > 0).sum())}/{len(par_date)}",
                         "med_sans_materiaux_pts": 100 * hors_mat[f"x{m}"].median(),
                         "p_hasard": p_aleatoire(base, masque, m) if nom != "Univers" else float("nan")})
    return pd.DataFrame(rows)


def par_date_selection(df: pd.DataFrame, noms: list[str]) -> pd.DataFrame:
    """Ecart median a l'indice, date par date, pour quelques selections."""
    rows = []
    for m in HORIZONS_MOIS:
        base = df.dropna(subset=[f"x{m}"])
        for d, g in base.groupby("date_T"):
            row = {"date_T": d, "horizon": f"{m} mois"}
            for nom, filtre in SELECTIONS:
                if nom in noms:
                    sel = g[filtre(g)]
                    row[f"{nom.split(' (')[0]} n"] = float(len(sel))
                    row[f"{nom.split(' (')[0]} med_pts"] = 100 * sel[f"x{m}"].median() if len(sel) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon", "date_T"])


def rafraichir_cours(prix: dict, symboles: list[str], debut: pd.Timestamp, taille: int = 50) -> int:
    """
    Complete en memoire (sans ecrire dans le store) les cours perimes, par
    telechargements groupes de `taille` titres : ~1 requete pour 50 titres, la
    forme que la regle de budget yfinance du projet autorise.
    Renvoie le nombre de titres mis a jour.
    """
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    n = 0
    for i in range(0, len(symboles), taille):
        lot = symboles[i:i + taille]
        try:
            raw = yf.download(lot, start=debut.date().isoformat(), progress=False, auto_adjust=False,
                              group_by="ticker", threads=True)
        except Exception as exc:
            logger.warning("[CAND] lot %d echoue : %s", i // taille, exc)
            continue
        for t in lot:
            try:
                d = raw[t][["Close", "Adj Close", "High", "Volume"]].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            if d.empty:
                continue
            d = d.rename(columns={"Close": "close", "Adj Close": "adj_close", "High": "high", "Volume": "volume"})
            d.index = pd.to_datetime(d.index).tz_localize(None)
            ancien = prix.get(t)
            prix[t] = d if ancien is None else pd.concat([ancien[ancien.index < d.index[0]], d]).sort_index()
            n += 1
    return n


def candidats_du_jour(out: Path) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------
    Purpose:
        Presélection a 0 requete yfinance : meme notation que le backtest, a la
        date du jour, sur tout le store. Les sélections retenues sont celles qui
        ont tenu dans le backtest (Dual Champion, Big Growth G >= 4, score >= 9) ;
        la liste ecrite sert a une confirmation en direct, a petit budget.

    Inputs:
        out (Path): dossier de sortie

    Outputs:
        df (DataFrame): titres notes ; ecrit candidats_<date>.csv et .symbols.txt
    --------------------------------------------------------------------------
    """
    prix, profils = charger_prix(), charger_profils()
    annuels, trims = charger_comptes()
    fx = charger_fx()
    univers = sorted(set(prix) & set(profils.index))
    T = pd.Timestamp.today().normalize()
    perimes = [s for s in univers if prix[s].index[-1] < T - pd.Timedelta(days=5)]
    logger.info("[CAND] %d titres aux cours perimes : rafraichissement groupe (%d requetes)",
                len(perimes), -(-len(perimes) // 50))
    logger.info("[CAND] %d titres rafraichis", rafraichir_cours(prix, perimes, T - pd.Timedelta(days=420)))
    # Indices : cache du backtest complete par un seul telechargement groupe recent.
    indices = charger_indices([benchmark_for(s) for s in univers], "2022-06-01", out / "indices.parquet")
    rafraichir_cours(indices, list(indices), T - pd.Timedelta(days=420), taille=len(indices))
    sf.EUR_RATES = taux_eur(fx, T)
    lignes = []
    for sym in univers:
        idx = indices.get(benchmark_for(sym))
        r = noter(sym, T, prix[sym], annuels.get(sym), trims.get(sym), profils.loc[sym],
                  idx["close"] if idx is not None else None)
        if r is not None:
            r["dernier_cours"] = prix[sym].index[-1].date().isoformat()
            lignes.append(r)
    df = pd.DataFrame(lignes)
    # Cours de plus de 30 jours : titre plus suivi par le store, ecarte.
    df = df[pd.to_datetime(df["dernier_cours"]) >= T - pd.Timedelta(days=30)]
    retenu = df["profil"].map(sf.est_dual) | (df["score_growth"] >= 4) | (df["score_total"] >= 9)
    sel = df[retenu].sort_values(["score_total", "score_growth"], ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    jour = T.date().isoformat()
    df.to_csv(out / f"candidats_{jour}_tous.csv", index=False)
    sel.to_csv(out / f"candidats_{jour}.csv", index=False)
    (out / f"candidats_{jour}.symbols.txt").write_text("\n".join(sel["ticker"]) + "\n")
    logger.info("[CAND] %d titres notes au %s (cours du store jusqu'au %s) ; %d retenus : %d Dual Champions, "
                "%d Big Growth G>=4, %d score >= 9", len(df), jour, max(p.index[-1] for p in prix.values()).date(),
                len(sel), sel["profil"].map(sf.est_dual).sum(), (sel["score_growth"] >= 4).sum(),
                (sel["score_total"] >= 9).sum())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest point-in-time du Combined scan.")
    ap.add_argument("--dates", nargs="+", default=DATES_DEFAUT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--candidats", action="store_true",
                    help="note l'univers du store a la date du jour (0 requete) et ecrit une liste restreinte "
                         "a confirmer avec Combined_scan.py --symbols-file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("core.scan_fondamentaux").setLevel(logging.ERROR)   # croissances extremes : bruit ici
    if args.candidats:
        candidats_du_jour(args.out)
        return

    prix = charger_prix()
    profils = charger_profils()
    annuels, trims = charger_comptes()
    fx = charger_fx()
    univers = sorted(set(prix) & set(profils.index))
    fin = min(max(p.index[-1] for p in prix.values()), fx.index[-1])
    logger.info("[BT] univers %d titres (cours + profil), cours jusqu'au %s", len(univers), fin.date())
    indices = charger_indices([benchmark_for(s) for s in univers], "2022-06-01", args.out / "indices.parquet")

    lignes = []
    for d in args.dates:
        T = pd.Timestamp(d)
        sf.EUR_RATES = taux_eur(fx, T)
        n0 = len(lignes)
        for sym in univers:
            idx = indices.get(benchmark_for(sym))
            r = noter(sym, T, prix[sym], annuels.get(sym), trims.get(sym), profils.loc[sym],
                      idx["close"] if idx is not None else None)
            if r is None:
                continue
            r["indice"] = benchmark_for(sym)
            for m in HORIZONS_MOIS:
                suspect = saut_suspect(prix[sym], T, m)
                r[f"suspect{m}"] = suspect
                r[f"r{m}"] = None if suspect else rendement(prix[sym], T, m, fin)
                ri = rendement(idx, T, m, fin) if idx is not None else None
                r[f"x{m}"] = r[f"r{m}"] - ri if (r[f"r{m}"] is not None and ri is not None) else None
            lignes.append(r)
        logger.info("[BT] %s : %d titres notes, %d Dual Champions", d, len(lignes) - n0,
                    sum(1 for x in lignes[n0:] if sf.est_dual(x["profil"])))

    df = pd.DataFrame(lignes)
    for m in HORIZONS_MOIS:
        df[f"r{m}"] = pd.to_numeric(df[f"r{m}"])
        df[f"x{m}"] = pd.to_numeric(df[f"x{m}"])
    df = ecreter(df)
    for m in HORIZONS_MOIS:
        # Ecart a la MEDIANE de l'univers a la meme date : neutralise le marche sans
        # se laisser tirer par quelques micro-capitalisations explosives.
        df[f"u{m}"] = df[f"r{m}"] - df.groupby("date_T")[f"r{m}"].transform("median")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "backtest_detail.csv", index=False)

    fiable = df[df["date_T"] >= PREMIERE_DATE_FIABLE]
    ancien = df[df["date_T"] < PREMIERE_DATE_FIABLE]
    bandes = lambda x: x.assign(score_bande=pd.cut(x["score_total"], [-1, 3, 5, 7, 9, 12],
                                                   labels=["0-3", "4-5", "6-7", "8-9", "10-12"]).astype(str))
    ordre_bandes = ["0-3", "4-5", "6-7", "8-9", "10-12"]
    par_profil = synthese(fiable, "profil", PROFILS)
    par_score = synthese(bandes(fiable), "score_bande", ordre_bandes)
    par_date = par_date_dual(df)
    par_profil.to_csv(args.out / "backtest_par_profil.csv", index=False)
    par_date.to_csv(args.out / "backtest_par_date.csv", index=False)
    n_susp = {m: int(df[f"suspect{m}"].sum()) for m in HORIZONS_MOIS}
    dc = fiable[fiable["profil"].map(sf.est_dual)]
    detail = dc[["date_T", "ticker", "nom", "score_total", "r6", "r12", "x6", "x12"]].copy()
    for c in ["r6", "r12", "x6", "x12"]:
        detail[c] = detail[c] * 100
    detail = detail.rename(columns={"r6": "perf6_%", "r12": "perf12_%", "x6": "ecart6_indice_pts",
                                    "x12": "ecart12_indice_pts"}).sort_values(["date_T", "score_total"],
                                                                               ascending=[True, False])
    md = ["# Backtest point-in-time du Combined scan", "",
          f"Dates T : {', '.join(args.dates)}. Univers : {len(univers)} titres du store, cours jusqu'au {fin.date()}.",
          "",
          "Méthode : à chaque date T, les 12 critères du scanner (core/scan_fondamentaux.py) sur les seules "
          "données connues avant T (exercice publié 90 j après clôture, trimestre 45 j). Rendement total "
          "(dividendes réinvestis) de la première séance après T à l'échéance. Écarts : à l'indice local, et à la "
          "médiane de l'univers à la même date. Rendements et écarts écrêtés aux 1er/99e centiles par date.",
          f"Rendements écartés pour saut de cours invraisemblable (x4 ou /4 en une séance) : "
          f"{n_susp[6]} à 6 mois, {n_susp[12]} à 12 mois.",
          f"Tableaux principaux : dates T >= {PREMIERE_DATE_FIABLE} seulement (avant, S6/S7 quasi incalculables). "
          "Limites : univers survivant, PEG rétrospectif, nombre d'actions actuel, bêta hebdomadaire 52 sem.",
          "", f"## Par profil (T >= {PREMIERE_DATE_FIABLE})", "", to_md(par_profil), "",
          f"## Par score total (T >= {PREMIERE_DATE_FIABLE})", "", to_md(par_score), "",
          "## Dual Champions par date (toutes dates ; avant avril 2025 : non comparable)", "", to_md(par_date), "",
          f"## Dual Champions (T >= {PREMIERE_DATE_FIABLE}, détail)", "", to_md(detail), "",
          f"## Filtres séparés : sélections (T >= {PREMIERE_DATE_FIABLE})", "",
          "`ecart_indice_*` : rendement moins indice local. `dates_positives` : dates où la médiane de la "
          "sélection bat l'indice. `med_sans_materiaux` : médiane hors secteur Basic Materials (mines d'or). "
          "`p_hasard` : part des tirages au hasard de même taille, date par date, qui font au moins aussi bien.",
          "", to_md(tableau_selections(fiable)), "",
          f"## Big Growth seul : par score G sur 5 (T >= {PREMIERE_DATE_FIABLE})", "",
          to_md(synthese(fiable.assign(score_G=fiable["score_growth"].astype(str)), "score_G",
                         [str(i) for i in range(6)])), "",
          f"## Sichere seul : par score S sur 7 (T >= {PREMIERE_DATE_FIABLE})", "",
          to_md(synthese(fiable.assign(score_S=fiable["score_safe"].astype(str)), "score_S",
                         [str(i) for i in range(8)])), "",
          "## Seuils des scanners, date par date (écart médian à l'indice)", "",
          to_md(par_date_selection(fiable, [SELECTIONS[0][0], SELECTIONS[3][0], SELECTIONS[6][0]])), "",
          f"## Annexe : par profil, dates antérieures à {PREMIERE_DATE_FIABLE}", "",
          to_md(synthese(ancien, "profil", PROFILS))]
    (args.out / "backtest_combined.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("[BT] écrit %s", args.out / "backtest_combined.md")


if __name__ == "__main__":
    main()
