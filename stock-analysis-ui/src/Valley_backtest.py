#!/usr/bin/env python3
"""
Valley_backtest - backtest point-in-time du detecteur de creux Valley_scan.py
(signaux INFLEXION, DIVERGENCE, PIEGE, A SUIVRE).

Rejoue, a chaque date T mensuelle, les fonctions MEMES du scanner
(mesures_cours, etat_tresorerie, classer) sur les seules donnees connues avant T,
puis mesure le rendement a 3, 6 et 12 mois contre l'indice du scanner et contre
la mediane de l'univers.

Pourquoi un script a part : l'option --asof de Valley_scan rejoue les cours mais
pas l'indicateur avance (fondamentaux « non historisables par l'API »), donc
INFLEXION et PIEGE n'y sont jamais testables. Ici l'indicateur avance vient de
la base locale, filtree par date de publication :
    - CA trimestriel (quarterly_financials, depuis mi-2024) : trimestre + 45 j
    - FCF et dette annuels (annual_financials, 2021+) : exercice + 90 j
Le chiffre d'affaires trimestriel n'existe qu'a partir de mi-2024 : INFLEXION
n'est mesurable qu'a partir de 2025 ; DIVERGENCE (cours seuls) des 2021.

Cours : un telechargement groupe par lots de 50 depuis 2018 (fenetre de 3 ans
du scanner), mis en cache dans backtests/valley/cours.parquet.
Biais connus : univers survivant (titres du store aujourd'hui), comptes
retraites par Yahoo, casse annuel au lieu de info["revenueGrowth"] quand
moins de 5 trimestres sont publies.

Usage :
    python Valley_backtest.py
    python Valley_backtest.py --debut 2022-01-01 --fin 2026-03-01
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import Valley_scan as vs
from Combined_backtest import (charger_comptes, ecreter, publies, rendement, saut_suspect,
                               to_md, DELAI_ANNUEL_J, DELAI_TRIM_J)

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent
OUT_DIR = SRC_DIR / "backtests" / "valley"
HORIZONS = (3, 6, 12)
SIGNAUX = ["↗️ INFLEXION", "🕳️ DIVERGENCE", "👀 À SUIVRE", "⚠️ PIÈGE", "—"]
SEUIL_BAISSE, SEUIL_BAS, MAX_PART_PROPRE = 25.0, 20.0, 75.0   # defauts du scanner


def univers_store() -> list[str]:
    import glob
    return sorted(d.split("symbol=", 1)[1] for d in glob.glob(str(SRC_DIR / "market_parquet" / "prices" / "symbol=*")))


def telecharger(symboles: list[str], debut: str, cache: Path, taille: int = 50) -> dict[str, pd.DataFrame]:
    """Close / Adj Close par titre, par lots groupes ; cache parquet long."""
    if cache.exists():
        data = pd.read_parquet(cache)
    else:
        import yfinance as yf
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        parts = []
        for i in range(0, len(symboles), taille):
            lot = symboles[i:i + taille]
            logger.info("[VB] lot %d/%d", i // taille + 1, -(-len(symboles) // taille))
            try:
                raw = yf.download(lot, start=debut, progress=False, auto_adjust=False,
                                  group_by="ticker", threads=True)
            except Exception as exc:
                logger.warning("[VB] lot echoue : %s", exc)
                continue
            for t in lot:
                try:
                    d = raw[t][["Close", "Adj Close"]].dropna(subset=["Close"])
                except (KeyError, TypeError):
                    continue
                if len(d):
                    d = d.rename(columns={"Close": "close", "Adj Close": "adj_close"})
                    d["ticker"] = t
                    parts.append(d)
        data = pd.concat(parts)
        data.index.name = "trade_date"
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    # Un titre present dans deux lots (indice aussi dans l'univers) arrive en double.
    return {t: g.drop(columns="ticker").sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
            for t, g in data.groupby("ticker")}


def indicateur_pit(trims: pd.DataFrame | None, annuels: pd.DataFrame | None, T: pd.Timestamp) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Equivalent point-in-time de Valley_scan.indicateur_avance : memes
        sorties, calculees sur les comptes publies avant T.

    Inputs:
        trims, annuels (DataFrame | None): comptes du titre (colonnes Combined_backtest)
        T (Timestamp): date de simulation

    Outputs:
        f (dict): ca_var_a1_%, indicateur_casse, indicateur_retourne,
                  tresorerie_degradee, fcf_sur_sommet_%, fcf_serie
    --------------------------------------------------------------------------
    """
    f = {"ca_var_a1_%": np.nan, "indicateur_casse": None, "indicateur_retourne": None}
    q = publies(trims, T, DELAI_TRIM_J).dropna(subset=["total_revenue"])
    rev = q["total_revenue"].to_numpy()
    if len(rev) >= 3:
        f["indicateur_retourne"] = bool(rev[-1] > rev[-2] > rev[-3])
    a = publies(annuels, T, DELAI_ANNUEL_J)
    if len(rev) >= 5 and rev[-5] > 0:
        var = (rev[-1] / rev[-5] - 1) * 100
    else:
        # Le scanner retombe ici sur info["revenueGrowth"] (introuvable a T) :
        # le dernier exercice sur le precedent en tient lieu.
        ra = a.dropna(subset=["total_revenue"])["total_revenue"].to_numpy()
        var = (ra[-1] / ra[-2] - 1) * 100 if len(ra) >= 2 and ra[-2] > 0 else np.nan
    if np.isfinite(var):
        f["ca_var_a1_%"] = var
        f["indicateur_casse"] = bool(var < -10.0)
    # etat_tresorerie attend du plus recent au plus ancien, comme yfinance.
    flux = a.dropna(subset=["free_cash_flow"]).sort_values("d", ascending=False)
    dette = a.dropna(subset=["total_debt"]).sort_values("d", ascending=False)
    f.update(vs.etat_tresorerie(list(flux["free_cash_flow"] / 1e6), list(dette["total_debt"] / 1e6)))
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest point-in-time de Valley_scan.")
    ap.add_argument("--debut", default="2021-07-01")
    ap.add_argument("--fin", default="2026-03-01")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    univers = univers_store()
    indices = sorted(set(vs.INDICES.values()) | {vs.INDICE_DEFAUT, vs.INDICE_REPLI})
    cours = telecharger(univers + indices, "2018-06-01", args.out / "cours.parquet")
    annuels, trims = charger_comptes()
    fin_donnees = max(c.index[-1] for c in cours.values())
    dates = pd.date_range(args.debut, args.fin, freq="MS")
    logger.info("[VB] %d titres avec cours, %d dates, cours jusqu'au %s",
                sum(1 for s in univers if s in cours), len(dates), fin_donnees.date())

    lignes = []
    for T in dates:
        idx_T = {i: cours[i]["close"][cours[i].index < T] for i in indices if i in cours}
        n0 = len(lignes)
        for s in univers:
            px = cours.get(s)
            if px is None:
                continue
            fen = px[(px.index < T) & (px.index >= T - pd.DateOffset(years=3))]["close"].dropna()
            if len(fen) < 60:          # meme plancher que Valley_scan._serie_close
                continue
            nom_idx = vs.indice_de(s)
            ci = idx_T.get(nom_idx)
            if ci is None or len(ci) < 60:
                ci = idx_T.get(vs.INDICE_REPLI)
            if ci is None:
                continue
            ci = ci[ci.index >= T - pd.DateOffset(years=3)]
            m = vs.mesures_cours(fen, ci)
            candidat = m["baisse_%"] <= -SEUIL_BAISSE or m["au_dessus_du_bas_%"] <= SEUIL_BAS
            f = indicateur_pit(trims.get(s), annuels.get(s), T) if candidat else {}
            signal, note, motif = vs.classer(m, f, SEUIL_BAISSE, SEUIL_BAS, MAX_PART_PROPRE)
            r = {"date_T": T.date().isoformat(), "ticker": s, "indice": nom_idx, "signal": signal,
                 "note": note, "baisse_%": m["baisse_%"], "au_dessus_du_bas_%": m["au_dessus_du_bas_%"],
                 "mouv_3m_%": m.get("mouv_3m_%"), "fraction_propre_3m_%": m.get("fraction_propre_3m_%"),
                 "ca_var_a1_%": f.get("ca_var_a1_%"), "indicateur_retourne": f.get("indicateur_retourne"),
                 "tresorerie_degradee": f.get("tresorerie_degradee")}
            idx_px = cours.get(nom_idx) if nom_idx in cours else cours.get(vs.INDICE_REPLI)
            for h in HORIZONS:
                susp = saut_suspect(px, T, h)
                r[f"r{h}"] = None if susp else rendement(px, T, h, fin_donnees)
                ri = rendement(idx_px, T, h, fin_donnees) if idx_px is not None else None
                r[f"x{h}"] = r[f"r{h}"] - ri if (r[f"r{h}"] is not None and ri is not None) else None
            lignes.append(r)
        nb = pd.Series([x["signal"] for x in lignes[n0:]]).value_counts().to_dict()
        logger.info("[VB] %s : %d titres, %s", T.date(), len(lignes) - n0,
                    {k: v for k, v in nb.items() if k != "—"})

    df = pd.DataFrame(lignes)
    for h in HORIZONS:
        df[f"r{h}"] = pd.to_numeric(df[f"r{h}"])
        df[f"x{h}"] = pd.to_numeric(df[f"x{h}"])
    # Ecretage par date (fonction de Combined_backtest, horizons 6 et 12) + horizon 3.
    df = ecreter(df)
    for col in ("r3", "x3"):
        b = df.groupby("date_T")[col].quantile([0.01, 0.99]).unstack()
        df[col] = df[col].clip(df["date_T"].map(b[0.01]), df["date_T"].map(b[0.99]))
    for h in HORIZONS:
        df[f"u{h}"] = df[f"r{h}"] - df.groupby("date_T")[f"r{h}"].transform("median")
    # Nouveau signal : absent le mois precedent pour ce titre (un meme creux
    # signale six mois de suite ne compte qu'une fois).
    df = df.sort_values(["ticker", "date_T"])
    df["nouveau"] = df["signal"] != df.groupby("ticker")["signal"].shift(1)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "valley_backtest_detail.csv", index=False)
    rapport(df, args.out, fin_donnees)


def resume(df: pd.DataFrame, rng_seed: int = 0) -> pd.DataFrame:
    """Par signal et horizon : effectifs, medianes, part qui bat l'indice, p du hasard."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for h in HORIZONS:
        base = df.dropna(subset=[f"x{h}"])
        for sig in SIGNAUX:
            sel = base[base["signal"] == sig]
            if sel.empty:
                continue
            obs, sims = [], []
            for _, g in base.groupby("date_T"):
                xs = g.loc[g["signal"] == sig, f"x{h}"].to_numpy()
                if len(xs) == 0:
                    continue
                pool = g[f"x{h}"].to_numpy()
                obs.append(np.median(xs))
                sims.append([np.median(rng.choice(pool, len(xs), replace=False)) for _ in range(500)])
            p = float(np.mean(np.mean(np.array(sims), axis=0) >= np.mean(obs))) if obs and sig != "—" else float("nan")
            par_date = sel.groupby("date_T")[f"x{h}"].median()
            rows.append({"signal": sig, "horizon": f"{h} mois", "signaux": len(sel),
                         "dates": sel["date_T"].nunique(),
                         "perf_med_%": 100 * sel[f"r{h}"].median(),
                         "ecart_indice_med_pts": 100 * sel[f"x{h}"].median(),
                         "ecart_indice_moy_pts": 100 * sel[f"x{h}"].mean(),
                         "ecart_med_univers_pts": 100 * sel[f"u{h}"].median(),
                         "battent_indice_%": 100 * (sel[f"x{h}"] > 0).mean(),
                         "dates_positives": f"{int((par_date > 0).sum())}/{len(par_date)}",
                         "p_hasard": p})
    return pd.DataFrame(rows)


def rapport(df: pd.DataFrame, out: Path, fin_donnees) -> None:
    neufs = df[df["nouveau"]]
    df["annee"] = df["date_T"].str[:4]
    par_annee = []
    for (an, sig), g in df[df["signal"] != "—"].groupby(["annee", "signal"]):
        g6 = g.dropna(subset=["x6"])
        if len(g6):
            par_annee.append({"annee": an, "signal": sig, "signaux_6m": len(g6),
                              "ecart_indice_med_6m_pts": 100 * g6["x6"].median(),
                              "battent_indice_6m_%": 100 * (g6["x6"] > 0).mean()})
    univ = df.dropna(subset=["x6"]).groupby("annee")["x6"].median() * 100
    pa = pd.DataFrame(par_annee)
    if not pa.empty:
        pa["univers_med_6m_pts"] = pa["annee"].map(univ)
    detail = df[(df["signal"] == "↗️ INFLEXION") & df["nouveau"]][
        ["date_T", "ticker", "au_dessus_du_bas_%", "ca_var_a1_%", "r6", "x6", "r12", "x12"]].copy()
    for c in ["r6", "x6", "r12", "x12"]:
        detail[c] = detail[c] * 100
    detail = detail.rename(columns={"r6": "perf6_%", "x6": "ecart6_pts", "r12": "perf12_%", "x12": "ecart12_pts"})
    md = ["# Backtest point-in-time du détecteur de creux (Valley_scan)", "",
          f"Dates T mensuelles {df['date_T'].min()} → {df['date_T'].max()}, "
          f"{df['ticker'].nunique()} titres, cours jusqu'au {fin_donnees.date()}. Fonctions de classement "
          "de Valley_scan.py (seuils par défaut : baisse 25 %, 20 % du plus bas, part propre 75 %), "
          "indicateur avancé reconstitué sur les comptes publiés avant T. Écart à l'indice du scanner, "
          "écrêté aux 1er/99e centiles par date ; sauts de cours x4 ou /4 écartés.",
          "INFLEXION et PIÈGE (trimestriels) ne sont mesurables qu'à partir de 2025 ; DIVERGENCE dès 2021. "
          "`p_hasard` : part des tirages au hasard de même taille, date par date, qui font au moins aussi bien.",
          "", "## Tous les signaux (un titre peut être compté plusieurs mois de suite)", "", to_md(resume(df)), "",
          "## Nouveaux signaux seulement (absents le mois précédent)", "", to_md(resume(neufs)), "",
          "## Par année (6 mois, tous signaux)", "", to_md(pa) if not pa.empty else "n.d.", "",
          "## Détail : nouveaux signaux INFLEXION", "",
          to_md(detail.sort_values("date_T")) if not detail.empty else "aucun"]
    (out / "valley_backtest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("[VB] écrit %s", out / "valley_backtest.md")


if __name__ == "__main__":
    main()
