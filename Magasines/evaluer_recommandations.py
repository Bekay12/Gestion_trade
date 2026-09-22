"""
evaluer_recommandations - mesure, N jours après parution, si les recommandations d'achat des
magazines ont tenu : objectif atteint ou stop touché en premier, performance, écart à l'indice
de la place de cotation, puis taux de réussite par magazine, type et horizon.

Entrée  : Recommandations_achat_*.csv (colonnes ticker, date_entree, cours, objectif, stop).
Sorties : Evaluation_recommandations_<asof>.md et .csv dans Magasines/.

Usage :
    .venv_new/bin/python Magasines/evaluer_recommandations.py                  # à aujourd'hui
    .venv_new/bin/python Magasines/evaluer_recommandations.py --asof 2027-03-20
    .venv_new/bin/python Magasines/evaluer_recommandations.py --sans-reseau    # store Parquet seul

Cours : lus d'abord dans le store Parquet (stock-analysis-ui/src/market_parquet/prices),
0 requête ; seuls les tickers absents ou incomplets sont téléchargés, par lots
(yf.download(chunk, group_by="ticker")), jamais titre par titre.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STORE_PRICES = BASE_DIR.parent / "stock-analysis-ui" / "src" / "market_parquet" / "prices"
# Cours téléchargés par ce script : réutilisés aux relances, jamais redemandés à yfinance.
CACHE_DIR = BASE_DIR / "pipeline_recos" / "cours"
HORIZON_JOURS = 182
CHUNK = 50

# Indice de référence d'après le suffixe Yahoo ; sans suffixe = cotation américaine.
BENCHMARKS = {
    ".DE": "^GDAXI", ".PA": "^FCHI", ".BR": "^BFX", ".AS": "^AEX", ".MI": "FTSEMIB.MI",
    ".MC": "^IBEX", ".VI": "^ATX", ".IR": "^ISEQ", ".L": "^FTSE", ".SW": "^SSMI",
    ".ST": "^OMX", ".OL": "OBX.OL", ".CO": "^OMXC25", ".T": "^N225", ".SI": "^STI",
    ".NZ": "^NZ50", ".TO": "^GSPTSE", "": "^GSPC",
}


def benchmark_for(ticker: str) -> str:
    """Indice de la place de cotation du ticker."""
    m = re.search(r"\.[A-Z]+$", ticker)
    return BENCHMARKS.get(m.group(0) if m else "", "^GSPC")


def parse_price(text: str) -> tuple[float | None, str | None]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lit un prix du tableau ("55,75 €", "1.190,00 €", "15,05 $", "190 Euro").
        Les cotes en % (obligations) et les champs "?" sont ignorés.

    Inputs:
        text (str): prix tel qu'imprimé

    Outputs:
        (valeur, devise) (tuple): (None, None) si illisible
    --------------------------------------------------------------------------
    """
    text = (text or "").strip()
    if not text or "?" in text or "%" in text:
        return None, None
    cur = "€" if re.search(r"€|euro", text, re.I) else "$" if "$" in text else "£" if "£" in text else None
    m = re.search(r"\d[\d.]*(?:,\d+)?", text)
    if not m:
        return None, None
    val = float(m.group(0).replace(".", "").replace(",", "."))
    return (val, cur) if val > 0 else (None, None)


def ratios(row: dict) -> tuple[float | None, float | None]:
    """Objectif et stop en multiple du cours imprimé (même devise requise)."""
    c, cc = parse_price(row.get("cours", ""))
    o, oc = parse_price(row.get("objectif", ""))
    s, sc = parse_price(row.get("stop", ""))
    if not c:
        return None, None
    tr = o / c if o and oc == cc and o > c else None
    sr = s / c if s and sc == cc and s < c else None
    return tr, sr


# ---------------------------------------------------------------------------
# Cours
# ---------------------------------------------------------------------------

def _from_store(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Historique OHLC depuis le store Parquet (0 requête réseau)."""
    import duckdb
    out = {}
    con = duckdb.connect()
    for t in tickers:
        files = glob.glob(str(STORE_PRICES / f"symbol={t}" / "*.parquet"))
        if not files:
            continue
        df = con.execute(
            "select trade_date, high, low, close from read_parquet(?) order by trade_date", [files]
        ).df()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            out[t] = df.set_index("trade_date")[["high", "low", "close"]].dropna()
    return out


def last_business_day(d: dt.date) -> dt.date:
    """Dernier jour ouvré <= d (samedi/dimanche -> vendredi ; jours fériés non gérés)."""
    return d - dt.timedelta(days=max(0, d.weekday() - 4))


def _cache_path(t: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9.\-]", "_", t) + ".parquet")


def _from_cache(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        p = _cache_path(t)
        if p.exists():
            out[t] = pd.read_parquet(p)
    return out


def _merge(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame | None:
    """Union de deux historiques ; b (plus récent) l'emporte sur les dates communes."""
    if a is None:
        return b
    if b is None:
        return a
    return pd.concat([a, b]).groupby(level=0).last().sort_index()


def _covers(df: pd.DataFrame | None, start: dt.date, end: dt.date) -> bool:
    return df is not None and not df.empty and df.index.min().date() <= start \
        and df.index.max().date() >= last_business_day(end)


def _from_yfinance(tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
    """Téléchargement groupé, par lots de CHUNK tickers."""
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    out = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        logger.info("[YF] téléchargement groupé de %d tickers", len(chunk))
        try:
            data = yf.download(chunk, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
                               group_by="ticker", auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            logger.error("[YF] échec du lot : %s", exc)
            continue
        for t in chunk:
            try:
                df = data[t][["High", "Low", "Close"]].dropna()
            except (KeyError, TypeError):
                continue
            if not df.empty:
                df.columns = ["high", "low", "close"]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                out[t] = df
    return out


def load_prices(needs: dict[str, tuple[dt.date, dt.date]], offline: bool) -> dict[str, pd.DataFrame]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Historiques couvrant la fenêtre voulue par ticker : store d'abord, puis
        yfinance groupé pour les tickers absents ou dont le store s'arrête trop tôt.

    Inputs:
        needs (dict): ticker -> (premier jour requis, dernier jour requis)
        offline (bool): True = store seul

    Outputs:
        prices (dict): ticker -> DataFrame(high, low, close) indexé par date
    --------------------------------------------------------------------------
    """
    store, cache = _from_store(list(needs)), _from_cache(list(needs))
    prices = {t: _merge(store.get(t), cache.get(t)) for t in needs}
    prices = {t: df for t, df in prices.items() if df is not None}
    missing = [t for t, (start, end) in needs.items() if not _covers(prices.get(t), start, end)]
    logger.info("[COURS] %d tickers couverts (store + cache), %d à télécharger", len(needs) - len(missing), len(missing))
    if missing and not offline:
        start = min(needs[t][0] for t in missing) - dt.timedelta(days=7)
        end = max(needs[t][1] for t in missing)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for t, df in _from_yfinance(missing, start, end).items():
            df.to_parquet(_cache_path(t))
            prices[t] = _merge(prices.get(t), df)
        still = [t for t in missing if not _covers(prices.get(t), *needs[t])]
        if still:
            logger.warning("[COURS] %d tickers toujours incomplets (limite yfinance ?) : relancer plus tard. %s",
                           len(still), ", ".join(still))
    return prices


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_row(px: pd.DataFrame | None, bench: pd.DataFrame | None, entree: dt.date,
                 asof: dt.date, horizon: int, target_ratio: float | None,
                 stop_ratio: float | None) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Évalue une recommandation sur [premier jour de cotation >= entrée,
        min(entrée + horizon, asof)]. L'objectif et le stop sont appliqués en
        multiples du cours d'entrée ; le premier touché (plus haut / plus bas
        journalier) l'emporte, le stop si les deux le sont le même jour.

    Inputs:
        px, bench (DataFrame | None): historiques high/low/close du titre et de l'indice
        entree, asof (date): date d'entrée, date d'évaluation
        horizon (int): jours calendaires jusqu'à l'échéance
        target_ratio, stop_ratio (float | None): objectif / cours, stop / cours

    Outputs:
        result (dict): statut, perf, perf_indice, exces, dates et cours
    --------------------------------------------------------------------------
    """
    echeance = entree + dt.timedelta(days=horizon)
    fin = min(echeance, asof)
    res = {"echeance": echeance.isoformat(), "statut": "sans cours", "cours_entree": None,
           "cours_fin": None, "perf": None, "perf_indice": None, "exces": None,
           "date_evenement": None}
    if px is None:
        return res
    if px.empty or px.index.max().date() < last_business_day(fin):
        # Historique tronqué : évaluer dessus donnerait un faux résultat.
        res["statut"] = "cours incomplets"
        return res
    win = px[(px.index >= pd.Timestamp(entree)) & (px.index <= pd.Timestamp(fin))]
    if win.empty:
        res["statut"] = "pas encore coté depuis l'entrée"
        return res
    if len(win) < 2:  # seulement le jour d'entrée : aucune performance mesurable
        res["statut"] = "en cours"
        return res
    p0 = float(win["close"].iloc[0])
    p1 = float(win["close"].iloc[-1])
    res.update(cours_entree=round(p0, 4), cours_fin=round(p1, 4), perf=p1 / p0 - 1)
    if bench is not None:
        bw = bench[(bench.index >= pd.Timestamp(entree)) & (bench.index <= pd.Timestamp(fin))]
        if not bw.empty:
            res["perf_indice"] = float(bw["close"].iloc[-1] / bw["close"].iloc[0] - 1)
            res["exces"] = res["perf"] - res["perf_indice"]
    after = win.iloc[1:]  # le jour d'entrée ne compte pas : on achète à sa clôture
    hit_t = after.index[after["high"] >= p0 * target_ratio] if target_ratio else []
    hit_s = after.index[after["low"] <= p0 * stop_ratio] if stop_ratio else []
    first_t = hit_t[0] if len(hit_t) else None
    first_s = hit_s[0] if len(hit_s) else None
    if first_s is not None and (first_t is None or first_s <= first_t):
        res.update(statut="stop touché", date_evenement=first_s.date().isoformat())
    elif first_t is not None:
        res.update(statut="objectif atteint", date_evenement=first_t.date().isoformat())
    elif asof >= echeance:
        res["statut"] = "échu gagnant" if res["perf"] > 0 else "échu perdant"
    else:
        res["statut"] = "en cours"
    return res


def summarize(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Taux de réussite par groupe ; seules les lignes clôturées (échues ou objectif/stop) comptent."""
    def agg(g: pd.DataFrame) -> pd.Series:
        clos = g[g["statut"].isin(["objectif atteint", "stop touché", "échu gagnant", "échu perdant"])]
        avec_obj = clos[clos["objectif_ratio"].notna()]
        cote = g[g["perf"].notna()]
        pct = lambda n, d: f"{100 * n / d:.0f} %" if d else "n.d."
        return pd.Series({
            "recos": len(g),
            "évaluables": len(cote),
            "clôturées": len(clos),
            "objectif atteint": pct((avec_obj["statut"] == "objectif atteint").sum(), len(avec_obj)),
            "stop touché": pct((avec_obj["statut"] == "stop touché").sum(), len(avec_obj)),
            "gagnantes": pct((cote["perf"] > 0).sum(), len(cote)),
            "battent l'indice": pct((cote["exces"] > 0).sum(), cote["exces"].notna().sum()),
            "perf. moyenne": f"{100 * cote['perf'].mean():+.1f} %" if len(cote) else "n.d.",
            "perf. médiane": f"{100 * cote['perf'].median():+.1f} %" if len(cote) else "n.d.",
            "écart moyen à l'indice": f"{100 * cote['exces'].mean():+.1f} pts" if cote["exces"].notna().any() else "n.d.",
        })
    if by is None:
        return agg(df).to_frame("Total").T
    return df.groupby(by, sort=True).apply(agg)


def to_md(df: pd.DataFrame, index_name: str) -> str:
    cols = [index_name] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Évalue les recommandations d'achat des magazines.")
    ap.add_argument("--csv", type=Path, help="tableau des recommandations (défaut : le plus récent)")
    ap.add_argument("--asof", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--horizon", type=int, default=HORIZON_JOURS, help="jours jusqu'à l'échéance")
    ap.add_argument("--sans-reseau", action="store_true", help="store Parquet seul, aucune requête")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    src = args.csv or sorted(BASE_DIR.glob("Recommandations_achat_*.csv"))[-1]
    recos = pd.read_csv(src, dtype=str).fillna("")
    recos = recos[recos["ticker"] != ""].copy()
    logger.info("[EVAL] %s : %d recommandations avec ticker, évaluées au %s", src.name, len(recos), args.asof)

    recos["entree_d"] = recos["date_entree"].map(dt.date.fromisoformat)
    needs: dict[str, tuple[dt.date, dt.date]] = {}
    for _, r in recos.iterrows():
        for t in (r["ticker"], benchmark_for(r["ticker"])):
            end = min(r["entree_d"] + dt.timedelta(days=args.horizon), args.asof)
            s0, e0 = needs.get(t, (r["entree_d"], end))
            needs[t] = (min(s0, r["entree_d"]), max(e0, end))
    prices = load_prices(needs, args.sans_reseau)

    out = []
    for _, r in recos.iterrows():
        tr, sr = ratios(r)
        res = evaluate_row(prices.get(r["ticker"]), prices.get(benchmark_for(r["ticker"])),
                           r["entree_d"], args.asof, args.horizon, tr, sr)
        out.append({"titre": r["entreprise"], "ticker": r["ticker"], "magazine": r["magazine"],
                    "page": r["page"], "type": r["type"].split(" (")[0], "horizon": r["horizon"],
                    "date_entree": r["date_entree"], "objectif_ratio": tr, "stop_ratio": sr,
                    "indice": benchmark_for(r["ticker"]), **res})
    ev = pd.DataFrame(out)

    stem = f"Evaluation_recommandations_{args.asof.isoformat()}"
    ev.to_csv(BASE_DIR / f"{stem}.csv", index=False)

    fmt = lambda v: "" if pd.isna(v) else f"{100 * v:+.1f} %"
    n_encours = (ev["statut"] == "en cours").sum()
    L = [f"# Évaluation des recommandations d'achat au {args.asof.isoformat()}", "",
         f"Source : `{src.name}`. Horizon : {args.horizon} jours après la date d'entrée "
         "(date du numéro pour Börse Online, date de création du PDF pour les mensuels). "
         "Entrée à la clôture du premier jour coté ; objectif et stop appliqués en % du cours imprimé ; "
         "le premier touché (plus haut/plus bas du jour) l'emporte, le stop en cas d'égalité. "
         "Indice = indice principal de la place de cotation.", ""]
    if n_encours:
        L += [f"> **Résultat provisoire** : {n_encours} recommandations sur {len(ev)} n'ont pas atteint "
              "leur échéance. Les taux « objectif atteint / stop touché » ne portent que sur les lignes "
              "clôturées ; « gagnantes » et « battent l'indice » portent sur la performance à ce jour.", ""]
    L += ["## Vue d'ensemble", "", to_md(summarize(ev, None), "Périmètre"), "",
          "## Par magazine", "", to_md(summarize(ev, "magazine"), "Magazine"), "",
          "## Par type de recommandation", "", to_md(summarize(ev, "type"), "Type"), "",
          "## Par horizon annoncé", "", to_md(summarize(ev, "horizon"), "Horizon"), "",
          "## Détail", "",
          "| Titre | Ticker | Source | Type | Entrée | Statut | Événement | Perf. | Indice | Écart |",
          "|---|---|---|---|---|---|---|--:|--:|--:|"]
    for _, r in ev.sort_values(["statut", "perf"], ascending=[True, False]).iterrows():
        L.append(f"| {r['titre']} | {r['ticker']} | {r['magazine']} p. {r['page']} | {r['type']} | "
                 f"{r['date_entree']} | {r['statut']} | {r['date_evenement'] or ''} | {fmt(r['perf'])} | "
                 f"{fmt(r['perf_indice'])} | {fmt(r['exces'])} |")
    (BASE_DIR / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("[EVAL] écrit %s.md et %s.csv", stem, stem)
    logger.info("[EVAL] statuts : %s", ev["statut"].value_counts().to_dict())


if __name__ == "__main__":
    main()
