#!/usr/bin/env python3
"""
Big Growth Scanner — Analyse les 5 conditions de croissance structurelle
sur tous les symboles de popular_symbols.txt.

Usage:
    python Big_Growth_scan.py                  # Tous les symboles
    python Big_Growth_scan.py --min-score 3    # Seulement score >= 3
    python Big_Growth_scan.py --top 50         # Top 50
    python Big_Growth_scan.py --workers 20     # 20 threads parallèles
"""
import os
import sys
import time
import logging
import argparse
import threading
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from cache_db import (
        refresh_symbol_incremental,
        get_latest_feature_row,
        ensure_market_data_schema,
        upsert_instrument,
        store_price_history,
        store_fundamental_snapshot,
        store_daily_feature_series,
        _build_daily_feature_frame,
    )
    MARKET_DB_AVAILABLE = True
    # Initialiser les répertoires Parquet dès le démarrage
    try:
        ensure_market_data_schema()
    except Exception:
        pass
except Exception:
    MARKET_DB_AVAILABLE = False

# Supprimer les logs HTTP parasites (404, rate-limit)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# Throttle global : délai minimum entre 2 appels yfinance
_throttle_lock = threading.Lock()
_last_call_time = 0.0
THROTTLE_DELAY = 0.25  # ~4 req/s max
USE_MARKET_DB = True
DB_REFRESH_HOURS = 20
DB_RECALC_WINDOW_DAYS = 420
DB_BOOTSTRAP_START = "2016-01-01"


def _safe_float(value):
    """Convertit en float fini si possible, sinon None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "y"}


def _is_valid_history(hist):
    """Valide rapidement l'historique de prix/volume."""
    if hist is None or getattr(hist, "empty", True):
        return False
    if "Close" not in hist.columns or "Volume" not in hist.columns:
        return False

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(hist["Volume"], errors="coerce").dropna()

    if len(close) < 130 or len(volume) < 90:
        return False
    if (close <= 0).all():
        return False
    if (volume < 0).any():
        return False
    return True


def _get_fast_info_snapshot(stock):
    """Récupère les champs rapides de yfinance sans lever d'exception."""
    try:
        fi = stock.fast_info
    except Exception:
        return {}

    if fi is None:
        return {}

    out = {}
    for src_key, dst_key in [
        ("lastPrice", "currentPrice"),
        ("last_price", "currentPrice"),
        ("marketCap", "marketCap"),
        ("market_cap", "marketCap"),
        ("currency", "currency"),
        ("quoteType", "quoteType"),
    ]:
        try:
            val = fi.get(src_key)
            if val is not None:
                out[dst_key] = val
        except Exception:
            continue

    return out


def _is_valid_info(info):
    """Vérifie qu'on a au moins une identité ou un prix exploitable."""
    if not isinstance(info, dict):
        return False
    if not info:
        return False

    has_identity = bool(info.get("shortName") or info.get("longName") or info.get("symbol"))
    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))

    return has_identity or (price is not None and price > 0)


def _fetch_info_with_retry(stock, ticker, max_retries=4):
    """Télécharge les métadonnées avec retry/backoff et fallback fast_info."""
    last_info = {}
    for attempt in range(max_retries):
        try:
            _throttle()
            info = stock.info or {}
            if isinstance(info, dict):
                last_info = info
        except Exception:
            info = {}

        fast_info = _get_fast_info_snapshot(stock)
        merged = dict(last_info) if isinstance(last_info, dict) else {}
        merged.update({k: v for k, v in fast_info.items() if v is not None})
        merged.setdefault("symbol", ticker)

        if _is_valid_info(merged):
            return merged

        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))

    return None


def _fetch_history_with_retry(stock, max_retries=4):
    """Télécharge l'historique avec retry/backoff et validation."""
    for attempt in range(max_retries):
        try:
            _throttle()
            hist = stock.history(period="18mo", auto_adjust=False)
            if _is_valid_history(hist):
                return hist
        except Exception:
            pass

        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))

    return None


def _result_from_db_row(ticker, row):
    """Convertit une ligne DB en format de sortie Big_Growth_scan."""
    if not row:
        return None

    price = _safe_float(row.get("price"))
    mc_b = _safe_float(row.get("market_cap_b"))
    if mc_b is None:
        imc = _safe_float(row.get("instrument_market_cap"))
        mc_b = round(imc / 1e9, 2) if imc else None

    name = row.get("name") or row.get("short_name") or row.get("long_name") or row.get("instrument_name") or ticker
    sector = row.get("sector") or row.get("instrument_sector") or "N/A"
    score = row.get("score")
    score = int(score) if score is not None else 0

    return {
        "ticker": ticker,
        "nom": name,
        "secteur": sector,
        "prix": round(price, 2) if price and price > 0 else None,
        "market_cap_B": mc_b,
        "score": score,
        "C1_CA_Growth_%": _safe_float(row.get("c1_revenue_growth_pct")),
        "C1_ok": _to_bool(row.get("c1_ok")),
        "C2_Marge_%": _safe_float(row.get("c2_gross_margin_pct")),
        "C2_ok": _to_bool(row.get("c2_ok")),
        "C3_Underval": row.get("c3_undervaluation_hits"),
        "C3_ok": _to_bool(row.get("c3_ok")),
        "C4_Momentum_%": _safe_float(row.get("c4_momentum_3m_pct")),
        "C4_ok": _to_bool(row.get("c4_ok")),
        "C5_VolRatio": _safe_float(row.get("c5_volume_ratio")),
        "C5_ok": _to_bool(row.get("c5_ok")),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    """Détecte une erreur 429 / rate limit yfinance."""
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "too many requests", "rate limited"))


def _analyze_from_market_db(ticker):
    """Pipeline DB: refresh incrémental puis lecture de la dernière ligne calculée.

    Si le refresh réseau échoue mais que des données existent déjà en cache,
    on retourne la dernière ligne connue plutôt que de tomber en fallback.
    En cas de rate-limit, on attend le throttle avant de laisser le fallback
    yfinance direct prendre le relai (ou on retourne le cache si disponible).
    """
    if not MARKET_DB_AVAILABLE:
        return None

    # Lire d'abord la ligne existante (peut être None pour un nouveau symbole)
    existing_row = get_latest_feature_row(ticker)

    # Throttle global avant tout appel réseau (évite les 429 en mode multi-thread)
    _throttle()

    try:
        refresh_symbol_incremental(
            ticker,
            min_refresh_hours=DB_REFRESH_HOURS,
            recalc_window_days=DB_RECALC_WINDOW_DAYS,
            bootstrap_start_date=DB_BOOTSTRAP_START,
            force=False,
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            # Rate-limit : retourner le cache si dispo pour ne pas bloquer les threads
            if existing_row is not None:
                return _result_from_db_row(ticker, existing_row)
            # Pas de cache → laisser le fallback yfinance tenter (avec son propre backoff)
            return None
        # Autre erreur réseau : même comportement
        if existing_row is not None:
            return _result_from_db_row(ticker, existing_row)
        return None

    row = get_latest_feature_row(ticker)
    return _result_from_db_row(ticker, row)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES SYMBOLES
# ─────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
POPULAR_FILE = SRC_DIR / "popular_symbols.txt"
OUTPUT_CSV = SRC_DIR / "big_growth_results.csv"


def load_symbols(path=POPULAR_FILE):
    """Charge les symboles depuis le fichier texte (un par ligne)."""
    symbols = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                symbols.append(s)
    return list(dict.fromkeys(symbols))  # dédoublonner en gardant l'ordre


# ─────────────────────────────────────────────────────────────
# 5 CONDITIONS (identiques à Big_Growth.py)
# ─────────────────────────────────────────────────────────────
def c1_revenue_growth(info):
    """C1 — Croissance structurelle du CA (>20% YoY)"""
    rg = info.get("revenueGrowth")
    if rg is None:
        return False, None, "N/A"
    passed = rg > 0.20
    return passed, round(rg * 100, 1), f"CA +{round(rg*100,1)}% YoY"


def c2_gross_margin(info):
    """C2 — Marges brutes en expansion (>30%)"""
    gm = info.get("grossMargins")
    if gm is None:
        return False, None, "N/A"
    passed = gm > 0.30
    return passed, round(gm * 100, 1), f"Marge brute {round(gm*100,1)}%"


def c3_undervaluation(info):
    """C3 — Sous-valorisation (2 proxies sur 3 : P/E<25, PEG<1.5, prix<75% 52wH)"""
    hits = 0
    details = []

    pe = info.get("trailingPE")
    if pe and 0 < pe < 25:
        hits += 1
        details.append(f"P/E={round(pe,1)}")

    peg = info.get("pegRatio")
    if peg and 0 < peg < 1.5:
        hits += 1
        details.append(f"PEG={round(peg,2)}")

    high_52w = info.get("fiftyTwoWeekHigh")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if high_52w and price and high_52w > 0:
        ratio = price / high_52w
        if ratio < 0.75:
            hits += 1
            details.append(f"Prix@{round(ratio*100)}%52wH")

    return hits >= 2, hits, " | ".join(details) or "N/A"


def c4_nascent_momentum(hist):
    """C4 — Momentum naissant (3m +8-60%, > SMA50, 6m < 150%)"""
    if hist is None or len(hist) < 130:
        return False, None, "Historique insuffisant"

    close = hist["Close"]
    current = float(close.iloc[-1])
    p3m = float(close.iloc[-63])
    p6m = float(close.iloc[-126])

    if p3m == 0 or p6m == 0:
        return False, None, "Prix nul"

    ret_3m = (current - p3m) / p3m
    ret_6m = (current - p6m) / p6m
    sma50 = float(close.rolling(50).mean().iloc[-1])

    passed = (0.08 < ret_3m < 0.60) and (current > sma50) and (ret_6m < 1.50)
    return passed, round(ret_3m * 100, 1), f"3m={round(ret_3m*100,1)}% 6m={round(ret_6m*100,1)}%"


def c5_volume_buildup(hist):
    """C5 — Accumulation institutionnelle (vol30j / vol90j > 1.20x)"""
    if hist is None or len(hist) < 90:
        return False, None, "Historique insuffisant"

    vol_30 = float(hist["Volume"].iloc[-30:].mean())
    vol_90 = float(hist["Volume"].iloc[-90:].mean())

    if vol_90 == 0:
        return False, None, "Volume nul"

    ratio = vol_30 / vol_90
    return ratio > 1.20, round(ratio, 2), f"Vol30/90={round(ratio,2)}x"


# ─────────────────────────────────────────────────────────────
# ANALYSE D'UN SYMBOLE
# ─────────────────────────────────────────────────────────────
def _throttle():
    """Limite globale du débit d'appels yfinance."""
    global _last_call_time
    with _throttle_lock:
        now = time.monotonic()
        wait = THROTTLE_DELAY - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


def analyze(ticker):
    """Retourne un dict de résultats ou None en cas d'erreur."""
    if USE_MARKET_DB and MARKET_DB_AVAILABLE:
        db_result = _analyze_from_market_db(ticker)
        if db_result is not None:
            return db_result

    # Fallback réseau direct (si DB indisponible ou vide)
    stock = yf.Ticker(ticker)
    info = _fetch_info_with_retry(stock, ticker)
    if info is None:
        return None

    # Vérifier que le ticker est valide et exploitable
    name = info.get("shortName") or info.get("longName") or info.get("symbol")
    price_probe = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if not name and (price_probe is None or price_probe <= 0):
        return None

    hist = _fetch_history_with_retry(stock)
    if hist is None:
        return None

    r1 = c1_revenue_growth(info)
    r2 = c2_gross_margin(info)
    r3 = c3_undervaluation(info)
    r4 = c4_nascent_momentum(hist)
    r5 = c5_volume_buildup(hist)

    score = sum(1 for r in [r1, r2, r3, r4, r5] if r[0])

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    mc = _safe_float(info.get("marketCap"))
    mc_b = round(mc / 1e9, 2) if mc else None

    result = {
        "ticker": ticker,
        "nom": name or ticker,
        "secteur": info.get("sector", "N/A"),
        "prix": round(price, 2) if price and price > 0 else None,
        "market_cap_B": mc_b,
        "score": score,
        "C1_CA_Growth_%": r1[1],
        "C1_ok": r1[0],
        "C2_Marge_%": r2[1],
        "C2_ok": r2[0],
        "C3_Underval": r3[1],
        "C3_ok": r3[0],
        "C4_Momentum_%": r4[1],
        "C4_ok": r4[0],
        "C5_VolRatio": r5[1],
        "C5_ok": r5[0],
    }

    # Stocker dans la DB pour que les prochains scans n'aient qu'à télécharger
    # les jours manquants au lieu de retélécharger tout l'historique.
    if USE_MARKET_DB and MARKET_DB_AVAILABLE:
        try:
            currency = info.get("currency") or "USD"
            upsert_instrument(ticker, info)
            store_fundamental_snapshot(ticker, info)
            store_price_history(ticker, hist, currency)
            feature_frame = _build_daily_feature_frame(ticker, hist, info)
            store_daily_feature_series(ticker, feature_frame)
        except Exception:
            pass  # Échec silencieux : le résultat calculé reste valide

    return result


def analyze_safe(ticker):
    """Wrapper avec gestion d'erreur + retry avec backoff exponentiel."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return analyze(ticker)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["429", "rate", "too many"])
            is_not_found = "404" in err_str or "not found" in err_str

            if is_not_found:
                return None  # ticker invalide, pas la peine de réessayer

            if is_rate_limit:
                # Rate-limit : attente plus longue mais on ne bloque pas les autres threads
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s
                else:
                    return None  # Abandon propre (pas d'erreur dans le résumé)
            elif attempt < max_retries - 1:
                time.sleep(0.3 * (2 ** attempt))
            else:
                return {"ticker": ticker, "nom": "ERREUR", "secteur": "N/A",
                        "prix": None, "market_cap_B": None, "score": -1,
                        "C1_CA_Growth_%": None, "C1_ok": False,
                        "C2_Marge_%": None, "C2_ok": False,
                        "C3_Underval": None, "C3_ok": False,
                        "C4_Momentum_%": None, "C4_ok": False,
                        "C5_VolRatio": None, "C5_ok": False}
    return None


# ─────────────────────────────────────────────────────────────
# SCAN PARALLÈLE
# ─────────────────────────────────────────────────────────────
def run_scan(symbols, max_workers=10, min_score=3, top_n=None, verbose=True):
    results = []
    total = len(symbols)
    done = 0
    errors = 0
    skipped = 0
    t0 = time.time()

    print(f"\n🚀 Big Growth Scan — {total} symboles, {max_workers} threads\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_safe, s): s for s in symbols}

        for future in as_completed(futures):
            done += 1
            sym = futures[future]
            result = future.result()

            if result is None:
                skipped += 1
            elif result.get("score", -1) < 0:
                errors += 1
            else:
                results.append(result)
                if verbose and result["score"] >= 3:
                    print(f"  🔥 {result['ticker']:8s} {result['nom'][:25]:25s} "
                          f"Score={result['score']}/5  "
                          f"Secteur={result['secteur']}")

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{total}] {rate:.1f}/s — "
                      f"{len(results)} valides, {errors} erreurs, {skipped} ignorés "
                      f"— ETA {eta:.0f}s")

    # Tri + filtrage
    df = pd.DataFrame(results)
    if df.empty:
        print("\n❌ Aucun résultat.")
        return df

    df = df.sort_values("score", ascending=False)

    if min_score > 0:
        df = df[df["score"] >= min_score]

    if top_n:
        df = df.head(top_n)

    elapsed = time.time() - t0
    print(f"\n✅ Scan terminé en {elapsed:.0f}s — {len(df)} résultats (score >= {min_score})")

    return df


def print_summary(df):
    if df.empty:
        return

    display_cols = ["ticker", "nom", "secteur", "prix", "market_cap_B", "score",
                    "C1_CA_Growth_%", "C2_Marge_%", "C3_Underval",
                    "C4_Momentum_%", "C5_VolRatio"]
    available = [c for c in display_cols if c in df.columns]

    # Tronquer les noms longs pour l'affichage console
    df_display = df[available].copy()
    if "nom" in df_display.columns:
        df_display["nom"] = df_display["nom"].astype(str).str[:25]

    print(f"\n{'='*120}")
    print("  CLASSEMENT BIG GROWTH")
    print(f"{'='*120}")

    # Par tranche de score
    for score_val in range(5, -1, -1):
        subset = df_display[df_display["score"] == score_val]
        if subset.empty:
            continue
        emoji = "🔥" * score_val if score_val > 0 else "⚪"
        print(f"\n  ─── Score {score_val}/5 {emoji} ({len(subset)} stocks) ───")
        print(subset.to_string(index=False))

    # Stats par secteur
    print(f"\n{'='*120}")
    print("  STATS PAR SECTEUR (score >= 3)")
    print(f"{'='*120}")
    high = df[df["score"] >= 3]
    if not high.empty:
        sector_stats = high.groupby("secteur").agg(
            count=("score", "size"),
            avg_score=("score", "mean"),
            avg_growth=("C1_CA_Growth_%", "mean"),
            avg_margin=("C2_Marge_%", "mean"),
        ).sort_values("count", ascending=False)
        print(sector_stats.to_string())
    else:
        print("  Aucun stock avec score >= 3")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Big Growth Scanner")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Score minimum à afficher (0-5)")
    parser.add_argument("--top", type=int, default=None,
                        help="Nombre max de résultats")
    parser.add_argument("--workers", type=int, default=5,
                        help="Threads parallèles (défaut: 5)")
    parser.add_argument("--throttle", type=float, default=0.25,
                        help="Délai minimum global entre appels yfinance, en secondes (défaut: 0.25)")
    parser.add_argument("--no-db", action="store_true",
                        help="Désactiver la base évolutive et forcer le mode yfinance direct")
    parser.add_argument("--db-refresh-hours", type=int, default=20,
                        help="Fréquence minimale de refresh DB par symbole (heures, défaut: 20)")
    parser.add_argument("--db-recalc-days", type=int, default=420,
                        help="Fenêtre de recalcul technique lors d'un refresh DB (jours, défaut: 420)")
    parser.add_argument("--db-bootstrap-start", type=str, default="2016-01-01",
                        help="Date de départ initiale pour un symbole absent de la DB (défaut: 2016-01-01)")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="Fichier de symboles (défaut: popular_symbols.txt)")
    parser.add_argument("--random", type=int, default=None, metavar="N",
                        help="Sélectionner N symboles au hasard dans le fichier")
    parser.add_argument("--seed", type=int, default=None,
                        help="Graine aléatoire pour reproductibilité (utiliser avec --random)")
    parser.add_argument("--quiet", action="store_true",
                        help="Réduire l'affichage pendant le scan")
    args = parser.parse_args()

    sym_file = Path(args.symbols_file) if args.symbols_file else POPULAR_FILE
    if not sym_file.exists():
        print(f"❌ Fichier introuvable: {sym_file}")
        sys.exit(1)

    symbols = load_symbols(sym_file)

    if args.random is not None:
        import random
        n = min(max(1, args.random), len(symbols))
        rng = random.Random(args.seed)
        symbols = rng.sample(symbols, n)
        seed_info = f", seed={args.seed}" if args.seed is not None else ""
        print(f"🎲 Sélection aléatoire: {n}/{len(load_symbols(sym_file))} symboles{seed_info}")

    THROTTLE_DELAY = max(0.05, args.throttle)
    USE_MARKET_DB = (not args.no_db) and MARKET_DB_AVAILABLE
    DB_REFRESH_HOURS = max(1, int(args.db_refresh_hours))
    DB_RECALC_WINDOW_DAYS = max(120, int(args.db_recalc_days))
    DB_BOOTSTRAP_START = args.db_bootstrap_start

    print(f"📋 {len(symbols)} symboles chargés depuis {sym_file.name}")
    print(f"⏱️ Throttle global yfinance: {THROTTLE_DELAY:.2f}s")
    if USE_MARKET_DB:
        print(
            f"🗄️ Mode DB évolutive actif (refresh={DB_REFRESH_HOURS}h, "
            f"recalc={DB_RECALC_WINDOW_DAYS}j, bootstrap={DB_BOOTSTRAP_START})"
        )
    elif not MARKET_DB_AVAILABLE and not args.no_db:
        print("⚠️ Module DB indisponible, fallback automatique en mode yfinance direct")
    else:
        print("🛰️ Mode yfinance direct (DB désactivée)")

    df = run_scan(
        symbols,
        max_workers=args.workers,
        min_score=args.min_score,
        top_n=args.top,
        verbose=not args.quiet,
    )

    if not df.empty:
        # Sauvegarder le CSV complet
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Résultats sauvegardés: {OUTPUT_CSV}")

        # Afficher le résumé
        print_summary(df)

        # Quick stats
        print(f"\n📊 Distribution des scores:")
        for s in range(5, -1, -1):
            count = len(df[df["score"] == s])
            if count > 0:
                bar = "█" * (count // 2) if count > 1 else "█"
                print(f"  {s}/5: {count:4d} {bar}")
