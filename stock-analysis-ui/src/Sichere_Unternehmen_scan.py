#!/usr/bin/env python3
"""
Sichere Unternehmen Scanner — Recherche les entreprises les plus sûres
basé sur 7 critères fondamentaux:

  C1 — Marktkapitalisierung  > 10 Mrd. €
  C2 — Verschuldungsgrad     < 100 %
  C3 — Beta                  < 0.8
  C4 — Dividendenrendite     > 0 %
  C5 — Free-Cashflow-Marge   > 5 %
  C6 — Ø Wachstum FCF 5J    > 0 %
  C7 — Ø Wachstum CA+EPS 5J > 3 %

Usage:
  python Sichere_Unternehmen_scan.py --min-score 5
  python Sichere_Unternehmen_scan.py --top 50
  python Sichere_Unternehmen_scan.py --workers 10
  python Sichere_Unternehmen_scan.py --random 100 --seed 42
"""
import os, sys, time, logging, argparse, threading, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from cache_db import (
        refresh_symbol_incremental, get_latest_feature_row,
        ensure_market_data_schema, upsert_instrument,
        store_price_history, store_fundamental_snapshot,
        store_daily_feature_series, _build_daily_feature_frame,
    )
    MARKET_DB_AVAILABLE = True
    try:
        ensure_market_data_schema()
    except Exception:
        pass
except Exception:
    MARKET_DB_AVAILABLE = False

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

_throttle_lock = threading.Lock()
_last_call_time = 0.0
THROTTLE_DELAY = 0.25
_skip_lock = threading.Lock()
_skip_reasons: dict[str, int] = {}
USE_MARKET_DB = True
EUR_USD_RATE = 1.08

# ── Utilitaires ──────────────────────────────────────────────
def _safe_float(value):
    if value is None: return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None

def _to_bool(value):
    if value is None: return False
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return value != 0
    return str(value).strip().lower() in {"1","true","yes","y"}

def _throttle():
    global _last_call_time
    with _throttle_lock:
        now = time.monotonic()
        wait = THROTTLE_DELAY - (now - _last_call_time)
        if wait > 0: time.sleep(wait)
        _last_call_time = time.monotonic()

def _record_skip(reason):
    with _skip_lock:
        _skip_reasons[reason] = _skip_reasons.get(reason, 0) + 1

def _is_valid_history(hist):
    if hist is None or getattr(hist, "empty", True): return False
    if "Close" not in hist.columns or "Volume" not in hist.columns: return False
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(hist["Volume"], errors="coerce").dropna()
    return len(close) >= 130 and len(volume) >= 90 and not (close <= 0).all()

def _get_fast_info_snapshot(stock):
    try: fi = stock.fast_info
    except Exception: return {}
    if fi is None: return {}
    out = {}
    for src, dst in [("lastPrice","currentPrice"),("last_price","currentPrice"),
                     ("marketCap","marketCap"),("market_cap","marketCap"),
                     ("currency","currency"),("quoteType","quoteType")]:
        try:
            val = fi.get(src)
            if val is not None: out[dst] = val
        except Exception: continue
    return out

def _is_valid_info(info):
    if not isinstance(info, dict) or not info: return False
    has_id = bool(info.get("shortName") or info.get("longName") or info.get("symbol"))
    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    return has_id or (price is not None and price > 0)

def _fetch_info_with_retry(stock, ticker, max_retries=4):
    last_info = {}
    for attempt in range(max_retries):
        try:
            _throttle()
            info = stock.info or {}
            if isinstance(info, dict): last_info = info
        except Exception: pass
        fast = _get_fast_info_snapshot(stock)
        merged = dict(last_info)
        merged.update({k: v for k, v in fast.items() if v is not None})
        merged.setdefault("symbol", ticker)
        if _is_valid_info(merged): return merged
        if attempt < max_retries - 1: time.sleep(0.8 * (2 ** attempt))
    return None

def _fetch_history_with_retry(stock, max_retries=4):
    for attempt in range(max_retries):
        try:
            _throttle()
            hist = stock.history(period="3y", auto_adjust=False)
            if _is_valid_history(hist): return hist
        except Exception: pass
        if attempt < max_retries - 1: time.sleep(0.8 * (2 ** attempt))
    return None

def _is_rate_limit_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ("429","rate limit","too many requests","rate limited"))

# ── 7 CRITÈRES ───────────────────────────────────────────────
def c1_market_cap(info):
    mc = _safe_float(info.get("marketCap"))
    if mc is None: return False, None, "N/A"
    currency = (info.get("currency") or "USD").upper()
    mc_eur = mc / EUR_USD_RATE if currency == "USD" else mc
    mc_b = mc_eur / 1e9
    return mc_b > 10.0, round(mc_b, 1), f"MCap={round(mc_b,1)}Mrd€"

def c2_debt_to_equity(info):
    de = _safe_float(info.get("debtToEquity"))
    if de is None: return False, None, "N/A"
    return de < 100.0, round(de, 1), f"D/E={round(de,1)}%"

def c3_beta(info, hist):
    beta = _safe_float(info.get("beta"))
    if beta is None: return False, None, "N/A"
    return 0 < beta < 0.8, round(beta, 2), f"Beta={round(beta,2)}"

def c4_dividend_yield(info):
    dy = _safe_float(info.get("dividendYield")) or _safe_float(info.get("trailingAnnualDividendYield"))
    if dy is None: return False, None, "N/A"
    dy_pct = dy * 100
    return dy_pct > 0, round(dy_pct, 2), f"Div={round(dy_pct,2)}%"

def c5_fcf_margin(info):
    fcf = _safe_float(info.get("freeCashflow"))
    rev = _safe_float(info.get("totalRevenue"))
    if fcf is None or rev is None or rev == 0:
        ocf = _safe_float(info.get("operatingCashflow"))
        capex = _safe_float(info.get("capitalExpenditures"))
        if ocf is not None and capex is not None and rev and rev > 0:
            fcf = ocf - abs(capex)
        else:
            return False, None, "N/A"
    margin = (fcf / rev) * 100
    return margin > 5.0, round(margin, 1), f"FCFmarge={round(margin,1)}%"

def c6_fcf_growth_5y(info):
    growth = _safe_float(info.get("earningsQuarterlyGrowth")) or _safe_float(info.get("earningsGrowth"))
    fcf = _safe_float(info.get("freeCashflow"))
    if fcf is None:
        ocf = _safe_float(info.get("operatingCashflow"))
        capex = _safe_float(info.get("capitalExpenditures"))
        if ocf is not None and capex is not None:
            fcf = ocf - abs(capex)
    if fcf is None: return False, None, "N/A"
    if growth is not None:
        g_pct = growth * 100
        return fcf > 0 and g_pct > 0, round(g_pct, 1), f"FCF+EG={round(g_pct,1)}%"
    rev_g = _safe_float(info.get("revenueGrowth"))
    if rev_g is not None:
        return fcf > 0 and rev_g > 0, round(rev_g*100,1), f"FCF+RG={round(rev_g*100,1)}%"
    return fcf > 0, None, f"FCF={'pos' if fcf > 0 else 'neg'}"

def c7_revenue_eps_growth_5y(info):
    rev_g = _safe_float(info.get("revenueGrowth"))
    eps_g = _safe_float(info.get("earningsGrowth"))
    if rev_g is None and eps_g is None: return False, None, "N/A"
    values = [g for g in [rev_g, eps_g] if g is not None]
    avg_pct = sum(values) / len(values) * 100
    if rev_g is not None and eps_g is not None:
        passed = rev_g * 100 > 3.0 and eps_g * 100 > 3.0
        detail = f"RevG={round(rev_g*100,1)}% EpsG={round(eps_g*100,1)}%"
    else:
        passed = avg_pct > 3.0
        detail = f"AvgG={round(avg_pct,1)}%"
    return passed, round(avg_pct, 1), detail

# ── FICHIERS ─────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
POPULAR_FILE = SRC_DIR / "popular_symbols.txt"
OUTPUT_CSV   = SRC_DIR / "sichere_unternehmen_results.csv"

def load_symbols(path=POPULAR_FILE):
    symbols = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                symbols.append(s)
    return list(dict.fromkeys(symbols))

# ── ANALYSE ──────────────────────────────────────────────────
def analyze(ticker):
    stock = yf.Ticker(ticker)
    info = _fetch_info_with_retry(stock, ticker)
    if info is None:
        _record_skip("Pas de données (yfinance)"); return None
    name = info.get("shortName") or info.get("longName") or info.get("symbol")
    price_p = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if not name and (price_p is None or price_p <= 0):
        _record_skip("Ticker invalide"); return None
    hist = _fetch_history_with_retry(stock)

    r1 = c1_market_cap(info)
    r2 = c2_debt_to_equity(info)
    r3 = c3_beta(info, hist)
    r4 = c4_dividend_yield(info)
    r5 = c5_fcf_margin(info)
    r6 = c6_fcf_growth_5y(info)
    r7 = c7_revenue_eps_growth_5y(info)
    score = sum(1 for r in [r1,r2,r3,r4,r5,r6,r7] if r[0])

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    mc = _safe_float(info.get("marketCap"))

    result = {
        "ticker": ticker, "nom": name or ticker,
        "secteur": info.get("sector","N/A"), "pays": info.get("country","N/A"),
        "devise": info.get("currency","N/A"),
        "prix": round(price,2) if price and price>0 else None,
        "market_cap_B": round(mc/1e9,2) if mc else None,
        "score": score,
        "C1_MCap_Mrd€": r1[1], "C1_ok": r1[0],
        "C2_DE_%":       r2[1], "C2_ok": r2[0],
        "C3_Beta":        r3[1], "C3_ok": r3[0],
        "C4_DivYield_%":  r4[1], "C4_ok": r4[0],
        "C5_FCF_Marge_%": r5[1], "C5_ok": r5[0],
        "C6_FCF_Growth_%":r6[1], "C6_ok": r6[0],
        "C7_RevEPS_%":    r7[1], "C7_ok": r7[0],
    }
    if USE_MARKET_DB and MARKET_DB_AVAILABLE and hist is not None:
        try:
            upsert_instrument(ticker, info)
            store_fundamental_snapshot(ticker, info)
            store_price_history(ticker, hist, info.get("currency","USD"))
            store_daily_feature_series(ticker, _build_daily_feature_frame(ticker, hist, info))
        except Exception: pass
    return result

def analyze_safe(ticker):
    for attempt in range(3):
        try:
            return analyze(ticker)
        except Exception as e:
            err = str(e).lower()
            if "404" in err or "not found" in err: return None
            if any(k in err for k in ["429","rate","too many"]):
                if attempt < 2: time.sleep(2**(attempt+1))
                else: _record_skip("Rate-limit (429)"); return None
            elif attempt < 2: time.sleep(0.3*(2**attempt))
            else:
                return {"ticker":ticker,"nom":"ERREUR","secteur":"N/A","pays":"N/A",
                        "devise":"N/A","prix":None,"market_cap_B":None,"score":-1,
                        "C1_MCap_Mrd€":None,"C1_ok":False,"C2_DE_%":None,"C2_ok":False,
                        "C3_Beta":None,"C3_ok":False,"C4_DivYield_%":None,"C4_ok":False,
                        "C5_FCF_Marge_%":None,"C5_ok":False,"C6_FCF_Growth_%":None,
                        "C6_ok":False,"C7_RevEPS_%":None,"C7_ok":False}
    return None

# ── SCAN ─────────────────────────────────────────────────────
def run_scan(symbols, max_workers=10, min_score=0, top_n=None, verbose=True):
    results, done, errors, skipped = [], 0, 0, 0
    total = len(symbols)
    t0 = time.time()
    print(f"\n🛡️  Sichere Unternehmen Scan — {total} symboles, {max_workers} threads")
    print(f"    C1=MCap>10Mrd€ | C2=D/E<100% | C3=Beta<0.8 | C4=Div>0%")
    print(f"    C5=FCF_M>5%    | C6=FCF_G>0% | C7=Rev+EPS>3%\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_safe, s): s for s in symbols}
        for future in as_completed(futures):
            done += 1
            sym = futures[future]
            result = future.result()
            if result is None: skipped += 1
            elif result.get("score",-1) < 0: errors += 1
            else:
                results.append(result)
                if verbose and result["score"] >= 5:
                    chk = "".join("✅" if result[f"C{i}_ok"] else "❌" for i in range(1,8))
                    print(f"  🛡️  {result['ticker']:8s} {result['nom'][:25]:25s} "
                          f"Score={result['score']}/7 {chk}")
            if done % 100 == 0 or done == total:
                el = time.time()-t0; rate = done/el if el>0 else 0
                eta = (total-done)/rate if rate>0 else 0
                print(f"  [{done:4d}/{total}] {rate:.1f}/s — {len(results)} valides, "
                      f"{errors} err, {skipped} skip — ETA {eta:.0f}s")

    df = pd.DataFrame(results)
    if df.empty: print("\n❌ Aucun résultat."); return df
    df = df.sort_values("score", ascending=False)
    if min_score > 0: df = df[df["score"] >= min_score]
    if top_n: df = df.head(top_n)
    elapsed = time.time()-t0
    print(f"\n✅ Terminé en {elapsed:.0f}s — {len(df)} résultats (score >= {min_score})")
    if skipped > 0 and _skip_reasons:
        print(f"\n⚠️  Raisons skip:")
        for r,c in sorted(_skip_reasons.items(), key=lambda x:-x[1]):
            print(f"  {c:4d}x {r}")
    return df

def print_summary(df):
    if df.empty: return
    cols = ["ticker","nom","secteur","pays","prix","market_cap_B","score",
            "C1_MCap_Mrd€","C2_DE_%","C3_Beta","C4_DivYield_%",
            "C5_FCF_Marge_%","C6_FCF_Growth_%","C7_RevEPS_%"]
    avail = [c for c in cols if c in df.columns]
    dfd = df[avail].copy()
    if "nom" in dfd.columns: dfd["nom"] = dfd["nom"].str[:25]
    print(f"\n{'='*150}")
    print(" SICHERSTE UNTERNEHMEN — Classement")
    print(f"{'='*150}")
    for s in range(7,-1,-1):
        sub = dfd[dfd["score"]==s]
        if sub.empty: continue
        print(f"\n  ─── Score {s}/7 {'🛡️'*s} ({len(sub)} stocks) ───")
        print(sub.to_string(index=False))
    print(f"\n{'='*150}")
    print(" STATS PAR SECTEUR (score >= 5)")
    high = df[df["score"]>=5]
    if not high.empty:
        print(high.groupby("secteur").agg(
            count=("score","size"), avg_score=("score","mean"),
            avg_beta=("C3_Beta","mean"), avg_div=("C4_DivYield_%","mean"),
            avg_fcf=("C5_FCF_Marge_%","mean"),
        ).sort_values("count",ascending=False).to_string())
    else:
        print("  Aucun stock avec score >= 5")

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sichere Unternehmen Scanner")
    parser.add_argument("--min-score",  type=int,   default=0)
    parser.add_argument("--top",        type=int,   default=None)
    parser.add_argument("--workers",    type=int,   default=5)
    parser.add_argument("--throttle",   type=float, default=0.25)
    parser.add_argument("--no-db",      action="store_true")
    parser.add_argument("--symbols-file", type=str, default=None)
    parser.add_argument("--random",     type=int,   default=None, metavar="N")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--eur-usd",    type=float, default=1.08)
    args = parser.parse_args()

    EUR_USD_RATE = args.eur_usd
    THROTTLE_DELAY = max(0.05, args.throttle)
    USE_MARKET_DB = (not args.no_db) and MARKET_DB_AVAILABLE

    sym_file = Path(args.symbols_file) if args.symbols_file else POPULAR_FILE
    if not sym_file.exists():
        print(f"❌ Fichier introuvable: {sym_file}"); sys.exit(1)

    symbols = load_symbols(sym_file)
    if args.random is not None:
        import random
        rng = random.Random(args.seed)
        symbols = rng.sample(symbols, min(max(1,args.random), len(symbols)))
        print(f"🎲 Sélection aléatoire: {len(symbols)} symboles")

    print(f"📋 {len(symbols)} symboles | ⏱️ Throttle={THROTTLE_DELAY}s | 💱 EUR/USD={EUR_USD_RATE}")

    df = run_scan(symbols, max_workers=args.workers, min_score=args.min_score,
                  top_n=args.top, verbose=not args.quiet)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Sauvegardé: {OUTPUT_CSV}")
        print_summary(df)
        print(f"\n📊 Distribution des scores:")
        for s in range(7,-1,-1):
            c = len(df[df["score"]==s])
            if c > 0: print(f"  {s}/7: {c:4d} {'█'*(c//2 or 1)}")