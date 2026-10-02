#!/usr/bin/env python3
"""
Combined Scanner — Big Growth + Sichere Unternehmen
=====================================================
Évalue SIMULTANÉMENT les deux stratégies :

  🚀 BIG GROWTH (5 critères)
  G1 — Croissance CA       > 20% YoY
  G2 — Marge brute         > 30%
  G3 — Sous-valorisation   >= 2/3 proxies (P/E<25, PEG<1.5, Prix<75% 52wH)
  G4 — Momentum naissant   3m +8-60%, > SMA50, 6m < 150%
  G5 — Accumulation volume Vol30j/Vol90j > 1.20x

  🛡️ SICHERE UNTERNEHMEN (7 critères)
  S1 — Marktkapitalisierung > 10 Mrd. €
  S2 — Verschuldungsgrad    < 100%
  S3 — Beta                 < 0.8
  S4 — Dividendenrendite    > 0%
  S5 — Free-Cashflow-Marge  > 5%
  S6 — Croissance FCF      > 0 %/an  (taux compose, exercices annuels yfinance : 4)
  S7 — Croissance CA et BPA > 3 %/an (les deux, exercices annuels)

  Criteres, devises et couche reseau : core/scan_fondamentaux.py (commun aux scanners).

  💎 PROFILS
  Dual Champion : G>=3 ET S>=5
  Dual Champion*: Dual Champion qui remplit aussi G3, G4 et S4 (backtest du 02.10.2026)
  Pure Growth   : G>=4, S<3
  Pure Safe     : S>=5, G<3
  Balanced      : G>=3, S>=3

Usage:
  python Combined_scan.py
  python Combined_scan.py --profile dual --top 30
  python Combined_scan.py --min-growth 3 --min-safe 4
  python Combined_scan.py --random 200 --workers 10
"""
import sys, time, logging, argparse, threading, math, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from cache_db import (
        ensure_market_data_schema, upsert_instrument,
        store_price_history, store_fundamental_snapshot,
        store_daily_feature_series, _build_daily_feature_frame,
    )
    MARKET_DB_AVAILABLE = True
    try: ensure_market_data_schema()
    except Exception: pass
except Exception:
    MARKET_DB_AVAILABLE = False

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Couche reseau, devises et criteres : code commun des scanners (voir le module).
from core import scan_fondamentaux as sf
from core.scan_fondamentaux import (
    g1_revenue_growth, g2_gross_margin, g3_undervaluation, g4_momentum, g5_volume,
    s1_market_cap, s2_debt_equity, s3_beta, s4_dividend, s5_fcf_margin,
    s6_fcf_growth, s7_rev_eps_growth, get_profile,
)

USE_MARKET_DB = True
_safe_float   = sf.safe_float
_record_skip  = sf.record_skip
_skip_reasons = sf.SKIP_REASONS

# ═══════════════════════════════════════════════════════════════
# FICHIERS
# ═══════════════════════════════════════════════════════════════

SRC_DIR      = Path(__file__).parent
POPULAR_FILE = SRC_DIR / "popular_symbols.txt"
OUTPUT_CSV   = SRC_DIR / "combined_results.csv"

def load_symbols(path=POPULAR_FILE):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return list(dict.fromkeys(out))

# ═══════════════════════════════════════════════════════════════
# ANALYSE D'UN SYMBOLE
# ═══════════════════════════════════════════════════════════════

def analyze(ticker):
    stock = yf.Ticker(ticker)
    info  = sf.fetch_info(stock, ticker)
    if info is None:
        _record_skip("Pas de données"); return None

    name = info.get("shortName") or info.get("longName") or ticker
    hist = sf.fetch_hist(stock)

    # ── Big Growth ──
    rg1 = g1_revenue_growth(info)
    rg2 = g2_gross_margin(info)
    rg3 = g3_undervaluation(info)
    rg4 = g4_momentum(hist)
    rg5 = g5_volume(hist)
    growth_score = sum(1 for r in [rg1,rg2,rg3,rg4,rg5] if r[0])

    # ── Sichere Unternehmen ──
    rs1 = s1_market_cap(info)
    rs2 = s2_debt_equity(info)
    rs3 = s3_beta(info)
    rs4 = s4_dividend(info)
    etats = sf.etats_annuels(*sf.fetch_statements(stock))
    rs5 = s5_fcf_margin(info, etats)
    rs6 = s6_fcf_growth(info, ticker, etats)
    rs7 = s7_rev_eps_growth(info, ticker, etats)
    safe_score = sum(1 for r in [rs1,rs2,rs3,rs4,rs5,rs6,rs7] if r[0])

    total_score = growth_score + safe_score
    profile     = get_profile(growth_score, safe_score, sf.est_etoile(rg3[0], rg4[0], rs4[0]))

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    mc    = _safe_float(info.get("marketCap"))

    result = {
        "ticker":       ticker,
        "nom":          name,
        "secteur":      info.get("sector", "N/A"),
        "pays":         info.get("country", "N/A"),
        "devise":       info.get("currency", "N/A"),
        "prix":         round(price, 2) if price and price > 0 else None,
        "market_cap_B": round(mc/1e9, 2) if mc else None,
        # Scores
        "score_total":  total_score,
        "score_growth": growth_score,
        "score_safe":   safe_score,
        "profil":       profile,
        # Big Growth détail
        "G1_CA_Growth_%":  rg1[1], "G1_ok": rg1[0],
        "G2_Marge_%":      rg2[1], "G2_ok": rg2[0],
        "G3_Underval":     rg3[1], "G3_ok": rg3[0],
        "G4_Momentum_%":   rg4[1], "G4_ok": rg4[0],
        "G5_VolRatio":     rg5[1], "G5_ok": rg5[0],
        # Sichere détail
        "S1_MCap_Mrd€":    rs1[1], "S1_ok": rs1[0],
        "S2_DE_%":         rs2[1], "S2_ok": rs2[0],
        "S3_Beta":         rs3[1], "S3_ok": rs3[0],
        "S4_DivYield_%":   rs4[1], "S4_ok": rs4[0],
        "S5_FCF_Marge_%":  rs5[1], "S5_ok": rs5[0],
        "S6_FCF_Growth_%": rs6[1], "S6_ok": rs6[0],
        "S7_RevEPS_%":     rs7[1], "S7_ok": rs7[0],
    }

    if USE_MARKET_DB and MARKET_DB_AVAILABLE and hist is not None:
        try:
            upsert_instrument(ticker, info)
            store_fundamental_snapshot(ticker, info)
            store_price_history(ticker, hist, info.get("currency", "USD"))
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
                else: _record_skip("Rate-limit 429"); return None
            elif attempt < 2: time.sleep(0.3*(2**attempt))
            else: return None
    return None

# ═══════════════════════════════════════════════════════════════
# SCAN PARALLÈLE
# ═══════════════════════════════════════════════════════════════

def run_scan(symbols, max_workers=10, min_growth=0, min_safe=0,
             profile_filter=None, top_n=None, verbose=True):
    results, done, skipped = [], 0, 0
    total = len(symbols)
    t0    = time.time()

    print(f"\n{'='*70}")
    print("  COMBINED SCANNER — Big Growth + Sichere Unternehmen")
    print(f"  {total} symboles | {max_workers} threads")
    print(f"{'='*70}")
    print("  🚀 G1=CA>20% | G2=Marge>30% | G3=Underval | G4=Momentum | G5=Volume")
    print("  🛡️  S1=MCap>10Mrd€ | S2=D/E<100% | S3=Beta<0.8 | S4=Div>0%")
    print("      S5=FCF_M>5% | S6=FCF_G>0% | S7=Rev+EPS>3%")
    print("  💎 Dual Champion = G>=3 ET S>=5  |  💎 Dual Champion* = Dual + G3 + G4 + S4")
    print(f"{'='*70}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_safe, s): s for s in symbols}
        for future in as_completed(futures):
            done   += 1
            result  = future.result()
            if result is None:
                skipped += 1
            else:
                results.append(result)
                gs, ss = result["score_growth"], result["score_safe"]
                if verbose and (gs >= 3 or ss >= 5):
                    g_chk = "".join("✅" if result[f"G{i}_ok"] else "❌" for i in range(1,6))
                    s_chk = "".join("✅" if result[f"S{i}_ok"] else "❌" for i in range(1,8))
                    print(f"  {result['profil']}  {result['ticker']:8s} "
                          f"{result['nom'][:20]:20s} "
                          f"G={gs}/5[{g_chk}] S={ss}/7[{s_chk}]")

            if done % 100 == 0 or done == total:
                el   = time.time()-t0
                rate = done/el if el > 0 else 0
                eta  = (total-done)/rate if rate > 0 else 0
                print(f"  [{done:4d}/{total}] {rate:.1f}/s — "
                      f"{len(results)} résultats — ETA {eta:.0f}s")

    df = pd.DataFrame(results)
    if df.empty:
        print("\n❌ Aucun résultat."); return df

    df = df.sort_values(["score_total","score_safe","score_growth"], ascending=False)

    if min_growth > 0:    df = df[df["score_growth"] >= min_growth]
    if min_safe   > 0:    df = df[df["score_safe"]   >= min_safe]
    if profile_filter:
        pmap = {"star":   "💎 Dual Champion*",
                "dual":   "💎 Dual Champion",
                "growth": "🚀 Pure Growth",
                "safe":   "🛡️  Pure Safe",
                "balanced":"⚖️  Balanced"}
        key = profile_filter.lower()
        if key == "dual":   # le Dual Champion* reste un Dual Champion
            df = df[df["profil"].map(sf.est_dual)]
        elif key in pmap:
            df = df[df["profil"] == pmap[key]]
    if top_n:
        df = df.head(top_n)

    elapsed = time.time()-t0
    print(f"\n✅ Terminé en {elapsed:.0f}s — {len(df)} résultats filtrés")
    if skipped > 0 and _skip_reasons:
        print("\n⚠️  Skips :")
        for r,c in sorted(_skip_reasons.items(), key=lambda x:-x[1]):
            print(f"  {c:4d}x {r}")
    return df

# ═══════════════════════════════════════════════════════════════
# RÉSUMÉ
# ═══════════════════════════════════════════════════════════════

def print_summary(df):
    if df.empty: return

    print(f"\n{'='*160}")
    print("  RÉSUMÉ — COMBINED SCANNER")
    print(f"{'='*160}")

    print("\n  📊 Répartition par profil :")
    for profil in ["💎 Dual Champion*","💎 Dual Champion","🚀 Pure Growth","🛡️  Pure Safe","⚖️  Balanced","⚪ Below"]:
        sub = df[df["profil"] == profil]
        if not sub.empty:
            bar = "█" * min(len(sub), 50)
            print(f"  {profil:22s}: {len(sub):4d}  {bar}")

    for profil in ["💎 Dual Champion*","💎 Dual Champion","🚀 Pure Growth","🛡️  Pure Safe","⚖️  Balanced"]:
        sub = df[df["profil"] == profil]
        if sub.empty: continue
        print(f"\n  {'─'*155}")
        print(f"  {profil}  ({len(sub)} stocks)")
        print(f"  {'─'*155}")
        cols = ["ticker","nom","secteur","pays","prix","market_cap_B",
                "score_growth","score_safe","score_total",
                "G1_CA_Growth_%","G2_Marge_%","G3_Underval","G4_Momentum_%","G5_VolRatio",
                "S1_MCap_Mrd€","S2_DE_%","S3_Beta","S4_DivYield_%","S5_FCF_Marge_%"]
        avail = [c for c in cols if c in sub.columns]
        dfd   = sub[avail].copy()
        if "nom" in dfd.columns:
            dfd["nom"] = dfd["nom"].str[:20]
        print(dfd.head(20).to_string(index=False))

    dual = df[df["profil"].map(sf.est_dual)]
    if not dual.empty:
        print(f"\n  {'='*155}")
        print("  SECTEURS — 💎 Dual Champions")
        stats = dual.groupby("secteur").agg(
            count          =("score_total","size"),
            avg_growth     =("score_growth","mean"),
            avg_safe       =("score_safe","mean"),
            avg_total      =("score_total","mean"),
            avg_beta       =("S3_Beta","mean"),
            avg_div        =("S4_DivYield_%","mean"),
            avg_ca_growth  =("G1_CA_Growth_%","mean"),
        ).sort_values("count", ascending=False)
        print(stats.to_string())

    print("\n  📊 Distribution score total (max=12) :")
    for s in range(12, -1, -1):
        c = len(df[df["score_total"] == s])
        if c > 0:
            bar = "█" * min(c//2 or 1, 50)
            print(f"  {s:2d}/12: {c:4d}  {bar}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Scanner — Big Growth + Sichere Unternehmen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python Combined_scan.py --profile dual --top 30
  python Combined_scan.py --min-growth 3 --min-safe 4
  python Combined_scan.py --random 300 --workers 10 --seed 42
  python Combined_scan.py --profile safe --min-safe 6
        """
    )
    parser.add_argument("--min-growth",   type=int,   default=0)
    parser.add_argument("--min-safe",     type=int,   default=0)
    parser.add_argument("--profile",      type=str,   default=None,
                        help="star | dual | growth | safe | balanced (dual inclut star)")
    parser.add_argument("--top",          type=int,   default=None)
    parser.add_argument("--workers",      type=int,   default=5)
    parser.add_argument("--throttle",     type=float, default=0.25)
    parser.add_argument("--no-db",        action="store_true")
    parser.add_argument("--symbols-file", type=str,   default=None)
    parser.add_argument("--random",       type=int,   default=None, metavar="N")
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--quiet",        action="store_true")
    parser.add_argument("--eur-usd",      type=float, default=1.08)
    args = parser.parse_args()

    sf.charger_taux_eur(verbose=not args.quiet, eur_usd_repli=args.eur_usd)
    THROTTLE_DELAY = sf.regler_debit(args.throttle)
    USE_MARKET_DB  = (not args.no_db) and MARKET_DB_AVAILABLE

    sym_file = Path(args.symbols_file) if args.symbols_file else (Path(__file__).parent / "popular_symbols.txt")
    if not sym_file.exists():
        print(f"❌ Fichier introuvable : {sym_file}"); sys.exit(1)

    symbols = load_symbols(sym_file)
    if args.random is not None:
        n   = min(max(1, args.random), len(symbols))
        rng = random.Random(args.seed)
        symbols = rng.sample(symbols, n)
        seed_info = f", seed={args.seed}" if args.seed is not None else ""
        print(f"🎲 Sélection aléatoire : {n} symboles{seed_info}")

    print(f"📋 {len(symbols)} symboles | ⏱️ Throttle={THROTTLE_DELAY}s | 💱 EUR/USD={sf.EUR_RATES.get('USD', float('nan')):.4f} ({len(sf.EUR_RATES)-1} devises)")
    if USE_MARKET_DB: print("🗄️  Mode DB évolutive actif")
    else:             print("🛰️  Mode yfinance direct")

    df = run_scan(
        symbols,
        max_workers    = args.workers,
        min_growth     = args.min_growth,
        min_safe       = args.min_safe,
        profile_filter = args.profile,
        top_n          = args.top,
        verbose        = not args.quiet,
    )

    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Résultats sauvegardés : {OUTPUT_CSV}")
        print_summary(df)