#!/usr/bin/env python3
"""
news_monitor_combined.py
------------------------
Pipeline complet fusionné :
  1. Surveille les flux RSS financiers
  2a. Analyse l'impact sur votre portfolio (LLM Ollama)
  2b. Découvre de nouveaux tickers dans les news (LLM Ollama)
  3. Valide via 12 critères combinés :
       🚀 Big Growth (G1-G5)   : CA>20%, Marge>30%, Underval, Momentum, Volume
       🛡️  Sichere (S1-S7)      : MCap>10Mrd€, D/E<100%, Beta<0.8, Div>0%,
                                   FCF_M>5%, FCF_G>0%, Rev+EPS>3%
  4. Attribue un profil : 💎 Dual / 🚀 Growth / 🛡️ Safe / ⚖️ Balanced
  5. Génère rapport Markdown + CSV enrichi

Usage :
  python news_monitor_combined.py          # boucle continue
  python news_monitor_combined.py --once   # une passe unique
  python news_monitor_combined.py --interval 600
  python news_monitor_combined.py --no-fundamentals
"""

import argparse, asyncio, csv, json, math, os, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp, feedparser, requests
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder-v2:16b"

BASE_DIR       = Path(__file__).parent
SRC_DIR        = BASE_DIR.parent
PORTFOLIO_FILE = SRC_DIR / "mes_symbols.txt"
ALERTS_CSV     = BASE_DIR / "news_alerts.csv"
TICKER_CSV     = BASE_DIR / "ticker_summary.csv"
SEEN_IDS_FILE  = BASE_DIR / ".news_seen_ids.json"
REPORT_DIR     = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

DEFAULT_INTERVAL_SEC  = 900
MAX_DISCOVERED_TICKERS = 15
FUNDAMENTAL_MIN_SCORE  = 2   # score Growth minimum pour apparaître (sur 5)
SAFE_MIN_SCORE         = 3   # score Sichere minimum pour apparaître (sur 7)

EUR_USD_RATE = 1.08          # ajustable via --eur-usd

RSS_FEEDS = {
    # News générales
    "Reuters Finance"  : "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Tech"     : "https://feeds.reuters.com/reuters/technologyNews",
    "Yahoo Finance"    : "https://finance.yahoo.com/news/rssindex",
    "MarketWatch"      : "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC"             : "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    # Sources officielles gratuites
    "SEC EDGAR 8-K"    : "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&owner=include&count=20&output=atom",
    "PR Newswire"      : "https://www.prnewswire.com/rss/news-releases-list.rss",
    # Europe
    "Zonebourse"       : "https://www.zonebourse.com/rss/actualites.xml",
    "Investing.com FR" : "https://fr.investing.com/rss/news.rss",
    "Les Echos"        : "https://syndication.lesechos.fr/rss/rss_finance.xml",
}

KEYWORD_MAP = {
    "NVDA": ["nvidia","nvda","blackwell","jensen huang","h100","b200","cuda"],
    "AVGO": ["broadcom","avgo","vmware","ai networking"],
    "MU"  : ["micron","hbm","dram","nand","memory chip"],
    "MRVL": ["marvell","mrvl","custom asic"],
    "AGEN": ["agenus","checkpoint inhibitor","immunotherapy"],
    "ALDX": ["aldeyra","aldx","reproxalap"],
    "IMNM": ["immunomedics","imnm","trodelvy"],
    "MBOT": ["microbot","mbot","robotic surgery micro"],
    "KGC" : ["kinross","kgc","gold mine","gold production"],
    "MSTR": ["microstrategy","mstr","michael saylor","bitcoin reserve"],
    "GS"  : ["goldman sachs","goldman"],
    "DB"  : ["deutsche bank","db"],
    "CF"  : ["cf industries","nitrogen fertilizer","ammonia"],
    "H"   : ["hyatt","hotel chain","hospitality earnings"],
    "SEMR": ["semrush","semr","seo platform"],
}

TICKER_FIELDS = [
    "ticker","nom","secteur","pays","devise","type","date_derniere_alerte","nb_alertes",
    "sentiment_dominant","intensite_max","horizons","catalyseurs","resume_principal",
    # Big Growth
    "score_growth","G1_CA","G1_ok","G2_Marge","G2_ok","G3_Underval","G3_ok",
    "G4_Momentum","G4_ok","G5_Volume","G5_ok",
    # Sichere
    "score_safe","S1_MCap","S1_ok","S2_DE","S2_ok","S3_Beta","S3_ok",
    "S4_Div","S4_ok","S5_FCF_M","S5_ok","S6_FCF_G","S6_ok","S7_RevEPS","S7_ok",
    # Synthèse
    "score_total","profil","conviction_opp","direction_opp",
    "prix","market_cap_B","titres","liens",
]

INTENSITY_RANK = {"fort": 3, "modere": 2, "faible": 1, "": 0}

# ── Utilitaires ──────────────────────────────────────────────

def _f(v):
    if v is None: return None
    try: f = float(v)
    except (TypeError, ValueError): return None
    return f if math.isfinite(f) else None

def load_portfolio():
    if not PORTFOLIO_FILE.exists(): return []
    return [l.strip() for l in PORTFOLIO_FILE.read_text().splitlines() if l.strip()]

def load_seen_ids():
    if SEEN_IDS_FILE.exists(): return set(json.loads(SEEN_IDS_FILE.read_text()))
    return set()

def save_seen_ids(ids):
    SEEN_IDS_FILE.write_text(json.dumps(list(ids)[-2000:]))

def is_market_hours():
    from datetime import time as dtime
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and dtime(4,0) <= now.time() <= dtime(20,0)

def get_price_context(tickers):
    lines = []
    for t in tickers[:6]:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                last  = hist["Close"].iloc[-1]
                prev  = hist["Close"].iloc[0]
                pct   = (last-prev)/prev*100
                vratio= hist["Volume"].iloc[-1] / hist["Volume"].mean()
                lines.append(f"{t}: ${last:.2f} ({pct:+.1f}% 5j) vol x{vratio:.1f}")
        except Exception: pass
    return " | ".join(lines) if lines else "N/A"

# ── RSS ──────────────────────────────────────────────────────

def fetch_news(max_per_feed=8):
    articles  = []
    all_feeds = dict(RSS_FEEDS)
    for t in load_portfolio()[:10]:
        all_feeds[f"Yahoo/{t}"] = f"https://finance.yahoo.com/rss/headline?s={t}"
    for source, url in all_feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                aid = entry.get("id") or entry.get("link","")
                articles.append({
                    "id"       : aid,
                    "source"   : source,
                    "title"    : entry.get("title",""),
                    "summary"  : entry.get("summary", entry.get("description",""))[:600],
                    "link"     : entry.get("link",""),
                    "published": entry.get("published",""),
                })
        except Exception as e:
            print(f"[WARN] Flux {source}: {e}")
    return articles

# ── LLM ──────────────────────────────────────────────────────

def _call_llm(prompt, label):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.1, "num_predict": 1024},
        }, timeout=180)
        r.raise_for_status()
        raw = r.json().get("response","")
        s, e = raw.find("{"), raw.rfind("}")+1
        if s == -1 or e == 0: return None
        return json.loads(raw[s:e])
    except Exception as ex:
        print(f"[WARN] LLM '{label[:50]}': {ex}")
        return None

def quick_filter(article, portfolio):
    text = (article["title"]+" "+article["summary"]).lower()
    return [t for t in portfolio if any(kw in text for kw in KEYWORD_MAP.get(t,[t.lower()]))]

def analyze_portfolio_impact(article, portfolio, matched_tickers):
    price_ctx = get_price_context(matched_tickers)
    prompt = f"""Tu es analyste financier quantitatif. Analyse cette actualité.

ACTUALITÉ:
Titre: {article['title']}
Résumé: {article['summary']}
Source: {article['source']}

TICKERS PORTFOLIO TOUCHÉS: {', '.join(matched_tickers)}
CONTEXTE PRIX (5j): {price_ctx}

Réponds UNIQUEMENT en JSON valide:
{{
  "pertinent": true/false,
  "confiance": 0-100,
  "horizon": "intraday"|"1-3j"|"1-2sem"|"1-3mois",
  "tickers_affectes": [{{"ticker":"X","direction":"hausse|baisse|neutre","magnitude":"<3%|3-8%|>8%"}}],
  "catalyseur": "earnings|macro|réglementaire|M&A|produit|concurrent|sectoriel",
  "sentiment": "positif"|"negatif"|"neutre",
  "intensite": "faible"|"modere"|"fort",
  "resume_impact": "1-2 phrases max",
  "raisonnement": "max 3 phrases",
  "risque_contraire": "facteur invalidant"
}}
Si aucun impact réel, retourne pertinent: false."""
    return _call_llm(prompt, article["title"])

def discover_tickers_from_batch(articles, portfolio):
    titles_block = "\n".join(
        f"[{i+1}] {a['title']} | {a['summary'][:200]}"
        for i, a in enumerate(articles[:20])
    )
    prompt = f"""Tu es un extracteur de tickers boursiers.

ARTICLES:
{titles_block}

PORTFOLIO DÉJÀ SUIVI (exclure sauf signal fort): {', '.join(portfolio)}

Réponds UNIQUEMENT en JSON:
{{
  "opportunites": [
    {{"ticker":"XXXX","exchange":"NASDAQ|NYSE|XETRA|...","article_index":1,
      "catalyseur":"...","direction_attendue":"hausse|baisse",
      "conviction":"faible|modere|fort","resume":"1 phrase"}}
  ]
}}
Limite: max 10 opportunités. JSON uniquement."""
    result = _call_llm(prompt, "batch_discovery")
    if not result: return []
    opportunites = result.get("opportunites", [])
    # Filtrer les placeholders/exemples renvoyés tels quels par le LLM
    return [
        o for o in opportunites
        if isinstance(o.get("ticker"), str)
        and o["ticker"] not in ("XXXX", "TICKER", "SYMBOL", "", "...")
        and not o["ticker"].startswith("NASDAQ")
        and len(o["ticker"]) <= 6
    ]

# ═══════════════════════════════════════════════════════════════
# 🚀 BIG GROWTH — 5 critères
# ═══════════════════════════════════════════════════════════════

def g1_revenue_growth(info):
    v = _f(info.get("revenueGrowth"))
    if v is None: return False, "N/A"
    return v > 0.20, f"{v*100:.1f}%"

def g2_gross_margin(info):
    v = _f(info.get("grossMargins"))
    if v is None: return False, "N/A"
    return v > 0.30, f"{v*100:.1f}%"

def g3_undervaluation(info):
    hits = 0
    pe  = _f(info.get("trailingPE"))
    peg = _f(info.get("pegRatio"))
    h52 = _f(info.get("fiftyTwoWeekHigh"))
    px  = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
    if pe  and 0 < pe  < 25:  hits += 1
    if peg and 0 < peg < 1.5: hits += 1
    if h52 and px and h52>0 and (px/h52)<0.75: hits += 1
    return hits >= 2, f"{hits}/3"

def g4_momentum(hist):
    if hist is None or len(hist) < 130: return False, "N/A"
    close = hist["Close"]
    cur, p3m, p6m = float(close.iloc[-1]), float(close.iloc[-63]), float(close.iloc[-126])
    if p3m==0 or p6m==0: return False, "N/A"
    r3 = (cur-p3m)/p3m; r6 = (cur-p6m)/p6m
    sma50 = float(close.rolling(50).mean().iloc[-1])
    return (0.08<r3<0.60) and (cur>sma50) and (r6<1.50), f"{r3*100:.1f}%"

def g5_volume(hist):
    if hist is None or len(hist) < 90: return False, "N/A"
    v30 = float(hist["Volume"].iloc[-30:].mean())
    v90 = float(hist["Volume"].iloc[-90:].mean())
    if v90==0: return False, "N/A"
    ratio = v30/v90
    return ratio > 1.20, f"{ratio:.2f}x"

# ═══════════════════════════════════════════════════════════════
# 🛡️ SICHERE UNTERNEHMEN — 7 critères
# ═══════════════════════════════════════════════════════════════

def s1_market_cap(info):
    mc = _f(info.get("marketCap"))
    if mc is None: return False, "N/A"
    cur = (info.get("currency") or "USD").upper()
    mc_b = (mc/EUR_USD_RATE if cur=="USD" else mc)/1e9
    return mc_b > 10.0, f"{mc_b:.1f}Mrd€"

def s2_debt_equity(info):
    de = _f(info.get("debtToEquity"))
    if de is None: return False, "N/A"
    return de < 100.0, f"{de:.1f}%"

def s3_beta(info):
    beta = _f(info.get("beta"))
    if beta is None: return False, "N/A"
    return 0 < beta < 0.8, f"{beta:.2f}"

def s4_dividend(info):
    dy = _f(info.get("dividendYield")) or _f(info.get("trailingAnnualDividendYield"))
    if dy is None: return False, "N/A"
    return dy*100 > 0, f"{dy*100:.2f}%"

def s5_fcf_margin(info):
    fcf = _f(info.get("freeCashflow"))
    rev = _f(info.get("totalRevenue"))
    if fcf is None or rev is None or rev==0:
        ocf   = _f(info.get("operatingCashflow"))
        capex = _f(info.get("capitalExpenditures"))
        if ocf is not None and capex is not None and rev and rev>0:
            fcf = ocf - abs(capex)
        else: return False, "N/A"
    m = (fcf/rev)*100
    return m > 5.0, f"{m:.1f}%"

def s6_fcf_growth(info):
    g   = _f(info.get("earningsQuarterlyGrowth")) or _f(info.get("earningsGrowth"))
    fcf = _f(info.get("freeCashflow"))
    if fcf is None:
        ocf   = _f(info.get("operatingCashflow"))
        capex = _f(info.get("capitalExpenditures"))
        if ocf is not None and capex is not None: fcf = ocf-abs(capex)
    if fcf is None: return False, "N/A"
    if g is not None: return fcf>0 and g*100>0, f"{g*100:.1f}%"
    rg = _f(info.get("revenueGrowth"))
    if rg is not None: return fcf>0 and rg>0, f"{rg*100:.1f}%"
    return fcf>0, "FCF+" if fcf>0 else "FCF-"

def s7_rev_eps_growth(info):
    rg = _f(info.get("revenueGrowth"))
    eg = _f(info.get("earningsGrowth"))
    if rg is None and eg is None: return False, "N/A"
    vals = [g for g in [rg,eg] if g is not None]
    avg  = sum(vals)/len(vals)*100
    if rg is not None and eg is not None:
        return rg*100>3.0 and eg*100>3.0, f"{avg:.1f}%"
    return avg>3.0, f"{avg:.1f}%"

def get_profile(gs, ss):
    if gs>=3 and ss>=5: return "💎 Dual Champion"
    if gs>=4 and ss< 3: return "🚀 Pure Growth"
    if ss>=5 and gs< 3: return "🛡️  Pure Safe"
    if gs>=3 and ss>=3: return "⚖️  Balanced"
    return "⚪ Below"

# ═══════════════════════════════════════════════════════════════
# ANALYSE FONDAMENTALE COMBINÉE (12 critères)
# ═══════════════════════════════════════════════════════════════

def run_fundamental_combined(ticker):
    """Remplace run_fundamental() — évalue les 12 critères + profil."""
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="18mo", auto_adjust=False)
        info  = stock.info
        if not info or not (info.get("regularMarketPrice") or info.get("currentPrice")):
            return None

        # Big Growth
        rg1 = g1_revenue_growth(info)
        rg2 = g2_gross_margin(info)
        rg3 = g3_undervaluation(info)
        rg4 = g4_momentum(hist if not hist.empty else None)
        rg5 = g5_volume(hist if not hist.empty else None)
        gs  = sum(1 for r in [rg1,rg2,rg3,rg4,rg5] if r[0])

        # Sichere
        rs1 = s1_market_cap(info)
        rs2 = s2_debt_equity(info)
        rs3 = s3_beta(info)
        rs4 = s4_dividend(info)
        rs5 = s5_fcf_margin(info)
        rs6 = s6_fcf_growth(info)
        rs7 = s7_rev_eps_growth(info)
        ss  = sum(1 for r in [rs1,rs2,rs3,rs4,rs5,rs6,rs7] if r[0])

        total   = gs + ss
        profile = get_profile(gs, ss)
        price   = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
        mc      = _f(info.get("marketCap"))

        return {
            "ticker": ticker,
            "nom"   : info.get("shortName", ticker),
            "secteur": info.get("sector","N/A"),
            "pays"  : info.get("country","N/A"),
            "devise": info.get("currency","N/A"),
            "prix"  : price,
            "market_cap_B": round(mc/1e9,2) if mc else None,
            # Growth
            "score_growth": gs,
            "G1_CA": rg1[1],    "G1_ok": rg1[0],
            "G2_Marge": rg2[1], "G2_ok": rg2[0],
            "G3_Underval": rg3[1], "G3_ok": rg3[0],
            "G4_Momentum": rg4[1], "G4_ok": rg4[0],
            "G5_Volume": rg5[1],   "G5_ok": rg5[0],
            # Sichere
            "score_safe": ss,
            "S1_MCap": rs1[1], "S1_ok": rs1[0],
            "S2_DE"  : rs2[1], "S2_ok": rs2[0],
            "S3_Beta": rs3[1], "S3_ok": rs3[0],
            "S4_Div" : rs4[1], "S4_ok": rs4[0],
            "S5_FCF_M": rs5[1],"S5_ok": rs5[0],
            "S6_FCF_G": rs6[1],"S6_ok": rs6[0],
            "S7_RevEPS": rs7[1],"S7_ok": rs7[0],
            # Synthèse
            "score_total": total,
            "profil"     : profile,
        }
    except Exception as e:
        print(f"[WARN] Fondamentaux {ticker}: {e}")
        return None

# ── Rapport Markdown enrichi ─────────────────────────────────

def generate_report(portfolio_alerts, opportunities, fundamentals, run_time):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_path   = REPORT_DIR / f"report_{timestamp}.md"
    csv_path  = REPORT_DIR / f"report_{timestamp}.csv"

    lines = [
        f"# 📊 Rapport Marché — {run_time}",
        f"> Modèle: {OLLAMA_MODEL} | 12 critères: 🚀G1-G5 + 🛡️S1-S7",
        "",
        "## 🔔 Alertes Portfolio", "",
    ]

    if portfolio_alerts:
        for a in sorted(portfolio_alerts, key=lambda x: x.get("intensite_rank",0), reverse=True):
            icon = {"positif":"🟢","negatif":"🔴","neutre":"🟡"}.get(a.get("sentiment",""),"⚪")
            lines += [
                f"### {icon} {a['titre'][:80]}",
                f"- **Source** : {a['source']} | **Intensité** : {a.get('intensite','?').upper()}",
                f"- **Tickers** : `{a.get('tickers_affectes','N/A')}` | **Horizon** : {a.get('horizon','?')}",
                f"- **Impact** : {a.get('resume_impact','')}",
                f"- **Risque** : {a.get('risque_contraire','N/A')}",
                f"- 🔗 {a.get('lien','')}", "",
            ]
    else:
        lines += ["*Aucune alerte portfolio dans ce cycle.*", ""]

    lines += ["## 🔍 Opportunités Découvertes", ""]
    if opportunities:
        for opp in opportunities:
            ci = {"fort":"🔥","modere":"⭐","faible":"💡"}.get(opp.get("conviction",""),"")
            lines.append(
                f"- {ci} **{opp['ticker']}** ({opp.get('exchange','?')}) — "
                f"{opp.get('direction_attendue','?').upper()} | "
                f"{opp.get('catalyseur','')} — {opp.get('resume','')}"
            )
    else:
        lines += ["*Aucune nouvelle opportunité détectée.*"]
    lines += [""]

    # Tableau fondamental enrichi 12 critères
    lines += ["## 📈 Analyse Fondamentale Combinée (12 critères)", ""]
    ok_f = lambda b: "✅" if b else "❌"

    scored = [f for f in fundamentals if f and
              (f["score_growth"] >= FUNDAMENTAL_MIN_SCORE or f["score_safe"] >= SAFE_MIN_SCORE)]
    scored.sort(key=lambda x: (x["score_total"], x["score_safe"]), reverse=True)

    if scored:
        lines.append("| Ticker | Nom | Profil | G/5 | S/7 | Tot | "
                     "G1 CA | G2 Mge | G3 Val | G4 Mom | G5 Vol | "
                     "S1 MCap | S2 D/E | S3 β | S4 Div | S5 FCFm | S6 FCFg | S7 RvEps |")
        lines.append("|--------|-----|--------|-----|-----|-----|"
                     "-------|--------|--------|--------|--------|"
                     "---------|--------|------|--------|---------|---------|----------|")
        for r in scored:
            lines.append(
                f"| **{r['ticker']}** | {r['nom'][:18]} | {r['profil']} | "
                f"{r['score_growth']}/5 | {r['score_safe']}/7 | **{r['score_total']}/12** | "
                f"{ok_f(r['G1_ok'])}{r['G1_CA']} | {ok_f(r['G2_ok'])}{r['G2_Marge']} | "
                f"{ok_f(r['G3_ok'])}{r['G3_Underval']} | {ok_f(r['G4_ok'])}{r['G4_Momentum']} | "
                f"{ok_f(r['G5_ok'])}{r['G5_Volume']} | "
                f"{ok_f(r['S1_ok'])}{r['S1_MCap']} | {ok_f(r['S2_ok'])}{r['S2_DE']} | "
                f"{ok_f(r['S3_ok'])}{r['S3_Beta']} | {ok_f(r['S4_ok'])}{r['S4_Div']} | "
                f"{ok_f(r['S5_ok'])}{r['S5_FCF_M']} | {ok_f(r['S6_ok'])}{r['S6_FCF_G']} | "
                f"{ok_f(r['S7_ok'])}{r['S7_RevEPS']} |"
            )
    else:
        lines += ["*Aucun ticker ne dépasse les seuils.*"]

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # CSV
    rows = []
    for a in portfolio_alerts:
        rows.append({"type":"ALERT","ticker":a.get("tickers_affectes",""),
                     "titre":a.get("titre",""),"sentiment":a.get("sentiment",""),
                     "intensite":a.get("intensite",""),"horizon":a.get("horizon",""),
                     "resume":a.get("resume_impact",""),"profil":"",
                     "score_growth":"","score_safe":"","score_total":"",
                     "catalyseur":a.get("catalyseur",""),"lien":a.get("lien",""),
                     "date":a.get("date","")})
    for f in scored:
        rows.append({"type":"FUNDAMENTAL","ticker":f["ticker"],"titre":f["nom"],
                     "sentiment":"","intensite":"","horizon":"","resume":"",
                     "profil":f["profil"],"score_growth":f["score_growth"],
                     "score_safe":f["score_safe"],"score_total":f["score_total"],
                     "catalyseur":"","lien":"","date":run_time})
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n[RAPPORT] {md_path}")
    print(f"[RAPPORT] {csv_path}")
    return md_path, csv_path

# ── Consolidation ticker_summary.csv ─────────────────────────

def _append_f(row, field, value):
    if not value or value in ("facteur invalidant","N/A"): return
    existing = row.get(field,"")
    if value in existing: return
    row[field] = (existing+" | "+value).strip(" | ") if existing else value

def _new_row(ticker):
    return {f:"" for f in TICKER_FIELDS} | {"ticker":ticker,"nb_alertes":"0"}

def consolidate_ticker_summary(portfolio_alerts, opportunities, fundamentals, run_time):
    existing = {}
    if TICKER_CSV.exists():
        with open(TICKER_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["ticker"]] = row
    data = dict(existing)

    for alert in portfolio_alerts:
        tickers = [t.strip() for t in alert.get("tickers_affectes","").split("|") if t.strip()]
        for ticker in tickers:
            if ticker not in data: data[ticker] = _new_row(ticker)
            row = data[ticker]
            row["type"] = "BOTH" if row.get("type") in ("OPPORTUNITY","VALIDATED","BOTH") else "PORTFOLIO"
            row["date_derniere_alerte"] = run_time
            row["nb_alertes"] = str(int(row.get("nb_alertes") or 0)+1)
            cur = INTENSITY_RANK.get(row.get("intensite_max",""),0)
            new = INTENSITY_RANK.get(alert.get("intensite",""),0)
            if new > cur:
                row["intensite_max"]      = alert.get("intensite","")
                row["sentiment_dominant"] = alert.get("sentiment","")
            _append_f(row,"horizons",alert.get("horizon",""))
            _append_f(row,"catalyseurs",alert.get("catalyseur",""))
            _append_f(row,"titres",alert.get("titre","")[:80])
            _append_f(row,"liens",alert.get("lien",""))

    for opp in opportunities:
        ticker = opp.get("ticker","").strip()
        if not ticker: continue
        if ticker not in data: data[ticker] = _new_row(ticker)
        row = data[ticker]
        if row.get("type") not in ("PORTFOLIO","BOTH"): row["type"] = "OPPORTUNITY"
        elif row.get("type") == "PORTFOLIO":            row["type"] = "BOTH"
        row["date_derniere_alerte"] = run_time
        row["conviction_opp"]  = opp.get("conviction","")
        row["direction_opp"]   = opp.get("direction_attendue","")
        _append_f(row,"catalyseurs",opp.get("catalyseur",""))
        if not row.get("resume_principal"): row["resume_principal"] = opp.get("resume","")

    for fund in fundamentals:
        if not fund: continue
        ticker = fund["ticker"]
        if ticker not in data: data[ticker] = _new_row(ticker)
        row = data[ticker]
        row.update({
            "nom": fund.get("nom",""), "secteur": fund.get("secteur",""),
            "pays": fund.get("pays",""), "devise": fund.get("devise",""),
            "prix": str(fund.get("prix") or ""),
            "market_cap_B": str(fund.get("market_cap_B") or ""),
            "date_derniere_alerte": run_time,
            # Growth
            "score_growth": fund["score_growth"],
            "G1_CA": f"{'✓' if fund['G1_ok'] else '✗'} {fund['G1_CA']}",
            "G1_ok": fund["G1_ok"],
            "G2_Marge": f"{'✓' if fund['G2_ok'] else '✗'} {fund['G2_Marge']}",
            "G2_ok": fund["G2_ok"],
            "G3_Underval": f"{'✓' if fund['G3_ok'] else '✗'} {fund['G3_Underval']}",
            "G3_ok": fund["G3_ok"],
            "G4_Momentum": f"{'✓' if fund['G4_ok'] else '✗'} {fund['G4_Momentum']}",
            "G4_ok": fund["G4_ok"],
            "G5_Volume": f"{'✓' if fund['G5_ok'] else '✗'} {fund['G5_Volume']}",
            "G5_ok": fund["G5_ok"],
            # Sichere
            "score_safe": fund["score_safe"],
            "S1_MCap": f"{'✓' if fund['S1_ok'] else '✗'} {fund['S1_MCap']}",
            "S1_ok": fund["S1_ok"],
            "S2_DE": f"{'✓' if fund['S2_ok'] else '✗'} {fund['S2_DE']}",
            "S2_ok": fund["S2_ok"],
            "S3_Beta": f"{'✓' if fund['S3_ok'] else '✗'} {fund['S3_Beta']}",
            "S3_ok": fund["S3_ok"],
            "S4_Div": f"{'✓' if fund['S4_ok'] else '✗'} {fund['S4_Div']}",
            "S4_ok": fund["S4_ok"],
            "S5_FCF_M": f"{'✓' if fund['S5_ok'] else '✗'} {fund['S5_FCF_M']}",
            "S5_ok": fund["S5_ok"],
            "S6_FCF_G": f"{'✓' if fund['S6_ok'] else '✗'} {fund['S6_FCF_G']}",
            "S6_ok": fund["S6_ok"],
            "S7_RevEPS": f"{'✓' if fund['S7_ok'] else '✗'} {fund['S7_RevEPS']}",
            "S7_ok": fund["S7_ok"],
            # Synthèse
            "score_total": fund["score_total"],
            "profil"     : fund["profil"],
        })
        if fund["score_growth"]>=FUNDAMENTAL_MIN_SCORE or fund["score_safe"]>=SAFE_MIN_SCORE:
            row["type"] = "BOTH" if row.get("type")=="PORTFOLIO" else "VALIDATED"

    if not data: return
    type_order = {"BOTH":0,"PORTFOLIO":1,"VALIDATED":2,"OPPORTUNITY":3}
    sorted_rows = sorted(data.values(),
        key=lambda r:(type_order.get(r.get("type",""),9), r.get("ticker","")))
    with open(TICKER_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TICKER_FIELDS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({k: row.get(k,"") for k in TICKER_FIELDS})
    print(f"[TICKER SUMMARY] {len(data)} tickers consolidés → {TICKER_CSV.name}")

# ── Save alert ────────────────────────────────────────────────

def save_alert(article, analysis):
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    tickers = "|".join(
        t["ticker"] if isinstance(t,dict) else t
        for t in analysis.get("tickers_affectes",[])
    )
    row = {
        "date": now, "source": article["source"], "titre": article["title"],
        "sentiment": analysis.get("sentiment",""),
        "intensite": analysis.get("intensite",""),
        "intensite_rank": INTENSITY_RANK.get(analysis.get("intensite",""),0),
        "horizon": analysis.get("horizon",""),
        "confiance": analysis.get("confiance",""),
        "catalyseur": analysis.get("catalyseur",""),
        "tickers_affectes": tickers,
        "resume_impact": analysis.get("resume_impact",""),
        "risque_contraire": analysis.get("risque_contraire",""),
        "raisonnement": analysis.get("raisonnement",""),
        "lien": article["link"],
    }
    file_exists = ALERTS_CSV.exists()
    with open(ALERTS_CSV,"a",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    icon = {"positif":"🟢","negatif":"🔴","neutre":"🟡"}.get(analysis.get("sentiment"),"⚪")
    print(f"\n{'─'*65}")
    print(f" {icon} [{analysis.get('intensite','?').upper()}] "
          f"confiance={analysis.get('confiance','?')}% horizon={analysis.get('horizon','?')}")
    print(f" {article['title'][:68]}")
    print(f" {analysis.get('resume_impact','')}")
    print(f" Tickers: {tickers or 'N/A'} | Catalyseur: {analysis.get('catalyseur','?')}")
    print(f"{'─'*65}")
    return {**row}

# ── Boucle principale ─────────────────────────────────────────

def run_once(run_fundamentals=True):
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M")
    portfolio = load_portfolio()
    if not portfolio:
        print("[ERROR] mes_symbols.txt vide."); return

    seen_ids    = load_seen_ids()
    articles    = fetch_news()
    new_articles= [a for a in articles if a["id"] not in seen_ids and a["title"].strip()]
    print(f"[{run_time}] {len(articles)} articles | {len(new_articles)} nouveaux")

    portfolio_alerts = []
    print("\n[PHASE 2a] Analyse impact portfolio...")
    for article in new_articles:
        seen_ids.add(article["id"])
        matched = quick_filter(article, portfolio)
        if not matched: continue
        print(f" → {article['title'][:60]}...", end="\r")
        analysis = analyze_portfolio_impact(article, portfolio, matched)
        if analysis and analysis.get("pertinent"):
            portfolio_alerts.append(save_alert(article, analysis))

    print("\n[PHASE 2b] Découverte d'opportunités...")
    opportunities      = discover_tickers_from_batch(new_articles, portfolio)
    discovered_tickers = [
        opp["ticker"] for opp in opportunities
        if opp.get("conviction") in ("modere","fort")
    ][:MAX_DISCOVERED_TICKERS]
    if opportunities:
        print(f" → {len(opportunities)} opportunités : {[o['ticker'] for o in opportunities]}")

    fundamentals = []
    if run_fundamentals and discovered_tickers:
        print(f"\n[PHASE 3] Analyse 12 critères sur {len(discovered_tickers)} tickers...")
        analyzed = set()
        for ticker in discovered_tickers:
            if ticker in analyzed: continue
            print(f" ↳ {ticker}...", end="\r")
            result = run_fundamental_combined(ticker)
            if result:
                fundamentals.append(result)
                p = result['profil']
                g = result['score_growth']
                s = result['score_safe']
                t = result['score_total']
                print(f" ✓ {ticker:8s} {p}  G={g}/5  S={s}/7  Total={t}/12")
            analyzed.add(ticker)
            time.sleep(0.5)

    if portfolio_alerts or fundamentals:
        print("\n[PHASE 4] Génération du rapport...")
        generate_report(portfolio_alerts, opportunities, fundamentals, run_time)

    print("\n[PHASE 5] Consolidation ticker_summary.csv...")
    consolidate_ticker_summary(portfolio_alerts, opportunities, fundamentals, run_time)

    save_seen_ids(seen_ids)
    nb_dual = len([f for f in fundamentals if f and f.get("profil")=="💎 Dual Champion"])
    print(f"\n✅ {len(portfolio_alerts)} alerte(s) | "
          f"{len(opportunities)} opportunité(s) | "
          f"{len(fundamentals)} analysés | "
          f"{nb_dual} 💎 Dual Champion")

def run_loop(interval, run_fundamentals):
    print(f"[INFO] Mode boucle | interval={interval}s")
    while True:
        try:
            run_once(run_fundamentals)
            sleep_time = 300 if is_market_hours() else 1800
            effective  = min(interval, sleep_time)
            print(f"[INFO] Prochain cycle dans {effective//60}min...\n")
            time.sleep(effective)
        except KeyboardInterrupt:
            print("\n[INFO] Arrêt."); break
        except Exception as e:
            print(f"[ERROR] {e}"); time.sleep(60)

# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="News Monitor + Combined Scanner (12 critères)"
    )
    parser.add_argument("--once",            action="store_true")
    parser.add_argument("--interval",        type=int,   default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--model",           type=str,   default=OLLAMA_MODEL)
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--eur-usd",         type=float, default=1.08)
    parser.add_argument("--min-growth",      type=int,   default=2)
    parser.add_argument("--min-safe",        type=int,   default=3)
    args = parser.parse_args()

    OLLAMA_MODEL        = args.model
    EUR_USD_RATE        = args.eur_usd
    FUNDAMENTAL_MIN_SCORE = args.min_growth
    SAFE_MIN_SCORE      = args.min_safe

    print(f"🚀 News Monitor Combined | modèle={OLLAMA_MODEL} | "
          f"min_growth={FUNDAMENTAL_MIN_SCORE} | min_safe={SAFE_MIN_SCORE}")

    if args.once: run_once(not args.no_fundamentals)
    else:         run_loop(args.interval, not args.no_fundamentals)