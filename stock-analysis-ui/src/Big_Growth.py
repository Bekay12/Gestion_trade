import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def _f(value):
    """Convertit en float ou retourne None si invalide/NaN."""
    if value is None:
        return None
    try:
        v = float(value)
        import math
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


Random_List= [
    "SNDK", "STX", "WDC",
    "BE", "CEG", "VST",
    "VRT", "SMR", "BWXT",
    "AVAV", "ADMA",
    "FCX", "IVN.TO",
    "XYL", "MWA",
    "ADBE",
    "KGC",
        "MU",    # Micron — mémoire HBM/NAND, IA
    "FIX",   # Comfort Systems — HVAC data centers
    "ABBV",  # AbbVie — pharma, sous-évalué
    "LYTS",  # LSI Industries — éclairage LED, croissance +16%/an
    "MC",    # Moelis & Co — banque d'investissement, EPS +311%
    "SHIP"
    
]

p_Ticker = [ "0005.HK", "0700.HK", "2388.HK", "601398.SS", "9618.HK", "9988.HK", "AAPL", "AAUC", "ABEO", "ABSO.ST", "ABUS", "ACS.MC", "ADC", "AER", "AFK.OL", "AG", "AFX.DE", "AHT", "AIXA.DE", "ALLO", "ALLT", "ALM", "ALV.DE", "AMAT", "ALTS", "ANAB", "AMS", "ARMK", "AROC", "ARX", "ASM", "ATHM", "ATRO", "ATON", "AWK", "BABA", "BAP", "BAYN.DE", "BEAT", "BFT.WA", "BGS", "BITF", "BMW.DE", "BN", "BOKF", "BP.L", "BRC", "BVN", "CALM", "CAN", "CAPR", "CCCC", "CBUS", "CEV", "CFG", "CIG", "CMPS", "CODX", "COK.DE", "COYA", "CRBP", "CRIS", "CRT", "CSCO", "CSIQ", "CSTE", "CWD", "CYN", "DAL", "DEZ.DE", "DHC", "DHI", "DLB", "DLO", "DMRC", "DRE2.F", "DTE.DE", "DSM", "DUE.DE", 
            "EC", "ED", "ECO", "ELE", "ENR.DE", "EOSE", "EOAN.DE", "EPD", "EQPT", "ER9.F", "ESP", "ETOR", "EVLV", "EWCZ", "EXE", "F", "FANG", "FATE", "FBYD", "FIG", "FNV", "FNV.TO", "FOSL", "FRMI", "FTLF", "GCTS", "GGAL", "GILT", "GLOB", "GO", "GOOS", "GOSS", "GROW", "GRWG", "GSK.L", "GWRS", "GUTS", "HEI.DE", "HMC", "HSBA.L", "HTFL", "HUMA", "HUYA", "HYMC", "IDR", "IHG", "IMB.L", "IMNM", "IMTX", "INFQ", "INGM", "INO", "IR", "IVR", "IVVD", "JACK"]
TICKERS =  p_Ticker
# ─────────────────────────────────────────────────────────────
# CONDITION 1 — Croissance structurelle du CA (>20% YoY)
# ─────────────────────────────────────────────────────────────
# Seagate avait +38% de croissance annuelle avant son envol.
# yfinance expose `revenueGrowth` = croissance YoY du dernier trimestre.
# Seuil conservateur fixé à +20% pour capturer les cas "en accélération".
def condition_1_revenue_growth(info):
    rev_growth = _f(info.get("revenueGrowth"))
    if rev_growth is None:
        return False, "N/A", "Donnée absente"
    passed = rev_growth > 0.20
    return passed, f"{round(rev_growth*100, 1)}%", f"CA +{round(rev_growth*100,1)}% YoY (seuil >20%)"


# ─────────────────────────────────────────────────────────────
# CONDITION 2 — Marges brutes en expansion (>30%)
# ─────────────────────────────────────────────────────────────
# Seagate est passé de 23% à 42% de marge brute en 12 mois.
# On cherche des boîtes qui ont déjà franchi 30% ET sont en territoire
# d'expansion (le niveau lui-même est déjà le signal, pas besoin du delta).
# Pour aller plus loin, on peut comparer grossMargins sur 2 trimestres
# consécutifs via stock.quarterly_financials.
def condition_2_gross_margin(info):
    gm = _f(info.get("grossMargins"))
    if gm is None:
        return False, "N/A", "Donnée absente"
    passed = gm > 0.30
    return passed, f"{round(gm*100, 1)}%", f"Marge brute {round(gm*100,1)}% (seuil >30%)"


# ─────────────────────────────────────────────────────────────
# CONDITION 3 — Sous-valorisation (action en retard sur ses fondamentaux)
# ─────────────────────────────────────────────────────────────
# Seagate était à $85 avec un P/E ~10x alors que ses fondamentaux
# justifiaient bien plus. On détecte la sous-valorisation via 3 proxies :
#   A) P/E trailing < 25 (pas encore re-valorisé par le marché)
#   B) PEG ratio < 1.5 (croissance pas encore intégrée dans le prix)
#   C) Prix < 75% du 52-week high (retard non rattrapé)
# La condition est validée si au moins 2 proxies sur 3 sont vrais.
def condition_3_undervaluation(info, hist):
    score = 0
    details = []

    pe = _f(info.get("trailingPE"))
    if pe is not None and 0 < pe < 25:
        score += 1
        details.append(f"P/E={round(pe,1)}")

    peg = _f(info.get("pegRatio"))
    if peg is not None and 0 < peg < 1.5:
        score += 1
        details.append(f"PEG={round(peg,2)}")

    high_52w = _f(info.get("fiftyTwoWeekHigh"))
    price = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
    if high_52w and price:
        ratio = price / high_52w
        if ratio < 0.75:
            score += 1
            details.append(f"Prix à {round(ratio*100,0)}% du 52wH")

    passed = score >= 2
    return passed, f"{score}/3 proxies", " | ".join(details) or "Aucune donnée"


# ─────────────────────────────────────────────────────────────
# CONDITION 4 — Momentum naissant (signal précoce, pas encore explosif)
# ─────────────────────────────────────────────────────────────
# L'idée clé : on veut attraper le titre AVANT l'explosion, pas pendant.
# Seagate montrait une hausse douce sur 3 mois (+8-15%) avant l'explosion.
# On vérifie :
#   - Retour sur 3 mois entre +8% et +60% (ni plat, ni déjà parti)
#   - Prix au-dessus de la SMA50 (tendance haussière confirmée)
#   - Retour sur 6 mois < 150% (pas encore trop tard pour entrer)
def condition_4_nascent_momentum(hist):
    if hist is None or len(hist) < 130:
        return False, "N/A", "Historique insuffisant"

    close = hist['Close']
    current = close.iloc[-1]
    price_3m = close.iloc[-63]   # ~63 jours de bourse = 3 mois
    price_6m = close.iloc[-126]  # ~126 jours = 6 mois

    ret_3m = (current - price_3m) / price_3m
    ret_6m = (current - price_6m) / price_6m
    sma50 = close.rolling(50).mean().iloc[-1]

    passed = (0.08 < ret_3m < 0.60) and (current > sma50) and (ret_6m < 1.50)
    detail = (f"3m={round(ret_3m*100,1)}% | "
              f"6m={round(ret_6m*100,1)}% | "
              f"SMA50={'OK' if current > sma50 else 'NON'}")
    return passed, f"{round(ret_3m*100,1)}%", detail


# ─────────────────────────────────────────────────────────────
# CONDITION 5 — Accumulation institutionnelle (volume croissant)
# ─────────────────────────────────────────────────────────────
# Les institutionnels accumulent en silence avant les grandes hausses.
# Détection : le volume moyen des 30 derniers jours dépasse de +20%
# le volume moyen sur 90 jours. C'est le signe d'un intérêt grandissant
# sans encore créer de bruit médiatique.
def condition_5_volume_buildup(hist):
    if hist is None or len(hist) < 90:
        return False, "N/A", "Historique insuffisant"

    vol_30 = hist['Volume'].iloc[-30:].mean()
    vol_90 = hist['Volume'].iloc[-90:].mean()

    if vol_90 == 0:
        return False, "N/A", "Volume nul"

    ratio = vol_30 / vol_90
    passed = ratio > 1.20
    return passed, f"{round(ratio, 2)}x", f"Vol30j/Vol90j = {round(ratio,2)}x (seuil >1.20x)"


# ─────────────────────────────────────────────────────────────
# MOTEUR PRINCIPAL — Analyse d'un ticker
# ─────────────────────────────────────────────────────────────
def analyze(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3y")
    info = stock.info

    c1 = condition_1_revenue_growth(info)
    c2 = condition_2_gross_margin(info)
    c3 = condition_3_undervaluation(info, hist)
    c4 = condition_4_nascent_momentum(hist)
    c5 = condition_5_volume_buildup(hist)

    score = sum(1 for c in [c1, c2, c3, c4, c5] if c[0])
    emoji = lambda b: "✅" if b else "❌"

    print(f"\n{'='*55}")
    print(f"  {ticker:8s} — {info.get('shortName', '?')}")
    print(f"  Score : {score}/5 {'🔥' * score}")
    print(f"{'='*55}")
    print(f"  {emoji(c1[0])} C1 CA Growth     : {c1[2]}")
    print(f"  {emoji(c2[0])} C2 Marge Brute   : {c2[2]}")
    print(f"  {emoji(c3[0])} C3 Valorisation  : {c3[2]}")
    print(f"  {emoji(c4[0])} C4 Momentum      : {c4[2]}")
    print(f"  {emoji(c5[0])} C5 Volume        : {c5[2]}")

    return {
        "ticker": ticker,
        "nom": info.get("shortName", ticker),
        "secteur": info.get("sector", "N/A"),
        "score": score,
        "C1_CA_Growth": c1[1], "C1_ok": c1[0],
        "C2_Marge": c2[1],     "C2_ok": c2[0],
        "C3_Valorisation": c3[1], "C3_ok": c3[0],
        "C4_Momentum": c4[1],  "C4_ok": c4[0],
        "C5_Volume": c5[1],    "C5_ok": c5[0],
    }


# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
results = []
for ticker in TICKERS:
    try:
        results.append(analyze(ticker))
    except Exception as e:
        print(f"Erreur sur {ticker}: {e}")

# Tri par score décroissant + export CSV
df = pd.DataFrame(results).sort_values("score", ascending=False)
df.to_csv("seagate_conditions_results.csv", index=False)

print("\n\n=== CLASSEMENT FINAL ===")
print(df[["ticker", "nom", "score",
          "C1_CA_Growth", "C2_Marge",
          "C3_Valorisation", "C4_Momentum", "C5_Volume"]].to_string(index=False))