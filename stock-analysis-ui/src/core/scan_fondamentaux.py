"""
scan_fondamentaux - code commun des scanners fondamentaux autonomes
(Combined_scan, Sichere_Unternehmen_scan, Big_Growth_scan, news_monitor_combined).

Avant ce module, chaque scanner portait sa propre copie de la couche reseau
yfinance et des criteres, aux constantes divergentes : un correctif applique a
l'un manquait aux autres (news_monitor_combined n'avait recu ni la conversion
de devise, ni le nettoyage de la seance en cours, ni l'echelle du dividende).
Un seul endroit desormais pour le debit, les reprises et chaque critere.

Contenu :
    - valeurs : safe_float, pct_croissance, pct_dividend_yield
    - devises : EUR_RATES, charger_taux_eur, mcap_en_mrd_eur
    - reseau  : regler_debit, throttle, fetch_info, fetch_hist, fetch_statements,
                is_rate_limit_error, record_skip / SKIP_REASONS
    - etats   : etats_annuels, taux_compose (S5-S7 sur exercices annuels)
    - criteres: g1..g5 (Big Growth), s1..s7 (Sichere Unternehmen), get_profile

Chaque critere renvoie (passe: bool, valeur: float | int | None) ; valeur None
signifie « non calculable », jamais un chiffre emprunte a un autre critere.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# VALEURS
# ═══════════════════════════════════════════════════════════════

# Bogue observe le 20.09.2026 sur DTG.DE: "dividendYield" valait 4.37 (deja
# un pourcentage) alors que "trailingAnnualDividendYield" valait 0.042937852
# (fraction) pour le MEME dividende reel (~4,3 %) - yfinance ne garantit pas
# l'echelle du premier champ, elle varie selon le titre et la version de
# l'API Yahoo. Multiplier "dividendYield" par 100 sans verification donnait
# 437 % au lieu de 4,37 %.
SEUIL_FRACTION_PLAUSIBLE = 1.5    # aucun rendement de dividende reel ne depasse 150 %
SEUIL_CROISSANCE_EXTREME = 300.0  # % au-dela desquels la valeur est signalee, pas corrigee
JOURS_DIVIDENDE_SUSPENDU = 400    # plus d'un an et un mois sans detachement
MIN_EXERCICES = 3                 # exercices annuels minimum pour une croissance (yfinance en donne 4)


def safe_float(v):
    """Float fini, ou None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def pct_croissance(fraction, ticker=None, champ=None):
    """
    --------------------------------------------------------------------------
    Purpose:
        Convertit une fraction de croissance en pourcentage. N'ALTERE PAS la
        valeur : une croissance extreme peut etre reelle (base proche de zero),
        mais elle est journalisee pour verification plutot qu'acceptee sans
        regard.

    Inputs:
        fraction (float): croissance en fraction (0.12 = 12 %)
        ticker, champ (str): pour le message de journalisation uniquement

    Outputs:
        pct (float): fraction * 100, journalisee si |pct| > 300 %
    --------------------------------------------------------------------------
    """
    pct = fraction * 100
    if abs(pct) > SEUIL_CROISSANCE_EXTREME:
        logger.warning(f"[SCAN] {ticker or '?'} {champ or 'croissance'}: {pct:.0f}% "
                       f"- valeur extreme, a verifier avant de la prendre au mot")
    return pct


def pct_dividend_yield(info):
    """
    --------------------------------------------------------------------------
    Purpose:
        Rendement du dividende en %, robuste au changement de convention de
        yfinance (fraction 0-1 vs pourcentage) sur "dividendYield". Quand
        "trailingAnnualDividendYield" (fraction) est disponible et positif, il
        sert d'etalon : l'echelle retenue est celle qui s'en rapproche le plus.
        A defaut, une fraction plausible ne depasse jamais
        SEUIL_FRACTION_PLAUSIBLE (150 %) ; au-dessus, le champ est deja en %.

    Inputs:
        info (dict): objet yfinance .info

    Outputs:
        pct (float | None): rendement en pourcentage, ou None si absent
    --------------------------------------------------------------------------
    """
    dy_brut = safe_float(info.get("dividendYield"))
    dy_annee = safe_float(info.get("trailingAnnualDividendYield"))
    # DEFAUT CORRIGE (02.10.2026, TTK.DE) : dividende suspendu. Le champ annuel
    # valait 0, le dernier detachement datait de 16 mois, mais "dividendYield"
    # (22.14) restait calcule sur un ancien dividende et passait le seuil de
    # plausibilite. Un zero annuel ne vaut suspension que si le dernier
    # detachement depasse JOURS_DIVIDENDE_SUSPENDU : un payeur annuel juste
    # avant son detachement peut aussi afficher 0 sur douze mois glissants.
    ex_div = safe_float(info.get("exDividendDate"))
    if dy_annee == 0 and ex_div is not None and \
            (time.time() - ex_div) / 86400 > JOURS_DIVIDENDE_SUSPENDU:
        return 0.0
    if dy_brut is not None:
        if dy_annee is not None and dy_annee > 0:
            comme_pourcentage = abs(dy_brut - dy_annee * 100)
            comme_fraction = abs(dy_brut * 100 - dy_annee * 100)
            return round(dy_brut if comme_pourcentage <= comme_fraction else dy_brut * 100, 2)
        return round(dy_brut if abs(dy_brut) > SEUIL_FRACTION_PLAUSIBLE else dy_brut * 100, 2)
    if dy_annee is not None:
        return round(dy_annee * 100, 2)
    return None


# ═══════════════════════════════════════════════════════════════
# DEVISES
# ═══════════════════════════════════════════════════════════════

# DEFAUT CORRIGE : seule l'USD etait convertie, toute autre devise etait
# traitee comme si elle etait deja en euros. Mesure du 14.09.2026 sur 43
# valeurs europeennes : 13 capitalisations fausses d'un facteur 7 a 11 (SEK,
# DKK, NOK, CHF) et une d'un facteur 85 (GBp, cotation en pence). Le critere
# « capitalisation > 10 Mrd € » en dependait directement.
EUR_RATES: dict[str, float] = {"EUR": 1.0}

# Paires Yahoo : unites de devise par euro.
_PAIRES_EUR = {"USD": "EURUSD=X", "SEK": "EURSEK=X", "DKK": "EURDKK=X",
               "NOK": "EURNOK=X", "CHF": "EURCHF=X", "GBP": "EURGBP=X",
               "PLN": "EURPLN=X", "CZK": "EURCZK=X", "HUF": "EURHUF=X",
               "JPY": "EURJPY=X", "CAD": "EURCAD=X", "AUD": "EURAUD=X"}


def charger_taux_eur(verbose: bool = True, eur_usd_repli: float = 1.08) -> dict:
    """Charge les taux de change en UN seul appel groupe.

    Un appel groupe et non un par devise : la contrainte de budget yfinance du
    projet vaut aussi ici. Les taux manquants restent absents du dictionnaire,
    ce qui rend la capitalisation non evaluable plutot que fausse.
    eur_usd_repli (option --eur-usd des scripts) ne sert que si EURUSD manque.
    """
    global EUR_RATES
    taux = {"EUR": 1.0}
    try:
        data = yf.download(list(_PAIRES_EUR.values()), period="5d",
                           progress=False, auto_adjust=False, group_by="ticker")
        for devise, paire in _PAIRES_EUR.items():
            try:
                serie = data[paire]["Close"].dropna()
                if len(serie):
                    taux[devise] = float(serie.iloc[-1])
            except Exception:
                continue
    except Exception as exc:
        if verbose:
            print(f"   ⚠️ Taux de change indisponibles ({exc}) : "
                  f"les capitalisations hors zone euro ne seront pas evaluees")
    if "USD" not in taux:
        taux["USD"] = eur_usd_repli
    if "GBP" in taux:
        taux["GBP_PENCE"] = taux["GBP"] * 100.0   # GBp est un centieme de livre
    EUR_RATES = taux
    if verbose:
        print(f"   💱 {len(taux)-1} taux de change charges "
              f"({', '.join(sorted(k for k in taux if k != 'EUR'))})")
    return taux


def mcap_en_mrd_eur(mc, devise):
    """Capitalisation en milliards d'euros, ou None si le taux manque.

    None, jamais un repli sur 1.0 : une devise non convertie qui passe pour de
    l'euro est precisement le defaut corrige ici.
    """
    if mc is None:
        return None
    cle = (devise or "USD").strip()
    # GBp / GBX : cotation en pence, cas particulier a traiter avant le .upper()
    if cle in ("GBp", "GBX", "GBPp"):
        taux = EUR_RATES.get("GBP_PENCE")
    else:
        taux = EUR_RATES.get(cle.upper())
    if not taux:
        return None
    return (float(mc) / taux) / 1e9


# ═══════════════════════════════════════════════════════════════
# RESEAU yfinance : un seul limiteur de debit, une seule politique de reprise
# ═══════════════════════════════════════════════════════════════

_throttle_lock = threading.Lock()
_last_call_time = 0.0
_debit = {"delai": 0.25}   # secondes minimum entre deux appels (~4 req/s)

_skip_lock = threading.Lock()
SKIP_REASONS: dict[str, int] = {}


def regler_debit(delai: float) -> float:
    """Fixe le delai minimum entre deux appels yfinance (option --throttle)."""
    _debit["delai"] = max(0.05, float(delai))
    return _debit["delai"]


def debit() -> float:
    return _debit["delai"]


def throttle():
    """Limite globale du debit d'appels yfinance, partagee par tous les threads."""
    global _last_call_time
    with _throttle_lock:
        now = time.monotonic()
        wait = _debit["delai"] - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


def record_skip(raison: str) -> None:
    with _skip_lock:
        SKIP_REASONS[raison] = SKIP_REASONS.get(raison, 0) + 1


def is_rate_limit_error(exc) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "too many requests", "rate limited"))


def is_valid_history(hist) -> bool:
    if hist is None or getattr(hist, "empty", True):
        return False
    if "Close" not in hist.columns or "Volume" not in hist.columns:
        return False
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(hist["Volume"], errors="coerce").dropna()
    if len(close) < 130 or len(volume) < 90:
        return False
    return not (close <= 0).all() and not (volume < 0).any()


def fast_info(stock) -> dict:
    """Champs rapides de yfinance, sans lever d'exception."""
    try:
        fi = stock.fast_info
    except Exception:
        return {}
    if fi is None:
        return {}
    out = {}
    for src, dst in [("lastPrice", "currentPrice"), ("last_price", "currentPrice"),
                     ("marketCap", "marketCap"), ("market_cap", "marketCap"),
                     ("currency", "currency"), ("quoteType", "quoteType")]:
        try:
            v = fi.get(src)
            if v is not None:
                out[dst] = v
        except Exception:
            continue
    return out


def is_valid_info(info) -> bool:
    if not isinstance(info, dict) or not info:
        return False
    has_id = bool(info.get("shortName") or info.get("longName") or info.get("symbol"))
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    return has_id or (price is not None and price > 0)


def fetch_info(stock, ticker, max_retries=4):
    """Metadonnees avec reprise/backoff et complement fast_info ; None si inutilisables."""
    last = {}
    for attempt in range(max_retries):
        try:
            throttle()
            info = stock.info or {}
            if isinstance(info, dict):
                last = info
        except Exception:
            pass
        merged = dict(last)
        merged.update({k: v for k, v in fast_info(stock).items() if v is not None})
        merged.setdefault("symbol", ticker)
        if is_valid_info(merged):
            return merged
        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))
    return None


def fetch_hist(stock, max_retries=4, period="3y"):
    """Historique de cours valide, ou None."""
    for attempt in range(max_retries):
        try:
            throttle()
            hist = stock.history(period=period, auto_adjust=False)
            if is_valid_history(hist):
                return hist
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))
    return None


def fetch_statements(stock, max_retries=3):
    """Etats annuels (cashflow, income_stmt) ; (None, None) si indisponibles."""
    for attempt in range(max_retries):
        try:
            throttle()
            cf = stock.cashflow
            throttle()
            inc = stock.income_stmt
            if (cf is not None and not cf.empty) or (inc is not None and not inc.empty):
                return cf, inc
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))
    return None, None


# ═══════════════════════════════════════════════════════════════
# ETATS ANNUELS
# ═══════════════════════════════════════════════════════════════

def _serie(etat, poste):
    """Valeurs d'un poste, du plus ancien au plus recent, exercices vides retires."""
    if etat is None or getattr(etat, "empty", True) or poste not in etat.index:
        return []
    s = pd.to_numeric(etat.loc[poste], errors="coerce").dropna()
    s = s[[math.isfinite(v) for v in s]]
    return [(pd.Timestamp(d), float(v)) for d, v in sorted(s.items())]


def etats_annuels(cashflow, income):
    """
    --------------------------------------------------------------------------
    Purpose:
        Series annuelles utiles a S5-S7, lues dans les etats financiers.
        Remplace les champs .info trimestriels (voir s6_fcf_growth).

    Inputs:
        cashflow (DataFrame | None): stock.cashflow (postes x exercices)
        income (DataFrame | None): stock.income_stmt

    Outputs:
        etats (dict): "fcf", "rev", "eps" -> liste [(date, valeur)] croissante
    --------------------------------------------------------------------------
    """
    return {"fcf": _serie(cashflow, "Free Cash Flow"),
            "rev": _serie(income, "Total Revenue"),
            "eps": _serie(income, "Diluted EPS")}


def taux_compose(serie):
    """
    Taux annuel compose en % entre le premier et le dernier exercice, ou None
    si moins de MIN_EXERCICES valeurs ou si une borne n'est pas positive (un
    taux de croissance depuis ou vers une valeur negative n'a pas de sens).
    """
    if len(serie) < MIN_EXERCICES:
        return None
    (d0, v0), (d1, v1) = serie[0], serie[-1]
    annees = (d1 - d0).days / 365.25
    if v0 <= 0 or v1 <= 0 or annees <= 0:
        return None
    return ((v1 / v0) ** (1 / annees) - 1) * 100


# ═══════════════════════════════════════════════════════════════
# 🚀 BIG GROWTH — 5 criteres
# ═══════════════════════════════════════════════════════════════

def g1_revenue_growth(info):
    """G1 — Croissance du CA > 20 % sur un an."""
    rg = safe_float(info.get("revenueGrowth"))
    if rg is None:
        return False, None
    return rg > 0.20, round(rg * 100, 1)


def g2_gross_margin(info):
    """G2 — Marge brute > 30 %."""
    gm = safe_float(info.get("grossMargins"))
    if gm is None:
        return False, None
    return gm > 0.30, round(gm * 100, 1)


def g3_undervaluation(info):
    """G3 — Au moins 2 proxies sur 3 : P/E < 25, PEG < 1,5, prix < 75 % du plus haut 52 sem."""
    hits = 0
    pe = safe_float(info.get("trailingPE"))
    peg = safe_float(info.get("pegRatio"))
    h52 = safe_float(info.get("fiftyTwoWeekHigh"))
    px = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    if pe and 0 < pe < 25:
        hits += 1
    if peg and 0 < peg < 1.5:
        hits += 1
    if h52 and px and h52 > 0 and (px / h52) < 0.75:
        hits += 1
    return hits >= 2, hits


def momentum_detail(hist):
    """(r3, r6, cur, sma50) sur cloture nettoyee, ou None si historique trop court."""
    if hist is None or len(hist) < 130:
        return None
    # DEFAUT CORRIGE : sur les bourses europeennes la derniere ligne est la
    # seance en cours et porte NaN. Lue brute, elle rendait cur, r3, r6 et
    # sma50 tous NaN, et le critere tombait en silence — calculable sur 2
    # valeurs sur 43 le 14.09.2026, aucune remplie.
    close = hist["Close"].dropna()
    if len(close) < 130:
        return None
    cur = float(close.iloc[-1])
    p3m = float(close.iloc[-63])
    p6m = float(close.iloc[-126])
    if p3m == 0 or p6m == 0:
        return None
    sma50 = float(close.rolling(50).mean().iloc[-1])
    return (cur - p3m) / p3m, (cur - p6m) / p6m, cur, sma50


def g4_momentum(hist):
    """G4 — Momentum naissant : 3 mois entre +8 et +60 %, au-dessus de la SMA50, 6 mois < +150 %."""
    m = momentum_detail(hist)
    if m is None:
        return False, None
    r3, r6, cur, sma50 = m
    return (0.08 < r3 < 0.60) and (cur > sma50) and (r6 < 1.50), round(r3 * 100, 1)


def g5_volume(hist):
    """G5 — Accumulation : volume moyen 30 j / 90 j > 1,20."""
    if hist is None or len(hist) < 90:
        return False, None
    v30 = float(hist["Volume"].iloc[-30:].mean())
    v90 = float(hist["Volume"].iloc[-90:].mean())
    if not v90:
        return False, None
    ratio = v30 / v90
    return ratio > 1.20, round(ratio, 2)


# ═══════════════════════════════════════════════════════════════
# 🛡️ SICHERE UNTERNEHMEN — 7 criteres
# ═══════════════════════════════════════════════════════════════

def s1_market_cap(info):
    """S1 — Capitalisation > 10 Mrd €, convertie depuis la devise de cotation."""
    mc_eur_b = mcap_en_mrd_eur(safe_float(info.get("marketCap")), info.get("currency"))
    if mc_eur_b is None:
        return False, None
    return mc_eur_b > 10.0, round(mc_eur_b, 1)


def s2_debt_equity(info):
    """S2 — Dette / fonds propres < 100 %."""
    de = safe_float(info.get("debtToEquity"))
    if de is None:
        return False, None
    return de < 100.0, round(de, 1)


def s3_beta(info):
    """S3 — Beta strictement entre 0 et 0,8."""
    beta = safe_float(info.get("beta"))
    if beta is None:
        return False, None
    return 0 < beta < 0.8, round(beta, 2)


def s4_dividend(info):
    """S4 — Dividende verse (rendement > 0)."""
    dy_pct = pct_dividend_yield(info)
    if dy_pct is None:
        return False, None
    return dy_pct > 0, dy_pct


def s5_fcf_margin(info, etats=None):
    """S5 — Marge de FCF > 5 % sur le dernier exercice."""
    # DEFAUT CORRIGE (02.10.2026) : info["freeCashflow"] est un FCF « levered »
    # TTM qui depassait parfois le flux operationnel (TTK.DE 46,3 M contre
    # 28,3 M ; FOSL +24,7 M contre -37,9 M). Le dernier exercice annuel
    # (FCF = OCF - capex) fait foi ; .info ne sert qu'a defaut d'etats.
    if etats and etats["fcf"] and etats["rev"] and etats["fcf"][-1][0] == etats["rev"][-1][0] \
            and etats["rev"][-1][1] > 0:
        margin = etats["fcf"][-1][1] / etats["rev"][-1][1] * 100
        return margin > 5.0, round(margin, 1)
    fcf = safe_float(info.get("freeCashflow"))
    rev = safe_float(info.get("totalRevenue"))
    if fcf is None or rev is None or rev == 0:
        ocf = safe_float(info.get("operatingCashflow"))
        capex = safe_float(info.get("capitalExpenditures"))
        if ocf is not None and capex is not None and rev and rev > 0:
            fcf = ocf - abs(capex)
        else:
            return False, None
    margin = (fcf / rev) * 100
    return margin > 5.0, round(margin, 1)


def s6_fcf_growth(info, ticker=None, etats=None):
    """S6 — Taux annuel compose du FCF > 0 % (exercices disponibles, 4 chez yfinance)."""
    # DEFAUT CORRIGE (02.10.2026) : le critere « Ø croissance FCF » ne lisait
    # aucun FCF. Il prenait earningsQuarterlyGrowth (trimestre sur trimestre :
    # ASRNL.AS 542 %, STM.DE 423 %), sinon revenueGrowth, qui recopiait G1
    # (TTK.DE, PWO.DE, FOSL). Sans etats ou avec un FCF negatif a une borne,
    # le critere est non calculable : un zero honnete vaut mieux qu'un chiffre
    # emprunte a un autre critere.
    taux = taux_compose(etats["fcf"]) if etats else None
    if taux is None:
        return False, None
    pct = pct_croissance(taux / 100, ticker, "S6 FCF compose")
    return pct > 0, round(pct, 1)


def s7_rev_eps_growth(info, ticker=None, etats=None):
    """S7 — Taux composes annuels du CA et du BPA dilue, les deux > 3 % ; valeur = moyenne."""
    # DEFAUT CORRIGE (02.10.2026) : lisait la croissance trimestrielle des
    # benefices (ASRNL.AS earningsGrowth 6,48 -> moyenne 332,8 %) ou, faute de
    # champ, recopiait revenueGrowth. Un BPA negatif a une borne rend son taux
    # non calculable, donc l'echec.
    if not etats:
        return False, None
    t_ca, t_bpa = taux_compose(etats["rev"]), taux_compose(etats["eps"])
    vals = [v for v in (t_ca, t_bpa) if v is not None]
    if not vals:
        return False, None
    moyenne = sum(vals) / len(vals)
    pct_croissance(moyenne / 100, ticker, "S7 CA+BPA compose")
    passe = t_ca is not None and t_bpa is not None and t_ca > 3.0 and t_bpa > 3.0
    return passe, round(moyenne, 1)


def s7_detail(etats):
    """(taux CA, taux BPA) pour l'affichage, chacun None si non calculable."""
    if not etats:
        return None, None
    return taux_compose(etats["rev"]), taux_compose(etats["eps"])


# ═══════════════════════════════════════════════════════════════
# 💎 PROFIL
# ═══════════════════════════════════════════════════════════════

DUAL = "💎 Dual Champion"
DUAL_ETOILE = "💎 Dual Champion*"


def est_etoile(g3_ok, g4_ok, s4_ok) -> bool:
    """
    Dual Champion* : Dual Champion qui remplit AUSSI G3 (sous-valorisation),
    G4 (momentum 3 mois) et S4 (dividende). Backtest point-in-time du
    02.10.2026 (Combined_backtest.py, T >= 2025-04-01) : ces trois criteres
    etaient les seuls associes aux meilleurs resultats ; G4+S4+G3 a battu
    l'indice a toutes les dates a 6 mois et dans 71 % des cas a 12 mois.
    Regle tiree des memes donnees qui la jugent : a revalider sur les dates
    suivantes avant de s'y fier.
    """
    return bool(g3_ok and g4_ok and s4_ok)


def est_dual(profil) -> bool:
    """Vrai pour Dual Champion ET Dual Champion* (filtres, comptages)."""
    return isinstance(profil, str) and profil.startswith(DUAL)


def get_profile(gs, ss, etoile=False):
    """Profil bi-score ; etoile=est_etoile(...) distingue le Dual Champion*."""
    if gs >= 3 and ss >= 5:
        return DUAL_ETOILE if etoile else DUAL
    if gs >= 4 and ss < 3:
        return "🚀 Pure Growth"
    if ss >= 5 and gs < 3:
        return "🛡️  Pure Safe"
    if gs >= 3 and ss >= 3:
        return "⚖️  Balanced"
    return "⚪ Below"
