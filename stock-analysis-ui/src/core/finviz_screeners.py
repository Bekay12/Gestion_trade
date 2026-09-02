"""
Screeners Finviz *market-wide*.

Contrairement aux screeners store-based (limités au catalogue local), ceux-ci
interrogent l'UNIVERS entier du marché US en 1 requête Finviz et renvoient les
tickers correspondants — pour DÉCOUVRIR de nouveaux titres, pas seulement filtrer
les siens. Coût : 1 requête Finviz par screener (0 requête yfinance).

Chaque preset mappe une stratégie vers un dict de filtres Finviz (clés/options
exactes issues de finvizfinance.constants.filter_dict).

`run_screen()` est le point d'entrée unique vers finvizfinance : il porte le
contournement du bot-blocking ET la lecture correcte de la colonne Ticker (voir
_OverviewTickerPropre). Tout nouvel écran Finviz doit passer par lui.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Presets : nom interne → {title, filters, order}
PRESETS = {
    "big_growth": {
        "title": "Big Growth (marché) — croissance forte + accumulation volume",
        # Fidèle aux critères G1/G4/G5 : croissance CA & EPS courante (YoY) + volume
        # relatif élevé (accumulation). Filtres AND ⇒ on garde les 4 plus discriminants.
        "filters": {
            "Sales growthqtr over qtr": "Over 20%",
            "EPS growththis year": "Over 20%",
            "P/E": "Profitable (>0)",
            "Relative Volume": "Over 1.5",
        },
    },
    "garp": {
        "title": "GARP — croissance à prix raisonnable (PEG<1, EPS>15%)",
        "filters": {
            "PEG": "Under 1",
            "EPS growthpast 5 years": "Over 15%",
            "P/E": "Under 25",
        },
    },
    "secure_growth": {
        "title": "Secure Growth — large cap qualité, croissance + faible risque (9 critères)",
        # Set exact repris de Code_Germain/Gap_Screen.py (preset "secure_Growth") :
        # grande cap, peu endettée, croissance CA/EPS 5 ans, forte marge brute,
        # valorisation raisonnable (PEG<2), beta bas, en repli mais au-dessus de la SMA50.
        "filters": {
            "Market Cap.": "+Large (over $10bln)",
            "Debt/Equity": "Under 1",
            "EPS growthpast 5 years": "Positive (>0%)",
            "Gross Margin": "Over 30%",
            "PEG": "Under 2",
            "Sales growthpast 5 years": "Over 5%",
            "Beta": "Under 1",
            "52-Week High/Low": "20% or more below High",
            "50-Day Simple Moving Average": "Price above SMA50",
        },
    },
    "minervini": {
        "title": "Minervini Trend Template — uptrend + proche du +haut 52s",
        # SEPA fidèle : prix au-dessus des SMA50/200 ET à ≤ 5% du plus-haut 52s
        # (leaders en tendance forte), avec liquidité minimale.
        "filters": {
            "50-Day Simple Moving Average": "Price above SMA50",
            "200-Day Simple Moving Average": "Price above SMA200",
            "52-Week High/Low": "0-5% below High",
            "Average Volume": "Over 500K",
        },
    },
    "magic_formula": {
        "title": "Magic Formula-like — qualité (ROA/ROE élevés) + valorisation basse",
        # Approxime Greenblatt (rendement du capital élevé + earnings yield élevé)
        # via ROA>15%, ROE>30% et P/E bas.
        "filters": {
            "Return on Assets": "Over +15%",
            "Return on Equity": "Very Positive (>30%)",
            "P/E": "Under 15",
        },
    },
    "rs_leaders": {
        "title": "RS Leaders — perf 6 mois ≥ +30%, au-dessus SMA200",
        "filters": {
            "Performance": "Half +30%",
            "Average Volume": "Over 500K",
            "200-Day Simple Moving Average": "Price above SMA200",
        },
    },
    "new_high": {
        "title": "Nouveaux plus-hauts 52 semaines (breakout)",
        "filters": {
            "52-Week High/Low": "New High",
            "Average Volume": "Over 500K",
        },
    },
    "oversold_quality": {
        "title": "Oversold Quality — RSI<30 sur large cap à dividende",
        "filters": {
            "RSI (14)": "Oversold (30)",
            "Market Cap.": "+Large (over $10bln)",
            "Dividend Yield": "Positive (>0%)",
        },
    },
    "low_vol_def": {
        "title": "Low-Vol Défensif — beta < 0.5, large cap, dividende > 2%",
        "filters": {
            "Market Cap.": "+Large (over $10bln)",
            "Beta": "Under 0.5",
            "Dividend Yield": "Over 2%",
        },
    },
    "gap_up": {
        "title": "Gap Up ≥ 5% (marché) — cassures haussières du jour",
        # Remplace le gap up sur symboles populaires : tout le marché, pas le catalogue.
        "filters": {
            "Gap": "Up 5%",
            "Average Volume": "Over 500K",
        },
        "order": "Change",
    },
    "gap_down": {
        "title": "Gap Down ≥ 5% (marché) — décrochages du jour",
        "filters": {
            "Gap": "Down 5%",
            "Average Volume": "Over 500K",
        },
        "order": "Change",
        "ascend": True,   # gaps les plus négatifs en premier
    },
}


def _num(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# Avatar-lettre placé par Finviz dans la cellule Ticker, devant le lien du
# symbole : <a class="company-ticker"><img …/><span>I</span></a>. Son <span> de
# repli (la 1re lettre, affichée le temps que le logo charge) fait partie du
# texte de la cellule.
_SELECTEUR_AVATAR = "a.company-ticker"

# Attribut du <td> qui porte le symbole non décoré (données de l'infobulle).
_ATTRIBUT_TICKER = "data-boxover-ticker"


def _ticker_fiable(ligne) -> str | None:
    """Symbole lu sur l'attribut `data-boxover-ticker` du <td>, ou None."""
    for cellule in ligne.find_all("td"):
        valeur = (cellule.get(_ATTRIBUT_TICKER) or "").strip()
        if valeur:
            return valeur.upper()
    return None


def _construire_overview():
    """Sous-classe d'Overview qui lit la colonne Ticker sans l'avatar-lettre.

    finvizfinance 1.3.0 remplit chaque cellule avec `td.text`, qui concatène TOUT
    le texte de la cellule. Depuis la refonte du screener Finviz, la cellule
    Ticker contient l'avatar-lettre en plus du symbole, d'où « IIESC » pour
    « IESC » : chaque symbole partait avec sa première lettre doublée, échouait
    côté yfinance (« No history available ») et polluait le store de profils.

    Deux mécanismes indépendants, du plus fiable au moins fiable :
      1. l'attribut `data-boxover-ticker` du <td>, qui fait autorité ;
      2. à défaut, la suppression de l'avatar du DOM avant que la librairie ne
         lise le texte de la cellule.
    `lignes_fiables` compte les lignes couvertes par l'un des deux, ce qui permet
    à run_screen() de détecter une 3e refonte de la structure Finviz.

    Aucune réparation par heuristique : « AAPL » ou « MMM » sont des symboles
    légitimes à première lettre doublée, indiscernables d'un symbole corrompu.
    """
    from finvizfinance.screener.overview import Overview

    class _OverviewTickerPropre(Overview):
        def __init__(self):
            super().__init__()
            self.lignes_fiables = 0

        def _get_table(self, rows, df, num_col_index, table_header, limit=-1):
            # Même découpe que Base._get_table, pour que l'ordre des tickers
            # relevés ici corresponde aux lignes ajoutées au DataFrame.
            lignes = rows[1:]
            if limit != -1:
                lignes = lignes[0:limit]

            tickers = []
            for ligne in lignes:
                fiable = _ticker_fiable(ligne)
                avatars = ligne.select(_SELECTEUR_AVATAR)
                for avatar in avatars:
                    avatar.decompose()
                if fiable or avatars:
                    self.lignes_fiables += 1
                tickers.append(fiable)

            resultat = super()._get_table(rows, df, num_col_index, table_header, limit)

            if tickers and "Ticker" in getattr(resultat, "columns", []):
                for position, ticker in zip(resultat.index[-len(tickers):], tickers):
                    if ticker:
                        resultat.at[position, "Ticker"] = ticker
            return resultat

    return _OverviewTickerPropre()


def _verifier_tickers(df: pd.DataFrame, lignes_fiables: int) -> None:
    """Échoue bruyamment si la structure Finviz n'est plus reconnue.

    Une liste de tickers corrompue est pire qu'une erreur : elle consomme le
    budget yfinance sur des symboles inexistants et écrit des profils fantômes
    dans le store. On ne lève que si AUCUNE ligne n'a pu être lue par un
    mécanisme fiable ET que les symboles portent la signature de la corruption
    (première lettre doublée en masse) — sinon une page légitimement sans logo
    déclencherait une fausse alerte.
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return
    if lignes_fiables:
        return

    tickers = [str(t).strip() for t in df["Ticker"] if str(t).strip()]
    suspects = [t for t in tickers if len(t) > 1 and t[0] == t[1]]
    if tickers and len(suspects) >= max(3, 0.6 * len(tickers)):
        raise RuntimeError(
            f"Finviz : structure de la cellule Ticker non reconnue "
            f"({len(suspects)}/{len(tickers)} symboles à première lettre doublée, "
            f"ex. {suspects[:3]}). Le sélecteur '{_SELECTEUR_AVATAR}' et l'attribut "
            f"'{_ATTRIBUT_TICKER}' de core/finviz_screeners.py sont à remettre à jour."
        )
    logger.warning("[FINVIZ] aucun ticker lu depuis une source fiable ; "
                   "structure de page inhabituelle mais symboles plausibles")


def run_screen(filters: dict, order: str = "Change", limit: int = 100,
               ascend: bool = False) -> pd.DataFrame:
    """Exécute un screen Finviz Overview avec contournement du bot-blocking
    (session curl_cffi impersonate) et colonne Ticker assainie. Retourne le
    DataFrame de finvizfinance. Lève une exception explicite si la dépendance
    manque ou si la structure de la page n'est plus reconnue."""
    import finvizfinance.util as _fv_util
    from curl_cffi.requests import Session as CurlSession

    _orig = _fv_util.session
    _fv_util.session = CurlSession(impersonate="chrome")
    try:
        fov = _construire_overview()
        fov.set_filter(filters_dict=filters)
        df = fov.screener_view(order=order, limit=limit, ascend=ascend)
    finally:
        _fv_util.session = _orig

    _verifier_tickers(df, fov.lignes_fiables)
    return df


def run_preset(key: str, limit: int = 100) -> dict:
    """Exécute un preset market-wide. Retourne {title, headers, rows} pour le dialog."""
    preset = PRESETS.get(key)
    if preset is None:
        raise ValueError(f"Preset Finviz inconnu : {key}")

    df = run_screen(
        preset["filters"], order=preset.get("order", "Change"),
        limit=limit, ascend=preset.get("ascend", False),
    )
    # Le nom vient de la colonne « Company » de la réponse Finviz : aucune
    # requête supplémentaire, et il est disponible même hors catalogue local.
    headers = ["Symbole", "Nom", "Secteur", "Pays", "Cap", "P/E", "Prix", "Var %"]
    rows = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            sym = str(r.get("Ticker") or "").strip().upper()
            if not sym:
                continue
            change = _num(r.get("Change"))
            rows.append([
                sym,
                str(r.get("Company") or "N/A"),
                str(r.get("Sector") or "N/A"),
                str(r.get("Country") or "N/A"),
                str(r.get("Market Cap") or "N/A"),
                _num(r.get("P/E")),
                _num(r.get("Price")),
                round(change * 100, 2) if change is not None else None,
            ])
    return {
        "title": f"{preset['title']} — {len(rows)} résultat(s)",
        "headers": headers,
        "rows": rows,
    }
