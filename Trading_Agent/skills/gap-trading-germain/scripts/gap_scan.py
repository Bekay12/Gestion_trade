#!/usr/bin/env python3
"""
gap_scan.py - scan des gappers (methode Germain) et enrichissement du sous-jacent.

Trois modes, deux fenetres (references/timing-et-sessions.md):

  --mode premarket   7h00-9h30 ET / 13h00-15h30 France. Produit une WATCHLIST.
                     La tenue du gap n'est pas encore observable.
  --mode close       16h00-17h00 ET / 22h00-23h00 France. Produit un VERDICT:
                     le gap a-t-il tenu ? C'est la seule fenetre qui autorise
                     une conclusion intraweek.
  --mode ticker      Qualifie des titres nommes, hors scan.

Deux regles d'architecture du depot, respectees ici:
  * Le screener Finviz passe par core.finviz_screeners.run_screen et non par
    finvizfinance directement: c'est run_screen qui porte la session curl_cffi
    (contournement du bot-blocking) et l'assainissement de la colonne Ticker.
  * Le budget de requetes yfinance est une contrainte dure: l'enrichissement se
    fait en UN SEUL appel groupe yf.download(..., group_by="ticker"), jamais en
    boucle par symbole.

Aucune donnee pre-marche n'est disponible sur Finviz gratuit; avant 9h30 ET la
couverture est partielle et le rapport doit le dire.

Aufruf:
  python3 scripts/gap_scan.py --mode premarket --min-gap 5 --limit 50
  python3 scripts/gap_scan.py --mode close --min-gap 5
  python3 scripts/gap_scan.py --mode ticker --tickers AAPL,TSLA
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_qualifier import Gap, qualifier, rendre           # noqa: E402

APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"
logger = logging.getLogger(__name__)

# Preset "morning_gappers" de Code_Germain/Gap_Screen.py, a l'identique.
FILTRES_GERMAIN = {
    "Market Cap.":     "-Small (under $2bln)",
    "Price":           "Under $20",
    "Average Volume":  "Over 500K",
    "Relative Volume": "Over 2",
    "Float Short":     "Over 10%",
}
GAP_PAR_SEUIL = {5: "Up 5%", 10: "Up 10%", 15: "Up 15%", 20: "Up 20%"}


def _fenetre_actuelle() -> str:
    """Fenetre de session au moment de l'appel, en heure de New York."""
    ny = timezone(timedelta(hours=-4))          # EDT; l'ecart exact importe peu ici
    h = datetime.now(ny)
    minutes = h.hour * 60 + h.minute
    if 7 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 10 * 60:
        return "ouverture"
    if 16 * 60 <= minutes < 17 * 60:
        return "close"
    if 10 * 60 <= minutes < 16 * 60:
        return "seance"
    return "hors-session"


def scanner_finviz(min_gap: int, limit: int):
    """Liste des gappers via le screener de l'application. Rend une liste de dict."""
    sys.path.insert(0, APP_SRC)
    from core.finviz_screeners import run_screen           # noqa: E402

    filtres = dict(FILTRES_GERMAIN)
    filtres["Gap"] = GAP_PAR_SEUIL.get(min_gap, "Up 5%")
    df = run_screen(filtres, order="Change", limit=limit, ascend=False)
    if df is None or len(df) == 0:
        return []

    def _num(v):
        s = str(v).replace(",", "").replace("%", "").strip()
        for suffixe, mult in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if s.upper().endswith(suffixe):
                try:
                    return float(s[:-1]) * mult
                except ValueError:
                    return None
        try:
            return float(s)
        except ValueError:
            return None

    lignes = []
    for _, r in df.iterrows():
        sym = str(r.get("Ticker") or "").strip().upper()
        if not sym:
            continue
        chg = _num(r.get("Change"))
        lignes.append({
            "ticker": sym,
            "nom": str(r.get("Company") or "N/A"),
            "secteur": str(r.get("Sector") or "N/A"),
            "pays": str(r.get("Country") or "N/A"),
            # Finviz rend Change en fraction (0.0477) -> pourcentage.
            "change_pct": round(chg * 100, 2) if chg is not None else None,
            "prix": _num(r.get("Price")),
            "volume": _num(r.get("Volume")),
            "market_cap": _num(r.get("Market Cap")),
        })
    return lignes


def enrichir(tickers: list) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        ATR(14), VWAP du jour, RVOL, float et short interest pour chaque titre.
        UN SEUL appel groupe yfinance pour l'historique (budget de requetes du
        depot); les champs de float/short interest viennent de .info, appele une
        fois par titre et uniquement pour les candidats deja filtres.

    Inputs:
        tickers (list): symboles

    Outputs:
        mesures (dict): {ticker: {atr_pct, vwap, rvol, float_actions, ...}}
    --------------------------------------------------------------------------
    """
    import yfinance as yf
    import pandas as pd

    mesures = {t: {} for t in tickers}
    if not tickers:
        return mesures

    lot = yf.download(tickers, period="1mo", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)

    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            if len(d) < 15:
                continue
            haut, bas, cloture = d["High"], d["Low"], d["Close"]
            prec = cloture.shift(1)
            tr = pd.concat([haut - bas, (haut - prec).abs(), (bas - prec).abs()],
                           axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            dernier = float(cloture.iloc[-1])
            mesures[t]["atr_pct"] = round(float(atr) / dernier * 100, 2) if dernier else None
            # VWAP approxime sur la derniere seance (typical price), faute de
            # donnees intrajournalieres gratuites fiables. Declare comme tel.
            tp = (float(haut.iloc[-1]) + float(bas.iloc[-1]) + dernier) / 3
            mesures[t]["vwap"] = round(tp, 4)
            mesures[t]["vwap_approx"] = True
            # Reference de volume: MEDIANE des 20 seances precedentes, pas la
            # moyenne. Mesure du 22.09.2026 sur VEEA: deux seances a 14 M juste
            # avant la detection tiraient la moyenne a ~9 M et ecrasaient le
            # RVOL a 0,1 alors que la seance etait ordinaire. La mediane resiste
            # au pic que le RVOL est precisement cense detecter.
            vol = d["Volume"]
            fenetre = vol.iloc[-21:-1]
            median = float(fenetre.median())
            mesures[t]["volume_moyen"] = median
            mesures[t]["volume_moyen_arith"] = float(fenetre.mean())
            mesures[t]["rvol"] = round(float(vol.iloc[-1]) / median, 2) if median else None
        except Exception as e:                     # un titre absent ne casse pas le lot
            logger.warning(f"[GAP] {t}: historique illisible ({e})")

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mesures[t]["float_actions"] = info.get("floatShares")
            si = info.get("shortPercentOfFloat")
            mesures[t]["short_interest_pct"] = round(si * 100, 2) if si else None
            mesures[t]["market_cap"] = info.get("marketCap")
        except Exception as e:
            logger.warning(f"[GAP] {t}: .info indisponible ({e})")
    return mesures


def construire_gaps(lignes: list, mesures: dict, mode: str) -> list:
    gaps = []
    for l in lignes:
        m = mesures.get(l["ticker"], {})
        gaps.append(Gap(
            ticker=l["ticker"],
            gap_pct=l.get("change_pct"),
            atr_pct=m.get("atr_pct"),
            rvol=m.get("rvol"),
            prix=l.get("prix"),
            vwap=m.get("vwap"),
            volume_moyen=m.get("volume_moyen") or l.get("volume"),
            float_actions=m.get("float_actions"),
            market_cap=m.get("market_cap") or l.get("market_cap"),
            short_interest_pct=m.get("short_interest_pct"),
            # Le catalyseur ne se devine pas: il est verifie a la main sur EDGAR
            # ou les news (SKILL.md, non negociable 1). Reste None ici.
            catalyseur=None,
            # En mode close, la tenue du gap est observable; en premarket, non.
            gap_tenu_30min=True if mode == "close" else None,
        ))
    return gaps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Scan de gaps - methode Germain")
    p.add_argument("--mode", choices=("premarket", "close", "ticker"), default="premarket")
    p.add_argument("--min-gap", type=int, default=5, choices=sorted(GAP_PAR_SEUIL))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--tickers", type=str, default=None, help="mode ticker: AAPL,TSLA")
    p.add_argument("--json", type=str, default=None, help="chemin de sortie JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    fenetre = _fenetre_actuelle()
    print(f"Fenetre de session actuelle (New York) : {fenetre}")
    if args.mode == "premarket" and fenetre not in ("premarket", "ouverture"):
        print("  ! Hors fenetre pre-marche : le champ Gap de Finviz reflete la "
              "derniere seance close, pas le jour a venir.")
    if args.mode == "close" and fenetre != "close":
        print("  ! Hors fenetre de cloture : la tenue du gap peut ne pas etre definitive.")

    if args.mode == "ticker":
        if not args.tickers:
            p.error("--mode ticker exige --tickers")
        lignes = [{"ticker": t.strip().upper(), "nom": "", "change_pct": None,
                   "prix": None, "volume": None, "market_cap": None}
                  for t in args.tickers.split(",") if t.strip()]
    else:
        lignes = scanner_finviz(args.min_gap, args.limit)
        if not lignes:
            print("Aucun gappeur ne passe les filtres. Le marche US doit etre ouvert "
                  "et des titres avoir gappe d'au moins "
                  f"{args.min_gap} % aujourd'hui.")
            return 1

    print(f"{len(lignes)} candidat(s) — enrichissement (1 appel groupe yfinance)…")
    mesures = enrichir([l["ticker"] for l in lignes])
    gaps = construire_gaps(lignes, mesures, args.mode)
    verdicts = [qualifier(g) for g in gaps]

    ordre = {"SQUEEZE": 0, "CONTINUATION": 1, "FADE": 2, "PUMP_RISK": 3, "INSUFFISANT": 4}
    verdicts.sort(key=lambda v: (ordre.get(v.classe, 9), -(v.comblement_attendu_pct or 0)))

    print("\n" + "=" * 72)
    for v in verdicts:
        print(rendre(v))
        print()
    repartition = {}
    for v in verdicts:
        repartition[v.classe] = repartition.get(v.classe, 0) + 1
    print("Repartition :", ", ".join(f"{k} {n}" for k, n in sorted(repartition.items())))
    print("\nRAPPEL : aucun catalyseur n'a ete verifie automatiquement. Aucune de ces "
          "lignes n'est un trade tant que le catalyseur n'est pas identifie et date "
          "sur EDGAR ou une source de news (SKILL.md, non negociable 1).")

    if args.json:
        # Enveloppe datee: la notation du soir (Trading_Agent/gaps/evaluer_gaps.py)
        # a besoin de la date et du mode pour savoir sur quelle fenetre juger.
        enveloppe = {
            "date": datetime.now(timezone(timedelta(hours=-4))).isoformat(),
            "mode": args.mode,
            "fenetre_au_scan": fenetre,
            "min_gap": args.min_gap,
            "verdicts": [v.__dict__ for v in verdicts],
        }
        with open(args.json, "w") as fo:
            json.dump(enveloppe, fo, indent=1, ensure_ascii=False)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
