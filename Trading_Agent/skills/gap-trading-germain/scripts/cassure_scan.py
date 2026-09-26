#!/usr/bin/env python3
"""
cassure_scan.py - scan des cassures intraday SANS gap (complement de gap_scan).

Pourquoi un second screener. Le 22.09.2026, huit des dix plus fortes hausses de
la seance americaine ont ouvert a plat ou en baisse puis sont montees toute la
journee. Le screener de gaps ne peut pas les voir: son filtre "Gap: Up 5%" les
exclut par construction. Ce n'est pas un reglage a corriger dans gap_scan, c'est
une configuration differente, qui se detecte et se juge autrement.

Une fenetre, et une seule. Une cassure intraday n'existe pas avant l'ouverture:
il n'y a rien a screener en pre-marche. Le scan tourne en seance ou a la
cloture, et la cloture est la fenetre qui tranche, parce que la position de la
cloture dans le range du jour est l'arbitre de la configuration.

Deux regles d'architecture du depot, respectees comme dans gap_scan:
  * Le screener Finviz passe par core.finviz_screeners.run_screen.
  * L'enrichissement se fait en UN SEUL appel groupe yfinance.

Aufruf:
  python3 scripts/cassure_scan.py --min-hausse 5 --limit 50
  python3 scripts/cassure_scan.py --tickers INDP,DNA,AVAT
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassure_qualifier import (Cassure, qualifier_cassure,          # noqa: E402
                               rendre)

APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"
logger = logging.getLogger(__name__)

# Filtres Finviz de la cassure. Trois differences avec FILTRES_GERMAIN, et
# chacune est le point meme de ce second screener:
#
#   Gap              ABSENT. C'est la definition de la configuration.
#   Change           remplace Gap comme filtre de mouvement: la hausse se
#                    constate sur la seance, pas a l'ouverture.
#   Relative Volume  abaisse a "Over 1". Le RVOL de Finviz compare le volume du
#                    jour a une moyenne 3 mois que la tendance en cours a deja
#                    gonflee. Mesure du 22.09.2026 sur INDP, plus forte hausse
#                    de la seance: RVOL Finviz environ 1,62, donc invisible
#                    d'un filtre "Over 2". Le vrai test de volume se fait en
#                    aval sur la base pre-tendance, ou INDP donne 11,34.
#
# Les deux moyennes mobiles ne sont pas decoratives: une cassure se produit dans
# une tendance, et c'est ce qui la distingue d'un rebond de titre casse.
FILTRES_CASSURE = {
    "Market Cap.":                      "-Small (under $2bln)",
    "Price":                            "Under $50",
    "Current Volume":                   "Over 500K",
    "Average Volume":                   "Over 100K",
    "Relative Volume":                  "Over 1",
    "20-Day Simple Moving Average":     "Price above SMA20",
    "50-Day Simple Moving Average":     "Price above SMA50",
}
HAUSSE_PAR_SEUIL = {5: "Up 5%", 10: "Up 10%", 15: "Up 15%", 20: "Up 20%"}

# Au-dela de cet ecart d'ouverture, le titre releve de gap_scan et non d'ici.
# La frontiere est celle du screener de gaps: en dessous de 5 %, Finviz ne le
# retient pas comme gappeur.
GAP_MAXIMAL_PCT = 5.0


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


# Une seule implementation pour les deux screeners: la lecture des colonnes
# Finviz est le point ou le format change sans prevenir (cf. 23.09.2026).
from gap_scan import _num, lire_variation_pct, vwap_seance      # noqa: E402
from edgar_depots import collecter as collecter_edgar           # noqa: E402


def scanner_finviz(min_hausse: int, limit: int):
    """Liste des titres en forte hausse de seance. Rend une liste de dict."""
    sys.path.insert(0, APP_SRC)
    from core.finviz_screeners import run_screen           # noqa: E402

    filtres = dict(FILTRES_CASSURE)
    filtres["Change"] = HAUSSE_PAR_SEUIL.get(min_hausse, "Up 5%")
    df = run_screen(filtres, order="Change", limit=limit, ascend=False)
    if df is None or len(df) == 0:
        return []

    lignes = []
    for _, r in df.iterrows():
        sym = str(r.get("Ticker") or "").strip().upper()
        if not sym:
            continue
        lignes.append({
            "ticker": sym,
            "nom": str(r.get("Company") or "N/A"),
            "secteur": str(r.get("Sector") or "N/A"),
            "pays": str(r.get("Country") or "N/A"),
            "change_pct": lire_variation_pct(r),
            "prix": _num(r.get("Price")),
            "volume": _num(r.get("Volume")),
            "market_cap": _num(r.get("Market Cap")),
        })
    return lignes


def enrichir(tickers: list) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Range de la seance, VWAP, les deux RVOL, longueur de la tendance, float
        et short interest. UN SEUL appel groupe yfinance pour l'historique.

        L'historique couvre six mois: la base pre-tendance se lit sur les
        seances [-60:-20]. C'est la mesure qui donne son interet au module, donc
        la fenetre courte ne suffit pas.

    Inputs:
        tickers (list): symboles

    Outputs:
        mesures (dict): {ticker: {haut_jour, bas_jour, vwap, rvol, ...}}
    --------------------------------------------------------------------------
    """
    import yfinance as yf
    import pandas as pd

    mesures = {t: {} for t in tickers}
    if not tickers:
        return mesures

    lot = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)

    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            if len(d) < 25:
                continue
            haut, bas, cloture, ouverture = d["High"], d["Low"], d["Close"], d["Open"]
            mesures[t]["haut_jour"] = float(haut.iloc[-1])
            mesures[t]["bas_jour"] = float(bas.iloc[-1])
            dernier = float(cloture.iloc[-1])
            # Le mode --tickers ne passe pas par Finviz: sans cette valeur, le
            # prix reste inconnu et la position de cloture, qui est l'arbitre
            # de la cassure, devient inmesurable.
            mesures[t]["prix"] = dernier
            ouv = float(ouverture.iloc[-1])
            veille = float(cloture.iloc[-2])
            mesures[t]["cloture_veille"] = veille
            mesures[t]["gap_pct"] = round((ouv / veille - 1) * 100, 2) if veille else None
            mesures[t]["variation_seance_pct"] = (
                round((dernier / ouv - 1) * 100, 2) if ouv else None)

            vol = d["Volume"]
            jour = float(vol.iloc[-1])
            mesures[t]["volume_jour"] = jour
            fenetre = vol.iloc[-21:-1]
            median = float(fenetre.median())
            mesures[t]["volume_moyen"] = median
            mesures[t]["rvol"] = round(jour / median, 2) if median else None
            base = vol.iloc[-60:-20]
            if len(base) >= 20:
                median_base = float(base.median())
                mesures[t]["rvol_pre_tendance"] = (
                    round(jour / median_base, 2) if median_base else None)
            else:
                mesures[t]["rvol_pre_tendance"] = None

            # Longueur de la tendance: seances consecutives cloturant en hausse,
            # en remontant depuis aujourd'hui. Sert a raccourcir l'horizon quand
            # la cassure arrive au bout d'un long mouvement.
            serie = 0
            for i in range(len(cloture) - 1, 0, -1):
                if float(cloture.iloc[i]) > float(cloture.iloc[i - 1]):
                    serie += 1
                else:
                    break
            mesures[t]["seances_de_hausse"] = serie
        except Exception as e:                     # un titre absent ne casse pas le lot
            logger.warning(f"[CASSURE] {t}: historique illisible ({e})")

    # VWAP reel sur les barres d'une minute. Une cassure ne se lit qu'en seance,
    # donc le VWAP existe toujours ici, contrairement au mode premarket des gaps.
    for t, v in vwap_seance(tickers).items():
        mesures[t]["vwap"] = v

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mesures[t]["float_actions"] = info.get("floatShares")
            si = info.get("shortPercentOfFloat")
            mesures[t]["short_interest_pct"] = round(si * 100, 2) if si else None
            mesures[t]["market_cap"] = info.get("marketCap")
        except Exception as e:
            logger.warning(f"[CASSURE] {t}: .info indisponible ({e})")
    return mesures


def construire_cassures(lignes: list, mesures: dict, edgar: dict = None) -> list:
    cassures = []
    for l in lignes:
        m = mesures.get(l["ticker"], {})
        cassures.append(Cassure(
            ticker=l["ticker"],
            gap_pct=m.get("gap_pct"),
            variation_seance_pct=m.get("variation_seance_pct"),
            rvol=m.get("rvol"),
            rvol_pre_tendance=m.get("rvol_pre_tendance"),
            prix=l.get("prix") or m.get("prix"),
            vwap=m.get("vwap"),
            haut_jour=m.get("haut_jour"),
            bas_jour=m.get("bas_jour"),
            cloture_veille=m.get("cloture_veille"),
            volume_jour=l.get("volume") or m.get("volume_jour"),
            volume_moyen=m.get("volume_moyen"),
            float_actions=m.get("float_actions"),
            market_cap=m.get("market_cap") or l.get("market_cap"),
            short_interest_pct=m.get("short_interest_pct"),
            # Le catalyseur ne se devine pas: il se verifie a la main sur EDGAR
            # ou les news (SKILL.md, non negociable 1). Reste None ici.
            catalyseur=None,
            formulaire_sec=(edgar or {}).get(l["ticker"], {}).get("formulaire_sec"),
            seances_de_hausse=m.get("seances_de_hausse"),
        ))
    return cassures


def separer_des_gaps(cassures: list):
    """Sort les titres qui ont gappe: ils relevent de gap_scan, pas d'ici.

    Un titre peut passer le filtre Change de Finviz tout en ayant gappe; le
    laisser dans la liste reviendrait a le juger avec la mauvaise grille. Un gap
    inconnu ne fait pas sortir le titre: inconnu n'est pas mesure.
    """
    retenues, renvoyees = [], []
    for c in cassures:
        if c.gap_pct is not None and abs(c.gap_pct) >= GAP_MAXIMAL_PCT:
            renvoyees.append(c)
        else:
            retenues.append(c)
    return retenues, renvoyees


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Scan des cassures intraday sans gap")
    p.add_argument("--min-hausse", type=int, default=5, choices=sorted(HAUSSE_PAR_SEUIL),
                   help="hausse minimale de la seance, en %%")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--tickers", type=str, default=None, help="qualifie des titres nommes")
    p.add_argument("--json", type=str, default=None, help="chemin de sortie JSON")
    p.add_argument("--sans-edgar", action="store_true",
                   help="ne pas interroger EDGAR (formulaire_sec restera vide)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    fenetre = _fenetre_actuelle()
    print(f"Fenetre de session actuelle (New York) : {fenetre}")
    if fenetre in ("premarket", "hors-session"):
        print("  ! Une cassure intraday n'existe pas avant l'ouverture. Hors seance, "
              "ce scan lit la derniere seance close.")
    elif fenetre == "seance":
        print("  ! En seance, la position de la cloture dans le range n'est pas "
              "definitive: le verdict peut changer d'ici 16h00 ET.")

    if args.tickers:
        lignes = [{"ticker": t.strip().upper(), "nom": "", "change_pct": None,
                   "prix": None, "volume": None, "market_cap": None}
                  for t in args.tickers.split(",") if t.strip()]
    else:
        lignes = scanner_finviz(args.min_hausse, args.limit)
        if not lignes:
            print("Aucune cassure ne passe les filtres. Le marche US doit etre ouvert "
                  f"et des titres avoir progresse d'au moins {args.min_hausse} % "
                  "au-dessus de leurs moyennes mobiles.")
            return 1

    print(f"{len(lignes)} candidat(s) — enrichissement (1 appel groupe yfinance)…")
    tickers = [l["ticker"] for l in lignes]
    mesures = enrichir(tickers)
    edgar = {}
    if not args.sans_edgar:
        print("Depots SEC des 10 derniers jours (edgar_depots)…")
        try:
            edgar = collecter_edgar(tickers, jours=10)
        except Exception as e:
            print(f"  ! EDGAR indisponible ({e}); formulaire_sec restera vide")
    cassures = construire_cassures(lignes, mesures, edgar)
    retenues, renvoyees = separer_des_gaps(cassures)

    if renvoyees:
        noms = ", ".join(c.ticker for c in renvoyees)
        print(f"\n{len(renvoyees)} titre(s) ont gappe de {GAP_MAXIMAL_PCT:.0f} % ou plus "
              f"et relevent de gap_scan : {noms}")

    verdicts = [qualifier_cassure(c) for c in retenues]
    ordre = {"CASSURE": 0, "A_SURVEILLER": 1, "EPUISEMENT": 2,
             "PUMP_RISK": 3, "INSUFFISANT": 4}
    verdicts.sort(key=lambda v: (ordre.get(v.classe, 9), -(v.position_cloture or 0)))

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
        # Mesures conservees a cote des verdicts, meme motif que dans gap_scan:
        # une detection doit pouvoir se rejuger hors ligne.
        enveloppe = {
            "date": datetime.now(timezone(timedelta(hours=-4))).isoformat(),
            "mode": "cassure",
            "fenetre_au_scan": fenetre,
            "min_hausse": args.min_hausse,
            "renvoyees_vers_gap_scan": [c.ticker for c in renvoyees],
            "verdicts": [v.__dict__ for v in verdicts],
            "mesures": {c.ticker: c.__dict__ for c in retenues},
            "depots_sec": edgar,
        }
        with open(args.json, "w") as fo:
            json.dump(enveloppe, fo, indent=1, ensure_ascii=False)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
