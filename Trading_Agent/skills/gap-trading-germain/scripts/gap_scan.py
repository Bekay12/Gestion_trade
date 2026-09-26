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
from edgar_depots import collecter as collecter_edgar      # noqa: E402

APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"
logger = logging.getLogger(__name__)

# Preset derive de "morning_gappers" (Code_Germain/Gap_Screen.py), revise le
# 22.09.2026 apres mesure sur la seance du jour. Trois ecarts avec l'original,
# chacun motive; le detail est dans docs/methode-gaps-et-cassures.md.
#
#   Current Volume  remplace Average Volume comme plancher de liquidite. Le
#                   seuil de 500 K de P7 Ch.01 est conserve, la colonne change:
#                   c'est le volume du jour qui permet d'executer. MAZE (ADV
#                   290 127, 2 726 280 titres le jour du gap, +26,8 %) etait
#                   ecarte par la moyenne; STFS (45 K le jour meme) reste
#                   ecarte, ce qui est correct.
#   Average Volume  conserve, mais rabaisse a un plancher structurel: un titre
#                   habituellement mort reste difficile a revendre.
#   Price           releve de 20 a 50 $. MAZE cotait 22,39 avant son gap et
#                   etait donc invisible. Le controle du risque reste la
#                   capitalisation (-Small), et la normalisation du gap par
#                   l'ATR mesure deja si le gap est significatif pour le titre.
#   Float Short     RETIRE des filtres durs. Le qualifier definit CONTINUATION
#                   sans short interest (catalyseur + RVOL + VWAP); l'exiger en
#                   amont transformait le screener en detecteur de squeeze et
#                   rendait la classe CONTINUATION inatteignable. Le seuil de
#                   10 % reste actif dans gap_qualifier.SHORT_INTEREST_SQUEEZE,
#                   ou il discrimine au lieu d'exclure.
FILTRES_GERMAIN = {
    "Market Cap.":     "-Small (under $2bln)",
    "Price":           "Under $50",
    "Current Volume":  "Over 500K",
    "Average Volume":  "Over 100K",
    "Relative Volume": "Over 2",
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


def _num(v):
    """Nombre Finviz ('1.2M', '45.3%', '3,200') vers float, None si illisible."""
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


# Colonnes possibles pour la variation, de la plus recente a la plus ancienne.
# Mesure du 23.09.2026: finvizfinance rend "Change %" avec une chaine deja en
# pourcentage ("180.75%"). Le code lisait "Change" et multipliait par 100, donc
# gap_pct valait None pour tous les candidats et aucune normalisation par l'ATR
# n'avait lieu. Les deux noms sont acceptes pour survivre au prochain changement.
COLONNES_VARIATION = ("Change %", "Change")


def lire_variation_pct(ligne) -> "float | None":
    """
    --------------------------------------------------------------------------
    Purpose:
        Variation en POURCENTAGE depuis la cloture precedente, quelle que soit
        la forme rendue par Finviz.

        Deux formats coexistent selon la version de finvizfinance:
          "180.75%" (chaine, deja un pourcentage)  -> rendu tel quel
          0.0477    (nombre, une fraction)          -> multiplie par 100

        Le marqueur "%" dans la chaine brute tranche sans ambiguite. Un nombre
        nu est traite selon l'ancienne convention, la seule documentee pour ce
        format.

        Recoupement du 23.09.2026 (WHLR): Finviz annoncait +180,75 % a 5,31,
        IBKR donnait une cloture de 1,87 la veille et 6,59 en direct. La valeur
        est bien le mouvement PRE-MARCHE du jour, pas la seance precedente.

    Inputs:
        ligne (Mapping): une ligne du DataFrame Finviz

    Outputs:
        variation (float | None): pourcentage, None si aucune colonne lisible
    --------------------------------------------------------------------------
    """
    for col in COLONNES_VARIATION:
        brut = ligne.get(col)
        if brut is None:
            continue
        texte = str(brut).strip()
        if not texte or texte.lower() in ("nan", "none", "-"):
            continue
        deja_pourcentage = texte.endswith("%")
        valeur = _num(brut)
        if valeur is None:
            continue
        return round(valeur if deja_pourcentage else valeur * 100, 2)
    return None


def vwap_seance(tickers: list) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        VWAP reel de la seance en cours, calcule sur les barres d'une minute
        ponderees par leur volume. UN SEUL appel groupe yfinance.

        Ce qu'il remplace, et pourquoi. Le code utilisait le prix typique de la
        derniere barre JOURNALIERE. En mode premarket, cette barre est celle de
        la VEILLE, puisque celle du jour n'existe pas encore. L'alerte
        "cours sous le VWAP, pression vendeuse" comparait donc le prix
        pre-marche au milieu de fourchette de la veille, et enoncait une
        affirmation sur le flux du jour a partir des donnees de la veille.

        Mesure du 25.09.2026 sur DCX: a 07h19 ET la derniere barre journaliere
        etait celle du 24.09 (prix typique 0,0597), et l'alerte annoncait
        "-16,2 %". Sur deux journees notees, cette alerte portait 3 des
        5 erreurs de classement, soit 50 % d'erreur contre 14 % sans elle.

        Le pre-marche reste hors de portee: yfinance rend les barres d'une
        minute avant 09h30 ET, mais avec un volume NUL (verifie le 25.09.2026,
        volume total identique avec et sans prepost). Sans volume, pas de
        ponderation, donc pas de VWAP. La fonction rend None pour ces titres
        plutot qu'une valeur fabriquee.

    Inputs:
        tickers (list): symboles

    Outputs:
        vwaps (dict): {ticker: float | None}
    --------------------------------------------------------------------------
    """
    import yfinance as yf
    import pandas as pd

    out = {t: None for t in tickers}
    if not tickers:
        return out
    try:
        lot = yf.download(tickers, period="1d", interval="1m", prepost=False,
                          group_by="ticker", auto_adjust=False, progress=False,
                          threads=True)
    except Exception as e:
        logger.warning(f"[GAP] barres 1 min indisponibles ({e}); VWAP non mesure")
        return out

    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            vol = d["Volume"]
            total = float(vol.sum())
            if d.empty or total <= 0:
                continue          # seance non commencee, ou aucun echange
            tp = (d["High"] + d["Low"] + d["Close"]) / 3
            out[t] = round(float((tp * vol).sum() / total), 4)
        except Exception as e:
            logger.warning(f"[GAP] {t}: VWAP non calculable ({e})")
    return out


def scanner_finviz(min_gap: int, limit: int):
    """Liste des gappers via le screener de l'application. Rend une liste de dict."""
    sys.path.insert(0, APP_SRC)
    from core.finviz_screeners import run_screen           # noqa: E402

    filtres = dict(FILTRES_GERMAIN)
    filtres["Gap"] = GAP_PAR_SEUIL.get(min_gap, "Up 5%")
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


def enrichir(tickers: list, mode: str = "close") -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        ATR(14), VWAP du jour, les deux RVOL, float et short interest pour
        chaque titre.
        UN SEUL appel groupe yfinance pour l'historique (budget de requetes du
        depot); les champs de float/short interest viennent de .info, appele une
        fois par titre et uniquement pour les candidats deja filtres.

        L'historique couvre six mois et non un mois: la base pre-tendance du
        second RVOL se lit sur les seances [-60:-20], anterieures a une tendance
        eventuellement en cours. Le nombre d'appels reseau ne change pas.

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

    lot = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
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
            # Idem cassure_scan: le mode --tickers ne passe pas par Finviz.
            mesures[t]["prix"] = dernier
            # Cloture de la veille: base du signal "gap efface" (Germain, VBIO
            # 24.09.2026). En pre-marche la derniere barre est celle de la
            # veille, donc c'est elle la reference; en seance c'est l'avant
            # derniere.
            if len(cloture) >= 2:
                mesures[t]["cloture_veille"] = float(cloture.iloc[-2])
            # L'ouverture du jour, et non gap_pct, porte la condition "a ouvert
            # en hausse": gap_pct vaut la variation courante de Finviz.
            mesures[t]["ouverture"] = float(d["Open"].iloc[-1])
            mesures[t]["atr_pct"] = round(float(atr) / dernier * 100, 2) if dernier else None
            # Reference de volume: MEDIANE des 20 seances precedentes, pas la
            # moyenne. Mesure du 22.09.2026 sur VEEA: deux seances a 14 M juste
            # avant la detection tiraient la moyenne a ~9 M et ecrasaient le
            # RVOL a 0,1 alors que la seance etait ordinaire. La mediane resiste
            # au pic que le RVOL est precisement cense detecter.
            vol = d["Volume"]
            jour = float(vol.iloc[-1])
            mesures[t]["volume_jour"] = jour
            fenetre = vol.iloc[-21:-1]
            median = float(fenetre.median())
            mesures[t]["volume_moyen"] = median
            mesures[t]["volume_moyen_arith"] = float(fenetre.mean())
            mesures[t]["rvol"] = round(jour / median, 2) if median else None
            # Seconde base de comparaison, anterieure a une tendance en cours.
            # La fenetre glissante ci-dessus est gonflee par le mouvement qu'elle
            # est censee detecter: mesure du 22.09.2026 sur INDP, plus forte
            # hausse de la seance, RVOL 1,99 sur 20 seances contre 11,34 sur la
            # base [-60:-20]. Sur un titre sans tendance prealable les deux
            # mesures coincident (MAZE le meme jour: 13,45 et 14,38), donc la
            # seconde ne peut pas degrader un classement.
            base = vol.iloc[-60:-20]
            if len(base) >= 20:
                median_base = float(base.median())
                mesures[t]["rvol_pre_tendance"] = (
                    round(jour / median_base, 2) if median_base else None)
            else:
                # Historique trop court (introduction recente): la mesure n'existe
                # pas, elle ne vaut pas zero.
                mesures[t]["rvol_pre_tendance"] = None
        except Exception as e:                     # un titre absent ne casse pas le lot
            logger.warning(f"[GAP] {t}: historique illisible ({e})")

    # VWAP reel, seulement quand une seance est en cours. En pre-marche il
    # n'existe pas encore, et une valeur inventee vaut moins que rien: elle
    # produit une alerte de "pression vendeuse" tiree de la veille.
    if mode == "premarket":
        for t in tickers:
            mesures[t]["vwap"] = None
            mesures[t]["vwap_motif"] = "seance non commencee: aucun VWAP n'existe"
    else:
        for t, v in vwap_seance(tickers).items():
            mesures[t]["vwap"] = v
            if v is None:
                mesures[t]["vwap_motif"] = "barres 1 min sans volume exploitable"

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


def construire_gaps(lignes: list, mesures: dict, mode: str, edgar: dict = None) -> list:
    gaps = []
    for l in lignes:
        m = mesures.get(l["ticker"], {})
        gaps.append(Gap(
            ticker=l["ticker"],
            gap_pct=l.get("change_pct"),
            atr_pct=m.get("atr_pct"),
            rvol=m.get("rvol"),
            rvol_pre_tendance=m.get("rvol_pre_tendance"),
            prix=l.get("prix") or m.get("prix"),
            vwap=m.get("vwap"),
            ouverture=m.get("ouverture"),
            cloture_veille=m.get("cloture_veille"),
            # Finviz rend le volume de la seance en cours dans la colonne Volume;
            # yfinance le redonne sur la derniere barre. La premiere source fait
            # foi quand elle existe, elle est plus fraiche en seance.
            volume_jour=l.get("volume") or m.get("volume_jour"),
            volume_moyen=m.get("volume_moyen"),
            float_actions=m.get("float_actions"),
            market_cap=m.get("market_cap") or l.get("market_cap"),
            short_interest_pct=m.get("short_interest_pct"),
            # Le catalyseur ne se devine pas: il est verifie a la main sur EDGAR
            # ou les news (SKILL.md, non negociable 1). Reste None ici.
            catalyseur=None,
            # En revanche le TYPE de depot se lit automatiquement. Renseigne
            # depuis le 26.09.2026 par edgar_depots: le champ existait depuis
            # l'origine et n'avait jamais ete rempli, alors que le motif
            # decisif etait chaque jour dans les depots.
            formulaire_sec=(edgar or {}).get(l["ticker"], {}).get("formulaire_sec"),
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
    p.add_argument("--sans-edgar", action="store_true",
                   help="ne pas interroger EDGAR (formulaire_sec restera vide)")
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

    tickers = [l["ticker"] for l in lignes]
    print(f"{len(lignes)} candidat(s) — enrichissement (1 appel groupe yfinance)…")
    mesures = enrichir(tickers, mode=args.mode)

    # Depots SEC: une requete par titre sur data.sec.gov, sans cle et sans
    # impact sur le budget yfinance.
    edgar = {}
    if not args.sans_edgar:
        print("Depots SEC des 10 derniers jours (edgar_depots)…")
        try:
            edgar = collecter_edgar(tickers, jours=10)
        except Exception as e:
            print(f"  ! EDGAR indisponible ({e}); formulaire_sec restera vide")

    gaps = construire_gaps(lignes, mesures, args.mode, edgar)
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
    a_lire = {t: r["a_lire"] for t, r in edgar.items() if r.get("a_lire")}
    if a_lire:
        print("\nDEPOTS A LIRE (le type ne dit pas le sens ; cas GLND du 25.09.2026 :")
        print("un item 1.01 annoncait un report de forage de deux ans)")
        for t, dep in sorted(a_lire.items()):
            for d in dep:
                items = f" [{d['items']}]" if d["items"] else ""
                print(f"  {t:6} {d['date']}  {d['form']:8}{items}")
                print(f"         {d['url']}")

    print("\nRAPPEL : aucun catalyseur n'a ete verifie automatiquement. Aucune de ces "
          "lignes n'est un trade tant que le catalyseur n'est pas identifie et date "
          "sur EDGAR ou une source de news (SKILL.md, non negociable 1).")

    if args.json:
        # Enveloppe datee: la notation du soir (Trading_Agent/gaps/evaluer_gaps.py)
        # a besoin de la date et du mode pour savoir sur quelle fenetre juger.
        #
        # Les MESURES sont conservees a cote des verdicts depuis le 25.09.2026.
        # Motif: le fichier ne portait que la classe et ses motifs, donc ni le
        # RVOL, ni le gap, ni le VWAP. Impossible alors de repondre apres coup a
        # "ce titre serait-il repasse sous le seuil si le catalyseur avait ete
        # verifie ?", ni de recalculer un verdict sans relancer tout le scan.
        # Une detection doit porter de quoi se rejuger sans le reseau.
        enveloppe = {
            "date": datetime.now(timezone(timedelta(hours=-4))).isoformat(),
            "mode": args.mode,
            "fenetre_au_scan": fenetre,
            "min_gap": args.min_gap,
            "verdicts": [v.__dict__ for v in verdicts],
            "mesures": {g.ticker: g.__dict__ for g in gaps},
            # Les depots sont conserves: ils portent les URL de ce qui reste a
            # lire, et permettent de rejuger une detection sans rappeler EDGAR.
            "depots_sec": edgar,
        }
        with open(args.json, "w") as fo:
            json.dump(enveloppe, fo, indent=1, ensure_ascii=False)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
