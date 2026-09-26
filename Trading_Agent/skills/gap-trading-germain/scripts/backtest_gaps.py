#!/usr/bin/env python3
"""
backtest_gaps.py - rejoue des seances passees a travers le qualifier des gaps.

Pourquoi. Au 26.09.2026 toutes les decisions de methode reposent sur 20 verdicts
notes sur deux journees. GRML a change de signe d'un jour a l'autre, -36,0 % puis
+16,8 %, avec les memes alertes; le rendement moyen d'une vente a decouvert sur
les PUMP_RISK passe de +4,13 % a +1,86 % si l'on retire une seule observation.
Aucun seuil ne peut etre defendu sur cette base. Ce module produit des centaines
de verdicts au lieu de vingt.

CE QU'IL MESURE, ET CE QU'IL NE MESURE PAS

Il mesure le CLASSIFICATEUR, pas la decouverte. Le screener Finviz n'a pas
d'historique: il est impossible de reconstruire la liste des titres qu'il aurait
rendus le 15 aout. Le backtest part donc d'un univers de tickers fourni, repere
dans leur historique les seances qui remplissent les criteres du screener, et
juge le verdict rendu. Un taux de justesse issu de ce module ne dit rien de la
couverture du screener.

TROIS ANACHRONISMES, TOUS DECLARES

  * flottant, capitalisation, short interest: yfinance ne rend que les valeurs
    COURANTES. Les appliquer a une seance de mars est faux, et l'ecart est
    maximal precisement sur les nano caps qui font des regroupements d'actions.
    L'option --sans-statique rejoue sans ces champs et donne la sensibilite.
  * VWAP: les barres d'une minute ne remontent pas au-dela d'environ 30 jours.
    Le VWAP vaut donc None, ce qui correspond exactement au mode premarket de
    production depuis la revision du 25.09.2026.
  * catalyseur: jamais disponible apres coup, et jamais renseigne en production
    non plus. Cette absence est donc fidele.

La notation reutilise evaluer_gaps.noter(), pour que le backtest juge avec les
memes promesses que la notation du soir. Une divergence entre les deux serait un
defaut, pas une nuance.

Aufruf:
  python3 scripts/backtest_gaps.py --tickers GRML,PFSA,DCX --annees 2
  python3 scripts/backtest_gaps.py --fichier-tickers univers.txt --sans-statique
  python3 scripts/backtest_gaps.py --depuis-detections <dossier> --json resultat.json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_qualifier import (Gap, qualifier, VOLUME_JOUR_MINIMAL,      # noqa: E402
                           VOLUME_MOYEN_PLANCHER)

AGENT = "/home/berkam/Projets/Gestion_trade/Trading_Agent/gaps"

# Fenetre de notation, en seances apres la detection. 3 correspond au --jours 3
# d'evaluer_gaps, qui est la fenetre dont SQUEEZE a besoin pour se juger.
FENETRE_JOURS = 3


def _noter():
    """Importe la notation de production. Echoue explicitement si absente:
    un backtest qui juge autrement que la production ne vaut rien."""
    sys.path.insert(0, AGENT)
    try:
        from evaluer_gaps import noter
    except ImportError as e:
        raise RuntimeError(f"evaluer_gaps introuvable dans {AGENT}: "
                           f"le backtest doit juger comme la production") from e
    return noter


def univers_depuis_detections(dossier: str) -> list:
    """Tickers deja rencontres par le screener, lus dans les detections passees."""
    vus = set()
    for f in glob.glob(os.path.join(dossier, "*_premarket.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for v in d.get("verdicts", []):
            t = str(v.get("ticker", "")).strip().upper()
            if t:
                vus.add(t)
    return sorted(vus)


def charger_statique(tickers: list, cache: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Flottant, capitalisation et short interest, une requete par titre, mis
        en cache sur disque pour que les rejeux suivants soient gratuits.

        Ces valeurs sont COURANTES et seront appliquees a des seances passees.
        C'est un anachronisme assume, borne par l'option --sans-statique.

    Inputs:
        tickers (list): symboles
        cache (str): chemin du fichier JSON de cache

    Outputs:
        statique (dict): {ticker: {float_actions, market_cap, short_interest_pct}}
    --------------------------------------------------------------------------
    """
    connu = {}
    if cache and os.path.exists(cache):
        try:
            connu = json.load(open(cache, encoding="utf-8"))
        except Exception:
            connu = {}
    manquants = [t for t in tickers if t not in connu]
    if manquants:
        import yfinance as yf
        for t in manquants:
            try:
                info = yf.Ticker(t).info
                si = info.get("shortPercentOfFloat")
                connu[t] = {"float_actions": info.get("floatShares"),
                            "market_cap": info.get("marketCap"),
                            "short_interest_pct": round(si * 100, 2) if si else None}
            except Exception:
                connu[t] = {"float_actions": None, "market_cap": None,
                            "short_interest_pct": None}
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            json.dump(connu, open(cache, "w", encoding="utf-8"), indent=1)
    return {t: connu.get(t, {}) for t in tickers}


def _atr_pct(haut, bas, cloture, i: int, n: int = 14):
    """ATR(n) a l'indice i, en pourcentage du cours de cloture. None si court."""
    if i < n:
        return None
    tr = []
    for k in range(i - n + 1, i + 1):
        prec = cloture[k - 1]
        tr.append(max(haut[k] - bas[k], abs(haut[k] - prec), abs(bas[k] - prec)))
    ref = cloture[i]
    return round(st.mean(tr) / ref * 100, 2) if ref else None


def seances_candidates(serie: dict, min_gap: float, statique: dict) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Repere dans l'historique d'un titre les seances qui auraient passe les
        criteres du screener, et construit le Gap correspondant.

        Les criteres repris sont ceux que l'historique journalier permet de
        verifier: ecart d'ouverture, volume du jour, plancher structurel. Le
        volume relatif n'est pas un critere d'entree ici, il est une mesure:
        l'appliquer en filtre reproduirait le biais de tendance corrige le
        23.09.2026.

    Inputs:
        serie (dict): listes dates/ouverture/haut/bas/cloture/volume
        min_gap (float): ecart d'ouverture minimal en pourcentage
        statique (dict): champs courants du titre, ou {} si --sans-statique

    Outputs:
        cas (list): [(date, Gap)]
    --------------------------------------------------------------------------
    """
    dates = serie["dates"]
    o, h, b, c, v = (serie["ouverture"], serie["haut"], serie["bas"],
                     serie["cloture"], serie["volume"])
    cas = []
    for i in range(61, len(dates) - 1):        # 61: base pre-tendance disponible
        veille = c[i - 1]
        if not veille:
            continue
        gap = (o[i] / veille - 1) * 100
        if gap < min_gap:
            continue
        if v[i] < VOLUME_JOUR_MINIMAL:
            continue
        fenetre = v[i - 20:i]
        median = st.median(fenetre) if fenetre else 0
        if median and median < VOLUME_MOYEN_PLANCHER:
            continue
        base = v[i - 60:i - 20]
        median_base = st.median(base) if len(base) >= 20 else None
        cas.append((dates[i], Gap(
            ticker=serie["ticker"],
            gap_pct=round(gap, 2),
            atr_pct=_atr_pct(h, b, c, i),
            rvol=round(v[i] / median, 2) if median else None,
            rvol_pre_tendance=(round(v[i] / median_base, 2)
                               if median_base else None),
            prix=c[i],
            # Barres d'une minute indisponibles au-dela de ~30 jours: pas de
            # VWAP. Identique au mode premarket de production.
            vwap=None,
            ouverture=o[i],
            cloture_veille=veille,
            volume_jour=v[i],
            volume_moyen=median or None,
            float_actions=statique.get("float_actions"),
            market_cap=statique.get("market_cap"),
            short_interest_pct=statique.get("short_interest_pct"),
            catalyseur=None,          # jamais disponible, en backtest comme en production
            gap_tenu_30min=None,      # non mesurable sur des barres journalieres
        )))
    return cas


def serie_de_notation(serie: dict, i0: int, jours: int) -> dict:
    """Fenetre de cours au format attendu par evaluer_gaps.noter()."""
    fin = min(i0 + jours, len(serie["dates"]))
    return {"ouverture": serie["ouverture"][i0:fin],
            "cloture": serie["cloture"][i0:fin],
            "bas": serie["bas"][i0:fin],
            "haut": serie["haut"][i0:fin]}


def charger_series(tickers: list, annees: int) -> dict:
    """Historique journalier, UN SEUL appel groupe yfinance."""
    import yfinance as yf
    import pandas as pd
    out = {}
    lot = yf.download(tickers, period=f"{annees}y", interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False,
                      threads=True)
    for t in tickers:
        try:
            d = lot[t] if isinstance(lot.columns, pd.MultiIndex) else lot
            d = d.dropna()
            if len(d) < 80:
                continue
            out[t] = {"ticker": t,
                      "dates": [x.date().isoformat() for x in d.index],
                      "ouverture": [float(x) for x in d["Open"]],
                      "haut": [float(x) for x in d["High"]],
                      "bas": [float(x) for x in d["Low"]],
                      "cloture": [float(x) for x in d["Close"]],
                      "volume": [float(x) for x in d["Volume"]]}
        except Exception:
            continue
    return out


def rejouer(series: dict, statiques: dict, min_gap: float, jours: int) -> list:
    """Produit un verdict note par seance candidate."""
    noter = _noter()
    resultats = []
    for t, serie in series.items():
        index = {d: i for i, d in enumerate(serie["dates"])}
        for date, g in seances_candidates(serie, min_gap, statiques.get(t, {})):
            v = qualifier(g)
            note = noter(v.__dict__, serie_de_notation(serie, index[date], jours))
            fen = serie_de_notation(serie, index[date], jours)
            resultats.append({"date": date, "ticker": t, "classe": v.classe,
                              "resultat": note["resultat"],
                              "variation_pct": note["variation_pct"],
                              "fenetre": fen,
                              "alertes": v.alertes,
                              "gap_pct": g.gap_pct, "rvol": g.rvol,
                              "rvol_pre_tendance": g.rvol_pre_tendance,
                              "atr_pct": g.atr_pct})
    return resultats


def short_avec_stop(fenetre: dict, stop_pct: float):
    """
    --------------------------------------------------------------------------
    Purpose:
        Rendement d'une vente a decouvert ouverte a l'ouverture du jour du gap,
        avec un stop sur le plus haut, debouclee en fin de fenetre sinon.

        Pourquoi cette fonction existe. Le backtest du 26.09.2026 a mesure, sans
        stop, 109 gagnants a +21,6 % en moyenne contre 52 perdants a -74,8 %,
        pour une esperance de -9,57 % malgre 68 % de reussite. Le pire cas, IPDN
        le 10.09.2026, perd 1750 %. Une strategie de cette forme ne se juge pas
        sur son signal mais sur sa gestion du risque; la mesurer sans stop
        revient a mesurer autre chose.

        Le stop est declenche sur le plus haut de la seance, donc au pire moment
        possible de la journee. C'est conservateur et cela evite de supposer une
        execution favorable qu'aucune barre journaliere ne peut prouver.

    Inputs:
        fenetre (dict): listes ouverture/haut/cloture sur la fenetre de notation
        stop_pct (float): perte maximale toleree, en pourcentage positif

    Outputs:
        rendement (float | None): en pourcentage, None si la fenetre est vide
    --------------------------------------------------------------------------
    """
    o = fenetre.get("ouverture") or []
    if not o:
        return None
    entree = o[0]
    if not entree:
        return None
    seuil = entree * (1 + stop_pct / 100)
    for i, haut in enumerate(fenetre.get("haut") or []):
        if haut >= seuil:
            return -stop_pct          # stoppe: perte bornee
    sortie = (fenetre.get("cloture") or [entree])[-1]
    return round((entree / sortie - 1) * 100, 2)


def _taux(lignes) -> str:
    juges = [x for x in lignes if x["resultat"] in ("juste", "faux")]
    if not juges:
        return "aucun verdict jugeable"
    j = sum(1 for x in juges if x["resultat"] == "juste")
    return f"{j}/{len(juges)} ({j / len(juges) * 100:.0f} %)"


def rapport(resultats: list) -> str:
    l = []
    juges = [x for x in resultats if x["resultat"] in ("juste", "faux")]
    l.append(f"{len(resultats)} verdicts produits, {len(juges)} jugeables")
    l.append(f"Taux de justesse global : {_taux(resultats)}")
    l.append("")
    l.append("Par classe")
    l.append(f"  {'classe':14}{'n':>5}{'justesse':>22}{'variation moyenne':>20}")
    for classe in sorted({x["classe"] for x in resultats}):
        sous = [x for x in resultats if x["classe"] == classe]
        var = [x["variation_pct"] for x in sous if x["variation_pct"] is not None]
        moy = f"{st.mean(var):+.2f} %" if var else "-"
        l.append(f"  {classe:14}{len(sous):>5}{_taux(sous):>22}{moy:>20}")
    l.append("")
    l.append("Par alerte portee (sur les verdicts jugeables)")
    l.append(f"  {'alerte':38}{'n':>5}{'justesse':>18}")
    motifs = {"VWAP": "cours sous le VWAP",
              "sans catalyseur": "montee sans catalyseur (RVOL eleve)",
              "nano cap": "nano cap", "low float": "low float",
              "cloture de la veille": "gap efface (voir mise en garde)"}
    for cle, libelle in motifs.items():
        avec = [x for x in juges if any(cle in a for a in x["alertes"])]
        if avec:
            l.append(f"  {libelle:38}{len(avec):>5}{_taux(avec):>18}")
    sans_alerte = [x for x in juges if not x["alertes"]]
    if sans_alerte:
        l.append(f"  {'(aucune alerte)':38}{len(sans_alerte):>5}{_taux(sans_alerte):>18}")
    l.append("")
    l.append("  MISE EN GARDE sur 'gap efface': l'alerte se declenche quand le cours")
    l.append("  passe sous la cloture de la veille, et PUMP_RISK est note juste quand")
    l.append("  le cours finit sous l'ouverture du jour du gap, laquelle est par")
    l.append("  construction au-dessus de cette meme cloture. Les deux mesures sont")
    l.append("  donc fortement liees par construction: la justesse affichee ne mesure")
    l.append("  pas un pouvoir predictif.")
    l.append("")
    l.append("Vente a decouvert des PUMP_RISK (rendement = -variation)")
    pr = [x for x in resultats if x["classe"] == "PUMP_RISK"
          and x["variation_pct"] is not None]
    if pr:
        r = [-x["variation_pct"] for x in pr]
        gagnants = sum(1 for x in r if x > 0)
        l.append(f"  n={len(r)}  moyenne {st.mean(r):+.2f} %  "
                 f"mediane {st.median(r):+.2f} %  gagnants {gagnants}/{len(r)}")
        # Une moyenne portee par une seule ligne n'est pas une moyenne.
        if len(r) > 2:
            sans_extreme = sorted(r)[:-1]
            l.append(f"  sans la meilleure ligne : moyenne {st.mean(sans_extreme):+.2f} %")
        perdants = [x for x in r if x < 0]
        gagnants = [x for x in r if x > 0]
        if perdants and gagnants:
            l.append(f"  asymetrie : {len(gagnants)} gagnants a {st.mean(gagnants):+.1f} % "
                     f"contre {len(perdants)} perdants a {st.mean(perdants):+.1f} %")
        l.append("")
        l.append("  Avec un stop, declenche sur le plus haut de seance")
        l.append(f"    {'stop':>8}{'esperance':>13}{'mediane':>11}{'stoppes':>10}")
        for stop in (10, 15, 20, 30, 50):
            rs = [short_avec_stop(x["fenetre"], stop) for x in pr if x.get("fenetre")]
            rs = [v for v in rs if v is not None]
            if not rs:
                continue
            stoppes = sum(1 for v in rs if abs(v + stop) < 1e-9)
            l.append(f"    {stop:>7} %{st.mean(rs):>12.2f} %{st.median(rs):>10.2f} %"
                     f"{stoppes:>6}/{len(rs)}")
    return "\n".join(l)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Backtest du qualifier de gaps")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--fichier-tickers", type=str, default=None)
    p.add_argument("--depuis-detections", type=str, default=None,
                   help=f"univers lu dans les detections passees (defaut {AGENT}/detections)")
    p.add_argument("--annees", type=int, default=2)
    p.add_argument("--min-gap", type=float, default=5.0)
    p.add_argument("--jours", type=int, default=FENETRE_JOURS)
    p.add_argument("--sans-statique", action="store_true",
                   help="rejoue sans flottant/capitalisation/short interest (sensibilite)")
    p.add_argument("--cache", type=str,
                   default=os.path.join(AGENT, "backtests", "_statique.json"))
    p.add_argument("--json", type=str, default=None)
    a = p.parse_args(argv)

    if a.tickers:
        tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    elif a.fichier_tickers:
        tickers = [l.strip().upper() for l in open(a.fichier_tickers)
                   if l.strip() and not l.startswith("#")]
    else:
        dossier = a.depuis_detections or os.path.join(AGENT, "detections")
        tickers = univers_depuis_detections(dossier)
        print(f"Univers lu dans {dossier} : {len(tickers)} tickers")
    if not tickers:
        print("Aucun ticker. Voir --tickers, --fichier-tickers ou --depuis-detections.")
        return 1

    print(f"Historique : {a.annees} an(s), gap minimal {a.min_gap} %, "
          f"fenetre de notation {a.jours} seances")
    print("RAPPEL : ce backtest mesure le CLASSIFICATEUR, pas la couverture du")
    print("screener. L'historique des ecrans Finviz n'existe pas.")
    if a.sans_statique:
        print("Mode sensibilite : flottant, capitalisation et short interest ecartes.")
    print()

    series = charger_series(tickers, a.annees)
    print(f"{len(series)}/{len(tickers)} titres avec un historique exploitable")
    statiques = ({} if a.sans_statique
                 else charger_statique(list(series), a.cache))
    if not a.sans_statique:
        print("ANACHRONISME DECLARE : flottant, capitalisation et short interest")
        print("sont les valeurs COURANTES, appliquees a des seances passees.")
    print()

    resultats = rejouer(series, statiques, a.min_gap, a.jours)
    if not resultats:
        print("Aucune seance candidate. Abaisser --min-gap ou elargir l'univers.")
        return 1
    print(rapport(resultats))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"parametres": {"annees": a.annees, "min_gap": a.min_gap,
                                  "jours": a.jours,
                                  "sans_statique": a.sans_statique,
                                  "tickers": len(series)},
                   "resultats": resultats},
                  open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
