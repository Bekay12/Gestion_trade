#!/usr/bin/env python3
"""
Valley Scanner — détecte les creux exploitables et les distingue des chutes justifiées
======================================================================================

Pourquoi ce scanner existe. Les trois autres scanners mesurent un NIVEAU :
est-ce que cette entreprise est grande, sûre, rentable, en croissance. Un niveau
ne dit rien du moment. Trois épisodes mesurés le 15.09.2026 le montrent :

  Neste, février 2025    −29,7 % sur le mois, −10,5 % le 13 février (résultats
                         2024 : EBITDA comparable divisé par 2,8, dividende coupé
                         de 83 %). Chute JUSTIFIÉE. Plus bas à 6,79 €, cours
                         aujourd'hui 32,89 € (+384 %).
  Prysmian, février 2025 −12,2 % le 27 février sur une PRÉVISION de 2,25-2,35 Md€
                         jugée prudente. L'exercice s'est clos à 2,398 Md€, au-dessus
                         du haut de la fourchette sanctionnée. Plus bas 40,65 €,
                         aujourd'hui 127,25 € (+213 %).
  GEA, octobre 2023      −8,7 % sans une seule séance de choc, volume normal, quand
                         le MDAX faisait −6,5 %, KION −19,7 % et Dürr −24,9 %.
                         Pure revalorisation de secteur liée aux taux. Plus bas
                         31,77 €, aujourd'hui 63,85 € (+101 %).

Un détecteur unique manquerait les trois. Ce scanner en produit donc DEUX,
volontairement séparés :

  Calibration du seuil de part propre. Rejoué sur les épisodes connus
  (option --asof), le recul de GEA sur les trois mois à fin octobre 2023
  ressort à 67,6 % de part propre : −15,6 % pour le titre quand son indice
  faisait −6,8 %. Le défaut est donc fixé à 75 %, au-dessus de cette mesure et
  non ajusté dessus. Les deux contre-exemples restent hors du signal sans
  ambiguïté : Neste en février 2025 sort à 128,6 % de part propre (l'indice
  montait de 8,6 % pendant que le titre perdait 39,8 %) et Prysmian en février
  2025 à 339,4 %. Ces trois nombres sont la seule justification du seuil ;
  l'option --max-part-propre existe pour ne pas avoir à me croire.

  🕳️  DIVERGENCE  Baisse importante, MAJORITAIREMENT SYSTÉMATIQUE (marché ou
                 secteur), indicateur avancé propre INTACT. C'est le cas GEA :
                 il ne s'est rien passé chez l'entreprise. Neste en février 2025
                 ne déclenche PAS, et c'est correct — sa marge s'était réellement
                 effondrée, ce n'était pas un creux mais une dégradation.

  ↗️  INFLEXION   Le cours reste près de son plus bas alors que l'indicateur avancé
                 se retourne sur deux trimestres consécutifs. C'est le signal qui
                 aurait parlé pour Neste au deuxième trimestre 2025, pas en février.

  ⚠️  PIÈGE      Baisse importante, majoritairement PROPRE à l'entreprise, et
                 indicateur cassé. Affiché exprès : c'est ce qui ressemble le plus
                 à une occasion sans en être une.

Le garde-fou contre le piège de valeur est l'indicateur avancé. Il ne remplace pas
l'analyse : ce scanner produit des candidats à examiner, pas des décisions.

Budget yfinance : un seul appel groupé pour tout l'historique des cours et des
indices. Les appels par titre (fondamentaux, trimestriels) ne concernent QUE les
candidats retenus par le filtre de cours.

Usage:
  python Valley_scan.py --symbols-file ma_liste.txt
  python Valley_scan.py --min-drawdown 30 --signal divergence
  python Valley_scan.py --symbols-file liste.txt --top 15 --quiet
"""
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

SRC_DIR = Path(__file__).resolve().parent
POPULAR_FILE = SRC_DIR / "popular_symbols.txt"
OUTPUT_CSV = SRC_DIR / "valley_results.csv"

# Indice de référence par place de cotation. Le suffixe du ticker désigne la
# bourse ; sans suffixe, on suppose une cotation américaine.
# ^OSEAX (Oslo) ne renvoie plus de données : on retombe sur l'indice européen
# large plutôt que d'inventer une référence.
INDICES = {
    ".DE": "^GDAXI", ".F": "^GDAXI", ".PA": "^FCHI", ".AS": "^AEX", ".BR": "^AEX",
    ".MI": "FTSEMIB.MI", ".ST": "^OMX", ".CO": "^OMXC25", ".HE": "^OMXH25",
    ".L": "^FTSE", ".SW": "^SSMI", ".OL": "^STOXX", ".VI": "^STOXX",
    ".LS": "^STOXX", ".MC": "^STOXX", ".IR": "^STOXX",
}
INDICE_DEFAUT = "^GSPC"
INDICE_REPLI = "^STOXX"

THROTTLE = 0.25
_dernier_appel = 0.0


def _throttle():
    global _dernier_appel
    delta = time.time() - _dernier_appel
    if delta < THROTTLE:
        time.sleep(THROTTLE - delta)
    _dernier_appel = time.time()


def indice_de(ticker: str) -> str:
    """Indice de référence du titre, d'après le suffixe de sa place."""
    for suffixe, idx in INDICES.items():
        if ticker.upper().endswith(suffixe):
            return idx
    return INDICE_DEFAUT


def load_symbols(path=POPULAR_FILE) -> list:
    symboles = []
    with open(path, "r") as f:
        for ligne in f:
            s = ligne.strip()
            if s and not s.startswith("#"):
                symboles.append(s)
    return list(dict.fromkeys(symboles))


def _serie_close(data, ticker):
    """Série de clôtures nettoyée.

    Le dropna n'est pas cosmétique : sur les bourses européennes la dernière
    ligne est la séance en cours et porte NaN. Lue brute, elle contamine tout
    calcul qui touche iloc[-1] — c'est exactement le défaut qui rendait le
    critère de momentum de Combined_scan.py inopérant sur 41 titres sur 43.
    """
    try:
        serie = data[ticker]["Close"]
    except Exception:
        return None
    serie = pd.to_numeric(serie, errors="coerce").dropna()
    return serie if len(serie) >= 60 else None


# ══════════════════════════════════════════════════════════════
# 📉 MESURES DE COURS — aucun appel réseau, tout vient du lot déjà téléchargé
# ══════════════════════════════════════════════════════════════
def mesures_cours(close: pd.Series, close_idx: pd.Series) -> dict:
    """Baisse depuis le plus haut, et sa décomposition marché / propre.

    La décomposition est le cœur du scanner. Une baisse de 30 % dont 28 points
    viennent du marché n'est pas la même information qu'une baisse de 30 % dont
    28 points sont propres à l'entreprise. Le bêta sert à convertir le mouvement
    de l'indice en mouvement attendu du titre ; l'écart est la part propre.
    """
    dernier = float(close.iloc[-1])
    haut = float(close.max())
    bas = float(close.min())
    date_haut = close.idxmax()

    baisse = (dernier / haut - 1.0) * 100.0
    au_dessus_du_bas = (dernier / bas - 1.0) * 100.0

    # Bêta sur un an de rendements quotidiens communs aux deux séries.
    commun = pd.concat([close.pct_change(), close_idx.pct_change()], axis=1).dropna()
    commun.columns = ["titre", "indice"]
    commun = commun.tail(252)
    beta = np.nan
    if len(commun) >= 60 and float(commun["indice"].var()) > 0:
        beta = float(commun["titre"].cov(commun["indice"]) / commun["indice"].var())

    # Mouvement de l'indice sur la MÊME fenêtre que la baisse du titre.
    idx_fenetre = close_idx[close_idx.index >= date_haut]
    baisse_idx = np.nan
    if len(idx_fenetre) >= 2:
        baisse_idx = (float(idx_fenetre.iloc[-1]) / float(idx_fenetre.iloc[0]) - 1.0) * 100.0

    part_systematique = beta * baisse_idx if np.isfinite(beta) and np.isfinite(baisse_idx) else np.nan
    part_propre = baisse - part_systematique if np.isfinite(part_systematique) else np.nan
    fraction_propre = (abs(part_propre) / abs(baisse) * 100.0
                       if np.isfinite(part_propre) and abs(baisse) > 1e-9 else np.nan)

    sortie = {
        "prix": dernier, "haut_3a": haut, "bas_3a": bas,
        "date_haut": date_haut.date().isoformat(),
        "baisse_%": baisse, "au_dessus_du_bas_%": au_dessus_du_bas,
        "beta": beta, "baisse_indice_%": baisse_idx,
        "part_systematique_pp": part_systematique, "part_propre_pp": part_propre,
        "fraction_propre_%": fraction_propre,
    }
    # FENÊTRES COURTES. Mesurer la décomposition sur la baisse depuis le plus
    # haut à trois ans ne détecte PAS une revalorisation de secteur : sur trois
    # ans l'indice a le plus souvent monté, la part propre dépasse alors 100 %
    # et le signal ne se déclenche jamais. Le cas GEA d'octobre 2023 — −8,7 %
    # quand le MDAX faisait −6,5 % — est une fenêtre d'UN MOIS. La baisse longue
    # mesure le potentiel, les fenêtres courtes mesurent la cause.
    for nom, jours in (("1m", 21), ("3m", 63)):
        sortie.update(_fenetre_courte(close, close_idx, beta, jours, nom))
    return sortie


def _fenetre_courte(close, close_idx, beta, jours, nom) -> dict:
    """Mouvement du titre et de son indice sur les N dernières séances."""
    vide = {f"mouv_{nom}_%": np.nan, f"indice_{nom}_%": np.nan,
            f"fraction_propre_{nom}_%": np.nan}
    if len(close) <= jours or not np.isfinite(beta):
        return vide
    mouv = (float(close.iloc[-1]) / float(close.iloc[-jours - 1]) - 1.0) * 100.0
    idx_aligne = close_idx[close_idx.index >= close.index[-jours - 1]]
    if len(idx_aligne) < 2:
        return vide
    mouv_idx = (float(idx_aligne.iloc[-1]) / float(idx_aligne.iloc[0]) - 1.0) * 100.0
    attendu = beta * mouv_idx
    propre = mouv - attendu
    frac = abs(propre) / abs(mouv) * 100.0 if abs(mouv) > 1e-9 else np.nan
    return {f"mouv_{nom}_%": mouv, f"indice_{nom}_%": mouv_idx,
            f"fraction_propre_{nom}_%": frac}


# ══════════════════════════════════════════════════════════════
# 🔎 INDICATEUR AVANCÉ — appelé UNIQUEMENT pour les candidats
# ══════════════════════════════════════════════════════════════
def indicateur_avance(ticker: str) -> dict:
    """Le chiffre d'affaires trimestriel, en niveau et en tendance.

    Choix assumé : le vrai indicateur avancé diffère d'une entreprise à l'autre
    — entrée de commandes pour un équipementier, marge de vente par tonne pour
    un raffineur, croissance organique pour un câblier. Aucun de ces trois n'est
    disponible de façon homogène par API. Le chiffre d'affaires trimestriel est
    le plus proche substitut comparable entre tous les titres, et il faut le lire
    comme tel : il détecte une rupture franche, pas une inflexion de marge.

    Deux sorties distinctes :
      casse    le dernier trimestre est en recul de plus de 10 % sur un an
      retourne les deux derniers trimestres progressent en séquence
    """
    sortie = {"ca_var_a1_%": np.nan, "ca_sequence": None,
              "indicateur_casse": None, "indicateur_retourne": None,
              "marge_brute_%": np.nan, "tresorerie_degradee": None,
              "fcf_sur_sommet_%": np.nan, "fcf_serie": None}
    try:
        _throttle()
        tk = yf.Ticker(ticker)
        q = tk.quarterly_income_stmt
        if q is not None and not q.empty and "Total Revenue" in q.index:
            rev = q.loc["Total Revenue"].dropna()
            rev = rev.sort_index()          # du plus ancien au plus récent
            if len(rev) >= 2:
                sortie["ca_sequence"] = " → ".join(f"{v/1e6:.0f}" for v in rev.tail(4))
                # Progression séquentielle sur les deux derniers trimestres
                sortie["indicateur_retourne"] = bool(
                    len(rev) >= 3 and rev.iloc[-1] > rev.iloc[-2] > rev.iloc[-3])
            if len(rev) >= 5:
                var = (float(rev.iloc[-1]) / float(rev.iloc[-5]) - 1.0) * 100.0
                sortie["ca_var_a1_%"] = var
                sortie["indicateur_casse"] = bool(var < -10.0)
        # Trésorerie annuelle : même appel réseau que le reste (yfinance met
        # en cache l'objet Ticker), donc le garde-fou ne coûte rien de plus.
        try:
            cf = tk.cashflow
            flux = (list(cf.loc["Free Cash Flow"].dropna())
                    if cf is not None and "Free Cash Flow" in cf.index else [])
            bs = tk.balance_sheet
            dette = (list(bs.loc["Total Debt"].dropna())
                     if bs is not None and "Total Debt" in bs.index else [])
            # En millions, comme le reste des montants affichés : la série
            # sert à être lue, pas seulement comparée.
            sortie.update(etat_tresorerie([x / 1e6 for x in flux],
                                          [x / 1e6 for x in dette]))
        except Exception:
            sortie.update(etat_tresorerie([], []))

        _throttle()
        info = tk.info or {}
        gm = info.get("grossMargins")
        if gm is not None:
            sortie["marge_brute_%"] = float(gm) * 100.0
        if sortie["indicateur_casse"] is None:
            rg = info.get("revenueGrowth")
            if rg is not None:
                sortie["ca_var_a1_%"] = float(rg) * 100.0
                sortie["indicateur_casse"] = bool(float(rg) < -0.10)
    except Exception:
        pass
    return sortie


def etat_tresorerie(flux_libre, dette, seuil_sommet: float = 0.70) -> dict:
    """Le flux libre s'est-il dégradé ? Deux conditions, mesurées sur trois cas réels.

    Le signal d'inflexion reposait sur le seul chiffre d'affaires trimestriel,
    et cela a produit deux faux positifs coûteux le 15.09.2026, tous deux
    vérifiés ensuite dans les 10-K :

      Pool Corp  flux libre 828 → 600 → 310 M USD en deux ans, chiffre
                 d'affaires +2,2 % sur un an. Classé « inflexion ».
      ESAB       flux libre 282 → 304 → 213, dette 1 117 → 1 163 → 1 345,
                 résultat net du 1er semestre 2026 en recul de 40 % pendant
                 que le chiffre d'affaires montait de 11,4 %. Classé « inflexion ».
      Honeywell  flux libre 4 599 → 5 226 → 5 422, au plus haut de la fenêtre.

    Deux conditions, chacune nécessaire pour attraper un des deux cas :
      1. le dernier flux libre tombe sous `seuil_sommet` fois le sommet de la
         fenêtre — écarte Pool (0,37) et laisse Honeywell (1,00) ;
      2. le flux libre recule pendant que la dette monte — écarte ESAB, que la
         première condition laissait passer de justesse (0,701).

    Une série trop courte ou absente rend None : s'abstenir, jamais interdire.
    Le mode rétrospectif n'a aucun fondamental, et un garde-fou qui refuse par
    défaut rendrait le signal impossible à rejouer.

    Inputs:
        flux_libre (Sequence[float]): flux libre annuel, du plus récent au plus ancien
        dette (Sequence[float]): dette totale, même ordre
        seuil_sommet (float): fraction du sommet sous laquelle la trésorerie est dite dégradée

    Outputs:
        etat (dict): tresorerie_degradee (bool | None), fcf_sur_sommet_%, fcf_serie
    """
    flux = [float(x) for x in (flux_libre or []) if x is not None and x == x]
    if len(flux) < 2:
        return {"tresorerie_degradee": None, "fcf_sur_sommet_%": None, "fcf_serie": None}

    sommet = max(flux)
    part = flux[0] / sommet * 100.0 if sommet > 0 else None
    sous_le_sommet = bool(sommet > 0 and flux[0] < seuil_sommet * sommet)

    dettes = [float(x) for x in (dette or []) if x is not None and x == x]
    recule_et_sendette = bool(
        flux[0] < flux[1] and len(dettes) >= 2 and dettes[0] > dettes[1])

    return {
        "tresorerie_degradee": bool(sous_le_sommet or recule_et_sendette),
        "fcf_sur_sommet_%": part,
        "fcf_serie": " → ".join(f"{x:.0f}" for x in reversed(flux)),
    }


def classer(m: dict, f: dict, seuil_baisse: float, seuil_bas: float,
            max_part_propre: float = 75.0) -> tuple:
    """Rend (signal, note, motif). La note sert au tri, pas au verdict."""
    baisse = m["baisse_%"]
    # La part propre de la baisse LONGUE n'entre plus dans la décision : sur
    # trois ans l'indice a le plus souvent monté, elle dépasse alors 100 % et
    # ne distingue rien. La décision se prend sur la fenêtre courte, plus bas.
    casse = f.get("indicateur_casse")
    retourne = f.get("indicateur_retourne")

    assez_bas = m["au_dessus_du_bas_%"] <= seuil_bas
    assez_tombe = baisse <= -seuil_baisse

    # La reprise séquentielle ne suffit pas : deux hausses de suite sont le
    # comportement NORMAL d'une activité saisonnière au premier semestre.
    # Mesuré le 15.09.2026, POOL (matériel de piscine) déclenchait le signal
    # avec 1451 → 982 → 1138 → 1823 alors que son activité reculait de 4,9 %
    # sur un an. Le dernier trimestre doit donc aussi dépasser celui de l'an
    # passé. Une valeur annuelle inconnue reste tolérée : l'exiger rendrait le
    # signal impossible à rejouer en mode rétrospectif.
    var_an = f.get("ca_var_a1_%")
    annee_ok = var_an is None or var_an != var_an or float(var_an) > 0
    # Garde-fou de trésorerie : un chiffre d'affaires qui repart ne vaut rien
    # si le flux libre se dégrade. Voir etat_tresorerie() pour les trois cas
    # mesurés qui ont motivé cette condition.
    if f.get("tresorerie_degradee") is True:
        return ("⚠️ PIÈGE", 1.0,
                "activité en reprise apparente mais flux de trésorerie libre dégradé")
    if assez_bas and retourne and annee_ok and casse is not True:
        note = 3.0 + min(2.0, (seuil_bas - m["au_dessus_du_bas_%"]) / 10.0)
        return ("↗️ INFLEXION", note,
                "cours près du plus bas, activité en reprise séquentielle et sur un an")

    # La divergence se juge sur la fenêtre COURTE : c'est là qu'une
    # revalorisation de secteur est visible. La baisse longue ne sert qu'à
    # exiger qu'il y ait quelque chose à récupérer.
    mouv3m = m.get("mouv_3m_%", np.nan)
    frac3m = m.get("fraction_propre_3m_%", np.nan)
    if (assez_tombe and np.isfinite(mouv3m) and mouv3m < -5.0
            and np.isfinite(frac3m) and frac3m < max_part_propre and casse is not True):
        note = 3.0 + min(2.0, abs(baisse) / 25.0)
        return ("🕳️ DIVERGENCE", note,
                f"recul de {mouv3m:.0f}% sur 3 mois dont {frac3m:.0f}% propre, indicateur intact")

    if assez_tombe and casse is True:
        return "⚠️ PIÈGE", 1.0, "baisse accompagnée d'une rupture du chiffre d'affaires"

    if assez_tombe:
        return "👀 À SUIVRE", 2.0, "baisse marquée, décomposition ou indicateur indécis"

    return "—", 0.0, "pas de baisse significative"


# ══════════════════════════════════════════════════════════════
# 🚀 EXÉCUTION
# ══════════════════════════════════════════════════════════════
def analyser(symboles: list, seuil_baisse: float, seuil_bas: float,
             quiet: bool = False, asof: str | None = None,
             max_part_propre: float = 75.0, progress=None) -> pd.DataFrame:
    """asof rejoue le détecteur tel qu'il aurait parlé à une date passée.

    Sans cette option le détecteur est invérifiable : on ne peut pas savoir s'il
    aurait reconnu GEA en octobre 2023 ni, ce qui compte autant, s'il aurait
    correctement REFUSÉ Neste en février 2025. Les fondamentaux ne sont pas
    historisables par l'API : en mode asof l'indicateur avancé reste inconnu, et
    la classification le traite comme tel plutôt que comme intact.
    """
    indices = sorted({indice_de(s) for s in symboles} | {INDICE_REPLI})
    if not quiet:
        print(f"   📥 Téléchargement groupé : {len(symboles)} titres + "
              f"{len(indices)} indices, 3 ans"
              + (f", arrêté au {asof}" if asof else ""))

    if asof:
        fin = pd.Timestamp(asof)
        lot = yf.download(symboles + indices, start=(fin - pd.DateOffset(years=3)).date(),
                          end=fin.date(), progress=False, auto_adjust=False,
                          group_by="ticker", threads=True)
    else:
        lot = yf.download(symboles + indices, period="3y", progress=False,
                          auto_adjust=False, group_by="ticker", threads=True)

    series_idx = {i: _serie_close(lot, i) for i in indices}
    repli = series_idx.get(INDICE_REPLI)

    lignes, candidats = [], []
    for s in symboles:
        close = _serie_close(lot, s)
        if close is None:
            continue
        idx_nom = indice_de(s)
        close_idx = series_idx.get(idx_nom)
        # `or` sur une Series lève « truth value is ambiguous » : le repli doit
        # être testé explicitement contre None.
        if close_idx is None:
            close_idx = repli
        if close_idx is None:
            continue
        m = mesures_cours(close, close_idx)
        m.update({"ticker": s, "indice": idx_nom})
        lignes.append(m)
        # Le filtre de cours décide qui mérite un appel réseau supplémentaire.
        if m["baisse_%"] <= -seuil_baisse or m["au_dessus_du_bas_%"] <= seuil_bas:
            candidats.append(s)

    if not quiet:
        print(f"   🔎 {len(candidats)} candidats sur {len(lignes)} passent le filtre de cours "
              f"→ appel des fondamentaux pour ceux-là uniquement")

    fond = {}
    # `progress` sert à l'interface : sans lui, une fenêtre figée pendant le
    # téléchargement de plusieurs dizaines de titres est inutilisable.
    if progress:
        try:
            progress(0, len(candidats), "")
        except Exception:
            pass
    if asof:
        if not quiet:
            print("   ⏳ Mode rétrospectif : les fondamentaux ne sont pas historisables, "
                  "l'indicateur avancé reste inconnu")
        candidats = []
    for i, s in enumerate(candidats, 1):
        if not quiet:
            print(f"      [{i}/{len(candidats)}] {s}", end="\r", flush=True)
        if progress:
            try:
                progress(i, len(candidats), s)
            except Exception:
                pass
        fond[s] = indicateur_avance(s)
    if not quiet and candidats:
        print(" " * 60, end="\r")

    # Colonnes fondamentales toujours présentes, même quand aucun appel n'a eu
    # lieu (mode rétrospectif, ou aucun candidat) : l'affichage et le CSV
    # doivent avoir la même forme dans tous les cas.
    VIDE = {"ca_var_a1_%": np.nan, "ca_sequence": None, "indicateur_casse": None,
            "indicateur_retourne": None, "marge_brute_%": np.nan,
            "tresorerie_degradee": None, "fcf_sur_sommet_%": np.nan, "fcf_serie": None}
    for m in lignes:
        f = fond.get(m["ticker"], {})
        m.update({**VIDE, **f})
        signal, note, motif = classer(m, f, seuil_baisse, seuil_bas, max_part_propre)
        m.update({"signal": signal, "note": note, "motif": motif})

    return pd.DataFrame(lignes)


def afficher(df: pd.DataFrame, top: int | None) -> None:
    ordre = {"↗️ INFLEXION": 0, "🕳️ DIVERGENCE": 1, "👀 À SUIVRE": 2, "⚠️ PIÈGE": 3, "—": 4}
    df = df.copy()
    df["_o"] = df["signal"].map(ordre).fillna(9)
    df = df.sort_values(["_o", "note"], ascending=[True, False])

    for signal in ["↗️ INFLEXION", "🕳️ DIVERGENCE", "👀 À SUIVRE", "⚠️ PIÈGE"]:
        bloc = df[df["signal"] == signal]
        if bloc.empty:
            continue
        if top:
            bloc = bloc.head(top)
        print(f"\n  ─── {signal}  ({len(bloc)} titres) " + "─" * 40)
        vue = bloc[["ticker", "prix", "baisse_%", "au_dessus_du_bas_%", "beta",
                    "mouv_3m_%", "indice_3m_%", "fraction_propre_3m_%",
                    "ca_var_a1_%", "motif"]]
        vue = vue.rename(columns={
            "baisse_%": "baisse 3a", "au_dessus_du_bas_%": "au-dessus du bas",
            "mouv_3m_%": "titre 3m", "indice_3m_%": "indice 3m",
            "fraction_propre_3m_%": "part propre 3m", "ca_var_a1_%": "CA a/a"})
        print(vue.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n  " + "═" * 78)
    print("  Rappel de lecture :")
    print("   🕳️ DIVERGENCE  la baisse vient surtout du marché, l'entreprise ne s'est pas cassée")
    print("   ↗️ INFLEXION   le cours est resté bas, le chiffre d'affaires repart")
    print("   ⚠️ PIÈGE       la baisse est propre à l'entreprise ET l'activité recule")
    print("  Ce scanner produit des candidats à analyser, pas des décisions.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Détecteur de creux (divergence / inflexion)")
    p.add_argument("--symbols-file", type=str, default=None)
    p.add_argument("--min-drawdown", type=float, default=25.0,
                   help="baisse minimale depuis le plus haut 3 ans, en %% (défaut 25)")
    p.add_argument("--near-low", type=float, default=20.0,
                   help="distance maximale au plus bas 3 ans pour le signal inflexion, en %% (défaut 20)")
    p.add_argument("--signal", type=str, default=None,
                   choices=["divergence", "inflexion", "piege", "suivre"])
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--random", type=int, default=None, metavar="N",
                   help="tirer N symboles au hasard dans la liste (comme les autres scanners)")
    p.add_argument("--seed", type=int, default=None,
                   help="graine du tirage, pour rejouer exactement le même échantillon")
    p.add_argument("--throttle", type=float, default=0.25)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--max-part-propre", type=float, default=75.0,
                   help="part propre maximale du recul 3 mois pour le signal divergence, "
                        "en %% (défaut 75, calibré sur GEA octobre 2023 à 67,6)")
    p.add_argument("--asof", type=str, default=None, metavar="AAAA-MM-JJ",
                   help="rejouer le détecteur tel qu'il aurait parlé à cette date")
    args = p.parse_args(argv)

    global THROTTLE
    THROTTLE = max(0.05, args.throttle)

    fichier = Path(args.symbols_file) if args.symbols_file else POPULAR_FILE
    symboles = load_symbols(fichier)

    if args.random:
        import random as _random
        # Graine explicite : un tirage non rejouable rend le résultat
        # invérifiable, et c'est précisément ce qu'on reproche aux détecteurs.
        rng = _random.Random(args.seed)
        symboles = rng.sample(symboles, min(max(1, args.random), len(symboles)))

    if not args.quiet:
        print("\n" + "═" * 80)
        print("  🕳️  VALLEY SCANNER — creux exploitables contre chutes justifiées")
        print("═" * 80)
        print(f"   📋 {len(symboles)} symboles"
              + (f" tirés au hasard (graine {args.seed})" if args.random else "")
              + f" · seuil de baisse {args.min_drawdown:.0f}% · "
              f"proximité du plus bas {args.near_low:.0f}%")

    df = analyser(symboles, args.min_drawdown, args.near_low, args.quiet,
                  args.asof, args.max_part_propre)
    if df.empty:
        print("   Aucun titre exploitable (historique insuffisant ?)")
        return 1

    if args.signal:
        cle = {"divergence": "🕳️ DIVERGENCE", "inflexion": "↗️ INFLEXION",
               "piege": "⚠️ PIÈGE", "suivre": "👀 À SUIVRE"}[args.signal]
        df_aff = df[df["signal"] == cle]
    else:
        df_aff = df

    afficher(df_aff, args.top)
    df.sort_values("note", ascending=False).to_csv(OUTPUT_CSV, index=False)
    print(f"\n  💾 {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
