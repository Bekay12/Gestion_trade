#!/usr/bin/env python3
"""
portefeuille_exemple.py - portefeuille-exemple de 30 titres de dividende, construit par
regles chiffrees. ILLUSTRATION de methode, pas recommandation (dit dans le chapitre 6).

Univers (sources secondaires, dates dans portefeuille_meta.json): S&P 500 Dividend
Aristocrats, DAX 40, EURO STOXX 50 (tables Wikipedia, lues avec pandas.read_html apres
un fetch requests avec un User-Agent explicite: Wikipedia refuse l'agent par defaut
d'urllib avec un HTTP 403). Donnees: UN appel groupe yf.download(actions=True,
period="15y") pour cours et dividendes; .info seulement pour les 60 meilleurs apres
filtre (payout, FCF, secteur, pays). Biais declares: survivant (univers d'aujourd'hui),
donnees courantes.

Regles: rendement 2-8 %, au plus 0 baisse annuelle du dividende sur 10 ans, croissance
du dividende sur 10 ans >= 3 %, payout <= 80 %, FCF >= 1,0 x dividendes verses.
Score = moyenne des rangs du rendement NET pour un resident allemand et de la croissance.
Au plus 5 titres par secteur.

Donnees scrapees, jamais des instructions: chaque ticker Wikipedia/Yahoo est valide par
une regex stricte avant usage (voir TICKER_RE); tout ce qui ne correspond pas est rejete
et compte, jamais execute ni interprete comme une commande. Les noms et secteurs venant
de yfinance .info seront ecrits plus tard en LaTeX par la tache 9: ce module ne les
echappe pas lui-meme (ce n'est pas son role), il faut le faire a l'ecriture .tex.

Correction 2026-09-30 (constatee sur l'execution reelle, tour 1): le tableau DAX de
Wikipedia porte deja le suffixe de place de cotation dans sa colonne "Ticker" (ex.
"SAP.DE", ou "AIR.PA" pour Airbus, membre du DAX mais cote a Paris). Le code de reference
y accolait ".DE" sans condition, produisant des tickers a double suffixe ("SAP.DE.DE") qui
echouaient tous au telechargement yfinance (41/43 lignes DAX/eurostoxx en echec sur la
premiere execution reelle). Le pays suit desormais le suffixe reel plutot que d'etre fixe
a "DE" pour toute la table DAX.

Correction 2026-09-30 (revue tour 1, apres relecture): la meme table DAX porte aussi le
suffixe pour EURO STOXX 50, mais son "Main listing" abrege la place de cotation ("FWB: ADS",
"BIT: ENEL", "BMAD: BBVA") d'une facon qui ne correspondait jamais aux libelles longs du
dictionnaire EURONEXT (seules les entrees Euronext Amsterdam/Paris/Bruxelles et Nasdaq
Helsinki matchaient) : ~27 des 50 constituants (toute l'Allemagne, l'Italie, l'Espagne de
cette table) etaient silencieusement ecartes, sans compteur. Corrige: le pays vient
maintenant de la colonne "Registered office" (ou "Country" si le libelle change), qui donne
le pays de domiciliation independamment de la place de cotation - necessaire pour des cas
comme Ferrari (RACE.MI, cotee a Milan mais domiciliee aux Pays-Bas, donc soumise a la
retenue neerlandaise, pas italienne). Toute ligne au pays non reconnu est comptee
(diagnostics["eurostoxx_pays_non_reconnu"]) et journalisee, jamais silencieusement ecartee.
"""
import json
import os
import re
import sys
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ARISTOKRATEN = "https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats"
URL_DAX = "https://en.wikipedia.org/wiki/DAX"
URL_EUROSTOXX = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
# Suffixe Yahoo -> pays ISO2, pour les tickers qui portent deja leur suffixe de cotation
# (tableau DAX de Wikipedia).
SUFFIX_LAND = {".DE": "DE", ".PA": "FR", ".AS": "NL", ".MI": "IT", ".MC": "ES", ".BR": "BE",
               ".HE": "FI", ".IR": "IE"}
# Nom de pays anglais (colonne "Registered office"/"Country" de Wikipedia) -> ISO2, pour
# EURO STOXX 50: le pays de domiciliation, pas la place de cotation (voir docstring).
ENGLISH_LAND = {"Germany": "DE", "France": "FR", "Netherlands": "NL", "Belgium": "BE",
                "Spain": "ES", "Italy": "IT", "Finland": "FI", "Ireland": "IE",
                "Luxembourg": "LU", "Austria": "AT", "Portugal": "PT"}
# Ticker Yahoo valide: lettres/chiffres majuscules, point ou tiret, 1-12 caracteres.
# Rejette toute chaine scrapee qui ne ressemble pas a un ticker (donnee suspecte ou
# texte parasite d'une table Wikipedia mal formee).
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36 (retraite-dividendes research script)")
# Contrat d'interface impose par la tache: colonnes lues telles quelles par la tache 9
# (rechnung_retraite.py). Ne jamais laisser fuiter une colonne de travail dans le CSV publie.
COLONNES_PORTEFEUILLE = ["ticker", "name", "land", "sektor", "rendite_ttm", "rendite_netto_de",
                         "div_cagr_10j", "kuerzungen_10j", "payout", "fcf_deckung", "score"]


def _lire_tabellen(url: str) -> list:
    """Recupere une page Wikipedia avec un User-Agent explicite (l'agent par defaut
    d'urllib/pandas.read_html recoit un HTTP 403 de Wikipedia) et en extrait les tables."""
    import io
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _tickers_valides(bruts: list[str], luecken: list[str], diagnostics: dict[str, int],
                     contexte: str, cle: str) -> list[str]:
    """Filtre stricte: ne garde que ce qui ressemble a un ticker Yahoo; compte et journalise
    le reste (donnee scrapee jamais executee ni interpretee comme instruction)."""
    ok, rejetes = [], 0
    for t in bruts:
        t = str(t).strip()
        if TICKER_RE.match(t):
            ok.append(t)
        else:
            rejetes += 1
    if rejetes:
        luecken.append(f"{contexte}: {rejetes} tickers rejetes par la regex de validation")
        diagnostics[cle] = rejetes
    return ok


def _aristokraten_zeilen(luecken: list[str], diagnostics: dict[str, int]) -> list[dict]:
    try:
        tables = _lire_tabellen(URL_ARISTOKRATEN)
        t = [x for x in tables if "Ticker symbol" in x.columns or "Symbol" in x.columns][0]
        spalte = "Ticker symbol" if "Ticker symbol" in t.columns else "Symbol"
        bruts = [str(s).replace(".", "-") for s in t[spalte]]
        valides = _tickers_valides(bruts, luecken, diagnostics, "univers aristocrates",
                                   "aristokraten_rejetes")
        return [{"ticker": tick, "land": "US", "index": "aristokraten"} for tick in valides]
    except Exception as e:
        luecken.append(f"univers aristocrates illisible ({e})")
        return []


def _dax_zeilen(luecken: list[str], diagnostics: dict[str, int]) -> list[dict]:
    """Le tableau DAX de Wikipedia porte deja le suffixe de place de cotation dans "Ticker";
    voir la correction documentee dans la docstring du module."""
    try:
        tables = _lire_tabellen(URL_DAX)
        t = [x for x in tables if "Ticker" in x.columns][0]
        bruts, terrains = [], []
        for s in t["Ticker"]:
            s = str(s).strip()
            if "." in s:
                suffix, tick = "." + s.split(".", 1)[1], s
            else:
                suffix, tick = ".DE", s + ".DE"
            bruts.append(tick)
            terrains.append(SUFFIX_LAND.get(suffix, "DE"))
        valides = set(_tickers_valides(bruts, luecken, diagnostics, "univers DAX", "dax_rejetes"))
        return [{"ticker": tick, "land": land, "index": "dax"}
                for tick, land in zip(bruts, terrains) if tick in valides]
    except Exception as e:
        luecken.append(f"univers DAX illisible ({e})")
        return []


def _eurostoxx_zeilen(luecken: list[str], diagnostics: dict[str, int]) -> list[dict]:
    """Le pays vient de "Registered office"/"Country" (domiciliation), pas de la place de
    cotation; voir la correction documentee dans la docstring du module. Toute ligne au
    pays non reconnu est comptee dans diagnostics["eurostoxx_pays_non_reconnu"]."""
    try:
        tables = _lire_tabellen(URL_EUROSTOXX)
        t = [x for x in tables if "Ticker" in x.columns
             and ("Registered office" in x.columns or "Country" in x.columns)][0]
        land_col = "Registered office" if "Registered office" in t.columns else "Country"
        bruts, terrains, nicht_zugeordnet = [], [], 0
        for _, r in t.iterrows():
            zelle = str(r[land_col])
            land = next((v for k, v in ENGLISH_LAND.items() if k in zelle), None)
            if land is None:
                nicht_zugeordnet += 1
                continue
            bruts.append(str(r["Ticker"]).strip())
            terrains.append(land)
        if nicht_zugeordnet:
            luecken.append(f"univers EURO STOXX 50: {nicht_zugeordnet} lignes au pays non "
                           f"reconnu (colonne {land_col}), ignorees")
            diagnostics["eurostoxx_pays_non_reconnu"] = nicht_zugeordnet
        valides = set(_tickers_valides(bruts, luecken, diagnostics, "univers EURO STOXX 50",
                                       "eurostoxx_rejetes"))
        return [{"ticker": tick, "land": land, "index": "eurostoxx"}
                for tick, land in zip(bruts, terrains) if tick in valides]
    except Exception as e:
        luecken.append(f"univers EURO STOXX 50 illisible ({e})")
        return []


def kennzahlen(dividenden: pd.Series, kurse: pd.Series) -> dict:
    jahres = dividenden.groupby(dividenden.index.year).sum()
    jahres = jahres[jahres.index < jahres.index.max()] if len(jahres) > 11 else jahres
    letzte = jahres.tail(11)
    cagr = (letzte.iloc[-1] / letzte.iloc[0]) ** (1 / 10) - 1 if len(letzte) == 11 and letzte.iloc[0] > 0 else None
    kuerz = int(((letzte.pct_change() < -0.05)).sum())
    ttm = dividenden[dividenden.index > dividenden.index.max() - pd.Timedelta(days=365)].sum()
    kurs = float(kurse.dropna().iloc[-1])
    k10 = kurse.dropna()
    kcagr = (k10.iloc[-1] / k10.iloc[0]) ** (365.25 / max((k10.index[-1] - k10.index[0]).days, 1)) - 1
    dd = float((k10 / k10.cummax() - 1).min())
    return {"rendite_ttm": ttm / kurs if kurs else None, "div_cagr_10j": cagr,
            "kuerzungen_10j": kuerz, "kurs_cagr_10j": kcagr, "max_drawdown": dd}


def auswahl(df: pd.DataFrame, n: int = 30, sektor_max: int = 5,
           cagr_min: float = 0.03, payout_max: float = 0.80) -> pd.DataFrame:
    """cagr_min/payout_max exposent les deux seuils que _waehle_mit_relaxation relache un
    seul a la fois, dans l'ordre impose par la tache."""
    f = df[(df["rendite_ttm"].between(0.02, 0.08)) & (df["kuerzungen_10j"] == 0)
           & (df["div_cagr_10j"] >= cagr_min) & (df["payout"] <= payout_max) & (df["fcf_deckung"] >= 1.0)].copy()
    f["score"] = (f["rendite_netto_de"].rank(pct=True) + f["div_cagr_10j"].rank(pct=True)) / 2
    f = f.sort_values("score", ascending=False)
    aus, zaehler = [], {}
    for _, r in f.iterrows():
        if zaehler.get(r["sektor"], 0) >= sektor_max:
            continue
        aus.append(r)
        zaehler[r["sektor"]] = zaehler.get(r["sektor"], 0) + 1
        if len(aus) == n:
            break
    return pd.DataFrame(aus).reset_index(drop=True)


def _waehle_mit_relaxation(df: pd.DataFrame, n: int = 30,
                          sektor_max: int = 5) -> tuple[pd.DataFrame, Optional[str], list]:
    """Selection avec, si la regle stricte ne fournit pas n titres, un seul relachement a la
    fois dans l'ordre impose par la tache: 1) croissance >= 2 % (payout reste <= 80 %),
    2) payout <= 90 % seul (croissance revient a >= 3 %, le defaut de auswahl()). S'arrete au
    premier palier qui atteint n; sinon garde le palier qui a produit le plus de titres.

    Retourne (selection, nom_du_palier_retenu_ou_None_si_strict, essais) ou essais est la
    liste [(nom_palier, effectif), ...] de tout ce qui a ete tente, pour journalisation.
    """
    paliers = [("strict", {}), ("croissance >= 2 %", {"cagr_min": 0.02}),
               ("payout <= 90 %", {"payout_max": 0.90})]
    essais: list = []
    meilleur_nom, meilleur = None, None
    for nom, kw in paliers:
        essai = auswahl(df, n=n, sektor_max=sektor_max, **kw)
        essais.append((nom, len(essai)))
        if meilleur is None or len(essai) > len(meilleur):
            meilleur_nom, meilleur = nom, essai
        if len(meilleur) >= n:
            break
    relache = None if meilleur_nom == "strict" else meilleur_nom
    return meilleur, relache, essais


def _portefeuille_csv(port: pd.DataFrame) -> pd.DataFrame:
    """Applique le contrat de colonnes (COLONNES_PORTEFEUILLE) et arrondit le score a 4
    decimales pour le CSV publie; ne laisse jamais fuiter une colonne de travail (index,
    kurs_cagr_10j, max_drawdown)."""
    if not len(port):
        return pd.DataFrame(columns=COLONNES_PORTEFEUILLE)
    out = port[COLONNES_PORTEFEUILLE].copy()
    out["score"] = out["score"].round(4)
    return out


def _marktdaten_wide(lot, tickers: list[str], feld: str) -> pd.DataFrame:
    """Colonnes = tickers, index = date, a partir du DataFrame renvoye par L'UNIQUE
    yf.download deja fait dans main() (jamais un second appel reseau: la tache 9 lit ces
    CSV pour ses figures au lieu de retelecharger)."""
    series = {}
    for t in tickers:
        try:
            series[t] = lot[t][feld]
        except Exception:
            continue
    out = pd.DataFrame(series).sort_index()
    out.index.name = "datum"
    return out


def universum() -> tuple[pd.DataFrame, dict[str, int]]:
    """Tickers Yahoo avec pays; ecrit data/universum.csv. Echec d'une table, ou ligne au
    pays/ticker non reconnu, -> LUECKEN.md (jamais silencieux); retourne aussi les
    compteurs structures (diagnostics) pour portefeuille_meta.json."""
    luecken: list[str] = []
    diagnostics: dict[str, int] = {}
    zeilen = (_aristokraten_zeilen(luecken, diagnostics) + _dax_zeilen(luecken, diagnostics)
              + _eurostoxx_zeilen(luecken, diagnostics))
    df = pd.DataFrame(zeilen).drop_duplicates("ticker") if zeilen else pd.DataFrame(
        columns=["ticker", "land", "index"])
    df.to_csv(os.path.join(ROOT, "data", "universum.csv"), index=False)
    if luecken:
        with open(os.path.join(ROOT, "LUECKEN.md"), "a") as fo:
            fo.write("".join(f"- portefeuille: {l}\n" for l in luecken))
    return df, diagnostics


def main() -> int:
    import yfinance as yf
    import fiscalite as fisc
    import hypotheses as h
    uni, univers_diag = universum()
    tickers = list(uni["ticker"])
    lot = yf.download(tickers, period="15y", interval="1mo", actions=True,
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    # Persistance des donnees du telechargement groupe unique pour la tache 9 (figures de
    # risque/rendement et d'historique de dividendes): jamais un second appel yfinance.
    kurse_wide = _marktdaten_wide(lot, tickers, "Close")
    div_wide = _marktdaten_wide(lot, tickers, "Dividends")
    kurse_wide.to_csv(os.path.join(ROOT, "data", "marktdaten_kurse.csv"))
    div_wide.to_csv(os.path.join(ROOT, "data", "marktdaten_dividenden.csv"))
    sq = h.wert("quellensteuer")
    zeilen = []
    manques_analyse = 0
    replis_pays: dict[str, int] = {}
    for _, u in uni.iterrows():
        try:
            d = lot[u["ticker"]]
            div = d["Dividends"][d["Dividends"] > 0]
            if len(div) < 8:
                manques_analyse += 1
                continue
            k = kennzahlen(div, d["Close"])
            zeilen.append({**u.to_dict(), **k})
        except Exception:
            manques_analyse += 1
            continue
    mesures_brutes_n = len(zeilen)   # historique de dividendes suffisant, avant filtre 2-8 %/0 baisse + .info
    df = pd.DataFrame(zeilen)
    pays_pour_impot = []
    for l in df["land"]:
        if l in sq:
            pays_pour_impot.append(l)
        else:
            # Repli declare (jamais silencieux): pays absent de la table BZSt/quellen.json,
            # impose comme "US" par prudence (retenue la plus courante des cas connus).
            replis_pays[l] = replis_pays.get(l, 0) + 1
            pays_pour_impot.append("US")
    if replis_pays:
        with open(os.path.join(ROOT, "LUECKEN.md"), "a") as fo:
            fo.write("- portefeuille: pays sans entree quellensteuer, replies sur US: "
                    + ", ".join(f"{p} ({n})" for p, n in sorted(replis_pays.items())) + "\n")
    df["rendite_netto_de"] = [fisc.posten_netto(r * 100, l, 0.0, sq=sq)[0] / 100
                              if r else None for r, l in zip(df["rendite_ttm"], pays_pour_impot)]
    vor = df[(df["rendite_ttm"].between(0.02, 0.08)) & (df["kuerzungen_10j"] == 0)]
    vor = vor.sort_values("div_cagr_10j", ascending=False).head(60)
    infos = []
    manques_info = 0
    for t in vor["ticker"]:                      # au plus 60 appels .info, declares
        try:
            i = yf.Ticker(t).info
            fcf, div_bezahlt = i.get("freeCashflow"), (i.get("dividendRate") or 0) * (i.get("sharesOutstanding") or 0)
            infos.append({"ticker": t, "name": i.get("shortName"), "sektor": i.get("sector") or "?",
                          "payout": i.get("payoutRatio") if i.get("payoutRatio") is not None else 1.0,
                          "fcf_deckung": (fcf / div_bezahlt) if fcf and div_bezahlt else 0.0})
        except Exception:
            manques_info += 1
            continue
    df = vor.merge(pd.DataFrame(infos), on="ticker", how="inner")
    port, relache, essais = _waehle_mit_relaxation(df, n=30)
    if len(port) < 30:
        resume = "; ".join(f"{nom}: {cnt}" for nom, cnt in essais)
        with open(os.path.join(ROOT, "LUECKEN.md"), "a") as fo:
            fo.write(f"- portefeuille: {len(port)}/30 titres retenus (univers {len(uni)} tickers"
                    f" sur 3 indices Wikipedia, {mesures_brutes_n} avec >= 8 dividendes annuels sur"
                    f" 15 ans, {len(df)} apres filtre rendement 2-8 %/0 baisse et fusion .info reussie);"
                    f" paliers essayes ({resume}); retenu {relache or 'strict'} avec"
                    f" {len(port)} titres. Le chapitre 6 presente les {len(port)} titres reels"
                    f" obtenus, la methode restant illustree meme sous 30 lignes.\n")
    port_csv = _portefeuille_csv(port)
    port_csv.to_csv(os.path.join(ROOT, "data", "portefeuille.csv"), index=False)
    mix = port["land"].value_counts(normalize=True).round(4).to_dict() if len(port) else {}
    abgerufen = pd.Timestamp.today().strftime("%Y-%m-%d")
    meta = {"laender_mix": mix, "abgerufen": abgerufen,
            "univers_n": int(len(uni)), "mesures_brutes_n": int(mesures_brutes_n),
            "mesures_n": int(len(df)), "choisis_n": int(len(port)),
            "manques_analyse": int(manques_analyse), "manques_info": int(manques_info),
            "quellensteuer_repli": replis_pays, "paliers_essayes": essais,
            "marktdaten": {"kurse": "data/marktdaten_kurse.csv",
                          "dividendes": "data/marktdaten_dividenden.csv", "abgerufen": abgerufen}}
    meta.update(univers_diag)
    if relache:
        meta["relache"] = relache
    json.dump(meta, open(os.path.join(ROOT, "data", "portefeuille_meta.json"), "w"), indent=1)
    print(f"[PORTEFEUILLE] univers {len(uni)}, mesures {len(df)}, choisis {len(port)}, "
          f"manques {manques_analyse}+{manques_info}; mix {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
