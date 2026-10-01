#!/usr/bin/env python3
r"""
projection.py - accumulation mensuelle puis revenu, en EUROS DE 2026 (reel).

Conventions (declarees au chapitre 5):
  * Tout est reel: rendements hors inflation, objectif fixe en euros de 2026. Les seuils
    fiscaux en euros n'ont pas tous le meme statut face a l'inflation:
      - Sparerpauschbetrag: fixe en NOMINAL par la loi (§20 Abs. 9 EStG), ne change que
        par une reforme legislative expresse; on le deflate donc chaque annee dans ce
        modele reel (effet de progression a froid, _pauschbetrag_default -> pausch dans
        simulieren). Aucun parametre fiscal n'est ecrit en dur ici: le forfait vient de
        hypotheses.wert("sparerpauschbetrag"), jamais d'un litteral dans ce module.
      - Plancher (Mindestbemessung) et plafond (BBG) de la cotisation maladie/dependance
        volontaire: REVALORISES CHAQUE ANNEE avec les salaires en droit (§6 Abs. 6-7 et
        §223 Abs. 4 SGB V pour la BBG; §18 SGB IV, Bezugsgroesse, pour la Mindestbemessung;
        sources completes dans data/quellen.json, cle "kv_schwellen_dynamisierung"). Comme
        les salaires croissent au moins autant que les prix et que la croissance salariale
        reelle est deja modelisee separement (salaire.py), la convention retenue ici est de
        les tenir CONSTANTS en euros de 2026 (reel): _saetze_real ne les deflate PAS. Les
        confondre avec le Sparerpauschbetrag (memes deflater par erreur) fait chuter le
        plafond de cotisation simule d'environ 1226 a 513 EUR/mois en 2071 et flatte a tort
        toute strategie a dividendes (corrige revue de code, tache 10, 2026-09-30).
  * Poche maison: dividende rendite_div_maison verse mensuellement, impose en fin
    d'annee selon LAENDER_MIX, net reinvesti. Poche ETF: capitalisante (Vorabpauschale)
    ou distribuante (Teilfreistellung).
  * Convention "dividende" pour la poche ETF: on suppose la poche ETF detenue (ou
    basculee au depart en retraite) en part distribuante, donc elle rapporte
    rendite_div_etf que la strategie soit capitalisante ou distribuante en phase
    d'accumulation; le cout fiscal du basculement de part n'est pas modelise
    (hypothese declaree comme \annahme au chapitre 5).
  * Vorabpauschale (S18 InvStG): la base compare la valeur de fin d'annee a la valeur de
    debut d'annee (Zuwachs). Les versements de l'annee ne sont pas un gain: ils sont
    retires de la valeur de fin d'annee avant l'appel, sinon un versement pur gonflerait
    artificiellement la base taxable.
  * ETF distribuant en phase d'accumulation: les distributions sont reinvesties (achat
    de parts supplementaires), donc elles s'ajoutent a la base de cout (basis_etf), au
    meme titre que les versements.
  * basis_etf (cout d'acquisition) est nominal en droit, alors que le reste du moteur
    est tenu en euros de 2026 (reel): un cout fixe en nominal perd du pouvoir d'achat
    reel chaque annee. En debut de chaque annee suivant l'annee 0 (annee d'initialisation,
    deja au niveau de prix de l'annee 0), basis_etf est deflate d'une annee d'inflation
    avant que les versements et reinvestissements de l'annee (deja exprimes au niveau de
    prix de cette annee) ne s'y ajoutent. Effet: a croissance reelle nulle, un retrait en
    mode "entnahme4" porte quand meme une part imposable non nulle (plus-value purement
    inflationniste, non indexee).
  * Base GKV des revenus de fonds (ETF): la Teilfreistellung InvStG (30 %) reduit
    l'assiette de cotisation GKV pour les Investmenterträge (source sourcee dans
    data/quellen.json sous "kv_teilfreistellung_etf": GKV-Spitzenverband, Katalog des
    revenus selon §240 SGB V, ligne "Investmenterträge ... ja, unter Beruecksichtigung
    der §§ 20 und 56 Abs. 6 InvStG"), mais pas les dividendes d'actions individuelles
    (poche maison, hors perimetre InvStG). Meme convention appliquee dans les deux
    branches "dividende" et "entnahme4"; jamais ecrite en dur, lue depuis hypotheses.py.
  * Impot retire une seule fois par annee: une seule variable `steuer` par annee, retiree
    d'une seule poche (celle qui porte l'essentiel du portefeuille); la comptabilite est
    volontairement simple (pas d'imputation poche par poche de l'impot).
  * Annee cible: premiere fin d'annee ou le revenu net mensuel soutenable atteint
    l'objectif. Revenu "dividende" = dividendes de l'annee suivante, nets d'impot et de
    KV/PV; revenu "entnahme4" = 4 % du portefeuille, impot sur la part de plus-value. Le
    ratio de plus-value (gewinnanteil) est applique au retrait total (maison + ETF); ce
    mode n'a de sens que pour la strategie de reference 100 % ETF (anteil_maison = 0),
    seul cas ou il est utilise dans ce projet.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fiscalite as f   # noqa: E402

LAENDER_MIX = {"US": 0.5, "DE": 0.2, "FR": 0.1, "CH": 0.1, "GB": 0.1}


@dataclass
class Markt:
    rendite_div_maison: float
    wachstum_maison: float
    rendite_div_etf: float
    wachstum_etf: float
    inflation: float
    basiszins: float


@dataclass
class Strategie:
    name: str
    anteil_maison: float
    etf_ausschuettend: bool
    einkommen: str


def endwert_ohne_steuer(monatlich: float, jahre: int, rendite: float, start: float) -> float:
    r = (1 + rendite) ** (1 / 12) - 1
    w = start
    for _ in range(jahre * 12):
        w = w * (1 + r) + monatlich
    return w


def sparplan_aus_quote(quote: float, minimum: float = 200.0, jahre: int = 45,
                       netto_monat: Optional[list[float]] = None) -> list[float]:
    if netto_monat is None:
        import salaire
        netto_monat = [t["netto_monat"] for t in salaire.trajektorie(2027, jahre)]
    return [max(minimum, quote * n) * 12 for n in netto_monat[:jahre]]


def _pauschbetrag_default() -> float:
    """Sparerpauschbetrag nominal, jamais ecrit en dur: seule source, hypotheses.py."""
    import hypotheses
    return hypotheses.wert("sparerpauschbetrag")


def _kv_teilfreistellung_etf_default() -> bool:
    """Teilfreistellung InvStG comptee ou non dans l'assiette GKV des revenus de fonds:
    jamais ecrit en dur, source hypotheses.wert('kv_teilfreistellung_etf')."""
    import hypotheses
    return bool(hypotheses.wert("kv_teilfreistellung_etf"))


def _saetze_real(saetze: dict, deflator: float) -> dict:
    """Plancher (Mindestbemessung) et plafond (BBG) de la cotisation KV/PV volontaire, en
    euros de 2026 (reel). CONSTANTS d'une annee a l'autre (deflator non applique a ces deux
    cles): en droit ils sont revalorises chaque annee avec les salaires, pas fixes en
    nominal (source: data/quellen.json, cle "kv_schwellen_dynamisierung"; voir aussi la
    convention declaree dans le docstring du module). `deflator` est conserve dans la
    signature pour compatibilite d'appel (simulieren le calcule deja pour le
    Sparerpauschbetrag, qui lui reste nominal et s'erode reellement, cf. `pausch` dans
    simulieren) mais n'intervient plus dans ce calcul."""
    return dict(saetze)


def _einkommen(strat: Strategie, markt: Markt, maison: float, etf: float, basis_etf: float,
               pausch: float, saetze: dict, sq: Optional[dict],
               kv_teilfreistellung_etf: bool) -> float:
    """Revenu net annuel soutenable a partir de l'annee suivante (reel).

    Convention "dividende": la poche ETF est supposee en part distribuante (voir docstring
    du module), donc elle rapporte rendite_div_etf sans branchement sur etf_ausschuettend
    (les deux branches du ternaire d'origine etaient identiques; supprime).

    kv_teilfreistellung_etf: si vrai, la Teilfreistellung InvStG (30 %) reduit la part ETF
    de l'assiette GKV (source: data/quellen.json, cle "kv_teilfreistellung_etf"), jamais la
    part maison (dividendes d'actions individuelles, hors perimetre InvStG). Meme
    convention dans les deux branches ci-dessous.
    """
    if strat.einkommen == "dividende":
        posten = [(maison * markt.rendite_div_maison * a, k) for k, a in LAENDER_MIX.items()]
        div_etf = etf * markt.rendite_div_etf
        posten.append((div_etf, "ETF"))
        maison_brutto = sum(b for b, k in posten if k != "ETF")
        div_etf_kv = div_etf * (1 - f.TEILFREI) if kv_teilfreistellung_etf else div_etf
        return f.jahres_netto(posten, pausch, sq=sq) - f.kv_beitrag_jahr(maison_brutto + div_etf_kv, saetze)
    entnahme = 0.04 * (maison + etf)
    gewinnanteil = max(0.0, 1 - basis_etf / etf) if etf > 0 else 0.0
    steuerpfl_brutto = entnahme * gewinnanteil
    netto = entnahme - steuerpfl_brutto + f.posten_netto(steuerpfl_brutto, "ETF", pausch, sq=sq)[0]
    steuerpfl_kv = steuerpfl_brutto * (1 - f.TEILFREI) if kv_teilfreistellung_etf else steuerpfl_brutto
    return netto - f.kv_beitrag_jahr(steuerpfl_kv, saetze)


def simulieren(strategie: Strategie, markt: Markt, sparplan: list[float], start_kapital: float,
               ziel_netto_monat_real: float, jahre: int = 45, sq: Optional[dict] = None,
               saetze: Optional[dict] = None, pauschbetrag_nominal: Optional[float] = None,
               kv_teilfreistellung_etf: Optional[bool] = None) -> dict:
    if sq is None:
        sq = f._sq_standard()
    if saetze is None:
        saetze = f.saetze_2026()                     # jamais de taux ecrit ici: quellen.json
    if pauschbetrag_nominal is None:
        pauschbetrag_nominal = _pauschbetrag_default()
    if kv_teilfreistellung_etf is None:
        kv_teilfreistellung_etf = _kv_teilfreistellung_etf_default()
    maison = start_kapital * strategie.anteil_maison
    etf = start_kapital * (1 - strategie.anteil_maison)
    basis_etf = etf
    gm = (1 + markt.wachstum_maison) ** (1 / 12) - 1
    ge = (1 + markt.wachstum_etf) ** (1 / 12) - 1
    out: dict = {"jahre": [], "wert": [], "einkommen_netto_monat_real": [], "eingezahlt": [],
                 "steuern": [], "ziel_jahr": None}
    eingezahlt = start_kapital
    for i in range(jahre):
        if i > 0:
            # basis_etf est un cout d'acquisition nominal; en euros de l'annee courante,
            # il perd du pouvoir d'achat chaque annee (annee 0 deja au niveau de prix de
            # l'annee 0, donc pas deflatee ici). Les versements/reinvestissements de
            # l'annee, ajoutes plus bas, sont deja au niveau de prix de cette annee.
            basis_etf /= 1 + markt.inflation
        deflator = (1 + markt.inflation) ** i
        pausch = pauschbetrag_nominal / deflator
        s_real = _saetze_real(saetze, deflator)
        monat = sparplan[i] / 12 if i < len(sparplan) else 0.0
        etf_anfang, div_maison, div_etf = etf, 0.0, 0.0
        for _ in range(12):
            maison *= 1 + gm
            etf *= 1 + ge
            div_maison += maison * markt.rendite_div_maison / 12
            div_etf += etf * markt.rendite_div_etf / 12
            maison += monat * strategie.anteil_maison
            etf += monat * (1 - strategie.anteil_maison)
            basis_etf += monat * (1 - strategie.anteil_maison)
        eingezahlt += monat * 12
        etf_beitrag_jahr = monat * 12 * (1 - strategie.anteil_maison)
        posten = [(div_maison * a, k) for k, a in LAENDER_MIX.items()]
        if strategie.etf_ausschuettend:
            posten.append((div_etf, "ETF"))
            netto = f.jahres_netto(posten, pausch, sq=sq)
            steuer = sum(b for b, _ in posten) - netto
            maison += div_maison                       # brut reinvesti, impot retire une fois plus bas
            etf += div_etf
            basis_etf += div_etf                        # distribution reinvestie = nouvelles parts
        else:
            etf += div_etf                              # thesauriert im Fonds
            # Zuwachs de la Vorabpauschale: hors versements de l'annee (pas un gain).
            vp = f.vorabpauschale(etf_anfang, etf - etf_beitrag_jahr, 0.0, markt.basiszins)
            posten.append((vp, "ETF"))
            netto = f.jahres_netto(posten, pausch, sq=sq)
            steuer = sum(b for b, _ in posten) - netto
            maison += div_maison
            basis_etf += vp
        # L'impot de l'annee est retire UNE fois, sur la poche qui le paie.
        if strategie.anteil_maison > 0:
            maison -= steuer
        else:
            etf -= steuer
        eink = _einkommen(strategie, markt, maison, etf, basis_etf, pausch, s_real, sq,
                         kv_teilfreistellung_etf) / 12
        out["jahre"].append(2027 + i)
        out["wert"].append(maison + etf)
        out["einkommen_netto_monat_real"].append(eink)
        out["eingezahlt"].append(eingezahlt)
        out["steuern"].append(steuer)
        if out["ziel_jahr"] is None and eink >= ziel_netto_monat_real:
            out["ziel_jahr"] = 2027 + i
    return out
