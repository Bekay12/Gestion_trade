#!/usr/bin/env python3
"""
rechnung_retraite.py - calcule tout ce que le document affiche et l'ecrit dans data/.
Un bloc par chapitre; aucun chiffre n'est ecrit ailleurs. Macros: noms sans chiffre
(annees et nombres en toutes lettres ou en romain: 2044 -> XLIV, 70 % -> Siebzig).

Partie A: infrastructure, markt_basis, kapitel_2_und_3, kapitel_4, kapitel_5, plus le
tableau de presse (data/tab_presse.tex, decision 4 du controleur).

Partie B (cette version): kapitel_6 (portefeuille-exemple), kapitel_7 (ce qui peut mal
tourner: Shiller, Monte Carlo, inflation), kapitel_8 (la retraite elle-meme: pont,
rente legale, reserve), kapitel_10 (frise du plan d'action), plus data/tab_portefeuille.tex.
Decisions notables (detaillees dans les docstrings des blocs concernes et dans
task-9-report.md, section "Partie B"):
  - fig_aufteilungen: "kapital_xlv"/"netto_monat_xlv" = capital et revenu net mensuel a
    la derniere annee du plan (annee 45 de l'horizon, XLV en romain), quote 20 %,
    capital de depart "Mitte", pour chacune des 5 strategies (le brief ne fixe pas
    l'annee de comparaison; l'horizon complet est le choix le plus neutre entre
    strategies qui n'atteignent pas toutes l'objectif au meme age). Quote laissee a 20 %
    apres le correctif du scenario de reference (tache 10, ci-dessous): cette figure
    compare 5 STRATEGIES a un taux d'epargne COMMUN fixe, ce n'est pas le scenario de
    reference (qui concerne une seule strategie, MaisonSiebzig) et elle ne nourrit aucune
    macro *Basis/*Yann; changer sa quote serait une decision independante, hors mandat.
  - fig_anzahl_titel: rendements MENSUELS DE PRIX (hors dividendes) des 23 titres
    retenus (data/marktdaten_kurse.csv, aucun nouvel appel yfinance), ecart-type annualise
    (x sqrt(12)) d'un portefeuille equipondere de n titres tires sans remise, moyenne sur
    500 tirages par n, graine fixe SEED_ANZAHL_TITEL (distincte de celle de montecarlo.py).
  - fig_div_beispiele: dividendes annuels reels; deflateur construit a partir de
    hypotheses.wert("inflation_destatis_2019_2025") (2019-2025) et, hors de cette
    fenetre, hypotheses.wert("inflation_ziel") en repli (aucune serie CPI mensuelle
    n'existe pour 2011-2026 dans ce depot), base 2026 (convention "euros de 2026" du
    reste du document). "Titre en hausse continue": la plus longue serie d'annees
    consecutives de dividende strictement croissant PARMI LES 23 titres retenus.
    "Titre en forte baisse": la plus forte baisse relative d'une annee sur l'autre
    PARMI TOUT L'UNIVERS (data/universum.csv, 142 tickers); aucune valeur recopiee de
    la litterature, tout est mesure sur data/marktdaten_dividenden.csv.
  - fig_mc_faecher/fig_mc_erfolg/fig_puffer: "jahre" (longueur des trajectoires
    Monte Carlo) est recherche automatiquement (_mc_jahre_sans_troncature) jusqu'a
    abgeschnitten == 0 pour le cas le plus exigeant utilise dans le bloc (rentenjahre
    maximal); le resultat est verifie par assert et non suppose. UN SEUL calcul Monte
    Carlo (pfade_base, scenario de reference REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM
    (70/30 a 30 %, tache 10 fix "objectif jamais atteint a 20 %", voir plus bas),
    rentenjahre=40) alimente a la fois \\ErfolgsquoteBasis, \\ErfolgsquoteMitPuffer
    (kapitel_7) et fig_puffer.csv (kapitel_8, via le cache module-level _MC_PUFFER): spec
    section 8, "meme macro, jamais deux calculs" (revue de code, tache 9 fix round 1 - un
    premier decoupage en deux calculs independants donnait deux estimations legerement
    differentes du meme scenario, 88,7 % contre 90,0 %, par bruit d'echantillonnage Monte
    Carlo).
  - fig_mc_erfolg: "div_schock" (colonne renommee, tache 9 fix round 1 - "rendite_start"
    ne decrivait plus ce que la colonne mesure) est un choc reel ADDITIF applique a la
    croissance du dividende de la PREMIERE ANNEE DE RETRAITE de chaque trajectoire
    (grille -40 % a +40 % par pas de 10 points, montecarlo.erfolg(...,
    div_schock_erstes_rentenjahr=...)), le reste de la trajectoire restant tire de
    l'historique (meme graine): stress-test de sequence des rendements a l'entree en
    retraite, pas une donnee historique brute ni un rendement de depart d'accumulation.
    Macro texte \\HypotheseDivSchock declarant ce sens pour la tache 10. Une premiere
    version chocait l'annee 0 CALENDAIRE du rendement de capitalisation et produisait un
    taux de reussite rigoureusement identique sur toute la grille (implausible, verifie
    et corrige plutot que publie: le capital demarre a 0, donc ce choc multipliait 0 par
    n'importe quoi, et la reussite post-retraite ne depend de toute facon jamais du
    rendement de capitalisation). Le stress-test vit desormais dans
    montecarlo.erfolg(div_schock_erstes_rentenjahr=...) (revue de code, tache 9 fix
    round 1: une premiere version dupliquait la boucle d'accumulation/reussite de
    montecarlo.erfolg() localement dans ce module; supprimee, montecarlo.py porte
    maintenant l'unique implementation, testee dans test_montecarlo.py).
  - fig_bruecke/fig_rente_alter: la rente legale est estimee par points
    (salaire_brut_annuel / Durchschnittsentgelt, somme sur les annees cotisees de 26 ans
    a l'age de sortie) x rentenwert_2026; "annees cotisees" = carriere continue depuis
    START_JAHR (26 ans), simplification declaree (pas de trous de carriere modelises).
  - EtfUmschichtung (decision 6 du controleur): modelisee comme un ETF distribuant DES
    LE DEBUT de l'accumulation (etf_ausschuettend=True dans STRATEGIEN), alors que la
    specification (section 3) decrit une bascule vers des ETF distribuants trois a cinq
    ans avant le depart. Modelisation simple du brief conservee, declaree en macro texte
    \\HypotheseUmschichtung (markt_basis(), meme motif que \\HypotheseUsFormulaire).
  - Poche ETF des strategies "Maison*" (revue finale, constat 5): capitalisante
    (etf_ausschuettend=False) pendant l'epargne, mais traitee comme distribuante
    (rendite_div_etf) des la retraite par projection._einkommen() sans modeliser le
    cout fiscal du changement de categorie de parts, meme angle mort que
    EtfUmschichtung mais sur la strategie de reference elle-meme. Declaree en macro
    texte \\HypotheseBasculeReference (markt_basis()).

Corrections du controleur (revue post-partie-A, apres le commit 89a4e659), ECART AU PLAN
DECIDE PAR LE CONTROLEUR documente ici:

1. rendite_real: le plan (et la premiere version de ce module) demandait la MEDIANE des
   rendements reels annuels Shiller 1950-2022. Remplace par le TCAC (taux de croissance
   annuel compose, moyenne geometrique: prod(1+r)^(1/n) - 1), car la mediane (12,1 %) et
   la moyenne arithmetique (8,6 %) des rendements ANNUELS ne sont pas des taux de
   capitalisation valides sur plusieurs decennies (elles ignorent l'effet multiplicatif
   des annees de krach); le TCAC vaut 7,3 %, la valeur correcte a composer sur 45 ans.
   La mediane reste exposee en macro a titre informatif (phrase de sensibilite), avec une
   macro indiquant explicitement la convention retenue.
2. Retenue americaine: fiscalite.py et projection.py ne sont PAS modifies (deja testes,
   hors perimetre). A la place, ce module construit sa propre copie de la table
   hypotheses.wert("quellensteuer") avec l'entree US substituee au taux W-8BEN (15 %,
   suppose deja depose par le courtier, pratique quasi automatique chez les courtiers
   europeens), jamais pour la Suisse ni la France (retenue standard conservee par
   defaut, le remboursement restant une demarche active separee, modelisee a part dans
   fig_steuer_100). Cette table de remplacement (_sq_de_base) est ce que _sim_kwargs()
   et tous les calculs manuels de ce module passent partout en argument sq=.
3. Scenario de reference releve a 30 % d'epargne (tache 10, correctif du 30.09.2026,
   apres task-10-kv-fix-report.md): le correctif "seuils KV constants en euros de 2026"
   (projection._saetze_real) fait que le scenario de reference historique (70/30,
   20 % d'epargne, capital de depart "Mitte") n'atteint plus jamais l'objectif dans
   JAHRE_HORIZONT ans. Decision du controleur: LE scenario de reference (celui qui
   alimente tous les macros *Basis/*Yann, le Monte Carlo de base, fig_bruecke,
   fig_zeitplan, fig_start_verzoegerung, \\BrueckeJahre, \\RenteBeiZielalter et
   \\ZielJahrBasis) passe a 30 % d'epargne; la strategie (MaisonSiebzig, 70/30) et le
   capital de depart ("Mitte") ne changent pas. Cette definition existe maintenant a UN
   SEUL endroit (REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM/REFERENCE_STARTKAPITAL_NOM
   ci-dessous, macros texte \\ReferenzStrategie/\\ReferenzSparquote pour la redaction).
   L'ancien resultat a 20 % reste publie tel quel sous ses propres macros, deja produites
   par la boucle ZUS["alter"] de kapitel_5() sur les 4 quotes de QUOTEN (\\Alter<Strategie>
   Zwanzig, \\EcartAgeEtfMaisonVingt, etc.): rien n'est supprime, seule LA REFERENCE change.
   Voir task-10-kv-fix-report.md, section "Scenario de reference 30 %", pour les valeurs.
"""
import inspect
import json
import os
import re
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fiscalite as f        # noqa: E402
import histoire              # noqa: E402
import hypotheses as h       # noqa: E402
import montecarlo as mc      # noqa: E402
import projection as p       # noqa: E402
import salaire               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ZIEL_REAL_MONAT = 3500.0
ZIEL_REAL_JAHR = ZIEL_REAL_MONAT * 12
# Variante basse (demande explicite de l'utilisateur, tache 11, 01.10.2026): epargner 50 %
# du salaire revient a vivre avec l'autre moitie, donc une cible plus basse que 3500 EUR est
# pertinente. Meme horizon/strategies/quotes/capital de depart que la cible principale;
# aucune macro *Basis/*Yann n'en depend, LA reference du document reste a ZIEL_REAL_MONAT.
ZIEL_NIEDRIG_MONAT = 2500.0
ZIEL_NIEDRIG_JAHR = ZIEL_NIEDRIG_MONAT * 12
GEBURTSJAHR = 2001
START_JAHR = 2027
JAHRE_HORIZONT = 45
QUOTEN = {"Zehn": 0.10, "Zwanzig": 0.20, "Dreissig": 0.30, "Fuenfzig": 0.50}
STARTKAPITAL = {"Niedrig": 2000.0, "Mitte": 6000.0, "Hoch": 10000.0}
# Bornes d'inflation de fig_ziel_nominal (la valeur centrale est hypotheses.wert("inflation_ziel")).
INFLATION_BAS, INFLATION_HAUT = 0.01, 0.03
# Impot d'Eglise: variante chiffree a 8 % et 9 % de l'impot (specification, section 2), jamais
# retenue par defaut.
KIST_VARIANTEN = {"Acht": 0.08, "Neun": 0.09}
# Chapitre 4 (tache 10, chapitres 04-05): taux de retrait de la reference ETF (regle des
# 4 %, meme taux que la branche "entnahme4" de projection.py) et variante prudente a 3 %
# (borne haute de la fourchette "ein bis drei Prozent" citee par Finanztip, data/presse.json,
# id finanztip_geldanlage); grille de rendement brut de la poche maison de fig_kapital_rendite.
TAUX_RETRAIT = 0.04
TAUX_RETRAIT_PRUDENT = 0.03
RENDITE_GRILLE = [0.02, 0.03, 0.04, 0.05, 0.06]
# Chapitre 5: controle croise du salaire net (docs/salaire_controle.md). Net annuel publie
# par Deutschland-Rechner, table Brutto-Netto 2026, Steuerklasse I, sans enfant, sans impot
# d'Eglise (https://www.deutschland-rechner.de/brutto-netto-tabelle, consulte le
# 30.09.2026), pour les deux paliers de la table qui encadrent le salaire d'entree; seuil
# d'ecart relatif accepte par la tache 3.
SALAIRE_CONTROLE = {"Bas": (55000.0, 34978.0), "Haut": (60000.0, 37561.0)}
SALAIRE_CONTROLE_SEUIL = 0.03
# Chapitre 5: ages vises par fig_sparbedarf (specification, section 7).
SPARBEDARF_ALTER = {"Quarante": 40, "QuaranteCinq": 45, "Cinquante": 50}
STRATEGIEN = [p.Strategie("MaisonFuenfzig", 0.5, False, "dividende"),
              p.Strategie("MaisonSiebzig", 0.7, False, "dividende"),
              p.Strategie("MaisonHundert", 1.0, False, "dividende"),
              p.Strategie("EtfReferenz", 0.0, False, "entnahme4"),
              p.Strategie("EtfUmschichtung", 0.0, True, "dividende")]
# --- Scenario de reference (decision du controleur, tache 10, correctif du 30.09.2026):
# LA SEULE definition de "strategie/quote d'epargne/capital de depart" pour tous les
# macros *Basis/*Yann, le Monte Carlo de base (\ErfolgsquoteBasis/\ErfolgsquoteMitPuffer/
# fig_mc_faecher/fig_puffer), fig_bruecke, fig_zeitplan, fig_start_verzoegerung et les
# macros \BrueckeJahre/\RenteBeiZielalter/\ZielJahrBasis. Avant ce correctif, ces blocs
# codaient chacun en dur QUOTEN["Zwanzig"] (20 %); a ce taux, projection._saetze_real
# (seuils KV/PV constants en euros de 2026, correctif precedent de la tache 10) fait que
# l'objectif n'est plus jamais atteint dans JAHRE_HORIZONT ans (voir
# task-10-kv-fix-report.md). Relever un seul de ces trois noms suffit desormais a
# deplacer LE scenario de reference partout ou il est utilise; l'ancien resultat a 20 %
# reste disponible sous ses propres macros (ZUS["alter"][...]["Zwanzig"],
# \AlterMaisonSiebzigZwanzig, \EcartAgeEtfMaisonVingt...), non affectees par ce bloc.
REFERENCE_STRATEGIE_NOM = "MaisonSiebzig"
REFERENCE_QUOTE_NOM = "Dreissig"
REFERENCE_STARTKAPITAL_NOM = "Mitte"
M, ZUS = {}, {"alter": {}}
# Cache module-level pour partager UN SEUL calcul Monte Carlo (scenario de reference
# 70/30 a 30 %, rentenjahre=40) entre kapitel_7 (macros \ErfolgsquoteBasis /
# \ErfolgsquoteMitPuffer) et kapitel_8 (fig_puffer.csv): revue de code, tache 9 fix
# round 1 - "meme macro, jamais deux calculs" (spec section 8). Rempli par kapitel_7,
# lu par kapitel_8; kapitel_7 s'execute toujours avant kapitel_8 dans main().
_MC_PUFFER: dict = {}
# Meme principe pour le rejeu historique du chapitre 7 (histoire.rueckspiel, scenario de
# reference): kapitel_8 compare la regle des 4 % aux MEMES annees de depart en retraite que
# ce rejeu (tache 10, chapitre 08), sans relancer un second rejeu. Rempli par kapitel_7.
_RUECKSPIEL: dict = {}


def schreiben(name, df):
    """
    --------------------------------------------------------------------------
    Purpose:
        Ecrit un CSV de figure (data/<name>.csv) consomme par \\linienfigur/
        \\balkenfigur. Ajoute, pour chaque colonne texte (dtype object), une
        colonne jumelle "<col>_disp" au texte echappe pour LaTeX (correctif
        infrastructure, tache 10): \\balkenfigur pose ses etiquettes de
        graduation avec "xticklabels from table" sur la colonne "<etiquette>
        _disp" plutot que sur la colonne brute (voir preamble.tex), pour que
        des valeurs comme "brut_necessaire" ou "capital_10000" (underscore,
        catcode 8 = indice mathematique hors mode math) compilent sans faire
        planter LaTeX, MEME quand la redaction n'ecrase pas les etiquettes a
        la main avec des libelles francais (ce que les chapitres 2-3 font
        deja via l'argument optionnel de la macro, qui reste prioritaire).

    Inputs:
        name (str): nom du CSV, sans extension ni prefixe data/.
        df (pandas.DataFrame): donnees de la figure.

    Outputs:
        None. Ecrit data/<name>.csv (colonnes d'origine + une "<col>_disp"
        par colonne texte).
    --------------------------------------------------------------------------
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[f"{col}_disp"] = df[col].map(escapieren)
    df.to_csv(os.path.join(DATA, name + ".csv"), index=False)


def fmt(x, stellen=0):
    """
    --------------------------------------------------------------------------
    Purpose:
        Formater un nombre pour data/kennzahlen.tex selon la convention
        francaise du document (decision 2 du controleur, tache 9): separateur
        de milliers "\\," (espace fine LaTeX), virgule decimale "{,}", signe
        moins "$-$". Les CSV de figures restent en point decimal brut pour
        pgfplots (non concernes par cette fonction).

    Inputs:
        x (float): valeur a formater.
        stellen (int): nombre de decimales, defaut 0.

    Outputs:
        result (str): ex. 3500 -> "3\\,500", 4.5 (stellen=1) -> "4{,}5",
            -1234.5 (stellen=1) -> "$-$1\\,234{,}5".
    --------------------------------------------------------------------------
    """
    s = f"{abs(x):,.{stellen}f}"
    entier, _, dec = s.partition(".")
    entier = entier.replace(",", "\\,")
    corps = entier if stellen == 0 else f"{entier}{{,}}{dec}"
    return ("$-$" if round(x, stellen) < 0 else "") + corps


_LATEX_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escapieren(texte) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Echapper tout texte venu de l'exterieur (noms/secteurs yfinance,
        citations de presse) avant ecriture dans un .tex ou une macro
        (decision 3 du controleur, tache 9: regle de securite du depot contre
        une macro LaTeX active telle que \\input ou \\write dans un champ
        scrape). Nettoie aussi les espaces et dechets en tete/fin de chaine
        (certains noms yfinance en portent, ex. "Allianz SE (espaces) v").

    Inputs:
        texte (Any): valeur brute, convertie en str.

    Outputs:
        result (str): texte nettoye puis echappe caractere par caractere
            (le backslash est lui-meme echappe en \\textbackslash{}, ce qui
            neutralise toute commande LaTeX presente dans le texte source).
    --------------------------------------------------------------------------
    """
    return "".join(_LATEX_ESCAPE.get(c, c) for c in str(texte).strip())


_ROEMISCH = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roemisch(n: int) -> str:
    """Chiffre romain d'un entier positif (algorithme standard, glouton)."""
    if n <= 0:
        raise ValueError("roemisch: n doit etre strictement positif")
    aus = []
    for valeur, chiffre in _ROEMISCH:
        while n >= valeur:
            aus.append(chiffre)
            n -= valeur
    return "".join(aus)


def annee_romaine(jahr: int) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Nom de macro sans chiffre pour une annee du document (toutes dans les
        annees 2000, cf. GEBURTSJAHR=2001 et horizon de 45 ans a partir de
        2027): seuls les deux derniers chiffres varient, convertis en romain
        (2044 -> XLIV), comme impose par le brief de la tache 9.

    Inputs:
        jahr (int): annee, ex. 2044.

    Outputs:
        result (str): ex. "XLIV".
    --------------------------------------------------------------------------
    """
    return roemisch(jahr % 100)


_ZAHLWORT = {0: "Null", 1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "Fuenf",
             6: "Sechs", 7: "Sieben", 8: "Acht", 9: "Neun", 10: "Zehn", 11: "Elf",
             12: "Zwoelf", 13: "Dreizehn", 14: "Vierzehn", 15: "Fuenfzehn",
             16: "Sechzehn", 17: "Siebzehn", 18: "Achtzehn", 19: "Neunzehn",
             20: "Zwanzig", 30: "Dreissig", 40: "Vierzig", 50: "Fuenfzig",
             60: "Sechzig", 70: "Siebzig", 80: "Achtzig", 90: "Neunzig", 100: "Hundert"}


def zahlwort(n: int) -> str:
    """Nombre entier 0-100 en toutes lettres (allemand), pour un nom de macro sans
    chiffre (ex. 18 -> Achtzehn); dizaine + unite composees pour le reste (21 -> EinsUndZwanzig)."""
    if n in _ZAHLWORT:
        return _ZAHLWORT[n]
    dizaine, unite = (n // 10) * 10, n % 10
    if 0 < n < 100 and dizaine in _ZAHLWORT and unite in _ZAHLWORT:
        return f"{_ZAHLWORT[unite]}Und{_ZAHLWORT[dizaine]}"
    raise ValueError(f"zahlwort: {n} hors de la plage geree (0-99)")


def _reference_strategie() -> p.Strategie:
    """Strategie du scenario de reference (constante REFERENCE_STRATEGIE_NOM): a lire
    partout ou un bloc calcule LE scenario de reference (macros *Basis/*Yann, Monte Carlo
    de base, fig_bruecke/fig_zeitplan/fig_start_verzoegerung), jamais un nom en dur."""
    return next(s for s in STRATEGIEN if s.name == REFERENCE_STRATEGIE_NOM)


def _reference_plan(jahre: int = JAHRE_HORIZONT, **kwargs) -> list:
    """Plan d'epargne du scenario de reference (constante REFERENCE_QUOTE_NOM): a lire
    partout ou un bloc calcule LE scenario de reference, jamais QUOTEN["Zwanzig"] en dur."""
    return p.sparplan_aus_quote(QUOTEN[REFERENCE_QUOTE_NOM], jahre=jahre, **kwargs)


def _sq_de_base() -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Table des retenues a la source utilisee PAR DEFAUT dans tout ce
        module (correction du controleur apres revue de la partie A):
        W-8BEN suppose deja depose pour les dividendes americains (retenue
        au taux DBA de 15 %, "mit_antrag"), jamais pour la Suisse ni la
        France (retenue standard "einbehalt" conservee: un remboursement
        actif y reste une demarche separee que l'investisseur peut ne
        jamais faire, modelisee a part dans fig_steuer_100 sous "avec
        remboursement"/"avec formulaire"). Ne modifie ni fiscalite.py ni
        projection.py (deja testes, hors perimetre de cette tache): copie
        profonde de la table sourcee dans hypotheses.py, seule l'entree US
        substituee.

    Inputs:
        Neant (lit hypotheses.wert("quellensteuer")).

    Outputs:
        result (dict): meme structure que hypotheses.wert("quellensteuer"),
            avec sq["US"]["einbehalt"] = sq["US"]["mit_antrag"] (15 %).
    --------------------------------------------------------------------------
    """
    sq = json.loads(json.dumps(h.wert("quellensteuer")))  # copie profonde
    sq["US"]["einbehalt"] = sq["US"]["mit_antrag"]
    return sq


def _sim_kwargs() -> dict:
    """Arguments imposes pour tout appel a projection.simulieren dans ce module
    (brief tache 9): jamais un taux ecrit en dur ici, tout vient de hypotheses.py.
    sq= utilise _sq_de_base() (W-8BEN suppose depose pour les dividendes americains,
    correction du controleur), pas la table brute de hypotheses.py."""
    return {"saetze": f.saetze_2026(), "pauschbetrag_nominal": h.wert("sparerpauschbetrag"),
            "sq": _sq_de_base()}


def _charger_laender_mix() -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Remplacer le mix pays par defaut de projection.py par celui du
        portefeuille-exemple (brief tache 9: "avant tout calcul"), et ramener
        a "US" tout pays absent de la table des retenues (avec une ligne
        LUECKEN.md), decision 9 du controleur. Renormalise ensuite le mix a
        somme exactement 1 (correctif infrastructure, tache 10): les parts
        de data/portefeuille_meta.json sont arrondies a 4 decimales et
        sommaient a 1,0001 (0,6522 + 0,2609 + 0,087), si bien que chaque
        poste de la cascade du chapitre 2 (data/fig_kaskade.csv) portait sur
        brut x 1,0001 au lieu de brut exact - la cascade ne se refermait pas
        tout a fait (ecart ~7,6 EUR/an sur ~76 000 EUR, documente comme non
        corrige dans le rapport de tache 10, chapitres 02-03, decision 3).

    Inputs:
        Neant (lit data/portefeuille_meta.json et data/quellen.json).

    Outputs:
        None. Effets de bord: p.LAENDER_MIX reassigne (parts renormalisees,
        somme exactement 1.0); LUECKEN.md complete si un pays manque (aucun
        cas attendu, US/DE/FR couverts).
    --------------------------------------------------------------------------
    """
    meta = json.load(open(os.path.join(DATA, "portefeuille_meta.json")))
    mix = dict(meta["laender_mix"])
    sq = h.wert("quellensteuer")
    manquants = sorted(land for land in mix if land not in sq)
    if manquants:
        with open(os.path.join(ROOT, "LUECKEN.md"), "a") as fo:
            fo.write(f"\n- Tache 9: pays du portefeuille absents de la table des "
                     f"retenues, ramenes a \"US\": {manquants}.\n")
        for land in manquants:
            mix["US"] = mix.get("US", 0.0) + mix.pop(land)
    somme = sum(mix.values())
    if somme:
        mix = {land: part / somme for land, part in mix.items()}
    p.LAENDER_MIX = mix


def markt_basis() -> p.Markt:
    """Hypotheses de marche REELLES: rendements tires des donnees Shiller (mediane
    historique) et du portefeuille-exemple (rendement et croissance du dividende)."""
    _charger_laender_mix()

    shiller = pd.read_csv(os.path.join(DATA, "shiller_jahr.csv"))
    # Les donnees Shiller s'arretent en 2022, pas 2023 comme l'ecrit le brief (note du
    # controleur, tache 9): fenetre 1950-2022.
    fenetre = shiller[(shiller["jahr"] >= FENETRE_MARCHE[0]) & (shiller["jahr"] <= FENETRE_MARCHE[1])]
    n_annees = len(fenetre)
    # TCAC (moyenne geometrique), PAS la mediane ni la moyenne arithmetique des
    # rendements annuels: correction du controleur (ecart au plan initial, voir
    # docstring du module) - la mediane/moyenne arithmetique d'une serie annuelle n'est
    # pas un taux de capitalisation valide sur plusieurs decennies (elle ignore l'effet
    # multiplicatif des annees de krach, cf. 2008 -38,8 %, 2022 -20,1 %).
    rendite_real = float((1 + fenetre["rendite_real"]).prod() ** (1 / n_annees) - 1)
    rendite_real_mediane = float(fenetre["rendite_real"].median())

    portefeuille = pd.read_csv(os.path.join(DATA, "portefeuille.csv"))
    rendite_div_maison = float(portefeuille["rendite_ttm"].median())
    wachstum_maison = rendite_real - rendite_div_maison

    # rendement d'un ETF monde: Annahme declaree, sourcee (ou repli explicite documente)
    # sous "etf_welt_rendite_div" dans data/quellen.json, voir LUECKEN.md.
    rendite_div_etf = h.wert("etf_welt_rendite_div")
    wachstum_etf = rendite_real - rendite_div_etf

    inflation = h.wert("inflation_ziel")
    basiszins = h.wert("basiszins_2026")

    M["RenditeReal"] = fmt(rendite_real * 100, 1)
    # Macros texte d'hypothese: accents en commandes LaTeX (\\'e), car elles s'impriment
    # telles quelles dans les encadres \\annahme des chapitres (tache 10, chapitres 02-03).
    M["RenditeReelleConvention"] = ("taux de croissance annuel compos\\'e (moyenne g\\'eom\\'etrique) "
                                    "1950--2022, pas la m\\'ediane")
    M["RenditeReelleMediane"] = fmt(rendite_real_mediane * 100, 1)
    # Nom de l'indice des donnees Shiller, en macro texte: son nom contient un nombre que
    # scripts/check_literals.py refuserait tape dans une section.
    M["IndiceActionsUs"] = "S\\&P~500"
    M["RenditeDivMaison"] = fmt(rendite_div_maison * 100, 1)
    M["RenditeDivEtf"] = fmt(rendite_div_etf * 100, 1)
    M["InflationZiel"] = fmt(inflation * 100, 1)
    M["HypotheseUsFormulaire"] = ("Retenue am\\'ericaine suppos\\'ee au taux conventionnel de "
                                  f"{fmt(h.wert('quellensteuer')['US']['mit_antrag'] * 100)}~\\%, "
                                  "formulaire W-8BEN suppos\\'e d\\'ej\\`a d\\'epos\\'e aupr\\`es du courtier")
    # Decision 6 du controleur (brief tache 9): le brief modelise EtfUmschichtung comme
    # un ETF distribuant DES LE DEPART (etf_ausschuettend=True dans STRATEGIEN), alors
    # que la specification (section 3) decrit une bascule vers des ETF distribuants
    # trois a cinq ans avant le depart. Modelisation simple du brief conservee, ecrite
    # ici en hypothese declaree (revue de code, tache 9 fix round 1: manquait comme
    # macro texte, n'existait qu'en commentaire du rapport partie A).
    M["HypotheseUmschichtung"] = ("La variante ETF distribuant est mod\\'elis\\'ee comme un ETF "
                                  "distribuant d\\`es le d\\'ebut de l'accumulation, pas comme une "
                                  "bascule trois \\`a cinq ans avant le d\\'epart (simplification du "
                                  "brief~; le co\\^ut fiscal du changement de part n'est pas "
                                  "mod\\'elis\\'e)")
    # Revue finale, constat 5: distinct de HypotheseUmschichtung ci-dessus (qui documente
    # EtfUmschichtung, etf_ausschuettend=True des le depart). Les strategies "Maison*"
    # (dont la reference MaisonSiebzig) ont etf_ausschuettend=False: leur poche ETF est
    # imposee comme capitalisante (Vorabpauschale) pendant l'epargne, mais
    # projection._einkommen() lui attribue rendite_div_etf a la retraite comme si elle
    # etait deja distribuante (convention "dividende" du module, voir sa docstring),
    # sans modeliser le cout fiscal du changement de categorie de parts. Favorise
    # legerement le resultat, comme HypotheseUmschichtung.
    M["HypotheseBasculeReference"] = ("La poche ETF des strat\\'egies \\`a dividendes (dont la "
                                      "r\\'ef\\'erence) est impos\\'ee comme un ETF capitalisant "
                                      "(Vorabpauschale) pendant l'\\'epargne, puis trait\\'ee comme "
                                      "un ETF distribuant d\\`es la retraite~; le co\\^ut fiscal du "
                                      "changement de cat\\'egorie de parts n'est pas mod\\'elis\\'e")

    # --- Scenario de reference (tache 10, correctif du 30.09.2026): macros texte pour que
    # la redaction (ch. 2-5, prochaine passe) cite \ReferenzStrategie/\ReferenzSparquote au
    # lieu de retaper "70/30" ou "30 %" en dur. Derivees de STRATEGIEN/QUOTEN via
    # REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM (definition unique, voir plus haut dans
    # le module): un futur changement de ces deux constantes suffit a les mettre a jour. ---
    ref_strat = _reference_strategie()
    M["ReferenzStrategie"] = (f"{fmt(ref_strat.anteil_maison * 100)}/"
                              f"{fmt((1 - ref_strat.anteil_maison) * 100)}")
    M["ReferenzSparquote"] = fmt(QUOTEN[REFERENCE_QUOTE_NOM] * 100)

    return p.Markt(rendite_div_maison=rendite_div_maison, wachstum_maison=wachstum_maison,
                    rendite_div_etf=rendite_div_etf, wachstum_etf=wachstum_etf,
                    inflation=inflation, basiszins=basiszins)


def kapitel_2_und_3():
    """fig_ziel_nominal, fig_kaskade, fig_steuer_100, fig_steuer_kumuliert."""
    markt = markt_basis()
    sim_kw = _sim_kwargs()
    sq = sim_kw["sq"]
    pausch = sim_kw["pauschbetrag_nominal"]
    saetze = sim_kw["saetze"]

    # --- Parametres du profil repris tels quels dans le texte (tache 10, chapitre 2):
    # memes objets que ceux qu'utilisent les calculs, jamais une seconde copie. ---
    M["ZielMonat"] = fmt(ZIEL_REAL_MONAT)
    M["ZielJahr"] = fmt(ZIEL_REAL_JAHR)
    M["EpargneMinimum"] = fmt(inspect.signature(p.sparplan_aus_quote).parameters["minimum"].default)
    M["KapitalDepartNiedrig"] = fmt(STARTKAPITAL["Niedrig"])
    M["KapitalDepartHoch"] = fmt(STARTKAPITAL["Hoch"])
    for quote_nom, quote in QUOTEN.items():
        M[f"Sparquote{quote_nom}"] = fmt(quote * 100)
    for land, part in p.LAENDER_MIX.items():
        M[f"Part{land.capitalize()}"] = fmt(part * 100, 1)

    # --- fig_ziel_nominal: 3500 EUR d'aujourd'hui en euros nominaux jusqu'en 2060 ---
    # Trois inflations: basse, cible (hypotheses.py, "inflation_ziel") et haute.
    inflation_ziel = h.wert("inflation_ziel")
    jahre = list(range(2026, 2061))
    lignes = []
    for jahr in jahre:
        n = jahr - 2026
        lignes.append({"jahr": jahr,
                        "inflation_1": ZIEL_REAL_MONAT * (1 + INFLATION_BAS) ** n,
                        "inflation_2": ZIEL_REAL_MONAT * (1 + inflation_ziel) ** n,
                        "inflation_3": ZIEL_REAL_MONAT * (1 + INFLATION_HAUT) ** n})
    schreiben("fig_ziel_nominal", pd.DataFrame(lignes))
    jahr_cible = 2026 + 18
    ziel_nominal_cible = ZIEL_REAL_MONAT * (1 + inflation_ziel) ** 18
    M[f"ZielNominal{annee_romaine(jahr_cible)}"] = fmt(ziel_nominal_cible)
    derniere = lignes[-1]
    fin = annee_romaine(derniere["jahr"])
    M[f"ZielNominal{fin}Bas"] = fmt(derniere["inflation_1"])
    M[f"ZielNominal{fin}"] = fmt(derniere["inflation_2"])
    M[f"ZielNominal{fin}Haut"] = fmt(derniere["inflation_3"])
    M["InflationBas"] = fmt(INFLATION_BAS * 100, 1)
    M["InflationHaut"] = fmt(INFLATION_HAUT * 100, 1)

    # --- Taux et plafonds sources (chapitre 3), lus dans hypotheses.py ---
    M["AbgeltungSatz"] = fmt(h.wert("abgeltungsteuer_satz") * 100)
    M["SoliSatz"] = fmt(h.wert("soli_satz") * 100, 1)
    M["SteuerSatzGesamt"] = fmt(h.wert("abgeltungsteuer_satz") * (1 + h.wert("soli_satz")) * 100, 3)
    M["Sparerpauschbetrag"] = fmt(pausch)
    M["Teilfreistellung"] = fmt(h.wert("teilfreistellung_aktienfonds") * 100)
    M["Basiszins"] = fmt(h.wert("basiszins_2026") * 100, 1)
    M["SoliFreigrenze"] = fmt(h.wert("soli_freigrenze_2026"))
    M["KvSatz"] = fmt(saetze["kv"] * 100, 1)
    M["KvZusatz"] = fmt(saetze["zusatz"] * 100, 1)
    M["PvSatz"] = fmt(saetze["pv"] * 100, 1)
    kv_pv_satz = saetze["kv"] + saetze["zusatz"] + saetze["pv"]
    M["KvPvSatz"] = fmt(kv_pv_satz * 100, 1)
    M["KvMindestMonat"] = fmt(saetze["min_monat"], 2)
    M["KvBbgMonat"] = fmt(saetze["bbg_monat"], 2)
    M["KvPvMinMonat"] = fmt(saetze["min_monat"] * kv_pv_satz)
    M["KvPvMaxMonat"] = fmt(saetze["bbg_monat"] * kv_pv_satz)
    sq_brut_taux = h.wert("quellensteuer")
    M["QstUsStandard"] = fmt(sq_brut_taux["US"]["einbehalt"] * 100)
    M["QstUsFormulaire"] = fmt(sq_brut_taux["US"]["mit_antrag"] * 100)
    M["QstCh"] = fmt(sq_brut_taux["CH"]["einbehalt"] * 100)
    M["QstChAnrechenbar"] = fmt(sq_brut_taux["CH"]["anrechenbar"] * 100)
    M["QstFr"] = fmt(sq_brut_taux["FR"]["einbehalt"] * 100, 1)
    M["QstNl"] = fmt(sq_brut_taux["NL"]["einbehalt"] * 100)
    M["QstGb"] = fmt(sq_brut_taux["GB"]["einbehalt"] * 100)
    for nom, k in KIST_VARIANTEN.items():
        M[f"Kist{nom}"] = fmt(k * 100)

    # --- fig_kaskade: brut -> retenues etrangeres -> impot allemand -> KV/PV -> disponible ---
    brutto = f.brutto_fuer_netto(ZIEL_REAL_JAHR, p.LAENDER_MIX, saetze, pausch, sq=sq)
    rest, total_einbehalt, total_steuer_de, total_netto_apres_impot = pausch, 0.0, 0.0, 0.0
    for land, part in p.LAENDER_MIX.items():
        b = brutto * part
        n, rest = f.posten_netto(b, land, rest, sq=sq)
        einbehalt = sq[land]["einbehalt"] * b
        total_einbehalt += einbehalt
        total_steuer_de += b - einbehalt - n
        total_netto_apres_impot += n
    kv_pv = f.kv_beitrag_jahr(brutto, saetze)
    disponible = total_netto_apres_impot - kv_pv
    schreiben("fig_kaskade", pd.DataFrame([
        {"posten": "brut_necessaire", "betrag": brutto},
        {"posten": "retenues_etrangeres", "betrag": -total_einbehalt},
        {"posten": "impot_allemand", "betrag": -total_steuer_de},
        {"posten": "kv_pv", "betrag": -kv_pv},
        {"posten": "disponible", "betrag": disponible},
    ]))
    M["BruttoNoetig"] = fmt(brutto)
    M["BruttoNoetigMois"] = fmt(brutto / 12)
    M["RetenuesEtrangeres"] = fmt(total_einbehalt)
    M["ImpotAllemand"] = fmt(total_steuer_de)
    M["KvPvAnnuel"] = fmt(kv_pv)
    M["PartDisponible"] = fmt(disponible / brutto * 100, 1)
    M["KvPvAnteilBrut"] = fmt(kv_pv / brutto * 100, 1)
    # Impot d'Eglise (variante du chapitre 3): brut necessaire pour le meme net.
    for nom, k in KIST_VARIANTEN.items():
        M[f"BruttoNoetigKist{nom}"] = fmt(f.brutto_fuer_netto(ZIEL_REAL_JAHR, p.LAENDER_MIX, saetze,
                                                              pausch, sq=sq, kist=k))
    # Guenstigerpruefung (§32d al. 6 EStG), meme brut: voir _guenstigerpruefung().
    g = _guenstigerpruefung(brutto, p.LAENDER_MIX, saetze, pausch, sq)
    M["GuenstigerRevenuImposable"] = fmt(g["zve"])
    M["GuenstigerImpotTarif"] = fmt(g["steuer_de"])
    M["GuenstigerEconomieAn"] = fmt(total_steuer_de - g["steuer_de"])
    M["GuenstigerEconomieMois"] = fmt((total_steuer_de - g["steuer_de"]) / 12)

    # --- fig_steuer_100: "100 EUR de dividende, combien pour toi ?" ---
    # Forfait suppose deja consomme par ailleurs (retraite vivant de dividendes,
    # cas marginal): pauschbetrag_rest=0 pour chaque poste isole ci-dessous.
    # sq (via sim_kw) applique deja le W-8BEN suppose depose pour "US" (correction du
    # controleur); sq_brut est la table sourcee non modifiee, pour la barre de
    # comparaison "US sans W-8BEN" (30 %, retenue standard sans formulaire).
    sq_brut = h.wert("quellensteuer")

    def net100(art, sq_utilise=None, **kw):
        return f.posten_netto(100.0, art, 0.0, sq=sq_utilise if sq_utilise is not None else sq, **kw)[0]

    lignes_100 = [
        {"art": "DE", "netto": net100("DE")},
        {"art": "US", "netto": net100("US")},
        {"art": "US_sans_w8ben", "netto": net100("US", sq_utilise=sq_brut)},
        {"art": "CH", "netto": net100("CH")},
        {"art": "CH_avec_remboursement", "netto": net100("CH", antrag=True)},
        {"art": "FR", "netto": net100("FR")},
        {"art": "FR_avec_formulaire", "netto": net100("FR", antrag=True)},
        {"art": "NL", "netto": net100("NL")},
        {"art": "GB", "netto": net100("GB")},
        {"art": "ETF", "netto": net100("ETF")},
        {"art": "DE_kirchensteuer_8", "netto": net100("DE", kist=KIST_VARIANTEN["Acht"])},
        {"art": "DE_kirchensteuer_9", "netto": net100("DE", kist=KIST_VARIANTEN["Neun"])},
    ]
    schreiben("fig_steuer_100", pd.DataFrame(lignes_100))
    par_art = {r["art"]: r["netto"] for r in lignes_100}
    M["NettoDe"] = fmt(par_art["DE"], 2)
    M["NettoUs"] = fmt(par_art["US"], 2)
    M["NettoUsSansFormulaire"] = fmt(par_art["US_sans_w8ben"], 2)
    M["NettoCh"] = fmt(par_art["CH"], 2)
    M["NettoEtf"] = fmt(par_art["ETF"], 2)
    M["NettoChRemboursement"] = fmt(par_art["CH_avec_remboursement"], 2)
    M["NettoFr"] = fmt(par_art["FR"], 2)
    M["NettoNl"] = fmt(par_art["NL"], 2)
    M["NettoGb"] = fmt(par_art["GB"], 2)
    M["NettoDeKistAcht"] = fmt(par_art["DE_kirchensteuer_8"], 2)
    M["NettoDeKistNeun"] = fmt(par_art["DE_kirchensteuer_9"], 2)

    # --- fig_steuer_kumuliert: impot cumule sur 18 ans, taux d'epargne 20 %, "Mitte" ---
    jahre_kum = 18
    plan = p.sparplan_aus_quote(0.20, jahre=jahre_kum)
    strat_maison = next(s for s in STRATEGIEN if s.name == "MaisonHundert")
    strat_etf_thes = next(s for s in STRATEGIEN if s.name == "EtfReferenz")
    strat_etf_aus = next(s for s in STRATEGIEN if s.name == "EtfUmschichtung")
    s_maison = p.simulieren(strat_maison, markt, plan, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                             jahre_kum, **sim_kw)
    s_etf_thes = p.simulieren(strat_etf_thes, markt, plan, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                               jahre_kum, **sim_kw)
    s_etf_aus = p.simulieren(strat_etf_aus, markt, plan, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                              jahre_kum, **sim_kw)
    cum_maison, cum_etf_thes, cum_etf_aus, lignes_kum = 0.0, 0.0, 0.0, []
    for i, jahr in enumerate(s_maison["jahre"]):
        cum_maison += s_maison["steuern"][i]
        cum_etf_thes += s_etf_thes["steuern"][i]
        cum_etf_aus += s_etf_aus["steuern"][i]
        lignes_kum.append({"jahr": jahr, "maison": cum_maison, "etf_thes": cum_etf_thes,
                            "etf_aus": cum_etf_aus})
    schreiben("fig_steuer_kumuliert", pd.DataFrame(lignes_kum))
    M["SteuerDifferenzAchtzehn"] = fmt(cum_maison - cum_etf_thes)
    M["SteuerKumMaison"] = fmt(cum_maison)
    M["SteuerKumEtfThes"] = fmt(cum_etf_thes)
    M["SteuerKumEtfAus"] = fmt(cum_etf_aus)


def _guenstigerpruefung(brutto: float, mix: dict, saetze: dict, pausch: float,
                         sq: dict) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Impot allemand sur les dividendes si Yann demande la Guenstigerpruefung
        (§32d al. 6 EStG): les revenus de capitaux passent au bareme progressif
        du §32a EStG au lieu du taux forfaitaire de 25 %. Pour un retraite qui
        vit de ses seuls dividendes, deux effets jouent: les cotisations
        d'assurance maladie et dependance de base deviennent deductibles
        (Sonderausgaben), et l'impot restant peut tomber sous la Freigrenze du
        Soli. Approximation declaree (chapitre 3, a faire valider par un
        conseiller): aucun autre revenu, aucune autre deduction que le
        Sparerpauschbetrag et les cotisations KV/PV; credit d'impot etranger
        (§34c EStG) plafonne pays par pays a la part de l'impot allemand qui
        revient a ce pays (part du brut); Soli sans zone de transition au-dessus
        de la Freigrenze (meme convention que salaire.netto_jahr). N'entre dans
        aucun autre calcul du document: le reste du modele garde le taux
        forfaitaire, plus prudent.

    Inputs:
        brutto (float): dividende brut annuel (euros de 2026).
        mix (dict[str, float]): part de chaque pays dans le brut.
        saetze (dict): taux KV/PV et plafonds (fiscalite.saetze_2026()).
        pausch (float): Sparerpauschbetrag.
        sq (dict): table des retenues a la source (W-8BEN suppose depose).

    Outputs:
        result (dict): "zve" (revenu imposable), "est" (impot du bareme avant
            credit), "credit" (retenues etrangeres imputees), "steuer_de"
            (impot allemand final, Soli compris).
    --------------------------------------------------------------------------
    """
    kv_pv = f.kv_beitrag_jahr(brutto, saetze)
    zve = max(0.0, brutto - pausch - kv_pv)
    est = salaire.est_32a(zve, h.wert("est_tarif_2026"))
    credit = 0.0
    for land, part in mix.items():
        b = brutto * part
        credit += min(sq[land]["anrechenbar"] * b, est * part)
    est_nach = max(0.0, est - credit)
    soli = h.wert("soli_satz") * est_nach if est_nach > h.wert("soli_freigrenze_2026") else 0.0
    return {"zve": zve, "est": est, "credit": credit, "steuer_de": est_nach + soli}


def _kapital_pour_ziel(m: float, rendite_maison: float, markt: p.Markt, saetze: dict,
                        sq: dict, pausch: float, kv_teilfreistellung_etf: bool,
                        ziel_jahr: float = None) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Capital total reel (poche maison + poche ETF, sans simulation
        temporelle) qui degage un revenu net de ziel_jahr (ZIEL_REAL_JAHR par
        defaut) par dividendes seuls, pour une part maison m et un rendement
        brut maison rendite_maison; generalisation, sur le capital K, de la
        formule de revenu "dividende" utilisee par projection._einkommen
        (meme traitement KV/Teilfreistellung), par bissection.

    Inputs:
        m (float): part du capital investie en maison, 0-1.
        rendite_maison (float): rendement brut de dividende de la poche
            maison (variable dans fig_kapital_rendite, fixe a
            markt.rendite_div_maison dans fig_div_vs_4).
        markt (projection.Markt): fournit rendite_div_etf pour la poche ETF.
        saetze, sq, pausch: memes conventions que partout ailleurs (sourcees
            hypotheses.py / fiscalite.saetze_2026 / quellensteuer).
        kv_teilfreistellung_etf (bool): meme convention que projection.py.
        ziel_jahr (float): revenu net annuel vise (euros de 2026); ZIEL_REAL_JAHR
            (3500 EUR/mois) si omis. Parametrage ajoute tache 11 (variante
            2500 EUR/mois), aucun appelant existant ne passe cet argument.

    Outputs:
        result (float): capital K (bissection sur 200 iterations).
    --------------------------------------------------------------------------
    """
    if ziel_jahr is None:
        ziel_jahr = ZIEL_REAL_JAHR

    def netto(k: float) -> float:
        maison_brutto = k * m * rendite_maison
        etf_brutto = k * (1 - m) * markt.rendite_div_etf
        posten = [(maison_brutto * a, land) for land, a in p.LAENDER_MIX.items()]
        posten.append((etf_brutto, "ETF"))
        etf_kv = etf_brutto * (1 - f.TEILFREI) if kv_teilfreistellung_etf else etf_brutto
        return f.jahres_netto(posten, pausch, sq=sq) - f.kv_beitrag_jahr(maison_brutto + etf_kv, saetze)

    lo, hi = 0.0, ziel_jahr * 200
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if netto(mid) < ziel_jahr else (lo, mid)
    return (lo + hi) / 2


def _kapital_etf_regle4(markt: p.Markt, saetze: dict, sq: dict, pausch: float,
                         kv_teilfreistellung_etf: bool, taux: float = None,
                         ziel_jahr: float = None) -> float:
    """Capital ETF pur (100 % ETF) pour la regle des 4 %: hypothese conservatrice
    declaree gewinnanteil=1 (portefeuille suppose entierement en plus-value, cas le
    plus defavorable fiscalement, puisqu'aucune duree de detention n'est simulee ici,
    contrairement a projection.simulieren qui suit basis_etf annee par annee).
    taux: taux de retrait annuel, TAUX_RETRAIT (4 %) par defaut; TAUX_RETRAIT_PRUDENT
    (3 %) pour la variante prudente du chapitre 4 (tache 10, chapitres 04-05).
    ziel_jahr: revenu net annuel vise, ZIEL_REAL_JAHR par defaut (parametrage ajoute
    tache 11, variante 2500 EUR/mois; aucun appelant existant ne passe cet argument)."""
    if taux is None:
        taux = TAUX_RETRAIT
    if ziel_jahr is None:
        ziel_jahr = ZIEL_REAL_JAHR

    def netto(k: float) -> float:
        entnahme = taux * k
        steuerpfl_brutto = entnahme  # gewinnanteil = 1, cf. docstring
        net_apres_impot = entnahme - steuerpfl_brutto + f.posten_netto(steuerpfl_brutto, "ETF",
                                                                        pausch, sq=sq)[0]
        steuerpfl_kv = steuerpfl_brutto * (1 - f.TEILFREI) if kv_teilfreistellung_etf else steuerpfl_brutto
        return net_apres_impot - f.kv_beitrag_jahr(steuerpfl_kv, saetze)

    lo, hi = 0.0, ziel_jahr * 300
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if netto(mid) < ziel_jahr else (lo, mid)
    return (lo + hi) / 2


def kapitel_4():
    """fig_kapital_rendite, fig_div_vs_4."""
    markt = markt_basis()
    sim_kw = _sim_kwargs()
    sq, pausch, saetze = sim_kw["sq"], sim_kw["pauschbetrag_nominal"], sim_kw["saetze"]
    kv_teilfrei = bool(h.wert("kv_teilfreistellung_etf"))

    lignes = []
    for r in RENDITE_GRILLE:
        lignes.append({"rendite": r,
                        "maison50": _kapital_pour_ziel(0.5, r, markt, saetze, sq, pausch, kv_teilfrei),
                        "maison70": _kapital_pour_ziel(0.7, r, markt, saetze, sq, pausch, kv_teilfrei),
                        "maison100": _kapital_pour_ziel(1.0, r, markt, saetze, sq, pausch, kv_teilfrei)})
    schreiben("fig_kapital_rendite", pd.DataFrame(lignes))

    r_ref = markt.rendite_div_maison
    kapital_50 = _kapital_pour_ziel(0.5, r_ref, markt, saetze, sq, pausch, kv_teilfrei)
    kapital_70 = _kapital_pour_ziel(0.7, r_ref, markt, saetze, sq, pausch, kv_teilfrei)
    kapital_100 = _kapital_pour_ziel(1.0, r_ref, markt, saetze, sq, pausch, kv_teilfrei)
    kapital_etf = _kapital_etf_regle4(markt, saetze, sq, pausch, kv_teilfrei)
    schreiben("fig_div_vs_4", pd.DataFrame([
        {"strategie": "MaisonFuenfzig", "kapital": kapital_50},
        {"strategie": "MaisonSiebzig", "kapital": kapital_70},
        {"strategie": "MaisonHundert", "kapital": kapital_100},
        {"strategie": "EtfReferenz", "kapital": kapital_etf},
    ]))
    M["KapitalMaisonSiebzig"] = fmt(kapital_70)
    M["KapitalEtfReferenz"] = fmt(kapital_etf)
    M["KapitalDifferenzProzent"] = fmt((kapital_100 / kapital_etf - 1) * 100, 1)

    # --- Macros de redaction du chapitre 4 (tache 10, chapitres 04-05). Toutes derivent
    # des MEMES appels que fig_div_vs_4/fig_kapital_rendite ci-dessus (aucun second
    # calcul du capital requis): le resume reprendra \KapitalMaisonSiebzig et
    # \KapitalEtfReferenz tels quels (controle croise, specification section 8). ---
    M["KapitalMaisonFuenfzig"] = fmt(kapital_50)
    M["KapitalMaisonHundert"] = fmt(kapital_100)
    M["KapitalDifferenzSiebzigProzent"] = fmt((kapital_70 / kapital_etf - 1) * 100, 1)
    M["KapitalDifferenzSiebzigEuro"] = fmt(kapital_70 - kapital_etf)
    # Capital exprime en annees de depense cible (multiple de ZIEL_REAL_JAHR).
    M["MultipleEtfReferenz"] = fmt(kapital_etf / ZIEL_REAL_JAHR, 1)
    M["MultipleMaisonSiebzig"] = fmt(kapital_70 / ZIEL_REAL_JAHR, 1)
    # Retrait brut annuel de la reference et taux de retrait (regle et variante prudente).
    M["TauxRetrait"] = fmt(TAUX_RETRAIT * 100)
    M["TauxRetraitPrudent"] = fmt(TAUX_RETRAIT_PRUDENT * 100)
    M["RetraitBrutEtf"] = fmt(TAUX_RETRAIT * kapital_etf)
    kapital_etf_prudent = _kapital_etf_regle4(markt, saetze, sq, pausch, kv_teilfrei,
                                              taux=TAUX_RETRAIT_PRUDENT)
    M["KapitalEtfPrudent"] = fmt(kapital_etf_prudent)
    M["KapitalDifferenzPrudentProzent"] = fmt((kapital_100 / kapital_etf_prudent - 1) * 100, 1)
    # Bornes de fig_kapital_rendite (poche maison seule) et rendement brut maison a partir
    # duquel la strategie tout-maison demande autant de capital que la reference ETF
    # (bissection sur le meme _kapital_pour_ziel, capital decroissant avec le rendement).
    M["RenditeGrilleBas"] = fmt(RENDITE_GRILLE[0] * 100)
    M["RenditeGrilleHaut"] = fmt(RENDITE_GRILLE[-1] * 100)
    M["KapitalMaisonHundertRenditeBas"] = fmt(lignes[0]["maison100"])
    M["KapitalMaisonHundertRenditeHaut"] = fmt(lignes[-1]["maison100"])
    lo, hi = RENDITE_GRILLE[0], RENDITE_GRILLE[-1] * 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if _kapital_pour_ziel(1.0, mid, markt, saetze, sq, pausch, kv_teilfrei) > kapital_etf:
            lo = mid
        else:
            hi = mid
    M["RenditeEquilibreEtf"] = fmt((lo + hi) / 2 * 100, 1)
    # Nom de la colonne "tout en maison" de fig_kapital_rendite et fig_jahre_sparquote, en
    # macro: les chapitres 4 et 5 la passent a \linienfigur, et scripts/check_literals.py
    # prendrait les chiffres de "maison100" tapes dans une section pour une valeur.
    M["ColonneToutMaison"] = "maison100"


def _sparbedarf_monatlich(strategie: p.Strategie, markt: p.Markt, jahre_n: int,
                           start_kapital: float, sim_kw: dict) -> float | None:
    """Epargne mensuelle reelle constante minimale (bissection) atteignant l'objectif
    en exactement jahre_n annees; None si meme un versement enorme n'y suffit pas
    (plafond de recherche 20000 EUR/mois, doublé jusqu'a 200000 EUR/mois)."""
    def atteint(m: float) -> bool:
        plan = [m * 12] * jahre_n
        s = p.simulieren(strategie, markt, plan, start_kapital, ZIEL_REAL_MONAT, jahre_n, **sim_kw)
        return s["ziel_jahr"] is not None

    hi = 20000.0
    while not atteint(hi):
        hi *= 2
        if hi > 2_000_000.0:
            return None
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if atteint(mid):
            hi = mid
        else:
            lo = mid
    return hi


def kapitel_5():
    """fig_gehalt, fig_jahre_sparquote, fig_kapital_zeit, fig_zinseszins, fig_sparbedarf,
    fig_start_verzoegerung. Remplit aussi ZUS["alter"] et les macros \\Alter<Strategie><Quote>
    pour les 5 strategies x 4 quotas, capital de depart "Mitte"."""
    markt = markt_basis()
    sim_kw = _sim_kwargs()

    # --- fig_gehalt ---
    traj = salaire.trajektorie(START_JAHR, JAHRE_HORIZONT)
    schreiben("fig_gehalt", pd.DataFrame([{"jahr": t["jahr"], "brutto": t["brutto"],
                                            "netto": t["netto"]} for t in traj]))

    # --- ZUS["alter"] et macros Alter<Strategie><Quote>, capital "Mitte" ---
    for strat in STRATEGIEN:
        ZUS["alter"][strat.name] = {}
        for quote_nom, quote_val in QUOTEN.items():
            plan = p.sparplan_aus_quote(quote_val, jahre=JAHRE_HORIZONT)
            s = p.simulieren(strat, markt, plan, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                              JAHRE_HORIZONT, **sim_kw)
            if s["ziel_jahr"] is None:
                ZUS["alter"][strat.name][quote_nom] = None
                M[f"Alter{strat.name}{quote_nom}"] = "non atteint"
            else:
                alter = s["ziel_jahr"] - GEBURTSJAHR
                ZUS["alter"][strat.name][quote_nom] = alter
                M[f"Alter{strat.name}{quote_nom}"] = fmt(alter)

    # --- fig_jahre_sparquote: 5 a 70 % par pas de 5, capital "Mitte" ---
    strat_50 = next(s for s in STRATEGIEN if s.name == "MaisonFuenfzig")
    strat_70 = next(s for s in STRATEGIEN if s.name == "MaisonSiebzig")
    strat_100 = next(s for s in STRATEGIEN if s.name == "MaisonHundert")
    strat_etf = next(s for s in STRATEGIEN if s.name == "EtfReferenz")
    lignes_q = []
    for pct in range(5, 71, 5):
        quote = pct / 100
        plan = p.sparplan_aus_quote(quote, jahre=JAHRE_HORIZONT)

        def annees(strat):
            s = p.simulieren(strat, markt, plan, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                              JAHRE_HORIZONT, **sim_kw)
            return None if s["ziel_jahr"] is None else s["ziel_jahr"] - START_JAHR + 1

        lignes_q.append({"quote": quote, "maison50": annees(strat_50), "maison70": annees(strat_70),
                          "maison100": annees(strat_100), "etf_ref": annees(strat_etf)})
    schreiben("fig_jahre_sparquote", pd.DataFrame(lignes_q))

    # --- fig_kapital_zeit: strategie 70/30, quotas 10/20/30/50, "Mitte" ---
    plan_10 = p.sparplan_aus_quote(QUOTEN["Zehn"], jahre=JAHRE_HORIZONT)
    plan_20 = p.sparplan_aus_quote(QUOTEN["Zwanzig"], jahre=JAHRE_HORIZONT)
    plan_30 = p.sparplan_aus_quote(QUOTEN["Dreissig"], jahre=JAHRE_HORIZONT)
    plan_50 = p.sparplan_aus_quote(QUOTEN["Fuenfzig"], jahre=JAHRE_HORIZONT)
    s10 = p.simulieren(strat_70, markt, plan_10, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    s20 = p.simulieren(strat_70, markt, plan_20, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    s30 = p.simulieren(strat_70, markt, plan_30, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    s50 = p.simulieren(strat_70, markt, plan_50, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    ziel_kapital_70 = _kapital_pour_ziel(0.7, markt.rendite_div_maison, markt,
                                         sim_kw["saetze"], sim_kw["sq"], sim_kw["pauschbetrag_nominal"],
                                         bool(h.wert("kv_teilfreistellung_etf")))
    schreiben("fig_kapital_zeit", pd.DataFrame([
        {"jahr": s20["jahre"][i], "q10": s10["wert"][i], "q20": s20["wert"][i],
         "q30": s30["wert"][i], "q50": s50["wert"][i], "ziel_kapital": ziel_kapital_70}
        for i in range(JAHRE_HORIZONT)
    ]))

    # --- fig_zinseszins: part des interets composes, scenario de reference (constante
    # REFERENCE_QUOTE_NOM, tache 10 correctif du 30.09.2026: lu dans le dict ci-dessous
    # plutot que resimule, les 4 simulations s10/s20/s30/s50 couvrent deja QUOTEN au
    # complet) ---
    s_ref = {"Zehn": s10, "Zwanzig": s20, "Dreissig": s30, "Fuenfzig": s50}[REFERENCE_QUOTE_NOM]
    schreiben("fig_zinseszins", pd.DataFrame([
        {"jahr": s_ref["jahre"][i], "eingezahlt": s_ref["eingezahlt"][i],
         "gewinn": s_ref["wert"][i] - s_ref["eingezahlt"][i]}
        for i in range(JAHRE_HORIZONT)
    ]))

    # --- fig_sparbedarf: epargne mensuelle constante pour 40/45/50 ans, strategie 70/30 ---
    lignes_sb = []
    for nom_sb, alter_cible in SPARBEDARF_ALTER.items():
        jahre_n = (GEBURTSJAHR + alter_cible) - START_JAHR + 1
        m = _sparbedarf_monatlich(strat_70, markt, jahre_n, STARTKAPITAL["Mitte"], sim_kw)
        lignes_sb.append({"alter": alter_cible, "monatlich": m})
        # Macros de redaction (tache 10, chapitres 04-05): meme valeur que la ligne du CSV,
        # et sa part du salaire net mensuel d'entree (au-dela de 100 %: hors de portee).
        M[f"Sparbedarf{nom_sb}"] = "non atteint" if m is None else fmt(m)
        M[f"Sparbedarf{nom_sb}PartNet"] = ("non atteint" if m is None else
                                           fmt(m / traj[0]["netto_monat"] * 100))
    schreiben("fig_sparbedarf", pd.DataFrame(lignes_sb))

    # --- fig_start_verzoegerung: -2 a +5 ans, scenario de reference (strategie
    # REFERENCE_STRATEGIE_NOM, quote REFERENCE_QUOTE_NOM, "Mitte"; tache 10 correctif du
    # 30.09.2026, auparavant code en dur a QUOTEN["Zwanzig"]) ---
    lignes_sv = []
    for d in range(-2, 6):
        traj_d = salaire.trajektorie(START_JAHR + d, JAHRE_HORIZONT)
        netto_monat_d = [t["netto_monat"] for t in traj_d]
        plan_d = p.sparplan_aus_quote(QUOTEN[REFERENCE_QUOTE_NOM], jahre=JAHRE_HORIZONT,
                                       netto_monat=netto_monat_d)
        s_d = p.simulieren(strat_70, markt, plan_d, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                            JAHRE_HORIZONT, **sim_kw)
        if s_d["ziel_jahr"] is None:
            ziel_alter = None
        else:
            # simulieren etiquette toujours ses annees "2027+i" en interne (voir sa
            # docstring/implementation): l'annee calendaire reelle avec un depart decale
            # de d annees est ce label + d (decision documentee ici, tache 9 partie A).
            ziel_alter = (s_d["ziel_jahr"] + d) - GEBURTSJAHR
        lignes_sv.append({"verzoegerung_jahre": d, "ziel_alter": ziel_alter})
    schreiben("fig_start_verzoegerung", pd.DataFrame(lignes_sv))

    _macros_redaction_kapitel_5(markt, sim_kw, traj, strat_70, s10, s20, s30, s50, s_ref,
                                ziel_kapital_70, lignes_q, lignes_sv)


def _macros_redaction_kapitel_5(markt: p.Markt, sim_kw: dict, traj: list, strat_70: p.Strategie,
                                s10: dict, s20: dict, s30: dict, s50: dict, s_ref: dict,
                                ziel_kapital_70: float, lignes_q: list, lignes_sv: list) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Macros de redaction du chapitre 5 (tache 10, chapitres 04-05), toutes
        lues dans les simulations et CSV deja produits par kapitel_5 (aucun
        second calcul de l'age ni du capital: specification section 8), sauf
        trois grandeurs nouvelles et declarees comme telles: l'age du cas de
        base pour les deux autres capitaux de depart de la fourchette, la
        cotisation KV/PV maximale a l'annee d'objectif en euros de 2026
        (explique l'ecart entre \\KapitalYannBasis et \\KapitalMaisonSiebzig),
        et le controle croise du salaire net. Correctif tache 10 (30.09.2026):
        les macros du scenario de reference (Zins*, KapitalYannSousCibleProzent,
        KvPvMaxMoisObjectif, capital de depart Niedrig/Hoch) lisent desormais
        s_ref (scenario de reference, constante REFERENCE_QUOTE_NOM) au lieu de
        s20 en dur; s10/s20/s30/s50 restent lus tels quels pour les macros qui
        nomment explicitement leur quote (KapitalHorizonZehn/Zwanzig/Dreissig/Fuenfzig).

    Inputs:
        markt (projection.Markt), sim_kw (dict): memes hypotheses que kapitel_5.
        traj (list[dict]): trajectoire de salaire (salaire.trajektorie).
        strat_70 (projection.Strategie): strategie de base 70/30.
        s10, s20, s30, s50 (dict): simulations 70/30 a 10, 20, 30 et 50 % (fig_kapital_zeit).
        s_ref (dict): simulation 70/30 au taux de reference (REFERENCE_QUOTE_NOM,
            aliase l'une de s10/s20/s30/s50 selon cette constante).
        ziel_kapital_70 (float): capital requis du chapitre 4 (meme appel).
        lignes_q (list[dict]): lignes de fig_jahre_sparquote.
        lignes_sv (list[dict]): lignes de fig_start_verzoegerung.

    Outputs:
        None. Remplit M.
    --------------------------------------------------------------------------
    """
    # --- Horizon et ages de reference ---
    annee_fin = START_JAHR + JAHRE_HORIZONT - 1
    M["HorizonJahre"] = fmt(JAHRE_HORIZONT)
    M["AnneeHorizonFin"] = str(annee_fin)
    M["AlterHorizon"] = fmt(annee_fin - GEBURTSJAHR)
    M["AlterDebut"] = fmt(START_JAHR - GEBURTSJAHR)
    M["AgeLegal"] = fmt(int(h.wert("regelaltersgrenze")))

    # --- Salaire (fig_gehalt) ---
    M["GehaltBrutEntree"] = fmt(traj[0]["brutto"])
    M["GehaltNetEntree"] = fmt(traj[0]["netto"])
    M["GehaltNetEntreeMois"] = fmt(traj[0]["netto_monat"])
    M["GehaltNetPart"] = fmt(traj[0]["netto"] / traj[0]["brutto"] * 100, 1)
    M["GehaltWachstum"] = fmt(h.wert("gehaltssteigerung_real") * 100, 1)
    # Forfait de frais professionnels du calcul du net (salaire.WERBUNGSKOSTEN), expose
    # pour l'hypothese du chapitre 5; pas de cle dans data/quellen.json (signale au rapport).
    M["FraisProfessionnels"] = fmt(salaire.WERBUNGSKOSTEN)
    M["GehaltBrutFin"] = fmt(traj[-1]["brutto"])
    M["GehaltNetFin"] = fmt(traj[-1]["netto"])
    # Premiere annee ou le net baisse alors que le brut monte: la contribution de
    # solidarite s'applique en entier des que l'impot depasse sa Freigrenze
    # (salaire.netto_jahr, sans zone de transition), artefact du modele signale au lecteur.
    creux = [t["jahr"] for t0, t in zip(traj, traj[1:]) if t["netto"] < t0["netto"]]
    M["GehaltCreuxAnnee"] = str(creux[0]) if creux else "aucune"

    # --- Controle croise du salaire net (docs/salaire_controle.md) ---
    tarif, sv = h.wert("est_tarif_2026"), h.wert("sv_arbeitnehmer")
    sfg = h.wert("soli_freigrenze_2026")
    for nom, (brut, publie) in SALAIRE_CONTROLE.items():
        calcule = salaire.netto_jahr(brut, tarif, sv, sfg)
        M[f"SalaireControleBrut{nom}"] = fmt(brut)
        M[f"SalaireControlePublie{nom}"] = fmt(publie)
        M[f"SalaireControleCalcule{nom}"] = fmt(calcule)
        M[f"SalaireControleEcart{nom}"] = fmt((calcule / publie - 1) * 100, 2)
    M["SalaireControleSeuil"] = fmt(SALAIRE_CONTROLE_SEUIL * 100)

    # --- Epargne mensuelle de la premiere annee, par taux (sparplan_aus_quote) ---
    for quote_nom, quote_val in QUOTEN.items():
        plan = p.sparplan_aus_quote(quote_val, jahre=JAHRE_HORIZONT)
        M[f"EpargneMois{quote_nom}"] = fmt(plan[0] / 12)

    # --- Ages: ecart ETF / 70-30 a 20 %, taux minimal par strategie (fig_jahre_sparquote).
    # Macro delibererement figee a 20 % (decision 3 du controleur, tache 10 correctif du
    # 30.09.2026: "EcartAgeEtfMaisonVingt" nomme sa propre quote, ce n'est PAS une macro
    # *Basis* et elle ne suit donc pas le changement de reference); l'ancien resultat
    # reste lisible tel quel via ZUS["alter"][...]["Zwanzig"]. ---
    a70, aetf = ZUS["alter"]["MaisonSiebzig"]["Zwanzig"], ZUS["alter"]["EtfReferenz"]["Zwanzig"]
    M["EcartAgeEtfMaisonVingt"] = "non atteint" if None in (a70, aetf) else fmt(a70 - aetf)
    for col, nom in (("maison50", "MaisonFuenfzig"), ("maison70", "MaisonSiebzig"),
                     ("maison100", "MaisonHundert"), ("etf_ref", "EtfReferenz")):
        atteints = [l["quote"] for l in lignes_q if l[col] is not None]
        M[f"QuoteMin{nom}"] = fmt(min(atteints) * 100) if atteints else "non atteint"

    # --- Revision des chapitres 02-05 (tache 10, 30.09.2026): macros qui SUIVENT le
    # scenario de reference (REFERENCE_*), pour que la prose ne cite plus une quote en dur,
    # lues dans ZUS["alter"] (meme calcul que \ZielJahrBasis, specification section 8). ---
    a_ref = ZUS["alter"][REFERENCE_STRATEGIE_NOM][REFERENCE_QUOTE_NOM]
    a_etf_ref = ZUS["alter"]["EtfReferenz"][REFERENCE_QUOTE_NOM]
    M["AlterYannBasis"] = "non atteint" if a_ref is None else fmt(a_ref)
    M["AlterEtfBasis"] = "non atteint" if a_etf_ref is None else fmt(a_etf_ref)
    M["EcartAgeEtfMaisonBasis"] = ("non atteint" if None in (a_ref, a_etf_ref)
                                   else fmt(a_ref - a_etf_ref))
    M["EpargneMoisBasis"] = M[f"EpargneMois{REFERENCE_QUOTE_NOM}"]
    # Depart anticipe: premier taux de la grille de fig_jahre_sparquote (pas de 5 points)
    # dont l'age d'atteinte est STRICTEMENT inferieur a l'age legal, et cet age.
    age_legal = int(h.wert("regelaltersgrenze"))
    for col, nom in (("maison70", "MaisonSiebzig"), ("etf_ref", "EtfReferenz")):
        avant = [(l["quote"], START_JAHR + l[col] - 1 - GEBURTSJAHR) for l in lignes_q
                 if l[col] is not None and START_JAHR + l[col] - 1 - GEBURTSJAHR < age_legal]
        M[f"QuoteAnticipee{nom}"] = fmt(avant[0][0] * 100) if avant else "non atteint"
        M[f"AlterAnticipee{nom}"] = fmt(avant[0][1]) if avant else "non atteint"
    # Revenu net reel (euros de 2026) atteint en fin d'horizon par 70/30 a 20 %: dit a quel
    # point l'ancien scenario de base manque la cible quand il n'aboutit pas.
    rev20 = s20["einkommen_netto_monat_real"][-1]
    M["RevenuHorizonZwanzig"] = fmt(rev20)
    M["RevenuHorizonZwanzigPartCible"] = fmt(rev20 / ZIEL_REAL_MONAT * 100)

    # --- Capital de depart: scenario de reference pour les bornes de la fourchette
    # (tache 10 correctif du 30.09.2026: plan_ref suit REFERENCE_QUOTE_NOM, auparavant
    # code en dur a QUOTEN["Zwanzig"]) ---
    plan_ref = p.sparplan_aus_quote(QUOTEN[REFERENCE_QUOTE_NOM], jahre=JAHRE_HORIZONT)
    for nom in ("Niedrig", "Hoch"):
        s_k = p.simulieren(strat_70, markt, plan_ref, STARTKAPITAL[nom], ZIEL_REAL_MONAT,
                            JAHRE_HORIZONT, **sim_kw)
        M[f"AlterBasisDepart{nom}"] = ("non atteint" if s_k["ziel_jahr"] is None
                                       else fmt(s_k["ziel_jahr"] - GEBURTSJAHR))
        M[f"KapitalHorizonDepart{nom}"] = fmt(s_k["wert"][-1])
        M[f"RevenuHorizonDepart{nom}"] = fmt(s_k["einkommen_netto_monat_real"][-1])

    # --- Capital dans le temps (fig_kapital_zeit): valeurs en fin d'horizon, les 4 quotas
    # (KapitalHorizonDreissig ajoutee tache 10: manquait a la symetrie de Zehn/Zwanzig/
    # Fuenfzig, utile depuis que Dreissig est la quote de reference). ---
    M["KapitalHorizonZehn"] = fmt(s10["wert"][-1])
    M["KapitalHorizonZwanzig"] = fmt(s20["wert"][-1])
    M["KapitalHorizonDreissig"] = fmt(s30["wert"][-1])
    M["KapitalHorizonFuenfzig"] = fmt(s50["wert"][-1])
    croise = [j for j, w in zip(s50["jahre"], s50["wert"]) if w >= ziel_kapital_70]
    M["AnneeCroisementFuenfzig"] = str(croise[0]) if croise else "non atteint"

    # --- Interets composes (fig_zinseszins, scenario de reference) a l'annee d'objectif.
    # Tache 10 correctif du 30.09.2026: lit s_ref (REFERENCE_QUOTE_NOM) au lieu de s20. ---
    idx = s_ref["jahre"].index(s_ref["ziel_jahr"]) if s_ref["ziel_jahr"] is not None else -1
    verse, total = s_ref["eingezahlt"][idx], s_ref["wert"][idx]
    M["ZinsVersements"] = fmt(verse)
    M["ZinsGains"] = fmt(total - verse)
    M["ZinsPartGains"] = fmt((total - verse) / total * 100, 1)
    bascule = [j for j, e, w in zip(s_ref["jahre"], s_ref["eingezahlt"], s_ref["wert"]) if w - e > e]
    M["ZinsAnneeBascule"] = str(bascule[0]) if bascule else "non atteint"

    # --- Ecart entre le capital du chapitre 4 et le capital atteint a l'objectif ---
    # CORRIGE tache 10 (2026-09-30): avant le correctif, projection._saetze_real
    # deflatait a tort le plancher et le plafond KV comme un montant nominal fixe, ce qui
    # rendait la cotisation KV/PV maximale artificiellement plus basse a l'annee
    # d'objectif et permettait d'atteindre l'objectif avec moins que \KapitalMaisonSiebzig
    # (12,1 % de moins dans le cas de base a 20 %, chiffre mesure avant correctif). Le
    # plafond BBG (et le plancher) est desormais tenu CONSTANT en euros de 2026 (source:
    # data/quellen.json, cle "kv_schwellen_dynamisierung"), donc \KvPvMaxMoisObjectif ne
    # depend plus de l'annee d'objectif: a 20 % d'epargne, le cas de base n'atteignait
    # plus l'objectif dans l'horizon de JAHRE_HORIZONT ans (voir task-10-kv-fix-report.md).
    # Correctif du 30.09.2026 (ce bloc): LE scenario de reference passe a
    # REFERENCE_QUOTE_NOM (30 %, voir la definition module-level plus haut), qui redonne
    # un objectif atteint (voir \ZielJahrBasis, kapitel_10, pour la valeur mesuree). La
    # branche "non atteint" ci-dessous reste geree si un futur changement de parametres
    # (quote, horizon, hypotheses) la fait a nouveau echouer. Un "capital requis a
    # l'annee d'objectif" n'est volontairement PAS publie: la simulation ne reequilibre
    # pas les poches (part maison derive legerement de 70 % en fin d'horizon), un tel
    # chiffre a 70/30 serait approximatif.
    if s_ref["ziel_jahr"] is not None:
        i_obj = s_ref["jahre"].index(s_ref["ziel_jahr"])
        deflator = (1 + markt.inflation) ** i_obj
        sous_cible = (1 - s_ref["wert"][i_obj] / ziel_kapital_70) * 100
        M["KapitalYannSousCibleProzent"] = fmt(sous_cible, 1)
        # Revision des chapitres 02-05: valeur absolue et sens en toutes lettres, pour que
        # la prose n'ecrive jamais "sous la cible de -4,5 %" (signe negatif = au-dessus).
        M["KapitalYannEcartCibleProzent"] = fmt(abs(sous_cible), 1)
        M["KapitalYannSensCible"] = "au-dessus du" if sous_cible < 0 else "au-dessous du"
        M["KvPvMaxMoisObjectif"] = fmt(f.kv_beitrag_jahr(1e12, p._saetze_real(sim_kw["saetze"],
                                                                              deflator)) / 12)
    else:
        for k in ("KapitalYannSousCibleProzent", "KapitalYannEcartCibleProzent",
                  "KapitalYannSensCible", "KvPvMaxMoisObjectif"):
            M[k] = "non atteint"

    # --- Depart plus tot ou plus tard (fig_start_verzoegerung), scenario de reference. ---
    sv_ok = [l for l in lignes_sv if l["ziel_alter"] is not None]
    if sv_ok:
        M["VerzoegerungAnsAvance"] = fmt(-sv_ok[0]["verzoegerung_jahre"])
        M["VerzoegerungAlterAvance"] = fmt(sv_ok[0]["ziel_alter"])
        M["VerzoegerungAnsRetard"] = fmt(sv_ok[-1]["verzoegerung_jahre"])
        M["VerzoegerungAlterRetard"] = fmt(sv_ok[-1]["ziel_alter"])
        pente = np.polyfit([l["verzoegerung_jahre"] for l in sv_ok],
                           [l["ziel_alter"] for l in sv_ok], 1)[0]
        M["VerzoegerungPente"] = fmt(pente, 1)
    else:
        for k in ("VerzoegerungAnsAvance", "VerzoegerungAlterAvance", "VerzoegerungAnsRetard",
                  "VerzoegerungAlterRetard", "VerzoegerungPente"):
            M[k] = "non atteint"


SEED_ANZAHL_TITEL = 20260930
# Nombre de tirages aleatoires par taille n dans fig_anzahl_titel (valeur deja utilisee
# avant la tache 10, sortie en constante pour la macro \TiragesDiversification du ch. 6).
TIRAGES_ANZAHL_TITEL = 500
# Libelles francais des secteurs yfinance (.info "sector") et des pays ISO2, pour les
# figures et le tableau du chapitre 6 (tache 10). Constantes internes et de confiance
# (pas des donnees scrapees); un secteur ou pays absent de la table garde son libelle
# d'origine, echappe par escapieren(). Accents en UTF-8: lus tels quels par
# pgfplotstable et \input sous inputenc utf8 (verifie par compilation, tache 10).
SEKTOR_FR = {"Basic Materials": "Matériaux de base", "Communication Services": "Communication",
             "Consumer Cyclical": "Consommation cyclique",
             "Consumer Defensive": "Consommation de base", "Energy": "Énergie",
             "Financial Services": "Services financiers", "Healthcare": "Santé",
             "Industrials": "Industrie", "Real Estate": "Immobilier",
             "Technology": "Technologie", "Utilities": "Services aux collectivités"}
LAND_FR = {"US": "États-Unis", "DE": "Allemagne", "FR": "France", "NL": "Pays-Bas",
           "ES": "Espagne", "IT": "Italie", "BE": "Belgique", "FI": "Finlande",
           "IE": "Irlande", "CH": "Suisse", "GB": "Royaume-Uni"}
INDEX_FR = {"aristokraten": "aristocrates du dividende américains", "dax": "DAX",
            "eurostoxx": "EURO STOXX~50"}
DIV_SCHOCK_GRID = [-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]
RENTENJAHRE_GRID = [30, 40, 50]
# Tache 10, chapitre 07: constantes nommees pour les deux litteraux deja utilises par
# kapitel_7 (duree de retraite du Monte Carlo de base, reserve publiee en macro), afin
# que les macros de redaction les lisent au lieu de les recopier. Valeurs inchangees.
RENTENJAHRE_BASIS = 40
PUFFER_JAHRE_MACRO = 2
# Tache 10, chapitre 08 (decision du controleur): test historique du retrait constant en
# termes reels (regle des TAUX_RETRAIT %), pour ces durees de retraite; noms de macros en
# toutes lettres (zahlwort). La duree longue doit egaler celle du rejeu du chapitre 7
# (verifie par inspect dans kapitel_8) pour la comparaison aux memes annees de depart.
RETRAIT_HIST_DUREES = {"Dreissig": 30, "Vierzig": 40}
# Tolerance flottante du test "le capital couvre encore le retrait de l'annee" (sans elle,
# 1 - 25 x 0,04 vaut -3e-17 et le 25e retrait d'un cas exact echouerait a tort).
TOL_RETRAIT = 1e-12
# Taux d'epargne (grille QUOTEN) du depart anticipe illustre par fig_bruecke_anticipee:
# celui de la grille du chapitre 5 qui donne le pont le plus long avec la repartition de
# reference (\AlterMaisonSiebzigFuenfzig), controle croise contre ZUS["alter"].
BRUECKE_QUOTE_NOM = "Fuenfzig"
# Ages de sortie cites en exemple au chapitre 8 (lus sur fig_rente_alter).
RENTE_AGES_EXEMPLE = (40, 50, 60)
# Premier age d'arret trace par fig_rente_alter (la figure va de cet age a l'age legal).
RENTE_AGE_MIN = 35
# Annee qui designe, dans fig_einbrueche, l'episode de la decennie d'inflation
# americaine (baisse REELLE du dividende pendant que les prix montaient), cite au
# chapitre 7 a cote de l'inflation de 2022. Lue comme "l'episode qui contient cette
# annee"; aucune valeur de baisse n'est recopiee.
EPISODE_INFLATION_ANNEE = 1974
# Fenetre du taux de capitalisation de markt_basis (TCAC 1950-2022), reprise par le
# chapitre 7 pour comparer la croissance du dividende sur la MEME fenetre.
FENETRE_MARCHE = (1950, 2022)
REFS = os.path.join(ROOT, "refs")


def _charger_portefeuille() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "portefeuille.csv"))


def _charger_shiller_komplett() -> pd.DataFrame:
    """Serie Shiller COMPLETE (1871-2022, pas la fenetre 1950-2022 de markt_basis):
    le chapitre 7 montre l'historique entier des crises, pas seulement la fenetre
    utilisee pour estimer le taux de capitalisation."""
    return histoire.jahresreihe(histoire.laden(os.path.join(REFS, "ie_data.xls")))


def _mc_jahre_sans_troncature(jahres_s: pd.DataFrame, sparplan: list, ziel: float,
                               rendite_div: float, cas_les_plus_exigeants: list,
                               n: int = 5000, depart: int = 85, pas: int = 25,
                               essais_max: int = 12):
    """
    --------------------------------------------------------------------------
    Purpose:
        Chercher automatiquement la plus petite longueur de trajectoire "jahre"
        telle que montecarlo.erfolg() ne tronque aucune trajectoire reussie
        (abgeschnitten == 0), pour TOUS les cas listes dans
        cas_les_plus_exigeants (chacun un dict de kwargs pour erfolg(), ex.
        {"rentenjahre": 50}). Decision du controleur (tache 9, brief): "jahre
        sized so abgeschnitten == 0 (assert or check and report)" - verifie
        ici par assert, jamais suppose. Le critere de troncature ne depend
        que de "erreicht + rentenjahre >= jahre" (voir montecarlo.erfolg()),
        donc seul rentenjahre fait varier la difficulte du cas; le choc
        div_schock_erstes_rentenjahr de montecarlo.erfolg() (utilise pour
        fig_mc_erfolg) est applique APRES ce critere et ne le modifie pas,
        donc n'a pas besoin d'etre recherche ici.

    Inputs:
        jahres_s (pandas.DataFrame): sortie de histoire.jahresreihe() (serie
            complete, pas la fenetre 1950-2022).
        sparplan (list[float]): plan d'epargne annuel reel.
        ziel (float): revenu reel annuel vise.
        rendite_div (float): rendement de dividende applique au capital.
        cas_les_plus_exigeants (list[dict]): chaque dict contient "rentenjahre"
            (int, defaut 40 si absent).
        n (int): nombre de trajectoires Monte Carlo, defaut 5000.
        depart, pas, essais_max: parametres de la recherche par paliers.

    Outputs:
        result (tuple): (jahre (int), pfade_de_base (numpy.ndarray) trajectoires
            non forcees, longueur jahre).
    --------------------------------------------------------------------------
    """
    jahre = depart
    for _ in range(essais_max):
        pf = mc.pfade(jahres_s, n, jahre)
        ok = True
        for cas in cas_les_plus_exigeants:
            e = mc.erfolg(pf, sparplan, ziel, rendite_div=rendite_div,
                          rentenjahre=cas.get("rentenjahre", 40))
            if e["abgeschnitten"] != 0:
                ok = False
        if ok:
            return jahre, pf
        jahre += pas
    raise RuntimeError(f"_mc_jahre_sans_troncature: abgeschnitten > 0 meme apres "
                       f"{essais_max} agrandissements (jahre final essaye: {jahre})")


def _inflation_annuelle_de(jahr: int) -> float:
    """Taux d'inflation allemande de l'annee `jahr`: serie sourcee Destatis
    2019-2025 si disponible, sinon repli sur inflation_ziel (aucune serie CPI
    mensuelle n'existe dans ce depot pour 2011-2018/2026, decision documentee
    dans le docstring du module)."""
    serie = h.wert("inflation_destatis_2019_2025")
    v = serie.get(str(jahr))
    return v if v is not None else h.wert("inflation_ziel")


def _deflateur_vers_2026(jahr: int) -> float:
    """Facteur multiplicatif pour exprimer un montant nominal de l'annee `jahr` en
    euros de 2026 (convention "reel" du reste du document): produit des inflations
    annuelles de jahr+1 a 2026 inclus."""
    facteur = 1.0
    for y in range(jahr + 1, 2027):
        facteur *= 1 + _inflation_annuelle_de(y)
    return facteur


def _dividendes_annuels(div: pd.DataFrame, ticker: str) -> pd.Series:
    """Dividende annuel nominal d'un ticker a partir de data/marktdaten_dividenden.csv
    (sommes mensuelles), annees completes (12 mois presents) seulement, la premiere et
    la derniere annee de cotation d'un titre etant typiquement incompletes."""
    g = div.groupby("jahr")[ticker].agg(["sum", "count"])
    return g.loc[g["count"] == 12, "sum"]


def _plus_longue_hausse(div: pd.DataFrame, tickers: list) -> tuple:
    """Ticker (parmi `tickers`) dont le dividende annuel a la plus longue serie
    d'annees CONSECUTIVES strictement croissantes; retourne (ticker, longueur, serie)."""
    meilleur = (None, 0, None)
    for t in tickers:
        s = _dividendes_annuels(div, t)
        if len(s) < 2:
            continue
        vals = s.to_numpy()
        run = best = 1
        for i in range(1, len(vals)):
            run = run + 1 if vals[i] > vals[i - 1] else 1
            best = max(best, run)
        if best > meilleur[1]:
            meilleur = (t, best, s)
    return meilleur


def _plus_forte_baisse(div: pd.DataFrame, tickers: list) -> tuple:
    """Ticker (parmi `tickers`) avec la plus forte baisse relative du dividende annuel
    d'une annee sur l'autre; retourne (ticker, annee_de_la_baisse, serie)."""
    pire = (None, None, 0.0, None)
    for t in tickers:
        s = _dividendes_annuels(div, t)
        if len(s) < 2:
            continue
        pct = s.pct_change().dropna()
        if pct.empty:
            continue
        annee_min = pct.idxmin()
        if pct.loc[annee_min] < pire[2]:
            pire = (t, annee_min, pct.loc[annee_min], s)
    return pire


def _nom_court(nom: str) -> str:
    """Nom de societe coupe a la premiere virgule ("McCormick & Company, Incorporat" ->
    "McCormick & Company"): retire une forme juridique, eventuellement tronquee par le
    scraping (LUECKEN.md, tache 9 partie B), sans rien completer de memoire."""
    return str(nom).split(",")[0].strip()


def _regles_selection() -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lire les seuils de la selection du portefeuille-exemple DANS
        scripts/portefeuille_exemple.py (source unique), pour que le chapitre 6
        les cite par macro sans les recopier. Les valeurs par defaut de
        auswahl()/_waehle_mit_relaxation() viennent de leur signature; les
        seuils ecrits dans le corps des fonctions (bande de rendement, seuil de
        baisse, couverture FCF, relachements...) sont extraits du code source
        par expression reguliere. Chaque motif doit donner UNE seule valeur
        distincte, sinon ValueError: un changement du script de selection qui
        casserait l'extraction echoue fort au lieu de publier un seuil faux.

    Inputs:
        Neant (lit le module portefeuille_exemple).

    Outputs:
        result (dict): rendement_min/max, baisse_seuil, annees_croissance,
            dividendes_min, historique_ans, info_max, fcf_min, cagr_min,
            cagr_relache, payout_max, payout_relache, sektor_max, n_vise.
    --------------------------------------------------------------------------
    """
    import portefeuille_exemple as pe
    src = inspect.getsource(pe)

    def unique(motif: str, conv=float):
        vals = {conv(v) for v in re.findall(motif, src)}
        if len(vals) != 1:
            raise ValueError(f"_regles_selection: motif {motif!r} -> {sorted(vals)}")
        return vals.pop()

    sig = inspect.signature(pe.auswahl).parameters
    sig_w = inspect.signature(pe._waehle_mit_relaxation).parameters
    regles = {
        "rendement_min": unique(r"between\((0\.\d+), 0\.\d+\)"),
        "rendement_max": unique(r"between\(0\.\d+, (0\.\d+)\)"),
        "baisse_seuil": unique(r"pct_change\(\) < -(0\.\d+)"),
        "annees_croissance": unique(r"\*\* \(1 / (\d+)\)", int),
        "dividendes_min": unique(r"len\(div\) < (\d+)", int),
        "historique_ans": unique(r'period="(\d+)y"', int),
        "info_max": unique(r"\.head\((\d+)\)", int),
        "fcf_min": unique(r'fcf_deckung"\] >= (\d+\.\d+)'),
        "cagr_relache": unique(r'\{"cagr_min": (0\.\d+)\}'),
        "payout_relache": unique(r'\{"payout_max": (0\.\d+)\}'),
        "cagr_min": float(sig["cagr_min"].default),
        "payout_max": float(sig["payout_max"].default),
        "sektor_max": int(sig["sektor_max"].default),
        "n_vise": int(sig_w["n"].default),
    }
    n_appel = unique(r"_waehle_mit_relaxation\(df, n=(\d+)\)", int)
    if n_appel != regles["n_vise"]:
        raise ValueError(f"_regles_selection: n appele ({n_appel}) != n par defaut "
                         f"({regles['n_vise']})")
    return regles


def _entonnoir_selection(regles: dict, meta: dict, univers: pd.DataFrame,
                         portefeuille: pd.DataFrame) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Reconstituer hors ligne l'entonnoir de selection du portefeuille-exemple
        a partir des donnees de marche deja enregistrees (marktdaten_*.csv,
        aucun appel yfinance), avec la fonction kennzahlen() du script de
        selection lui-meme, et le verifier contre les compteurs de
        portefeuille_meta.json (ValueError en cas d'ecart: jamais publie sans
        concordance). Sert a dire, au chapitre 6, combien des titres ecartes au
        dernier filtre avaient pourtant une croissance suffisante: les criteres
        payout, couverture FCF et secteur (lus dans .info) ne sont pas conserves
        pour les titres ecartes, donc ce calcul ne peut pas dire lequel a joue.

    Inputs:
        regles (dict): sortie de _regles_selection().
        meta (dict): data/portefeuille_meta.json.
        univers (pandas.DataFrame): data/universum.csv.
        portefeuille (pandas.DataFrame): data/portefeuille.csv.

    Outputs:
        result (dict): ecartes (int, titres mesures non retenus),
            croissance_ok (int, dont croissance >= seuil strict),
            sans_recul (int, dont croissance sur dix ans non calculable).
    --------------------------------------------------------------------------
    """
    import portefeuille_exemple as pe
    div = pd.read_csv(os.path.join(DATA, "marktdaten_dividenden.csv"),
                      parse_dates=["datum"]).set_index("datum")
    kurse = pd.read_csv(os.path.join(DATA, "marktdaten_kurse.csv"),
                        parse_dates=["datum"]).set_index("datum")
    lignes = []
    for t in univers["ticker"]:
        if t not in div.columns or t not in kurse.columns:
            continue
        d = div[t][div[t] > 0]
        if len(d) < regles["dividendes_min"]:
            continue
        lignes.append({"ticker": t, **pe.kennzahlen(d, kurse[t])})
    mesures = pd.DataFrame(lignes)
    if len(mesures) != meta["mesures_brutes_n"]:
        raise ValueError(f"_entonnoir_selection: {len(mesures)} titres mesures, "
                         f"meta dit {meta['mesures_brutes_n']}")
    vor = mesures[mesures["rendite_ttm"].between(regles["rendement_min"], regles["rendement_max"])
                  & (mesures["kuerzungen_10j"] == 0)]
    vor = vor.sort_values("div_cagr_10j", ascending=False).head(regles["info_max"])
    if len(vor) - meta["manques_info"] != meta["mesures_n"]:
        raise ValueError(f"_entonnoir_selection: {len(vor)} titres apres filtre, "
                         f"meta dit {meta['mesures_n']} (+{meta['manques_info']} sans .info)")
    choisis = set(portefeuille["ticker"])
    if not choisis <= set(vor["ticker"]):
        raise ValueError("_entonnoir_selection: titres retenus hors de l'entonnoir reconstitue")
    ecartes = vor[~vor["ticker"].isin(choisis)]
    cagr = pd.to_numeric(ecartes["div_cagr_10j"], errors="coerce")
    return {"ecartes": len(ecartes), "croissance_ok": int((cagr >= regles["cagr_min"]).sum()),
            "sans_recul": int(cagr.isna().sum())}


def _bornes_plus_longue_hausse(s: pd.Series) -> tuple:
    """(annee de debut, annee de fin, nombre de hausses) de la premiere plus longue serie
    d'annees consecutives a dividende strictement croissant (meme regle que
    _plus_longue_hausse, qui ne rend que la longueur)."""
    vals, annees = s.to_numpy(), list(s.index)
    best, fin, run = 1, 0, 1
    for i in range(1, len(vals)):
        run = run + 1 if vals[i] > vals[i - 1] else 1
        if run > best:
            best, fin = run, i
    return int(annees[fin - best + 1]), int(annees[fin]), best - 1


def _macros_redaction_kapitel_6(markt: p.Markt, portefeuille: pd.DataFrame, lignes_auft: list,
                                lignes_n: list, kurse: pd.DataFrame, netto_w8: list,
                                t_hausse: str, s_hausse: pd.Series, s_hausse_reel: dict,
                                t_baisse: str, s_baisse: pd.Series, s_baisse_reel: dict,
                                univers: pd.DataFrame) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Macros de redaction du chapitre 6 (tache 10), toutes lues dans les CSV
        et objets deja produits par kapitel_6, dans portefeuille_meta.json ou
        dans le script de selection (_regles_selection): regles chiffrees et
        relachements, entonnoir de selection, secteurs, rendement contre
        croissance, diversification (fig_anzahl_titel), comparaison des
        repartitions a 20 % (fig_aufteilungen; le 70/30 reutilise les macros
        \\KapitalHorizonZwanzig/\\RevenuHorizonZwanzig du chapitre 5 apres
        controle d'egalite, jamais un second chiffre), et historique des deux
        dividendes (fig_div_beispiele). Aucune valeur de la litterature.

    Inputs:
        markt (projection.Markt): hypotheses de marche (croissance du modele).
        portefeuille (pandas.DataFrame): data/portefeuille.csv.
        lignes_auft, lignes_n (list[dict]): lignes de fig_aufteilungen et
            fig_anzahl_titel.
        kurse (pandas.DataFrame): data/marktdaten_kurse.csv (colonne datum).
        netto_w8 (list[float]): rendement net par titre, W-8BEN suppose.
        t_hausse, s_hausse, s_hausse_reel: ticker, dividendes annuels nominaux
            et reels du titre en plus longue hausse.
        t_baisse, s_baisse, s_baisse_reel: idem pour le titre en plus forte baisse.
        univers (pandas.DataFrame): data/universum.csv.

    Outputs:
        None. Remplit le dictionnaire de macros M.
    --------------------------------------------------------------------------
    """
    with open(os.path.join(DATA, "portefeuille_meta.json"), encoding="utf-8") as fi:
        meta = json.load(fi)
    regles = _regles_selection()
    n = len(portefeuille)
    if meta["choisis_n"] != n or meta["univers_n"] != len(univers):
        raise ValueError("_macros_redaction_kapitel_6: portefeuille_meta.json ne concorde pas")

    # --- Regles chiffrees et relachements (portefeuille_exemple.py) ---
    M["RegleRendementMin"] = fmt(regles["rendement_min"] * 100)
    M["RegleRendementMax"] = fmt(regles["rendement_max"] * 100)
    M["RegleCroissanceMin"] = fmt(regles["cagr_min"] * 100)
    M["RegleCroissanceRelache"] = fmt(regles["cagr_relache"] * 100)
    M["ReglePayoutMax"] = fmt(regles["payout_max"] * 100)
    M["ReglePayoutRelache"] = fmt(regles["payout_relache"] * 100)
    M["RegleCouvertureFcf"] = fmt(regles["fcf_min"], 1)
    M["RegleSecteurMax"] = fmt(regles["sektor_max"])
    M["RegleBaisseSeuil"] = fmt(regles["baisse_seuil"] * 100)
    M["RegleAnneesCroissance"] = fmt(regles["annees_croissance"])
    M["RegleDividendesMin"] = fmt(regles["dividendes_min"])
    M["RegleHistoriqueAns"] = fmt(regles["historique_ans"])
    M["RegleInfoMax"] = fmt(regles["info_max"])
    M["NombreTitresVise"] = fmt(regles["n_vise"])
    M["NombreTitresManquants"] = fmt(regles["n_vise"] - n)
    M["DateDonneesPortefeuille"] = _date_fr(meta["abgerufen"])

    # --- Entonnoir: univers, historique suffisant, filtre rendement/baisse, selection ---
    par_indice = univers["index"].value_counts()
    M["UniversTitres"] = fmt(meta["univers_n"])
    M["UniversAristocrates"] = fmt(int(par_indice.get("aristokraten", 0)))
    M["UniversDax"] = fmt(int(par_indice.get("dax", 0)))
    M["UniversEurostoxx"] = fmt(int(par_indice.get("eurostoxx", 0)))
    M["UniversPays"] = fmt(univers["land"].nunique())
    M["PortefeuillePays"] = fmt(portefeuille["land"].nunique())
    M["UniversHistorique"] = fmt(meta["mesures_brutes_n"])
    M["UniversSansHistorique"] = fmt(meta["univers_n"] - meta["mesures_brutes_n"])
    M["UniversFiltreRendement"] = fmt(meta["mesures_n"])
    ent = _entonnoir_selection(regles, meta, univers, portefeuille)
    M["TitresEcartesFin"] = fmt(ent["ecartes"])
    M["TitresEcartesCroissanceOk"] = fmt(ent["croissance_ok"])
    M["TitresEcartesSansRecul"] = fmt(ent["sans_recul"])
    essais = meta["paliers_essayes"]
    if len(essais) != 3:
        raise ValueError(f"_macros_redaction_kapitel_6: {len(essais)} paliers, 3 attendus")
    M["PalierStrictTitres"] = fmt(essais[0][1])
    M["PalierCroissanceTitres"] = fmt(essais[1][1])
    M["PalierPayoutTitres"] = fmt(essais[2][1])

    # --- Secteurs (parts equiponderees) ---
    comptes = portefeuille["sektor"].value_counts()
    au_plafond = [SEKTOR_FR.get(s, s) for s in comptes.index if comptes[s] >= regles["sektor_max"]]
    M["SecteursNombre"] = fmt(len(comptes))
    M["SecteursAuPlafond"] = fmt(len(au_plafond))
    noms = [f"\\enquote{{{escapieren(s)}}}" for s in au_plafond]
    M["SecteursAuPlafondNoms"] = (", ".join(noms[:-1]) + " et " + noms[-1]) if len(noms) > 1 \
        else (noms[0] if noms else "aucun")
    M["SecteurPartMax"] = fmt(comptes.max() / n * 100, 1)
    M["SecteurPartMin"] = fmt(comptes.min() / n * 100, 1)
    M["PoidsParTitre"] = fmt(100 / n, 1)

    # --- Rendement contre croissance du dividende ---
    r, g = portefeuille["rendite_ttm"], portefeuille["div_cagr_10j"]
    us = portefeuille["land"] == "US"
    M["RenditeTitreMin"] = fmt(r.min() * 100, 1)
    M["RenditeTitreMinTicker"] = escapieren(portefeuille.loc[r.idxmin(), "ticker"])
    M["RenditeTitreMax"] = fmt(r.max() * 100, 1)
    M["RenditeTitreMaxTicker"] = escapieren(portefeuille.loc[r.idxmax(), "ticker"])
    M["CroissanceTitreMin"] = fmt(g.min() * 100, 1)
    M["CroissanceTitreMinTicker"] = escapieren(portefeuille.loc[g.idxmin(), "ticker"])
    M["CroissanceTitreMax"] = fmt(g.max() * 100, 1)
    M["CroissanceTitreMaxTicker"] = escapieren(portefeuille.loc[g.idxmax(), "ticker"])
    M["CroissanceMediane"] = fmt(g.median() * 100, 1)
    M["RenditeMedianeUs"] = fmt(r[us].median() * 100, 1)
    M["RenditeMedianeEurope"] = fmt(r[~us].median() * 100, 1)
    M["CroissanceMedianeUs"] = fmt(g[us].median() * 100, 1)
    M["CroissanceMedianeEurope"] = fmt(g[~us].median() * 100, 1)
    M["RenditeNetteMediane"] = fmt(float(np.median(netto_w8)) * 100, 1)
    M["PayoutMedian"] = fmt(portefeuille["payout"].median() * 100)
    M["CorrelationRendementCroissance"] = fmt(float(r.corr(g)), 2)
    M["CroissanceModeleMaison"] = fmt(markt.wachstum_maison * 100, 1)

    # --- Diversification (fig_anzahl_titel) ---
    risques = [l["risiko"] for l in lignes_n]
    r1, r_n = risques[0], risques[-1]
    seuil = r_n + 0.1 * (r1 - r_n)
    M["RisqueUnTitre"] = fmt(r1 * 100, 1)
    M["RisqueTousTitres"] = fmt(r_n * 100, 1)
    M["RisqueReductionProzent"] = fmt((1 - r_n / r1) * 100)
    M["TitresNeufDixiemes"] = fmt(next(l["n"] for l in lignes_n if l["risiko"] <= seuil))
    M["TiragesDiversification"] = fmt(TIRAGES_ANZAHL_TITEL)
    dates = pd.to_datetime(kurse["datum"])
    M["DonneesDebut"] = str(dates.min().year)
    M["DonneesFin"] = str(dates.max().year)

    # --- Repartitions a 20 % (fig_aufteilungen); 70/30 = macros du chapitre 5 ---
    auft = {l["strategie"]: l for l in lignes_auft}
    if (fmt(auft["MaisonSiebzig"]["kapital_xlv"]) != M["KapitalHorizonZwanzig"]
            or fmt(auft["MaisonSiebzig"]["netto_monat_xlv"]) != M["RevenuHorizonZwanzig"]):
        raise ValueError("fig_aufteilungen (70/30 a 20 %) != \\KapitalHorizonZwanzig/"
                         "\\RevenuHorizonZwanzig du chapitre 5")
    for nom in ("MaisonFuenfzig", "MaisonHundert", "EtfReferenz", "EtfUmschichtung"):
        M["KapitalAufteilung" + nom] = fmt(auft[nom]["kapital_xlv"])
        M["RevenuAufteilung" + nom] = fmt(auft[nom]["netto_monat_xlv"])
    for nom in ("MaisonFuenfzig", "MaisonHundert"):
        M["RevenuAufteilung" + nom + "PartCible"] = fmt(
            auft[nom]["netto_monat_xlv"] / ZIEL_REAL_MONAT * 100)
    # Cible mensuelle en nombre brut (sans separateur): lisible par pgfmath, pour tracer la
    # ligne de l'objectif dans la figure des revenus (extra y ticks), jamais pour la prose.
    M["ZielMonatNombre"] = str(int(round(ZIEL_REAL_MONAT)))
    M["RevenuAufteilungEtfReferenzMultiple"] = fmt(
        auft["EtfReferenz"]["netto_monat_xlv"] / ZIEL_REAL_MONAT, 1)
    M["KapitalAufteilungEcartProzent"] = fmt(
        (auft["MaisonFuenfzig"]["kapital_xlv"] / auft["MaisonHundert"]["kapital_xlv"] - 1) * 100, 1)
    M["RevenuAufteilungEcartProzent"] = fmt(
        (auft["MaisonHundert"]["netto_monat_xlv"] / auft["MaisonFuenfzig"]["netto_monat_xlv"] - 1)
        * 100, 1)

    # --- Titre en plus longue hausse (parmi les retenus) ---
    debut, fin, n_hausses = _bornes_plus_longue_hausse(s_hausse)
    reel = pd.Series(s_hausse_reel).sort_index()
    var_reel = reel.pct_change().loc[debut + 1:fin]
    annee_min = int(var_reel.idxmin())
    M["BeispielHausseTicker"] = escapieren(t_hausse)
    M["BeispielHausseHausses"] = fmt(n_hausses)
    M["BeispielHausseDebut"] = str(debut)
    M["BeispielHausseFin"] = str(fin)
    M["BeispielHausseReelTotal"] = fmt((reel[fin] / reel[debut] - 1) * 100)
    M["BeispielHausseReelMin"] = fmt(var_reel.min() * 100, 1)
    M["BeispielHausseReelMinAnnee"] = str(annee_min)
    M["BeispielHausseNominalMinAnnee"] = fmt(s_hausse.pct_change().loc[annee_min] * 100, 1)
    M["BeispielHausseInflationMinAnnee"] = fmt(_inflation_annuelle_de(annee_min) * 100, 1)

    # --- Titre en plus forte baisse (tout l'univers) ---
    pct = s_baisse.pct_change()
    annee_b = int(pct.idxmin())
    baisses = pct[pct < -regles["baisse_seuil"]]
    autres = baisses.drop(annee_b)
    if autres.empty:
        raise ValueError("_macros_redaction_kapitel_6: le titre en plus forte baisse n'a pas "
                         "d'autre baisse; le texte du chapitre 6 en suppose une")
    annees_b = list(s_baisse.index)
    avant = int(annees_b[annees_b.index(annee_b) - 1])
    sans_baisse = 0
    for y in reversed(annees_b[1:annees_b.index(annee_b)]):
        if pct.loc[y] < -regles["baisse_seuil"]:
            break
        sans_baisse += 1
    reel_b = pd.Series(s_baisse_reel).sort_index()
    derniere = int(reel_b.index.max())
    ecart = reel_b[derniere] / reel_b[avant] - 1
    indice = univers.set_index("ticker")["index"].get(t_baisse)
    M["BeispielBaisseAnnee"] = str(annee_b)
    M["BeispielBaisseProzent"] = fmt(-pct.min() * 100)
    M["BeispielBaisseNombreBaisses"] = fmt(len(baisses))
    M["BeispielBaisseAutreAnnee"] = str(int(autres.idxmin()))
    M["BeispielBaisseAutreProzent"] = fmt(-autres.min() * 100)
    M["BeispielBaisseAnneesSansBaisse"] = fmt(sans_baisse)
    M["BeispielBaisseAnneeAvant"] = str(avant)
    M["BeispielBaisseDerniereAnnee"] = str(derniere)
    M["BeispielBaisseEcartFinProzent"] = fmt(abs(ecart) * 100)
    M["BeispielBaisseEcartFinSens"] = "sous" if ecart < 0 else "au-dessus de"
    M["BeispielBaisseIndice"] = INDEX_FR.get(indice, escapieren(indice))
    M["BeispielBaisseStatut"] = ("fait partie" if t_baisse in set(portefeuille["ticker"])
                                 else "ne fait pas partie")


def kapitel_5_ziel_niedrig():
    """
    --------------------------------------------------------------------------
    Purpose:
        Variante basse (demande explicite de l'utilisateur, tache 11,
        01.10.2026): tout ce que ce bloc produit concerne la cible de
        ZIEL_NIEDRIG_MONAT EUR nets par mois, jamais la reference du document
        (ZIEL_REAL_MONAT, 3500 EUR), qui reste inchangee. Reprend exactement
        la logique de kapitel_5 (5 strategies x 4 taux QUOTEN, capital de
        depart "Mitte", projection.simulieren, "non atteint" si ziel_jahr est
        None) et son epargne mensuelle de la premiere annee
        (sparplan_aus_quote, independante de la cible, memes valeurs que
        M["EpargneMois<Quote>"] deja ecrites par kapitel_5). Ecrit
        data/tab_ziel_niedrig.tex (corps de tableau, meme convention que
        data/tab_portefeuille.tex) et data/fig_alter_ziel_vergleich.csv (age
        d'atteinte par taux d'epargne, 70/30 et ETF regle des 4 %, les deux
        cibles cote a cote, grille de 5 a 70 % par pas de 5 points comme
        fig_jahre_sparquote). Doit s'executer APRES kapitel_5 (lit
        M["EpargneMois<Quote>"]).

    Inputs:
        Neant (lit M, QUOTEN, STRATEGIEN, STARTKAPITAL, markt_basis(), _sim_kwargs()).

    Outputs:
        None. Remplit M (ZielNiedrigMonat, Alter<Strategie><Quote>Niedrig,
        KapitalNiedrigMaisonSiebzig, KapitalNiedrigEtf); ecrit
        data/tab_ziel_niedrig.tex et data/fig_alter_ziel_vergleich.csv.
    --------------------------------------------------------------------------
    """
    markt = markt_basis()
    sim_kw = _sim_kwargs()

    M["ZielNiedrigMonat"] = fmt(ZIEL_NIEDRIG_MONAT)

    # --- Age d'atteinte par strategie x taux (meme grille que ZUS["alter"] de kapitel_5,
    # cible ZIEL_NIEDRIG_MONAT au lieu de ZIEL_REAL_MONAT) ---
    alter_niedrig = {}
    for strat in STRATEGIEN:
        alter_niedrig[strat.name] = {}
        for quote_nom, quote_val in QUOTEN.items():
            plan = p.sparplan_aus_quote(quote_val, jahre=JAHRE_HORIZONT)
            s = p.simulieren(strat, markt, plan, STARTKAPITAL["Mitte"], ZIEL_NIEDRIG_MONAT,
                              JAHRE_HORIZONT, **sim_kw)
            alter = None if s["ziel_jahr"] is None else s["ziel_jahr"] - GEBURTSJAHR
            alter_niedrig[strat.name][quote_nom] = alter
            M[f"Alter{strat.name}{quote_nom}Niedrig"] = "non atteint" if alter is None else fmt(alter)

    # Les deux ages cites dans la prose (70/30 et ETF regle des 4 %, a 30 % d'epargne) sont
    # deja ecrits par la boucle ci-dessus: M["AlterMaisonSiebzigDreissigNiedrig"],
    # M["AlterEtfReferenzDreissigNiedrig"].

    # --- Capital necessaire pour ZIEL_NIEDRIG_MONAT (70/30 dividendes et ETF regle 4 %) ---
    strat_70 = next(s for s in STRATEGIEN if s.name == "MaisonSiebzig")
    kv_teilfrei = bool(h.wert("kv_teilfreistellung_etf"))
    M["KapitalNiedrigMaisonSiebzig"] = fmt(_kapital_pour_ziel(
        strat_70.anteil_maison, markt.rendite_div_maison, markt, sim_kw["saetze"],
        sim_kw["sq"], sim_kw["pauschbetrag_nominal"], kv_teilfrei, ziel_jahr=ZIEL_NIEDRIG_JAHR))
    M["KapitalNiedrigEtf"] = fmt(_kapital_etf_regle4(
        markt, sim_kw["saetze"], sim_kw["sq"], sim_kw["pauschbetrag_nominal"], kv_teilfrei,
        ziel_jahr=ZIEL_NIEDRIG_JAHR))

    # --- data/tab_ziel_niedrig.tex: corps de tableau, lignes = strategies, colonnes = les
    # 4 taux d'epargne, derniere ligne = epargne mensuelle de la premiere annee. ---
    strats_label = [("MaisonFuenfzig", "Moiti\\'e en maison"), ("MaisonSiebzig", "70~\\% en maison"),
                     ("MaisonHundert", "Tout en maison"),
                     ("EtfReferenz", f"ETF, retrait {fmt(TAUX_RETRAIT * 100)}~\\%"),
                     ("EtfUmschichtung", "ETF distribuant")]
    quotes_ordre = ("Zehn", "Zwanzig", "Dreissig", "Fuenfzig")
    lignes_tab = []
    for nom, label in strats_label:
        cellules = [("non atteint" if alter_niedrig[nom][q] is None else fmt(alter_niedrig[nom][q]))
                    for q in quotes_ordre]
        lignes_tab.append(f"{label} & " + " & ".join(cellules) + " \\\\")
    epargne_cellules = [f"{M[f'EpargneMois{q}']}~€" for q in quotes_ordre]
    lignes_tab.append("\\'Epargne mensuelle, 1\\textsuperscript{re} ann\\'ee & "
                      + " & ".join(epargne_cellules) + " \\\\")
    with open(os.path.join(DATA, "tab_ziel_niedrig.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (tab_ziel_niedrig) - "
                 "ne pas modifier a la main\n")
        fo.write("\n".join(lignes_tab) + "\n")

    # --- data/fig_alter_ziel_vergleich.csv: age d'atteinte par taux d'epargne, 70/30 et
    # ETF regle 4 %, les deux cibles (3500 et 2500 EUR/mois), grille de 5 a 70 points par
    # 5 comme fig_jahre_sparquote. ---
    strat_etf = next(s for s in STRATEGIEN if s.name == "EtfReferenz")

    def age_a(strat, ziel_monat, quote):
        plan = p.sparplan_aus_quote(quote, jahre=JAHRE_HORIZONT)
        s = p.simulieren(strat, markt, plan, STARTKAPITAL["Mitte"], ziel_monat,
                          JAHRE_HORIZONT, **sim_kw)
        return None if s["ziel_jahr"] is None else s["ziel_jahr"] - GEBURTSJAHR

    lignes_cmp = []
    for pct in range(5, 71, 5):
        quote = pct / 100
        lignes_cmp.append({
            "quote": quote,
            "maison70_ziel_haut": age_a(strat_70, ZIEL_REAL_MONAT, quote),
            "maison70_ziel_bas": age_a(strat_70, ZIEL_NIEDRIG_MONAT, quote),
            "etf_ziel_haut": age_a(strat_etf, ZIEL_REAL_MONAT, quote),
            "etf_ziel_bas": age_a(strat_etf, ZIEL_NIEDRIG_MONAT, quote),
        })
    schreiben("fig_alter_ziel_vergleich", pd.DataFrame(lignes_cmp))


def kapitel_6():
    """fig_aufteilungen, fig_anzahl_titel, fig_sektoren, fig_laender,
    fig_rendite_wachstum, fig_div_beispiele, data/tab_portefeuille.tex.

    Decision 1 du controleur: partout ou le brief dit "30 titres", nombre reel de
    lignes de data/portefeuille.csv (23 ici); fig_anzahl_titel va de 1 a ce nombre.
    """
    markt = markt_basis()
    sim_kw = _sim_kwargs()
    portefeuille = _charger_portefeuille()
    n_titres = len(portefeuille)
    M["NombreTitres"] = fmt(n_titres)

    # --- fig_aufteilungen: capital et revenu net mensuel a l'annee 45 (XLV) du plan,
    # quote 20 %, capital de depart "Mitte", pour les 5 strategies (decision documentee
    # dans le docstring du module: le brief ne fixe pas l'annee de comparaison).
    # Quote laissee a 20 % apres le correctif du scenario de reference (tache 10, voir
    # docstring du module, point 3): comparaison a 5 STRATEGIES et taux commun, distincte
    # du scenario de reference (une seule strategie, MaisonSiebzig) et sans macro
    # *Basis/*Yann en aval, donc hors mandat de ce correctif. ---
    plan_20 = p.sparplan_aus_quote(QUOTEN["Zwanzig"], jahre=JAHRE_HORIZONT)
    lignes_auft = []
    for strat in STRATEGIEN:
        s = p.simulieren(strat, markt, plan_20, STARTKAPITAL["Mitte"], ZIEL_REAL_MONAT,
                          JAHRE_HORIZONT, **sim_kw)
        ziel_alter = (s["ziel_jahr"] - GEBURTSJAHR) if s["ziel_jahr"] is not None else None
        lignes_auft.append({"strategie": strat.name, "kapital_xlv": s["wert"][-1],
                            "netto_monat_xlv": s["einkommen_netto_monat_real"][-1],
                            "ziel_alter": ziel_alter})
    schreiben("fig_aufteilungen", pd.DataFrame(lignes_auft))

    # --- fig_anzahl_titel: ecart-type annualise d'un portefeuille equipondere de n
    # titres tires sans remise parmi les n_titres retenus, rendements MENSUELS DE PRIX
    # (data/marktdaten_kurse.csv, aucun nouvel appel yfinance), 500 tirages par n,
    # graine fixe. Aucune valeur de la litterature recopiee (decision 1). ---
    tickers = list(portefeuille["ticker"])
    kurse = pd.read_csv(os.path.join(DATA, "marktdaten_kurse.csv"))
    rendements = kurse[tickers].pct_change().dropna()
    rng = np.random.default_rng(SEED_ANZAHL_TITEL)
    lignes_n = []
    for n in range(1, n_titres + 1):
        ecarts = []
        for _ in range(TIRAGES_ANZAHL_TITEL):
            choix = rng.choice(tickers, size=n, replace=False)
            port_ret = rendements[list(choix)].mean(axis=1)
            ecarts.append(port_ret.std() * (12 ** 0.5))
        lignes_n.append({"n": n, "risiko": float(np.mean(ecarts))})
    schreiben("fig_anzahl_titel", pd.DataFrame(lignes_n))

    # --- fig_sektoren, fig_laender: parts equiponderees (le modele traite les 23
    # titres comme une seule poche "maison", cf. projection.py: pas de ponderation
    # par score/capitalisation dans ce projet). ---
    # Colonne "libelle" (tache 10, chapitre 6): libelle francais de la categorie
    # (SEKTOR_FR/LAND_FR), repli sur le libelle d'origine; schreiben() en tire
    # "libelle_disp" echappe, lu par \balkenfigur{...}{libelle}{anteil}.
    sekt = (portefeuille.groupby("sektor").size() / n_titres).reset_index()
    sekt.columns = ["kategorie", "anteil"]
    sekt["libelle"] = [SEKTOR_FR.get(k, k) for k in sekt["kategorie"]]
    schreiben("fig_sektoren", sekt)
    land = (portefeuille.groupby("land").size() / n_titres).reset_index()
    land.columns = ["kategorie", "anteil"]
    land["libelle"] = [LAND_FR.get(k, k) for k in land["kategorie"]]
    schreiben("fig_laender", land)

    # --- data/tab_portefeuille.tex: ticker, pays, secteur, rendement, rendement net
    # DE, croissance 10 ans, payout (colonnes du brief, texte scrape echappe).
    # Tache 10 (chapitre 6): secteur en francais (SEKTOR_FR); rendement net RECALCULE
    # avec _sq_de_base() (W-8BEN suppose depose, ecart au plan n. 2 du document), au lieu
    # de la colonne rendite_netto_de de portefeuille.csv, calculee par
    # portefeuille_exemple.py avec la retenue americaine SANS formulaire (30 %): sans ce
    # recalcul, le tableau contredirait le chapitre 3 (\NettoUs) pour les titres americains.
    # portefeuille.csv (livrable des taches 7/8) n'est pas modifie. ---
    sq_base = _sq_de_base()
    netto_w8 = [f.posten_netto(r * 100, l, 0.0, sq=sq_base)[0] / 100
                for r, l in zip(portefeuille["rendite_ttm"], portefeuille["land"])]
    lignes_tab = []
    for (_, r), net in zip(portefeuille.iterrows(), netto_w8):
        lignes_tab.append(
            f"{escapieren(r['ticker'])} & {escapieren(r['land'])} & "
            f"{escapieren(SEKTOR_FR.get(r['sektor'], r['sektor']))} & "
            f"{fmt(r['rendite_ttm'] * 100, 1)}\\% & {fmt(net * 100, 1)}\\% & "
            f"{fmt(r['div_cagr_10j'] * 100, 1)}\\% & {fmt(r['payout'] * 100, 1)}\\% \\\\")
    with open(os.path.join(DATA, "tab_portefeuille.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (tab_portefeuille) - "
                 "ne pas modifier a la main\n")
        fo.write("\n".join(lignes_tab) + "\n")

    # --- fig_rendite_wachstum: rendement vs croissance du dividende, un point par titre. ---
    # Colonne "land" ajoutee (tache 10, chapitre 6): classes de marqueurs par pays dans
    # le nuage de points (scatter/classes de pgfplots).
    rw = portefeuille[["ticker", "rendite_ttm", "div_cagr_10j", "land"]].copy()
    rw.columns = ["ticker", "rendite", "wachstum", "land"]
    schreiben("fig_rendite_wachstum", rw)

    # --- fig_div_beispiele: dividendes annuels REELS (euros de 2026) d'un titre du
    # portefeuille a la plus longue hausse continue et d'un titre de l'UNIVERS
    # (data/universum.csv) a la plus forte baisse. ---
    div = pd.read_csv(os.path.join(DATA, "marktdaten_dividenden.csv"))
    div["jahr"] = pd.to_datetime(div["datum"]).dt.year
    t_hausse, _, s_hausse = _plus_longue_hausse(div, tickers)
    univers = pd.read_csv(os.path.join(DATA, "universum.csv"))
    t_baisse, _, _, s_baisse = _plus_forte_baisse(div, list(univers["ticker"]))

    s_hausse_reel = {int(j): v * _deflateur_vers_2026(int(j)) for j, v in s_hausse.items()}
    s_baisse_reel = {int(j): v * _deflateur_vers_2026(int(j)) for j, v in s_baisse.items()}
    toutes_annees = sorted(set(s_hausse_reel) | set(s_baisse_reel))
    schreiben("fig_div_beispiele", pd.DataFrame([
        {"jahr": j, "titre_hausse": s_hausse_reel.get(j), "titre_baisse": s_baisse_reel.get(j)}
        for j in toutes_annees
    ]))
    nom_hausse = portefeuille.loc[portefeuille["ticker"] == t_hausse, "name"]
    # Nom court (tache 10, chapitre 6): coupe a la premiere virgule, ce qui retire la
    # forme juridique tronquee par le scraping ("McCormick & Company, Incorporat",
    # signale dans LUECKEN.md) sans rien inventer.
    M["BeispielHausse"] = (escapieren(_nom_court(nom_hausse.iloc[0])) if len(nom_hausse)
                           else escapieren(t_hausse))
    M["BeispielBaisse"] = escapieren(t_baisse)

    _macros_redaction_kapitel_6(markt, portefeuille, lignes_auft, lignes_n, kurse, netto_w8,
                                t_hausse, s_hausse, s_hausse_reel, t_baisse, s_baisse,
                                s_baisse_reel, univers)


def _kapital_requis_reference(markt: p.Markt) -> float:
    """Capital requis du chapitre 4 pour la repartition du scenario de reference
    (_reference_strategie), MEME appel _kapital_pour_ziel et memes arguments que
    kapitel_4 (aucun second modele): sert au chapitre 7 a placer ce capital, calcule
    avec impot et assurance maladie, face au seuil du Monte Carlo qui n'en a pas."""
    sim_kw = _sim_kwargs()
    return _kapital_pour_ziel(_reference_strategie().anteil_maison, markt.rendite_div_maison,
                              markt, sim_kw["saetze"], sim_kw["sq"], sim_kw["pauschbetrag_nominal"],
                              bool(h.wert("kv_teilfreistellung_etf")))


def _seuil_revenu_modeles() -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lire, dans le code de montecarlo.erfolg et de histoire.rueckspiel, la
        part de l'objectif en dessous de laquelle une annee de retraite compte
        comme un echec (motif "<x> * ziel_real_jahr"), au lieu de la recopier
        dans ce module (meme approche que _regles_selection pour le chapitre 6).
        Les deux modeles doivent employer une seule et meme valeur.

    Inputs:
        Neant (lit le source des deux fonctions via inspect).

    Outputs:
        result (float): la part commune (ex. 0.8).
    --------------------------------------------------------------------------
    """
    motif = re.compile(r"(\d+(?:\.\d+)?)\s*\*\s*ziel_real_jahr")
    valeurs = set()
    for fn in (mc.erfolg, histoire.rueckspiel):
        trouvees = {float(v) for v in motif.findall(inspect.getsource(fn))}
        if len(trouvees) != 1:
            raise ValueError(f"_seuil_revenu_modeles: {fn.__name__} -> {trouvees}")
        valeurs |= trouvees
    if len(valeurs) != 1:
        raise ValueError(f"_seuil_revenu_modeles: seuils differents {valeurs}")
    return valeurs.pop()


def _liste_fr(elements: list) -> str:
    """Liste francaise "a, b et c" (tache 10, chapitre 07)."""
    if len(elements) == 1:
        return elements[0]
    return ", ".join(elements[:-1]) + " et " + elements[-1]


def _macros_redaction_kapitel_7(markt: p.Markt, jahres_s: pd.DataFrame, einbrueche: list,
                                rj: pd.DataFrame, pfade_base: np.ndarray, jahre_mc: int,
                                plan_ref: list, e_base: dict, df_erfolg: pd.DataFrame,
                                lignes_infl: list, kapital_ref: float, seuil_mc: float) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Macros de redaction du chapitre 7 (tache 10), toutes lues dans les
        objets deja produits par kapitel_7 (serie Shiller, episodes de baisse,
        rejeu, trajectoires Monte Carlo de base, grille du choc, serie
        d'inflation) ou dans le code des modeles (seuil de revenu, longueur des
        blocs, nombre de trajectoires lus par inspect). Aucune valeur recopiee.
        Deux calculs complementaires seulement, sans nouveau modele:
        (1) la reussite Monte Carlo si l'objectif doit etre atteint DANS
        l'horizon, obtenue avec mc.erfolg sur les memes trajectoires coupees a
        JAHRE_HORIZONT + RENTENJAHRE_BASIS ans (une trajectoire qui atteint
        l'objectif plus tard y est "abgeschnitten", donc exclue des reussites);
        (2) pour chaque echec du rejeu, la premiere annee ou le revenu passe
        sous le seuil, pour verifier qu'elle tombe dans un episode de
        fig_einbrueche (ValueError sinon: la phrase du chapitre deviendrait
        fausse).

    Inputs:
        markt (projection.Markt): rendement du dividende maison (Monte Carlo).
        jahres_s (pandas.DataFrame): serie Shiller complete (jahresreihe).
        einbrueche (list[dict]): sortie de histoire.div_einbrueche.
        rj (pandas.DataFrame): sortie de histoire.rueckspiel (colonne ziel_jahr).
        pfade_base (numpy.ndarray): trajectoires Monte Carlo de base.
        jahre_mc (int): longueur des trajectoires.
        plan_ref (list[float]): plan d'epargne du scenario de reference.
        e_base (dict): mc.erfolg de base (sans reserve, RENTENJAHRE_BASIS).
        df_erfolg (pandas.DataFrame): contenu de fig_mc_erfolg.
        lignes_infl (list[dict]): lignes de fig_inflation_2022.
        kapital_ref (float): capital requis du chapitre 4 (reference).
        seuil_mc (float): capital a partir duquel le Monte Carlo juge
            l'objectif atteint (sans impot).

    Outputs:
        None. Remplit le dictionnaire de macros M.
    --------------------------------------------------------------------------
    """
    # --- Controle croise: le capital requis de la figure est celui du chapitre 4. ---
    if _reference_strategie().anteil_maison == 0.7 and "KapitalMaisonSiebzig" in M:
        if fmt(kapital_ref) != M["KapitalMaisonSiebzig"]:
            raise ValueError("_macros_redaction_kapitel_7: capital requis != \\KapitalMaisonSiebzig")
    M["KapitalRequisReference"] = fmt(kapital_ref)

    # --- Serie Shiller: bornes, croissance reelle du dividende, rendement total. ---
    d = jahres_s.set_index("jahr")["div_real"]
    a_debut, a_fin = int(d.index[0]), int(d.index[-1])
    M["ShillerDebut"], M["ShillerFin"] = str(a_debut), str(a_fin)
    M["ShillerDivMultiple"] = fmt(d.iloc[-1] / d.iloc[0], 1)
    M["ShillerDivCroissance"] = fmt(((d.iloc[-1] / d.iloc[0]) ** (1 / (a_fin - a_debut)) - 1) * 100, 1)
    f0, f1 = FENETRE_MARCHE
    M["FenetreMarcheDebut"], M["FenetreMarcheFin"] = str(f0), str(f1)
    croissance_fenetre = (d.loc[f1] / d.loc[f0]) ** (1 / (f1 - f0)) - 1
    M["ShillerDivCroissanceFenetre"] = fmt(croissance_fenetre * 100, 1)
    # Croissance reelle du dividende maison supposee par le modele (\CroissanceModeleMaison)
    # rapportee a celle du dividende de l'indice sur la meme fenetre.
    M["CroissanceModeleRapport"] = fmt(markt.wachstum_maison / croissance_fenetre, 1)
    M["ShillerDivCroissanceDerniere"] = fmt((d.iloc[-1] / d.iloc[-2] - 1) * 100, 1)
    # Premiere ligne de jahresreihe = artefact de bord (voir montecarlo.pfade), exclue.
    r_hist = jahres_s["rendite_real"].iloc[1:]
    M["McRenditeHistorique"] = fmt(((1 + r_hist).prod() ** (1 / len(r_hist)) - 1) * 100, 1)
    # Rendement du dividende de l'indice (dividende annuel moyen / cours moyen de l'annee),
    # aux deux bornes de la fenetre du taux de capitalisation.
    brut = histoire.laden(os.path.join(REFS, "ie_data.xls"))
    brut["jahr"] = brut["datum"].astype(int)
    moy = brut.groupby("jahr").agg(d=("d", "mean"), p=("p", "mean"))
    rendement_indice = moy["d"] / moy["p"]
    M["ShillerRendementDebutFenetre"] = fmt(rendement_indice.loc[f0] * 100, 1)
    M["ShillerRendementFinFenetre"] = fmt(rendement_indice.loc[f1] * 100, 1)

    # --- Episodes de baisse (fig_einbrueche) et annees passees sous le sommet. ---
    seuil_baisse = inspect.signature(histoire.div_einbrueche).parameters["schwelle"].default
    M["SeuilBaisseHistorique"] = fmt(abs(seuil_baisse) * 100)
    M["NombreBaisses"] = fmt(len(einbrueche))

    def retour(e):
        apres = d.loc[e["bis"]:]
        rattrape = apres[apres >= d.loc[e["von"]]]
        return int(rattrape.index[0]) if len(rattrape) else None

    pire = min(einbrueche, key=lambda e: e["rueckgang"])
    if retour(pire) is None:
        raise ValueError("_macros_redaction_kapitel_7: la pire baisse n'est jamais rattrapee")
    M["GroessterEinbruchDebut"] = str(pire["von"])
    M["GroessterEinbruchRetour"] = str(retour(pire))
    M["GroessterEinbruchSousSommetAns"] = fmt(retour(pire) - pire["von"])
    rattrapes = [e for e in einbrueche if retour(e) is not None]
    longue = max(rattrapes, key=lambda e: retour(e) - e["von"])
    M["BaisseLongue"] = fmt(abs(longue["rueckgang"]) * 100, 1)
    M["BaisseLongueDebut"], M["BaisseLongueFin"] = str(longue["von"]), str(longue["bis"])
    M["BaisseLongueRetour"] = str(retour(longue))
    M["BaisseLongueAns"] = fmt(retour(longue) - longue["von"])
    recent = einbrueche[-1]
    M["EinbruchRecent"] = fmt(abs(recent["rueckgang"]) * 100, 1)
    M["EinbruchRecentDebut"], M["EinbruchRecentFin"] = str(recent["von"]), str(recent["bis"])
    M["EinbruchRecentRetour"] = str(retour(recent)) if retour(recent) else "non rattrap\\'e"
    infl = [e for e in einbrueche if e["von"] <= EPISODE_INFLATION_ANNEE <= e["bis"]]
    if len(infl) != 1:
        raise ValueError("_macros_redaction_kapitel_7: episode d'inflation introuvable")
    M["EinbruchInflation"] = fmt(abs(infl[0]["rueckgang"]) * 100, 1)
    M["EinbruchInflationDebut"], M["EinbruchInflationFin"] = str(infl[0]["von"]), str(infl[0]["bis"])

    # --- Rejeu historique (fig_rueckspiel). ---
    seuil_rev = _seuil_revenu_modeles()
    if inspect.signature(histoire.rueckspiel).parameters["rentenjahre"].default != RENTENJAHRE_BASIS:
        raise ValueError("_macros_redaction_kapitel_7: duree de retraite du rejeu != RENTENJAHRE_BASIS")
    issue = rj["ueberlebt"].map(lambda v: "tenu" if v is True else ("echec" if v is False else None))
    atteints = rj[rj["jahre_bis_ziel"].notna()]
    juges, echecs = rj[issue.notna()], rj[issue == "echec"]
    n_tenus = int((issue == "tenu").sum())
    M["RueckspielDebuts"] = fmt(len(rj))
    M["RueckspielPremier"], M["RueckspielDernier"] = str(int(rj["start"].min())), str(int(rj["start"].max()))
    M["RueckspielAtteints"] = fmt(len(atteints))
    M["RueckspielDernierAtteint"] = str(int(atteints["start"].max()))
    M["RueckspielJuges"] = fmt(len(juges))
    M["RueckspielDernierJuge"] = str(int(juges["start"].max()))
    M["RueckspielTenus"] = fmt(n_tenus)
    M["RueckspielEchecs"] = fmt(len(echecs))
    M["RueckspielTenusProzent"] = fmt(n_tenus / len(juges) * 100, 1)
    M["RueckspielAnsMin"] = fmt(atteints["jahre_bis_ziel"].min())
    M["RueckspielAnsMedian"] = fmt(atteints["jahre_bis_ziel"].median())
    M["RueckspielAnsMax"] = fmt(atteints["jahre_bis_ziel"].max())
    wachstum = jahres_s.set_index("jahr")["div_wachstum_real"]
    episodes_echec = []
    for _, ligne in echecs.iterrows():
        z = int(ligne["ziel_jahr"])
        revenu = (1 + wachstum.loc[z + 1:z + RENTENJAHRE_BASIS]).cumprod()
        sous = revenu[revenu < seuil_rev]
        if sous.empty:
            raise ValueError(f"_macros_redaction_kapitel_7: echec {ligne['start']} sans annee sous le seuil")
        premiere = int(sous.index[0])
        ep = [e for e in einbrueche if e["von"] <= premiere <= e["bis"]]
        if not ep:
            raise ValueError(f"_macros_redaction_kapitel_7: echec {ligne['start']} hors episode de baisse")
        if ep[0] not in episodes_echec:
            episodes_echec.append(ep[0])
    episodes_echec.sort(key=lambda e: e["von"])
    M["RueckspielEchecsEpisodes"] = _liste_fr([f"{e['von']}--{e['bis']}" for e in episodes_echec])
    M["RueckspielEchecsNombreEpisodes"] = fmt(len(episodes_echec))

    # --- Monte Carlo: parametres lus dans le code, seuil de capital, dates d'objectif. ---
    n = pfade_base.shape[0]
    M["McTrajectoires"] = fmt(n)
    M["McBloc"] = fmt(inspect.signature(mc.pfade).parameters["block"].default)
    M["McDureeAns"] = fmt(jahre_mc)
    M["SeuilReussite"] = fmt(seuil_rev * 100)
    M["McRenteAns"] = fmt(RENTENJAHRE_BASIS)
    M["McRenteAnsCourt"], M["McRenteAnsLong"] = fmt(min(RENTENJAHRE_GRID)), fmt(max(RENTENJAHRE_GRID))
    M["McKapitalSeuil"] = fmt(seuil_mc)
    M["McKapitalSeuilPartProzent"] = fmt(seuil_mc / kapital_ref * 100)
    zj = e_base["ziel_jahr_perzentile"]
    centiles = sorted(zj)
    M["McCentileBas"], M["McCentileHaut"] = fmt(centiles[0]), fmt(centiles[-1])
    for nom, c in (("Bas", centiles[0]), ("Median", centiles[1]), ("Haut", centiles[-1])):
        annee = START_JAHR + int(round(zj[c]))
        M[f"McAnneeObjectif{nom}"] = str(annee)
        M[f"McAgeObjectif{nom}"] = fmt(annee - GEBURTSJAHR)
    t_fin = JAHRE_HORIZONT - 1
    if "AnneeHorizonFin" in M and M["AnneeHorizonFin"] != str(START_JAHR + t_fin):
        raise ValueError("_macros_redaction_kapitel_7: annee de fin d'horizon incoherente")
    wp = e_base["wert_perzentile"][t_fin]
    M["McCapitalHorizonBas"], M["McCapitalHorizonMedian"], M["McCapitalHorizonHaut"] = (
        fmt(wp[0]), fmt(wp[1]), fmt(wp[2]))
    M["McEventailRapport"] = fmt(wp[2] / wp[0], 1)
    e_h = mc.erfolg(pfade_base[:, :JAHRE_HORIZONT + RENTENJAHRE_BASIS], plan_ref, ZIEL_REAL_JAHR,
                    rendite_div=markt.rendite_div_maison, rentenjahre=RENTENJAHRE_BASIS)
    M["McErfolgDansHorizon"] = fmt(e_h["erfolgsquote"] * 100, 1)
    M["McErfolgEcartHorizon"] = fmt((e_base["erfolgsquote"] - e_h["erfolgsquote"]) * 100, 1)

    # --- Choc sur la premiere annee de retraite (fig_mc_erfolg). ---
    def reussite(choc, duree):
        ligne = df_erfolg[np.isclose(df_erfolg["div_schock"], choc)]
        if len(ligne) != 1:
            raise ValueError(f"_macros_redaction_kapitel_7: choc {choc} absent de la grille")
        return ligne[f"erfolg_{duree}"].iloc[0] * 100

    if fmt(reussite(0.0, RENTENJAHRE_BASIS), 1) != M["ErfolgsquoteBasis"]:
        raise ValueError("_macros_redaction_kapitel_7: fig_mc_erfolg != \\ErfolgsquoteBasis")
    M["ChocGrilleMax"] = fmt(max(DIV_SCHOCK_GRID) * 100)
    M["ChocGrillePas"] = fmt((DIV_SCHOCK_GRID[1] - DIV_SCHOCK_GRID[0]) * 100)
    M["McErfolgChocMin"] = fmt(reussite(min(DIV_SCHOCK_GRID), RENTENJAHRE_BASIS), 1)
    M["McErfolgChocMax"] = fmt(reussite(max(DIV_SCHOCK_GRID), RENTENJAHRE_BASIS), 1)
    M["McErfolgChocMoinsDix"] = fmt(reussite(-0.10, RENTENJAHRE_BASIS), 1)
    M["McErfolgChocMoinsVingt"] = fmt(reussite(-0.20, RENTENJAHRE_BASIS), 1)
    M["McErfolgCourt"] = fmt(reussite(0.0, min(RENTENJAHRE_GRID)), 1)
    M["McErfolgLong"] = fmt(reussite(0.0, max(RENTENJAHRE_GRID)), 1)

    # --- Reserve de liquidites (meme calcul que \ErfolgsquoteMitPuffer). ---
    M["PufferJahreMit"] = fmt(PUFFER_JAHRE_MACRO)
    M["PufferMontantMit"] = fmt(PUFFER_JAHRE_MACRO * ZIEL_REAL_JAHR)
    M["PufferPartCapitalProzent"] = fmt(PUFFER_JAHRE_MACRO * ZIEL_REAL_JAHR / kapital_ref * 100, 1)

    # --- Inflation 2019-2025 et rente non indexee (fig_inflation_2022). ---
    serie = h.wert("inflation_destatis_2019_2025")
    annees = sorted(int(a) for a in serie)
    annee_pic = max(annees, key=lambda a: serie[str(a)])
    # La valeur provisoire n'existe dans quellen.json que pour 2022 (cle inflation_2022_de).
    if annee_pic != 2022:
        raise ValueError(f"_macros_redaction_kapitel_7: pic d'inflation en {annee_pic}, pas 2022")
    M["InflationSerieDebut"], M["InflationSerieFin"] = str(annees[0]), str(annees[-1])
    M["InflationPicAnnee"] = str(annee_pic)
    M["InflationPicRevisee"] = fmt(serie[str(annee_pic)] * 100, 1)
    M["InflationPicProvisoire"] = fmt(h.wert("inflation_2022_de") * 100, 1)
    fin = lignes_infl[-1]["nicht_indexiert"]
    M["RenteNonIndexeeFin"] = fmt(fin)
    M["RentePerteProzent"] = fmt((1 - fin / ZIEL_REAL_MONAT) * 100, 1)
    M["RentePerteMois"] = fmt(ZIEL_REAL_MONAT - fin)
    M["PrixHausseCumulee"] = fmt((ZIEL_REAL_MONAT / fin - 1) * 100, 1)
    i_ziel = h.wert("inflation_ziel")
    M["RenteMoitieAns"] = fmt(np.log(2) / np.log(1 + i_ziel))
    M["RenteApresRetraite"] = fmt(ZIEL_REAL_MONAT / (1 + i_ziel) ** RENTENJAHRE_BASIS)

    # --- Encadre Yann: dividende brut mensuel apres la pire baisse historique (MEME appel
    # brutto_fuer_netto que la cascade du chapitre 2). ---
    sim_kw = _sim_kwargs()
    brutto = f.brutto_fuer_netto(ZIEL_REAL_JAHR, p.LAENDER_MIX, sim_kw["saetze"],
                                 sim_kw["pauschbetrag_nominal"], sq=sim_kw["sq"])
    if "BruttoNoetigMois" in M and fmt(brutto / 12) != M["BruttoNoetigMois"]:
        raise ValueError("_macros_redaction_kapitel_7: brut mensuel != \\BruttoNoetigMois")
    M["YannBrutApresPireBaisse"] = fmt(brutto / 12 * (1 + pire["rueckgang"]))


def kapitel_7():
    """fig_shiller_div, fig_einbrueche, fig_rueckspiel, fig_mc_faecher, fig_mc_erfolg,
    fig_inflation_2022. Macros: \\ErfolgsquoteBasis, \\ErfolgsquoteMitPuffer,
    \\GroessterEinbruch, \\GroessterEinbruchJahr."""
    markt = markt_basis()
    jahres_s = _charger_shiller_komplett()

    # --- fig_shiller_div: dividende reel du S&P 500 depuis 1871 (serie complete). ---
    schreiben("fig_shiller_div", jahres_s[["jahr", "div_real"]])

    # --- fig_einbrueche: baisses de dividende >= 10 % depuis le sommet precedent. ---
    einbrueche = histoire.div_einbrueche(jahres_s)
    schreiben("fig_einbrueche", pd.DataFrame([
        {"episode": f"{e['von']}-{e['bis']}", "rueckgang_prozent": e["rueckgang"] * 100}
        for e in einbrueche
    ]))
    if einbrueche:
        pire = min(einbrueche, key=lambda e: e["rueckgang"])
        M["GroessterEinbruch"] = fmt(abs(pire["rueckgang"]) * 100, 1)
        M["GroessterEinbruchJahr"] = str(pire["bis"])

    # --- fig_rueckspiel: rejeu du scenario de reference (strategie
    # REFERENCE_STRATEGIE_NOM, quote REFERENCE_QUOTE_NOM; tache 10 correctif du
    # 30.09.2026, auparavant code en dur a 70/30-20%) par annee de depart historique,
    # rendite_div = rendement reel mesure du portefeuille. ---
    plan_ref = p.sparplan_aus_quote(QUOTEN[REFERENCE_QUOTE_NOM], jahre=JAHRE_HORIZONT)
    rj = histoire.rueckspiel(jahres_s, plan_ref, ZIEL_REAL_JAHR, rendite_div=markt.rendite_div_maison)
    # Tache 10, chapitre 07: trois colonnes numeriques jumelles de jahre_bis_ziel, une par
    # issue (revenu tenu, revenu tombe sous le seuil, retraite non observable jusqu'au bout
    # dans l'historique), vides ailleurs, pour tracer un nuage a trois classes avec
    # \linienfigur sans lire la colonne booleenne (pgfplots ne sait pas filtrer dessus).
    # Colonnes d'origine inchangees.
    df_rj = rj[["start", "jahre_bis_ziel", "ueberlebt"]].copy()
    issue = df_rj["ueberlebt"].map(lambda v: "tenu" if v is True else ("echec" if v is False else "non_juge"))
    for nom in ("tenu", "echec", "non_juge"):
        df_rj[f"ans_{nom}"] = df_rj["jahre_bis_ziel"].where(issue == nom)
    schreiben("fig_rueckspiel", df_rj)
    _RUECKSPIEL["rj"] = rj

    # --- Monte Carlo (fig_mc_faecher, fig_mc_erfolg, fig_puffer): "jahre" recherche
    # pour eviter toute troncature (rentenjahre max de RENTENJAHRE_GRID), verifie par
    # assert (decision du controleur, brief tache 9). Le choc de fig_mc_erfolg
    # (mc.erfolg(..., div_schock_erstes_rentenjahr=...)) ne modifie ni le rendement
    # d'accumulation ni la date d'atteinte de l'objectif, donc le critere de troncature
    # (erreicht + rentenjahre >= jahre) est inchange par ce choc: un seul cas exigeant
    # suffit ici. ---
    cas_exigeants = [{"rentenjahre": max(RENTENJAHRE_GRID)}]
    jahre_mc, pfade_base = _mc_jahre_sans_troncature(jahres_s, plan_ref, ZIEL_REAL_JAHR,
                                                      markt.rendite_div_maison, cas_exigeants)

    # --- fig_puffer + \ErfolgsquoteBasis/\ErfolgsquoteMitPuffer: UN SEUL calcul Monte
    # Carlo (pfade_base ci-dessus) pour les deux, mis en cache dans _MC_PUFFER pour que
    # kapitel_8 reutilise EXACTEMENT ces resultats au lieu de les recalculer
    # independamment (revue de code, tache 9 fix round 1: spec section 8, "meme macro,
    # jamais deux calculs"; l'ecart de 1,3 point mesure entre les deux calculs
    # independants precedents disparait). rentenjahre=40: scenario de reference du
    # document (voir fig_kapital_zeit/fig_zinseszins/fig_sparbedarf; quote relevee a
    # REFERENCE_QUOTE_NOM par le correctif tache 10 du 30.09.2026). ---
    lignes_puffer, e_base = [], None
    for k in range(4):
        e_k = mc.erfolg(pfade_base, plan_ref, ZIEL_REAL_JAHR, rendite_div=markt.rendite_div_maison,
                        rentenjahre=RENTENJAHRE_BASIS, puffer_jahre=k)
        assert e_k["abgeschnitten"] == 0, f"fig_puffer: puffer_jahre={k} abgeschnitten={e_k['abgeschnitten']}"
        lignes_puffer.append({"puffer_jahre": k, "erfolgsquote": e_k["erfolgsquote"]})
        if k == 0:
            e_base = e_k   # reutilise pour fig_mc_faecher: wert_perzentile ne depend pas de puffer_jahre
    df_puffer = pd.DataFrame(lignes_puffer)
    _MC_PUFFER["df"] = df_puffer
    M["ErfolgsquoteBasis"] = fmt(df_puffer.loc[df_puffer["puffer_jahre"] == 0, "erfolgsquote"].iloc[0] * 100, 1)
    M["ErfolgsquoteMitPuffer"] = fmt(df_puffer.loc[df_puffer["puffer_jahre"] == PUFFER_JAHRE_MACRO, "erfolgsquote"].iloc[0] * 100, 1)

    # --- fig_mc_faecher: eventail p10/p50/p90 du capital, affiche sur les 60 premieres
    # annees (au-dela, la fourchette de compoundage domine visuellement sans ajouter
    # d'information utile au lecteur). ---
    fenetre = min(60, jahre_mc)
    perz = e_base["wert_perzentile"][:fenetre]
    # Tache 10, chapitre 07: deux lignes horizontales ajoutees pour que la figure montre
    # elle-meme l'ecart entre les deux modeles: "seuil_mc" = capital a partir duquel le
    # Monte Carlo juge l'objectif atteint (ZIEL_REAL_JAHR / rendement maison, SANS impot),
    # "capital_requis" = capital requis du chapitre 4 pour la repartition de reference
    # (MEME appel _kapital_pour_ziel que kapitel_4, AVEC impot et assurance maladie).
    kapital_ref = _kapital_requis_reference(markt)
    seuil_mc = ZIEL_REAL_JAHR / markt.rendite_div_maison
    schreiben("fig_mc_faecher", pd.DataFrame([
        {"jahr": START_JAHR + t, "p10": perz[t, 0], "p50": perz[t, 1], "p90": perz[t, 2],
         "seuil_mc": seuil_mc, "capital_requis": kapital_ref}
        for t in range(fenetre)
    ]))

    # --- fig_mc_erfolg: probabilite de reussite selon "div_schock", la croissance
    # reelle du dividende FORCEE (choc ADDITIF, mc.erfolg(..., div_schock_erstes_
    # rentenjahre=...)) sur la PREMIERE ANNEE DE RETRAITE de chaque trajectoire
    # (stress-test de sequence des rendements a l'entree en retraite; l'annee 0
    # calendaire de l'accumulation aurait ete inerte, capital nul a cet instant et
    # reussite post-retraite independante de rendite_real, voir LUECKEN.md) et la duree
    # de retraite testee (30/40/50 ans). Axe declare en macro texte pour la tache 10. ---
    # Texte accentue en LaTeX et sans le nom de colonne (tache 10, chapitre 07: la macro
    # s'imprime telle quelle dans un encadre \annahme); sens inchange.
    M["HypotheseDivSchock"] = ("Le choc est un \\'ecart additif, en points de pourcentage, "
                               "appliqu\\'e \\`a la croissance r\\'eelle du dividende pendant la "
                               "premi\\`ere ann\\'ee de retraite de chaque trajectoire Monte Carlo~; "
                               "les ann\\'ees suivantes restent tir\\'ees de l'historique. Ce n'est "
                               "pas un rendement de d\\'epart")
    lignes_erfolg = []
    for choc in DIV_SCHOCK_GRID:
        ligne = {"div_schock": choc}
        for rj_ in RENTENJAHRE_GRID:
            e = mc.erfolg(pfade_base, plan_ref, ZIEL_REAL_JAHR, rendite_div=markt.rendite_div_maison,
                          rentenjahre=rj_, div_schock_erstes_rentenjahr=choc)
            assert e["abgeschnitten"] == 0, (
                f"fig_mc_erfolg: div_schock={choc} rentenjahre={rj_} "
                f"abgeschnitten={e['abgeschnitten']} > 0")
            ligne[f"erfolg_{rj_}"] = e["erfolgsquote"]
        lignes_erfolg.append(ligne)
    df_erfolg = pd.DataFrame(lignes_erfolg)
    df_erfolg.columns = ["div_schock", "erfolg_30", "erfolg_40", "erfolg_50"]
    schreiben("fig_mc_erfolg", df_erfolg)

    # --- fig_inflation_2022: 3500 EUR d'une rente NON indexee depuis 2019, inflation
    # Destatis 2019-2025 (le choc de 2022 y figure), contre une rente indexee (reelle
    # constante par definition). ---
    serie_infl = h.wert("inflation_destatis_2019_2025")
    annees_infl = sorted(int(a) for a in serie_infl)
    lignes_infl, cum = [], 1.0
    for i, jahr in enumerate(annees_infl):
        if i > 0:
            cum *= 1 + serie_infl[str(jahr)]
        lignes_infl.append({"jahr": jahr, "indexiert": ZIEL_REAL_MONAT,
                            "nicht_indexiert": ZIEL_REAL_MONAT / cum})
    schreiben("fig_inflation_2022", pd.DataFrame(lignes_infl))

    _macros_redaction_kapitel_7(markt, jahres_s, einbrueche, rj, pfade_base, jahre_mc, plan_ref,
                                e_base, df_erfolg, lignes_infl, kapital_ref, seuil_mc)


def _rente_mensuelle(ausstiegsalter: int, traj: list, durchschnittsentgelt: float,
                     rentenwert: float, bbg_rv: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rente legale mensuelle estimee (euros de 2026): points = salaire brut
        annuel / Durchschnittsentgelt, le salaire etant plafonne au plafond de
        cotisation de l'assurance pension (tache 10, chapitre 08: un salaire
        au-dessus du plafond ne cotise pas au-dela, salaire.sv_anteil le
        plafonne deja pour la cotisation; les points ne l'etaient pas, ce qui
        surestimait la rente des dernieres annees), somme sur les annees
        cotisees de 26 ans (START_JAHR - GEBURTSJAHR) a ausstiegsalter, fois
        rentenwert_2026. Simplification declaree: carriere continue depuis
        START_JAHR, aucun trou de carriere, rente prise a l'age legal sans
        abattement, valeurs de 2026 tenues constantes en euros de 2026.

    Inputs:
        ausstiegsalter (int): age d'arret du travail (fin des cotisations).
        traj (list[dict]): salaire.trajektorie (cle "brutto", euros de 2026).
        durchschnittsentgelt (float): salaire moyen de 2026 (EUR/an).
        rentenwert (float): valeur du point de 2026 (EUR/mois).
        bbg_rv (float): plafond annuel de cotisation de l'assurance pension.

    Outputs:
        result (float): rente brute mensuelle, avant impot et cotisations.
    --------------------------------------------------------------------------
    """
    n_jahre = max(0, ausstiegsalter - (START_JAHR - GEBURTSJAHR))
    punkte = sum(min(t["brutto"], bbg_rv) / durchschnittsentgelt for t in traj[:n_jahre])
    return punkte * rentenwert


def _retrait_constant_historique(jahres: pd.DataFrame, taux: float, dauer: int) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------
    Purpose:
        Test historique de la regle de retrait (tache 10, chapitre 08): pour
        chaque annee de depart, un capital de 1 retire en DEBUT d'annee
        taux x capital initial (montant constant en termes reels), puis recoit
        le rendement reel total de l'annee (dividendes reinvestis). Le capital
        "tient" si les dauer retraits sont tous couverts. Sans impot, sans
        frais, marche americain seul (serie Shiller).

    Inputs:
        jahres (pandas.DataFrame): sortie de histoire.jahresreihe (colonnes
            "jahr", "rendite_real"). La premiere ligne (rendement non defini,
            rempli par 0.0 dans jahresreihe) est ecartee, comme au chapitre 7.
        taux (float): retrait annuel en part du capital initial (ex. 0.04).
        dauer (int): duree de retraite testee, en annees.

    Outputs:
        result (pandas.DataFrame): une ligne par annee de depart dont les
            dauer annees sont toutes observees: "start", "tenu" (bool),
            "annees_tenues" (retraits entierement couverts, <= dauer),
            "capital_fin" (capital reel final en part du capital initial,
            0.0 si le capital n'a pas tenu).
    --------------------------------------------------------------------------
    """
    serie = jahres.set_index("jahr")["rendite_real"].iloc[1:]
    annees, rendements = [int(a) for a in serie.index], list(serie.values)
    lignes = []
    for i in range(len(annees) - dauer + 1):
        wert, faits = 1.0, 0
        for t in range(dauer):
            if wert < taux - TOL_RETRAIT:
                break
            wert = (wert - taux) * (1 + rendements[i + t])
            faits += 1
        tenu = faits == dauer
        lignes.append({"start": annees[i], "tenu": tenu, "annees_tenues": faits,
                       "capital_fin": wert if tenu else 0.0})
    return pd.DataFrame(lignes)


def _resume_retrait_historique(df: pd.DataFrame) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Resume d'un test _retrait_constant_historique: part des departs ou le
        capital tient et pire annee de depart (la plus courte duree tenue, et
        a egalite le plus petit capital final; si tout tient, le plus petit
        capital final).

    Inputs:
        df (pandas.DataFrame): sortie de _retrait_constant_historique.

    Outputs:
        result (dict): "debuts", "tenus", "part", "pire_start",
            "pire_annees", "pire_capital", "echecs" (annees de depart en
            echec, triees), "premier", "dernier", "capital_median".
    --------------------------------------------------------------------------
    """
    pire = df.sort_values(["annees_tenues", "capital_fin", "start"]).iloc[0]
    tenus = int(df["tenu"].sum())
    return {"debuts": len(df), "tenus": tenus, "part": tenus / len(df),
            "pire_start": int(pire["start"]), "pire_annees": int(pire["annees_tenues"]),
            "pire_capital": float(pire["capital_fin"]),
            "echecs": sorted(int(a) for a in df.loc[~df["tenu"].astype(bool), "start"]),
            "premier": int(df["start"].min()), "dernier": int(df["start"].max()),
            "capital_median": float(df["capital_fin"].median())}


def kapitel_8():
    """fig_bruecke, fig_rente_alter, fig_puffer, et (tache 10, chapitre 08)
    fig_bruecke_anticipee, fig_regle_quatre_histoire. Macros: \\RenteBeiZielalter,
    \\BrueckeJahre (plus \\KapitalYannBasis, \\RevenuNetYannBasis, \\AnneeYannBasis,
    \\KapitalDepartMitte pour l'exemple chiffre "Yann" du chapitre 8, decision du
    controleur/dispatch: quelques macros sans chiffre dans le nom par chapitre pour
    ces exemples, calcules ici et non ecrits a la main en tache 10), puis les macros de
    redaction de _macros_redaction_kapitel_8."""
    markt = markt_basis()
    sim_kw = _sim_kwargs()
    traj = salaire.trajektorie(START_JAHR, JAHRE_HORIZONT)
    durchschnittsentgelt = h.wert("durchschnittsentgelt_2026")
    rentenwert = h.wert("rentenwert_2026")
    bbg_rv = h.wert("sv_arbeitnehmer")["bbg_rv_jahr"]
    regelalter = int(h.wert("regelaltersgrenze"))

    def rente(alter: int) -> float:
        return _rente_mensuelle(alter, traj, durchschnittsentgelt, rentenwert, bbg_rv)

    # Scenario de reference (constante REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM,
    # correctif tache 10 du 30.09.2026, auparavant code en dur a 70/30-20%).
    strat_70 = _reference_strategie()
    plan_ref = _reference_plan()
    s = p.simulieren(strat_70, markt, plan_ref, STARTKAPITAL[REFERENCE_STARTKAPITAL_NOM],
                      ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    ziel_jahr = s["ziel_jahr"]

    # --- fig_rente_alter: rente selon l'age de sortie du monde du travail, 35-67 ans
    # (points plafonnes au plafond de cotisation, tache 10, chapitre 08). ---
    lignes_rente = [{"ausstiegsalter": a, "rente_monat": rente(a)}
                    for a in range(RENTE_AGE_MIN, regelalter + 1)]
    schreiben("fig_rente_alter", pd.DataFrame(lignes_rente))

    # KapitalDepartMitte ne depend pas de l'atteinte de l'objectif (c'est un capital de
    # depart, pas un resultat de simulation): toujours defini, meme cas "non atteint" ci-
    # dessous, car plusieurs sections/*.tex le citent hors de tout contexte conditionnel.
    M["KapitalDepartMitte"] = fmt(STARTKAPITAL["Mitte"])

    if ziel_jahr is None:
        # Resultat a publier tel quel (decision 7 du controleur): le scenario de
        # reference n'atteint jamais l'objectif dans l'horizon (correctif tache 10,
        # seuils KV constants en reel: cf. projection._saetze_real; le scenario de
        # reference a ete releve a REFERENCE_QUOTE_NOM par le correctif du 30.09.2026,
        # voir task-10-kv-fix-report.md, mais cette branche reste geree pour tout futur
        # changement de parametres qui le ferait a nouveau echouer). Le pont et les
        # macros associees restent alors non definis, signale sans etre ajuste. Les
        # macros \KapitalYannBasis/\RevenuNetYannBasis/\AnneeYannBasis DOIVENT rester
        # definies (sections/*.tex les cite hors de tout \IfDefined): "non atteint" plutot
        # qu'une exception de compilation LaTeX ulterieure sur une commande indefinie.
        M["RenteBeiZielalter"] = "non atteint"
        M["BrueckeJahre"] = "non atteint"
        M["ZielNominalBasis"] = "non atteint"
        M["KapitalYannBasis"] = "non atteint"
        M["RevenuNetYannBasis"] = "non atteint"
        M["AnneeYannBasis"] = "non atteint"
        schreiben("fig_bruecke", pd.DataFrame(columns=["alter", "dividende", "rente"]))
    else:
        ziel_alter = ziel_jahr - GEBURTSJAHR
        idx = s["jahre"].index(ziel_jahr)
        dividende_monat = s["einkommen_netto_monat_real"][idx]
        rente_a_zielalter = rente(ziel_alter)

        # --- fig_bruecke: de l'age de depart a 90 ans; la rente legale demarre a
        # regelalter (67 ans), montant fige a celui calcule pour un arret a ziel_alter. ---
        lignes_bruecke = [{"alter": a, "dividende": dividende_monat,
                           "rente": rente_a_zielalter if a >= regelalter else 0.0}
                          for a in range(ziel_alter, 91)]
        schreiben("fig_bruecke", pd.DataFrame(lignes_bruecke))
        M["RenteBeiZielalter"] = fmt(rente_a_zielalter)
        M["BrueckeJahre"] = fmt(max(0, regelalter - ziel_alter))

        # --- Macros "Yann" (exemple chiffre du chapitre 8, scenario de reference:
        # REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM/REFERENCE_STARTKAPITAL_NOM). ---
        M["KapitalYannBasis"] = fmt(s["wert"][idx])
        M["RevenuNetYannBasis"] = fmt(dividende_monat)
        M["AnneeYannBasis"] = str(ziel_jahr)
        # Objectif de ZIEL_REAL_MONAT euros de 2026 exprime en euros nominaux de l'annee
        # d'objectif (encadre "Yann" du chapitre 2), a l'inflation cible.
        M["ZielNominalBasis"] = fmt(ZIEL_REAL_MONAT * (1 + markt.inflation) ** (ziel_jahr - 2026))

    # --- fig_puffer: probabilite de reussite selon la reserve de liquidites (0-3 ans),
    # meme scenario de reference (REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM,
    # rentenjahre=40) que \ErfolgsquoteBasis/\ErfolgsquoteMitPuffer. Reutilise TEL QUEL
    # le calcul deja fait par kapitel_7 (cache
    # _MC_PUFFER) plutot que de relancer une seconde recherche Monte Carlo independante:
    # spec section 8, "meme macro, jamais deux calculs" (revue de code, tache 9 fix
    # round 1). kapitel_7 s'execute toujours avant kapitel_8 dans main(). ---
    assert "df" in _MC_PUFFER, "fig_puffer: kapitel_7() doit s'executer avant kapitel_8()"
    schreiben("fig_puffer", _MC_PUFFER["df"])

    _macros_redaction_kapitel_8(markt, sim_kw, strat_70, traj, rente, regelalter, bbg_rv,
                                durchschnittsentgelt)


def _macros_redaction_kapitel_8(markt: p.Markt, sim_kw: dict, strat_70: p.Strategie, traj: list,
                                rente, regelalter: int, bbg_rv: float,
                                durchschnittsentgelt: float) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Macros de redaction du chapitre 8 (tache 10) et deux nouvelles figures:
        (1) fig_bruecke_anticipee, le pont jusqu'a l'age legal pour un depart
        anticipe (repartition de reference, taux d'epargne BRUECKE_QUOTE_NOM de
        la grille du chapitre 5; age controle contre ZUS["alter"]);
        (2) fig_regle_quatre_histoire, test historique du retrait constant de
        TAUX_RETRAIT % du capital initial (reel) sur la serie Shiller, pour
        chaque duree de RETRAIT_HIST_DUREES, et sa comparaison aux MEMES
        annees de depart en retraite que le rejeu du chapitre 7 (cache
        _RUECKSPIEL, aucun second rejeu). Reserve: valeurs lues dans
        fig_puffer (meme calcul que \\ErfolgsquoteMitPuffer).

    Inputs:
        markt (projection.Markt): hypotheses de marche de reference.
        sim_kw (dict): _sim_kwargs().
        strat_70 (projection.Strategie): repartition de reference.
        traj (list[dict]): trajectoire de salaire (salaire.trajektorie).
        rente (callable): age d'arret -> rente mensuelle (_rente_mensuelle).
        regelalter (int): age legal de la retraite.
        bbg_rv (float): plafond annuel de cotisation de l'assurance pension.
        durchschnittsentgelt (float): salaire moyen de 2026.

    Outputs:
        None. Remplit M, ecrit data/fig_bruecke_anticipee.csv et
        data/fig_regle_quatre_histoire.csv.
    --------------------------------------------------------------------------
    """
    # --- Rente legale: grandeurs de calcul et exemples lus sur la meme fonction. ---
    age_debut = START_JAHR - GEBURTSJAHR
    M["AgeDebutCotisation"] = fmt(age_debut)
    M["Durchschnittsentgelt"] = fmt(durchschnittsentgelt)
    M["Rentenwert"] = fmt(h.wert("rentenwert_2026"), 2)
    M["BbgRvJahr"] = fmt(bbg_rv)
    M["PointsEntree"] = fmt(min(traj[0]["brutto"], bbg_rv) / durchschnittsentgelt, 2)
    M["PointsPlafond"] = fmt(bbg_rv / durchschnittsentgelt, 2)
    au_plafond = [t["jahr"] for t in traj if t["brutto"] >= bbg_rv]
    M["AgePlafondRv"] = fmt(au_plafond[0] - GEBURTSJAHR) if au_plafond else "non atteint"
    M["RenteAgeMinGraphique"] = fmt(RENTE_AGE_MIN)
    for age in RENTE_AGES_EXEMPLE:
        M[f"RenteArret{zahlwort(age)}"] = fmt(rente(age))
    M["RenteDerniereAnnee"] = fmt(rente(regelalter) - rente(regelalter - 1))
    M["RenteLegalePartCible"] = fmt(rente(regelalter) / ZIEL_REAL_MONAT * 100)

    # --- Pont anticipe: meme simulation que kapitel_5 (capital de depart de reference). ---
    plan_b = p.sparplan_aus_quote(QUOTEN[BRUECKE_QUOTE_NOM], jahre=JAHRE_HORIZONT)
    s_b = p.simulieren(strat_70, markt, plan_b, STARTKAPITAL[REFERENCE_STARTKAPITAL_NOM],
                       ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)
    if s_b["ziel_jahr"] is None:
        raise ValueError("_macros_redaction_kapitel_8: depart anticipe non atteint")
    age_b = s_b["ziel_jahr"] - GEBURTSJAHR
    if age_b != ZUS["alter"][REFERENCE_STRATEGIE_NOM][BRUECKE_QUOTE_NOM]:
        raise ValueError("_macros_redaction_kapitel_8: age anticipe != chapitre 5")
    if age_b >= regelalter:
        raise ValueError("_macros_redaction_kapitel_8: le depart 'anticipe' n'est pas anticipe")
    idx_b = s_b["jahre"].index(s_b["ziel_jahr"])
    div_b = s_b["einkommen_netto_monat_real"][idx_b]
    rente_b = rente(age_b)
    schreiben("fig_bruecke_anticipee", pd.DataFrame([
        {"alter": a, "dividende": div_b, "rente": rente_b if a >= regelalter else 0.0,
         "cible": ZIEL_REAL_MONAT} for a in range(age_b, 91)]))
    M["BrueckeJahreFuenfzig"] = fmt(regelalter - age_b)
    M["BrueckeAnneeDebutFuenfzig"] = str(s_b["ziel_jahr"])
    M["BrueckeAnneeRente"] = str(GEBURTSJAHR + regelalter)
    M["RenteBrueckeFuenfzig"] = fmt(rente_b)
    M["RevenuNetBrueckeFuenfzig"] = fmt(div_b)
    M["KapitalBrueckeFuenfzig"] = fmt(s_b["wert"][idx_b])
    M["RenteBrueckePerteProzent"] = fmt((1 - rente_b / rente(regelalter)) * 100)
    M["RenteBrueckePerteMois"] = fmt(rente(regelalter) - rente_b)
    # Depart anticipe au taux minimal du chapitre 5 (\QuoteAnticipeeMaisonSiebzig).
    if M.get("AlterAnticipeeMaisonSiebzig", "non atteint") != "non atteint":
        age_a = int(M["AlterAnticipeeMaisonSiebzig"].replace("\\,", ""))
        M["BrueckeJahreAnticipee"] = fmt(regelalter - age_a)
        M["RenteBrueckeAnticipee"] = fmt(rente(age_a))
    else:
        M["BrueckeJahreAnticipee"] = "non atteint"
        M["RenteBrueckeAnticipee"] = "non atteint"

    # --- Reserve: valeurs lues dans fig_puffer (meme calcul que \ErfolgsquoteMitPuffer). ---
    dfp = _MC_PUFFER["df"].set_index("puffer_jahre")["erfolgsquote"]
    k_max = int(dfp.index.max())
    M["ErfolgsquotePufferEins"] = fmt(dfp.loc[1] * 100, 1)
    M["PufferJahreMax"] = fmt(k_max)
    M["ErfolgsquotePufferMax"] = fmt(dfp.loc[k_max] * 100, 1)
    M["PufferMontantMax"] = fmt(k_max * ZIEL_REAL_JAHR)
    M["PufferGainEins"] = fmt((dfp.loc[1] - dfp.loc[0]) * 100, 1)
    M["PufferGainMit"] = fmt((dfp.loc[PUFFER_JAHRE_MACRO] - dfp.loc[PUFFER_JAHRE_MACRO - 1]) * 100, 1)
    M["PufferGainMax"] = fmt((dfp.loc[k_max] - dfp.loc[k_max - 1]) * 100, 1)

    # --- Test historique de la regle des TAUX_RETRAIT % (decision du controleur). ---
    jahres_s = _charger_shiller_komplett()
    tables = {}
    for nom, duree in RETRAIT_HIST_DUREES.items():
        df = _retrait_constant_historique(jahres_s, TAUX_RETRAIT, duree)
        tables[nom] = df
        res = _resume_retrait_historique(df)
        M[f"RetraitHistAns{nom}"] = fmt(duree)
        M[f"RetraitHistDebuts{nom}"] = fmt(res["debuts"])
        M[f"RetraitHistTenus{nom}"] = fmt(res["tenus"])
        M[f"RetraitHistEchecs{nom}"] = fmt(len(res["echecs"]))
        M[f"RetraitHistEchecsListe{nom}"] = _liste_fr([str(a) for a in res["echecs"]]) if res["echecs"] else "aucune"
        M[f"RetraitHistReussite{nom}"] = fmt(res["part"] * 100, 1)
        M[f"RetraitHistPremier{nom}"] = str(res["premier"])
        M[f"RetraitHistDernier{nom}"] = str(res["dernier"])
        M[f"RetraitHistPireDebut{nom}"] = str(res["pire_start"])
        M[f"RetraitHistPireDuree{nom}"] = fmt(res["pire_annees"])
        M[f"RetraitHistFinMedianeFois{nom}"] = fmt(res["capital_median"], 1)
        prudent = _resume_retrait_historique(
            _retrait_constant_historique(jahres_s, TAUX_RETRAIT_PRUDENT, duree))
        M[f"RetraitHistReussitePrudent{nom}"] = fmt(prudent["part"] * 100, 1)
        M[f"RetraitHistPrudentFinMin{nom}"] = fmt(prudent["pire_capital"] * 100)
    df_csv = None
    for nom, df in tables.items():
        cols = df[["start", "capital_fin", "annees_tenues"]].rename(
            columns={"capital_fin": f"fin_{nom.lower()}", "annees_tenues": f"ans_{nom.lower()}"})
        df_csv = cols if df_csv is None else df_csv.merge(cols, on="start", how="outer")
    schreiben("fig_regle_quatre_histoire", df_csv.sort_values("start"))

    # --- Memes annees de depart en retraite que le rejeu du chapitre 7: la retraite du
    # rejeu commence l'annee qui suit l'objectif (histoire.rueckspiel, boucle k >= 1). ---
    assert "rj" in _RUECKSPIEL, "kapitel_7() doit s'executer avant kapitel_8()"
    duree_rejeu = inspect.signature(histoire.rueckspiel).parameters["rentenjahre"].default
    nom_long = next(n for n, d in RETRAIT_HIST_DUREES.items() if d == duree_rejeu)
    tenu_par_start = tables[nom_long].set_index("start")["tenu"]
    rj = _RUECKSPIEL["rj"]
    juges = rj[rj["ueberlebt"].notna()]
    paires = [(bool(u), bool(tenu_par_start.loc[int(z) + 1]))
              for u, z in zip(juges["ueberlebt"], juges["ziel_jahr"])]
    debuts_retraite = [int(z) + 1 for z in juges["ziel_jahr"]]
    # Plusieurs departs d'epargne aboutissent a la meme annee de retraite: le nombre
    # d'annees distinctes dit combien d'histoires differentes la comparaison contient.
    M["RetraitHistMemesDepartsAnnees"] = fmt(len(set(debuts_retraite)))
    for nom_classe, (d_ok, r_ok) in (("Aucun", (False, False)), ("SeulDividende", (True, False)),
                                     ("SeulRetrait", (False, True))):
        annees = sorted({a for a, pr in zip(debuts_retraite, paires) if pr == (d_ok, r_ok)})
        M[f"RetraitHistAnnees{nom_classe}"] = _liste_fr([str(a) for a in annees]) if annees else "aucune"
    M["RetraitHistMemesDepartsJuges"] = fmt(len(paires))
    M["RetraitHistMemesDepartsTenus"] = fmt(sum(r for _, r in paires))
    M["RetraitHistMemesDepartsEchecs"] = fmt(sum(not r for _, r in paires))
    M["RetraitHistMemesDepartsProzent"] = fmt(sum(r for _, r in paires) / len(paires) * 100, 1)
    M["RetraitHistLesDeuxTiennent"] = fmt(sum(d and r for d, r in paires))
    M["RetraitHistSeulRetraitTient"] = fmt(sum(r and not d for d, r in paires))
    M["RetraitHistSeulDividendeTient"] = fmt(sum(d and not r for d, r in paires))
    M["RetraitHistAucunNeTient"] = fmt(sum(not d and not r for d, r in paires))


# --- Chapitre 10 (tache 10, passage chapitres 09-10): seuils de capital de la frise
# fig_zeitplan (valeurs inchangees, auparavant une liste litterale dans kapitel_10), nommes
# pour que la prose cite \JalonAnnee<Nom>/\JalonSeuil<Nom> au lieu d'un chiffre tape. ---
JALONS_CAPITAL = {"DixMille": 10000.0, "CinquanteMille": 50000.0, "CentMille": 100000.0,
                  "DeuxCentCinquanteMille": 250000.0, "CinqCentMille": 500000.0}
# Reserve d'urgence du plan d'action: nombre de salaires nets mensuels. Convention de ce
# document (regle de prudence courante), SANS source dans quellen.json: le chapitre 10 la
# presente dans un \annahme comme un choix, pas comme une regle sourcee.
RESERVE_URGENCE_MOIS = 3
# Revue generale du plan: tant d'annees avant l'annee d'objectif du scenario de base.
REVUE_AVANT_OBJECTIF_ANS = 5


def kapitel_10():
    """fig_zeitplan. Macro: \\ZielJahrBasis. Scenario de reference (constante
    REFERENCE_STRATEGIE_NOM/REFERENCE_QUOTE_NOM, correctif tache 10 du 30.09.2026,
    auparavant code en dur a 70/30-20%). Colonne "libelle" (etiquette francaise du
    jalon, lue par la frise du chapitre 10) ajoutee au passage chapitres 09-10; les
    colonnes d'origine sont inchangees."""
    markt = markt_basis()
    sim_kw = _sim_kwargs()
    strat_70 = _reference_strategie()
    plan_ref = _reference_plan()
    s = p.simulieren(strat_70, markt, plan_ref, STARTKAPITAL[REFERENCE_STARTKAPITAL_NOM],
                      ZIEL_REAL_MONAT, JAHRE_HORIZONT, **sim_kw)

    seuils = list(JALONS_CAPITAL.values())
    lignes, atteints = [], set()
    for i, jahr in enumerate(s["jahre"]):
        for seuil in seuils:
            if seuil not in atteints and s["wert"][i] >= seuil:
                lignes.append({"jahr": jahr, "ereignis": f"capital_{int(seuil)}",
                               "kapital": s["wert"][i], "libelle": f"{int(seuil / 1000)} k€"})
                atteints.add(seuil)
    if s["ziel_jahr"] is not None:
        idx = s["jahre"].index(s["ziel_jahr"])
        lignes.append({"jahr": s["ziel_jahr"], "ereignis": "objectif", "kapital": s["wert"][idx],
                       "libelle": "Objectif"})
        M["ZielJahrBasis"] = str(s["ziel_jahr"])
    else:
        M["ZielJahrBasis"] = "non atteint"
    schreiben("fig_zeitplan", pd.DataFrame(lignes))


def _theme_normalise(thema: str) -> str:
    """Theme de presse.json sans accent ni casse ("fiscalité" et "fiscalite" sont le meme
    theme: les deux graphies coexistent dans les donnees collectees)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", thema.lower())
                   if unicodedata.category(c) != "Mn")


def _annee_depuis_age(age_texte: str) -> str:
    """Annee civile d'un age donne par une macro (meme convention que \\AnneeYannBasis =
    GEBURTSJAHR + \\AlterYannBasis); "non atteint" reste "non atteint"."""
    if age_texte == "non atteint":
        return age_texte
    return str(GEBURTSJAHR + int(age_texte.replace("\\,", "")))


def _macros_redaction_kapitel_9_10() -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Macros des chapitres 9 (presse) et 10 (plan d'action), toutes relues
        dans une donnee existante: data/presse.json pour les comptes de la
        presse, data/fig_zeitplan.csv pour les jalons, les macros deja posees
        par kapitel_5/kapitel_6 pour les ages, annees et epargnes. Aucun
        second calcul du scenario de base.

    Inputs:
        Neant (lit M, data/presse.json, data/fig_zeitplan.csv; doit etre
        appelee apres kapitel_5, kapitel_6 et kapitel_10).

    Outputs:
        None. Ecrit dans M. Leve ValueError si une phrase du chapitre 9
        cessait d'etre vraie (une citation sur la fiscalite qui ne serait
        pas defavorable, une position hors pro/contra/neutre).
    --------------------------------------------------------------------------
    """
    import math
    presse = json.load(open(os.path.join(DATA, "presse.json"), encoding="utf-8"))
    positions = [e["position"] for e in presse]
    if set(positions) - {"pro", "contra", "neutre"}:
        raise ValueError(f"position de presse inattendue: {set(positions)}")
    M["PresseCitations"] = fmt(len(presse))
    M["PresseArticles"] = fmt(len({e["id"] for e in presse}))
    M["PressePublications"] = fmt(len({e["quelle"] for e in presse}))
    for pos, nom in (("pro", "Pro"), ("contra", "Contra"), ("neutre", "Neutre")):
        M[f"Presse{nom}"] = fmt(positions.count(pos))
    fisc = [e for e in presse if _theme_normalise(e["thema"]) == "fiscalite"]
    if not fisc or any(e["position"] != "contra" for e in fisc):
        raise ValueError("chapitre 9: les citations sur la fiscalite ne sont plus toutes contra")
    M["PresseCitationsFiscalite"] = fmt(len(fisc))
    M["PresseDateConsultation"] = _date_fr(max(e["abgerufen"] for e in presse))

    # --- Chapitre 10: jalons de la frise, relus dans le CSV ecrit par kapitel_10 ---
    zp = pd.read_csv(os.path.join(DATA, "fig_zeitplan.csv"))
    for nom, seuil in JALONS_CAPITAL.items():
        ligne = zp[zp["ereignis"] == f"capital_{int(seuil)}"]
        M[f"JalonSeuil{nom}"] = fmt(seuil)
        M[f"JalonAnnee{nom}"] = str(int(ligne["jahr"].iloc[0])) if len(ligne) else "non atteint"

    # --- Annees des choix qui avancent le depart (ages du chapitre 5) ---
    M["AnneeEtfBasis"] = _annee_depuis_age(M["AlterEtfBasis"])
    for nom in ("MaisonSiebzig", "EtfReferenz"):
        M[f"AnneeAnticipee{nom}"] = _annee_depuis_age(M[f"AlterAnticipee{nom}"])
        q = M[f"QuoteAnticipee{nom}"]
        M[f"EpargneMoisAnticipee{nom}"] = ("non atteint" if q == "non atteint" else
                                           fmt(p.sparplan_aus_quote(int(q) / 100,
                                                                    jahre=JAHRE_HORIZONT)[0] / 12))
    M["RevueAvantObjectifAns"] = fmt(REVUE_AVANT_OBJECTIF_ANS)
    M["AnneeRevueAvantObjectif"] = ("non atteint" if M["ZielJahrBasis"] == "non atteint"
                                    else str(int(M["ZielJahrBasis"]) - REVUE_AVANT_OBJECTIF_ANS))

    # --- Epargne du scenario de base: part de l'ETF maison et montant par titre ---
    base = _reference_plan()[0] / 12
    maison = base * _reference_strategie().anteil_maison
    M["EpargneMoisMaisonBasis"] = fmt(maison)
    M["EpargneMoisParTitre"] = fmt(maison / int(M["NombreTitres"].replace("\\,", "")))

    # --- Reserve d'urgence (convention RESERVE_URGENCE_MOIS, salaire net d'entree) ---
    net = salaire.trajektorie(START_JAHR, 1)[0]["netto_monat"]
    reserve = RESERVE_URGENCE_MOIS * net
    complement = max(0.0, reserve - STARTKAPITAL[REFERENCE_STARTKAPITAL_NOM])
    M["ReserveUrgenceMois"] = fmt(RESERVE_URGENCE_MOIS)
    M["ReserveUrgence"] = fmt(reserve)
    M["ReserveUrgenceComplement"] = fmt(complement)
    M["ReserveUrgenceMoisComplement"] = fmt(math.ceil(complement / base))


def _ligne_presse(p_: dict) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Une ligne de data/tab_presse.tex pour une citation de data/presse.json.
        Extrait de tab_presse() (revue de code, tache 9 fix round 1) pour etre
        testable isolement: la citation est TRONQUEE SUR LE TEXTE BRUT (avant
        echappement), donc la troncature ne peut jamais couper une sequence
        d'echappement en deux (une sequence comme "\\&" n'existe pas encore au
        moment de la troncature, elle n'apparait qu'apres escapieren()
        applique a la chaine deja tronquee).

    Inputs:
        p_ (dict): entree de presse.json (cles au moins quelle, position,
            thema, zitat).

    Outputs:
        result (str): une ligne de tableau LaTeX, texte echappe, terminee par
            " \\\\".
    --------------------------------------------------------------------------
    """
    citation = p_.get("zitat", "")
    if len(citation) > 140:
        citation = citation[:137] + "..."
    return (f"{escapieren(p_.get('quelle', ''))} & {escapieren(p_.get('position', ''))} & "
            f"{escapieren(p_.get('thema', ''))} & {escapieren(citation)} \\\\")


_URL_DANGEREUX = re.compile(r"[%#&\\]")


def _date_fr(iso: str) -> str:
    """Convertit une date ISO (AAAA-MM-JJ, format de quellen.json/presse.json) au
    format francais JJ.MM.AAAA pour affichage dans une note de bas de page (meme
    convention que les dates deja ecrites en dur dans LUECKEN.md et les tests de
    check_footnote_pages.py, ex. "31.07.2026")."""
    if not iso:
        return ""
    a, m, j = iso.split("-")
    return f"{j}.{m}.{a}"


def _domaine_affichable(url: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Domaine court d'une URL ("www.bzst.de"), pour l'affichage d'un lien
        dans une note de bas de page en mode footmisc[para] (correctif
        infrastructure, tache 10): le corps de chaque note y est compose
        par footmisc dans un \\hbox non brisable (\\@footnotetext, variante
        "para" de footmisc.sty, confirme par lecture directe du fichier
        installe), ou aucun point de coupure de \\url n'est insere, meme
        avec le paquet xurl (teste isolement au passage precedent: Overfull
        \\hbox 152 a 461 pt "while \\output is active" sur les notes portant
        une URL longue). Le lien complet reste actif au clic (cible \\href
        cachee, hyperref[hidelinks]); l'URL complete reste lisible en clair
        dans l'annexe "Sources" (quellen_annexe.tex/presse_annexe.tex), qui
        n'est pas composee dans ce hbox restreint.

    Inputs:
        url (str): URL brute.

    Outputs:
        result (str): hote sans "www.", echappe pour LaTeX (le domaine ne
            contient normalement aucun caractere special, mais escapieren()
            reste applique par principe de defense en profondeur).
    --------------------------------------------------------------------------
    """
    hote = urlparse(url).netloc or url
    if hote.startswith("www."):
        hote = hote[4:]
    return escapieren(hote)


def _verifier_url(url: str, cle: str) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Garde-fou avant d'ecrire une URL brute dans le corps d'une macro
        \\newcommand generee (data/quellen.tex, data/presse_notes.tex): les
        caracteres %, #, & et \\ corrompent silencieusement la DEFINITION de
        la macro au moment ou LaTeX LIT le fichier (catcode de commentaire,
        de parametre ou de tabulation s'applique AVANT que \\url ne puisse
        basculer ses propres catcodes), contrairement a un \\url{} tape a la
        main dans le corps du document, ou \\url bascule ses catcodes avant
        que ces caracteres ne soient lus. Aucune URL de data/quellen.json ni
        data/presse.json n'en contient au 30.09.2026 (verifie par balayage
        complet, tache 10); si une future donnee en introduit une, on echoue
        fort ici plutot que de produire un PDF silencieusement corrompu.

    Inputs:
        url (str): URL brute.
        cle (str): cle ou id source, pour un message d'erreur exploitable.

    Outputs:
        None. Leve ValueError si un caractere dangereux est present.
    --------------------------------------------------------------------------
    """
    if _URL_DANGEREUX.search(url):
        raise ValueError(
            f"quellen/presse '{cle}': l'URL contient %, #, & ou \\, non geree "
            f"par la generation de macro (voir docstring de _verifier_url); "
            f"echapper a la main ou adapter quellen_tex()/presse_notes_tex().")


# Revue finale, constat 7: cle (quellen.json) et id (presse.json) servent de suffixe
# a \csname ... \endcsname et a \label{src:.../presse:...} dans quellen_tex()/
# presse_notes_tex() ci-dessous, sans validation. Un caractere hors de ce motif
# (espace, accolade, backslash) romprait silencieusement la compilation LaTeX au
# lieu d'etre refuse au moment ou il est ecrit.
_CLE_VALIDE = re.compile(r"^[a-z0-9_]+$")


def _verifier_cle(cle: str, origine: str) -> None:
    if not _CLE_VALIDE.match(cle):
        raise ValueError(f"{origine}: cle/id hors motif {_CLE_VALIDE.pattern!r}: {cle!r}")


def quellen_tex():
    """
    --------------------------------------------------------------------------
    Purpose:
        Ecrit data/quellen.tex (une macro \\QHfn@<cle> par cle de
        data/quellen.json, corps de la note de bas de page consommee par
        \\QH dans preamble.tex) et data/quellen_annexe.tex (liste labellisee
        \\label{src:<cle>} pour l'annexe "Sources", tache 10 du plan de
        nuit). Seuls quelle, seite (si present), url, abgerufen et primaer
        sont repris dans la note; "wert" (la valeur chiffree elle-meme) n'est
        pas le role de la couche de citation et reste dans
        data/kennzahlen.tex, ecrit par les blocs kapitel_*. Le champ interne
        "hinweis" (remarque de collecte, souvent en allemand, p. ex.
        "ATTENTION ecart de revision" pour inflation_destatis_2019_2025)
        n'apparait QUE dans l'annexe, jamais en bas de page (tache 10,
        passage resume et annexes: il s'imprimait tel quel dans les notes
        des chapitres 7 et 9).

    Inputs:
        Neant (lit data/quellen.json: dict de cle -> {quelle, url, abgerufen,
        seite?, primaer, hinweis?}).

    Outputs:
        None. Ecrit data/quellen.tex et data/quellen_annexe.tex.
    --------------------------------------------------------------------------
    """
    quellen = json.load(open(os.path.join(DATA, "quellen.json")))
    notes, annexe = [], []
    for cle in sorted(quellen):
        _verifier_cle(cle, "quellen_tex")
        q = quellen[cle]
        url = q.get("url", "")
        _verifier_url(url, cle)
        titre = escapieren(q.get("quelle", cle))
        date = _date_fr(q.get("abgerufen", ""))
        seite = q.get("seite")
        seite_txt = f", {escapieren(seite)}" if seite else ""
        primaire = "source primaire" if q.get("primaer") else "source secondaire"
        hinweis = q.get("hinweis")
        hinweis_txt = f" \\textit{{Remarque de collecte~: {escapieren(hinweis)}}}" if hinweis else ""
        # Note de bas de page: lien court affiche (domaine), cible complete au
        # clic - voir docstring de _domaine_affichable pour la raison (hbox non
        # brisable de footmisc[para]). Sans hinweis (annexe seulement).
        lien_note = f"\\href{{{url}}}{{{_domaine_affichable(url)}}}"
        corps_note = f"{titre}{seite_txt}. {lien_note}, consulté le {date} ({primaire})."
        # Annexe "Sources": URL complete en clair, composee en paragraphe normal
        # (pas dans le hbox restreint des notes), ou \url casse normalement.
        lien_annexe = f"\\url{{{url}}}"
        corps_annexe = f"{titre}{seite_txt}. {lien_annexe}, consulté le {date} ({primaire}).{hinweis_txt}"
        notes.append(f"\\expandafter\\newcommand\\csname QHfn@{cle}\\endcsname{{{corps_note}}}")
        # \phantomsection: ancre hyperref propre a l'entree (un item de description
        # n'en pose aucune), cible des liens du tableau d'hypotheses de l'annexe.
        annexe.append(f"\\item[\\texttt{{{escapieren(cle)}}}] \\phantomsection\\label{{src:{cle}}} {corps_annexe}")
    with open(os.path.join(DATA, "quellen.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (quellen_tex) - ne pas modifier a la main\n")
        fo.write("\n".join(notes) + "\n")
    with open(os.path.join(DATA, "quellen_annexe.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (quellen_tex) - ne pas modifier a la main\n")
        fo.write("\\begin{description}\n")
        fo.write("\n".join(annexe) + "\n")
        fo.write("\\end{description}\n")


def presse_notes_tex():
    """
    --------------------------------------------------------------------------
    Purpose:
        Ecrit data/presse_notes.tex (une macro \\QPfn@<id> par source de
        data/presse.json, corps de note consommee par \\QP dans
        preamble.tex) et data/presse_annexe.tex (liste labellisee
        \\label{presse:<id>} pour l'annexe "Sources"). data/presse.json est
        une liste de CITATIONS (plusieurs lignes peuvent partager le meme
        id de source, verifie manuellement le 30.09.2026: quelle/titel/url/
        abgerufen sont identiques pour toutes les lignes d'un meme id);
        cette fonction deduplique par id et ne reprend PAS zitat/aussage/
        position/thema, qui restent dans data/tab_presse.tex (tableau du
        chapitre 9, ecrit par tab_presse()) et n'ont pas leur place dans une
        note de bas de page generique.

    Inputs:
        Neant (lit data/presse.json: liste de dicts avec au moins id,
        quelle, titel, url, abgerufen).

    Outputs:
        None. Ecrit data/presse_notes.tex et data/presse_annexe.tex.
    --------------------------------------------------------------------------
    """
    presse = json.load(open(os.path.join(DATA, "presse.json")))
    par_id = {}
    for zeile in presse:
        par_id.setdefault(zeile["id"], zeile)  # premiere occurrence: meta partagee
    notes, annexe = [], []
    for id_ in sorted(par_id):
        _verifier_cle(id_, "presse_notes_tex")
        zeile = par_id[id_]
        url = zeile.get("url", "")
        _verifier_url(url, id_)
        titre = escapieren(zeile.get("titel", id_))
        source = escapieren(zeile.get("quelle", ""))
        date = _date_fr(zeile.get("abgerufen", ""))
        # Meme correctif que quellen_tex(): lien court (domaine) dans la note,
        # URL complete dans l'annexe (voir docstring de _domaine_affichable).
        lien_note = f"\\href{{{url}}}{{{_domaine_affichable(url)}}}"
        corps_note = f"{source}, \\textit{{{titre}}}. {lien_note}, consulté le {date}."
        lien_annexe = f"\\url{{{url}}}"
        corps_annexe = f"{source}, \\textit{{{titre}}}. {lien_annexe}, consulté le {date}."
        notes.append(f"\\expandafter\\newcommand\\csname QPfn@{id_}\\endcsname{{{corps_note}}}")
        annexe.append(f"\\item[\\texttt{{{escapieren(id_)}}}] \\phantomsection\\label{{presse:{id_}}} {corps_annexe}")
    with open(os.path.join(DATA, "presse_notes.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (presse_notes_tex) - ne pas modifier a la main\n")
        fo.write("\n".join(notes) + "\n")
    with open(os.path.join(DATA, "presse_annexe.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (presse_notes_tex) - ne pas modifier a la main\n")
        fo.write("\\begin{description}\n")
        fo.write("\n".join(annexe) + "\n")
        fo.write("\\end{description}\n")


def tab_presse():
    """
    --------------------------------------------------------------------------
    Purpose:
        Ecrire data/tab_presse.tex (corps de tableau: source, position, theme,
        citation courte echappee) a partir de data/presse.json, consomme par
        la tache 10 (decision 4 du controleur, tache 9: ecrit ici car aucun
        des blocs kapitel_2_und_3..kapitel_10 imposes ne porte le chapitre 9
        "presse et recherche").

    Inputs:
        Neant (lit data/presse.json, liste de dicts avec au moins quelle,
        position, thema, zitat).

    Outputs:
        None. Ecrit data/tab_presse.tex, une ligne de tableau LaTeX par
        citation (voir _ligne_presse), texte echappe via escapieren().
    --------------------------------------------------------------------------
    """
    presse = json.load(open(os.path.join(DATA, "presse.json")))
    lignes = [_ligne_presse(p_) for p_ in presse]
    with open(os.path.join(DATA, "tab_presse.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (tab_presse) - ne pas modifier a la main\n")
        fo.write("\n".join(lignes) + "\n")


def _pct(v, stellen: int = 1) -> str:
    """Part (0.25) -> "25~\\%" au format francais du document (via fmt)."""
    return f"{fmt(float(v) * 100, stellen)}~\\%"


def _eur(v, stellen: int = 0, suffixe: str = "") -> str:
    """Montant -> "1\\,000~€ par an" au format francais du document (via fmt)."""
    return f"{fmt(float(v), stellen)}~€{(' ' + suffixe) if suffixe else ''}"


def _oui_non(v) -> str:
    return "oui" if v is True or str(v).lower() == "true" else "non"


def _valeur_quellensteuer(w: dict) -> str:
    """Retenues par pays: les pays que le document commente, plus le decompte."""
    montres = [c for c in ("US", "CH", "FR", "NL", "GB") if c in w]
    parties = []
    for c in montres:
        txt = f"{LAND_FR.get(c, c)} {_pct(w[c]['einbehalt'], 1 if round(w[c]['einbehalt'] * 1000) % 10 else 0)}"
        if c == "US" and w[c].get("mit_antrag") != w[c]["einbehalt"]:
            txt += f" ({_pct(w[c]['mit_antrag'], 0)} avec W-8BEN)"
        parties.append(txt)
    return ", ".join(parties) + f"~; table de {len(w)} pays"


def _valeur_est_tarif(w: dict) -> str:
    """Bareme par zones: seuil d'exoneration et taux marginal maximal."""
    zonen = w["zonen"]
    return (f"{len(zonen)} zones, exon\\'eration jusqu'\\`a {_eur(zonen[0][0])}, taux marginal "
            f"maximal {_pct(zonen[-1][2], 0)}")


def _valeur_sv(w: dict) -> str:
    return (f"part salariale retraite {_pct(w['rv'])}, ch\\^omage {_pct(w['av'])}~; plafonds "
            f"{_eur(w['bbg_rv_jahr'])} (retraite) et {_eur(w['bbg_kv_jahr'])} (maladie) par an")


def _valeur_inflation_serie(w: dict) -> str:
    annees = sorted(w)
    bas = min(annees, key=lambda a: w[a])
    haut = max(annees, key=lambda a: w[a])
    return (f"{annees[0]} \\`a {annees[-1]}~: de {_pct(w[bas])} ({bas}) \\`a {_pct(w[haut])} "
            f"({haut}), s\\'erie r\\'evis\\'ee")


# Annexe "Hypotheses declarees" (tache 10, passage resume et annexes): une ligne par cle
# de data/quellen.json -> (libelle francais, formatteur de la valeur, label de la section
# qui l'emploie). annexe_hypotheses() leve ValueError si une cle manque ou est en trop,
# pour qu'une nouvelle hypothese sourcee ne sorte jamais de l'annexe sans qu'on le voie.
ANNEXE_HYPOTHESES = {
    "abgeltungsteuer_satz": ("Taux de l'imp\\^ot sur les revenus du capital (Abgeltungsteuer)",
                             lambda w: _pct(w, 0), "sec:fisc-abgeltung"),
    "soli_satz": ("Contribution de solidarit\\'e, en part de l'imp\\^ot", _pct, "sec:fisc-abgeltung"),
    "sparerpauschbetrag": ("Forfait d'\\'epargnant, personne seule",
                           lambda w: _eur(w, suffixe="par an"), "sec:fisc-abgeltung"),
    "teilfreistellung_aktienfonds": ("Exon\\'eration partielle d'un ETF actions",
                                     lambda w: _pct(w, 0), "sec:fisc-teilfreistellung"),
    "basiszins_2026": ("Basiszins 2026 (Vorabpauschale)", _pct, "sec:fisc-teilfreistellung"),
    "quellensteuer": ("Retenues \\`a la source par pays", _valeur_quellensteuer,
                      "sec:fisc-quellensteuer"),
    "est_tarif_2026": ("Bar\\`eme 2026 de l'imp\\^ot sur le revenu", _valeur_est_tarif,
                       "sec:fisc-guenstiger"),
    "soli_freigrenze_2026": ("Seuil d'exon\\'eration de la contribution de solidarit\\'e",
                             lambda w: _eur(w, suffixe="d'imp\\^ot par an"), "sec:fisc-guenstiger"),
    "kv_satz_ermaessigt": ("Taux r\\'eduit de l'assurance maladie", _pct, "sec:fisc-kv"),
    "kv_zusatzbeitrag_2026": ("Cotisation compl\\'ementaire moyenne 2026", _pct, "sec:fisc-kv"),
    "pv_satz_kinderlos_2026": ("Assurance d\\'ependance, sans enfant", _pct, "sec:fisc-kv"),
    "kv_mindestbemessung_monat_2026": ("Plancher de l'assiette maladie",
                                       lambda w: _eur(w, suffixe="par mois"), "sec:fisc-kv"),
    "kv_bbg_monat_2026": ("Plafond de l'assiette maladie",
                          lambda w: _eur(w, 2, "par mois"), "sec:fisc-kv"),
    "kv_schwellen_dynamisierung": ("Plancher et plafond maladie revaloris\\'es avec les salaires",
                                   _oui_non, "sec:fisc-kv"),
    "kv_teilfreistellung_etf": ("Exon\\'eration partielle des ETF dans l'assiette maladie",
                                _oui_non, "sec:fisc-kv"),
    "inflation_ziel": ("Inflation retenue (objectif de la BCE)", _pct, "sec:depart-cible"),
    "shiller_url": ("S\\'erie de march\\'e am\\'ericaine (cours, dividendes, prix)",
                    lambda w: "donn\\'ees mensuelles de Robert Shiller", "sec:risques-histoire"),
    "etf_welt_rendite_div": ("Rendement de distribution de l'ETF monde (repli)", _pct,
                             "sec:capital-methode"),
    "sv_arbeitnehmer": ("Cotisations salariales et plafonds 2026", _valeur_sv, "sec:accu-salaire"),
    "einstiegsgehalt_brutto": ("Salaire brut d'entr\\'ee d'un ing\\'enieur",
                               lambda w: _eur(w, suffixe="par an"), "sec:accu-salaire"),
    "gehaltssteigerung_real": ("Progression r\\'eelle du salaire", lambda w: _pct(w) + " par an",
                               "sec:accu-salaire"),
    "inflation_2022_de": ("Inflation allemande 2022, annonce provisoire", _pct,
                          "sec:risques-inflation"),
    "inflation_destatis_2019_2025": ("Inflation allemande annuelle", _valeur_inflation_serie,
                                     "sec:risques-inflation"),
    "durchschnittsentgelt_2026": ("Salaire moyen 2026 (points de rente)",
                                  lambda w: _eur(w, suffixe="par an"), "sec:retraite-rente"),
    "rentenwert_2026": ("Valeur du point de rente", lambda w: _eur(w, 2, "par mois"),
                        "sec:retraite-rente"),
    "regelaltersgrenze": ("\\^Age l\\'egal de la retraite", lambda w: f"{int(w)}~ans",
                          "sec:retraite-rente"),
}

# Conventions du modele (non sourcees, fixees par le plan): (libelle, valeur en macros de
# data/kennzahlen.tex, label de section). Chaque macro citee est verifiee dans M par
# annexe_hypotheses(): une macro renommee fait echouer le calcul, pas la compilation.
ANNEXE_CONVENTIONS = [
    ("Cible", "\\ZielMonat~€ nets par mois, en euros de 2026", "sec:depart-cible"),
    ("Horizon de calcul", "\\HorizonJahre~ans, de 2027 \\`a \\AnneeHorizonFin", "sec:accu-age"),
    ("Capital de d\\'epart", "\\KapitalDepartNiedrig{} \\`a \\KapitalDepartHoch~€, milieu "
     "\\KapitalDepartMitte~€", "sec:depart-profil"),
    ("Taux d'\\'epargne compar\\'es", "\\SparquoteZehn, \\SparquoteZwanzig, \\SparquoteDreissig{} et "
     "\\SparquoteFuenfzig~\\% du salaire net", "sec:accu-taux"),
    ("Sc\\'enario de base", "r\\'epartition \\ReferenzStrategie, \\ReferenzSparquote~\\% d'\\'epargne",
     "sec:depart-profil"),
    ("Rendement r\\'eel du march\\'e (\\'ecart au plan)", "\\RenditeReal~\\% par an, moyenne "
     "g\\'eom\\'etrique \\FenetreMarcheDebut--\\FenetreMarcheFin{} (m\\'ediane~: \\RenditeReelleMediane~\\%)",
     "sec:depart-cible"),
    ("Rendement du dividende de l'ETF maison", "\\RenditeDivMaison~\\% (m\\'ediane du "
     "portefeuille-exemple)", "sec:capital-methode"),
    ("Croissance r\\'eelle du dividende maison", "\\CroissanceModeleMaison~\\% par an (indice "
     "historique~: \\ShillerDivCroissanceFenetre~\\%)", "sec:risques-histoire"),
    ("Retenue am\\'ericaine (\\'ecart au plan)", "\\QstUsFormulaire~\\%, formulaire W-8BEN "
     "suppos\\'e d\\'epos\\'e", "sec:fisc-quellensteuer"),
    ("R\\`egles fiscales et sociales", "celles de 2026, fig\\'ees, taux forfaitaire, sans "
     "G\\\"unstigerpr\\\"ufung ni imp\\^ot d'\\'Eglise", "sec:fiscalite"),
    ("Taux de retrait de la r\\'ef\\'erence", "\\TauxRetrait~\\% (variante prudente "
     "\\TauxRetraitPrudent~\\%)", "sec:capital-regle"),
    ("R\\'eussite d'une retraite simul\\'ee", "revenu d'au moins \\SeuilReussite~\\% de la cible "
     "pendant \\McRenteAns~ans", "sec:risques-modeles"),
    ("Monte Carlo", "\\McTrajectoires{} trajectoires, blocs de \\McBloc~ans, graine fixe",
     "sec:risques-montecarlo"),
    ("R\\'eserve de retraite simul\\'ee", "\\PufferJahreMit~ans de cible, \\PufferMontantMit~€",
     "sec:risques-reserve"),
    ("R\\'eserve d'urgence", "\\ReserveUrgenceMois~salaires nets, \\ReserveUrgence~€",
     "sec:plan-premiers"),
]

_MACRO = re.compile(r"\\([A-Za-z]+)")
_BIB_ENTREE = re.compile(r"^\s*@(?!comment|string|preamble)\w+\s*\{", re.IGNORECASE | re.MULTILINE)


def _ligne_hypothese(cle: str, q: dict) -> str:
    libelle, formatteur, section = ANNEXE_HYPOTHESES[cle]
    typ = "primaire" if q.get("primaer") else "secondaire"
    return (f"{libelle}\\newline{{\\scriptsize\\normalfont\\texttt{{{escapieren(cle)}}}}} & "
            f"{formatteur(q['wert'])} & \\ref{{{section}}} & \\hyperref[src:{cle}]{{{typ}}} \\\\")


def annexe_hypotheses():
    """
    --------------------------------------------------------------------------
    Purpose:
        Ecrit les corps de tableaux de l'annexe "Hypotheses declarees"
        (tache 10, passage resume et annexes), jamais tapes a la main:
        data/tab_hypotheses_sources.tex (une ligne par cle de
        data/quellen.json: libelle, valeur formatee a la francaise, section
        qui l'emploie, lien vers la source en annexe) et
        data/tab_hypotheses_modele.tex (conventions non sourcees du plan,
        exprimees avec les macros de data/kennzahlen.tex). Pose aussi la
        macro \\NombreReferencesAcademiques (entrees de data/literatur.bib),
        qui decide si l'annexe compose la bibliographie ou dit qu'elle est
        vide.

    Inputs:
        Neant (lit data/quellen.json et data/literatur.bib; lit M pour
        verifier les macros des conventions).

    Outputs:
        None. Ecrit les deux corps de tableaux et remplit M. Leve ValueError
        si ANNEXE_HYPOTHESES ne couvre pas exactement les cles de
        quellen.json, ou si une convention cite une macro absente de M.
    --------------------------------------------------------------------------
    """
    quellen = json.load(open(os.path.join(DATA, "quellen.json"), encoding="utf-8"))
    manquantes = sorted(set(quellen) - set(ANNEXE_HYPOTHESES))
    en_trop = sorted(set(ANNEXE_HYPOTHESES) - set(quellen))
    if manquantes or en_trop:
        raise ValueError(f"ANNEXE_HYPOTHESES a mettre a jour: manquantes {manquantes}, "
                         f"en trop {en_trop}")
    ordre = list(ANNEXE_HYPOTHESES)   # ordre du document: fiscalite, marche, salaire, rente
    lignes = [_ligne_hypothese(c, quellen[c]) for c in ordre]
    with open(os.path.join(DATA, "tab_hypotheses_sources.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (annexe_hypotheses) - ne pas modifier a la main\n")
        fo.write("\n".join(lignes) + "\n")
    conv = []
    for libelle, valeur, section in ANNEXE_CONVENTIONS:
        absentes = [m for m in _MACRO.findall(valeur) if m not in M and m not in ("ref",)]
        if absentes:
            raise ValueError(f"convention '{libelle}': macros absentes de kennzahlen {absentes}")
        conv.append(f"{libelle} & {valeur} & \\ref{{{section}}} \\\\")
    with open(os.path.join(DATA, "tab_hypotheses_modele.tex"), "w", encoding="utf-8") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py (annexe_hypotheses) - ne pas modifier a la main\n")
        fo.write("\n".join(conv) + "\n")
    bib = open(os.path.join(DATA, "literatur.bib"), encoding="utf-8").read()
    M["NombreReferencesAcademiques"] = str(len(_BIB_ENTREE.findall(bib)))
    M["NombreHypothesesSourcees"] = str(len(quellen))
    M["NombreHypothesesPrimaires"] = str(sum(1 for q in quellen.values() if q.get("primaer")))
    M["NombreHypothesesSecondaires"] = str(sum(1 for q in quellen.values() if not q.get("primaer")))


def main():
    # Revue finale, constat 2: main() est rappele plusieurs fois dans le meme processus
    # (suite de tests, 13 appels dans test_rechnung.py seul). Sans remise a zero, une
    # macro qu'un appel ulterieur omettrait de reecrire garderait silencieusement la
    # valeur de l'appel precedent au lieu de faire echouer le test qui la controle.
    M.clear()
    ZUS["alter"].clear()
    _MC_PUFFER.clear()
    _RUECKSPIEL.clear()
    for k in (kapitel_2_und_3, kapitel_4, kapitel_5, kapitel_5_ziel_niedrig, kapitel_6,
              kapitel_7, kapitel_8, kapitel_10):
        k()
    tab_presse()
    _macros_redaction_kapitel_9_10()
    quellen_tex()
    presse_notes_tex()
    annexe_hypotheses()
    with open(os.path.join(DATA, "kennzahlen.tex"), "w") as fo:
        fo.write("% genere par scripts/rechnung_retraite.py - ne pas modifier a la main\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    json.dump(ZUS, open(os.path.join(DATA, "zusammenfassung.json"), "w"), indent=1)
    print(f"[RECHNUNG] {len(M)} macros")


if __name__ == "__main__":
    main()
