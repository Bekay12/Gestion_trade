#!/usr/bin/env python3
"""
gen_figures_doc.py - Rafraichit la colonne "Valeur" de docs/figures.md a partir de
data/kennzahlen.tex (correctif infrastructure, tache 10, second passage).

Probleme corrige: docs/figures.md est un inventaire ecrit a la main (tache 10, premier
passage) qui recopiait les valeurs des macros au moment de sa redaction; ces valeurs
DERIVENT a chaque fois que scripts/rechnung_retraite.py est relance avec des donnees ou
un calcul different (ex. \\ZielJahrBasis documente comme "non atteint" alors qu'il vaut
2071 depuis un passage ulterieur, \\KapitalMaisonSiebzig documente a 2 973 818 alors
qu'il vaut desormais 2 974 126 - constate par le controleur, tache 10, decision 6).

Ce script ne reecrit QUE la colonne "Valeur" des lignes de tableau "| `\\Macro` | valeur
| sens |" (et la valeur d'exemple "\\AlterMaisonSiebzigZwanzig = N" du chapitre 5), en
relisant data/kennzahlen.tex a chaque execution. Il ne touche jamais la colonne "Sens"
(texte redige a la main) ni le reste du fichier (contrat des macros, listes de CSV).

Prudence deliberee: seules les valeurs "sures" (nombre au format fmt() de
rechnung_retraite.py, sentinelle "non atteint", ou identifiant court type ticker) sont
recopiees automatiquement. Une valeur de macro qui est en realite un texte de plusieurs
mots (hypothese, phrase) N'EST JAMAIS recopiee telle quelle dans le tableau (illisible
dans une cellule) - la cellule existante ("texte", "ticker", etc.) est laissee intacte.
Une ligne groupant plusieurs macros ("`\\A` / `\\B`" | "val_A / val_B") n'est rafraichie
que si TOUTES les macros de la ligne sont "sures"; sinon la ligne est laissee intacte.

Appele par ./build.sh juste apres scripts/rechnung_retraite.py, pour que ce document ne
puisse plus se perimer silencieusement.

Usage: python3 scripts/gen_figures_doc.py
Sortie: nombre de lignes rafraichies, nombre de macros ignorees (valeur non sure).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs", "figures.md")

_MACRO_DEF = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", re.M)
# Valeur "sure": uniquement des chiffres/virgules/accolades/espace fine LaTeX/signe moins
# fmt()-style (voir docstring de fmt() dans rechnung_retraite.py), ex. "76\,058", "7{,}3",
# "$-$1\,234{,}5".
_NUM_SUR = re.compile(r"^(?:\$-\$)?[0-9](?:[0-9,{}\\]*[0-9}])?$")
# Identifiant court type ticker (lettres/chiffres/point, sans espace ni backslash), ex.
# "ADS.DE": affichable tel quel dans le tableau.
_TICKER_SUR = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")
_SENTINELLE = "non atteint"


def charger_macros() -> dict:
    """Lit data/kennzahlen.tex, renvoie {nom_macro: valeur_brute_LaTeX}."""
    texte = open(os.path.join(DATA, "kennzahlen.tex"), encoding="utf-8").read()
    return dict(_MACRO_DEF.findall(texte))


def _affichable(brut: str):
    """
    --------------------------------------------------------------------------
    Purpose:
        Decide si une valeur de macro brute peut etre recopiee sans risque
        dans une cellule de tableau Markdown, et la met en forme le cas
        echeant (voir regle de prudence dans le docstring du module).

    Inputs:
        brut (str): valeur telle qu'ecrite dans data/kennzahlen.tex.

    Outputs:
        result (str | None): texte a afficher, ou None si la valeur n'est
            pas "sure" (la cellule existante doit alors etre laissee intacte).
    --------------------------------------------------------------------------
    """
    if brut == _SENTINELLE:
        return brut
    if _NUM_SUR.match(brut):
        return brut.replace("{,}", ",").replace("$-$", "-")
    if _TICKER_SUR.match(brut):
        return brut
    return None


_ROW = re.compile(r"^\| ((?:`\\[A-Za-z]+`(?: / )?)+) \| (.+?) \| (.+) \|$")
_MACRO_IN_CELL = re.compile(r"`\\([A-Za-z]+)`")
# La valeur est tout ce qui suit "=" jusqu'a la parenthese explicative, pas un seul mot:
# avec "(\S+)", seule la premiere partie de "non atteint" etait remplacee et un "atteint"
# s'ajoutait a chaque build (tache 10, chapitre 07; test_gen_figures_doc.py). Un texte
# deja corrompu est ainsi repare au passage suivant.
_EXEMPLE = re.compile(r"(`\\AlterMaisonSiebzigZwanzig`\s*=\s*)([^(\n]*?)(?=\s*\()")


def rafraichir(texte: str, macros: dict) -> tuple:
    """
    --------------------------------------------------------------------------
    Purpose:
        Reecrit la colonne "Valeur" de chaque ligne de tableau
        "| `\\Macro` [/ `\\Macro2` ...] | valeur | sens |" de docs/figures.md
        avec la valeur COURANTE de data/kennzahlen.tex, plus l'exemple
        "\\AlterMaisonSiebzigZwanzig = N" du chapitre 5.

    Inputs:
        texte (str): contenu actuel de docs/figures.md.
        macros (dict): {nom_macro: valeur_brute}, voir charger_macros().

    Outputs:
        (nouveau_texte, n_rafraichies, n_ignorees) (tuple[str, int, int]).
    --------------------------------------------------------------------------
    """
    n_rafraichies, n_ignorees = 0, 0
    lignes = texte.splitlines()
    for i, ligne in enumerate(lignes):
        m = _ROW.match(ligne)
        if not m:
            continue
        noms = _MACRO_IN_CELL.findall(m.group(1))
        if not noms:
            continue
        valeurs = []
        sur = True
        for nom in noms:
            brut = macros.get(nom)
            if brut is None:
                sur = False  # macro documentee mais absente de kennzahlen.tex (perimee)
                break
            aff = _affichable(brut)
            if aff is None:
                sur = False
                break
            valeurs.append(aff)
        if not sur:
            n_ignorees += len(noms)
            continue
        nouvelle_valeur = " / ".join(valeurs)
        # Conserve la mise en forme Markdown (guillemets code `...`) de la cellule
        # d'origine, ex. "`ADS.DE`" doit le rester, pas devenir "ADS.DE" nu.
        ancienne = m.group(2)
        if ancienne.startswith("`") and ancienne.endswith("`") and "/" not in nouvelle_valeur:
            nouvelle_valeur = f"`{nouvelle_valeur}`"
        if nouvelle_valeur != m.group(2):
            n_rafraichies += 1
        lignes[i] = f"| {m.group(1)} | {nouvelle_valeur} | {m.group(3)} |"

    nouveau_texte = "\n".join(lignes) + ("\n" if texte.endswith("\n") else "")

    def _remplacer_exemple(m):
        nonlocal n_rafraichies
        brut = macros.get("AlterMaisonSiebzigZwanzig")
        if brut is None:
            return m.group(0)
        aff = _affichable(brut) or brut
        if aff != m.group(2):
            n_rafraichies += 1
        return f"{m.group(1)}{aff}"

    nouveau_texte = _EXEMPLE.sub(_remplacer_exemple, nouveau_texte)
    return nouveau_texte, n_rafraichies, n_ignorees


def main() -> int:
    macros = charger_macros()
    texte = open(DOCS, encoding="utf-8").read()
    nouveau_texte, n_rafraichies, n_ignorees = rafraichir(texte, macros)
    if nouveau_texte != texte:
        with open(DOCS, "w", encoding="utf-8") as fo:
            fo.write(nouveau_texte)
    print(f"[FIGURES.MD] {n_rafraichies} valeur(s) rafraichie(s), "
          f"{n_ignorees} macro(s) a valeur non sure laissee(s) intacte(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
