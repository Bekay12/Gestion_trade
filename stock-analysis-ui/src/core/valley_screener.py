"""
Moteur de screening « creux » pour l'interface — deux signaux, une sortie.

Pourquoi ce module existe à côté de Valley_scan.py. Le script en ligne de
commande reste la référence : toute la logique de mesure et de classification y
vit, et rien n'en est recopié ici. Ce module ne fait que trois choses que
l'interface exige et que la ligne de commande n'a pas :

  • rendre le contrat {title, headers, rows} de ScreenerResultsDialog,
    symbole en colonne 0, comme core/store_screeners.py ;
  • accepter un rappel de progression, parce qu'un téléchargement de plusieurs
    dizaines de titres derrière une fenêtre figée est inutilisable ;
  • choisir un univers par défaut à partir des listes de l'application.

Budget yfinance : un appel groupé pour l'historique de tous les titres et de
leurs indices, puis un appel par CANDIDAT seulement (voir Valley_scan).
"""
from Valley_scan import analyser

MAX_ROWS = 80

# Ordre d'affichage : les signaux exploitables d'abord, le piège en dernier
# mais VISIBLE — c'est ce qui ressemble le plus à une occasion sans en être une.
ORDRE = {"↗️ INFLEXION": 0, "🕳️ DIVERGENCE": 1, "👀 À SUIVRE": 2, "⚠️ PIÈGE": 3}
SIGNAUX = {"divergence": "🕳️ DIVERGENCE", "inflexion": "↗️ INFLEXION",
           "piege": "⚠️ PIÈGE", "suivre": "👀 À SUIVRE"}

HEADERS = ["Symbole", "Signal", "Prix", "Baisse 3a %", "Au-dessus du bas %",
           "Titre 3m %", "Indice 3m %", "Part propre 3m %", "CA a/a %",
           "Indice", "Motif"]


def univers_par_defaut() -> list:
    """Liste de travail de l'application, avec repli sur le fichier texte."""
    try:
        from symbol_manager import get_symbols_by_list_type
        symboles = get_symbols_by_list_type("optimization", active_only=True)
        if symboles:
            return list(symboles)
    except Exception:
        pass
    try:
        from Valley_scan import POPULAR_FILE, load_symbols
        return load_symbols(POPULAR_FILE)
    except Exception:
        return []


def _cellule(valeur, decimales=1):
    if valeur is None:
        return "—"
    try:
        if valeur != valeur:          # NaN
            return "—"
        return f"{float(valeur):.{decimales}f}"
    except (TypeError, ValueError):
        return str(valeur)


def run_valley(signal: str = None, universe: list = None,
               min_drawdown: float = 25.0, near_low: float = 20.0,
               max_part_propre: float = 75.0, progress=None) -> dict:
    """
    --------------------------------------------------------------------------
    Objectif:
        Exécuter le détecteur de creux et rendre le résultat au format attendu
        par la fenêtre de résultats des screeners.

    Inputs:
        signal (str | None): 'divergence', 'inflexion', 'piege', 'suivre' ou None
        universe (list | None): symboles ; à défaut, la liste de l'application
        min_drawdown (float): baisse minimale depuis le plus haut 3 ans, en %
        near_low (float): distance maximale au plus bas 3 ans, en %
        max_part_propre (float): part propre maximale du recul 3 mois, en %
        progress (Callable | None): rappel (i, total, symbole)

    Outputs:
        resultat (dict): {title, headers, rows}
    --------------------------------------------------------------------------
    """
    symboles = list(universe) if universe else univers_par_defaut()
    if not symboles:
        return {"title": "🕳️ Creux — aucun univers", "headers": HEADERS, "rows": []}

    df = analyser(symboles, min_drawdown, near_low, quiet=True,
                  max_part_propre=max_part_propre, progress=progress)
    if df is None or df.empty:
        return {"title": "🕳️ Creux — aucun résultat", "headers": HEADERS, "rows": []}

    # Un titre sans signal n'a rien à faire dans la liste : le screener répond
    # « lesquels regarder », pas « voici les 43 titres ».
    df = df[df["signal"].isin(ORDRE)].copy()
    if signal:
        df = df[df["signal"] == SIGNAUX[signal]]

    df["_o"] = df["signal"].map(ORDRE).fillna(9)
    df = df.sort_values(["_o", "note"], ascending=[True, False]).head(MAX_ROWS)

    rows = [[
        r["ticker"], r["signal"], _cellule(r.get("prix"), 2),
        _cellule(r.get("baisse_%")), _cellule(r.get("au_dessus_du_bas_%")),
        _cellule(r.get("mouv_3m_%")), _cellule(r.get("indice_3m_%")),
        _cellule(r.get("fraction_propre_3m_%")), _cellule(r.get("ca_var_a1_%")),
        str(r.get("indice", "—")), str(r.get("motif", "")),
    ] for _, r in df.iterrows()]

    nom = {"divergence": "🕳️ Divergence", "inflexion": "↗️ Inflexion",
           "piege": "⚠️ Pièges", "suivre": "👀 À suivre"}.get(signal, "🕳️ Creux (tous signaux)")
    return {
        "title": f"{nom} — {len(rows)} titre(s) sur {len(symboles)} analysés "
                 f"(baisse ≥ {min_drawdown:.0f} %, part propre < {max_part_propre:.0f} %)",
        "headers": HEADERS, "rows": rows,
    }
