"""
combined_finviz - forme « Finviz + Combined » du screener Combined pour l'interface.

Deux formes coexistent dans l'application, volontairement :
    - Combined pur : core.store_screeners.screen_combined, catalogue local, 0 requete ;
    - Finviz + Combined (ce module) : decouverte Finviz sur tout le marche US
      (preset « dual_star », 1 requete), puis le Combined EN DIRECT
      (Combined_scan.analyze_safe) sur chaque titre trouve.
Les deux signalent l'etoile (Dual Champion*, core.scan_fondamentaux.est_etoile).

Pourquoi cette forme : backtest point-in-time du 02.10.2026, dates T >= 2025-04 :
la regle Finviz seule battait l'indice de +4,6 pts a 6 mois (4 dates sur 4),
confirmee Dual de +7,7 pts (+37,1 a 12 mois) ; en direct, 8 Dual Champion* sur
53 titres Finviz. Finviz trouve, le Combined trie.

Cout : 1 requete Finviz + 1 appel groupe de taux de change + ~4 requetes
yfinance par titre trouve (~50 titres, environ une minute). Le Combined ecrit
aussi chaque titre dans le store, ce qui rafraichit le catalogue au passage.
"""
from __future__ import annotations

PRESET = "dual_star"
HEADERS = ["Symbole", "Nom", "Pays", "Profil", "⭐", "Growth /5", "Safe /7",
           "G3 sous-val.", "G4 momentum %", "S4 dividende %"]
ORDRE = {"💎 Dual Champion*": 0, "💎 Dual Champion": 1, "🛡️  Pure Safe": 2,
         "🚀 Pure Growth": 3, "⚖️  Balanced": 4, "⚪ Below": 5}


def _cellule(v, dec=1):
    if v is None or v != v:
        return "—"
    try:
        return f"{float(v):.{dec}f}"
    except (TypeError, ValueError):
        return str(v)


def lignes(resultats: list[dict]) -> list[list]:
    """Resultats de Combined_scan.analyze -> lignes triees, etoile en tete et signalee."""
    res = [r for r in resultats if r]
    res.sort(key=lambda r: (ORDRE.get(r["profil"], 9), -r["score_total"], r["ticker"]))
    return [[r["ticker"], r.get("nom") or "N/A", r.get("pays") or "N/A", r["profil"],
             "⭐" if r["profil"] == "💎 Dual Champion*" else "",
             r["score_growth"], r["score_safe"],
             r.get("G3_Underval", "—"), _cellule(r.get("G4_Momentum_%")),
             _cellule(r.get("S4_DivYield_%"), 2)] for r in res]


def run_finviz_combined(limit: int = 500, progress=None) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Decouverte Finviz (preset dual_star) puis Combined en direct sur chaque
        titre ; rend le contrat {title, headers, rows} de ScreenerResultsDialog.

    Inputs:
        limit (int): plafond de lignes Finviz
        progress (Callable | None): rappel (i, total, symbole) -> bool ; un
            retour True interrompt l'analyse (bouton Annuler de l'interface)

    Outputs:
        resultat (dict): {title, headers, rows, decouverts, analyses}
    --------------------------------------------------------------------------
    """
    from core.finviz_screeners import run_preset
    from core import scan_fondamentaux as sf
    import Combined_scan as cs

    fv = run_preset(PRESET, limit=limit)
    tickers = [r[0] for r in (fv.get("rows") or [])]
    if not tickers:
        return {"title": "Finviz + Combined — Finviz ne renvoie aucun titre", "headers": HEADERS,
                "rows": [], "decouverts": 0, "analyses": 0}
    if len(sf.EUR_RATES) == 1:          # taux de change : un seul appel groupe
        sf.charger_taux_eur(verbose=False)
    resultats, interrompu = [], False
    for i, t in enumerate(tickers, 1):
        if progress and progress(i, len(tickers), t):
            interrompu = True
            break
        resultats.append(cs.analyze_safe(t))
    rows = lignes(resultats)
    n_dual = sum(1 for r in rows if r[3].startswith("💎"))
    n_star = sum(1 for r in rows if r[4] == "⭐")
    titre = (f"Finviz + Combined — {len(tickers)} titres Finviz, {len(rows)} analysés : "
             f"{n_star} ⭐ Dual Champion*, {n_dual - n_star} Dual Champion"
             + (" (interrompu)" if interrompu else ""))
    return {"title": titre, "headers": HEADERS, "rows": rows,
            "decouverts": len(tickers), "analyses": len(rows)}
