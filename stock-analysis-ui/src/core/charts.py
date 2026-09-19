"""
Visualisation : graphiques unifiés prix/volume/signaux.
À migrer depuis qsi.py :
  plot_unified_chart.
"""
import numpy as np
import pandas as pd

# TODO: migrer plot_unified_chart() depuis qsi.py

# Couleurs de fond par zone RSI (lightcoral / lightgreen / lightgray en hex,
# pour être lues aussi bien par matplotlib que par pyqtgraph).
RSI_ZONE_COLORS = {
    "surachat": "#F08080",
    "survente": "#90EE90",
    "neutre": "#D3D3D3",
}


def rsi_zones(index, rsi, upper: float = 70.0, lower: float = 30.0) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Regroupe les jours consécutifs de même zone RSI en bandes de fond.
        Remplace un axvspan par jour (7 500 rectangles sur 30 ans, 92 Mo et
        plus de 2 s par redessin) par quelques dizaines de bandes ; la
        couverture est identique : le segment [index[i], index[i+1]] prend la
        zone de rsi[i], et un RSI manquant tombe en « neutre ».

    Inputs:
        index (pd.Index): dates des prix, dans l'ordre.
        rsi (pd.Series | array-like): RSI aligné sur `index`.
        upper (float): seuil de surachat (strictement supérieur).
        lower (float): seuil de survente (strictement inférieur).

    Outputs:
        zones (list[tuple]): (début, fin, zone) avec zone dans RSI_ZONE_COLORS.
    --------------------------------------------------------------------------
    """
    n = len(index)
    if n < 2:
        return []
    vals = np.asarray(rsi, dtype="float64")[: n - 1]
    with np.errstate(invalid="ignore"):
        codes = np.where(vals > upper, 0, np.where(vals < lower, 1, 2))
    names = ("surachat", "survente", "neutre")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1))
    ends = np.concatenate((starts[1:], [n - 1]))
    return [(index[s], index[e], names[codes[s]]) for s, e in zip(starts, ends)]
