#!/usr/bin/env python3
"""
fiscalite.py - du dividende brut au montant disponible, pour un resident fiscal allemand.

Regles (sources dans data/quellen.json):
  * Abgeltungsteuer 25 % + Soli 5,5 % de l'impot; avec impot d'Eglise k, la formule
    statutaire de l'§32d al. 1 EStG est ESt = (e - 4q)/(4+k), ou e est le revenu
    imposable et q le credit d'impot etranger impute (q = 0 -> ESt = e/(4+k) = e*25 %
    quand k = 0). Soli 5,5 % de l'ESt, KiSt k x ESt.
  * Retenue etrangere: imputable jusqu'au taux conventionnel, au plus l'impot allemand
    du poste (calcule sans credit, e/(4+k)); l'excedent est perdu sauf remboursement
    (antrag=True -> taux "mit_antrag").
  * ETF actions: 30 % du revenu exonere (Teilfreistellung); la retenue au niveau du fonds
    est deja dans le rendement distribue et n'est pas imputable.
  * Sparerpauschbetrag: s'impute sur la base imposable, poste par poste dans l'ordre
    donne; une retenue etrangere sur la part couverte par le forfait est perdue.
  * Dividendes allemands (art="DE"): la table quellensteuer par defaut porte
    einbehalt = anrechenbar = 0 pour "DE" (pas de retenue etrangere, pas de DBA); c'est
    posten_netto qui calcule l'impot allemand via steuer_kap. Mettre un einbehalt non nul
    sur "DE" doublerait l'impot (25 % deja modelise par SATZ, plus une seconde retenue).
  * Assurance maladie et dependance volontaire: taux x revenu mensuel, borne par le
    plancher (Mindestbemessung) et le plafond (BBG).
"""
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SATZ, SOLI, TEILFREI = 0.25, 0.055, 0.30


def _sq_standard() -> dict:
    import hypotheses
    return hypotheses.wert("quellensteuer")


def steuer_kap(betrag_steuerpflichtig: float, kist: float = 0.0) -> float:
    """ESt + Soli + KiSt sur un revenu de capitaux imposable (sans retenue etrangere)."""
    if betrag_steuerpflichtig <= 0:
        return 0.0
    est = betrag_steuerpflichtig / (4 + kist) if kist else betrag_steuerpflichtig * SATZ
    return est * (1 + SOLI + kist)


def posten_netto(brutto: float, art: str, pauschbetrag_rest: float, antrag: bool = False,
                  kist: float = 0.0, sq: dict | None = None) -> tuple[float, float]:
    """(net disponible, reste du forfait) pour un poste de revenu.

    Avec impot d'Eglise (kist > 0), l'ESt apres credit d'impot etranger suit la formule
    statutaire de l'§32d al. 1 EStG, ESt = (e - 4q)/(4+k), et non e/(4+k) - q: avec k > 0
    le credit q pese moins que son montant nominal dans le calcul de l'ESt (il est
    "dilue" par le facteur 4/(4+k)). Les deux formules coincident pour k = 0.
    """
    if art == "ETF":
        steuerpfl = brutto * (1 - TEILFREI)
        einbehalt = anrechenbar = 0.0
    else:
        s = (sq or _sq_standard())[art]
        einbehalt = s["mit_antrag"] if antrag else s["einbehalt"]
        anrechenbar = s["anrechenbar"]
        steuerpfl = brutto
    genutzt = min(pauschbetrag_rest, steuerpfl)
    steuerpfl -= genutzt
    est_ohne_credit = steuerpfl / (4 + kist) if kist else steuerpfl * SATZ
    # Le credit est plafonne a l'impot allemand du poste avant credit (regle du DBA).
    anrechnung = min(anrechenbar * brutto, est_ohne_credit)
    if kist:
        rest_est = (steuerpfl - 4 * anrechnung) / (4 + kist)
    else:
        rest_est = est_ohne_credit - anrechnung
    steuer = rest_est * (1 + SOLI + kist)
    return brutto - einbehalt * brutto - steuer, pauschbetrag_rest - genutzt


def jahres_netto(posten: list[tuple[float, str]], pauschbetrag: float, **kw: Any) -> float:
    rest, summe = pauschbetrag, 0.0
    for brutto, art in posten:
        n, rest = posten_netto(brutto, art, rest, **kw)
        summe += n
    return summe


def kv_beitrag_jahr(einkommen_jahr: float, saetze: dict) -> float:
    monat = min(max(einkommen_jahr / 12, saetze["min_monat"]), saetze["bbg_monat"])
    return monat * 12 * (saetze["kv"] + saetze["zusatz"] + saetze["pv"])


def brutto_fuer_netto(ziel_netto_jahr: float, mix: dict[str, float], saetze: dict,
                       pauschbetrag: float, **kw: Any) -> float:
    """Dividende brut annuel qui laisse ziel_netto_jahr apres impot et KV/PV (bissection)."""
    def netto(b: float) -> float:
        posten = [(b * a, k) for k, a in mix.items()]
        return jahres_netto(posten, pauschbetrag, **kw) - kv_beitrag_jahr(b, saetze)
    lo, hi = 0.0, ziel_netto_jahr * 5
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if netto(mid) < ziel_netto_jahr else (lo, mid)
    return (lo + hi) / 2


def vorabpauschale(wert_anfang: float, wert_ende: float, ausschuettung: float,
                    basiszins: float) -> float:
    """Base de la Vorabpauschale (avant Teilfreistellung), §18 InvStG."""
    basisertrag = wert_anfang * basiszins * 0.7
    zuwachs = wert_ende - wert_anfang + ausschuettung
    return max(0.0, min(basisertrag, zuwachs) - ausschuettung)


def saetze_2026() -> dict:
    import hypotheses as h
    return {"kv": h.wert("kv_satz_ermaessigt"), "zusatz": h.wert("kv_zusatzbeitrag_2026"),
            "pv": h.wert("pv_satz_kinderlos_2026"),
            "min_monat": h.wert("kv_mindestbemessung_monat_2026"),
            "bbg_monat": h.wert("kv_bbg_monat_2026")}
