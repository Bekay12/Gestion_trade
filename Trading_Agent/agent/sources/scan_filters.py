"""
scan_filters - Filtres avances du balayage TWS.

Separe du balayage lui-meme parce que la faute qu'on y commet ne ressemble pas
a une faute : TWS REFUSE certains filtres selon le compte, le refus arrive par
le canal d'erreurs, et la requete rend une liste VIDE. Un marche calme lui
ressemble trait pour trait.

Les valeurs booleennes viennent du XML de reqScannerParameters() du compte, pas
d'une supposition. Aucun seuil n'est ecrit ici : le plafond de flottant vient de
R6 et de R9, dans gate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..gate import LOW_FLOAT_CEILING, R9_FLOAT_CEILING

# Valeurs exactes des filtres booleens, relevees dans le XML du compte.
FILTRE_VRAI, FILTRE_FAUX = "true", "false"


class ScanUnavailable(RuntimeError):
    """Le balayage n'a pas abouti."""


@dataclass(frozen=True)
class ScanFilters:
    """
    --------------------------------------------------------------------------
    Purpose:
        Filtres avances du balayage. Comme les bornes, ils ne portent aucun
        seuil propre : le plafond de flottant vient de R6 et de R9.

        `exclude_halted` est vrai par defaut parce que R6 REFUSE un titre
        suspendu : le laisser occuper une des cinquante lignes du balayage
        gaspille la place d'un candidat que le crible aurait accepte. Le
        scanner applique ici la meme regle que le crible, dans le meme sens —
        il ne prend aucune decision que gate.py ne prendrait pas.

    Inputs:
        low_float (bool): borner le flottant au plafond de R6
        exclude_halted (bool): ecarter les titres suspendus (R6)
        shortable_only (bool): ecarter les titres non empruntables (R7)
        ssr_only (bool): ne garder que les titres sous Rule 201 (R9)
        change_pct_above (float | None): variation minimale en pourcentage

    Outputs:
        (dataclass immuable)
    --------------------------------------------------------------------------
    """

    low_float: bool = False
    exclude_halted: bool = True
    shortable_only: bool = False
    ssr_only: bool = False
    change_pct_above: float | None = None


# La part de R9 que TWS sait filtrer. Balayer cette configuration ne sert pas a
# la negocier — elle est bloquante — mais a savoir quels titres NE PAS vendre a
# decouvert avant d'en avoir envie.
#
# Le flottant n'y figure PAS, bien qu'il soit une condition de R9 : le compte de
# test a repondu « erreur 10360, Scan filter floatSharesBelow is not allowed ».
# L'autorisation depend du compte et du balayage, d'ou le drapeau low_float qui
# reste disponible pour qui l'a. Les deux conditions restantes — flottant reduit
# et interet vendeur — sont evaluees en aval par le crible, sur le Snapshot :
# le scanner retrecit, gate.py decide.
R9_TRAP = ScanFilters(ssr_only=True, exclude_halted=True)


def build_filters(filters: ScanFilters | None = None) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Traduire les filtres en TagValue, forme attendue par
        scannerSubscriptionFilterOptions.

    Inputs:
        filters (ScanFilters | None): filtres voulus

    Outputs:
        tags (list[TagValue]): filtres avances, vide si aucun

    Raises:
        ScanUnavailable: si ib_async est absent
    --------------------------------------------------------------------------
    """
    filters = filters or ScanFilters()
    try:
        from ib_async import TagValue
    except ImportError as error:                       # dependance optionnelle
        raise ScanUnavailable("ib_async absent") from error

    tags = []
    if filters.low_float:
        # R6 et R9 partagent ce plafond ; on prend le plus strict des deux.
        plafond = min(LOW_FLOAT_CEILING, R9_FLOAT_CEILING)
        tags.append(TagValue("floatSharesBelow", str(int(plafond))))
    if filters.exclude_halted:
        tags.append(TagValue("haltedIs", FILTRE_FAUX))
    if filters.shortable_only:
        tags.append(TagValue("unshortableIs", FILTRE_FAUX))
    if filters.ssr_only:
        tags.append(TagValue("shortSaleRestrictionIs", FILTRE_VRAI))
    if filters.change_pct_above is not None:
        tags.append(TagValue("changePercAbove", str(filters.change_pct_above)))
    return tags
