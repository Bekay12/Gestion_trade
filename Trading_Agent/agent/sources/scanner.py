"""
scanner - Decouverte de candidats (F1) par le scanner TWS.

Le dernier trou structurant du pipeline : jusqu'ici le moteur qualifiait bien,
mais ne voyait que ce qu'on lui mettait sous les yeux. La watchlist etait
saisie a la main.

Ce module ne cree AUCUN seuil. Les bornes du balayage sont importees de
gate.py : prix, volume moyen, capitalisation viennent de R6 et de nulle part
ailleurs. Une regle modifiee change donc le balayage sans qu'on y touche. Un
scanner qui reciterait ses propres chiffres divergerait du crible en silence,
et le premier symptome serait une liste de candidats que le crible refuse tous.

Ce que le scanner NE fait PAS, et qu'il ne faut pas lui demander :

    - il ne qualifie pas. Un titre remonte par le balayage est un symbole a
      passer au crible, pas un candidat. Le verdict reste dans gate.py ;
    - il ne CALCULE pas le volume relatif, filtre decisif de R6. IB n'offre
      aucun filtre de volume relatif ; il offre un classement ("rvol"), ce qui
      n'est pas la meme chose. Le rapport au volume habituel se mesure ensuite,
      sur le Snapshot ;
    - il ne lit aucun catalyseur. C'est la couche 2, toujours absente.

Ce qu'il sait faire et qu'un screener grand public ne sait pas : filtrer sur la
restriction Rule 201, sur le caractere non empruntable et sur la suspension.
Trois des quatre conditions de R9 sont donc filtrables A LA SOURCE. Les valeurs
exactes de ces filtres viennent du XML de reqScannerParameters() du compte, pas
d'une supposition : un libelle errone est accepte sans erreur par TWS et rend un
balayage vide, ce qui ressemble a un marche calme.

Limites du protocole, publiees par IB : 50 lignes maximum par balayage, et dix
balayages actifs au plus par connexion.

Reference : docu/methode/07-etat-automatisation.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..gate import MAX_MARKET_CAP, MAX_PRICE, MIN_AVG_VOLUME, MIN_PRICE
from ..eastern import market_open
from .scan_filters import (  # re-exportes : point d'entree unique
    FILTRE_FAUX,
    FILTRE_VRAI,
    R9_TRAP,
    ScanFilters,
    ScanUnavailable,
    build_filters,
)
from .ibkr import IbkrConnector, IbkrUnavailable

logger = logging.getLogger(__name__)

# Plafonds imposes par IB, pas par ce module.
IB_MAX_ROWS = 50
IB_MAX_ACTIVE_SCANS = 10

# Le cours recommande de demarrer avec trois a cinq alertes au maximum : le
# volume par defaut est ingerable et produit l'effet inverse de celui
# recherche. Ce defaut modeste est cette reserve, pas une limite technique.
DEFAULT_ROWS = 10

# Marche vise. Les petites capitalisations de la methode cotent sur les places
# principales americaines.
US_STOCKS = "STK.US.MAJOR"

# Balayages utiles a la methode. Les codes viennent de la documentation TWS ;
# reqScannerParameters() reste la seule liste faisant autorite pour un compte
# donne, d'ou dump_parameters() plus bas.
SCAN_CODES: dict[str, str] = {
    # Le titre qui s'envole : la situation meme du short sur spike.
    "spike": "TOP_PERC_GAIN",
    # L'anomalie de volume, cousine directe du filtre decisif de R6.
    "volume": "HOT_BY_VOLUME",
    # Les plus echanges du jour, filet plus large.
    "actifs": "MOST_ACTIVE",
    # Les ecarts d'ouverture : le moment ou la methode travaille reellement,
    # et la seule famille que le screener gratuit de reference ne donne pas.
    "gap": "TOP_OPEN_PERC_GAIN",
    "gap_haut": "HIGH_OPEN_GAP",
    "gap_bas": "LOW_OPEN_GAP",
    # Volume relatif sur cinq minutes : le classement le plus proche du filtre
    # decisif de R6, la ou aucun FILTRE de volume relatif n'existe.
    "rvol": "SCAN_stVolumeVsAvg5min_DESC",
    # Titres suspendus. A consulter, pas a negocier : R6 les refuse.
    "suspendus": "HALTED",
}
DEFAULT_SCANS = ("spike", "volume")

# Balayages qui n'ont de sens qu'en seance. Hors seance ils ne rendent pas une
# erreur : ils rendent une liste ALPHABETIQUE, faute de classement a calculer.
# C'est la pire forme d'echec — un resultat qui a l'air d'un resultat. Observe
# le 2026-08-23 : "rvol" a rendu AADX, AAL, AAOG, AAOX, AAOZ.
SCANS_INTRAJOURNALIERS = frozenset({"rvol", "gap", "gap_haut", "gap_bas", "volume"})

# TWS REFUSE certains filtres selon le compte et le balayage, et le refus n'est
# pas une exception : il arrive par le canal d'erreurs, la requete rend une
# liste VIDE, et un marche calme lui ressemble trait pour trait. Observe le
# 2026-08-23 : erreur 10360, « Scan filter floatSharesBelow is not allowed ».
# Sans ce releve, un filtre refuse viderait la liste de pre-marche en silence.
ERREURS_BALAYAGE = frozenset({
    10360,   # filtre non autorise
    162,     # echec de la requete de balayage historique/scanner
    165,     # avertissement du scanner
    366,     # aucune donnee de balayage
})


def build_subscription(
    scan: str = "spike",
    rows: int = DEFAULT_ROWS,
    location: str = US_STOCKS,
):
    """
    --------------------------------------------------------------------------
    Purpose:
        Construire l'abonnement de balayage, borne par R6. Les quatre bornes
        sont importees de gate.py : aucune valeur numerique n'est ecrite ici.

    Inputs:
        scan (str): cle de SCAN_CODES, ou un code TWS brut
        rows (int): nombre de lignes voulu, plafonne a IB_MAX_ROWS
        location (str): place de cotation

    Outputs:
        subscription (ScannerSubscription): abonnement pret

    Raises:
        ScanUnavailable: si ib_async est absent
    --------------------------------------------------------------------------
    """
    try:
        from ib_async import ScannerSubscription
    except ImportError as error:                       # dependance optionnelle
        raise ScanUnavailable(
            "ib_async absent : .venv\\Scripts\\python.exe -m pip install ib_async"
        ) from error

    if rows > IB_MAX_ROWS:
        logger.warning(
            "[SCAN] %d lignes demandees, IB en rend %d au plus", rows, IB_MAX_ROWS
        )
    code = SCAN_CODES.get(scan, scan)

    return ScannerSubscription(
        instrument="STK",
        locationCode=location,
        scanCode=code,
        numberOfRows=min(rows, IB_MAX_ROWS),
        # R6, importe. Le prix borne le terrain de jeu de la methode.
        abovePrice=MIN_PRICE,
        belowPrice=MAX_PRICE,
        # R6, importe. Volume de seance : le rapport au volume habituel se
        # mesure plus tard, sur le Snapshot.
        aboveVolume=int(MIN_AVG_VOLUME),
        # R6, importe. Au-dela, le titre sort du terrain habituel.
        marketCapBelow=float(MAX_MARKET_CAP),
    )


def symbols_from_scan(rows) -> list[str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Extraire les tickers d'une reponse de balayage, dans l'ordre du rang.
        Une ligne dont le contrat est illisible est ecartee et signalee, jamais
        completee.

    Inputs:
        rows: ScanDataList, ou tout iterable de ScanData

    Outputs:
        symbols (list[str]): tickers, sans doublon
    --------------------------------------------------------------------------
    """
    found: list[str] = []
    for row in rows or []:
        details = getattr(row, "contractDetails", None)
        contract = getattr(details, "contract", None)
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        if not symbol:
            logger.warning("[SCAN] ligne sans symbole exploitable, ecartee")
            continue
        if symbol not in found:
            found.append(symbol)
    return found


class MarketScanner:
    """
    Balayage du marche par la connexion TWS deja utilisee pour les cotations.
    Le cout marginal est nul : c'est la meme session, le meme port.
    """

    def __init__(self, connector: IbkrConnector | None = None) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer le scanner. Le connecteur est injectable pour les tests.

        Inputs:
            connector (IbkrConnector | None): connexion TWS

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self.connector = connector or IbkrConnector()

    def scan(
        self,
        scan: str = "spike",
        rows: int = DEFAULT_ROWS,
        filters: ScanFilters | None = None,
    ) -> list[str]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Un balayage. Rend des symboles, pas des candidats : la difference
            est tout le propos de gate.py.

        Inputs:
            scan (str): cle de SCAN_CODES, ou code TWS brut
            rows (int): nombre de lignes
            filters (ScanFilters | None): filtres avances

        Outputs:
            symbols (list[str]): tickers dans l'ordre du rang

        Raises:
            ScanUnavailable: si TWS refuse le balayage
        ----------------------------------------------------------------------
        """
        subscription = build_subscription(scan, rows)
        try:
            ib = self.connector.connect()
        except IbkrUnavailable as error:
            raise ScanUnavailable(str(error)) from error

        # On ecoute le canal d'erreurs pendant la requete : un filtre refuse
        # n'y leve pas d'exception, il vide la reponse.
        refus: list[str] = []

        def _noter(reqId, code, message, contract=None):
            if code in ERREURS_BALAYAGE:
                refus.append(f"{code} : {message}")

        evenement = getattr(ib, "errorEvent", None)
        if evenement is not None:
            evenement += _noter
        try:
            data = ib.reqScannerData(subscription, [], build_filters(filters))
        except Exception as error:                     # droits, code invalide
            raise ScanUnavailable(
                f"balayage '{scan}' refuse par TWS ({error}) — droits de donnees "
                "de marche, ou code de balayage inconnu de ce compte"
            ) from error
        finally:
            if evenement is not None:
                try:
                    evenement -= _noter
                except Exception:                      # desabonnement best effort
                    pass

        if refus:
            raise ScanUnavailable(
                f"balayage '{scan}' rejete par TWS — " + " ; ".join(refus)
                + ". Un filtre refuse rend une liste vide, pas une erreur : "
                "relancer sans le filtre en cause."
            )

        symbols = symbols_from_scan(data)
        if scan in SCANS_INTRAJOURNALIERS and not market_open()[0]:
            logger.warning(
                "[SCAN] %s : marche ferme — ce balayage n'a pas de classement a "
                "calculer et rend un ordre alphabetique, pas un resultat", scan
            )
        logger.info("[SCAN] %s : %d symbole(s)", scan, len(symbols))
        return symbols

    def sweep(
        self,
        scans: tuple[str, ...] = DEFAULT_SCANS,
        rows: int = DEFAULT_ROWS,
        filters: ScanFilters | None = None,
    ) -> dict[str, list[str]]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Plusieurs balayages, en gardant la trace de leur origine. Un titre
            remonte par deux balayages est plus interessant qu'un titre remonte
            par un seul : l'information est conservee plutot qu'aplatie.

        Inputs:
            scans (tuple[str, ...]): cles de balayage
            rows (int): lignes par balayage
            filters (ScanFilters | None): filtres avances, communs aux balayages

        Outputs:
            origins (dict): {symbole: [balayages qui l'ont remonte]}
        ----------------------------------------------------------------------
        """
        if len(scans) > IB_MAX_ACTIVE_SCANS:
            raise ScanUnavailable(
                f"{len(scans)} balayages demandes, IB en accepte "
                f"{IB_MAX_ACTIVE_SCANS} au plus par connexion"
            )

        origins: dict[str, list[str]] = {}
        echecs: list[str] = []
        for name in scans:
            try:
                for symbol in self.scan(name, rows, filters):
                    origins.setdefault(symbol, []).append(name)
            except ScanUnavailable as error:
                # Un balayage refuse n'annule pas les autres : on le dit et on
                # continue, plutot que de rendre une liste vide sans raison.
                logger.warning("[SCAN] %s ignore : %s", name, error)
                echecs.append(f"{name} ({error})")

        # Un marche calme rend zero symbole ; une source morte aussi. Les deux
        # ne veulent pas dire la meme chose, et les confondre reviendrait a
        # presenter une panne comme une seance sans candidat.
        if echecs and len(echecs) == len(scans):
            raise ScanUnavailable(
                "aucun balayage n'a abouti — " + " ; ".join(echecs)
            )
        return origins

    def dump_parameters(self) -> str:
        """
        ----------------------------------------------------------------------
        Purpose:
            Rendre le XML des balayages et filtres valides POUR CE COMPTE.
            C'est la seule liste faisant autorite : les codes ecrits dans
            SCAN_CODES viennent de la documentation, ce qu'un compte donne
            accepte reellement peut differer.

        Inputs:
            None

        Outputs:
            xml (str): reponse brute de reqScannerParameters
        ----------------------------------------------------------------------
        """
        ib = self.connector.connect()
        return ib.reqScannerParameters()
