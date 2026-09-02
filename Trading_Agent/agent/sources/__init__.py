"""
sources - Couche d'acquisition des donnees.

Chaque module couvre une fonction de docu/methode/03-outils.md. La regle qui
gouverne cette couche : une donnee qu'on ne peut pas obtenir reste None et
remonte telle quelle au moteur, qui la traite comme un refus. Aucun module ne
comble un trou par une valeur plausible.

Etat :
    edgar   - F4, depots reglementaires. Implemente, gratuit.
    ssr     - F5, restriction Rule 201. Implemente, gratuit.
    market  - F1/F5, prix et flottant. Implemente, differe.
    broker  - pont JSON, seam entre le moteur et le producteur d'instantanes.
    ibkr    - producteur TWS/Gateway : carnet, suspension, ET statut d'emprunt.
    scanner - F1, decouverte de candidats. Bornes importees de gate.py.
"""

from .broker import BrokerBridge, from_ibkr_snapshot
from .ibkr import (
    IbkrConnector,
    IbkrSettings,
    IbkrUnavailable,
    borrow_from_shortable,
    halted_from_tick,
)
from .scanner import MarketScanner, ScanUnavailable, build_subscription
from .edgar import (
    DilutionReport,
    EdgarClient,
    EdgarError,
    Filing,
    OwnershipReport,
    assess_dilution,
    assess_ownership,
)

__all__ = [
    "EdgarClient", "EdgarError", "Filing",
    "DilutionReport", "assess_dilution",
    "OwnershipReport", "assess_ownership",
    "BrokerBridge", "from_ibkr_snapshot",
    "IbkrConnector", "IbkrSettings", "IbkrUnavailable",
    "borrow_from_shortable", "halted_from_tick",
    "MarketScanner", "ScanUnavailable", "build_subscription",
]
