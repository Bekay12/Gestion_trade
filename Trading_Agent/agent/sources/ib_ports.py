"""
ib_ports - Ou TWS ou Gateway ecoute.

Preoccupation volontairement separee du connecteur : une erreur de port produit
exactement le meme message qu'un logiciel ferme, ce qui envoie chercher la panne
au mauvais endroit. Sonder plutot que demander a l'operateur de retenir lequel
des quatre ports correspond a son installation.

Ce module ne depend ni d'ib_async, ni d'un compte, ni de droits de donnees.
"""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger(__name__)

# Ports par defaut d'Interactive Brokers, dans l'ordre de sondage. Les comptes
# papier viennent d'abord : ce depot n'a aucune raison de viser un compte
# finance, et le premier port qui repond est celui qu'on utilise.
PAPER_TWS_PORT = 7497
PAPER_GATEWAY_PORT = 4002
LIVE_TWS_PORT = 7496
LIVE_GATEWAY_PORT = 4001

KNOWN_PORTS: tuple[tuple[int, str], ...] = (
    (PAPER_TWS_PORT, "TWS papier"),
    (PAPER_GATEWAY_PORT, "Gateway papier"),
    (LIVE_TWS_PORT, "TWS reel"),
    (LIVE_GATEWAY_PORT, "Gateway reel"),
)


def probe_port(host: str, port: int, timeout: float = 0.6) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Verifier qu'un port local accepte une connexion TCP. Seul controle qui
        ne demande ni ib_async, ni compte, ni droits de donnees.

    Inputs:
        host (str): hote
        port (int): port
        timeout (float): delai d'attente

    Outputs:
        listening (bool): vrai si quelque chose ecoute
    --------------------------------------------------------------------------
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def scan_ports(host: str = "127.0.0.1") -> list[tuple[int, str, bool]]:
    """Sonder les quatre ports connus, dans l'ordre le plus probable."""
    return [(port, label, probe_port(host, port)) for port, label in KNOWN_PORTS]


def detect_port(host: str = "127.0.0.1") -> int | None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Trouver le port ou TWS ou Gateway ecoute. Evite d'imposer a l'operateur
        de retenir lequel des quatre correspond a son installation : une erreur
        de port produit exactement le meme message qu'un logiciel ferme, ce qui
        envoie chercher la panne au mauvais endroit.

    Inputs:
        host (str): hote

    Outputs:
        port (int | None): premier port en ecoute, None si aucun
    --------------------------------------------------------------------------
    """
    for port, label, listening in scan_ports(host):
        if listening:
            logger.info("[IBKR] %s detecte sur le port %d", label, port)
            return port
    return None



