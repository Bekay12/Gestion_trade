"""
notify - Notification ntfy.sh pour la veille Dual Champions (canal separe du
cycle gaps de Trading_Agent - demande explicite du 13.10.2026 : "un deuxieme
canal de notifications").

Meme principe que Trading_Agent/gaps/notify.py, sujet distinct pour ne pas
mélanger les deux veilles sur le telephone. DUALCHAMP_NTFY_TOPIC en
environnement remplace le defaut sans toucher au code.

Echoue silencieusement cote appelant (ne leve jamais) : une notification
manquee ne doit jamais faire echouer un scan deja ecrit en etat.
"""

from __future__ import annotations

import os
import sys

import requests

DEFAULT_TOPIC = "bertin-dualchamp-r8n3w5"
TOPIC = os.environ.get("DUALCHAMP_NTFY_TOPIC", DEFAULT_TOPIC)


def notify(title: str, body: str, priority: str = "default") -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Envoie une notification ntfy.sh. N'arrete jamais l'appelant.

    Inputs:
        title (str): titre affiche
        body (str): corps du message
        priority (str): "min"|"low"|"default"|"high"|"urgent" (ntfy)

    Outputs:
        sent (bool): vrai si la requete a reussi
    --------------------------------------------------------------------------
    """
    if not TOPIC:
        return False
    try:
        requests.post(
            f"https://ntfy.sh/{TOPIC}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=15,
        )
        return True
    except requests.RequestException as err:
        print(f"[DUALCHAMP] ntfy a echoue: {err}", file=sys.stderr)
        return False
