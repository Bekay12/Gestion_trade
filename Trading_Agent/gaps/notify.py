"""
notify - Notification ntfy.sh pour le cycle quotidien gaps (cron local).

Meme principe que agent/finviz_watch.py : ntfy.sh est gratuit, sans compte,
et un POST simple suffit. Le sujet par defaut est celui donne par
l'utilisateur le 09.10.2026 ("bertin-finviz-a7f3k9", deja dans la forme
longue et peu devinable recommandee par security.md) ; GAPS_NTFY_TOPIC en
environnement le remplace sans toucher au code.

Echoue silencieusement cote appelant (ne leve jamais) : une notification
manquee ne doit jamais faire echouer un scan ou un bilan deja ecrit sur
disque. Si l'envoi echoue, le message est imprime sur stderr pour rester
visible dans les logs cron.
"""

from __future__ import annotations

import os
import sys

import requests

DEFAULT_TOPIC = "bertin-finviz-a7f3k9"
TOPIC = os.environ.get("GAPS_NTFY_TOPIC", DEFAULT_TOPIC)


def notify(title: str, body: str, priority: str = "default") -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Envoie une notification ntfy.sh. N'arrete jamais l'appelant : une
        notification est un confort, pas une etape du pipeline.

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
        print(f"[GAPS] ntfy a echoue: {err}", file=sys.stderr)
        return False
