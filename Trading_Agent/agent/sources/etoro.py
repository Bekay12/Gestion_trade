"""
etoro - Client REST eToro (Public API, lancee 2026), demo et reel.

Specification verifiee le 09.10.2026 contre api-portal.etoro.com (pages
openapi.json, api-reference/trading--real/*, api-reference/trading--demo/*,
api-reference/market-data/*, api-reference/balances/*) - pas devinee. Les
champs non verifies directement sur une reponse reelle (parce qu'aucune cle
n'etait encore disponible au moment d'ecrire ce module) sont marques LACUNE
dans leur docstring plutot que supposes corrects.

SECURITE
    Les cles ne sont JAMAIS en dur dans ce fichier. Elles viennent des
    variables d'environnement ETORO_API_KEY et ETORO_USER_KEY (meme
    convention que SCOPUS_API_KEY, voir ~/.claude/rules/security.md), avec
    fallback optionnel sur un fichier gitignore si ces variables sont
    absentes - a l'image du fallback .scopus_key du skill scopus.

EXCEPTION A L'INVARIANT DU MOTEUR
    "Le moteur ne decide jamais d'entrer" (agent/CLAUDE.md) reste vrai pour
    IBKR et gate.py/evaluate(), inchanges. Ce module EST l'exception
    consciente actee avec l'utilisateur le 09.10.2026, pour eToro
    uniquement : create_order() peut etre appelee par une boucle autonome
    (voir agent/autonomy.py pour les bornes qui l'encadrent), sans
    confirmation humaine par trade. Toute ouverture doit passer par
    autonomy.can_open_new_position() et autonomy.kill_switch_active() AVANT
    cet appel - ce module ne les verifie pas lui-meme, il execute ce qu'on
    lui demande.

ENVIRONNEMENT
    ETORO_SANDBOX=1 (ou appeler les fonctions avec sandbox=True) tant que la
    phase de preuve n'est pas terminee - decision actee le 09.10.2026.
    Aucune fonction de ce module ne choisit elle-meme demo vs reel : c'est
    toujours un parametre explicite de l'appelant, jamais une valeur par
    defaut silencieuse qui pourrait toucher le compte reel par erreur.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://public-api.etoro.com"
# Budget de requetes : 20/60s partage sur les endpoints de trading, 60/60s
# sur les endpoints de lecture (marche, solde, portefeuille). Verifie le
# 09.10.2026 ; a reverifier si eToro change son quota.
_TIMEOUT_S = 15


class EtoroUnavailable(RuntimeError):
    """Cle absente, reseau indisponible, ou reponse HTTP en erreur."""


def _credentials() -> tuple[str, str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lit les cles depuis l'environnement. Ne lit jamais une cle en dur
        dans ce fichier ni dans un argument de fonction : betterleaks (voir
        securite du depot) refuse une cle en dur, pas une cle lue ainsi.

    Outputs:
        (api_key, user_key) (tuple[str, str])

    Raises:
        EtoroUnavailable: une des deux cles est absente de l'environnement
    --------------------------------------------------------------------------
    """
    api_key = os.environ.get("ETORO_API_KEY", "")
    user_key = os.environ.get("ETORO_USER_KEY", "")
    if not api_key or not user_key:
        raise EtoroUnavailable(
            "ETORO_API_KEY et/ou ETORO_USER_KEY absentes de l'environnement"
        )
    return api_key, user_key


def _headers() -> dict[str, str]:
    api_key, user_key = _credentials()
    return {
        "x-api-key": api_key,
        "x-user-key": user_key,
        "x-request-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, **kwargs: Any) -> dict:
    """Appel HTTP avec en-tetes d'authentification, erreurs traduites en
    EtoroUnavailable plutot que laissees comme exception requests brute."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(method, url, headers=_headers(), timeout=_TIMEOUT_S, **kwargs)
    except requests.RequestException as e:
        raise EtoroUnavailable(f"reseau indisponible ({type(e).__name__}: {e})") from e
    if resp.status_code == 429:
        raise EtoroUnavailable("limite de requetes atteinte (429)")
    if not resp.ok:
        raise EtoroUnavailable(f"HTTP {resp.status_code} sur {path}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as e:
        raise EtoroUnavailable(f"reponse non-JSON sur {path}") from e


def search_instrument(query: str, limit: int = 10) -> list[dict]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Resout un symbole (ex. "FRGT") en instrumentId eToro. Necessaire
        avant tout ordre : create_order() prend un instrumentId, pas
        seulement un symbole texte, et le mappage n'est pas garanti stable.

    Inputs:
        query (str): texte libre, 1 a 100 caracteres (symbole ou nom)
        limit (int): nombre maximal de resultats, 1 a 50

    Outputs:
        results (list[dict]): candidats avec au moins instrumentId, symbol,
            displayName - LACUNE: champs exacts non verifies contre une
            reponse reelle, voir en-tete du module
    --------------------------------------------------------------------------
    """
    data = _request("GET", "/api/v2/market-data/instruments/search",
                    params={"query": query, "limit": limit})
    return data.get("results", [])


def get_balances(display_currency: str = "USD") -> dict:
    """Solde agrege du compte. GET /api/v1/balances."""
    return _request("GET", "/api/v1/balances", params={"displayCurrency": display_currency})


def get_portfolio(sandbox: bool) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Positions ouvertes, agregees par instrument. Donne netUnits (signe =
        sens), avgOpenRate, accountCurrencyReturn (P&L latent) par
        instrumentId - PAS de positionId individuel (voir get_position_id
        pour fermer une position precise).

    Inputs:
        sandbox (bool): environnement demo si vrai, reel sinon - jamais de
            defaut implicite, voir en-tete du module

    Outputs:
        snapshot (dict): reponse brute (instrumentAggregates[], mirrors[])
    --------------------------------------------------------------------------
    """
    if sandbox:
        # Le chemin demo de cet endpoint precis n'a pas ete confirme page par
        # page (contrairement a create_order/close_position ci-dessous) :
        # deviner une URL plausible ici serait exactement l'estimation que
        # la regle 5 du depot interdit. A verifier contre
        # api-portal.etoro.com avant d'implementer.
        raise NotImplementedError(
            "chemin demo de get_portfolio non verifie - confirmer contre "
            "api-portal.etoro.com/api-reference/trading--demo/ avant usage"
        )
    return _request("GET", "/api/v1/trading/info/aggregate-portfolio")


@dataclass(frozen=True)
class OrderRequest:
    """Parametres d'un ordre, traduits vers le schema eToro par create_order."""

    instrument_id: int
    transaction: str  # "buy" (long) ou "sellShort" (short)
    amount: float  # montant investi, devise orderCurrency
    stop_loss_rate: float  # OBLIGATOIRE ici par convention du depot, voir Purpose
    leverage: int = 1
    take_profit_rate: float | None = None
    settlement_type: str | None = None  # "cfd" pour un short, "real" pour un long
    order_currency: str = "usd"


def create_order(order: OrderRequest, sandbox: bool) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Ouvre une position. POST /api/v2/trading/execution/orders (reel) ou
        /api/v2/trading/execution/demo/orders (demo) - chemins verifies le
        09.10.2026 contre la documentation officielle, pas devines.

        Le stop-loss est rendu obligatoire COTE APPEL (OrderRequest.stop_loss_rate
        n'a pas de defaut) meme si eToro ne l'exige lui-meme que pour
        sellShort/levier>1/realFutures : ce depot n'ouvre jamais de position
        nue, decision actee le 09.10.2026, independante de ce que le
        courtier tolere.

        Un 200 signifie "accepte pour traitement", PAS "execute". L'appelant
        doit confirmer via lookup_order() avant de considerer la position
        reellement ouverte - ne jamais supposer le remplissage depuis ce
        seul retour.

    Inputs:
        order (OrderRequest): parametres de l'ordre
        sandbox (bool): environnement demo si vrai, reel sinon

    Outputs:
        response (dict): {"token", "orderId", "referenceId"} a conserver pour
            lookup_order()
    --------------------------------------------------------------------------
    """
    if order.transaction not in ("buy", "sellShort"):
        raise ValueError(f"transaction non supportee: {order.transaction}")
    path = ("/api/v2/trading/execution/demo/orders" if sandbox
            else "/api/v2/trading/execution/orders")
    body: dict[str, Any] = {
        "action": "open",
        "transaction": order.transaction,
        "instrumentId": order.instrument_id,
        "orderType": "mkt",
        "leverage": order.leverage,
        "amount": order.amount,
        "orderCurrency": order.order_currency,
        "stopLossRate": order.stop_loss_rate,
        "stopLossType": "fixed",
    }
    if order.take_profit_rate is not None:
        body["takeProfitRate"] = order.take_profit_rate
    if order.settlement_type is not None:
        body["settlementType"] = order.settlement_type
    logger.info("[ETORO] ordre %s instrument=%s montant=%s levier=%s stop=%s sandbox=%s",
               order.transaction, order.instrument_id, order.amount, order.leverage,
               order.stop_loss_rate, sandbox)
    return _request("POST", path, json=body)


def lookup_order(order_id: int | None = None, reference_id: str | None = None,
                 sandbox: bool = True) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Confirme l'execution d'un ordre souris par create_order() et expose
        le positionId necessaire pour fermer la position plus tard.
        GET /api/v2/trading/info/orders:lookup.

    Inputs:
        order_id (int | None): exactement un des deux parametres
        reference_id (str | None): exactement un des deux parametres
        sandbox (bool): environnement demo si vrai, reel sinon

    Outputs:
        info (dict): contient positionExecutions[] avec positionId,
            instrumentId (dans asset), openingData.units,
            openingData.avgPrice, stopLossRate, takeProfitRate,
            asset.leverage - LACUNE: non verifie contre une reponse reelle
    --------------------------------------------------------------------------
    """
    if (order_id is None) == (reference_id is None):
        raise ValueError("fournir exactement un de order_id ou reference_id")
    params = {"orderId": order_id} if order_id is not None else {"referenceId": reference_id}
    # Chemin demo confirme (page create-an-order--demo, 09.10.2026) : cette
    # confirmation est explicitement requise apres un ordre demo, dont la
    # reponse 200 ne garantit pas l'execution.
    path = ("/api/v2/trading/info/demo/orders:lookup" if sandbox
            else "/api/v2/trading/info/orders:lookup")
    return _request("GET", path, params=params)


def close_position(position_id: int, instrument_id: int, sandbox: bool,
                   units_to_deduct: float | None = None) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Ferme une position, en tout ou en partie. POST
        /api/v1/trading/execution/market-close-orders/positions/{positionId}.
        Un 200 signifie "soumis", pas "execute" - meme reserve que
        create_order : confirmer avant de considerer la position fermee.

    Inputs:
        position_id (int): identifiant de position (obtenu via lookup_order)
        instrument_id (int): instrument de la position
        sandbox (bool): environnement demo si vrai, reel sinon
        units_to_deduct (float | None): None = fermer la position entiere

    Outputs:
        response (dict): contient orderID pour confirmation ulterieure
    --------------------------------------------------------------------------
    """
    if sandbox:
        # Meme reserve que get_portfolio : seul le chemin REEL de fermeture
        # a ete confirme page par page le 09.10.2026. Ne pas deviner un
        # suffixe d'URL pour la version demo - verifier contre
        # api-portal.etoro.com/api-reference/trading--demo/ avant d'ouvrir
        # cette fonction au sandbox.
        raise NotImplementedError(
            "chemin demo de close_position non verifie - confirmer contre "
            "api-portal.etoro.com/api-reference/trading--demo/ avant usage"
        )
    path = f"/api/v1/trading/execution/market-close-orders/positions/{position_id}"
    body = {"InstrumentId": instrument_id, "UnitsToDeduct": units_to_deduct}
    logger.info("[ETORO] fermeture position=%s instrument=%s sandbox=%s",
               position_id, instrument_id, sandbox)
    return _request("POST", path, json=body)
