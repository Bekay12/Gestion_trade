"""
ib_ticks - Traduction des ticks IB en valeurs du moteur.

Fonctions pures, sans reseau ni connexion : ce sont elles qui decident du sens
d'une valeur recue, et c'est la qu'une erreur passerait inapercue. Le
connecteur qui les appelle vit dans ibkr.py.

Trois pieges du protocole, verifies sur la documentation TWS puis confrontes a
un vrai Gateway le 2026-08-23 :

    - le tick 46 (« shortable ») porte des seuils publies : > 2.5 facile,
      > 1.5 disponible apres locate, <= 1.5 non empruntable ;
    - le tick 49 (« halted ») vaut -1 pour « statut indisponible », qui n'est
      PAS « non suspendu ». Sa valeur 0 n'est renvoyee que si le contrat figure
      dans une liste TWS, donc l'ignorance est le cas courant ;
    - le tick de volume differe (74) arrive en VIRGULE FIXE, echelle 10^6.
      Mesure directe : le callback brut porte 48 591 578 764 254 pour AAPL,
      alors que le tick 89 arrive juste par le meme chemin. Ce n'est donc ni
      ib_async ni la locale — c'est l'encodage du tick. Voir session_volume().
      S'y ajoute l'unite (actions ou lots de 100), qui depend d'un reglage de
      TWS que l'API ne rapporte pas.

Reference : docu/methode/07-etat-automatisation.md
"""

from __future__ import annotations

import logging
from math import isnan

from ..models import BorrowStatus

logger = logging.getLogger(__name__)

# Tick generique 236 : « Shortable ». Sans lui, ni le statut d'emprunt ni le
# nombre d'actions empruntables ne sont diffuses.
SHORTABLE_TICKS = "236"

# Seuils du tick 46, tels que publies par la documentation TWS.
SHORTABLE_EASY = 2.5
SHORTABLE_HARD = 1.5

# Unite du volume sur les actions americaines. Ce n'est PAS une constante du
# protocole : c'est un reglage de TWS (API > Einstellungen), et l'API ne dit pas
# dans quel etat il se trouve.
#
#   Depuis TWS 985, avec « Send volumes in lots ... for US stocks » DECOCHE
#   — le defaut moderne — les volumes arrivent deja en actions.
#   Coche, TWS revient au mode historique et envoie des lots de 100.
#
# Le defaut est 1, pour deux raisons. C'est le comportement moderne, et c'est
# le sens d'erreur sur : sous-estimer le volume fait echouer le filtre R6, donc
# refuse ; le surestimer d'un facteur cent le fait accepter ce qu'il devrait
# rejeter. Une regle bloquante doit se tromper du cote du refus.
SHARES_VOLUME_MULTIPLIER = 1
LOT_VOLUME_MULTIPLIER = 100
US_VOLUME_MULTIPLIER = SHARES_VOLUME_MULTIPLIER

# Garde-fou d'absurdite, pas un filtre de selection. Aucune action americaine
# n'echange vingt milliards de titres dans une seance ; au-dela, la valeur est
# corrompue ou encodee autrement, pas extreme. Laisser passer un tel nombre
# rendrait le volume relatif de R6 absurde, et un filtre qui accepte tout ne
# refuse plus rien.
MAX_PLAUSIBLE_VOLUME = 2e10

# Echelle en virgule fixe du tick de volume differe (tick 74).
#
# Mesure du 2026-08-23, Gateway reel, flux differe : TWS envoie lui-meme
# 48 591 578 764 254 pour AAPL. Ce n'est pas ib_async qui deforme — le callback
# brut porte deja cette valeur, alors que le tick 89 (actions empruntables)
# arrive juste par le MEME chemin de code. Divisee par un million, la valeur
# donne 48,6 M, du meme ordre que les 42,2 M rapportes par la source publique.
#
# Corroboration sur cinq titres couvrant trois ordres de grandeur : JUNS 52,4 M,
# SDOT 19,6 M, ADXN 727 069, AAOX 97 242. Tous plausibles.
#
# Le decodage n'est applique QUE si la valeur brute est impossible ET que la
# valeur decodee devient plausible. Hors de cette fenetre, on refuse plutot que
# de mettre a l'echelle au jugé : une inference silencieuse sur le volume
# fausserait le RVOL dans le sens permissif.
#
# A revalider en seance sur un flux DIRECT : cette mesure ne porte que sur le
# differe. Voir docu/methode/07-etat-automatisation.md, questions ouvertes.
TWS_DECIMAL_SCALE = 1_000_000

# Un flux en direct porte le type 1. Le type 3 est le differe, offert sans
# abonnement. Demander le direct a un compte qui n'y a pas droit ne degrade pas
# la reponse, il l'annule (erreur 10089) : le titre remonte alors sans prix du
# tout. Le differe vaut mieux que rien, a condition de ne jamais le presenter
# comme du direct — c'est ce que garantit le drapeau `realtime`, qui reste faux.
MARKET_DATA_LIVE = 1
MARKET_DATA_DELAYED = 3


def num(value: float | None) -> float | None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Valeur exploitable, ou None. IB signale l'absence par nan et par -1
        selon le champ ; les deux valent inconnu, jamais zero.

    Inputs:
        value (float | None): valeur brute d'un tick

    Outputs:
        number (float | None): valeur, ou None si absente
    --------------------------------------------------------------------------
    """
    if value is None:
        return None
    try:
        if isnan(value) or value < 0:
            return None
    except TypeError:
        return None
    return float(value)


def session_volume(
    raw: float | None, multiplier: int = US_VOLUME_MULTIPLIER, symbol: str = ""
) -> float | None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Volume de seance en actions, ou None si la valeur est inutilisable.

        Le garde-fou d'absurdite est le point important : une valeur corrompue
        remonte None, donc le moteur retombe sur le volume public et le volume
        relatif de R6 reste calculable. La servir telle quelle donnerait un
        RVOL absurde, et un filtre qui accepte tout ne refuse plus rien.

    Inputs:
        raw (float | None): valeur du tick de volume
        multiplier (int): 1 si TWS envoie des actions, 100 s'il envoie des lots
        symbol (str): ticker, pour la journalisation

    Outputs:
        volume (float | None): volume en actions, ou None
    --------------------------------------------------------------------------
    """
    value = num(raw)
    if value is None:
        return None
    value *= multiplier
    if value <= MAX_PLAUSIBLE_VOLUME:
        return value

    # Valeur impossible : tenter le decodage en virgule fixe, et ne l'accepter
    # que s'il produit un nombre plausible. Sinon refuser.
    decode = value / TWS_DECIMAL_SCALE
    if 0 < decode <= MAX_PLAUSIBLE_VOLUME:
        logger.info(
            "[IBKR] %s : volume decode en virgule fixe (%.0f -> %.0f)",
            symbol or "?", value, decode,
        )
        return decode

    logger.warning(
        "[IBKR] %s : volume %.0f invraisemblable meme decode, ignore",
        symbol or "?", value,
    )
    return None


def borrow_from_shortable(shortable: float | None) -> BorrowStatus | None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Traduire le tick 46 en statut d'emprunt du moteur (R7, R9). Un tick
        absent remonte None, que le crible traite comme un refus : c'est le
        comportement voulu, pas une panne.

    Inputs:
        shortable (float | None): valeur du tick 46

    Outputs:
        status (BorrowStatus | None): "easy", "hard", "none", ou None
    --------------------------------------------------------------------------
    """
    value = num(shortable)
    if value is None:
        return None
    if value > SHORTABLE_EASY:
        return "easy"
    if value > SHORTABLE_HARD:
        return "hard"
    return "none"


def halted_from_tick(halted: float | None) -> tuple[bool, bool]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Traduire le tick 49. Ses valeurs sont -1 (statut indisponible), 0 (non
        suspendu, renvoye seulement si le contrat est dans une liste TWS), 1
        (suspension reglementaire) et 2 (suspension de volatilite).

        L'ignorance remonte False, conformement a la convention du projet, mais
        elle est signalee : un titre dont on ignore la suspension n'est pas un
        titre dont on sait qu'il cote.

    Inputs:
        halted (float | None): valeur du tick 49

    Outputs:
        (flag, known) (tuple[bool, bool]): suspension, et si le statut est su
    --------------------------------------------------------------------------
    """
    if halted is None:
        return False, False
    try:
        if isnan(halted):
            return False, False
    except TypeError:
        return False, False
    if halted < 0:
        return False, False
    return halted >= 1, True
