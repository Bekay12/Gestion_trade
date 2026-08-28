"""
Caches LRU en mémoire (process-local, sans lien avec Parquet/SQLite).
Sert uniquement à mémoïser les calculs d'indicateurs déjà faits pendant
la durée de vie du process pour éviter de les recalculer.
"""
from collections import OrderedDict
from typing import Dict


class _BoundedCache(OrderedDict):
    """
    Dict en mémoire borné en taille avec éviction LRU, sûr en multithread.

    Sûreté : chaque opération d'OrderedDict prise isolément est atomique sous
    le GIL, mais les séquences « tester puis agir » de cette classe ne le sont
    pas. `TA_CACHE` est lu et écrit depuis un `ThreadPoolExecutor` de symboles
    imbriqué dans un second de populations (`optimisateur_hybride.py`), donc
    entre le test d'appartenance et le `move_to_end` d'un thread, un autre
    thread peut évincer exactement cette clé par `popitem(last=False)`. Le
    `move_to_end` lèverait alors une `KeyError` qui remonterait jusqu'au
    `except Exception` par barre de `qsi_optimized.py`, lequel ajoute un
    signal 'NEUTRE' et continue : la panne se traduirait par un résultat
    silencieusement différent, exactement ce que ce lot s'interdit.

    Les seules issues par lesquelles cela peut lever sont donc neutralisées
    ici. Le dépassement transitoire du plafond ou une éviction de trop restent
    possibles sous course : ce sont des approximations de capacité, pas des
    erreurs de résultat, la valeur rendue restant toujours soit la valeur
    stockée, soit un défaut de cache légitime.
    """

    def __init__(self, maxsize: int = 500):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:
        if key in self:
            try:
                self.move_to_end(key)
            except KeyError:
                # Évincée par un autre thread entre le test et ici : l'écriture
                # ci-dessous la réinsère de toute façon en fin d'ordre LRU.
                pass
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            try:
                self.popitem(last=False)
            except KeyError:
                # Vidé par un autre thread : plus rien à évincer.
                break

    def get(self, key, default=None):
        if key in self:
            try:
                self.move_to_end(key)
            except KeyError:
                # Évincée entre le test et ici : c'est un défaut de cache.
                return default
        return super().get(key, default)


# Cache des dérivées de prix par (symbol, len(prices)).
# Clés: price_slope_rel, price_acc_rel, rsi_slope_rel, volume_slope_rel.
#
# Meme defaut que TA_CACHE ci-dessous, et pour la meme raison : la cle porte
# `prices_len`, donc un backtest de 1160 barres produit 1160 entrees pour UN
# symbole. A 500, les premieres barres etaient evincees avant d'avoir pu
# resservir et le taux de reussite etait nul. Le lot 2 avait corrige TA_CACHE
# et laisse celui-ci en l'etat, si bien que le repli recalculait le RSI complet
# (ta.momentum.RSIIndicator sur toute la tranche) a chaque barre des que les
# features de prix etaient actives. Mesure au profil du 2026-08-07, vecteur
# avec extras de prix : 1160 reconstructions du RSI pour 52 % du temps de
# l'evaluation.
# Une entree ne porte que 5 flottants, contre 24 champs pour un instantane
# TA_CACHE : a plafond egal elle pese donc nettement moins. Comme pour
# TA_CACHE, c'est un PLAFOND et non une allocation, et le cache est vide aux
# deux memes frontieres (fin de groupe d'optimisation, fin de boucle de
# backtest de l'interface).
DERIV_CACHE_MAXSIZE = 100_000

DERIV_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=DERIV_CACHE_MAXSIZE)

# Cache des indicateurs techniques (scalaires instantanés) par (symbol, len(prices)).
# Clés: last_close, last_ema20/50/200, last_rsi, prev_rsi, delta_rsi,
#       last_macd, prev_macd, last_signal, prev_signal, variation_30j/180j,
#       volume_mean/std, current_volume, last_bb_percent, last_adx,
#       last_ichimoku_base/conversion.
# Un backtest de 5 ans parcourt environ 1160 barres et produit donc 1160
# instantanes pour UN seul symbole. A 500, le cache evincait les premieres
# barres avant de pouvoir les reutiliser : son taux de reussite etait nul et
# chaque evaluation repayait le calcul complet des indicateurs.
# Mesure du 2026-08-06, meme serie, resultats identiques au bit pres :
#     maxsize=500    eval1 7,23 s   eval2 7,18 s   eval3 7,13 s
#     maxsize=5000   eval1 7,09 s   eval2 2,36 s   eval3 2,35 s
# 100 000 est un PLAFOND, pas une allocation. Mesure tracemalloc du 2026-08-06
# sur la structure reelle ecrite en qsi.py:402-427 (24 champs par instantane,
# cle a 4 elements) : un cache rempli a 100 000 entrees pese environ 196 Mo,
# soit ~2,06 Ko/entree (le dict Python par instantane domine, pas les
# flottants bruts). A cette echelle, 1160 entrees (un symbole) pesent environ
# 2,4 Mo et 58 000 entrees (un groupe de cinquante symboles) environ 114 Mo.
#
# CYCLE DE VIE REEL. Sans point de liberation, le caractere LRU ne borne rien
# a l'echelle d'un process long : la cle porte le nom du symbole, donc les
# entrees d'un symbole deja traite ne sont jamais reutilisees mais restent en
# place jusqu'a ce que le plafond les evince, et l'etat stable d'un run
# complet serait le plafond, tenu jusqu'a la sortie du process. Le plafond est
# atteignable depuis l'interface graphique et pas seulement depuis le CLI :
# le bouton « Analyser + Backtester » lance un backtest par symbole sur une
# periode allant jusqu'a 10y (2520 barres, ~2470 instantanes par symbole),
# soit une quarantaine de symboles a 10y ou environ 500 a la periode par
# defaut pour saturer, alors que popular_symbols.txt en compte 3243.
# Le cache est donc vide a deux frontieres naturelles, ou l'ensemble de
# travail change entierement :
#   - fin de l'optimisation d'un groupe secteur x cap_range
#     (optimisateur_hybride.optimize_sector_coefficients_hybrid, bloc finally) ;
#   - fin de la boucle de backtest de l'interface graphique (qsi.py, apres la
#     boucle sur signals_to_backtest).
# Les groupes ne partagent pas de symboles (un symbole appartient a un seul
# secteur et y est range dans un seul cap_range), donc vider entre deux
# groupes ne jette aucune entree qui aurait pu resservir.
TA_CACHE_MAXSIZE = 100_000

TA_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=TA_CACHE_MAXSIZE)
