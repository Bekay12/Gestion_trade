"""
Caches LRU en mémoire (process-local, sans lien avec Parquet/SQLite).
Sert uniquement à mémoïser les calculs d'indicateurs déjà faits pendant
la durée de vie du process pour éviter de les recalculer.
"""
from collections import OrderedDict
from typing import Dict


class _BoundedCache(OrderedDict):
    """Dict en mémoire borné en taille avec éviction LRU."""

    def __init__(self, maxsize: int = 500):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            self.popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
        return super().get(key, default)


# Cache des dérivées de prix par (symbol, len(prices)).
# Clés: price_slope_rel, price_acc_rel, rsi_slope_rel, volume_slope_rel.
DERIV_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=500)

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
# 100 000 entrees couvrent 1160 barres pour une cinquantaine de symboles, soit
# environ 20 Mo : un instantane porte une vingtaine de flottants plus sa cle.
TA_CACHE_MAXSIZE = 100_000

TA_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=TA_CACHE_MAXSIZE)
