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
TA_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=500)
