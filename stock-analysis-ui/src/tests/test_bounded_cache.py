"""
Tests unitaires purs pour qsi._BoundedCache (pas de réseau, pas de DB).
"""
from qsi import _BoundedCache


def test_bounded_cache_respects_maxsize():
    cache = _BoundedCache(maxsize=3)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    assert list(cache.keys()) == ['a', 'b', 'c']

    cache['d'] = 4
    assert len(cache) == 3
    assert 'a' not in cache
    assert list(cache.keys()) == ['b', 'c', 'd']


def test_bounded_cache_get_marks_recently_used():
    cache = _BoundedCache(maxsize=3)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3

    cache.get('a')  # 'a' devient le plus récemment utilisé
    cache['d'] = 4  # doit évincer 'b' (le moins récemment utilisé), pas 'a'

    assert 'a' in cache
    assert 'b' not in cache
    assert 'c' in cache
    assert 'd' in cache


def test_bounded_cache_get_missing_returns_default():
    cache = _BoundedCache(maxsize=3)
    assert cache.get('missing') is None
    assert cache.get('missing', 'fallback') == 'fallback'


def test_bounded_cache_overwrite_existing_key_no_growth():
    cache = _BoundedCache(maxsize=2)
    cache['a'] = 1
    cache['b'] = 2
    cache['a'] = 99  # mise à jour, pas d'éviction nécessaire
    assert len(cache) == 2
    assert cache['a'] == 99
