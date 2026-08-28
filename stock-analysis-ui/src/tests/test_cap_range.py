"""
Verrouillage de la fonction classify_cap_range et sa chaîne d'import.
"""

import pytest

from symbol_manager import classify_cap_range


def test_classify_cap_range_is_importable_from_symbol_manager() -> None:
    """Verrouille : from symbol_manager import classify_cap_range doit réussir."""
    assert callable(classify_cap_range)


def test_qsi_import_chain_is_intact() -> None:
    """Verrouille : les 5 noms que qsi importe de symbol_manager restent disponibles."""
    # qsi importe ces cinq noms d'un seul bloc try/except ImportError. Un seul
    # nom manquant faisait echouer l'import entier et basculait qsi sur un
    # repli « methode txt », en perdant silencieusement les quatre autres.
    from qsi import (
        init_symbols_table,
        sync_txt_to_sqlite,
        get_symbols_by_list_type,
        get_symbols_by_sector_and_cap,
        classify_cap_range as qsi_classify_cap_range,
    )

    for fonction in (init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type,
                     get_symbols_by_sector_and_cap, qsi_classify_cap_range):
        assert callable(fonction)


@pytest.mark.parametrize("input_value,expected", [
    (0.0, 'Unknown'),
    (1.0, 'Small'),
    (1.99, 'Small'),
    (2.0, 'Mid'),
    (5.0, 'Mid'),
    (9.99, 'Mid'),
    (10.0, 'Large'),
    (50.0, 'Large'),
    (99.99, 'Large'),
    (100.0, 'Mega'),
    (5000.0, 'Mega')
])
def test_thresholds(input_value: float, expected: str) -> None:
    """Verrouille : bornes incluses/excluses testées."""
    assert classify_cap_range(input_value) == expected


def test_returns_unknown_on_none_and_negative() -> None:
    """Verrouille : None, 0, -5 retournent 'Unknown'."""
    assert classify_cap_range(None) == 'Unknown'
    assert classify_cap_range(0) == 'Unknown'
    assert classify_cap_range(-5) == 'Unknown'


def test_returns_unknown_on_non_numeric() -> None:
    """Verrouille : 'abc' ne lève pas d'exception, retourne 'Unknown'."""
    assert classify_cap_range('abc') == 'Unknown'


def test_consistency_with_hardcoded_values() -> None:
    """Verrouille : les valeurs 1.0, 5.0, 50.0, 500.0 retournent les étiquettes attendues."""
    assert classify_cap_range(1.0) == 'Small'
    assert classify_cap_range(5.0) == 'Mid'
    assert classify_cap_range(50.0) == 'Large'
    assert classify_cap_range(500.0) == 'Mega'
