"""Verrouille le câblage du détecteur de creux dans l'interface.

Le module cœur peut être parfait et le screener rester injoignable : il suffit
que l'aiguillage ou l'entrée de menu manque. Ces deux tests couvrent exactement
cet espace, sans construire de fenêtre Qt.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent


def test_les_entrees_de_menu_existent():
    """Un screener sans entrée dans la liste déroulante est inatteignable."""
    source = (SRC / 'ui' / 'main_window.py').read_text()
    for cle in ('_valley', '_valley_divergence', '_valley_inflexion', '_valley_piege'):
        assert re.search(rf'\("{cle}",', source), f"entrée de menu manquante : {cle}"


def test_laiguillage_route_chaque_variante(monkeypatch):
    """`_show_yahoo_screener` doit appeler le bon signal pour chaque clé."""
    from ui.mixins.screeners import ScreenersMixin

    class _FauxCombo:
        def __init__(self, cle):
            self._cle = cle

        def currentData(self):
            return self._cle

        def currentText(self):
            return self._cle

    class _Fenetre(ScreenersMixin):
        def __init__(self, cle):
            self.screener_combo = _FauxCombo(cle)
            self.recu = None

        def _show_valley_screener(self, signal=None):
            self.recu = signal

    attendu = {
        '_valley': None,
        '_valley_divergence': 'divergence',
        '_valley_inflexion': 'inflexion',
        '_valley_piege': 'piege',
    }
    for cle, signal in attendu.items():
        f = _Fenetre(cle)
        f._show_yahoo_screener()
        assert f.recu == signal, f"{cle} devrait router vers signal={signal!r}, reçu {f.recu!r}"


def test_une_cle_inconnue_ne_route_pas_vers_les_creux():
    """Garde contre un `startswith` trop large qui capturerait d'autres clés."""
    from ui.mixins.screeners import ScreenersMixin

    class _FauxCombo:
        def currentData(self):
            return '_store_combined'

        def currentText(self):
            return 'store'

    class _Fenetre(ScreenersMixin):
        def __init__(self):
            self.screener_combo = _FauxCombo()
            self.vers_creux = False
            self.vers_store = False

        def _show_valley_screener(self, signal=None):
            self.vers_creux = True

        def _show_store_screener(self, engine_key):
            self.vers_store = True

    f = _Fenetre()
    f._show_yahoo_screener()
    assert f.vers_store and not f.vers_creux
