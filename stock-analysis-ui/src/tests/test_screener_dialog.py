"""
Verrouillage de ScreenerResultsDialog apres l'ajout de la colonne « Nom ».

Le dialog prefixe une colonne « # » et lit les symboles coches dans la colonne
symbole. Une colonne inseree du mauvais cote de celle-ci ferait injecter des noms
d'entreprise dans le champ d'analyse au lieu des tickers.
"""
import os

# Doit etre pose avant l'import de PyQt : ces tests instancient des widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from ui.dialogs import ScreenerResultsDialog

HEADERS = ["Symbole", "Nom", "Pays", "Variation (%)"]
ROWS = [
    ("AAPL", "Apple Inc.", "United States", 1.25),
    ("TSM", "Taiwan Semiconductor Manufacturing Company Limited", "Taiwan", -0.5),
]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_symboles_coches_sont_des_tickers(app) -> None:
    """Verrouille : selected_symbols() renvoie les tickers, jamais les noms."""
    dlg = ScreenerResultsDialog("Test", HEADERS, ROWS, preselect=True)

    assert dlg.selected_symbols() == ["AAPL", "TSM"]


def test_nom_long_tronque_avec_infobulle(app) -> None:
    """Verrouille : le nom long n'etire pas la colonne, il passe en infobulle."""
    dlg = ScreenerResultsDialog("Test", HEADERS, ROWS)
    colonne_nom = ["#"] + HEADERS
    cellule = dlg.table.item(1, colonne_nom.index("Nom"))

    assert cellule.text().endswith("…")
    assert cellule.toolTip() == ROWS[1][1]


def test_sans_colonne_nom_le_dialog_reste_fonctionnel(app) -> None:
    """Verrouille : les screeners qui n'exposent pas de nom continuent de marcher."""
    dlg = ScreenerResultsDialog("Test", ["Symbole", "Pays"], [("MSFT", "United States")])

    assert dlg.selected_symbols() == ["MSFT"]
