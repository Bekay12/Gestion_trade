"""
Verrouillage de la disposition du tableau de resultats (`MainWindow.merged_table`).

La colonne « Nom » a ete inseree en position 1, entre « Symbole » et « Signal ».
Une vingtaine d'acces designaient auparavant les colonnes par un index litteral
(coloration, stats par domaine, tableau comparatif) : tout decalage se traduisait
par une lecture silencieuse de la mauvaise colonne. Ces tests verrouillent la
correspondance cle -> index ET le remplissage effectif de la ligne.

Aucun acces reseau ni au store : `_name_map` est remplace par un dictionnaire.
"""
import os

# Doit etre pose avant l'import de PyQt : ces tests instancient des widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QTableWidget

from ui.main_window import MERGED_COL, MERGED_COLUMNS, MERGED_LABELS, MainWindow


@pytest.fixture(scope="module")
def app():
    """QApplication unique pour le module (Qt en interdit plusieurs)."""
    return QApplication.instance() or QApplication([])


class _FauxSpin:
    """Substitut de fiab_threshold_spin : aucun filtrage sur la fiabilite."""

    @staticmethod
    def value() -> int:
        return 0


class _FenetreMinimale:
    """Porte la vraie methode update_results_table sur un etat minimal.

    Instancier MainWindow chargerait les listes de symboles et la base ; seule la
    logique de remplissage du tableau est en jeu ici.
    """

    update_results_table = MainWindow.update_results_table

    def __init__(self, resultats, noms):
        self.current_results = resultats
        self._noms = noms
        self.merged_table = QTableWidget()
        self.merged_table.setColumnCount(len(MERGED_LABELS))
        self.merged_table.setHorizontalHeaderLabels(MERGED_LABELS)
        self.fiab_threshold_spin = _FauxSpin()
        self.backtest_map = {}

    def _name_map(self, symboles):
        return dict(self._noms)

    def cellule(self, ligne: int, cle: str):
        return self.merged_table.item(ligne, MERGED_COL[cle])


def _signal(symbole: str = "AAPL") -> dict:
    return {
        "Symbole": symbole,
        "Signal": "ACHAT",
        "Score": 5.5,
        "Prix": 210.25,
        "Tendance": "Hausse",
        "RSI": 61.0,
        "Volume moyen": 1234567.0,
        "Domaine": "Technologie",
        "CapRange": "Large",
        "Fiabilite": 72.0,
        "Consensus": "Strong Buy",
    }


def test_disposition_declaree_une_seule_fois() -> None:
    """Verrouille : cles uniques, labels alignes, « Nom » juste apres « Symbole »."""
    cles = [cle for cle, _ in MERGED_COLUMNS]

    assert len(set(cles)) == len(cles)
    assert len(MERGED_LABELS) == len(MERGED_COL) == len(MERGED_COLUMNS)
    assert MERGED_COL["symbole"] == 0
    assert MERGED_COL["nom"] == 1
    assert MERGED_LABELS[MERGED_COL["nom"]] == "Nom"


def test_nom_rempli_depuis_le_store(app) -> None:
    """Verrouille : la colonne Nom porte le nom d'entreprise du store."""
    fenetre = _FenetreMinimale([_signal()], {"AAPL": "Apple Inc."})

    fenetre.update_results_table()

    assert fenetre.cellule(0, "symbole").text() == "AAPL"
    assert fenetre.cellule(0, "nom").text() == "Apple Inc."


def test_nom_absent_du_store_affiche_na(app) -> None:
    """Verrouille : un symbole sans profil n'empeche pas l'affichage de la ligne."""
    fenetre = _FenetreMinimale([_signal("ZZZZ")], {})

    fenetre.update_results_table()

    assert fenetre.cellule(0, "nom").text() == "N/A"
    assert fenetre.cellule(0, "signal").text() == "ACHAT"


def test_nom_long_tronque_avec_infobulle(app) -> None:
    """Verrouille : nom tronque a l'affichage, complet en infobulle."""
    complet = "Taiwan Semiconductor Manufacturing Company Limited"
    fenetre = _FenetreMinimale([_signal("TSM")], {"TSM": complet})

    fenetre.update_results_table()

    cellule = fenetre.cellule(0, "nom")
    assert cellule.text().endswith("…")
    assert len(cellule.text()) < len(complet)
    assert cellule.toolTip() == complet


def test_colonnes_suivantes_non_decalees(app) -> None:
    """Verrouille l'absence de decalage : chaque valeur reste dans SA colonne.

    C'est le test qui aurait attrape l'insertion de « Nom » sans remaniement des
    index litteraux : chaque valeur se serait retrouvee dans la colonne voisine.
    """
    fenetre = _FenetreMinimale([_signal()], {"AAPL": "Apple Inc."})

    fenetre.update_results_table()

    assert fenetre.cellule(0, "signal").text() == "ACHAT"
    assert fenetre.cellule(0, "tendance").text() == "Hausse"
    assert fenetre.cellule(0, "domaine").text() == "Technologie"
    assert fenetre.cellule(0, "cap_range").text() == "Large"
    assert fenetre.cellule(0, "consensus").text() == "Strong Buy"
    assert float(fenetre.cellule(0, "prix").text()) == pytest.approx(210.25)
    assert float(fenetre.cellule(0, "fiabilite").text()) == pytest.approx(72.0)
