"""
Verrouillage de la disposition du tableau de resultats (`MainWindow.merged_table`).

Les colonnes « Nom » et « Pays » ont ete inserees en positions 1 et 2, entre
« Symbole » et « Signal ». Une vingtaine d'acces designaient auparavant les
colonnes par un index litteral (coloration, stats par domaine, tableau
comparatif) : tout decalage se traduisait par une lecture silencieuse de la
mauvaise colonne. Ces tests verrouillent la correspondance cle -> index, le
remplissage effectif de la ligne, l'arrondi d'affichage des cellules chiffrees
et leur tri numerique.

Aucun acces reseau ni au store : `_name_map` et `_country_map` sont remplaces par
des dictionnaires.
"""
import os

# Doit etre pose avant l'import de PyQt : ces tests instancient des widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidget

from ui.main_window import (
    DECIMALES_MAX,
    MERGED_COL,
    MERGED_COLUMNS,
    MERGED_LABELS,
    MainWindow,
    formater_nombre,
)


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
    compute_domain_stats = MainWindow.compute_domain_stats

    def __init__(self, resultats, noms, pays=None):
        self.current_results = resultats
        self._noms = noms
        self._pays = pays or {}
        self.merged_table = QTableWidget()
        self.merged_table.setColumnCount(len(MERGED_LABELS))
        self.merged_table.setHorizontalHeaderLabels(MERGED_LABELS)
        self.merged_table.setSortingEnabled(True)
        self.fiab_threshold_spin = _FauxSpin()
        self.backtest_map = {}

    def _name_map(self, symboles):
        return dict(self._noms)

    def _country_map(self, symboles):
        return dict(self._pays)

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
    assert MERGED_COL["pays"] == 2
    assert MERGED_LABELS[MERGED_COL["nom"]] == "Nom"
    assert MERGED_LABELS[MERGED_COL["pays"]] == "Pays"


def test_nom_et_pays_remplis_depuis_le_store(app) -> None:
    """Verrouille : les colonnes Nom et Pays portent les valeurs du store."""
    fenetre = _FenetreMinimale([_signal()], {"AAPL": "Apple Inc."},
                               {"AAPL": "United States"})

    fenetre.update_results_table()

    assert fenetre.cellule(0, "symbole").text() == "AAPL"
    assert fenetre.cellule(0, "nom").text() == "Apple Inc."
    assert fenetre.cellule(0, "pays").text() == "United States"


def test_nom_et_pays_absents_du_store_affichent_na(app) -> None:
    """Verrouille : un symbole sans profil n'empeche pas l'affichage de la ligne."""
    fenetre = _FenetreMinimale([_signal("ZZZZ")], {})

    fenetre.update_results_table()

    assert fenetre.cellule(0, "nom").text() == "N/A"
    assert fenetre.cellule(0, "pays").text() == "N/A"
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


# ---------------------------------------------------------------------------
# Affichage des nombres : au plus DECIMALES_MAX decimales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valeur, attendu", [
    (5.5123456789012, "5.512346"),      # arrondi a 6 decimales
    (210.25, "210.25"),                 # pas de zeros ajoutes
    (0.0000123456789, "0.000012"),       # jamais de notation scientifique
    (2.00000000001, "2"),               # bruit de precision efface
    (0, "0"),
    (-1.987654321, "-1.987654"),
    (-0.00000001, "0"),                 # pas de « -0 »
    (1234567.89, "1234567.89"),         # grand nombre garde ses entiers
    ("N/A", "N/A"),                     # non numerique rendu tel quel
])
def test_formater_nombre(valeur, attendu) -> None:
    """Verrouille le format d'affichage des cellules chiffrees."""
    assert formater_nombre(valeur) == attendu


def test_aucune_cellule_ne_depasse_six_decimales(app) -> None:
    """Verrouille : sur une ligne entiere, aucune valeur n'affiche plus de 6 decimales."""
    signal = _signal()
    signal.update({
        "Score": 5.5123456789012,
        "Prix": 210.256789123456,
        "RSI": 61.0987654321,
        "Rev. Growth (%)": 0.123456789012345,
        "D/E Ratio": 1.23456789012345,
        "Market Cap (B$)": 3456.789123456789,
        "dPrice": 0.0000123456789,
        "dRSI": 2.00000000001,
    })
    fenetre = _FenetreMinimale([signal], {"AAPL": "Apple Inc."})

    fenetre.update_results_table()

    for colonne in range(fenetre.merged_table.columnCount()):
        cellule = fenetre.merged_table.item(0, colonne)
        if cellule is None:
            continue
        texte = cellule.text()
        assert "e-" not in texte and "e+" not in texte, texte
        if "." in texte:
            decimales = len(texte.split(".", 1)[1])
            assert decimales <= DECIMALES_MAX, f"{texte} ({decimales} decimales)"


def test_tri_numerique_et_non_lexicographique(app) -> None:
    """Verrouille le tri des colonnes chiffrees.

    Mesure avant correction : l'ordre croissant de « Score » donnait 10.2, 100,
    puis 9.5. QTableWidgetItem compare data(DisplayRole) ; le texte affiche etant
    desormais arrondi, la comparaison passe par la valeur reelle (CelluleNumerique).
    """
    signaux = []
    for symbole, score in (("AAA", 9.5), ("BBB", 10.2), ("CCC", 100.0), ("DDD", -3.75)):
        signal = _signal(symbole)
        signal["Score"] = score
        signaux.append(signal)
    fenetre = _FenetreMinimale(signaux, {})

    fenetre.update_results_table()
    fenetre.merged_table.sortItems(MERGED_COL["score"], Qt.AscendingOrder)

    ordre = [fenetre.cellule(ligne, "symbole").text()
             for ligne in range(fenetre.merged_table.rowCount())]
    assert ordre == ["DDD", "AAA", "BBB", "CCC"]


def test_statistiques_par_domaine_lisent_la_valeur_reelle(app) -> None:
    """Verrouille : les stats agregent la valeur portee, pas le texte arrondi.

    L'ancien `int(item.data(Qt.EditRole))` levait une ValueError sur un « 3.0 » et
    la ligne etait abandonnee en silence, faussant les totaux.
    """
    signal = _signal()
    signal.update({"NbTrades": 3.0, "Gagnants": 2.0, "Gain_total": 1234.567891234})
    fenetre = _FenetreMinimale([signal], {})

    fenetre.update_results_table()
    stats = fenetre.compute_domain_stats()

    assert stats["global"]["trades"] == 3
    assert stats["global"]["gagnants"] == 2
    # La precision complete est conservee, malgre l'arrondi de la cellule.
    assert stats["by_domain"]["Technologie"]["gain"] == pytest.approx(1234.567891234)
