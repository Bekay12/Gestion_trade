"""
Dialogs réutilisables de l'interface.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QAbstractItemView, QHeaderView, QDialogButtonBox,
)
from PyQt5.QtCore import Qt


class ScreenerResultsDialog(QDialog):
    """Affiche les résultats d'un screener dans une table triable avec sélection
    par cases à cocher (colonne symbole), et renvoie les symboles cochés.

    Remplace l'ancien dump texte en QMessageBox : l'utilisateur peut trier,
    cocher un sous-ensemble, puis injecter uniquement sa sélection.
    """

    def __init__(self, title, headers, rows, parent=None, symbol_col=0, preselect=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(580, 500)
        # Une colonne « # » (rang) est préfixée pour porter l'ordre du screener :
        # le symbole de l'appelant glisse donc d'une position vers la droite.
        self._symbol_col = symbol_col + 1
        full_headers = ["#"] + list(headers)

        layout = QVBoxLayout(self)

        info = QLabel(
            f"{len(rows)} résultat(s) — cochez les symboles à injecter dans le champ d'analyse, "
            "puis « Injecter la sélection ». Cliquez un en-tête pour trier (« # » = ordre du screener)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(len(rows), len(full_headers))
        self.table.setHorizontalHeaderLabels(full_headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)  # rempli d'abord, tri activé ensuite

        for r, row in enumerate(rows):
            full_row = (r + 1,) + tuple(row)   # rang 1-based = ordre du screener
            for c, val in enumerate(full_row):
                item = QTableWidgetItem()
                if val is None:
                    item.setText("n/a")
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    item.setData(Qt.EditRole, val)        # tri numérique correct
                    item.setText(self._fmt(val))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setText(str(val))
                if c == self._symbol_col:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if preselect else Qt.Unchecked)
                self.table.setItem(r, c, item)

        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)   # affiche l'ordre du screener (col #)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        if header.count() > 1:
            header.setSectionResizeMode(header.count() - 1, QHeaderView.Stretch)
        self.table.itemDoubleClicked.connect(self._toggle_row)
        layout.addWidget(self.table)

        sel_row = QHBoxLayout()
        btn_all = QPushButton("Tout cocher")
        btn_none = QPushButton("Tout décocher")
        btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        self._count_label = QLabel()
        sel_row.addWidget(self._count_label)
        layout.addLayout(sel_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Injecter la sélection")
        buttons.button(QDialogButtonBox.Cancel).setText("Annuler")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.table.itemChanged.connect(self._refresh_count)
        self._refresh_count()

    @staticmethod
    def _fmt(val):
        if isinstance(val, int):
            return f"{val:,}"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    def _toggle_row(self, item):
        sym_item = self.table.item(item.row(), self._symbol_col)
        if sym_item is not None:
            new = Qt.Unchecked if sym_item.checkState() == Qt.Checked else Qt.Checked
            sym_item.setCheckState(new)

    def _set_all(self, state):
        self.table.setSortingEnabled(False)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, self._symbol_col)
            if it is not None:
                it.setCheckState(state)
        self.table.setSortingEnabled(True)
        self._refresh_count()

    def _refresh_count(self, *args):
        self._count_label.setText(f"{len(self.selected_symbols())} sélectionné(s)")

    def selected_symbols(self):
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, self._symbol_col)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.text())
        return out
