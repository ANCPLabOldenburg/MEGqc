"""
qc_viewer_file_panel.py – Left-hand file-explorer widget.

Uses QTreeView + QFileSystemModel with MEG/HTML/BIDS-aware filtering.
Emits ``fileSelected(str)`` when the user clicks a file.

Integrated into meg_qc package as part of the QC Viewer module.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QTreeView, QFileDialog, QLabel, QComboBox,
)
from PyQt6.QtCore import pyqtSignal, QDir, QModelIndex
from PyQt6.QtGui import QFileSystemModel


# Extensions we care about (shown in the filter)
MEG_EXTENSIONS = {".fif", ".ds", ".con", ".sqd", ".4d", ".pdf", ".raw", ".bdf", ".edf", ".set", ".vhdr", ".mef"}
REPORT_EXTENSIONS = {".html", ".htm"}
DATA_EXTENSIONS = {".tsv", ".csv", ".json", ".ini"}
ALL_EXTENSIONS = MEG_EXTENSIONS | REPORT_EXTENSIONS | DATA_EXTENSIONS


class FilePanel(QWidget):
    """File explorer panel with directory navigation."""

    fileSelected = pyqtSignal(str)  # absolute path of the selected file

    def __init__(self, initial_dir: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._build_ui(initial_dir)

    def _build_ui(self, initial_dir: Optional[str]):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Root directory…")
        self.path_edit.returnPressed.connect(self._on_path_entered)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_root)
        btn_up = QPushButton("↑")
        btn_up.setFixedWidth(30)
        btn_up.setToolTip("Go to parent directory")
        btn_up.clicked.connect(self._go_up)
        top.addWidget(self.path_edit, stretch=1)
        top.addWidget(btn_browse)
        top.addWidget(btn_up)
        layout.addLayout(top)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["All supported", "MEG files", "Reports (HTML)", "Data files (TSV/JSON)", "All files"])
        self.cmb_filter.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self.cmb_filter, stretch=1)
        layout.addLayout(filter_row)

        self.model = QFileSystemModel()
        self.model.setReadOnly(True)
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAnimated(True)
        self.tree.setSortingEnabled(True)
        self.tree.setHeaderHidden(False)
        for col in (1, 2, 3):
            self.tree.setColumnHidden(col, True)
        self.tree.clicked.connect(self._on_item_clicked)
        self.tree.doubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, stretch=1)

        root = initial_dir or QDir.homePath()
        self._set_root(root)
        self._apply_filter()

    def _set_root(self, path: str):
        p = Path(path)
        if not p.is_dir():
            p = p.parent
        root = str(p)
        self.model.setRootPath(root)
        self.tree.setRootIndex(self.model.index(root))
        self.path_edit.setText(root)

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select root directory", self.path_edit.text())
        if d:
            self._set_root(d)

    def _on_path_entered(self):
        p = Path(self.path_edit.text())
        if p.is_dir():
            self._set_root(str(p))

    def _go_up(self):
        current = Path(self.path_edit.text())
        parent = current.parent
        if parent != current:
            self._set_root(str(parent))

    def set_root_directory(self, path: str):
        """Public method to set root from outside."""
        self._set_root(path)

    def _apply_filter(self, _idx=None):
        choice = self.cmb_filter.currentText()
        if "All files" in choice:
            self.model.setNameFilters([])
        elif "MEG" in choice:
            self.model.setNameFilters([f"*{ext}" for ext in MEG_EXTENSIONS])
        elif "Reports" in choice:
            self.model.setNameFilters([f"*{ext}" for ext in REPORT_EXTENSIONS])
        elif "Data" in choice:
            self.model.setNameFilters([f"*{ext}" for ext in DATA_EXTENSIONS])
        else:
            self.model.setNameFilters([f"*{ext}" for ext in ALL_EXTENSIONS])

    @staticmethod
    def _is_meg_directory(p: Path) -> bool:
        """Return True if *p* is a directory-based MEG format (e.g. CTF .ds)."""
        return p.is_dir() and p.suffix.lower() in (".ds",)

    def _on_item_clicked(self, idx: QModelIndex):
        path = self.model.filePath(idx)
        p = Path(path)
        if p.is_file() or self._is_meg_directory(p):
            self.fileSelected.emit(path)

    def _on_item_double_clicked(self, idx: QModelIndex):
        path = self.model.filePath(idx)
        p = Path(path)
        if self._is_meg_directory(p):
            self.fileSelected.emit(path)
        elif p.is_dir():
            self._set_root(path)
        else:
            self.fileSelected.emit(path)

