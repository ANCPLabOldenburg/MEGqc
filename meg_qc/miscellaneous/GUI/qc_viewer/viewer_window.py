"""
qc_viewer_viewer_window.py – Main QC Viewer window.

A QMainWindow with:
  • Left panel: file explorer (FilePanel)
  • Right panel: content viewer (ContentPanel)
  • Status bar with file info
  • Menu bar with useful actions

Integrated into meg_qc package as part of the QC Viewer module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QStatusBar,
    QLabel, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .file_panel import FilePanel
from .content_panel import ContentPanel


class QCViewerWindow(QMainWindow):
    """
    Dedicated MEG raw-data and QA/QC report viewer window.

    Can be launched standalone or from the MEGqc GUI.
    """

    def __init__(self, parent=None, initial_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("MEGqc – QC Viewer")
        self.resize(1400, 900)
        self.setMinimumSize(800, 500)

        self._initial_dir = initial_dir

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()

    # ================================================================ #
    # UI Construction                                                  #
    # ================================================================ #
    def _build_menu_bar(self):
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")

        act_open_dir = QAction("Open Directory…", self)
        act_open_dir.setShortcut("Ctrl+O")
        act_open_dir.triggered.connect(self._open_directory)
        file_menu.addAction(act_open_dir)

        act_open_file = QAction("Open File…", self)
        act_open_file.setShortcut("Ctrl+Shift+O")
        act_open_file.triggered.connect(self._open_file)
        file_menu.addAction(act_open_file)

        file_menu.addSeparator()

        act_close = QAction("Close Viewer", self)
        act_close.setShortcut("Ctrl+W")
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        # View menu
        view_menu = menu.addMenu("&View")

        act_ts = QAction("Switch to Time-Series", self)
        act_ts.triggered.connect(self._switch_to_timeseries)
        view_menu.addAction(act_ts)

        act_welcome = QAction("Show Welcome Page", self)
        act_welcome.triggered.connect(lambda: self._content_panel._stack.setCurrentIndex(0))
        view_menu.addAction(act_welcome)

        view_menu.addSeparator()

        act_toggle_explorer = QAction("Toggle File Explorer", self)
        act_toggle_explorer.setShortcut("Ctrl+E")
        act_toggle_explorer.triggered.connect(self._toggle_explorer)
        view_menu.addAction(act_toggle_explorer)

        # Tools menu
        tools_menu = menu.addMenu("&Tools")

        act_load_annot = QAction("Load MEGqc Annotations…", self)
        act_load_annot.triggered.connect(
            lambda: self._content_panel.get_timeseries_widget()._load_annotations_dialog()
        )
        tools_menu.addAction(act_load_annot)

        act_psd = QAction("Compute PSD", self)
        act_psd.triggered.connect(
            lambda: self._content_panel.get_timeseries_widget()._show_psd()
        )
        tools_menu.addAction(act_psd)

        act_topo = QAction("Show Topomap", self)
        act_topo.triggered.connect(
            lambda: self._content_panel.get_timeseries_widget()._show_topomap()
        )
        tools_menu.addAction(act_topo)

        # Help menu
        help_menu = menu.addMenu("&Help")
        act_about = QAction("About QC Viewer", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        act_shortcuts = QAction("Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_shortcuts)

    def _build_central(self):
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(8)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStyleSheet(
            "QSplitter::handle:horizontal {"
            "  background-color: #c0c0c0;"
            "  border: 1px solid #a0a0a0;"
            "  border-radius: 2px;"
            "  margin: 2px 0px;"
            "  min-width: 8px;"
            "}"
            "QSplitter::handle:horizontal:hover {"
            "  background-color: #2196F3;"
            "}"
        )

        # Left: file explorer
        self._file_panel = FilePanel(initial_dir=self._initial_dir)
        self._file_panel.fileSelected.connect(self._on_file_selected)
        self._file_panel.setMinimumWidth(150)
        self._file_panel.setMaximumWidth(800)

        # Right: content viewer
        self._content_panel = ContentPanel()
        self._content_panel.statusMessage.connect(self._update_status)
        self._content_panel.setMinimumWidth(300)

        self._splitter.addWidget(self._file_panel)
        self._splitter.addWidget(self._content_panel)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes([300, 1100])

        self.setCentralWidget(self._splitter)

    def _build_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label, stretch=1)
        self._file_label = QLabel("")
        self._status_bar.addPermanentWidget(self._file_label)

    # ================================================================ #
    # Slots                                                            #
    # ================================================================ #
    def _on_file_selected(self, filepath: str):
        self._file_label.setText(Path(filepath).name)
        self._content_panel.load_file(filepath)

    def _open_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select directory to browse")
        if d:
            self._file_panel.set_root_directory(d)

    def _open_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open file",
            filter="All supported (*.fif *.ds *.html *.tsv *.json *.csv *.ini *.edf *.bdf *.set *.vhdr);;"
                   "MEG files (*.fif *.ds *.con *.sqd *.4d *.raw *.bdf *.edf *.set *.vhdr);;"
                   "HTML (*.html *.htm);;"
                   "All files (*)")
        if f:
            self._content_panel.load_file(f)
            self._file_label.setText(Path(f).name)

    def _switch_to_timeseries(self):
        self._content_panel.show_timeseries()

    def _toggle_explorer(self):
        self._file_panel.setVisible(not self._file_panel.isVisible())

    def _update_status(self, msg: str):
        self._status_label.setText(msg)

    def _show_about(self):
        QMessageBox.about(
            self, "About QC Viewer",
            "<h3>MEGqc QC Viewer</h3>"
            "<p>A dedicated visualiser for MEG raw data, HTML reports, and MEGqc QA/QC annotations.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>File explorer with BIDS-aware filtering</li>"
            "<li>HTML report rendering via WebEngine</li>"
            "<li>MEG metadata display via MNE-Python</li>"
            "<li>Interactive time-series viewer with pyqtgraph</li>"
            "<li>Auto-detect available channel types (CTF .ds support)</li>"
            "<li>Individual channel selection within types</li>"
            "<li>Signal processing: filtering, resampling, PSD, topomaps</li>"
            "<li>MEGqc annotation overlays: STD, PTP, PSD, ECG, EOG, Muscle</li>"
            "<li>Epoch-level noisy/flat annotation matrices</li>"
            "<li>BIDS events.tsv overlay from EventSummary JSON</li>"
            "<li>Configurable event line thickness and color</li>"
            "<li>Background-threaded loading for all heavy operations</li>"
            "</ul>"
            "<p>Part of the MEGqc pipeline.</p>"
        )

    def _show_shortcuts(self):
        QMessageBox.information(
            self, "Keyboard Shortcuts",
            "<table>"
            "<tr><td><b>Ctrl+O</b></td><td>Open directory</td></tr>"
            "<tr><td><b>Ctrl+Shift+O</b></td><td>Open file</td></tr>"
            "<tr><td><b>Ctrl+E</b></td><td>Toggle file explorer</td></tr>"
            "<tr><td><b>Ctrl+W</b></td><td>Close viewer</td></tr>"
            "<tr><td><b>Mouse wheel</b></td><td>Scroll channels</td></tr>"
            "<tr><td><b>Click+drag</b></td><td>Pan time axis</td></tr>"
            "</table>"
        )

    # ================================================================ #
    # Public API                                                       #
    # ================================================================ #
    def set_directory(self, path: str):
        """Set the file explorer root directory."""
        self._file_panel.set_root_directory(path)

    def load_file(self, filepath: str):
        """Load a specific file in the content viewer."""
        self._content_panel.load_file(filepath)
        self._file_label.setText(Path(filepath).name)

    # ================================================================ #
    # Lifecycle                                                        #
    # ================================================================ #
    def closeEvent(self, event):
        self._content_panel.cleanup()
        super().closeEvent(event)

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                self.setPalette(app.palette())
            self._content_panel.refresh_theme()
        super().changeEvent(event)

    def refresh_theme(self):
        """Explicitly called by the parent GUI when the theme changes."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            self.setPalette(app.palette())
        self._content_panel.refresh_theme()
        self.update()

