"""
live_terminal.py – Floating, minimisable window that streams real-time
stdout/stderr from MEGqc background worker processes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QPlainTextEdit, QCheckBox, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QFont


class LiveTerminalDialog(QDialog):
    """Floating window that streams real-time stdout/stderr from worker processes.

    * Non-modal – stays open while the user interacts with the main window.
    * Minimisable – has a standard title-bar with a minimise button.
    * Singleton – only one instance is ever shown at a time; call
      ``LiveTerminalDialog.get_instance()`` instead of the constructor.
    * Auto-scroll – scrolls to the newest line unless the user unticks the
      "Auto-scroll" checkbox.
    """

    # Shared singleton reference so only one instance is ever alive.
    _instance: Optional["LiveTerminalDialog"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MEGqc — Live Terminal Output")
        self.resize(860, 540)
        self.setMinimumSize(480, 280)

        # Non-blocking; user can minimise, resize and interact with main window.
        self.setModal(False)
        # Ensure the window has a native title bar with minimise / close buttons.
        # Qt.WindowType.Tool hides the minimise button on some platforms,
        # so we use the plain Window type and add exactly the hints we need.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Terminal-style read-only text area with monospace font.
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Pick the best monospace font available on each platform.
        if sys.platform == "darwin":
            font_family = "Menlo"
        elif sys.platform == "win32":
            font_family = "Consolas"
        else:
            font_family = "DejaVu Sans Mono"
        mono = QFont(font_family, 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.output.setFont(mono)

        # Always use a dark terminal palette regardless of the app theme –
        # this widget intentionally mimics a terminal emulator.
        pal = self.output.palette()
        pal.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Text, QColor(204, 204, 204))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
        self.output.setPalette(pal)

        lay.addWidget(self.output, 1)

        # Controls row: auto-scroll checkbox + Clear + Save
        btn_row = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        btn_row.addWidget(self.chk_autoscroll)
        btn_row.addStretch(1)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.output.clear)
        btn_row.addWidget(self.btn_clear)

        self.btn_save = QPushButton("Save log…")
        self.btn_save.clicked.connect(self._save_log)
        btn_row.addWidget(self.btn_save)

        lay.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def append_text(self, text: str) -> None:
        """Append *text* to the terminal output and optionally auto-scroll."""
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        if self.chk_autoscroll.isChecked():
            sb = self.output.verticalScrollBar()
            sb.setValue(sb.maximum())

    @classmethod
    def get_instance(cls, parent=None) -> "LiveTerminalDialog":
        """Return the existing singleton or create a new one."""
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent)
        return cls._instance

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _save_log(self) -> None:
        """Prompt the user to save the accumulated output to a file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save terminal log",
            str(Path.home() / "megqc_terminal_log.txt"),
            "Text files (*.txt *.log);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.output.toPlainText())
        except Exception as exc:
            QMessageBox.warning(self, "Save log", f"Failed to save:\n{exc}")

