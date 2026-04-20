"""
qc_viewer_content_panel.py – Right-hand content viewer.

A QStackedWidget that switches between:
  0. Welcome / placeholder page
  1. HTML report viewer (QWebEngineView)
  2. MEG metadata viewer (QTextEdit table)
  3. Time-series viewer (TimeSeriesWidget)
  4. Generic text/TSV/JSON viewer

All heavy I/O operations run in background QThread workers with
a loading overlay so the GUI remains responsive.

Integrated into meg_qc package as part of the QC Viewer module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel,
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QGroupBox, QFormLayout, QSplitter,
    QProgressBar, QApplication,
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QThread, QTimer

from .timeseries_widget import TimeSeriesWidget, _LoadWorker, _LoadingOverlay

# Try to import WebEngine; fall back gracefully if not installed.
# megqcGUI.py also imports this at module level (before QApplication) which
# is required on macOS for Chromium to initialise correctly.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    HAS_WEBENGINE = True
except Exception as _webengine_exc:
    import sys as _sys
    print(f"[MEGqc] WebEngine unavailable: {_webengine_exc}", file=_sys.stderr)
    QWebEngineView = None  # type: ignore[assignment,misc]
    QWebEngineSettings = None  # type: ignore[assignment,misc]
    HAS_WEBENGINE = False


# ── Localhost HTTP server for large Plotly reports ────────────────────────
# Chromium in PyQt6 imposes strict CSP restrictions on file:// URLs that can
# prevent large Plotly reports from rendering.  Serving over http://127.0.0.1
# bypasses all those restrictions (same approach used by Jupyter & Electron).
_LARGE_FILE_THRESHOLD = 5 * 1024 * 1024  # 5 MB


class _LocalHTTPServer:
    """Background HTTP server for large local HTML files."""
    _instance = None
    _thread = None

    @classmethod
    def serve(cls, filepath: str) -> str:
        """Return an ``http://127.0.0.1:<port>/filename`` URL."""
        import threading
        from http.server import HTTPServer, SimpleHTTPRequestHandler

        p = Path(filepath).resolve()
        directory = str(p.parent)
        filename = p.name

        # Shut down any previous server instance
        if cls._instance is not None:
            try:
                cls._instance.shutdown()
            except Exception:
                pass
            cls._instance = None

        handler = lambda *args, **kw: SimpleHTTPRequestHandler(
            *args, directory=directory, **kw
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cls._instance = server
        cls._thread = thread

        return f"http://127.0.0.1:{port}/{filename}"

    @classmethod
    def shutdown(cls):
        """Shut down the background server if running."""
        if cls._instance is not None:
            try:
                cls._instance.shutdown()
            except Exception:
                pass
            cls._instance = None
            cls._thread = None


# MEG file extensions recognised by MNE
MEG_EXTENSIONS = {".fif", ".ds", ".con", ".sqd", ".4d", ".pdf", ".raw", ".bdf", ".edf", ".set", ".vhdr", ".mef"}


class ContentPanel(QWidget):
    """Multi-format content viewer with threaded loading."""

    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_raw = None
        self._current_file: Optional[str] = None
        self._active_worker: Optional[_LoadWorker] = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()

        # Page 0: Welcome
        welcome = QWidget()
        wlay = QVBoxLayout(welcome)
        wlay.addStretch(1)
        lbl = QLabel("Select a file from the explorer to view its content.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 16px;")
        wlay.addWidget(lbl)
        lbl2 = QLabel("Supported: HTML reports  |  MEG data files  |  TSV/JSON/INI files")
        lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl2.setStyleSheet("font-size: 12px;")
        wlay.addWidget(lbl2)
        wlay.addStretch(1)
        self._stack.addWidget(welcome)  # index 0

        # Page 1: HTML viewer container
        # Following the BIDS-Manager pattern we create a *fresh*
        # QWebEngineView for every HTML file and destroy the previous one.
        self._html_container = QWidget()
        self._html_container_layout = QVBoxLayout(self._html_container)
        self._html_container_layout.setContentsMargins(0, 0, 0, 0)
        self._web_view = None  # created on demand
        if not HAS_WEBENGINE:
            fallback = QLabel(
                "PyQt6-WebEngine not installed.\n"
                "HTML files will be opened in the system browser."
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._html_container_layout.addWidget(fallback)
        self._stack.addWidget(self._html_container)  # index 1

        # Page 2: Metadata viewer
        self._meta_page = QWidget()
        meta_lay = QVBoxLayout(self._meta_page)

        self._meta_info = QTextEdit()
        self._meta_info.setReadOnly(True)

        self._btn_open_timeseries = QPushButton("Open in Time-Series Viewer")
        self._btn_open_timeseries.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; "
            "border-radius: 4px; font-size: 13px; font-weight: bold; } "
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self._btn_open_timeseries.clicked.connect(self._open_in_timeseries)

        meta_lay.addWidget(self._meta_info, stretch=1)
        meta_lay.addWidget(self._btn_open_timeseries)
        self._stack.addWidget(self._meta_page)  # index 2

        # Page 3: Time-series viewer
        self._ts_widget = TimeSeriesWidget()
        self._ts_widget.statusMessage.connect(self.statusMessage.emit)
        self._stack.addWidget(self._ts_widget)  # index 3

        # Page 4: Generic text viewer
        self._text_view = QWidget()
        text_lay = QVBoxLayout(self._text_view)
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_table = QTableWidget()
        self._text_table.setVisible(False)
        text_lay.addWidget(self._text_edit, stretch=1)
        text_lay.addWidget(self._text_table, stretch=1)
        self._stack.addWidget(self._text_view)  # index 4

        layout.addWidget(self._stack)

        # Loading overlay for metadata loading
        self._loading_overlay = _LoadingOverlay(self)

    # ------------------------------------------------------------------ #
    # File loading dispatch                                              #
    # ------------------------------------------------------------------ #
    def load_file(self, filepath: str):
        """Load a file and show the appropriate viewer page."""
        if self._current_file != filepath:
            self._release_previous_data()
        self._current_file = filepath
        p = Path(filepath)
        ext = p.suffix.lower()

        # Directory-based MEG formats (CTF .ds, etc.)
        if p.is_dir() and ext in (".ds",):
            self._load_meg_metadata(filepath)
        elif ext in (".html", ".htm"):
            self._load_html(filepath)
        elif ext in MEG_EXTENSIONS or p.name.endswith(".fif"):
            self._load_meg_metadata(filepath)
        elif ext == ".tsv":
            self._load_tsv(filepath)
        elif ext == ".json":
            self._load_json(filepath)
        elif ext in (".ini", ".cfg", ".txt", ".log", ".py", ".md"):
            self._load_text(filepath)
        elif ext == ".csv":
            self._load_tsv(filepath, sep=",")
        else:
            self._load_text(filepath)

    def _release_previous_data(self):
        """Free MEG raw data and web views when switching to a different file."""
        if self._current_raw is not None:
            del self._current_raw
            self._current_raw = None
        if hasattr(self._ts_widget, '_raw') and self._ts_widget._raw is not None:
            self._ts_widget.unload_data()
        self._destroy_web_view()

    # ------------------------------------------------------------------ #
    # Specific loaders                                                   #
    # ------------------------------------------------------------------ #
    def _load_html(self, filepath: str):
        """Display an HTML file.

        When PyQt6-WebEngine is available a fresh QWebEngineView is created
        for every file (BIDS-Manager pattern) giving full JavaScript /
        Plotly rendering.  When WebEngine is NOT available the file is
        opened in the system default browser.
        """
        if not HAS_WEBENGINE:
            import webbrowser
            webbrowser.open(str(Path(filepath).resolve()))
            self.statusMessage.emit(
                f"Opened in browser: {Path(filepath).name} "
                "(install PyQt6-WebEngine for in-app rendering)"
            )
            return

        # Destroy the previous web view (BIDS-Manager fresh-view pattern)
        self._destroy_web_view()

        view = QWebEngineView()

        # Enable local file access — PyQt6's Chromium disables these by
        # default, unlike PyQt5 where they were enabled.
        if QWebEngineSettings is not None:
            settings = view.settings()
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        # For large files, serve via localhost to bypass Chromium's file:// CSP
        resolved = str(Path(filepath).resolve())
        try:
            file_size = Path(filepath).stat().st_size
        except OSError:
            file_size = 0

        if file_size > _LARGE_FILE_THRESHOLD:
            url = QUrl(_LocalHTTPServer.serve(resolved))
        else:
            url = QUrl.fromLocalFile(resolved)

        view.setUrl(url)
        self._web_view = view
        self._html_container_layout.addWidget(view)

        self._stack.setCurrentIndex(1)
        self.statusMessage.emit(f"Loading HTML: {Path(filepath).name}")

    def _destroy_web_view(self):
        """Remove and schedule deletion of the current QWebEngineView.

        Mirrors the BIDS-Manager ``clear()`` pattern: the widget is removed
        from the layout and ``deleteLater()`` is called so Qt cleans it up
        safely in the next event-loop cycle.
        """
        if self._web_view is not None:
            self._html_container_layout.removeWidget(self._web_view)
            self._web_view.deleteLater()
            self._web_view = None

    def _load_meg_metadata(self, filepath: str):
        """Load MEG file metadata on the main thread (deferred).

        MNE's CTF reader uses memory-mapped I/O internally which is not
        thread-safe on macOS and causes SIGBUS when run inside a QThread.
        We therefore run the I/O on the main thread, using QTimer.singleShot
        so the loading overlay has time to render first.
        """
        self._loading_overlay.show_message(f"Loading MEG metadata…")
        self.statusMessage.emit(f"Loading MEG metadata: {Path(filepath).name}…")
        QApplication.processEvents()

        # Defer actual I/O to next event-loop tick so overlay paints first
        QTimer.singleShot(50, lambda: self._do_load_meg_metadata(filepath))

    def _do_load_meg_metadata(self, filepath: str):
        """Main-thread MEG metadata loader (called by deferred timer)."""
        try:
            import mne
            raw = mne.io.read_raw(filepath, preload=False, verbose=False)
            self._loading_overlay.hide()
            self._current_raw = raw
            self._display_meg_metadata(raw, filepath)
        except Exception as exc:
            self._loading_overlay.hide()
            msg = str(exc)
            self._meta_info.setPlainText(f"Error loading MEG file:\n{msg}")
            self._btn_open_timeseries.setEnabled(False)
            self._stack.setCurrentIndex(2)
            self.statusMessage.emit(f"Error: {msg.split(chr(10))[0]}")

    def _display_meg_metadata(self, raw, filepath: str):
        """Build and display the metadata HTML for a loaded raw object."""
        import mne
        info = raw.info
        tc = self._get_theme_colors()

        html = "<style>"
        html += f"body {{ font-family: sans-serif; font-size: 12px; background: {tc['body_bg']}; color: {tc['text']}; }}"
        html += f"table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}"
        html += f"th, td {{ border: 1px solid {tc['border']}; padding: 6px 10px; text-align: left; }}"
        html += f"th {{ background-color: {tc['th_bg']}; color: {tc['th_text']}; font-weight: bold; }}"
        html += f"tr:nth-child(even) {{ background-color: {tc['even_row']}; }}"
        html += f".section-title {{ color: {tc['title']}; font-size: 14px; margin-top: 12px; }}"
        html += "</style>"

        html += f"<h2>MEG Metadata: {Path(filepath).name}</h2>"

        html += "<p class='section-title'><b>Recording Info</b></p>"
        html += "<table>"
        rows = [
            ("File", filepath),
            ("# Channels", str(len(info["ch_names"]))),
            ("Sampling frequency", f"{info['sfreq']:.1f} Hz"),
            ("Duration", f"{raw.times[-1]:.1f} s ({raw.times[-1]/60:.1f} min)"),
            ("# Time points", str(raw.n_times)),
            ("Highpass filter", f"{info.get('highpass', 'N/A')} Hz"),
            ("Lowpass filter", f"{info.get('lowpass', 'N/A')} Hz"),
            ("Line frequency", f"{info.get('line_freq', 'N/A')} Hz"),
        ]
        if info.get("meas_date"):
            rows.append(("Measurement date", str(info["meas_date"])))
        if info.get("subject_info"):
            rows.append(("Subject info", str(info["subject_info"])))
        if info.get("experimenter"):
            rows.append(("Experimenter", str(info["experimenter"])))
        if info.get("description"):
            rows.append(("Description", str(info["description"])))
        for label, value in rows:
            html += f"<tr><th>{label}</th><td>{value}</td></tr>"
        html += "</table>"

        ch_type_counts = {}
        for i in range(len(info["ch_names"])):
            ct = mne.channel_type(info, i)
            ch_type_counts[ct] = ch_type_counts.get(ct, 0) + 1

        # Detect CTF system: has "mag" but no "grad", and has ref_meg or
        # the file is a .ds directory, or compensation_grade is set.
        is_ctf = (
            "mag" in ch_type_counts
            and "grad" not in ch_type_counts
            and (
                "ref_meg" in ch_type_counts
                or Path(filepath).suffix.lower() == ".ds"
                or getattr(raw, "compensation_grade", None) is not None
            )
        )

        html += "<p class='section-title'><b>Channel Types</b></p>"
        html += "<table><tr><th>Type</th><th>Count</th></tr>"
        for ct, count in sorted(ch_type_counts.items()):
            display_name = ct
            if is_ctf and ct == "mag":
                display_name = "mag (axial gradiometers)"
            html += f"<tr><td>{display_name}</td><td>{count}</td></tr>"
        html += "</table>"

        # Note for CTF datasets
        if is_ctf:
            html += ("<p style='color: #4fc3f7;'><b>Note:</b> This is a CTF dataset. "
                     "The MEG sensors are <b>axial gradiometers</b>. MNE labels them "
                     "as &ldquo;mag&rdquo; internally, but they are physically "
                     "gradiometers measuring dBz/dz.</p>")

        bads = info.get("bads", [])
        if bads:
            html += f"<p class='section-title'><b>Bad Channels ({len(bads)})</b></p>"
            html += f"<p>{', '.join(bads)}</p>"

        projs = info.get("projs", [])
        if projs:
            html += f"<p class='section-title'><b>Projectors ({len(projs)})</b></p>"
            html += "<table><tr><th>Name</th><th>Active</th><th>N vectors</th></tr>"
            for proj in projs:
                html += (f"<tr><td>{proj.get('desc', 'N/A')}</td>"
                         f"<td>{proj.get('active', 'N/A')}</td>"
                         f"<td>{proj['data']['nrow'] if 'data' in proj else 'N/A'}</td></tr>")
            html += "</table>"

        dig = info.get("dig")
        if dig:
            html += f"<p class='section-title'><b>Digitization Points: {len(dig)}</b></p>"

        dev_info = info.get("dev_head_t")
        if dev_info is not None:
            html += "<p class='section-title'><b>Device → Head Transform</b></p>"
            html += "<p>Available</p>"

        self._meta_info.setHtml(html)
        self._btn_open_timeseries.setEnabled(True)
        self._stack.setCurrentIndex(2)
        self.statusMessage.emit(
            f"MEG: {len(info['ch_names'])} ch, {info['sfreq']:.0f} Hz, "
            f"{raw.times[-1]:.1f}s | Types: {', '.join(sorted(ch_type_counts.keys()))}"
        )

    def _open_in_timeseries(self):
        """Load the current MEG file into the time-series viewer.

        Runs on the main thread (deferred via QTimer) because MNE's CTF
        reader is not thread-safe on macOS.
        """
        if self._current_raw is None:
            QMessageBox.information(self, "No data", "No MEG data loaded.")
            return

        self._loading_overlay.show_message("Preloading data for time-series viewer…")
        self.statusMessage.emit("Loading data into time-series viewer (preloading)…")
        QApplication.processEvents()

        filepath = self._current_file
        QTimer.singleShot(50, lambda: self._do_open_in_timeseries(filepath))

    def _do_open_in_timeseries(self, filepath: str):
        """Main-thread timeseries preloader (called by deferred timer)."""
        try:
            import mne
            raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self._loading_overlay.hide()
            self._ts_widget.load_raw(raw)
            if self._current_file:
                self._ts_widget.set_current_filepath(self._current_file)
            self._stack.setCurrentIndex(3)
            self._try_auto_load_annotations()
        except Exception as exc:
            self._loading_overlay.hide()
            msg = str(exc)
            QMessageBox.warning(self, "Load Error",
                                f"Failed to load data for visualisation:\n{msg}")
            self.statusMessage.emit(f"Error: {msg.split(chr(10))[0]}")


    def _try_auto_load_annotations(self):
        """Try to find and auto-load MEGqc annotations for the current file."""
        if self._current_file is None:
            return

        p = Path(self._current_file)
        for parent in p.parents:
            calc_dir = parent / "calculation"
            if calc_dir.is_dir():
                # Search in direct sub-XXX children (legacy layout)
                for sub_dir in calc_dir.iterdir():
                    if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                        prefixes = self._ts_widget._annotation_mgr.auto_detect_recordings(
                            str(sub_dir))
                        if prefixes:
                            self._ts_widget.load_annotations_from_dir(str(sub_dir), prefixes[0])
                            self.statusMessage.emit(
                                f"Auto-loaded annotations from {sub_dir.name}")
                            return
                    # Search inside modality subfolders (meg/, eeg/)
                    elif sub_dir.is_dir() and sub_dir.name in ("meg", "eeg"):
                        for mod_sub in sub_dir.iterdir():
                            if mod_sub.is_dir() and mod_sub.name.startswith("sub-"):
                                prefixes = self._ts_widget._annotation_mgr.auto_detect_recordings(
                                    str(mod_sub))
                                if prefixes:
                                    self._ts_widget.load_annotations_from_dir(
                                        str(mod_sub), prefixes[0])
                                    self.statusMessage.emit(
                                        f"Auto-loaded annotations from {sub_dir.name}/{mod_sub.name}")
                                    return

    def _load_tsv(self, filepath: str, sep="\t"):
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=500)
            self._text_edit.setVisible(False)
            self._text_table.setVisible(True)

            self._text_table.setRowCount(len(df))
            self._text_table.setColumnCount(len(df.columns))
            self._text_table.setHorizontalHeaderLabels([str(c) for c in df.columns])

            for i in range(len(df)):
                for j in range(len(df.columns)):
                    val = df.iloc[i, j]
                    item = QTableWidgetItem(str(val) if pd.notna(val) else "")
                    self._text_table.setItem(i, j, item)

            self._text_table.resizeColumnsToContents()
            self._stack.setCurrentIndex(4)
            self.statusMessage.emit(
                f"TSV: {len(df)} rows × {len(df.columns)} cols – {Path(filepath).name}")
        except Exception as e:
            self._show_text_fallback(filepath, f"Error reading TSV: {e}")

    def _load_json(self, filepath: str):
        try:
            with open(filepath) as f:
                data = json.load(f)
            pretty = json.dumps(data, indent=2, default=str)
            self._text_edit.setVisible(True)
            self._text_table.setVisible(False)
            self._text_edit.setPlainText(pretty)
            self._stack.setCurrentIndex(4)
            self.statusMessage.emit(f"JSON: {Path(filepath).name}")
        except Exception as e:
            self._show_text_fallback(filepath, f"Error reading JSON: {e}")

    def _load_text(self, filepath: str):
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                text = f.read(500_000)
            self._text_edit.setVisible(True)
            self._text_table.setVisible(False)
            self._text_edit.setPlainText(text)
            self._stack.setCurrentIndex(4)
            self.statusMessage.emit(f"Text: {Path(filepath).name}")
        except Exception as e:
            self._show_text_fallback(filepath, f"Error reading file: {e}")

    def _show_text_fallback(self, filepath, error_msg):
        self._text_edit.setVisible(True)
        self._text_table.setVisible(False)
        self._text_edit.setPlainText(f"{error_msg}\n\nFile: {filepath}")
        self._stack.setCurrentIndex(4)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def get_timeseries_widget(self) -> TimeSeriesWidget:
        return self._ts_widget

    def show_timeseries(self):
        self._stack.setCurrentIndex(3)

    def cleanup(self):
        self._release_previous_data()
        self._destroy_web_view()
        _LocalHTTPServer.shutdown()

    def refresh_theme(self):
        self._ts_widget.refresh_theme()
        if self._stack.currentIndex() == 2 and self._current_file:
            # Re-render metadata to pick up new colors
            if self._current_raw is not None:
                self._display_meg_metadata(self._current_raw, self._current_file)
        self.update()

    def _get_theme_colors(self) -> dict:
        app = QApplication.instance()
        from PyQt6.QtGui import QPalette
        dark = False
        if app:
            bg = app.palette().color(QPalette.ColorRole.Window)
            dark = bg.lightness() < 128
        if dark:
            return {
                "body_bg": "#1e1e1e", "text": "#e0e0e0",
                "th_bg": "#2a3a4a", "th_text": "#e0e0e0",
                "even_row": "#252525", "border": "#555555",
                "title": "#64b5f6",
            }
        return {
            "body_bg": "#ffffff", "text": "#000000",
            "th_bg": "#e8eef4", "th_text": "#000000",
            "even_row": "#f5f5f5", "border": "#cccccc",
            "title": "#1565c0",
        }

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.refresh_theme()
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._loading_overlay.isVisible():
            self._loading_overlay.setGeometry(self.rect())


