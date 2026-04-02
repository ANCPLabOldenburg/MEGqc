"""
qc_viewer_timeseries_widget.py – Interactive MEG time-series viewer built on pyqtgraph.

Features
--------
* Scrollable / zoomable multi-channel trace display
* Auto-detected channel-type filtering (CTF .ds support: no grad option when absent)
* Individual channel selection within a type
* Configurable visible time window with navigation controls
* Resampling & decimation
* Filtering: high-pass, low-pass, band-pass, notch (via MNE)
* PSD computation popup
* MEGqc annotation overlays (coloured regions + channel label tinting)
* Epoch grid overlay
* Events.tsv overlay from EventSummary JSON
* Thicker event lines with width/color controls
* QThread-based loading with loading overlay

Integrated into meg_qc package as part of the QC Viewer module.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox, QScrollBar,
    QMessageBox, QSlider, QSplitter, QScrollArea, QListWidget, QListWidgetItem,
    QLineEdit, QProgressBar, QColorDialog, QApplication, QToolTip,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSettings, QThread, pyqtSlot
from PyQt6.QtGui import QColor, QCursor

from .annotation_manager import (
    AnnotationManager, AnnotationSet, EpochAnnotation, EventMarker,
)


# Colour scheme for annotation overlays
ANNOTATION_COLORS = {
    "std_noisy":        (255, 100, 100, 50),
    "std_flat":         (100, 100, 255, 50),
    "ptp_noisy":        (255, 165, 0, 50),
    "ptp_flat":         (0, 200, 200, 50),
    "psd_noisy":        (200, 0, 200, 50),
    "ecg":              (255, 0, 0, 40),
    "eog":              (0, 128, 0, 40),
    "muscle":           (180, 180, 0, 60),
    "noisy_std_epochs": (255, 80, 80, 35),
    "flat_std_epochs":  (80, 80, 255, 35),
    "noisy_ptp_epochs": (255, 140, 0, 35),
    "flat_ptp_epochs":  (0, 180, 180, 35),
    "epoch_windows":    (160, 160, 160, 30),
    "ecg_events":       (220, 50, 50, 70),
    "eog_events":       (50, 160, 50, 70),
}

CHANNEL_LABEL_COLORS = {
    "std_noisy": "#ff4444",
    "std_flat":  "#4444ff",
    "ptp_noisy": "#ff8800",
    "ptp_flat":  "#00aaaa",
    "psd_noisy": "#cc00cc",
    "ecg":       "#ff0000",
    "eog":       "#008800",
}


# ================================================================ #
# Background worker thread for heavy I/O operations               #
# ================================================================ #
class _LoadWorker(QThread):
    """Runs a callable in a background thread and emits result/error."""
    finished = pyqtSignal(object)     # result object
    errored = pyqtSignal(str)         # error message

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.errored.emit(f"{exc}\n{traceback.format_exc()}")


class _LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with spinner and message."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 120);")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("Loading…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        lay.addWidget(self._label)
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setFixedWidth(250)
        lay.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.hide()

    def show_message(self, msg: str = "Loading…"):
        self._label.setText(msg)
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()

    def resizeEvent(self, event):
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)


class TimeSeriesWidget(QWidget):
    """Interactive multi-channel MEG time-series viewer."""

    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw = None
        self._data_cache = {}
        self._sfreq = 1000.0
        self._ch_names: List[str] = []
        self._ch_types: List[str] = []
        self._available_ch_types: List[str] = []
        self._is_ctf: bool = False
        self._duration = 0.0
        self._n_samples = 0

        # Display state
        self._visible_channels = 20
        self._channel_offset = 0
        self._time_start = 0.0
        self._time_window = 10.0
        self._scale_factor = 1.0
        self._active_ch_type = "all"
        self._selected_channels: Optional[Set[str]] = None  # None = all
        self._normalize = False  # False = raw mode (default), True = per-channel normalization

        self._display_indices: List[int] = []
        self._shown_ch_info: List = []  # (y_offset, ch_name, ch_type) for hover

        # Annotation state
        self._annotation_mgr = AnnotationManager()
        self._annotations: Optional[AnnotationSet] = None
        self._enabled_annotations: Dict[str, bool] = {}
        self._overlay_items: List = []
        self._ecg_top_n = 10
        self._eog_top_n = 10

        # Event display
        self._show_events = False
        self._show_bids_events = False
        self._events_from_raw: list = []
        self._active_stim_channel = "all"
        self._event_line_width = 2
        self._event_color_override: Optional[QColor] = None

        # Annotation overlay alpha
        self._annotation_alpha = 50

        # Epoch window overlay state
        self._show_epoch_windows = False
        self._epoch_window_alpha = 30

        # ECG/EOG event overlay state
        self._show_ecg_events = False
        self._show_eog_events = False

        # Overlay border lines
        self._show_overlay_borders = False

        # Processing state
        self._current_filter = None
        self._notch_freq = None
        self._resample_freq = None
        self._current_filepath: Optional[str] = None
        self._external_annot_paths: List[str] = []

        # Worker thread reference
        self._active_worker: Optional[_LoadWorker] = None

        # Preferences
        self._settings = QSettings("ANCP", "MEGqc_Viewer")
        self._dark_plot = self._settings.value("viewer/dark_plot", False, type=bool)

        self._build_ui()

    # ================================================================ #
    # UI Construction                                                  #
    # ================================================================ #
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)

        # Plot area
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._label_area = QWidget()
        self._label_layout = QVBoxLayout(self._label_area)
        self._label_layout.setContentsMargins(2, 0, 2, 0)
        self._label_layout.setSpacing(0)
        label_scroll = QScrollArea()
        label_scroll.setWidget(self._label_area)
        label_scroll.setWidgetResizable(True)
        label_scroll.setFixedWidth(90)
        label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._plot_widget = pg.PlotWidget()
        self._apply_plot_theme()
        self._plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setMouseEnabled(x=True, y=False)
        self._plot_widget.getPlotItem().getAxis("left").setWidth(0)
        self._plot_widget.getPlotItem().getAxis("left").setStyle(showValues=False)
        self._plot_widget.wheelEvent = self._on_plot_wheel

        # Mouse hover for channel name tooltip
        self._hover_proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved, rateLimit=30,
            slot=self._on_mouse_moved)

        splitter.addWidget(label_scroll)
        splitter.addWidget(self._plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.sld_channels = QScrollBar(Qt.Orientation.Vertical)
        self.sld_channels.setToolTip("Scroll channels")
        self.sld_channels.valueChanged.connect(self._on_channel_scroll)

        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(0)
        plot_row.addWidget(splitter, stretch=1)
        plot_row.addWidget(self.sld_channels)

        plot_container = QWidget()
        plot_container.setLayout(plot_row)
        main_layout.addWidget(plot_container, stretch=1)

        nav = self._build_navigation()
        main_layout.addWidget(nav)

        self._annot_panel = self._build_annotation_panel()
        main_layout.addWidget(self._annot_panel)

        # Loading overlay (must be last to stay on top)
        self._loading_overlay = _LoadingOverlay(self)

    def _build_toolbar(self) -> QWidget:
        container = QWidget()
        container_lay = QVBoxLayout(container)
        container_lay.setContentsMargins(0, 0, 0, 0)
        container_lay.setSpacing(0)

        # Row 1: Display
        row1 = QWidget()
        lay1 = QHBoxLayout(row1)
        lay1.setContentsMargins(4, 2, 4, 2)
        lay1.setSpacing(6)

        lay1.addWidget(QLabel("Ch type:"))
        self.cmb_ch_type = QComboBox()
        self.cmb_ch_type.addItems(["all", "mag", "grad", "eeg", "eog", "ecg", "stim", "misc"])
        self.cmb_ch_type.currentTextChanged.connect(self._on_ch_type_changed)
        lay1.addWidget(self.cmb_ch_type)

        # Individual channel selection button
        self.btn_select_channels = QPushButton("Select Channels…")
        self.btn_select_channels.setToolTip("Pick specific channels to display")
        self.btn_select_channels.clicked.connect(self._open_channel_selector)
        lay1.addWidget(self.btn_select_channels)

        lay1.addWidget(QLabel("Channels:"))
        self.spn_n_channels = QSpinBox()
        self.spn_n_channels.setRange(1, 500)
        self.spn_n_channels.setValue(20)
        self.spn_n_channels.valueChanged.connect(self._on_n_channels_changed)
        lay1.addWidget(self.spn_n_channels)

        lay1.addWidget(QLabel("Scale:"))
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(0.01, 1000.0)
        self.spn_scale.setValue(1.0)
        self.spn_scale.setSingleStep(0.1)
        self.spn_scale.setDecimals(2)
        self.spn_scale.valueChanged.connect(self._on_scale_changed)
        lay1.addWidget(self.spn_scale)

        lay1.addWidget(QLabel("Window (s):"))
        self.spn_time_window = QDoubleSpinBox()
        self.spn_time_window.setRange(0.1, 600.0)
        self.spn_time_window.setValue(10.0)
        self.spn_time_window.setSingleStep(1.0)
        self.spn_time_window.valueChanged.connect(self._on_time_window_changed)
        lay1.addWidget(self.spn_time_window)

        lay1.addWidget(self._vsep())

        self.chk_dark_plot = QCheckBox("Dark plot")
        self.chk_dark_plot.setToolTip("Toggle dark/light background for the signal plot")
        self.chk_dark_plot.setChecked(self._dark_plot)
        self.chk_dark_plot.toggled.connect(self._on_dark_plot_toggled)
        lay1.addWidget(self.chk_dark_plot)

        self.chk_normalize = QCheckBox("Normalize")
        self.chk_normalize.setToolTip(
            "Normalize each channel independently to prevent overlap.\n"
            "When unchecked, raw signals are shown with fixed spacing (may overlap).")
        self.chk_normalize.setChecked(self._normalize)
        self.chk_normalize.toggled.connect(self._on_normalize_toggled)
        lay1.addWidget(self.chk_normalize)

        lay1.addWidget(self._vsep())

        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.setToolTip("Reset zoom, scroll, and filters to defaults")
        self.btn_reset_view.clicked.connect(self._reset_view)
        lay1.addWidget(self.btn_reset_view)

        self.btn_close_data = QPushButton("Close Data")
        self.btn_close_data.setToolTip("Unload current data from memory")
        self.btn_close_data.clicked.connect(self.unload_data)
        lay1.addWidget(self.btn_close_data)

        lay1.addStretch(1)
        container_lay.addWidget(row1)

        # Row 2: Signal processing
        row2 = QWidget()
        lay2 = QHBoxLayout(row2)
        lay2.setContentsMargins(4, 2, 4, 2)
        lay2.setSpacing(6)

        lay2.addWidget(QLabel("HP (Hz):"))
        self.spn_hp = QDoubleSpinBox()
        self.spn_hp.setRange(0.0, 500.0)
        self.spn_hp.setValue(0.0)
        self.spn_hp.setDecimals(1)
        self.spn_hp.setSpecialValueText("Off")
        lay2.addWidget(self.spn_hp)

        lay2.addWidget(QLabel("LP (Hz):"))
        self.spn_lp = QDoubleSpinBox()
        self.spn_lp.setRange(0.0, 5000.0)
        self.spn_lp.setValue(0.0)
        self.spn_lp.setDecimals(1)
        self.spn_lp.setSpecialValueText("Off")
        lay2.addWidget(self.spn_lp)

        lay2.addWidget(QLabel("Notch (Hz):"))
        self.spn_notch = QDoubleSpinBox()
        self.spn_notch.setRange(0.0, 1000.0)
        self.spn_notch.setValue(0.0)
        self.spn_notch.setDecimals(1)
        self.spn_notch.setSpecialValueText("Off")
        lay2.addWidget(self.spn_notch)

        self.btn_apply_filter = QPushButton("Apply Filter")
        self.btn_apply_filter.clicked.connect(self._apply_filters)
        lay2.addWidget(self.btn_apply_filter)

        self.btn_reset_filter = QPushButton("Reset Filters")
        self.btn_reset_filter.clicked.connect(self._reset_filters)
        lay2.addWidget(self.btn_reset_filter)

        lay2.addWidget(self._vsep())

        lay2.addWidget(QLabel("Resample (Hz):"))
        self.spn_resample = QDoubleSpinBox()
        self.spn_resample.setRange(0.0, 10000.0)
        self.spn_resample.setValue(0.0)
        self.spn_resample.setDecimals(0)
        self.spn_resample.setSpecialValueText("Off")
        lay2.addWidget(self.spn_resample)

        self.btn_resample = QPushButton("Resample")
        self.btn_resample.clicked.connect(self._apply_resample)
        lay2.addWidget(self.btn_resample)

        lay2.addWidget(self._vsep())

        self.btn_psd = QPushButton("PSD")
        self.btn_psd.setToolTip("Compute and display Power Spectral Density")
        self.btn_psd.clicked.connect(self._show_psd)
        lay2.addWidget(self.btn_psd)

        self.btn_topo = QPushButton("Topomap")
        self.btn_topo.setToolTip("Show sensor topography of current time window")
        self.btn_topo.clicked.connect(self._show_topomap)
        lay2.addWidget(self.btn_topo)

        lay2.addStretch(1)
        container_lay.addWidget(row2)

        return container

    def _build_navigation(self) -> QWidget:
        nav = QWidget()
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)

        self.btn_start = QPushButton("|<<")
        self.btn_start.setFixedWidth(36)
        self.btn_start.clicked.connect(lambda: self._navigate("start"))
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(lambda: self._navigate("prev"))
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(lambda: self._navigate("next"))
        self.btn_end = QPushButton(">>|")
        self.btn_end.setFixedWidth(36)
        self.btn_end.clicked.connect(lambda: self._navigate("end"))

        self.sld_time = QSlider(Qt.Orientation.Horizontal)
        self.sld_time.setRange(0, 1000)
        self.sld_time.valueChanged.connect(self._on_time_slider)
        self.lbl_time_pos = QLabel("0.0 / 0.0 s")

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.sld_time, stretch=1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_end)
        lay.addWidget(self.lbl_time_pos)
        return nav

    def _build_annotation_panel(self) -> QGroupBox:
        box = QGroupBox("MEGqc Annotation Overlays")
        outer_lay = QHBoxLayout(box)
        outer_lay.setContentsMargins(4, 4, 4, 4)
        outer_lay.setSpacing(12)

        self._annot_checkboxes: Dict[str, QCheckBox] = {}

        # Group 1: Channel Quality
        grp_quality = QGroupBox("Channel Quality")
        quality_lay = QVBoxLayout(grp_quality)
        quality_lay.setContentsMargins(4, 8, 4, 4)
        quality_lay.setSpacing(2)
        for key, label in [("std_noisy", "STD Noisy"), ("std_flat", "STD Flat"),
                           ("ptp_noisy", "PTP Noisy"), ("ptp_flat", "PTP Flat"),
                           ("psd_noisy", "PSD Noisy")]:
            cb = QCheckBox(label)
            cb.setEnabled(False)
            r, g, b, _ = ANNOTATION_COLORS.get(key, (200, 200, 200, 50))
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); font-weight: bold; }}")
            cb.toggled.connect(lambda checked, k=key: self._on_annotation_toggled(k, checked))
            quality_lay.addWidget(cb)
            self._annot_checkboxes[key] = cb
            self._enabled_annotations[key] = False
        outer_lay.addWidget(grp_quality)

        # Group 2: Physiological
        grp_physio = QGroupBox("Physiological")
        physio_lay = QVBoxLayout(grp_physio)
        physio_lay.setContentsMargins(4, 8, 4, 4)
        physio_lay.setSpacing(2)

        for metric, label, color_rgb, top_attr in [
            ("ecg", "ECG", (255, 0, 0), "_ecg_top_n"),
            ("eog", "EOG", (0, 128, 0), "_eog_top_n"),
        ]:
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setEnabled(False)
            cb.setStyleSheet(f"QCheckBox {{ color: rgb{color_rgb}; font-weight: bold; }}")
            cb.toggled.connect(lambda checked, k=metric: self._on_annotation_toggled(k, checked))
            row.addWidget(cb)
            row.addWidget(QLabel("Top:"))
            spn = QSpinBox()
            spn.setRange(1, 500)
            spn.setValue(getattr(self, top_attr))
            spn.setToolTip(f"Number of most-affected {label} channels to highlight")
            spn.setFixedWidth(55)
            if metric == "ecg":
                spn.valueChanged.connect(self._on_ecg_top_changed)
                self.spn_ecg_top = spn
            else:
                spn.valueChanged.connect(self._on_eog_top_changed)
                self.spn_eog_top = spn
            row.addWidget(spn)
            physio_lay.addLayout(row)
            self._annot_checkboxes[metric] = cb
            self._enabled_annotations[metric] = False

        cb_muscle = QCheckBox("Muscle Artifacts")
        cb_muscle.setEnabled(False)
        cb_muscle.setStyleSheet("QCheckBox { color: rgb(180,180,0); font-weight: bold; }")
        cb_muscle.toggled.connect(lambda checked: self._on_annotation_toggled("muscle", checked))
        physio_lay.addWidget(cb_muscle)
        self._annot_checkboxes["muscle"] = cb_muscle
        self._enabled_annotations["muscle"] = False

        # ECG detected events (vertical segments)
        cb_ecg_events = QCheckBox("ECG Events")
        cb_ecg_events.setEnabled(False)
        cb_ecg_events.setStyleSheet("QCheckBox { color: rgb(220,50,50); font-weight: bold; }")
        cb_ecg_events.toggled.connect(lambda checked: self._on_ecg_events_toggled(checked))
        physio_lay.addWidget(cb_ecg_events)
        self._annot_checkboxes["ecg_events"] = cb_ecg_events
        self._enabled_annotations["ecg_events"] = False

        # EOG detected events (vertical segments)
        cb_eog_events = QCheckBox("EOG Events")
        cb_eog_events.setEnabled(False)
        cb_eog_events.setStyleSheet("QCheckBox { color: rgb(50,160,50); font-weight: bold; }")
        cb_eog_events.toggled.connect(lambda checked: self._on_eog_events_toggled(checked))
        physio_lay.addWidget(cb_eog_events)
        self._annot_checkboxes["eog_events"] = cb_eog_events
        self._enabled_annotations["eog_events"] = False

        outer_lay.addWidget(grp_physio)

        # Group 3: Epoch-Level
        grp_epochs = QGroupBox("Epoch-Level")
        epochs_lay = QVBoxLayout(grp_epochs)
        epochs_lay.setContentsMargins(4, 8, 4, 4)
        epochs_lay.setSpacing(2)
        for key, label in [("noisy_std_epochs", "Noisy (STD)"), ("flat_std_epochs", "Flat (STD)"),
                           ("noisy_ptp_epochs", "Noisy (PTP)"), ("flat_ptp_epochs", "Flat (PTP)")]:
            cb = QCheckBox(label)
            cb.setEnabled(False)
            r, g, b, _ = ANNOTATION_COLORS.get(key, (200, 200, 200, 50))
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); font-weight: bold; }}")
            cb.toggled.connect(lambda checked, k=key: self._on_annotation_toggled(k, checked))
            epochs_lay.addWidget(cb)
            self._annot_checkboxes[key] = cb
            self._enabled_annotations[key] = False

        # Epoch windows overlay (analysis windows)
        cb_epoch_windows = QCheckBox("Epoch Windows")
        cb_epoch_windows.setEnabled(False)
        cb_epoch_windows.setStyleSheet("QCheckBox { color: rgb(160,160,160); font-weight: bold; }")
        cb_epoch_windows.toggled.connect(self._on_epoch_windows_toggled)
        epochs_lay.addWidget(cb_epoch_windows)
        self._annot_checkboxes["epoch_windows"] = cb_epoch_windows
        self._enabled_annotations["epoch_windows"] = False

        # Epoch window alpha slider
        ew_alpha_row = QHBoxLayout()
        ew_alpha_row.addWidget(QLabel("Win α:"))
        self.sld_epoch_window_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sld_epoch_window_alpha.setRange(5, 150)
        self.sld_epoch_window_alpha.setValue(self._epoch_window_alpha)
        self.sld_epoch_window_alpha.setToolTip("Adjust transparency of epoch window overlays")
        self.sld_epoch_window_alpha.valueChanged.connect(self._on_epoch_window_alpha_changed)
        ew_alpha_row.addWidget(self.sld_epoch_window_alpha, stretch=1)
        self.lbl_ew_alpha_val = QLabel(str(self._epoch_window_alpha))
        self.lbl_ew_alpha_val.setFixedWidth(28)
        ew_alpha_row.addWidget(self.lbl_ew_alpha_val)
        epochs_lay.addLayout(ew_alpha_row)

        outer_lay.addWidget(grp_epochs)

        # Group 4: Events (with thickness and color controls)
        grp_events = QGroupBox("Events")
        events_lay = QVBoxLayout(grp_events)
        events_lay.setContentsMargins(4, 8, 4, 4)
        events_lay.setSpacing(2)
        self.chk_show_events = QCheckBox("Show Stim Events")
        self.chk_show_events.setEnabled(False)
        self.chk_show_events.setStyleSheet("QCheckBox { color: rgb(255,215,0); font-weight: bold; }")
        self.chk_show_events.toggled.connect(self._on_show_events_toggled)
        events_lay.addWidget(self.chk_show_events)

        self.chk_show_bids_events = QCheckBox("Show BIDS Events")
        self.chk_show_bids_events.setEnabled(False)
        self.chk_show_bids_events.setStyleSheet("QCheckBox { color: rgb(100,200,255); font-weight: bold; }")
        self.chk_show_bids_events.toggled.connect(self._on_show_bids_events_toggled)
        events_lay.addWidget(self.chk_show_bids_events)

        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Channel:"))
        self.cmb_stim_channel = QComboBox()
        self.cmb_stim_channel.addItem("all")
        self.cmb_stim_channel.setEnabled(False)
        self.cmb_stim_channel.currentTextChanged.connect(self._on_stim_channel_changed)
        ev_row.addWidget(self.cmb_stim_channel, stretch=1)
        events_lay.addLayout(ev_row)

        # Event line thickness control
        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Line width:"))
        self.spn_event_width = QSpinBox()
        self.spn_event_width.setRange(1, 10)
        self.spn_event_width.setValue(self._event_line_width)
        self.spn_event_width.setToolTip("Event line thickness in pixels")
        self.spn_event_width.setFixedWidth(50)
        self.spn_event_width.valueChanged.connect(self._on_event_width_changed)
        thickness_row.addWidget(self.spn_event_width)
        events_lay.addLayout(thickness_row)

        # Event color override
        color_row = QHBoxLayout()
        self.btn_event_color = QPushButton("Event Color…")
        self.btn_event_color.setToolTip("Override event line color (clear to use per-ID colors)")
        self.btn_event_color.clicked.connect(self._pick_event_color)
        self.btn_event_color_reset = QPushButton("Reset")
        self.btn_event_color_reset.setFixedWidth(50)
        self.btn_event_color_reset.clicked.connect(self._reset_event_color)
        color_row.addWidget(self.btn_event_color)
        color_row.addWidget(self.btn_event_color_reset)
        events_lay.addLayout(color_row)

        outer_lay.addWidget(grp_events)

        # Group 5: Actions
        grp_actions = QGroupBox("Actions")
        actions_lay = QVBoxLayout(grp_actions)
        actions_lay.setContentsMargins(4, 8, 4, 4)
        actions_lay.setSpacing(4)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Overlay α:"))
        self.sld_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sld_alpha.setRange(5, 255)
        self.sld_alpha.setValue(self._annotation_alpha)
        self.sld_alpha.setToolTip("Adjust transparency of annotation overlays")
        self.sld_alpha.valueChanged.connect(self._on_alpha_changed)
        alpha_row.addWidget(self.sld_alpha, stretch=1)
        self.lbl_alpha_val = QLabel(str(self._annotation_alpha))
        self.lbl_alpha_val.setFixedWidth(28)
        alpha_row.addWidget(self.lbl_alpha_val)
        actions_lay.addLayout(alpha_row)

        # Border lines toggle for overlay regions
        self.chk_overlay_borders = QCheckBox("Show Overlay Borders")
        self.chk_overlay_borders.setToolTip(
            "Draw border lines on segment/epoch overlay edges "
            "to distinguish adjacent overlays"
        )
        self.chk_overlay_borders.toggled.connect(self._on_overlay_borders_toggled)
        actions_lay.addWidget(self.chk_overlay_borders)

        self.btn_scan_annot = QPushButton("Scan for Annotations")
        self.btn_scan_annot.setToolTip("Automatically scan BIDS derivatives for annotations matching the loaded file")
        self.btn_scan_annot.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 4px 8px; border-radius: 3px; } "
            "QPushButton:hover { background-color: #388E3C; }")
        self.btn_scan_annot.clicked.connect(self._scan_annotations)
        actions_lay.addWidget(self.btn_scan_annot)

        self.btn_load_annot = QPushButton("Load Annotations...")
        self.btn_load_annot.clicked.connect(self._load_annotations_dialog)
        actions_lay.addWidget(self.btn_load_annot)

        self.btn_ext_path = QPushButton("Set External Path...")
        self.btn_ext_path.setToolTip("Set an external output directory where MEGqc derivatives may reside")
        self.btn_ext_path.clicked.connect(self._set_external_path_dialog)
        actions_lay.addWidget(self.btn_ext_path)

        self.lbl_ext_path = QLabel("")
        self.lbl_ext_path.setWordWrap(True)
        self.lbl_ext_path.setStyleSheet("font-size: 9px;")
        actions_lay.addWidget(self.lbl_ext_path)

        btn_row = QHBoxLayout()
        self.btn_all_on = QPushButton("All On")
        self.btn_all_on.clicked.connect(self._enable_all_annotations)
        btn_row.addWidget(self.btn_all_on)
        self.btn_all_off = QPushButton("All Off")
        self.btn_all_off.clicked.connect(self._disable_all_annotations)
        btn_row.addWidget(self.btn_all_off)
        actions_lay.addLayout(btn_row)
        actions_lay.addStretch(1)
        outer_lay.addWidget(grp_actions)

        outer_lay.addStretch(1)
        return box

    @staticmethod
    def _vsep() -> QLabel:
        sep = QLabel("|")
        sep.setStyleSheet("color: gray;")
        return sep

    # ================================================================ #
    # Channel Selection Dialog                                         #
    # ================================================================ #
    def _open_channel_selector(self):
        """Open a dialog to select individual channels."""
        if not self._ch_names:
            QMessageBox.information(self, "Channels", "No data loaded.")
            return

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Channels")
        dlg.resize(350, 500)
        lay = QVBoxLayout(dlg)

        # Search box
        search = QLineEdit()
        search.setPlaceholderText("Search channels…")
        lay.addWidget(search)

        # Type filter
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        cmb_type = QComboBox()
        display_ch_types = []
        for ct in self._available_ch_types:
            if self._is_ctf and ct == "mag":
                display_ch_types.append("mag (axial grad)")
            else:
                display_ch_types.append(ct)
        cmb_type.addItems(["all"] + display_ch_types)
        type_row.addWidget(cmb_type, stretch=1)
        lay.addLayout(type_row)

        # Channel list
        ch_list = QListWidget()
        ch_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i, name in enumerate(self._ch_names):
            ct_display = self._ch_types[i]
            if self._is_ctf and ct_display == "mag":
                ct_display = "mag (axial grad)"
            item = QListWidgetItem(f"{name}  [{ct_display}]")
            item.setData(Qt.ItemDataRole.UserRole, name)
            if self._selected_channels is None or name in self._selected_channels:
                item.setSelected(True)
            ch_list.addItem(item)
        lay.addWidget(ch_list, stretch=1)

        # Quick buttons
        qrow = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Select None")
        btn_all.clicked.connect(lambda: ch_list.selectAll())
        btn_none.clicked.connect(lambda: ch_list.clearSelection())
        qrow.addWidget(btn_all)
        qrow.addWidget(btn_none)
        lay.addLayout(qrow)

        lbl_count = QLabel(f"{ch_list.count()} channels")
        lay.addWidget(lbl_count)

        def _filter_channels():
            text = search.text().lower()
            ct = cmb_type.currentText()
            # Map display label back to MNE type
            if ct == "mag (axial grad)":
                ct = "mag"
            for idx in range(ch_list.count()):
                item = ch_list.item(idx)
                ch_name = item.data(Qt.ItemDataRole.UserRole)
                ch_idx = self._ch_names.index(ch_name)
                ch_type = self._ch_types[ch_idx]
                visible = True
                if text and text not in ch_name.lower():
                    visible = False
                if ct != "all" and ch_type != ct:
                    visible = False
                item.setHidden(not visible)
            visible_count = sum(1 for i in range(ch_list.count()) if not ch_list.item(i).isHidden())
            lbl_count.setText(f"{visible_count} / {ch_list.count()} channels")

        search.textChanged.connect(_filter_channels)
        cmb_type.currentTextChanged.connect(lambda: _filter_channels())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = set()
            for idx in range(ch_list.count()):
                item = ch_list.item(idx)
                if item.isSelected():
                    selected.add(item.data(Qt.ItemDataRole.UserRole))
            if len(selected) == len(self._ch_names):
                self._selected_channels = None  # All selected = no filter
            else:
                self._selected_channels = selected
            self._update_display_indices()
            self._redraw()
            n = len(selected) if self._selected_channels else len(self._ch_names)
            self.statusMessage.emit(f"Displaying {n} channels")

    # ================================================================ #
    # Data Loading (threaded)                                          #
    # ================================================================ #
    def load_raw(self, raw):
        """Accept an already-loaded MNE Raw object."""
        self._raw = raw
        self._sfreq = raw.info["sfreq"]
        self._ch_names = raw.ch_names
        self._n_samples = raw.n_times
        self._duration = raw.times[-1] if len(raw.times) > 0 else 0.0
        import mne
        self._ch_types = [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))]
        self._time_start = 0.0
        self._channel_offset = 0
        self._data_cache.clear()
        self._current_filter = None
        self._notch_freq = None
        self._resample_freq = None
        self._selected_channels = None
        self.spn_time_window.setMaximum(self._duration)

        # Auto-detect available channel types
        self._available_ch_types = sorted(set(self._ch_types))

        # Detect CTF system: has "mag" but no "grad", plus ref_meg or
        # compensation_grade or the file path ends with .ds
        self._is_ctf = (
            "mag" in self._available_ch_types
            and "grad" not in self._available_ch_types
            and (
                "ref_meg" in self._available_ch_types
                or getattr(raw, "compensation_grade", None) is not None
                or (self._current_filepath and
                    Path(self._current_filepath).suffix.lower() == ".ds")
            )
        )

        # Build channel type combo: add "mag+grad" if both mag and grad exist
        self.cmb_ch_type.blockSignals(True)
        self.cmb_ch_type.clear()
        self.cmb_ch_type.addItem("all")
        has_mag = "mag" in self._available_ch_types
        has_grad = "grad" in self._available_ch_types
        if has_mag and has_grad:
            self.cmb_ch_type.addItem("mag+grad")
        for ct in self._available_ch_types:
            # For CTF, show "mag" as "mag (axial grad)" in the combo box
            display_label = ct
            if self._is_ctf and ct == "mag":
                display_label = "mag (axial grad)"
            self.cmb_ch_type.addItem(display_label, userData=ct)
        self.cmb_ch_type.setCurrentText("all")
        self.cmb_ch_type.blockSignals(False)

        self._update_display_indices()
        self._update_channel_scrollbar()

        # Extract events from stim channels – wrapped safely for CTF data
        self._events_from_raw = []
        try:
            stim_chs = [ch for ch, t in zip(raw.ch_names, self._ch_types) if t == "stim"]
            if stim_chs:
                try:
                    events = mne.find_events(raw, stim_channel=stim_chs,
                                             shortest_event=1, verbose=False)
                except Exception:
                    # CTF data may not support find_events the same way;
                    # try individual channels as fallback
                    events = np.empty((0, 3), dtype=int)
                    for sc in stim_chs:
                        try:
                            ev = mne.find_events(raw, stim_channel=sc,
                                                 shortest_event=1, verbose=False)
                            if len(ev) > 0:
                                events = np.vstack([events, ev]) if len(events) > 0 else ev
                        except Exception:
                            continue
                for ev in events:
                    t = ev[0] / self._sfreq
                    eid = int(ev[2])
                    if eid != 0:
                        self._events_from_raw.append(
                            EventMarker(time=t, event_id=eid, channel="raw", source="raw"))
                if self._events_from_raw:
                    self.chk_show_events.setEnabled(True)
                    self.cmb_stim_channel.setEnabled(True)
        except Exception:
            pass

        self._redraw()
        # Build display-friendly type list for status bar
        display_types = []
        for ct in self._available_ch_types:
            if self._is_ctf and ct == "mag":
                display_types.append("mag (axial grad)")
            else:
                display_types.append(ct)
        self.statusMessage.emit(f"Loaded: {len(self._ch_names)} channels, "
                                f"{self._duration:.1f}s @ {self._sfreq:.0f} Hz"
                                f" | Types: {', '.join(display_types)}")

    def load_raw_threaded(self, filepath: str):
        """Load a MEG file on the main thread (deferred) with loading overlay.

        MNE's CTF reader uses memory-mapped I/O that is not thread-safe on
        macOS (causes SIGBUS).  We run I/O on the main thread, using
        QTimer.singleShot so the loading overlay renders first.
        """
        self._show_loading("Loading MEG data…")
        self._current_filepath = filepath
        QTimer.singleShot(50, lambda: self._do_load_raw_deferred(filepath))

    def _do_load_raw_deferred(self, filepath: str):
        """Main-thread raw loader (called by deferred timer)."""
        try:
            import mne
            raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self._hide_loading()
            self.load_raw(raw)
            if self._current_filepath:
                self.set_current_filepath(self._current_filepath)
        except Exception as exc:
            self._hide_loading()
            QMessageBox.warning(self, "Load Error", f"Failed to load data:\n{exc}")
            self.statusMessage.emit("Load error")

    @pyqtSlot(object)
    def _on_raw_loaded(self, raw):
        self._hide_loading()
        self._active_worker = None
        self.load_raw(raw)
        if self._current_filepath:
            self.set_current_filepath(self._current_filepath)

    @pyqtSlot(str)
    def _on_load_error(self, msg: str):
        self._hide_loading()
        self._active_worker = None
        QMessageBox.warning(self, "Load Error", f"Failed to load data:\n{msg}")
        self.statusMessage.emit("Load error")

    # ================================================================ #
    # Loading overlay helpers                                          #
    # ================================================================ #
    def _show_loading(self, msg: str = "Loading…"):
        self._loading_overlay.show_message(msg)
        QApplication.processEvents()

    def _hide_loading(self):
        self._loading_overlay.hide()

    # ================================================================ #
    # Display Logic                                                    #
    # ================================================================ #
    def _update_display_indices(self):
        if self._active_ch_type == "all":
            indices = list(range(len(self._ch_names)))
        elif self._active_ch_type == "mag+grad":
            indices = [
                i for i, t in enumerate(self._ch_types) if t in ("mag", "grad")]
        else:
            indices = [
                i for i, t in enumerate(self._ch_types) if t == self._active_ch_type]
        # Apply individual channel selection filter
        if self._selected_channels is not None:
            indices = [i for i in indices if self._ch_names[i] in self._selected_channels]
        self._display_indices = indices
        self._update_channel_scrollbar()

    def _update_channel_scrollbar(self):
        n = len(self._display_indices)
        visible = min(self._visible_channels, n)
        self.sld_channels.setRange(0, max(0, n - visible))
        self.sld_channels.setValue(self._channel_offset)

    def _get_data_segment(self, ch_indices, tmin, tmax):
        if self._raw is None:
            return None, None
        smin = max(0, int(tmin * self._sfreq))
        smax = min(self._n_samples, int(tmax * self._sfreq))
        if smax <= smin:
            return None, None
        try:
            data, times = self._raw[ch_indices, smin:smax]
        except Exception:
            return None, None
        if self._current_filter or self._notch_freq:
            import mne
            l_freq = (self._current_filter[0]
                      if self._current_filter and self._current_filter[0] is not None
                      and self._current_filter[0] > 0 else None)
            h_freq = (self._current_filter[1]
                      if self._current_filter and self._current_filter[1] is not None
                      and self._current_filter[1] > 0 else None)
            if l_freq or h_freq:
                try:
                    data = mne.filter.filter_data(data, self._sfreq, l_freq, h_freq, verbose=False)
                except Exception:
                    pass
            if self._notch_freq and self._notch_freq > 0:
                try:
                    data = mne.filter.notch_filter(data, self._sfreq, self._notch_freq, verbose=False)
                except Exception:
                    pass
        return data, times

    def _redraw(self):
        self._plot_widget.clear()
        self._clear_labels()
        self._clear_overlays()
        self._shown_ch_info = []
        if self._raw is None:
            return
        n_disp = len(self._display_indices)
        if n_disp == 0:
            return
        vis = min(self._visible_channels, n_disp)
        start_ch = min(self._channel_offset, max(0, n_disp - vis))
        end_ch = min(start_ch + vis, n_disp)
        shown_indices = self._display_indices[start_ch:end_ch]
        n_shown = len(shown_indices)
        if n_shown == 0:
            return
        tmin = self._time_start
        tmax = min(self._time_start + self._time_window, self._duration)
        data, times = self._get_data_segment(shown_indices, tmin, tmax)
        if data is None:
            return
        flagged_channels = self._get_active_channel_flags()
        scale = self._scale_factor
        plot_item = self._plot_widget.getPlotItem()
        default_label_color = "#cccccc" if self._dark_plot else "#000000"

        if self._normalize:
            # Normalize mode: per-channel normalization, each channel
            # is scaled to fit within its lane — no overlap.
            for i in range(n_shown):
                ch_idx = shown_indices[i]
                ch_name = self._ch_names[ch_idx]
                ch_type = self._ch_types[ch_idx]
                trace = data[i]
                data_range = np.ptp(trace) if np.ptp(trace) > 0 else 1.0
                offset = (n_shown - 1 - i)
                y = ((trace - np.mean(trace)) / data_range) * scale + offset
                color = self._get_channel_color(ch_name, ch_type, flagged_channels)
                pen = pg.mkPen(color=color, width=1)
                plot_item.plot(times, y, pen=pen)
                self._add_channel_label(ch_name, ch_type, flagged_channels, n_shown, i,
                                        default_label_color)
                self._shown_ch_info.append((offset, ch_name, ch_type))
        else:
            # Raw mode (default): common scale per channel type,
            # signals keep their original relative amplitudes and
            # may overlap between channels.
            type_ranges: Dict[str, list] = {}
            for i in range(n_shown):
                ch_type = self._ch_types[shown_indices[i]]
                ptp_val = np.ptp(data[i])
                type_ranges.setdefault(ch_type, []).append(ptp_val)
            type_scale: Dict[str, float] = {}
            for ct, ranges in type_ranges.items():
                valid = [r for r in ranges if r > 0]
                if valid:
                    type_scale[ct] = float(np.median(valid))
                else:
                    type_scale[ct] = 1.0

            for i in range(n_shown):
                ch_idx = shown_indices[i]
                ch_name = self._ch_names[ch_idx]
                ch_type = self._ch_types[ch_idx]
                trace = data[i]
                ref_range = type_scale.get(ch_type, 1.0)
                if ref_range <= 0:
                    ref_range = 1.0
                offset = (n_shown - 1 - i)
                # Centre signal around offset: remove mean so it sits on its baseline
                y = ((trace - np.mean(trace)) / ref_range) * scale + offset
                color = self._get_channel_color(ch_name, ch_type, flagged_channels)
                pen = pg.mkPen(color=color, width=1)
                plot_item.plot(times, y, pen=pen)
                self._add_channel_label(ch_name, ch_type, flagged_channels, n_shown, i,
                                        default_label_color)
                self._shown_ch_info.append((offset, ch_name, ch_type))

        plot_item.setXRange(tmin, tmax, padding=0)
        plot_item.setYRange(-0.5, n_shown - 0.5, padding=0.02)
        self._draw_annotation_overlays(shown_indices, tmin, tmax, n_shown)
        self.lbl_time_pos.setText(f"{tmin:.1f} - {tmax:.1f} / {self._duration:.1f} s")
        if self._duration > 0:
            slider_val = int((tmin / max(0.01, self._duration - self._time_window)) * 1000)
            self.sld_time.blockSignals(True)
            self.sld_time.setValue(max(0, min(1000, slider_val)))
            self.sld_time.blockSignals(False)

    def _clear_labels(self):
        while self._label_layout.count() > 0:
            item = self._label_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _clear_overlays(self):
        for item in self._overlay_items:
            try:
                self._plot_widget.removeItem(item)
            except Exception:
                pass
        self._overlay_items.clear()

    def _add_channel_label(self, ch_name, ch_type, flagged_channels, n_shown, index,
                           default_color="#000000"):
        lbl = QLabel(ch_name)
        lbl.setFixedHeight(max(1, int(self._plot_widget.height() / max(n_shown, 1))))
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        color = default_color
        for metric, channels in flagged_channels.items():
            if ch_name in channels:
                color = CHANNEL_LABEL_COLORS.get(metric, "#ff0000")
                break
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-size: 9px; }}")
        self._label_layout.addWidget(lbl)

    def _get_channel_color(self, ch_name, ch_type, flagged_channels):
        if self._dark_plot:
            base_colors = {"mag": (100, 140, 255), "grad": (100, 220, 100),
                           "eeg": (200, 100, 200), "eog": (100, 200, 200),
                           "ecg": (220, 100, 100), "stim": (200, 200, 100)}
            base = base_colors.get(ch_type, (180, 180, 180))
        else:
            base_colors = {"mag": (0, 0, 180), "grad": (0, 100, 0),
                           "eeg": (100, 0, 100), "eog": (0, 128, 128),
                           "ecg": (128, 0, 0), "stim": (128, 128, 0)}
            base = base_colors.get(ch_type, (80, 80, 80))
        for metric, channels in flagged_channels.items():
            if ch_name in channels:
                hc = CHANNEL_LABEL_COLORS.get(metric, "#ff0000")
                return (int(hc[1:3], 16), int(hc[3:5], 16), int(hc[5:7], 16))
        return base

    def _get_active_channel_flags(self) -> Dict[str, Set[str]]:
        result = {}
        if self._annotations is None:
            return result
        for metric in ["std_noisy", "std_flat", "ptp_noisy", "ptp_flat",
                       "psd_noisy", "ecg", "eog"]:
            if not self._enabled_annotations.get(metric, False):
                continue
            if metric == "ecg":
                channels = self._annotation_mgr.get_flagged_channels(metric, max_channels=self._ecg_top_n)
            elif metric == "eog":
                channels = self._annotation_mgr.get_flagged_channels(metric, max_channels=self._eog_top_n)
            else:
                channels = self._annotation_mgr.get_flagged_channels(metric)
            if channels:
                result[metric] = channels
        return result

    def _draw_annotation_overlays(self, shown_indices, tmin, tmax, n_shown):
        if (self._annotations is None
                and not self._show_events
                and not self._show_bids_events
                and not self._show_epoch_windows):
            return

        alpha = self._annotation_alpha

        if self._annotations is not None:
            # Muscle artifact overlays
            if self._enabled_annotations.get("muscle", False):
                for interval in self._annotation_mgr.get_muscle_intervals():
                    if interval.end >= tmin and interval.start <= tmax:
                        r, g, b, _ = ANNOTATION_COLORS["muscle"]
                        border_pen = pg.mkPen(r, g, b, 180, width=1) if self._show_overlay_borders else pg.mkPen(None)
                        region = pg.LinearRegionItem(
                            values=[max(interval.start, tmin), min(interval.end, tmax)],
                            orientation="vertical", brush=pg.mkBrush(r, g, b, alpha),
                            pen=border_pen, movable=False)
                        region.setToolTip(
                            f"Muscle artifact\nStart: {interval.start:.3f}s\n"
                            f"End: {interval.end:.3f}s\nDuration: {interval.end - interval.start:.3f}s\n"
                            f"Score: {interval.score:.4f}"
                        )
                        self._plot_widget.addItem(region)
                        self._overlay_items.append(region)

            # Epoch-level noisy/flat overlays
            for ui_key, (metric_key, _) in {
                "noisy_std_epochs": ("noisy_std", ""), "flat_std_epochs": ("flat_std", ""),
                "noisy_ptp_epochs": ("noisy_ptp", ""), "flat_ptp_epochs": ("flat_ptp", ""),
            }.items():
                if not self._enabled_annotations.get(ui_key, False):
                    continue
                for ch_type in ("mag", "grad", "eeg"):
                    ea = self._annotation_mgr.get_epoch_matrix(metric_key, ch_type)
                    if ea is None:
                        continue
                    self._draw_epoch_overlay(ea, shown_indices, tmin, tmax, n_shown, ui_key)

            # Epoch analysis window overlays (6a)
            if self._show_epoch_windows:
                self._draw_epoch_window_overlay(tmin, tmax, n_shown)

            # ECG detected event overlays (6b)
            if self._show_ecg_events:
                self._draw_physio_event_segments(
                    self._annotations.ecg_event_times, tmin, tmax, n_shown, "ecg_events",
                    label_prefix="ECG", rate=self._annotations.ecg_event_rate,
                )

            # EOG detected event overlays (6b)
            if self._show_eog_events:
                self._draw_physio_event_segments(
                    self._annotations.eog_event_times, tmin, tmax, n_shown, "eog_events",
                    label_prefix="EOG", rate=self._annotations.eog_event_rate,
                )

        # Draw stim event markers
        if self._show_events:
            self._draw_event_markers(tmin, tmax, n_shown, include_bids=False)

        # Draw BIDS events.tsv markers
        if self._show_bids_events:
            self._draw_event_markers(tmin, tmax, n_shown, include_bids=True, bids_only=True)

    def _draw_epoch_window_overlay(self, tmin, tmax, n_shown):
        """Draw all epoched analysis windows as gray transparent regions (6a)."""
        a = self._annotations
        if a is None or a.epoch_duration <= 0 or a.n_epochs <= 0:
            return

        epoch_dur = a.epoch_duration
        r, g, b, _ = ANNOTATION_COLORS["epoch_windows"]
        ew_alpha = self._epoch_window_alpha
        # Always use a clear black border to separate epoch windows
        border_pen = pg.mkPen(0, 0, 0, 200, width=2)

        # Determine first/last epoch visible
        first_epoch = max(0, int(tmin / epoch_dur))
        last_epoch = min(a.n_epochs - 1, int(tmax / epoch_dur))

        # Cap at 200 visible epochs for performance
        if last_epoch - first_epoch > 200:
            step = max(1, (last_epoch - first_epoch) // 200)
        else:
            step = 1

        for ep_idx in range(first_epoch, last_epoch + 1, step):
            ep_start = ep_idx * epoch_dur
            ep_end = ep_start + epoch_dur
            if ep_end < tmin or ep_start > tmax:
                continue
            region = pg.LinearRegionItem(
                values=[max(ep_start, tmin), min(ep_end, tmax)],
                orientation="vertical",
                brush=pg.mkBrush(r, g, b, ew_alpha),
                pen=border_pen,
                movable=False,
            )
            region.setToolTip(
                f"Epoch window {ep_idx}\nOnset: {ep_start:.3f}s\n"
                f"Duration: {epoch_dur:.3f}s"
            )
            self._plot_widget.addItem(region)
            self._overlay_items.append(region)

    def _draw_physio_event_segments(
        self, event_times, tmin, tmax, n_shown, color_key,
        label_prefix="", rate=0.0,
    ):
        """Draw ECG/EOG detected events as vertical segment lines (6b).

        Similar to how muscle events are displayed, these appear as thin
        vertical regions (segment lines) across all channels.
        """
        if not event_times:
            return

        r, g, b, _ = ANNOTATION_COLORS.get(color_key, (200, 200, 200, 70))
        alpha = self._annotation_alpha

        # Filter to visible window and cap at 500 for performance
        visible = [t for t in event_times if tmin <= t <= tmax]
        if len(visible) > 500:
            step = len(visible) // 500
            visible = visible[::step]

        # Segment half-width (thin region instead of line for better visibility)
        half_w = max(0.005, (tmax - tmin) * 0.001)

        for idx, t in enumerate(visible):
            border_pen = pg.mkPen(r, g, b, 180, width=1) if self._show_overlay_borders else pg.mkPen(None)
            region = pg.LinearRegionItem(
                values=[t - half_w, t + half_w],
                orientation="vertical",
                brush=pg.mkBrush(r, g, b, alpha),
                pen=border_pen,
                movable=False,
            )
            tooltip = f"{label_prefix} event\nTime: {t:.3f}s"
            if rate > 0:
                tooltip += f"\nRate: {rate:.1f}/min"
            region.setToolTip(tooltip)
            self._plot_widget.addItem(region)
            self._overlay_items.append(region)

    def _draw_epoch_overlay(self, ea: EpochAnnotation, shown_indices, tmin, tmax, n_shown, color_key):
        if ea.matrix.shape[0] == 0 or ea.matrix.shape[1] == 0:
            return
        epoch_dur = self._annotations.epoch_duration if self._annotations.epoch_duration > 0 else (
            self._duration / max(ea.matrix.shape[1], 1))
        color = ANNOTATION_COLORS.get(color_key, (200, 200, 200, 40))
        r, g, b, _ = color
        alpha = self._annotation_alpha
        # Black thick border for noisy/flat epoch overlays
        border_pen_color = pg.mkPen(0, 0, 0, 220, width=2)
        for ch_row_idx, ch_name in enumerate(ea.channel_names):
            try:
                ch_global_idx = self._ch_names.index(ch_name)
            except ValueError:
                continue
            if ch_global_idx not in shown_indices:
                continue
            shown_pos = shown_indices.index(ch_global_idx)
            y_pos = n_shown - 1 - shown_pos
            for ep_col_idx, ep_idx in enumerate(ea.epoch_indices):
                if ep_col_idx >= ea.matrix.shape[1]:
                    break
                if not ea.matrix[ch_row_idx, ep_col_idx]:
                    continue
                ep_start = ep_idx * epoch_dur
                ep_end = ep_start + epoch_dur
                if ep_end < tmin or ep_start > tmax:
                    continue
                x0 = max(ep_start, tmin)
                x1 = min(ep_end, tmax)
                rect = pg.QtWidgets.QGraphicsRectItem(
                    x0, y_pos - 0.4,
                    x1 - x0, 0.8)
                rect.setBrush(pg.mkBrush(r, g, b, alpha))
                # Always show black thick border for noisy/flat epochs
                rect.setPen(border_pen_color)
                # Enable hover events so tooltip shows on mouse over
                rect.setAcceptHoverEvents(True)
                # Hover tooltip (6c)
                label = color_key.replace("_", " ").title()
                rect.setToolTip(
                    f"{label}\nChannel: {ch_name}\nEpoch: {ep_idx}\n"
                    f"Onset: {ep_start:.3f}s\nDuration: {epoch_dur:.3f}s\n"
                    f"Ch type: {ea.ch_type}"
                )
                self._plot_widget.addItem(rect)
                self._overlay_items.append(rect)

    # Colour palette for distinct event IDs
    _EVENT_COLORS = [
        (255, 215, 0), (0, 200, 255), (255, 100, 200), (100, 255, 100),
        (255, 150, 50), (150, 100, 255), (0, 255, 200), (255, 80, 80),
    ]

    def _draw_event_markers(self, tmin, tmax, n_shown,
                            include_bids: bool = False, bids_only: bool = False):
        """Draw vertical lines at event onset times within the visible window."""
        events = []
        if not bids_only:
            if self._annotations is not None:
                ch_filter = self._active_stim_channel if self._active_stim_channel != "all" else ""
                events.extend(self._annotation_mgr.get_events(ch_filter, include_bids=False))
            events.extend(self._events_from_raw)

        if include_bids and self._annotations is not None:
            events.extend(self._annotation_mgr.get_events("", include_bids=True))
            # Deduplicate: remove non-bids events that are already in bids list
            if not bids_only:
                pass  # keep both
            else:
                events = [e for e in events if e.source == "bids_events"]

        if not events:
            return

        # Filter to visible window and cap at 500 for performance
        visible = [e for e in events if tmin <= e.time <= tmax]
        if len(visible) > 500:
            step = len(visible) // 500
            visible = visible[::step]

        # Map event IDs to colours
        unique_ids = sorted(set(e.event_id for e in visible))
        id_color_map = {}
        for i, eid in enumerate(unique_ids):
            id_color_map[eid] = self._EVENT_COLORS[i % len(self._EVENT_COLORS)]

        line_width = self._event_line_width
        color_override = self._event_color_override

        for ev in visible:
            if color_override and color_override.isValid():
                r, g, b = color_override.red(), color_override.green(), color_override.blue()
            else:
                r, g, b = id_color_map[ev.event_id]

            # Use solid line style for thicker lines, dash for thin
            style = Qt.PenStyle.SolidLine if line_width >= 2 else Qt.PenStyle.DashLine
            pen = pg.mkPen(color=(r, g, b, 200), width=line_width, style=style)
            line = pg.InfiniteLine(pos=ev.time, angle=90, pen=pen, movable=False)
            self._plot_widget.addItem(line)
            self._overlay_items.append(line)

            # Label: use trial_type label if available, else event ID
            label_text = ev.label if ev.label else str(ev.event_id)
            label = pg.TextItem(label_text, color=(r, g, b, 220), anchor=(0.5, 1.0))
            label.setPos(ev.time, n_shown - 0.3)
            font = label.textItem.font()
            font.setPointSize(7)
            label.setFont(font)
            self._plot_widget.addItem(label)
            self._overlay_items.append(label)

    # ================================================================ #
    # Event Handlers                                                   #
    # ================================================================ #
    def _on_ch_type_changed(self, text):
        # Use userData if set (for CTF "mag (axial grad)" → "mag"), else text
        idx = self.cmb_ch_type.currentIndex()
        user_data = self.cmb_ch_type.itemData(idx)
        self._active_ch_type = user_data if user_data else text
        self._channel_offset = 0
        self._update_display_indices()
        self._redraw()

    def _on_n_channels_changed(self, val):
        self._visible_channels = val
        self._update_channel_scrollbar()
        self._redraw()

    def _on_scale_changed(self, val):
        self._scale_factor = val
        self._redraw()

    def _on_time_window_changed(self, val):
        self._time_window = val
        self._redraw()

    def _on_time_slider(self, val):
        if self._duration <= 0:
            return
        max_start = max(0.0, self._duration - self._time_window)
        self._time_start = (val / 1000.0) * max_start
        self._redraw()

    def _on_channel_scroll(self, val):
        self._channel_offset = val
        self._redraw()

    def _on_plot_wheel(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            step = -1 if delta > 0 else 1
            new_val = max(self.sld_channels.minimum(),
                          min(self.sld_channels.maximum(), self.sld_channels.value() + step))
            self.sld_channels.setValue(new_val)
        event.accept()

    def _navigate(self, direction):
        step = self._time_window * 0.8
        if direction == "start":
            self._time_start = 0.0
        elif direction == "prev":
            self._time_start = max(0.0, self._time_start - step)
        elif direction == "next":
            self._time_start = min(self._duration - self._time_window, self._time_start + step)
        elif direction == "end":
            self._time_start = max(0.0, self._duration - self._time_window)
        self._time_start = max(0.0, self._time_start)
        self._redraw()

    def _on_normalize_toggled(self, checked):
        self._normalize = checked
        if self._raw is not None:
            self._redraw()

    def _on_mouse_moved(self, evt):
        """Show channel name tooltip when hovering near a trace."""
        pos = evt[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return
        if not self._shown_ch_info:
            return
        mouse_point = self._plot_widget.getPlotItem().vb.mapSceneToView(pos)
        y = mouse_point.y()
        # Find nearest channel
        min_dist = float('inf')
        nearest_name = ""
        for y_off, ch_name, ch_type in self._shown_ch_info:
            dist = abs(y - y_off)
            if dist < min_dist:
                min_dist = dist
                nearest_name = ch_name
        if min_dist < 0.6:
            QToolTip.showText(QCursor.pos(), nearest_name, self._plot_widget)
        else:
            QToolTip.hideText()

    def _on_dark_plot_toggled(self, checked):
        self._dark_plot = checked
        self._settings.setValue("viewer/dark_plot", checked)
        self._apply_plot_theme()
        if self._raw is not None:
            self._redraw()

    def _on_ecg_top_changed(self, val):
        self._ecg_top_n = val
        if self._enabled_annotations.get("ecg", False):
            self._redraw()

    def _on_eog_top_changed(self, val):
        self._eog_top_n = val
        if self._enabled_annotations.get("eog", False):
            self._redraw()

    def _on_show_events_toggled(self, checked):
        self._show_events = checked
        self._redraw()

    def _on_show_bids_events_toggled(self, checked):
        self._show_bids_events = checked
        self._redraw()

    def _on_stim_channel_changed(self, text):
        self._active_stim_channel = text
        if self._show_events:
            self._redraw()

    def _on_alpha_changed(self, val):
        self._annotation_alpha = val
        self.lbl_alpha_val.setText(str(val))
        if any(self._enabled_annotations.values()):
            self._redraw()

    def _on_epoch_windows_toggled(self, checked):
        self._show_epoch_windows = checked
        self._enabled_annotations["epoch_windows"] = checked
        self._redraw()

    def _on_epoch_window_alpha_changed(self, val):
        self._epoch_window_alpha = val
        self.lbl_ew_alpha_val.setText(str(val))
        if self._show_epoch_windows:
            self._redraw()

    def _on_ecg_events_toggled(self, checked):
        self._show_ecg_events = checked
        self._enabled_annotations["ecg_events"] = checked
        self._redraw()

    def _on_eog_events_toggled(self, checked):
        self._show_eog_events = checked
        self._enabled_annotations["eog_events"] = checked
        self._redraw()

    def _on_overlay_borders_toggled(self, checked):
        self._show_overlay_borders = checked
        if any(self._enabled_annotations.values()):
            self._redraw()

    def _on_event_width_changed(self, val):
        self._event_line_width = val
        if self._show_events or self._show_bids_events:
            self._redraw()

    def _pick_event_color(self):
        color = QColorDialog.getColor(
            self._event_color_override if self._event_color_override else QColor(255, 215, 0),
            self, "Pick Event Line Color")
        if color.isValid():
            self._event_color_override = color
            self.btn_event_color.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: white; }}")
            if self._show_events or self._show_bids_events:
                self._redraw()

    def _reset_event_color(self):
        self._event_color_override = None
        self.btn_event_color.setStyleSheet("")
        if self._show_events or self._show_bids_events:
            self._redraw()

    # ================================================================ #
    # Reset View                                                       #
    # ================================================================ #
    def _reset_view(self):
        if self._raw is None:
            return
        self._time_start = 0.0
        self._channel_offset = 0
        self._scale_factor = 1.0
        self._time_window = 10.0
        self._active_ch_type = "all"
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        for w in (self.spn_scale, self.spn_time_window, self.spn_n_channels, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_normalize):
            w.blockSignals(True)
        self.spn_scale.setValue(1.0)
        self.spn_time_window.setValue(10.0)
        self.spn_n_channels.setValue(20)
        self.cmb_ch_type.setCurrentText("all")
        self.spn_hp.setValue(0.0)
        self.spn_lp.setValue(0.0)
        self.spn_notch.setValue(0.0)
        self.chk_normalize.setChecked(False)
        self._normalize = False
        for w in (self.spn_scale, self.spn_time_window, self.spn_n_channels, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_normalize):
            w.blockSignals(False)
        self._visible_channels = 20
        self._update_display_indices()
        self._update_channel_scrollbar()
        self._redraw()
        self.statusMessage.emit("View reset to defaults")

    # ================================================================ #
    # Filtering                                                        #
    # ================================================================ #
    def _apply_filters(self):
        hp = self.spn_hp.value()
        lp = self.spn_lp.value()
        notch = self.spn_notch.value()
        self._current_filter = (hp if hp > 0 else None, lp if lp > 0 else None)
        self._notch_freq = notch if notch > 0 else None
        self._redraw()
        self.statusMessage.emit(f"Filter applied: HP={hp}Hz, LP={lp}Hz, Notch={notch}Hz")

    def _reset_filters(self):
        self.spn_hp.setValue(0.0)
        self.spn_lp.setValue(0.0)
        self.spn_notch.setValue(0.0)
        self._current_filter = None
        self._notch_freq = None
        self._redraw()
        self.statusMessage.emit("Filters reset")

    def _apply_resample(self):
        freq = self.spn_resample.value()
        if freq <= 0 or self._raw is None:
            return
        self._show_loading(f"Resampling to {freq:.0f} Hz…")
        QTimer.singleShot(50, lambda: self._do_resample_deferred(freq))

    def _do_resample_deferred(self, freq: float):
        """Main-thread resampler (deferred via QTimer)."""
        try:
            resampled_raw = self._raw.copy().resample(freq, verbose=False)
            self._hide_loading()
            self._raw = resampled_raw
            self._sfreq = self._raw.info["sfreq"]
            self._n_samples = self._raw.n_times
            self._duration = self._raw.times[-1] if len(self._raw.times) > 0 else 0.0
            self.spn_time_window.setMaximum(self._duration)
            self._redraw()
            self.statusMessage.emit(f"Resampled to {freq:.0f} Hz ({self._n_samples} samples)")
        except Exception as exc:
            self._hide_loading()
            QMessageBox.warning(self, "Resample Error", str(exc))

    # ================================================================ #
    # PSD & Topography                                                 #
    # ================================================================ #
    def _show_psd(self):
        if self._raw is None:
            QMessageBox.information(self, "PSD", "No data loaded.")
            return
        self._show_loading("Computing PSD…")
        QTimer.singleShot(50, self._do_psd_deferred)

    def _do_psd_deferred(self):
        """Main-thread PSD computation (deferred via QTimer)."""
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            psd = self._raw.compute_psd(fmin=0.1, fmax=min(self._sfreq / 2, 150), verbose=False)
            fig = psd.plot(show=False)
            fig.suptitle("Power Spectral Density")
            fig.set_size_inches(12, 6)
            fig.tight_layout()
            self._hide_loading()
            fig.show()
            self.statusMessage.emit("PSD computed and displayed")
        except Exception as e:
            self._hide_loading()
            QMessageBox.warning(self, "PSD Error", f"Failed to compute PSD:\n{e}")

    def _show_topomap(self):
        if self._raw is None:
            QMessageBox.information(self, "Topomap", "No data loaded.")
            return
        self._show_loading("Computing topomap…")
        QTimer.singleShot(50, self._do_topomap_deferred)

    def _do_topomap_deferred(self):
        """Main-thread topomap computation (deferred via QTimer)."""
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import mne
            t_center = self._time_start + self._time_window / 2
            sample_idx = int(t_center * self._sfreq)
            sample_idx = max(0, min(sample_idx, self._n_samples - 1))
            half_win = max(1, int(0.05 * self._sfreq))
            s_start = max(0, sample_idx - half_win)
            s_end = min(self._n_samples, sample_idx + half_win + 1)
            data_segment = self._raw.get_data()[:, s_start:s_end].mean(axis=1, keepdims=True)
            evoked = mne.EvokedArray(data_segment, self._raw.info, tmin=0.0)
            fig = evoked.plot_topomap(times=[0.0], show=False)
            fig.suptitle(f"Topomap at t = {t_center:.2f} s")
            self._hide_loading()
            fig.show()
            self.statusMessage.emit(f"Topomap displayed at t = {t_center:.2f} s")
        except Exception as e:
            self._hide_loading()
            QMessageBox.warning(self, "Topomap Error", f"Failed:\n{e}")

    # ================================================================ #
    # Annotations                                                      #
    # ================================================================ #
    def _load_annotations_dialog(self):
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        calc_dir = QFileDialog.getExistingDirectory(self, "Select MEGqc calculation directory (sub-XXX/)")
        if not calc_dir:
            return
        self._show_loading("Loading annotations…")

        def _do_load():
            prefixes = self._annotation_mgr.auto_detect_recordings(calc_dir)
            return prefixes

        worker = _LoadWorker(_do_load)

        def _on_done(prefixes):
            self._hide_loading()
            self._active_worker = None
            if not prefixes:
                QMessageBox.warning(self, "No annotations found",
                                    "Could not find MEGqc derivative files in this directory.")
                return
            if len(prefixes) == 1:
                prefix = prefixes[0]
            else:
                prefix, ok = QInputDialog.getItem(self, "Select recording",
                                                  "Multiple recordings found:", prefixes, 0, False)
                if not ok:
                    return
            self._do_load_annotations(calc_dir, prefix)

        def _on_err(msg):
            self._hide_loading()
            self._active_worker = None
            QMessageBox.warning(self, "Annotation Error", msg)

        worker.finished.connect(_on_done)
        worker.errored.connect(_on_err)
        self._active_worker = worker
        worker.start()

    def _do_load_annotations(self, calc_dir: str, prefix: str):
        """Load annotations in a background thread."""
        self._show_loading(f"Loading annotations for {prefix}…")

        def _load():
            return self._annotation_mgr.load_for_recording(calc_dir, prefix)

        worker = _LoadWorker(_load)

        def _on_done(annot_set):
            self._hide_loading()
            self._active_worker = None
            self._annotations = annot_set
            if self._annotations.n_epochs == 0 and self._raw is not None:
                for store in [self._annotations.noisy_epochs_std, self._annotations.flat_epochs_std]:
                    for ea in store.values():
                        if len(ea.epoch_indices) > 0:
                            self._annotations.n_epochs = max(ea.epoch_indices) + 1
                            self._annotations.epoch_duration = (
                                self._duration / max(self._annotations.n_epochs, 1))
                            break
                    if self._annotations.n_epochs > 0:
                        break
            avail = self._annotation_mgr.get_available_metrics()
            for key, cb in self._annot_checkboxes.items():
                if key == "epoch_windows":
                    # Epoch windows available when we know the epoch structure
                    cb.setEnabled(self._annotations.n_epochs > 0 and self._annotations.epoch_duration > 0)
                else:
                    cb.setEnabled(key in avail)
            # Enable BIDS events checkbox if available
            self.chk_show_bids_events.setEnabled("bids_events" in avail)
            self._update_ecg_eog_limits()
            self._update_events_controls()
            self.statusMessage.emit(
                f"Loaded annotations for '{prefix}': {len(avail)} metric categories available")

        def _on_err(msg):
            self._hide_loading()
            self._active_worker = None
            QMessageBox.warning(self, "Annotation Error", f"Failed to load:\n{msg}")

        worker.finished.connect(_on_done)
        worker.errored.connect(_on_err)
        self._active_worker = worker
        worker.start()

    def load_annotations_from_dir(self, calc_dir: str, prefix: str):
        try:
            self._annotations = self._annotation_mgr.load_for_recording(calc_dir, prefix)
            if self._raw is not None and self._annotations.n_epochs == 0:
                for store in [self._annotations.noisy_epochs_std, self._annotations.flat_epochs_std]:
                    for ea in store.values():
                        if len(ea.epoch_indices) > 0:
                            self._annotations.n_epochs = max(ea.epoch_indices) + 1
                            self._annotations.epoch_duration = (
                                self._duration / max(self._annotations.n_epochs, 1))
                            break
                    if self._annotations.n_epochs > 0:
                        break
            avail = self._annotation_mgr.get_available_metrics()
            for key, cb in self._annot_checkboxes.items():
                if key == "epoch_windows":
                    cb.setEnabled(self._annotations.n_epochs > 0 and self._annotations.epoch_duration > 0)
                else:
                    cb.setEnabled(key in avail)
            self.chk_show_bids_events.setEnabled("bids_events" in avail)
            self._update_ecg_eog_limits()
            self._update_events_controls()
        except Exception:
            pass

    def _update_ecg_eog_limits(self):
        if self._annotations is None:
            return
        n_ecg = len(self._annotations.ecg_ranked_channels)
        if n_ecg > 0:
            self.spn_ecg_top.setMaximum(n_ecg)
            self.spn_ecg_top.setToolTip(f"Top N of {n_ecg} ranked ECG channels")
        n_eog = len(self._annotations.eog_ranked_channels)
        if n_eog > 0:
            self.spn_eog_top.setMaximum(n_eog)
            self.spn_eog_top.setToolTip(f"Top N of {n_eog} ranked EOG channels")

    def _update_events_controls(self):
        """Enable the events checkbox and populate the stim channel selector."""
        if self._annotations is None:
            return
        has_events = bool(self._annotations.events)
        has_bids = bool(self._annotations.bids_event_onsets)
        self.chk_show_events.setEnabled(has_events or bool(self._events_from_raw))
        self.chk_show_bids_events.setEnabled(has_bids)
        if has_events or has_bids:
            self.cmb_stim_channel.blockSignals(True)
            self.cmb_stim_channel.clear()
            self.cmb_stim_channel.addItem("all")
            for ch in self._annotations.stim_channels:
                self.cmb_stim_channel.addItem(ch)
            if has_bids:
                self.cmb_stim_channel.addItem("events.tsv")
            self.cmb_stim_channel.blockSignals(False)
            self.cmb_stim_channel.setEnabled(True)

    def _on_annotation_toggled(self, key: str, checked: bool):
        self._enabled_annotations[key] = checked
        self._redraw()

    def _enable_all_annotations(self):
        for key, cb in self._annot_checkboxes.items():
            if cb.isEnabled():
                cb.setChecked(True)

    def _disable_all_annotations(self):
        for cb in self._annot_checkboxes.values():
            cb.setChecked(False)

    def set_current_filepath(self, filepath: str):
        self._current_filepath = filepath

    def _scan_annotations(self):
        if self._raw is None:
            QMessageBox.information(self, "Scan", "No data loaded. Load a MEG file first.")
            return
        filepath = self._current_filepath
        if not filepath:
            QMessageBox.information(self, "Scan", "No file path available for scanning.")
            return

        self._show_loading("Scanning for annotations…")

        def _do_scan():
            import re
            p = Path(filepath)
            fname = p.stem
            # Handle CTF .ds directories: p is the directory, stem is e.g. "sub-009_task-X_meg"
            if p.suffix.lower() == ".ds":
                # Strip trailing _meg suffix from CTF directory names
                fname = re.sub(r"_meg$", "", fname)
            sub_match = re.search(r"(sub-[^_]+)", fname)
            sub_entity = sub_match.group(1) if sub_match else None
            return sub_entity

        worker = _LoadWorker(_do_scan)

        def _on_done(sub_entity):
            self._hide_loading()
            self._active_worker = None
            if not sub_entity:
                QMessageBox.warning(self, "Scan",
                                    "Could not determine subject from filename.\n"
                                    "Use 'Load Annotations...' to select manually.")
                return
            self._continue_scan(filepath, sub_entity)

        def _on_err(msg):
            self._hide_loading()
            self._active_worker = None
            QMessageBox.warning(self, "Scan Error", msg)

        worker.finished.connect(_on_done)
        worker.errored.connect(_on_err)
        self._active_worker = worker
        worker.start()

    def _continue_scan(self, filepath: str, sub_entity: str):
        """Second phase of scan: find dirs and load annotations."""
        import re
        p = Path(filepath)
        fname = p.stem
        if p.suffix.lower() == ".ds":
            fname = re.sub(r"_meg$", "", fname)

        found_dirs = self._collect_scan_dirs(filepath, sub_entity)
        if not found_dirs:
            QMessageBox.information(self, "Scan Result",
                                    f"No MEGqc calculation directory found for '{sub_entity}'.\n\n"
                                    "Make sure the derivatives are in a 'derivatives/Meg_QC/calculation/' "
                                    "directory structure.")
            return

        found_dirs = self._select_profile_if_needed(found_dirs)
        best_dir = None
        best_prefix = None
        for calc_dir in found_dirs:
            prefixes = self._annotation_mgr.auto_detect_recordings(calc_dir)
            if not prefixes:
                continue
            ses_match = re.search(r"(ses-[^_]+)", fname)
            task_match = re.search(r"(task-[^_]+)", fname)
            run_match = re.search(r"(run-[^_]+)", fname)
            scored = []
            for pfx in prefixes:
                score = 0
                if sub_entity and sub_entity in pfx: score += 10
                if ses_match and ses_match.group(1) in pfx: score += 5
                if task_match and task_match.group(1) in pfx: score += 3
                if run_match and run_match.group(1) in pfx: score += 2
                scored.append((score, pfx))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][0] > 0:
                best_dir = calc_dir
                best_prefix = scored[0][1]
                break
        if best_dir is None or best_prefix is None:
            for calc_dir in found_dirs:
                prefixes = self._annotation_mgr.auto_detect_recordings(calc_dir)
                if prefixes:
                    best_dir = calc_dir
                    best_prefix = prefixes[0]
                    break
        if best_dir and best_prefix:
            self.load_annotations_from_dir(best_dir, best_prefix)
            avail = self._annotation_mgr.get_available_metrics()
            self.statusMessage.emit(
                f"Auto-loaded annotations for '{best_prefix}': "
                f"{len(avail)} metrics from {Path(best_dir).name}")
            if len(avail) > 0:
                QMessageBox.information(self, "Scan Result",
                                        f"Found and loaded {len(avail)} annotation categories\n"
                                        f"for '{best_prefix}'.\n\nSource: {best_dir}")
            else:
                QMessageBox.information(self, "Scan Result",
                                        f"Found calculation directory but no annotation data:\n{best_dir}")
        else:
            QMessageBox.information(self, "Scan Result",
                                    "No matching MEGqc annotations found.\n\n"
                                    "Searched directories:\n" + "\n".join(found_dirs))

    # ================================================================ #
    # External Annotation Paths                                        #
    # ================================================================ #
    def set_external_annotation_paths(self, paths: List[str]):
        self._external_annot_paths = list(paths)
        self._refresh_ext_path_label()

    def add_external_annotation_path(self, path: str):
        if path not in self._external_annot_paths:
            self._external_annot_paths.append(path)
        self._refresh_ext_path_label()

    def _set_external_path_dialog(self):
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, "Select external MEGqc output directory")
        if d:
            self.add_external_annotation_path(d)
            self.statusMessage.emit(f"External annotation path added: {d}")

    def _refresh_ext_path_label(self):
        if self._external_annot_paths:
            text = "Ext: " + "; ".join(str(Path(p).name) for p in self._external_annot_paths[-2:])
            if len(self._external_annot_paths) > 2:
                text += f" (+{len(self._external_annot_paths) - 2} more)"
            self.lbl_ext_path.setText(text)
            self.lbl_ext_path.setToolTip("\n".join(self._external_annot_paths))
        else:
            self.lbl_ext_path.setText("")

    def _collect_scan_dirs(self, filepath: str, sub_entity: str) -> list:
        p = Path(filepath)
        found_dirs = []
        # For CTF .ds directories, start from the parent of the .ds dir
        start_path = p.parent if p.suffix.lower() == ".ds" else p
        for parent in start_path.parents:
            for calc_root in [
                parent / "derivatives" / "Meg_QC" / "calculation" / sub_entity,
                parent / "derivatives" / "Meg_QC" / "calculation",
                parent / "Meg_QC" / "calculation" / sub_entity,
                parent / "calculation" / sub_entity,
            ]:
                if calc_root.is_dir():
                    if calc_root.name == "calculation":
                        for sub_dir in calc_root.iterdir():
                            if sub_dir.is_dir() and sub_dir.name == sub_entity:
                                found_dirs.append(str(sub_dir))
                            # Check modality subfolders (meg/, eeg/)
                            elif sub_dir.is_dir() and sub_dir.name in ("meg", "eeg"):
                                mod_sub = sub_dir / sub_entity
                                if mod_sub.is_dir():
                                    found_dirs.append(str(mod_sub))
                    else:
                        found_dirs.append(str(calc_root))
            profile_root = parent / "derivatives" / "Meg_QC" / "profiles"
            if profile_root.is_dir():
                for profile_dir in profile_root.iterdir():
                    if profile_dir.is_dir():
                        calc_in = profile_dir / "calculation" / sub_entity
                        if calc_in.is_dir():
                            found_dirs.append(str(calc_in))
                        # Check modality subfolders inside profiles
                        for modality in ("meg", "eeg"):
                            mod_calc = profile_dir / "calculation" / modality / sub_entity
                            if mod_calc.is_dir():
                                found_dirs.append(str(mod_calc))
        for ext_root in self._external_annot_paths:
            ext_p = Path(ext_root)
            if not ext_p.is_dir():
                continue
            for candidate in [
                ext_p / "derivatives" / "Meg_QC" / "calculation" / sub_entity,
                ext_p / "Meg_QC" / "calculation" / sub_entity,
                ext_p / "calculation" / sub_entity,
            ]:
                if candidate.is_dir():
                    found_dirs.append(str(candidate))
                # Also check modality subfolders
                for modality in ("meg", "eeg"):
                    mod_candidate = candidate.parent / modality / sub_entity
                    if mod_candidate.is_dir():
                        found_dirs.append(str(mod_candidate))
            for prof_root in [ext_p / "derivatives" / "Meg_QC" / "profiles",
                              ext_p / "Meg_QC" / "profiles"]:
                if prof_root.is_dir():
                    for pd in prof_root.iterdir():
                        if pd.is_dir():
                            calc_in = pd / "calculation" / sub_entity
                            if calc_in.is_dir():
                                found_dirs.append(str(calc_in))
                            # Check modality subfolders inside profiles
                            for modality in ("meg", "eeg"):
                                mod_calc = pd / "calculation" / modality / sub_entity
                                if mod_calc.is_dir():
                                    found_dirs.append(str(mod_calc))
            if ext_p.name == sub_entity:
                found_dirs.append(str(ext_p))
            elif (ext_p / sub_entity).is_dir():
                found_dirs.append(str(ext_p / sub_entity))
        seen = set()
        unique = []
        for d in found_dirs:
            rd = str(Path(d).resolve())
            if rd not in seen:
                seen.add(rd)
                unique.append(d)
        return unique

    def _select_profile_if_needed(self, found_dirs: list) -> list:
        import re as _re
        profile_map: Dict[str, list] = {}
        non_profile: list = []
        for d in found_dirs:
            m = _re.search(r"profiles[/\\]([^/\\]+)", d)
            if m:
                profile_map.setdefault(m.group(1), []).append(d)
            else:
                non_profile.append(d)
        if len(profile_map) <= 1:
            return found_dirs
        from PyQt6.QtWidgets import QInputDialog
        profiles = sorted(profile_map.keys())
        chosen, ok = QInputDialog.getItem(
            self, "Select Profile",
            f"Multiple MEGqc profiles found ({len(profiles)}).\n"
            "Choose the profile to load annotations from:",
            profiles, 0, False)
        if not ok:
            return found_dirs
        return non_profile + profile_map.get(chosen, [])

    # ================================================================ #
    # Memory Management                                                #
    # ================================================================ #
    def unload_data(self):
        self._plot_widget.clear()
        self._clear_labels()
        self._clear_overlays()
        self._raw = None
        self._data_cache.clear()
        self._ch_names = []
        self._ch_types = []
        self._available_ch_types = []
        self._is_ctf = False
        self._duration = 0.0
        self._n_samples = 0
        self._annotations = None
        self._current_filter = None
        self._notch_freq = None
        self._resample_freq = None
        self._current_filepath = None
        self._selected_channels = None
        for cb in self._annot_checkboxes.values():
            cb.setChecked(False)
            cb.setEnabled(False)
        self.chk_show_bids_events.setChecked(False)
        self.chk_show_bids_events.setEnabled(False)
        self.statusMessage.emit("Data unloaded from memory")

    # ================================================================ #
    # Theme                                                            #
    # ================================================================ #
    def _apply_plot_theme(self):
        plot_bg = "#1e1e1e" if self._dark_plot else "#ffffff"
        self._plot_widget.setBackground(plot_bg)
        fg = "#cccccc" if self._dark_plot else "#000000"
        for axis_name in ("bottom", "left"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setPen(pg.mkPen(color=fg))
            axis.setTextPen(pg.mkPen(color=fg))

    def refresh_theme(self):
        if self._raw is not None:
            self._redraw()

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            if self._raw is not None:
                QTimer.singleShot(50, self._redraw)
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep loading overlay sized correctly
        if self._loading_overlay.isVisible():
            self._loading_overlay.setGeometry(self.rect())





