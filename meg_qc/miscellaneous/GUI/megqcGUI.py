# main_window.py
# PyQt6 GUI for MEG QC pipeline: run/stop calculation & plotting, edit settings.ini
#
# Developer Notes:
# - Worker class spawns a separate OS process group for each task so that
#   sending SIGTERM to the group reliably stops joblib workers.
# - System info (CPU & total RAM) displayed in status bar via psutil or /proc/meminfo fallback.
# - All imports are at top, and key code sections are annotated for clarity.
# - The "Info" button next to the jobs spinbox shows detailed recommendations for n_jobs.

import sys
import os
import signal
import json
import time
import subprocess
import configparser
import hashlib
import shutil
import ssl
import tempfile
import ctypes


# ── Linux xcb-cursor fix ─────────────────────────────────────────────────
# Starting with Qt 6.5, the xcb platform plugin requires libxcb-cursor.so.0
# which is NOT installed by default on many Linux distros (e.g. Ubuntu 22.04).
# On Windows and macOS the Qt wheels bundle everything needed, but on Linux
# this one library slips through.  The helpers below fetch it into the active
# .venv (no root needed) so that end-users never have to run
# ``sudo apt install``.

_XCB_SENTINEL = "_MEGQC_XCB_CURSOR_FIXED"
_XCB_SO_NAME = "libxcb-cursor.so.0"


def _ensure_xcb_cursor() -> None:
    """Make sure ``libxcb-cursor.so.0`` is loadable on Linux.

    Only runs on Linux.  Strategy:
      1. Try to load the lib from the default search path — return if OK.
      2. If a cached copy already exists in ``<venv>/lib/megqc_xcb/``,
         preload it globally via ``ctypes.CDLL(..., RTLD_GLOBAL)`` so Qt's
         xcb plugin finds it.  No process restart needed.
      3. If no cached copy exists, fetch the ``.deb`` with
         ``apt-get download`` (no root), extract the ``.so``, store it in
         the venv, then re-exec the process with ``LD_LIBRARY_PATH`` so
         the dynamic linker picks it up on the fresh start.
    A sentinel env-var prevents infinite re-exec loops.
    """
    if sys.platform != "linux":
        return
    if os.environ.get(_XCB_SENTINEL):
        return  # already attempted — do not loop

    # 1. Already installed system-wide? Nothing to do.
    try:
        ctypes.cdll.LoadLibrary(_XCB_SO_NAME)
        return
    except OSError:
        pass

    # Store inside the virtual-env so it lives/dies with the venv.
    venv_lib_dir = os.path.join(sys.prefix, "lib", "megqc_xcb")
    so_path = os.path.join(venv_lib_dir, _XCB_SO_NAME)

    # 2. Cached from a previous run? Preload it into the process globally
    #    so Qt's dlopen() finds it — no re-exec overhead.
    if os.path.isfile(so_path):
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            print(
                f"[MEGqc] Preloaded {_XCB_SO_NAME} from {so_path}",
                file=sys.stderr,
            )
            return
        except OSError:
            # Corrupted or incompatible — delete and re-fetch below.
            os.remove(so_path)

    # 3. Not available anywhere — download into the venv.
    print(
        f"[MEGqc] {_XCB_SO_NAME} is not installed on this system.\n"
        f"[MEGqc] Qt 6.5+ requires it for the xcb (X11) platform plugin.\n"
        f"[MEGqc] Attempting automatic download into the active .venv …",
        file=sys.stderr,
    )

    try:
        _fetch_xcb_cursor_lib(venv_lib_dir)
    except Exception as exc:
        print(
            f"[MEGqc] Auto-fetch failed: {exc}\n"
            "[MEGqc] Please install the library manually:\n"
            "[MEGqc]   sudo apt install libxcb-cursor0",
            file=sys.stderr,
        )
        return
    if not os.path.isfile(so_path):
        return

    print(
        f"[MEGqc] Library cached at {so_path}\n"
        f"[MEGqc] Restarting with LD_LIBRARY_PATH → {venv_lib_dir}",
        file=sys.stderr,
    )

    # Re-exec so LD_LIBRARY_PATH takes effect for the dynamic linker.
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = venv_lib_dir + (":" + ld if ld else "")
    env[_XCB_SENTINEL] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def _fetch_xcb_cursor_lib(dest_dir: str) -> None:
    """Download ``libxcb-cursor0`` and extract the ``.so`` into *dest_dir*.

    Uses ``apt-get download`` (works without root on Debian/Ubuntu) and
    ``dpkg-deb -x`` to extract — both are always present on these systems.
    """
    os.makedirs(dest_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(
            "[MEGqc] Running: apt-get download libxcb-cursor0  (no root required)",
            file=sys.stderr,
        )
        subprocess.run(
            ["apt-get", "download", "libxcb-cursor0"],
            cwd=tmp, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        debs = [f for f in os.listdir(tmp) if f.endswith(".deb")]
        if not debs:
            raise RuntimeError("apt-get download did not produce a .deb file")

        deb_path = os.path.join(tmp, debs[0])
        print(
            f"[MEGqc] Downloaded {debs[0]}",
            file=sys.stderr,
        )

        extract_dir = os.path.join(tmp, "extracted")
        print(
            f"[MEGqc] Extracting {_XCB_SO_NAME} into {dest_dir}",
            file=sys.stderr,
        )
        subprocess.run(
            ["dpkg-deb", "-x", deb_path, extract_dir],
            check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        # Walk the extraction tree looking for the .so (resolve symlinks)
        for root, _dirs, files in os.walk(extract_dir):
            for fname in files:
                if "libxcb-cursor" in fname:
                    src = os.path.realpath(os.path.join(root, fname))
                    if os.path.isfile(src):
                        shutil.copy2(
                            src,
                            os.path.join(dest_dir, _XCB_SO_NAME),
                        )
                        return

        raise RuntimeError(f"{_XCB_SO_NAME} not found inside the .deb")


# ── Linux WebEngine GPU fix ───────────────────────────────────────────────
# On some Linux systems (especially hybrid-GPU laptops like NVIDIA Optimus),
# QtWebEngine's Chromium backend fails to composite via DMA-BUF / Vulkan,
# resulting in blank WebEngine views and floods of
#   "Failed to get native pixmap due to dma_buf acquisition failure"
#   "Backend texture is not a Vulkan texture"
#   "Compositor returned null texture"
#
# Two targeted flags fix this without killing *all* GPU acceleration:
#   --in-process-gpu        keep GPU ops in the main process so DMA-BUF
#                           buffer-sharing between processes is not needed.
#   --disable-gpu-compositing   fall back to a software compositor while
#                               GPU rasterisation stays active.
#
# Together they fix the blank-page bug while still letting Chromium use
# the GPU for rasterisation (faster Plotly / heavy-SVG reports).
# ``setdefault`` ensures a user-supplied value always takes precedence.
if sys.platform == "linux":
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--in-process-gpu --disable-gpu-compositing",
    )

# ── WebEngine early init ──────────────────────────────────────────────────
# AA_ShareOpenGLContexts must be set *before* any WebEngine import or
# QApplication creation.  On Qt6 the attribute may already be implicit but
# setting it explicitly ensures Chromium's GPU process can share the GL
# context (critical for macOS hardware-accelerated rendering of large HTML).
from PyQt6.QtCore import QCoreApplication as _QCA, Qt as _Qt
try:
    _QCA.setAttribute(_Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
except Exception:
    pass  # not fatal on Qt6 — context sharing is on by default

# ── WebEngine import ──────────────────────────────────────────────────────
# Importing here (before QApplication) is required on macOS for Chromium to
# initialise correctly.  The broad except catches any crash on platforms
# where the binary is missing or incompatible; in that case HTML files fall
# back to the system browser.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception as _webengine_exc:
    import sys as _sys
    print(f"[MEGqc] WebEngine unavailable: {_webengine_exc}", file=_sys.stderr)
    QWebEngineView = None  # type: ignore[assignment,misc]
    HAS_WEBENGINE = False
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Attempt to import psutil for accurate RAM info; if unavailable, fallback later
try:
    import psutil

    has_psutil = True
except ImportError:
    has_psutil = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QLabel,
    QFileDialog, QPlainTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSpinBox, QTabWidget, QScrollArea, QFrame, QMessageBox,
    QListWidget, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QComboBox, QDoubleSpinBox, QDialogButtonBox, QGridLayout,
    QInputDialog, QTreeView, QListView, QAbstractItemView, QSizePolicy,
    QAbstractScrollArea,
)
from PyQt6.QtCore import (
    QObject,
    QProcess,
    pyqtSignal,
    Qt,
    QCoreApplication,
    QSettings,
    QTimer,
    QStandardPaths,
)
from PyQt6.QtGui import QPixmap, QIcon, QPalette, QColor

# Core MEG QC pipeline functions
from meg_qc.test import (
    run_calculation_dispatch,
    run_all_dispatch,
    run_plotting_dispatch,
    run_gqi_dispatch,
    validate_plot_request,
)
from meg_qc.calculation.meg_qc_pipeline import list_analysis_profiles
from meg_qc import __version__ as MEGQC_VERSION
# Output monitoring: live terminal streaming + CLI launcher
from .output_monitoring import LiveTerminalDialog, open_cli_terminal
# Use Qt built-in dialogs (consistent cross-platform behaviour)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)

# Locate bundled settings and logo files within the package
# Locate bundled settings and logo files within the package by filepath
# GUI_DIR es la carpeta donde está este mismo archivo
GUI_DIR = Path(__file__).parent
# PKG_ROOT apunta a la raíz del paquete meg_qc (dos niveles arriba)
PKG_ROOT = GUI_DIR.parent.parent

SETTINGS_PATH = PKG_ROOT / "settings" / "settings.ini"
INTERNAL_PATH = PKG_ROOT / "settings" / "settings_internal.ini"
# Logo shown in the header of the Run tab
LOGO_PATH = GUI_DIR / "logo.png"

# Window / taskbar icon — platform-specific bundled assets
_ASSETS = GUI_DIR / "assets"
_WIN_ICON  = _ASSETS / "windows" / "AppIcon.ico"
_MAC_ICON  = _ASSETS / "macos"   / "AppIcon.icns"
# PNG sizes available for Linux (and as fallback on any platform)
_PNG_SIZES = [16, 32, 64, 128, 256, 512]


def _build_app_icon() -> "QIcon":
    """Return the best available QIcon for the current platform."""
    import sys as _sys
    if _sys.platform == "win32" and _WIN_ICON.exists():
        return QIcon(str(_WIN_ICON))
    if _sys.platform == "darwin" and _MAC_ICON.exists():
        return QIcon(str(_MAC_ICON))
    # Linux (or fallback): compose a multi-resolution icon from the PNG set
    icon = QIcon()
    macos_dir = _ASSETS / "macos"
    for size in _PNG_SIZES:
        p = macos_dir / f"AppIcon{size}.png"
        if p.exists():
            icon.addPixmap(QPixmap(str(p)))
    if icon.isNull() and LOGO_PATH.exists():
        icon = QIcon(str(LOGO_PATH))
    return icon



# megqcGUI.py

# -----------------------------------------------------------------------------
# Helper that mirrors the robust shutdown logic used in BIDS-Manager.
# Given a PID, we terminate the full process tree so that joblib/loky
# workers disappear alongside the main task.  We try ``killpg`` first to
# cover POSIX platforms, then fall back to psutil (when available) or a
# plain SIGTERM as a last resort.  The GUI process itself is never targeted.
# -----------------------------------------------------------------------------
def _terminate_process_tree(pid: int) -> None:
    if pid <= 0:
        return

    # Avoid killing our own process group, which would close the GUI.
    try:
        pgid = os.getpgid(pid)
        if pgid != os.getpgid(0):
            os.killpg(pgid, signal.SIGTERM)
            return
    except Exception:
        # killpg not available (e.g. Windows) or call failed → fall through
        pass

    if has_psutil:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=3)
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        return

    # Fallback when psutil is not present: send SIGTERM to the PID.
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


class Worker(QObject):
    """Run MEG-QC tasks in a detached Python process managed via ``QProcess``.

    The design closely follows the strategy used in the external BIDS-Manager
    project: each task runs in its own interpreter so the GUI remains
    responsive, and we keep the process ID handy so we can terminate the
    entire joblib tree if the user clicks “Stop”.  Signals mirror the previous
    ``QThread`` API so the rest of the GUI code does not need to change.
    """

    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)
    output_ready = pyqtSignal(str)  # emits captured stdout/stderr lines

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args
        self.process: Optional[QProcess] = None
        self._stopped_by_user = False
        self._had_error = False

    # ------------------------------------------------------------------
    # Process lifecycle helpers
    # ------------------------------------------------------------------
    def _build_payload(self) -> str:
        """JSON-encode ``self.args`` converting Paths to strings recursively."""

        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(val) for key, val in value.items()}
            return value

        normalized = [convert(arg) for arg in self.args]
        return json.dumps(normalized, ensure_ascii=False)

    def _cleanup(self) -> None:
        """Release the ``QProcess`` instance and reset transient flags."""

        if self.process is not None:
            try:
                self.process.finished.disconnect(self._on_finished)
            except Exception:
                pass
            try:
                self.process.errorOccurred.disconnect(self._on_error)
            except Exception:
                pass
            try:
                self.process.started.disconnect(self._on_started)
            except Exception:
                pass
            self.process.deleteLater()
        self.process = None
        self._stopped_by_user = False
        self._had_error = False

    # ------------------------------------------------------------------
    # Slots connected to QProcess signals
    # ------------------------------------------------------------------
    def _on_started(self) -> None:
        self.started.emit()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if self._stopped_by_user:
            return

        self._had_error = True
        if error == QProcess.ProcessError.FailedToStart:
            message = "Failed to start background process"
        elif error == QProcess.ProcessError.Crashed:
            message = "Background process crashed"
        else:
            message = "Background process encountered an unknown error"
        self.error.emit(message)
        self._cleanup()

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        if self._stopped_by_user or self._had_error:
            self._cleanup()
            return

        if status != QProcess.ExitStatus.NormalExit:
            self.error.emit("Background process exited unexpectedly")
            self._cleanup()
            return

        if exit_code == 0:
            self.finished.emit()
        else:
            self.error.emit(f"Process exited with code {exit_code}")
        self._cleanup()

    def _on_stdout_ready(self) -> None:
        """Read available stdout/stderr data, emit via output_ready AND
        echo to the parent-process stdout so both the Live Terminal dialog
        *and* any real terminal (Linux/Windows) receive the output.
        """
        if self.process is None:
            return
        data = self.process.readAllStandardOutput()
        if data:
            text = bytes(data).decode("utf-8", errors="replace")
            # GUI live terminal
            self.output_ready.emit(text)
            # Real terminal (for users on Linux / Windows or who launched
            # megqc from a terminal on macOS).
            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API used by the GUI
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch the worker process using ``python -m ...worker_entry``."""

        if self.process is not None:
            raise RuntimeError("Worker is already running")

        entry_module = "meg_qc.miscellaneous.GUI.worker_entry"
        func_path = f"{self.func.__module__}:{self.func.__name__}"
        payload = self._build_payload()

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_stdout_ready)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        process.started.connect(self._on_started)
        process.setProgram(sys.executable)
        process.setArguments(["-u", "-m", entry_module, "--func", func_path, "--args", payload])
        self.process = process
        process.start()

        if process.error() == QProcess.ProcessError.FailedToStart:
            # ``start()`` is asynchronous, but ``error()`` already knows if
            # the executable could not be launched.  Trigger the handler
            # manually so callers receive the signal immediately.
            self._on_error(QProcess.ProcessError.FailedToStart)

    def stop(self) -> None:
        """Terminate the running process and its children, mirroring BIDS-Manager."""

        if self.process is None:
            return

        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._cleanup()
            return

        self._stopped_by_user = True
        pid = int(self.process.processId())
        if pid > 0:
            _terminate_process_tree(pid)

        # ``kill`` complements the explicit tree teardown above.  When the child
        # already exited this is a no-op, otherwise it ensures the GUI regains
        # control even if the task ignores SIGTERM.
        self.process.kill()


def _runtime_config_dir() -> Path:
    """Return and create the writable folder used for runtime config copies."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    if not base:
        base = str(Path.home() / ".config" / "MEGqc")
    path = Path(base) / "runtime_configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class SettingsEditorDialog(QDialog):
    """
    Typed INI editor used by GUI config manager buttons.

    Design goals:
    - Never modify package defaults in-place.
    - Prefer selectable controls (bool/combo/numeric) over free typing.
    - Display sections in two columns for better use of screen width.
    """

    _ENUM_OPTIONS: Dict[Tuple[str, str], List[str]] = {
        ("Filtering", "method"): ["iir", "fir"],
        ("Epoching", "event_repeated"): ["merge", "drop", "error"],
        ("EEG", "reference_method"): ["average", "REST", "none"],
        ("EEG", "montage"): [
            "auto", "standard_1020", "standard_1010", "standard_1005",
            "biosemi64", "biosemi128", "biosemi256",
            "GSN-HydroCel-129", "GSN-HydroCel-256",
            "mgh60", "mgh70",
        ],
    }
    # These fields intentionally allow blank values in settings.ini.
    # Using text widgets here avoids forcing a numeric value during resets.
    _OPTIONAL_NUMERIC_KEYS = {
        ("GENERAL", "data_crop_tmin"),
        ("GENERAL", "data_crop_tmax"),
    }

    # Fields that accept either an integer value OR the literal "False" to disable the feature.
    _INT_OR_FALSE_KEYS = {
        ("Filtering", "downsample_to_hz"),
    }

    def __init__(self, source_path: Union[str, Path], defaults_path: Union[str, Path], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1080, 760)

        self.source_path = Path(source_path)
        self.defaults_path = Path(defaults_path)
        # Build an editor model from package defaults first, then overlay
        # user/runtime values. This keeps the GUI aligned to supported keys and
        # silently drops deprecated legacy options from older profiles.
        self.template_config = configparser.ConfigParser()
        self.template_config.optionxform = str
        self.template_config.read(self.defaults_path)

        source_cfg = configparser.ConfigParser()
        source_cfg.optionxform = str
        source_cfg.read(self.source_path)

        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        for section in self.template_config.sections():
            self.config.add_section(section)
            for key, default_val in self.template_config[section].items():
                if section in source_cfg and key in source_cfg[section]:
                    self.config[section][key] = source_cfg[section][key]
                else:
                    self.config[section][key] = default_val

        # Fallback for corrupted/missing defaults: preserve old behavior and
        # show whatever exists in source config instead of failing.
        if not self.config.sections():
            self.config.read(self.source_path)

        self.comment_map = self._build_comment_map(self.defaults_path)
        self.widgets: Dict[Tuple[str, str], Tuple[QWidget, str]] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        sections = self._ordered_sections()
        for idx, section in enumerate(sections):
            box = QGroupBox(section)
            box.setMinimumWidth(260)
            box.setMaximumWidth(360)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
            for key, val in self._section_items(section):
                widget, widget_type = self._build_widget(section, key, str(val))
                self.widgets[(section, key)] = (widget, widget_type)
                tip = self.comment_map.get((section, key), "")
                label = QLabel(key)
                if tip:
                    label.setToolTip(tip)
                    widget.setToolTip(tip)
                form.addRow(label, widget)
            grid.addWidget(box, idx // 3, idx % 3)

        scroll.setWidget(container)
        root.addWidget(scroll)

        # Wire apply_filtering toggle AFTER all widgets are built
        self._wire_apply_filtering_toggle()

        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to defaults")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        self.btn_save_as = QPushButton("Save as...")
        self.btn_save_as.clicked.connect(self._save_as)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_save_as)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _wire_apply_filtering_toggle(self):
        """
        When 'apply_filtering' is unchecked, gray-out the filter frequency and
        method fields so the user understands they have no effect.
        Downsampling remains independently controllable.
        """
        if ("Filtering", "apply_filtering") not in self.widgets:
            return
        chk_widget, _ = self.widgets[("Filtering", "apply_filtering")]
        # These keys only make sense when filtering is on
        filter_dependent_keys = ["l_freq", "h_freq", "method"]

        def _set_enabled(state):
            enabled = bool(state)
            for k in filter_dependent_keys:
                if ("Filtering", k) in self.widgets:
                    w, _ = self.widgets[("Filtering", k)]
                    w.setEnabled(enabled)

        chk_widget.stateChanged.connect(_set_enabled)
        # Apply initial state on dialog open
        _set_enabled(chk_widget.isChecked())

    def _ordered_sections(self) -> List[str]:
        sections = list(self.config.sections())
        if "GENERAL" in sections:
            sections = ["GENERAL"] + [s for s in sections if s != "GENERAL"]
        return sections

    def _section_items(self, section: str):
        return list(self.config[section].items())

    @staticmethod
    def _build_comment_map(path: Path) -> Dict[Tuple[str, str], str]:
        """
        Parse INI comments and map them to (section, key) tooltips.
        Uses the package default INI where comments are preserved.
        """
        mapping: Dict[Tuple[str, str], str] = {}
        current_section = "DEFAULT"
        pending: List[str] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    stripped = line.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        current_section = stripped[1:-1]
                        pending = []
                    elif stripped.startswith("#") or stripped.startswith(";"):
                        txt = stripped.lstrip("#; ").strip()
                        if txt:
                            pending.append(txt)
                    elif "=" in stripped and not stripped.startswith("#") and not stripped.startswith(";"):
                        key = stripped.split("=", 1)[0].strip()
                        mapping[(current_section, key)] = " ".join(pending).strip()
                        pending = []
        except Exception:
            return {}
        return mapping

    def _build_widget(self, section: str, key: str, raw: str) -> Tuple[QWidget, str]:
        val = raw.strip()
        key_lower = key.lower()
        sec_key = (section, key)

        if section == "GENERAL" and key_lower == "ch_types":
            container = QWidget()
            lay = QHBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            chk_mag = QCheckBox("mag")
            chk_grad = QCheckBox("grad")
            chk_eeg = QCheckBox("eeg")
            chk_mag.setMinimumWidth(72)
            chk_grad.setMinimumWidth(72)
            chk_eeg.setMinimumWidth(72)
            parts = [p.strip().lower() for p in val.split(",") if p.strip()]
            chk_mag.setChecked("mag" in parts)
            chk_grad.setChecked("grad" in parts)
            chk_eeg.setChecked("eeg" in parts)
            lay.addWidget(chk_mag)
            lay.addWidget(chk_grad)
            lay.addWidget(chk_eeg)
            lay.addStretch(1)
            container.setProperty("ch_types_widgets", (chk_mag, chk_grad, chk_eeg))
            return container, "ch_types"

        if sec_key in self._ENUM_OPTIONS:
            combo = QComboBox()
            options = self._ENUM_OPTIONS[sec_key]
            combo.addItems(options)
            if val in options:
                combo.setCurrentText(val)
            combo.setMinimumWidth(135)
            combo.setMaximumWidth(205)
            return combo, "enum"

        if val.lower() in {"true", "false"} and sec_key not in self._INT_OR_FALSE_KEYS:
            chk = QCheckBox()
            chk.setChecked(val.lower() == "true")
            return chk, "bool"

        if sec_key in self._OPTIONAL_NUMERIC_KEYS:
            line = QLineEdit(val)
            line.setPlaceholderText("empty")
            line.setMinimumWidth(145)
            line.setMaximumWidth(215)
            return line, "text"

        if sec_key in self._INT_OR_FALSE_KEYS:
            container = QWidget()
            lay = QHBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            chk = QCheckBox("enable")
            spin = QSpinBox()
            spin.setRange(1, 1_000_000)
            spin.setMinimumWidth(100)
            spin.setMaximumWidth(170)
            spin.setSuffix(" Hz")
            is_disabled = val.strip().lower() in ("false", "0", "no", "none", "")
            if is_disabled:
                chk.setChecked(False)
                spin.setValue(1000)
                spin.setEnabled(False)
            else:
                try:
                    spin.setValue(int(val.strip()))
                except (ValueError, TypeError):
                    spin.setValue(1000)
                chk.setChecked(True)
                spin.setEnabled(True)
            chk.stateChanged.connect(lambda state, s=spin: s.setEnabled(bool(state)))
            lay.addWidget(chk)
            lay.addWidget(spin)
            container.setProperty("int_or_false_widgets", (chk, spin))
            return container, "int_or_false"

        if val == "":
            line = QLineEdit("")
            line.setPlaceholderText("empty")
            line.setMinimumWidth(145)
            line.setMaximumWidth(215)
            return line, "text"

        try:
            ival = int(val)
            spin = QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(ival)
            spin.setMinimumWidth(145)
            spin.setMaximumWidth(215)
            return spin, "int"
        except Exception:
            pass

        try:
            fval = float(val)
            dspin = QDoubleSpinBox()
            # Keep float editing concise and readable in the settings dialog.
            dspin.setDecimals(3)
            dspin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            dspin.setValue(fval)
            dspin.setMinimumWidth(145)
            dspin.setMaximumWidth(215)
            return dspin, "float"
        except Exception:
            pass

        line = QLineEdit(val)
        line.setMinimumWidth(145)
        line.setMaximumWidth(215)
        return line, "text"

    def _widget_to_value(self, widget: QWidget, widget_type: str) -> str:
        if widget_type == "ch_types":
            chk_mag, chk_grad, chk_eeg = widget.property("ch_types_widgets")
            chosen = []
            if chk_mag.isChecked():
                chosen.append("mag")
            if chk_grad.isChecked():
                chosen.append("grad")
            if chk_eeg.isChecked():
                chosen.append("eeg")
            return ", ".join(chosen)
        if widget_type == "enum":
            return str(widget.currentText())
        if widget_type == "bool":
            return "True" if widget.isChecked() else "False"
        if widget_type == "int_or_false":
            chk, spin = widget.property("int_or_false_widgets")
            return str(spin.value()) if chk.isChecked() else "False"
        if widget_type == "int":
            return str(int(widget.value()))
        if widget_type == "float":
            return str(float(widget.value()))
        return str(widget.text())

    def to_config(self) -> configparser.ConfigParser:
        out = configparser.ConfigParser()
        out.optionxform = str
        for section in self._ordered_sections():
            out.add_section(section)
            for key, _ in self._section_items(section):
                widget, widget_type = self.widgets[(section, key)]
                out[section][key] = self._widget_to_value(widget, widget_type)
        return out

    def _reset_to_defaults(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(self.defaults_path)
        for (section, key), (widget, widget_type) in self.widgets.items():
            if section not in cfg or key not in cfg[section]:
                continue
            val = cfg[section][key]
            if widget_type == "ch_types":
                chk_mag, chk_grad, chk_eeg = widget.property("ch_types_widgets")
                parts = [p.strip().lower() for p in val.split(",") if p.strip()]
                chk_mag.setChecked("mag" in parts)
                chk_grad.setChecked("grad" in parts)
                chk_eeg.setChecked("eeg" in parts)
            elif widget_type == "enum":
                widget.setCurrentText(val)
            elif widget_type == "bool":
                widget.setChecked(str(val).lower() == "true")
            elif widget_type == "int":
                try:
                    widget.setValue(int(val))
                except Exception:
                    pass
            elif widget_type == "int_or_false":
                chk, spin = widget.property("int_or_false_widgets")
                is_disabled = val.strip().lower() in ("false", "0", "no", "none", "")
                if is_disabled:
                    chk.setChecked(False)
                    spin.setValue(1000)
                    spin.setEnabled(False)
                else:
                    try:
                        spin.setValue(int(val.strip()))
                    except Exception:
                        spin.setValue(1000)
                    chk.setChecked(True)
                    spin.setEnabled(True)
            elif widget_type == "float":
                try:
                    widget.setValue(float(val))
                except Exception:
                    pass
            else:
                widget.setText(str(val))

    def _save_as(self):
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Save settings as",
            str(Path.home() / "settings.ini"),
            "INI files (*.ini);;All files (*)",
        )
        if not target:
            return
        cfg = self.to_config()
        with open(target, "w", encoding="utf-8") as f:
            cfg.write(f)

    def _write_to_source(self) -> None:
        cfg = self.to_config()
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.source_path, "w", encoding="utf-8") as f:
            cfg.write(f)

    def accept(self):
        try:
            self._write_to_source()
        except Exception as exc:
            QMessageBox.critical(self, "Settings", f"Failed to save settings:\n{exc}")
            return
        super().accept()


class GUISettingsDialog(QDialog):
    """Popup window for GUI-only preferences (theme, system info)."""

    def __init__(self, parent, themes: Dict[str, QPalette], active_theme: str, cpu_count: int, total_gb: float):
        super().__init__(parent)
        self.setWindowTitle("GUI Settings")
        self.resize(420, 220)
        self._themes = themes

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(themes.keys()))
        if active_theme in themes:
            self.theme_combo.setCurrentText(active_theme)
        form.addRow("Theme:", self.theme_combo)
        form.addRow("CPUs:", QLabel(str(cpu_count)))
        form.addRow("Total RAM (GB):", QLabel(f"{total_gb:.1f}"))
        lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


class SubjectSelectionDialog(QDialog):
    """Dataset subject selector with All/Specific modes."""

    def __init__(self, dataset_label: str, subjects: List[str], current_value: Union[str, List[str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Subjects: {dataset_label}")
        self.resize(520, 620)

        lay = QVBoxLayout(self)
        self.chk_all = QCheckBox("All subjects")
        self.chk_all.toggled.connect(self._on_all_toggled)
        lay.addWidget(self.chk_all)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for sub in subjects:
            self.list_widget.addItem(sub)
        lay.addWidget(self.list_widget)

        helper = QLabel("Tip: use Ctrl/Cmd-click or Shift-click to select multiple subjects.")
        helper.setWordWrap(True)
        lay.addWidget(helper)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._set_initial_selection(current_value)

    def _set_initial_selection(self, current_value: Union[str, List[str]]) -> None:
        if current_value == "all":
            self.chk_all.setChecked(True)
            return
        chosen = {str(s) for s in (current_value or [])}
        if not chosen:
            self.chk_all.setChecked(True)
            return
        self.chk_all.setChecked(False)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setSelected(item.text() in chosen)

    def _on_all_toggled(self, checked: bool) -> None:
        self.list_widget.setEnabled(not checked)

    def selected_value(self) -> Union[str, List[str]]:
        if self.chk_all.isChecked():
            return "all"
        selected = [item.text() for item in self.list_widget.selectedItems()]
        return selected if selected else "all"


class _DetachedSectionDialog(QDialog):
    """Floating container used by detachable main sections."""

    def __init__(self, title: str, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self._content_widget: Optional[QWidget] = None
        self._closing_by_owner = False
        self.setWindowTitle(f"{title} (detached)")
        self.resize(900, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.content_layout, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("Reattach")
        btn.clicked.connect(self.owner.reattach_section)
        row.addWidget(btn)
        root.addLayout(row)

    def set_content_widget(self, widget: QWidget) -> None:
        self._content_widget = widget
        self.content_layout.addWidget(widget)

    def take_content_widget(self) -> Optional[QWidget]:
        if self._content_widget is None:
            return None
        widget = self._content_widget
        self.content_layout.removeWidget(widget)
        widget.setParent(None)
        self._content_widget = None
        return widget

    def close_by_owner(self) -> None:
        self._closing_by_owner = True
        self.close()

    def closeEvent(self, event):
        if not self._closing_by_owner and self.owner is not None:
            self.owner._handle_detached_dialog_closed()
        super().closeEvent(event)


class CollapsibleDetachableSection(QWidget):
    """Section wrapper with collapse and detach/reattach controls."""

    def __init__(self, title: str, content_widget: QWidget, parent=None):
        super().__init__(parent)
        self.title = title
        self.content_widget = content_widget
        self.detached_dialog: Optional[_DetachedSectionDialog] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Single visual frame for the whole section (header + body) to avoid
        # rendering as two separate "bubbles" in expanded state.
        self.frame = QWidget()
        self.frame.setObjectName("sectionFrame")
        # Expanding policy so the frame grows when the parent allocates more space.
        self.frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_lay = QVBoxLayout(self.frame)
        # 1px side/bottom margins keep children inside the 1px border and
        # prevent rectangular child widgets from overlapping the rounded
        # corners of the sectionFrame (border-radius: 8px).
        frame_lay.setContentsMargins(4, 0, 4, 4)
        frame_lay.setSpacing(0)
        root.addWidget(self.frame, stretch=1)

        header = QWidget()
        header.setObjectName("sectionHeader")
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(4, 2, 4, 2)
        header_lay.setSpacing(6)

        self.btn_toggle = QPushButton(f"▼ {self.title}")
        self.btn_toggle.setObjectName("sectionToggle")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.clicked.connect(self._toggle_collapsed)
        self.btn_toggle.setFlat(True)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_detach = QPushButton("Detach")
        self.btn_detach.setObjectName("sectionDetach")
        self.btn_detach.clicked.connect(self._toggle_detached)
        self.btn_detach.setCursor(Qt.CursorShape.PointingHandCursor)

        header_lay.addWidget(self.btn_toggle)
        header_lay.addStretch(1)
        header_lay.addWidget(self.btn_detach)
        frame_lay.addWidget(header, stretch=0)

        self.body = QWidget()
        self.body.setObjectName("sectionBody")
        # Expanding policy so the body fills all space below the header.
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        # Inner "mainSection" group boxes are embedded into the outer frame,
        # so they should not draw a second border.
        self.content_widget.setProperty("embeddedSection", True)
        self.body_lay.addWidget(self.content_widget)
        # stretch=1 so body takes all remaining vertical space inside the frame
        # (the header stays at its natural height).
        frame_lay.addWidget(self.body, stretch=1)

    def _toggle_collapsed(self, checked: bool) -> None:
        arrow = "▼" if checked else "▶"
        self.btn_toggle.setText(f"{arrow} {self.title}")
        if self.detached_dialog is None:
            self.body.setVisible(checked)

    def _toggle_detached(self) -> None:
        if self.detached_dialog is None:
            self.detach_section()
        else:
            self.reattach_section()

    def detach_section(self) -> None:
        if self.detached_dialog is not None:
            return
        self.body_lay.removeWidget(self.content_widget)
        self.content_widget.setParent(None)
        self.detached_dialog = _DetachedSectionDialog(self.title, self, self.window())
        self.detached_dialog.set_content_widget(self.content_widget)
        self.detached_dialog.show()
        self.btn_detach.setText("Reattach")
        self.body.setVisible(False)

    def reattach_section(self) -> None:
        if self.detached_dialog is None:
            return
        widget = self.detached_dialog.take_content_widget()
        if widget is not None:
            self.body_lay.addWidget(widget)
        dialog = self.detached_dialog
        self.detached_dialog = None
        self.btn_detach.setText("Detach")
        self.body.setVisible(self.btn_toggle.isChecked())
        dialog.close_by_owner()
        dialog.deleteLater()

    def _handle_detached_dialog_closed(self) -> None:
        if self.detached_dialog is None:
            return
        widget = self.detached_dialog.take_content_widget()
        if widget is not None:
            self.body_lay.addWidget(widget)
        self.detached_dialog = None
        self.btn_detach.setText("Detach")
        self.body.setVisible(self.btn_toggle.isChecked())



class MainWindow(QMainWindow):
    """
    Main application window for dataset execution and report generation.
    """

    # ──────────────────────────────── #
    # constructor                      #
    # ──────────────────────────────── #
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MEGqc")
        self.resize(700, 780)
        self.setWindowIcon(_build_app_icon())

        # Runtime config manager state.
        # Package defaults are read-only; GUI edits always go to runtime copies.
        self.default_settings_path = Path(SETTINGS_PATH)
        self.global_config_path: Optional[str] = None
        self.dataset_config_paths: Dict[str, str] = {}

        # Persistent GUI preferences (theme etc.) are stored cross-platform
        # through QSettings backend (plist on macOS, registry on Windows, INI on Linux).
        self.settings_store = QSettings("ANCP", "MEGqc")

        self.themes = self._build_theme_dict()
        self.cpu_count, self.total_ram_gb = self._detect_system_resources()
        self.active_theme_name = "Ocean"
        self.installed_megqc_version = str(MEGQC_VERSION)

        # Bottom execution status controls (rendered inside main layout).
        self.spinner_frames = ["|", "/", "-", "\\"]
        self.spinner_idx = 0
        self.spinner_label = QLabel("")
        self.spinner_label.setFixedWidth(14)
        self.elapsed_label = QLabel("Active tasks: none")

        self.workers: Dict[str, Worker] = {}
        self.task_started_at: Dict[str, float] = {}

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self._refresh_elapsed_label)

        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(130)
        self.spinner_timer.timeout.connect(self._animate_spinner)

        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(8, 4, 8, 8)
        vlay.setSpacing(4)

        if LOGO_PATH and LOGO_PATH.exists():
            logo = QLabel()
            pix = QPixmap(str(LOGO_PATH))
            pix = pix.scaledToHeight(54, Qt.TransformationMode.SmoothTransformation)
            logo.setPixmap(pix)
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vlay.addWidget(logo)

        vlay.addWidget(self._create_run_tab(), stretch=1)
        self.setCentralWidget(central)
        self.statusBar().setVisible(False)

        saved_theme = str(self.settings_store.value("ui/theme", "Ocean"))
        if saved_theme not in self.themes:
            saved_theme = "Ocean"
        self.apply_theme(saved_theme, persist=False)


    # ──────────────────────────────── #
    # palette dictionary builder       #
    # ──────────────────────────────── #
    def _build_theme_dict(self) -> Dict[str, QPalette]:
        """Return dictionary: theme label → QPalette."""
        themes: dict[str, QPalette] = {}

        # DARK ☾
        dark = QPalette()
        dark.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        dark.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark.setColor(QPalette.ColorRole.ToolTipBase, QColor(65, 65, 65))
        dark.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        # Use blue selection in Dark theme (instead of the legacy purple).
        dark.setColor(QPalette.ColorRole.Highlight, QColor(78, 163, 255))
        dark.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        dark.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
        themes["Dark"] = dark

        # LIGHT ☀
        light = QPalette()
        light.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)
        light.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        light.setColor(QPalette.ColorRole.Base, QColor(245, 245, 245))
        light.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.white)
        light.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        light.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        light.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        light.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        light.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        light.setColor(QPalette.ColorRole.Highlight, QColor(100, 149, 237))
        light.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        light.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))
        themes["Light"] = light

        # BEIGE 🏜
        beige = QPalette()
        beige.setColor(QPalette.ColorRole.Window, QColor(243, 232, 210))
        beige.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        beige.setColor(QPalette.ColorRole.Base, QColor(250, 240, 222))
        beige.setColor(QPalette.ColorRole.AlternateBase, QColor(246, 236, 218))
        beige.setColor(QPalette.ColorRole.ToolTipBase, QColor(236, 224, 200))
        beige.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        beige.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        beige.setColor(QPalette.ColorRole.Button, QColor(242, 231, 208))
        beige.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        beige.setColor(QPalette.ColorRole.Highlight, QColor(196, 148, 70))
        beige.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        beige.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 140, 110))
        themes["Beige"] = beige

        # OCEAN 🌊
        ocean = QPalette()
        ocean.setColor(QPalette.ColorRole.Window, QColor(225, 238, 245))  # pale teal
        ocean.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        ocean.setColor(QPalette.ColorRole.Base, QColor(240, 248, 252))  # alice blue
        ocean.setColor(QPalette.ColorRole.AlternateBase, QColor(230, 240, 247))
        ocean.setColor(QPalette.ColorRole.ToolTipBase, QColor(215, 230, 240))
        ocean.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        ocean.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        ocean.setColor(QPalette.ColorRole.Button, QColor(213, 234, 242))
        ocean.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        ocean.setColor(QPalette.ColorRole.Highlight, QColor(0, 123, 167))  # deep ocean blue
        ocean.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        ocean.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 155, 175))
        themes["Ocean"] = ocean

        # CONTRAST 🌓
        hc = QPalette()
        hc.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        hc.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        hc.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        hc.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        hc.setColor(QPalette.ColorRole.Highlight, QColor(255, 215, 0))  # vivid gold
        hc.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        hc.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 140))
        themes["Contrast"] = hc

        # SOLARIZED 🌞 (light variant)
        solar = QPalette()
        solar.setColor(QPalette.ColorRole.Window, QColor(253, 246, 227))  # solarized base3
        solar.setColor(QPalette.ColorRole.WindowText, QColor(101, 123, 131))  # base00
        solar.setColor(QPalette.ColorRole.Base, QColor(255, 250, 240))  # linen-ish
        solar.setColor(QPalette.ColorRole.AlternateBase, QColor(253, 246, 227))
        solar.setColor(QPalette.ColorRole.ToolTipBase, QColor(238, 232, 213))  # base2
        solar.setColor(QPalette.ColorRole.ToolTipText, QColor(88, 110, 117))  # base01
        solar.setColor(QPalette.ColorRole.Text, QColor(88, 110, 117))
        solar.setColor(QPalette.ColorRole.Button, QColor(238, 232, 213))
        solar.setColor(QPalette.ColorRole.ButtonText, QColor(88, 110, 117))
        solar.setColor(QPalette.ColorRole.Highlight, QColor(38, 139, 210))  # solarized blue
        solar.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        solar.setColor(QPalette.ColorRole.PlaceholderText, QColor(147, 161, 161))
        themes["Solar"] = solar

        # CYBERPUNK 🕶
        cyber = QPalette()
        cyber.setColor(QPalette.ColorRole.Window, QColor(20, 20, 30))  # near black
        cyber.setColor(QPalette.ColorRole.WindowText, QColor(0, 255, 255))  # neon cyan
        cyber.setColor(QPalette.ColorRole.Base, QColor(30, 30, 45))
        cyber.setColor(QPalette.ColorRole.AlternateBase, QColor(25, 25, 35))
        cyber.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 65))
        cyber.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 0, 255))  # neon magenta
        cyber.setColor(QPalette.ColorRole.Text, QColor(0, 255, 255))
        cyber.setColor(QPalette.ColorRole.Button, QColor(40, 40, 55))
        cyber.setColor(QPalette.ColorRole.ButtonText, QColor(255, 0, 255))
        cyber.setColor(QPalette.ColorRole.Highlight, QColor(255, 0, 128))  # neon pink
        cyber.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        cyber.setColor(QPalette.ColorRole.PlaceholderText, QColor(0, 140, 140))
        themes["Cyber"] = cyber

        # DRACULA  🧛
        drac = QPalette()
        drac.setColor(QPalette.ColorRole.Window, QColor("#282a36"))
        drac.setColor(QPalette.ColorRole.WindowText, QColor("#f8f8f2"))
        drac.setColor(QPalette.ColorRole.Base, QColor("#1e1f29"))
        drac.setColor(QPalette.ColorRole.AlternateBase, QColor("#282a36"))
        drac.setColor(QPalette.ColorRole.ToolTipBase, QColor("#44475a"))
        drac.setColor(QPalette.ColorRole.ToolTipText, QColor("#f8f8f2"))
        drac.setColor(QPalette.ColorRole.Text, QColor("#f8f8f2"))
        drac.setColor(QPalette.ColorRole.Button, QColor("#44475a"))
        drac.setColor(QPalette.ColorRole.ButtonText, QColor("#f8f8f2"))
        drac.setColor(QPalette.ColorRole.Highlight, QColor("#bd93f9"))  # purple
        drac.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        drac.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6272a4"))
        themes["Dracula"] = drac

        # NORD  🧊
        nord = QPalette()
        nord.setColor(QPalette.ColorRole.Window, QColor("#2e3440"))
        nord.setColor(QPalette.ColorRole.WindowText, QColor("#d8dee9"))
        nord.setColor(QPalette.ColorRole.Base, QColor("#3b4252"))
        nord.setColor(QPalette.ColorRole.AlternateBase, QColor("#434c5e"))
        nord.setColor(QPalette.ColorRole.ToolTipBase, QColor("#4c566a"))
        nord.setColor(QPalette.ColorRole.ToolTipText, QColor("#eceff4"))
        nord.setColor(QPalette.ColorRole.Text, QColor("#e5e9f0"))
        nord.setColor(QPalette.ColorRole.Button, QColor("#4c566a"))
        nord.setColor(QPalette.ColorRole.ButtonText, QColor("#d8dee9"))
        nord.setColor(QPalette.ColorRole.Highlight, QColor("#88c0d0"))  # icy cyan
        nord.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        nord.setColor(QPalette.ColorRole.PlaceholderText, QColor("#4c566a"))
        themes["Nord"] = nord

        # GRUVBOX DARK  🪵
        gruv = QPalette()
        gruv.setColor(QPalette.ColorRole.Window, QColor("#282828"))
        gruv.setColor(QPalette.ColorRole.WindowText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.ColorRole.Base, QColor("#32302f"))
        gruv.setColor(QPalette.ColorRole.AlternateBase, QColor("#3c3836"))
        gruv.setColor(QPalette.ColorRole.ToolTipBase, QColor("#504945"))
        gruv.setColor(QPalette.ColorRole.ToolTipText, QColor("#fbf1c7"))
        gruv.setColor(QPalette.ColorRole.Text, QColor("#ebdbb2"))
        gruv.setColor(QPalette.ColorRole.Button, QColor("#504945"))
        gruv.setColor(QPalette.ColorRole.ButtonText, QColor("#ebdbb2"))
        gruv.setColor(QPalette.ColorRole.Highlight, QColor("#d79921"))  # warm yellow
        gruv.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        gruv.setColor(QPalette.ColorRole.PlaceholderText, QColor("#928374"))
        themes["Gruvbox"] = gruv

        # MONOKAI  🌵
        mono = QPalette()
        mono.setColor(QPalette.ColorRole.Window, QColor("#272822"))
        mono.setColor(QPalette.ColorRole.WindowText, QColor("#f8f8f2"))
        mono.setColor(QPalette.ColorRole.Base, QColor("#1e1f1c"))
        mono.setColor(QPalette.ColorRole.AlternateBase, QColor("#272822"))
        mono.setColor(QPalette.ColorRole.ToolTipBase, QColor("#3e3d32"))
        mono.setColor(QPalette.ColorRole.ToolTipText, QColor("#f8f8f2"))
        mono.setColor(QPalette.ColorRole.Text, QColor("#f8f8f2"))
        mono.setColor(QPalette.ColorRole.Button, QColor("#3e3d32"))
        mono.setColor(QPalette.ColorRole.ButtonText, QColor("#f8f8f2"))
        mono.setColor(QPalette.ColorRole.Highlight, QColor("#a6e22e"))  # neon green
        mono.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        mono.setColor(QPalette.ColorRole.PlaceholderText, QColor("#75715e"))
        themes["Monokai"] = mono

        # TOKYO NIGHT  🗼
        tokyo = QPalette()
        tokyo.setColor(QPalette.ColorRole.Window, QColor("#1a1b26"))  # night background
        tokyo.setColor(QPalette.ColorRole.WindowText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.ColorRole.Base, QColor("#1f2335"))
        tokyo.setColor(QPalette.ColorRole.AlternateBase, QColor("#24283b"))
        tokyo.setColor(QPalette.ColorRole.ToolTipBase, QColor("#414868"))
        tokyo.setColor(QPalette.ColorRole.ToolTipText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.ColorRole.Text, QColor("#c0caf5"))
        tokyo.setColor(QPalette.ColorRole.Button, QColor("#414868"))
        tokyo.setColor(QPalette.ColorRole.ButtonText, QColor("#c0caf5"))
        tokyo.setColor(QPalette.ColorRole.Highlight, QColor("#7aa2f7"))  # soft blue
        tokyo.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        tokyo.setColor(QPalette.ColorRole.PlaceholderText, QColor("#565f89"))
        themes["Tokyo"] = tokyo

        # CATPPUCCIN MOCHA 🐈
        mocha = QPalette()
        mocha.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
        mocha.setColor(QPalette.ColorRole.WindowText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.ColorRole.Base, QColor("#181825"))
        mocha.setColor(QPalette.ColorRole.AlternateBase, QColor("#1e1e2e"))
        mocha.setColor(QPalette.ColorRole.ToolTipBase, QColor("#313244"))
        mocha.setColor(QPalette.ColorRole.ToolTipText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.ColorRole.Text, QColor("#cdd6f4"))
        mocha.setColor(QPalette.ColorRole.Button, QColor("#313244"))
        mocha.setColor(QPalette.ColorRole.ButtonText, QColor("#cdd6f4"))
        mocha.setColor(QPalette.ColorRole.Highlight, QColor("#f38ba8"))
        mocha.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        mocha.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6c7086"))
        themes["Mocha"] = mocha

        # PALENIGHT 🎆
        pale = QPalette()
        pale.setColor(QPalette.ColorRole.Window, QColor("#292d3e"))
        pale.setColor(QPalette.ColorRole.WindowText, QColor("#a6accd"))
        pale.setColor(QPalette.ColorRole.Base, QColor("#1b1d2b"))
        pale.setColor(QPalette.ColorRole.AlternateBase, QColor("#222436"))
        pale.setColor(QPalette.ColorRole.ToolTipBase, QColor("#444267"))
        pale.setColor(QPalette.ColorRole.ToolTipText, QColor("#a6accd"))
        pale.setColor(QPalette.ColorRole.Text, QColor("#a6accd"))
        pale.setColor(QPalette.ColorRole.Button, QColor("#444267"))
        pale.setColor(QPalette.ColorRole.ButtonText, QColor("#a6accd"))
        pale.setColor(QPalette.ColorRole.Highlight, QColor("#82aaff"))
        pale.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        pale.setColor(QPalette.ColorRole.PlaceholderText, QColor("#585c79"))
        themes["Palenight"] = pale

        return themes

    # ──────────────────────────────── #
    # apply selected theme             #
    # ──────────────────────────────── #
    def apply_theme(self, name: str, persist: bool = True):
        """Apply palette chosen from the Theme menu."""
        self.active_theme_name = name
        QApplication.instance().setPalette(self.themes[name])
        self._apply_section_highlight_style()
        if persist:
            self.settings_store.setValue("ui/theme", name)
        # Propagate to the QC Viewer window if it is open
        viewer = getattr(self, "_qc_viewer_window", None)
        if viewer is not None:
            try:
                viewer.refresh_theme()
            except Exception:
                pass

    def _build_section_highlight_stylesheet(self) -> str:
        """
        Build a section style from the active theme palette so highlighted
        section chrome always follows user-selected colors.
        """
        pal = QApplication.instance().palette()
        hl = pal.color(QPalette.ColorRole.Highlight)
        base = pal.color(QPalette.ColorRole.Window)
        text = pal.color(QPalette.ColorRole.WindowText)

        # Explicit request: keep blue section accents when Dark theme is active.
        if getattr(self, "active_theme_name", "") == "Dark":
            hl = QColor(126, 200, 255)

        border_alpha = 185
        title_alpha = 225
        # Slight theme-aware tint so section boxes are visible in both
        # dark and light palettes without hard-coded color names.
        if base.lightness() < 110:
            section_bg = base.lighter(108)
        else:
            section_bg = base.darker(103)

        return (
            "QWidget#sectionFrame {"
            f" border: 1px solid rgba({hl.red()}, {hl.green()}, {hl.blue()}, {border_alpha});"
            " border-radius: 8px;"
            f" background-color: rgba({section_bg.red()}, {section_bg.green()}, {section_bg.blue()}, 35);"
            "}"
            "QGroupBox#mainSection {"
            f" border: 1px solid rgba({hl.red()}, {hl.green()}, {hl.blue()}, {border_alpha});"
            " border-radius: 8px;"
            " margin-top: 6px;"
            " font-weight: 600;"
            f" background-color: rgba({section_bg.red()}, {section_bg.green()}, {section_bg.blue()}, 35);"
            "}"
            "QGroupBox#mainSection[embeddedSection=\"true\"] {"
            " border: none;"
            " border-radius: 0px;"
            " margin-top: 0px;"
            " background: transparent;"
            "}"
            "QGroupBox#mainSection::title {"
            " subcontrol-origin: margin;"
            " left: 10px;"
            " padding: 0 6px;"
            f" color: rgba({hl.red()}, {hl.green()}, {hl.blue()}, {title_alpha});"
            "}"
            "QLabel#sectionCaption {"
            f" color: rgba({text.red()}, {text.green()}, {text.blue()}, 190);"
            " font-size: 11px;"
            "}"
            "QWidget#sectionHeader {"
            " border: none;"
            " border-top-left-radius: 8px;"
            " border-top-right-radius: 8px;"
            f" background-color: rgba({section_bg.red()}, {section_bg.green()}, {section_bg.blue()}, 60);"
            "}"
            "QWidget#sectionBody {"
            " border: none;"
            " border-bottom-left-radius: 8px;"
            " border-bottom-right-radius: 8px;"
            " background: transparent;"
            "}"
            "QPushButton#sectionToggle {"
            " border: none;"
            " text-align: left;"
            " font-weight: 700;"
            " padding: 4px 8px;"
            f" color: rgba({hl.red()}, {hl.green()}, {hl.blue()}, 230);"
            "}"
            "QPushButton#sectionToggle:hover {"
            f" background-color: rgba({hl.red()}, {hl.green()}, {hl.blue()}, 25);"
            " border-radius: 5px;"
            "}"
            "QPushButton#sectionDetach {"
            f" border: 1px solid rgba({hl.red()}, {hl.green()}, {hl.blue()}, 170);"
            " border-radius: 6px;"
            " padding: 2px 10px;"
            f" color: rgba({text.red()}, {text.green()}, {text.blue()}, 225);"
            f" background-color: rgba({section_bg.red()}, {section_bg.green()}, {section_bg.blue()}, 40);"
            "}"
            "QPushButton#sectionDetach:hover {"
            f" background-color: rgba({hl.red()}, {hl.green()}, {hl.blue()}, 30);"
            "}"
        )

    def _apply_section_highlight_style(self) -> None:
        if hasattr(self, "run_root_widget"):
            self.run_root_widget.setStyleSheet(self._build_section_highlight_stylesheet())

    def _detect_system_resources(self) -> Tuple[int, float]:
        cpu_cnt = os.cpu_count() or 1
        total_bytes = psutil.virtual_memory().total if has_psutil else 0
        if not has_psutil:
            try:
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_bytes = int(line.split()[1]) * 1024
                            break
            except Exception:
                total_bytes = 0
        total_gb = total_bytes / (1024**3) if total_bytes else 0.0
        return cpu_cnt, total_gb

    def _open_gui_settings_dialog(self):
        active_theme = str(self.settings_store.value("ui/theme", "Ocean"))
        dialog = GUISettingsDialog(
            self,
            themes=self.themes,
            active_theme=active_theme,
            cpu_count=self.cpu_count,
            total_gb=self.total_ram_gb,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_theme(dialog.theme_combo.currentText(), persist=True)

    # ------------------------------------------------------------------
    # PyPI helpers
    # ------------------------------------------------------------------

    def _fetch_pypi_payload(self) -> dict:
        """Fetch the raw PyPI JSON payload for meg_qc (shared SSL retry logic).

        SSL strategy:
        1. certifi CA bundle when available.
        2. System default CA context.
        3. Unverified fallback (read-only check — does not install anything).
        """
        req = Request(
            "https://pypi.org/pypi/meg_qc/json",
            headers={"User-Agent": "MEGqc-GUI-update-check"},
        )
        contexts = []
        try:
            import certifi
            contexts.append(ssl.create_default_context(cafile=certifi.where()))
        except Exception:
            pass
        contexts.append(ssl.create_default_context())

        last_exc: Optional[Exception] = None
        for ctx in contexts:
            try:
                with urlopen(req, timeout=8, context=ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc

        try:
            insecure_ctx = ssl._create_unverified_context()
            with urlopen(req, timeout=8, context=insecure_ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self._log("Update check used unverified SSL fallback (certificate validation unavailable).")
            return payload
        except Exception as exc:
            raise (last_exc or exc)

    def _fetch_latest_pypi_version(self, include_pre: bool = False) -> str:
        """Return the latest meg_qc version from PyPI.

        When *include_pre* is False (default) this returns ``info.version``
        which PyPI always sets to the latest *stable* release.  When True, the
        full ``releases`` dict is scanned so pre-releases are considered too.
        """
        payload = self._fetch_pypi_payload()
        if not include_pre:
            return str(payload.get("info", {}).get("version", "")).strip()
        try:
            from packaging.version import Version
            versions = [Version(v) for v in payload.get("releases", {}).keys()]
            return str(max(versions)) if versions else str(payload.get("info", {}).get("version", "")).strip()
        except Exception:
            return str(payload.get("info", {}).get("version", "")).strip()

    def _fetch_all_pypi_versions(self, include_pre: bool = True) -> List[str]:
        """Return every published meg_qc version, sorted newest → oldest.

        Parameters
        ----------
        include_pre:
            When False, alpha/beta/rc releases are omitted.
        """
        payload = self._fetch_pypi_payload()
        raw = list(payload.get("releases", {}).keys())
        try:
            from packaging.version import Version
            parsed = []
            for v in raw:
                try:
                    parsed.append(Version(v))
                except Exception:
                    pass
            if not include_pre:
                parsed = [v for v in parsed if not v.is_prerelease]
            parsed.sort(reverse=True)
            return [str(v) for v in parsed]
        except Exception:
            # packaging unavailable — crude filter + sort
            if not include_pre:
                raw = [v for v in raw if not any(t in v for t in ("a", "b", "rc", "dev", "post"))]
            return sorted(raw, reverse=True)

    @staticmethod
    def _is_newer_version(latest: str, current: str) -> bool:
        """Return True when *latest* is strictly newer than *current*."""
        try:
            from packaging.version import Version
            return Version(latest) > Version(current)
        except Exception:
            return bool(latest and latest != current)

    def _run_megqc_install(
        self,
        version: Optional[str] = None,
        include_pre: bool = False,
    ) -> Tuple[bool, str]:
        """Run pip to install meg_qc.

        Parameters
        ----------
        version:
            Pin to this exact version (``pip install meg_qc==X.Y.Z``).
            When *None*, upgrades to the latest release.
        include_pre:
            Append ``--pre`` when *version* is *None* (upgrade path only).
        """
        if version:
            cmd = [sys.executable, "-m", "pip", "install", f"meg_qc=={version}"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "meg_qc"]
            if include_pre:
                cmd.append("--pre")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except Exception as exc:
            return False, str(exc)
        if proc.returncode == 0:
            return True, (proc.stdout or "").strip()
        return False, (proc.stderr or proc.stdout or "Unknown pip error.").strip()

    def _finish_install(self, ok: bool, out: str) -> None:
        """Show success / failure dialog after any pip install."""
        if ok:
            QMessageBox.information(
                self, "Update completed",
                "MEGqc was updated successfully.\n"
                "Please restart the GUI to load the new version.",
            )
            self._log("Self-update completed successfully.")
        else:
            QMessageBox.warning(self, "Update failed", f"pip update failed.\n\n{out}")
            self._log(f"Self-update failed: {out}")

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    def _check_for_updates(self) -> None:
        """Check PyPI and optionally upgrade MEGqc from the GUI.

        Three paths are offered via an initial choice dialog:
        • Latest stable   — upgrade to the latest non-pre-release.
        • Latest (β/rc)   — upgrade to the highest version including betas.
        • Choose version… — pick any published version from a dropdown list.
        """
        current = self.installed_megqc_version

        # ── Initial choice dialog ──────────────────────────────────────
        msg = QMessageBox(self)
        msg.setWindowTitle("Check updates")
        msg.setText(
            f"Installed: MEGqc  {current}\n\n"
            "How would you like to check for updates?"
        )
        btn_stable  = msg.addButton("Latest stable",        QMessageBox.ButtonRole.AcceptRole)
        btn_pre     = msg.addButton("Latest (incl. beta)",  QMessageBox.ButtonRole.AcceptRole)
        btn_pick    = msg.addButton("Choose version…",      QMessageBox.ButtonRole.ActionRole)
        _btn_cancel = msg.addButton("Cancel",               QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_stable)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is _btn_cancel or clicked is None:
            return

        # ── "Choose version…" path ─────────────────────────────────────
        if clicked is btn_pick:
            self._log("Fetching all available versions from PyPI…")
            try:
                all_versions = self._fetch_all_pypi_versions(include_pre=True)
            except Exception as exc:
                QMessageBox.warning(self, "Check updates", f"Could not fetch version list.\n\n{exc}")
                self._log(f"Version fetch failed: {exc}")
                return

            if not all_versions:
                QMessageBox.warning(self, "Check updates", "No versions found on PyPI.")
                return

            # Mark the installed version so the user can spot it easily.
            labelled = [
                f"{v}  ← installed" if v == current else v
                for v in all_versions
            ]

            picked_label, ok = QInputDialog.getItem(
                self,
                "Choose MEGqc version",
                f"Select a version to install  (installed: {current}):",
                labelled,
                0,      # default selection: newest
                False,  # not editable
            )
            if not ok or not picked_label:
                return

            # Strip annotation back to a plain version string.
            picked_version = picked_label.split("  ←")[0].strip()

            if picked_version == current:
                QMessageBox.information(
                    self, "No change",
                    f"Version {picked_version} is already installed."
                )
                return

            try:
                from packaging.version import Version as _V
                is_pre = _V(picked_version).is_prerelease
            except Exception:
                is_pre = any(t in picked_version for t in ("a", "b", "rc", "dev"))

            pre_note = "\n⚠️  This is a pre-release (beta / rc) version." if is_pre else ""
            answer = QMessageBox.question(
                self, "Install version",
                f"Install MEGqc  {picked_version}?{pre_note}\n\n"
                f"Currently installed: {current}",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

            self._log(f"Running: pip install meg_qc=={picked_version}")
            self._finish_install(*self._run_megqc_install(version=picked_version))
            return

        # ── "Latest stable" / "Latest incl. beta" path ────────────────
        include_pre = (clicked is btn_pre)
        self._log(f"Checking PyPI for updates (installed: {current}, include_pre={include_pre})…")

        try:
            latest = self._fetch_latest_pypi_version(include_pre=include_pre)
        except Exception as exc:
            QMessageBox.warning(self, "Check updates", f"Could not check PyPI for updates.\n\n{exc}")
            self._log(f"Update check failed: {exc}")
            return

        if not latest:
            QMessageBox.warning(self, "Check updates", "PyPI responded without a valid version string.")
            self._log("Update check failed: empty PyPI version response.")
            return

        self._log(f"PyPI latest: {latest}  (pre-release channel: {include_pre})")

        if not self._is_newer_version(latest, current):
            QMessageBox.information(
                self, "Check updates",
                f"You are up to date.\nInstalled: {current}\nPyPI: {latest}",
            )
            return

        pre_note = "\n⚠️  This is a pre-release (beta / rc) version." if include_pre else ""
        answer = QMessageBox.question(
            self, "Update available",
            f"A newer MEGqc version is available.{pre_note}\n\n"
            f"Installed: {current}\n"
            f"PyPI:      {latest}\n\n"
            "Do you want to update now using pip?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._log("Update skipped by user.")
            return

        self._log(f"Running: pip install --upgrade meg_qc{' --pre' if include_pre else ''}")
        self._finish_install(*self._run_megqc_install(include_pre=include_pre))

    def _animate_spinner(self) -> None:
        if not self.task_started_at:
            self.spinner_label.setText("")
            return
        self.spinner_label.setText(self.spinner_frames[self.spinner_idx % len(self.spinner_frames)])
        self.spinner_idx += 1

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {message}")

    def _start_task_timer(self, key: str) -> None:
        self.task_started_at[key] = time.time()
        if not self.elapsed_timer.isActive():
            self.elapsed_timer.start()
        if not self.spinner_timer.isActive():
            self.spinner_timer.start()
        self._refresh_elapsed_label()

    def _stop_task_timer(self, key: str) -> float:
        started = self.task_started_at.pop(key, None)
        self._refresh_elapsed_label()
        if not self.task_started_at and self.elapsed_timer.isActive():
            self.elapsed_timer.stop()
        if not self.task_started_at and self.spinner_timer.isActive():
            self.spinner_timer.stop()
            self.spinner_label.setText("")
        if started is None:
            return 0.0
        return max(0.0, time.time() - started)

    def _refresh_elapsed_label(self) -> None:
        if not self.task_started_at:
            self.elapsed_label.setText("Active tasks: none")
            return
        parts = []
        now = time.time()
        for key in sorted(self.task_started_at.keys()):
            elapsed = int(max(0.0, now - self.task_started_at[key]))
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            parts.append(f"{key} {h:02d}:{m:02d}:{s:02d}")
        self.elapsed_label.setText("Active tasks: " + " | ".join(parts))

    def _set_active_run_task(self, task_key: Optional[str]) -> None:
        """Keep run controls mutually exclusive to prevent accidental overlap.

        Only one pipeline action can run at a time (calculation, plotting, GQI,
        or run-all). When a task is active, all run buttons are disabled and
        only the corresponding stop button stays enabled.
        """
        buttons = {
            "calc": (getattr(self, "btn_calc_run", None), getattr(self, "btn_calc_stop", None)),
            "plot": (getattr(self, "btn_plot_run", None), getattr(self, "btn_plot_stop", None)),
            "gqi": (getattr(self, "btn_gqi_run", None), getattr(self, "btn_gqi_stop", None)),
            "all": (getattr(self, "btn_all_run", None), getattr(self, "btn_all_stop", None)),
        }
        if task_key is None:
            for _key, (run_btn, stop_btn) in buttons.items():
                if run_btn is not None:
                    run_btn.setEnabled(True)
                if stop_btn is not None:
                    stop_btn.setEnabled(False)
            return
        for key, (run_btn, stop_btn) in buttons.items():
            if run_btn is not None:
                run_btn.setEnabled(False)
            if stop_btn is not None:
                stop_btn.setEnabled(key == task_key)

    def _reset_run_controls_if_idle(self) -> None:
        """Restore default run controls when no worker task is alive."""
        active = any(name in self.workers for name in ("calc", "plot", "gqi", "all"))
        if not active:
            self._set_active_run_task(None)

    def _ensure_no_active_pipeline_task(self, requester_label: str) -> bool:
        """Guard against overlapping task launches from multiple run buttons."""
        active = [name for name in ("calc", "plot", "gqi", "all") if name in self.workers]
        if active:
            QMessageBox.warning(
                self,
                requester_label,
                "Another task is already running.\n"
                "Stop the active task before starting a new one.",
            )
            return False
        return True

    # ──────────────────────────────── #
    # build “Run” tab                  #
    # ──────────────────────────────── #
    def _create_run_tab(self) -> QWidget:
        w = QWidget()
        self.run_root_widget = w
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 4, 10, 10)
        lay.setSpacing(10)

        inputs_box = QGroupBox("")
        inputs_box.setObjectName("mainSection")
        # Mark as embedded so the stylesheet suppresses the inner border
        # (avoids a double-line with the CollapsibleDetachableSection frame).
        inputs_box.setProperty("embeddedSection", "true")
        inputs_layout = QVBoxLayout(inputs_box)
        inputs_layout.setContentsMargins(4, 2, 4, 4)
        inputs_layout.setSpacing(4)

        global_row = QWidget()
        global_lay = QHBoxLayout(global_row)
        global_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_cfg_edit_global = QPushButton("Global settings...")
        self.btn_cfg_edit_global.clicked.connect(self._edit_global_settings)
        self.btn_cfg_reset = QPushButton("Reset to package defaults")
        self.btn_cfg_reset.clicked.connect(self._reset_runtime_settings_selection)
        global_lay.addWidget(self.btn_cfg_edit_global)
        global_lay.addWidget(self.btn_cfg_reset)
        global_lay.addStretch(1)
        inputs_layout.addWidget(global_row)

        dataset_input_row = QWidget()
        dataset_input_lay = QHBoxLayout(dataset_input_row)
        dataset_input_lay.setContentsMargins(0, 0, 0, 0)
        self.dataset_input = QLineEdit()
        self.dataset_input.setPlaceholderText("Add BIDS dataset path(s) — Browse supports multiple selection")
        btn_dataset_browse = QPushButton("Browse")
        btn_dataset_browse.clicked.connect(self._browse_dataset_input)
        btn_dataset_add = QPushButton("Add")
        btn_dataset_add.clicked.connect(self._add_dataset_from_input)
        dataset_input_lay.addWidget(self.dataset_input)
        dataset_input_lay.addWidget(btn_dataset_browse)
        dataset_input_lay.addWidget(btn_dataset_add)
        inputs_layout.addWidget(dataset_input_row)

        self.dataset_table = QTableWidget(0, 2)
        self.dataset_table.setHorizontalHeaderLabels(["Dataset path", "Settings"])
        self.dataset_table.setMinimumHeight(60)  # ensure at least 2 rows visible
        self.dataset_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.dataset_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.dataset_table.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        inputs_layout.addWidget(self.dataset_table)

        options_row = QWidget()
        options_lay = QHBoxLayout(options_row)
        options_lay.setContentsMargins(0, 0, 0, 0)
        self.chk_use_per_dataset_config = QCheckBox("Use per-dataset configuration")
        self.chk_use_per_dataset_config.toggled.connect(self._toggle_per_dataset_config)
        self.chk_use_per_dataset_config.setChecked(False)
        options_lay.addWidget(self.chk_use_per_dataset_config)
        options_lay.addStretch(1)
        inputs_layout.addWidget(options_row)

        # Analysis profile + runtime policy controls.
        # These settings are shared by calculation/plotting/GQI so users can
        # choose legacy output or profile-based runs from one place.
        profile_box = QGroupBox("Analysis profile + policies")
        profile_lay = QGridLayout(profile_box)

        self.cmb_analysis_mode = QComboBox()
        self.cmb_analysis_mode.addItems(["legacy", "new", "reuse", "latest"])
        self.cmb_analysis_mode.setCurrentText("new")

        self.edit_analysis_id = QLineEdit()
        self.edit_analysis_id.setPlaceholderText("Optional profile ID (required for mode=reuse)")
        self.btn_load_profiles = QPushButton("Load profiles...")
        self.btn_load_profiles.clicked.connect(self._load_available_profiles)
        self.btn_refresh_profiles = QPushButton("Refresh profiles")
        self.btn_refresh_profiles.clicked.connect(self._refresh_profiles_cache)

        self.cmb_existing_cfg_policy = QComboBox()
        self.cmb_existing_cfg_policy.addItems(["provided", "latest_saved", "fail"])
        self.cmb_existing_cfg_policy.setCurrentText("provided")

        self.cmb_processed_sub_policy = QComboBox()
        self.cmb_processed_sub_policy.addItems(["skip", "rerun", "fail"])
        self.cmb_processed_sub_policy.setCurrentText("skip")

        profile_lay.addWidget(QLabel("Mode:"), 0, 0)
        profile_lay.addWidget(self.cmb_analysis_mode, 0, 1)
        profile_lay.addWidget(QLabel("Profile ID:"), 0, 2)
        profile_lay.addWidget(self.edit_analysis_id, 0, 3)
        profile_lay.addWidget(self.btn_load_profiles, 0, 4)
        profile_lay.addWidget(self.btn_refresh_profiles, 0, 5)
        profile_lay.addWidget(QLabel("Config policy:"), 1, 0)
        profile_lay.addWidget(self.cmb_existing_cfg_policy, 1, 1)
        profile_lay.addWidget(QLabel("Processed subjects:"), 1, 2)
        profile_lay.addWidget(self.cmb_processed_sub_policy, 1, 3)
        profile_lay.setColumnStretch(3, 1)
        inputs_layout.addWidget(profile_box)

        dataset_btn_row = QWidget()
        dataset_btn_lay = QHBoxLayout(dataset_btn_row)
        dataset_btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_remove = QPushButton("Remove selected")
        btn_remove.clicked.connect(self._remove_selected_dataset)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_datasets)
        btn_up = QPushButton("Move up")
        btn_up.clicked.connect(lambda: self._move_selected_dataset(-1))
        btn_down = QPushButton("Move down")
        btn_down.clicked.connect(lambda: self._move_selected_dataset(1))
        dataset_btn_lay.addWidget(btn_remove)
        dataset_btn_lay.addWidget(btn_clear)
        dataset_btn_lay.addWidget(btn_up)
        dataset_btn_lay.addWidget(btn_down)
        inputs_layout.addWidget(dataset_btn_row)

        self.section_inputs = CollapsibleDetachableSection("Inputs", inputs_box, parent=w)
        lay.addWidget(self.section_inputs, stretch=3)

        output_box = QGroupBox("")
        output_box.setObjectName("mainSection")
        output_form = QFormLayout(output_box)
        self.derivatives_dir = QLineEdit()
        derivatives_browse = QPushButton("Browse")
        derivatives_browse.clicked.connect(lambda: self._browse(self.derivatives_dir))
        deriv_row = QWidget()
        deriv_lay = QHBoxLayout(deriv_row)
        deriv_lay.setContentsMargins(0, 0, 0, 0)
        deriv_lay.addWidget(self.derivatives_dir)
        deriv_lay.addWidget(derivatives_browse)
        output_form.addRow("Derivatives output:", deriv_row)
        self.section_output = CollapsibleDetachableSection("Output folder (optional)", output_box, parent=w)
        lay.addWidget(self.section_output)

        # — Calculation section —
        # Tab label already states "QA/QC calculation"; keep inner panel untitled
        # to avoid repeated heading text and accent separator.
        calc_box = QGroupBox("")
        calc_form = QFormLayout(calc_box)

        self.jobs = QSpinBox()
        self.jobs.setRange(-1, os.cpu_count() or 1)
        self.jobs.setValue(-1)
        self.jobs.valueChanged.connect(lambda _: self._refresh_calc_jobs_table())
        btn_info = QPushButton("Info")
        btn_info.setToolTip("Parallel jobs info")

        def show_jobs_info():
            info = """
            Number of parallel jobs to use during
            processing.
            Default is 1. Use -1 to utilize all
            available CPU cores.

            ⚠️ Recommendation based on system
            memory:

            - 8 GB → up to 1 job
            - 16 GB → up to 2 jobs
            - 32 GB → up to 6 jobs
            - 64 GB → up to 16 jobs
            - 128 GB → up to 30 jobs

            Using -1 will use all cores. Optimal RAM ≳ 3.5×
            #cores.
                    """
            QMessageBox.information(self, "Jobs Recommendation", info)

        btn_info.clicked.connect(show_jobs_info)
        row_jobs = QWidget()
        jobs_lay = QHBoxLayout(row_jobs)
        jobs_lay.setContentsMargins(0, 0, 0, 0)
        jobs_lay.addWidget(self.jobs)
        jobs_lay.addWidget(btn_info)
        calc_form.addRow("Calculation n_jobs:", row_jobs)

        self.chk_calc_per_dataset_jobs = QCheckBox("Use per-dataset n_jobs override")
        self.chk_calc_per_dataset_jobs.toggled.connect(self._toggle_calc_jobs_table)
        calc_form.addRow(self.chk_calc_per_dataset_jobs)

        self.calc_jobs_table = QTableWidget(0, 4)
        self.calc_jobs_table.setHorizontalHeaderLabels(["Dataset", "Calc n_jobs", "Subjects", "Scan/select"])
        self.calc_jobs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.calc_jobs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.calc_jobs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.calc_jobs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        calc_form.addRow(self.calc_jobs_table)

        self.btn_calc_run = QPushButton("Run Calculation")
        self.btn_calc_run.clicked.connect(self.start_calc)
        self.btn_calc_stop = QPushButton("Stop Calculation")
        self.btn_calc_stop.clicked.connect(self.stop_calc)
        row_btns = QWidget()
        btns_lay = QHBoxLayout(row_btns)
        btns_lay.setContentsMargins(0, 0, 0, 0)
        btns_lay.addWidget(self.btn_calc_run)
        btns_lay.addWidget(self.btn_calc_stop)
        calc_form.addRow(row_btns)

        self.btn_gqi_run = QPushButton("Run GQI")
        self.btn_gqi_run.clicked.connect(self.start_gqi)
        self.btn_gqi_stop = QPushButton("Stop GQI")
        self.btn_gqi_stop.clicked.connect(self.stop_gqi)
        row_gqi = QWidget()
        gqi_lay = QHBoxLayout(row_gqi)
        gqi_lay.setContentsMargins(0, 0, 0, 0)
        gqi_lay.addWidget(self.btn_gqi_run)
        gqi_lay.addWidget(self.btn_gqi_stop)
        calc_form.addRow(row_gqi)

        # — Plotting section —
        # Tab label already states "QA/QC plotting"; keep inner panel untitled
        # to avoid repeated heading text and accent separator.
        plot_box = QGroupBox("")
        plot_form = QFormLayout(plot_box)

        self.chk_qa_subject = QCheckBox("QA subject reports")
        self.chk_qa_group = QCheckBox("QA group reports")
        self.chk_qa_multisample = QCheckBox("QA multisample report")
        self.chk_qc_group = QCheckBox("QC group report(s)")
        self.chk_qc_multisample = QCheckBox("QC multisample report")
        self.chk_qa_subject.setChecked(True)

        mode_row = QWidget()
        mode_lay = QVBoxLayout(mode_row)
        mode_lay.setContentsMargins(0, 0, 0, 0)
        mode_lay.addWidget(self.chk_qa_subject)
        mode_lay.addWidget(self.chk_qa_group)
        mode_lay.addWidget(self.chk_qa_multisample)
        mode_lay.addWidget(self.chk_qc_group)
        mode_lay.addWidget(self.chk_qc_multisample)
        plot_form.addRow("Modes:", mode_row)

        preset_row = QWidget()
        preset_lay = QHBoxLayout(preset_row)
        preset_lay.setContentsMargins(0, 0, 0, 0)
        btn_qa_all = QPushButton("QA all")
        btn_qa_all.clicked.connect(self._set_plot_preset_qa)
        btn_qc_all = QPushButton("QC all")
        btn_qc_all.clicked.connect(self._set_plot_preset_qc)
        btn_all = QPushButton("All")
        btn_all.clicked.connect(self._set_plot_preset_all)
        btn_clear_modes = QPushButton("Clear modes")
        btn_clear_modes.clicked.connect(self._clear_plot_modes)
        preset_lay.addWidget(btn_qa_all)
        preset_lay.addWidget(btn_qc_all)
        preset_lay.addWidget(btn_all)
        preset_lay.addWidget(btn_clear_modes)
        plot_form.addRow("", preset_row)

        self.plot_attempt = QSpinBox()
        self.plot_attempt.setRange(0, 9999)
        self.plot_attempt.setValue(0)
        self.plot_attempt.setToolTip("0 means auto/latest attempt")
        plot_form.addRow("QC attempt (0=auto):", self.plot_attempt)

        self.plot_jobs = QSpinBox()
        self.plot_jobs.setRange(-1, os.cpu_count() or 1)
        self.plot_jobs.setValue(self.jobs.value())
        plot_form.addRow("Plotting n_jobs:", self.plot_jobs)

        self.plot_input_tsv = QLineEdit()
        self.plot_input_tsv.setPlaceholderText("Optional: Global_Quality_Index_attempt_*.tsv")
        btn_input_tsv = QPushButton("Browse")
        btn_input_tsv.clicked.connect(self._browse_plot_input_tsv)
        tsv_row = QWidget()
        tsv_lay = QHBoxLayout(tsv_row)
        tsv_lay.setContentsMargins(0, 0, 0, 0)
        tsv_lay.addWidget(self.plot_input_tsv)
        tsv_lay.addWidget(btn_input_tsv)
        plot_form.addRow("QC input TSV:", tsv_row)

        self.plot_output_report = QLineEdit()
        self.plot_output_report.setPlaceholderText("Optional: explicit output HTML path")
        btn_output_report = QPushButton("Browse")
        btn_output_report.clicked.connect(self._browse_plot_output_report)
        out_row = QWidget()
        out_lay = QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.addWidget(self.plot_output_report)
        out_lay.addWidget(btn_output_report)
        plot_form.addRow("Output report:", out_row)

        for chk in [
            self.chk_qa_subject,
            self.chk_qa_group,
            self.chk_qa_multisample,
            self.chk_qc_group,
            self.chk_qc_multisample,
        ]:
            chk.toggled.connect(self._update_plot_form_state)

        self.btn_plot_run = QPushButton("Run Plotting")
        self.btn_plot_run.clicked.connect(self.start_plot)
        self.btn_plot_stop = QPushButton("Stop Plotting")
        self.btn_plot_stop.clicked.connect(self.stop_plot)
        prow2 = QWidget()
        pl2 = QHBoxLayout(prow2)
        pl2.setContentsMargins(0, 0, 0, 0)
        pl2.addWidget(self.btn_plot_run)
        pl2.addWidget(self.btn_plot_stop)
        plot_form.addRow("", prow2)

        mode_tabs = QTabWidget()
        calc_tab = QWidget()
        calc_tab_lay = QVBoxLayout(calc_tab)
        calc_tab_lay.setContentsMargins(0, 0, 0, 0)
        calc_tab_lay.addWidget(calc_box)
        calc_scroll = QScrollArea()
        calc_scroll.setWidgetResizable(True)
        calc_scroll.setWidget(calc_tab)
        mode_tabs.addTab(calc_scroll, "QA/QC calculation")

        plot_tab = QWidget()
        plot_tab_lay = QVBoxLayout(plot_tab)
        plot_tab_lay.setContentsMargins(0, 0, 0, 0)
        plot_tab_lay.addWidget(plot_box)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(plot_tab)
        mode_tabs.addTab(plot_scroll, "QA/QC plotting")

        # Full execution area = calculation + plotting tabs + run-all controls.
        # This is the section users requested to treat as one "Pipeline orchestration" block.
        orchestration_box = QGroupBox("")
        orchestration_box.setObjectName("mainSection")
        orchestration_lay = QVBoxLayout(orchestration_box)
        orchestration_lay.setContentsMargins(8, 8, 8, 8)
        orchestration_lay.setSpacing(8)
        orchestration_lay.addWidget(mode_tabs)

        all_row = QWidget()
        all_box_lay = QHBoxLayout(all_row)
        all_box_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_all_run = QPushButton("Run ALL (calc + plotting)")
        self.btn_all_run.clicked.connect(self.start_all)
        self.btn_all_stop = QPushButton("Stop ALL")
        self.btn_all_stop.clicked.connect(self.stop_all)
        all_box_lay.addStretch(1)
        all_box_lay.addWidget(self.btn_all_run)
        all_box_lay.addWidget(self.btn_all_stop)
        all_box_lay.addStretch(1)
        orchestration_lay.addWidget(all_row)

        self.section_orchestration = CollapsibleDetachableSection(
            "Pipeline orchestration",
            orchestration_box,
            parent=w,
        )
        lay.addWidget(self.section_orchestration)

        # — Log output —
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        # Header row: "Log:" label + action buttons side by side
        log_header = QWidget()
        log_header_lay = QHBoxLayout(log_header)
        log_header_lay.setContentsMargins(0, 0, 0, 2)
        log_header_lay.setSpacing(6)
        log_header_lay.addWidget(QLabel("Log:"))
        log_header_lay.addStretch(1)
        self.btn_live_output = QPushButton("Live terminal output")
        self.btn_live_output.setToolTip(
            "Open a floating window that streams real-time stdout/stderr "
            "from background pipeline tasks."
        )
        self.btn_live_output.clicked.connect(self._open_live_terminal)
        log_header_lay.addWidget(self.btn_live_output)
        self.btn_open_cli = QPushButton("Open CLI")
        self.btn_open_cli.setToolTip(
            "Open a system terminal with the MEGqc Python environment activated,\n"
            "ready to run CLI commands like: run-megqc --inputdata /path/to/dataset"
        )
        self.btn_open_cli.clicked.connect(self._open_cli_terminal)
        log_header_lay.addWidget(self.btn_open_cli)
        lay.addWidget(log_header)
        lay.addWidget(self.log, stretch=1)
        lay.addSpacing(10)

        bottom_row = QWidget()
        bottom_lay = QHBoxLayout(bottom_row)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(10)
        self.lbl_version = QLabel(f"MEGqc v{self.installed_megqc_version}")
        self.lbl_version.setToolTip("Installed package version.")
        self.btn_check_updates = QPushButton("Check updates")
        self.btn_check_updates.setToolTip("Check PyPI for newer MEGqc versions and optionally update.")
        self.btn_check_updates.clicked.connect(self._check_for_updates)
        bottom_lay.addWidget(self.lbl_version)
        bottom_lay.addWidget(self.btn_check_updates)
        bottom_lay.addStretch(1)
        self.btn_qc_viewer = QPushButton("QC Viewer")
        self.btn_qc_viewer.setToolTip("Open the integrated QC data & report viewer.")
        self.btn_qc_viewer.clicked.connect(self._open_qc_viewer)
        bottom_lay.addWidget(self.btn_qc_viewer)
        self.btn_gui_settings = QPushButton("GUI settings")
        self.btn_gui_settings.clicked.connect(self._open_gui_settings_dialog)
        bottom_lay.addWidget(self.btn_gui_settings)


        bottom_lay.addWidget(self.spinner_label)
        bottom_lay.addWidget(self.elapsed_label)
        lay.addWidget(bottom_row)

        self._refresh_calc_jobs_table()
        self._update_config_status()
        self._update_plot_form_state()
        self._apply_section_highlight_style()
        self._set_active_run_task(None)
        return w

    # ──────────────────────────────── #
    # helper: browse directory         #
    # ──────────────────────────────── #
    def _browse(self, edit: QLineEdit):
        options = (
                QFileDialog.Option.ShowDirsOnly
                | QFileDialog.Option.DontUseNativeDialog
        )
        start_dir = edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            start_dir,
            options
        )
        if path:
            edit.setText(path)

    def _browse_dataset_input(self):
        """Open a folder dialog that allows selecting multiple directories at once."""
        dlg = QFileDialog(self, "Select one or more BIDS dataset folders")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        start_dir = self.dataset_input.text().strip()
        if start_dir and os.path.isdir(start_dir):
            dlg.setDirectory(start_dir)
        if dlg.exec():
            selected = dlg.selectedFiles()
            dirs = [p for p in selected if os.path.isdir(p)]
            for d in dirs:
                self._add_dataset_path(d)
            self.dataset_input.clear()

    def _browse_plot_input_tsv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GQI TSV",
            self.plot_input_tsv.text().strip() or "",
            "TSV files (*.tsv);;All files (*)",
        )
        if path:
            self.plot_input_tsv.setText(path)

    def _browse_plot_output_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output HTML",
            self.plot_output_report.text().strip() or "",
            "HTML files (*.html);;All files (*)",
        )
        if path:
            self.plot_output_report.setText(path)

    def _add_dataset_from_input(self):
        path = self.dataset_input.text().strip()
        if not path:
            return
        self._add_dataset_path(path)
        self.dataset_input.clear()

    def _populate_dataset_table(self, paths: List[str], selected_row: Optional[int] = None) -> None:
        self.dataset_table.setRowCount(0)
        for ds in paths:
            row = self.dataset_table.rowCount()
            self.dataset_table.insertRow(row)
            item = QTableWidgetItem(ds)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.dataset_table.setItem(row, 0, item)
            cfg_btn = QPushButton("Settings...")
            cfg_btn.setProperty("dataset_path", ds)
            cfg_btn.clicked.connect(self._edit_dataset_settings_from_button)
            self.dataset_table.setCellWidget(row, 1, cfg_btn)
        if selected_row is not None and paths:
            selected_row = max(0, min(selected_row, len(paths) - 1))
            self.dataset_table.selectRow(selected_row)
        self._refresh_dataset_settings_button_state()

    def _add_dataset_path(self, dataset_path: str):
        normalized = os.path.normpath(dataset_path)
        existing = self._collect_dataset_paths()
        if normalized in existing:
            return
        existing.append(normalized)
        self._populate_dataset_table(existing, selected_row=len(existing) - 1)
        self._refresh_calc_jobs_table()
        self._update_config_status()
        self._update_plot_form_state()

    def _remove_selected_dataset(self):
        selected = self.dataset_table.currentRow()
        if selected < 0:
            return
        paths = self._collect_dataset_paths()
        removed_path = paths[selected]
        del paths[selected]
        self._populate_dataset_table(paths, selected_row=min(selected, len(paths) - 1))
        if removed_path:
            self.dataset_config_paths.pop(removed_path, None)
        self._refresh_calc_jobs_table()
        self._update_config_status()
        self._update_plot_form_state()

    def _clear_datasets(self):
        self._populate_dataset_table([])
        self.dataset_config_paths.clear()
        self._refresh_calc_jobs_table()
        self._update_config_status()
        self._update_plot_form_state()

    def _move_selected_dataset(self, offset: int):
        row = self.dataset_table.currentRow()
        if row < 0:
            return
        paths = self._collect_dataset_paths()
        new_row = row + offset
        if new_row < 0 or new_row >= len(paths):
            return
        paths[row], paths[new_row] = paths[new_row], paths[row]
        self._populate_dataset_table(paths, selected_row=new_row)
        self._refresh_calc_jobs_table()
        self._update_config_status()

    def _on_dataset_selection_changed(self):
        self._update_config_status()
        self._update_plot_form_state()

    def _collect_dataset_paths(self) -> List[str]:
        paths: List[str] = []
        if not hasattr(self, "dataset_table"):
            return paths
        for i in range(self.dataset_table.rowCount()):
            item = self.dataset_table.item(i, 0)
            if item is not None and item.text().strip():
                paths.append(os.path.normpath(item.text().strip()))
        return paths

    def _selected_dataset_path(self) -> Optional[str]:
        row = self.dataset_table.currentRow()
        if row < 0:
            return None
        item = self.dataset_table.item(row, 0)
        if item is None:
            return None
        return os.path.normpath(item.text().strip())

    def _runtime_global_settings_path(self) -> Path:
        return _runtime_config_dir() / "settings_global.ini"

    def _runtime_dataset_settings_path(self, dataset_path: str) -> Path:
        normalized = os.path.normpath(dataset_path)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        base = Path(normalized).name.replace(" ", "_")
        return _runtime_config_dir() / f"settings_{base}_{digest}.ini"

    def _ensure_runtime_settings_copy(self, target_path: Path) -> Path:
        if not target_path.exists():
            shutil.copy(self.default_settings_path, target_path)
        return target_path

    def _open_settings_dialog(self, ini_path: Path, title: str) -> bool:
        self._ensure_runtime_settings_copy(ini_path)
        dialog = SettingsEditorDialog(
            source_path=ini_path,
            defaults_path=self.default_settings_path,
            title=title,
            parent=self,
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _edit_global_settings(self):
        runtime_path = self._runtime_global_settings_path()
        if self._open_settings_dialog(runtime_path, "Global settings profile"):
            self.global_config_path = str(runtime_path)
            self._log(f"Global config active: {runtime_path}")
            self._update_config_status()

    def _edit_selected_dataset_settings(self):
        selected = self._selected_dataset_path()
        if not selected:
            QMessageBox.warning(self, "Settings", "Select one dataset first.")
            return
        runtime_path = self._runtime_dataset_settings_path(selected)
        if self._open_settings_dialog(runtime_path, f"Settings profile: {Path(selected).name}"):
            self.dataset_config_paths[selected] = str(runtime_path)
            self._log(f"Dataset override active: {selected} -> {runtime_path}")
            self._update_config_status()

    def _edit_dataset_settings_from_button(self):
        sender = self.sender()
        if sender is None:
            return
        ds = sender.property("dataset_path")
        if not ds:
            return
        if not self.chk_use_per_dataset_config.isChecked():
            return
        selected = os.path.normpath(str(ds))
        runtime_path = self._runtime_dataset_settings_path(selected)
        if self._open_settings_dialog(runtime_path, f"Settings profile: {Path(selected).name}"):
            self.dataset_config_paths[selected] = str(runtime_path)
            self._log(f"Dataset override active: {selected} -> {runtime_path}")
            self._update_config_status()

    def _refresh_dataset_settings_button_state(self):
        enabled = bool(getattr(self, "chk_use_per_dataset_config", None) and self.chk_use_per_dataset_config.isChecked())
        if not hasattr(self, "dataset_table"):
            return
        for row in range(self.dataset_table.rowCount()):
            btn = self.dataset_table.cellWidget(row, 1)
            if isinstance(btn, QPushButton):
                btn.setEnabled(enabled)

    def _toggle_per_dataset_config(self, checked: bool):
        self._refresh_dataset_settings_button_state()
        self._update_config_status()

    def _clear_selected_dataset_config(self):
        selected = self._selected_dataset_path()
        if not selected:
            QMessageBox.warning(self, "Settings", "Select one dataset first.")
            return
        if selected in self.dataset_config_paths:
            self.dataset_config_paths.pop(selected, None)
            self._log(f"Removed dataset override: {selected}")
            self._update_config_status()

    def _reset_runtime_settings_selection(self):
        self.global_config_path = None
        self.dataset_config_paths.clear()
        self._log("Config selection reset to package defaults.")
        self._update_config_status()

    def _collect_dataset_config_overrides(self) -> Optional[Dict[str, str]]:
        if not hasattr(self, "chk_use_per_dataset_config") or not self.chk_use_per_dataset_config.isChecked():
            return None
        if not self.dataset_config_paths:
            return None
        active = {os.path.normpath(p) for p in self._collect_dataset_paths()}
        valid = {
            os.path.normpath(ds): cfg
            for ds, cfg in self.dataset_config_paths.items()
            if os.path.normpath(ds) in active
        }
        return valid or None

    def _update_config_status(self):
        datasets = {os.path.normpath(p) for p in self._collect_dataset_paths()}
        self.dataset_config_paths = {
            os.path.normpath(ds): cfg
            for ds, cfg in self.dataset_config_paths.items()
            if os.path.normpath(ds) in datasets
        }
        self._refresh_dataset_settings_button_state()

    def _collect_analysis_profile_settings(self) -> Tuple[str, Optional[str], str, str]:
        mode = self.cmb_analysis_mode.currentText().strip().lower()
        analysis_id = self.edit_analysis_id.text().strip() or None
        cfg_policy = self.cmb_existing_cfg_policy.currentText().strip().lower()
        sub_policy = self.cmb_processed_sub_policy.currentText().strip().lower()
        return mode, analysis_id, cfg_policy, sub_policy

    @staticmethod
    def _generate_shared_analysis_id() -> str:
        return f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"

    def _validate_analysis_selection(self, mode, analysis_id, cfg_policy=None, sub_policy=None) -> bool:
        if mode == "reuse" and not analysis_id:
            QMessageBox.warning(self, "Analysis profile",
                "analysis_mode='reuse' requires a profile ID.\nSet a profile ID or use 'Load profiles...'.")
            return False
        return True

    def _discover_profiles_for_datasets(self) -> Tuple[Dict[str, List[str]], List[str]]:
        datasets = self._collect_dataset_paths()
        derivatives_base = self.derivatives_dir.text().strip() or None
        profile_map: Dict[str, List[str]] = {}
        for ds in datasets:
            try:
                profile_map[ds] = list_analysis_profiles(dataset_path=ds, external_derivatives_root=derivatives_base)
            except Exception:
                profile_map[ds] = []
        common: List[str] = []
        if profile_map:
            common_set = set(next(iter(profile_map.values())))
            for values in profile_map.values():
                common_set &= set(values)
            common = sorted(common_set)
        return profile_map, common

    def _refresh_profiles_cache(self):
        datasets = self._collect_dataset_paths()
        if not datasets:
            QMessageBox.warning(self, "Profiles", "Add at least one dataset first.")
            return
        profile_map, common = self._discover_profiles_for_datasets()
        summary = ", ".join(f"{Path(ds).name}:{len(vals)}" for ds, vals in profile_map.items())
        self._log(f"Profile scan completed ({summary}); common={len(common)}.")
        if len(datasets) > 1 and not common:
            QMessageBox.information(self, "Profiles",
                "No common profile ID across all selected datasets.\nUse legacy mode or create/reuse a shared analysis_id.")

    def _load_available_profiles(self):
        datasets = self._collect_dataset_paths()
        if not datasets:
            QMessageBox.warning(self, "Profiles", "Add at least one dataset first.")
            return
        selected_ds = self._selected_dataset_path() or datasets[0]
        profile_map, common = self._discover_profiles_for_datasets()
        if len(datasets) > 1:
            options = common
            label = "Available common profiles across selected datasets:"
        else:
            options = profile_map.get(selected_ds, [])
            label = f"Available profiles for {Path(selected_ds).name}:"
        if not options:
            QMessageBox.information(self, "Profiles", "No compatible profiles found for current selection.")
            return
        picked, ok = QInputDialog.getItem(self, "Select profile", label, options, 0, False)
        if ok and picked:
            self.edit_analysis_id.setText(str(picked))
            self.cmb_analysis_mode.setCurrentText("reuse")
            self._log(f"Selected profile: {picked}")

    def _validate_multisample_profile_compatibility(self, mode, analysis_id, *, qa_multisample, qc_multisample, require_existing_profiles=True) -> bool:
        if not (qa_multisample or qc_multisample):
            return True
        if mode == "legacy":
            return True
        if mode not in {"reuse", "new"}:
            QMessageBox.warning(self, "Profile compatibility",
                "Multisample plotting requires a shared profile strategy.\nUse mode='reuse'/'new' or legacy mode.")
            return False
        if not analysis_id:
            QMessageBox.warning(self, "Profile compatibility",
                "Multisample plotting requires a profile ID for mode='reuse'/'new'.")
            return False
        if mode == "new" and not require_existing_profiles:
            return True
        profile_map, _common = self._discover_profiles_for_datasets()
        missing = [Path(ds).name for ds, vals in profile_map.items() if analysis_id not in set(vals)]
        if missing:
            QMessageBox.warning(self, "Profile compatibility",
                "Selected profile is missing for:\n- " + "\n- ".join(missing))
            return False
        return True

    def _toggle_calc_jobs_table(self, checked: bool):
        self._refresh_calc_jobs_table()

    def _refresh_calc_jobs_table(self):
        if not hasattr(self, "calc_jobs_table"):
            return
        datasets = self._collect_dataset_paths()
        current_values: Dict[str, int] = {}
        current_subjects: Dict[str, str] = {}
        for row in range(self.calc_jobs_table.rowCount()):
            item = self.calc_jobs_table.item(row, 0)
            spin = self.calc_jobs_table.cellWidget(row, 1)
            subs_widget = self.calc_jobs_table.cellWidget(row, 2)
            if item is not None and spin is not None:
                current_values[item.text()] = int(spin.value())
            if item is not None and isinstance(subs_widget, QLineEdit):
                current_subjects[item.text()] = subs_widget.text().strip() or "all"
        self.calc_jobs_table.setRowCount(len(datasets))
        max_jobs = os.cpu_count() or 1
        allow_njobs_override = self.chk_calc_per_dataset_jobs.isChecked()
        for row, ds in enumerate(datasets):
            ds_item = QTableWidgetItem(ds)
            ds_item.setFlags(ds_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.calc_jobs_table.setItem(row, 0, ds_item)
            spin = QSpinBox()
            spin.setRange(-1, max_jobs)
            spin.setValue(current_values.get(ds, self.jobs.value()))
            spin.setEnabled(allow_njobs_override)
            self.calc_jobs_table.setCellWidget(row, 1, spin)
            subs_edit = QLineEdit()
            subs_edit.setPlaceholderText("all or comma-separated IDs")
            subs_edit.setText(current_subjects.get(ds, "all"))
            self.calc_jobs_table.setCellWidget(row, 2, subs_edit)
            scan_btn = QPushButton("Scan...")
            scan_btn.setProperty("dataset_path", ds)
            scan_btn.clicked.connect(self._open_subject_selector_for_row)
            self.calc_jobs_table.setCellWidget(row, 3, scan_btn)

    def _scan_dataset_subjects(self, dataset_path: str) -> List[str]:
        import ancpbids
        from ancpbids import DatasetOptions
        dataset = ancpbids.load_dataset(dataset_path, DatasetOptions(lazy_loading=True))
        entities = dataset.query_entities(scope="raw")
        subjects = sorted(list(entities.get("subject", [])))
        return [str(s) for s in subjects if str(s).strip()]

    def _open_subject_selector_for_row(self):
        sender = self.sender()
        if sender is None:
            return
        ds = sender.property("dataset_path")
        if not ds:
            return
        dataset_path = os.path.normpath(str(ds))
        row_idx = None
        for row in range(self.calc_jobs_table.rowCount()):
            item = self.calc_jobs_table.item(row, 0)
            if item is not None and os.path.normpath(item.text()) == dataset_path:
                row_idx = row
                break
        if row_idx is None:
            return
        subs_widget = self.calc_jobs_table.cellWidget(row_idx, 2)
        if not isinstance(subs_widget, QLineEdit):
            return
        current_raw = subs_widget.text().strip()
        if not current_raw or current_raw.lower() == "all":
            current_value: Union[str, List[str]] = "all"
        else:
            current_value = [s.strip() for s in current_raw.split(",") if s.strip()]
        try:
            subjects = self._scan_dataset_subjects(dataset_path)
        except Exception as exc:
            QMessageBox.warning(self, "Subject scan", f"Failed to scan dataset subjects:\n{exc}")
            return
        if not subjects:
            QMessageBox.information(self, "Subject scan", "No subjects found in this dataset.")
            return
        dialog = SubjectSelectionDialog(Path(dataset_path).name, subjects, current_value, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_value()
        if selected == "all":
            subs_widget.setText("all")
        else:
            subs_widget.setText(",".join(selected))

    def _collect_calc_njobs_overrides(self) -> Optional[Dict[str, int]]:
        if not self.chk_calc_per_dataset_jobs.isChecked():
            return None
        overrides: Dict[str, int] = {}
        for row in range(self.calc_jobs_table.rowCount()):
            item = self.calc_jobs_table.item(row, 0)
            spin = self.calc_jobs_table.cellWidget(row, 1)
            if item is None or spin is None:
                continue
            overrides[item.text()] = int(spin.value())
        return overrides or None

    def _collect_dataset_sub_overrides(self) -> Optional[Dict[str, Union[str, List[str]]]]:
        overrides: Dict[str, Union[str, List[str]]] = {}
        for row in range(self.calc_jobs_table.rowCount()):
            item = self.calc_jobs_table.item(row, 0)
            subs_widget = self.calc_jobs_table.cellWidget(row, 2)
            if item is None or not isinstance(subs_widget, QLineEdit):
                continue
            raw = subs_widget.text().strip()
            if not raw or raw.lower() == "all":
                overrides[item.text()] = "all"
                continue
            parsed = [s.strip() for s in raw.split(",") if s.strip()]
            overrides[item.text()] = parsed if parsed else "all"
        return overrides or None

    def _set_plot_modes(self, *, qa_subject, qa_group, qa_multisample, qc_group, qc_multisample):
        self.chk_qa_subject.setChecked(qa_subject)
        self.chk_qa_group.setChecked(qa_group)
        self.chk_qa_multisample.setChecked(qa_multisample)
        self.chk_qc_group.setChecked(qc_group)
        self.chk_qc_multisample.setChecked(qc_multisample)
        self._update_plot_form_state()

    def _set_plot_preset_qa(self):
        self._set_plot_modes(qa_subject=True, qa_group=True, qa_multisample=True, qc_group=False, qc_multisample=False)

    def _set_plot_preset_qc(self):
        self._set_plot_modes(qa_subject=False, qa_group=False, qa_multisample=False, qc_group=True, qc_multisample=True)

    def _set_plot_preset_all(self):
        self._set_plot_modes(qa_subject=True, qa_group=True, qa_multisample=True, qc_group=True, qc_multisample=True)

    def _clear_plot_modes(self):
        self._set_plot_modes(qa_subject=False, qa_group=False, qa_multisample=False, qc_group=False, qc_multisample=False)

    def _update_plot_form_state(self):
        dataset_count = len(self._collect_dataset_paths())
        allow_multisample = dataset_count >= 2
        for chk in [self.chk_qa_multisample, self.chk_qc_multisample]:
            if not allow_multisample and chk.isChecked():
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)
            chk.setEnabled(allow_multisample)
        qa_subject_selected = self.chk_qa_subject.isChecked()
        qa_group_selected = self.chk_qa_group.isChecked()
        qa_multisample_selected = self.chk_qa_multisample.isChecked()
        qc_group_selected = self.chk_qc_group.isChecked()
        qc_multisample_selected = self.chk_qc_multisample.isChecked()
        input_tsv_allowed = qc_group_selected and dataset_count == 1 and not qc_multisample_selected
        self.plot_input_tsv.setEnabled(input_tsv_allowed)
        if not input_tsv_allowed:
            self.plot_input_tsv.clear()
        attempt_allowed = qc_group_selected or qc_multisample_selected
        self.plot_attempt.setEnabled(attempt_allowed)
        if not attempt_allowed:
            self.plot_attempt.setValue(0)
        mode_count = sum(1 for s in [qa_subject_selected, qa_group_selected, qa_multisample_selected, qc_group_selected, qc_multisample_selected] if s)
        output_allowed = mode_count == 1 and (qa_multisample_selected or qc_multisample_selected or (qc_group_selected and dataset_count == 1))
        self.plot_output_report.setEnabled(output_allowed)
        if not output_allowed:
            self.plot_output_report.clear()

    # ──────────────────────────────── #
    # start / stop handlers            #
    # ──────────────────────────────── #
    def start_calc(self):
        if not self._ensure_no_active_pipeline_task("Calculation"):
            return
        dataset_paths = self._collect_dataset_paths()
        if not dataset_paths:
            QMessageBox.warning(self, "Calculation", "Please add at least one dataset.")
            return
        derivatives_dir = self.derivatives_dir.text().strip() or None
        subs = "all"
        dataset_subs = self._collect_dataset_sub_overrides()
        global_n_jobs = self.jobs.value()
        dataset_njobs = self._collect_calc_njobs_overrides()
        dataset_cfg_overrides = self._collect_dataset_config_overrides()
        analysis_mode, analysis_id, cfg_policy, sub_policy = self._collect_analysis_profile_settings()
        if not self._validate_analysis_selection(analysis_mode, analysis_id, cfg_policy=cfg_policy, sub_policy=sub_policy):
            return
        args = (
            dataset_paths, str(self.default_settings_path), str(INTERNAL_PATH),
            subs, global_n_jobs, derivatives_dir, dataset_njobs, dataset_subs,
            self.global_config_path, dataset_cfg_overrides,
            analysis_mode, analysis_id, cfg_policy, sub_policy, False, False,
        )
        worker = Worker(run_calculation_dispatch, *args)
        self._set_active_run_task("calc")

        def on_started():
            self._start_task_timer("calc")
            cfg_note = f"global={self.global_config_path or 'default'}"
            self._log(f"Calculation started for {len(dataset_paths)} dataset(s) ({cfg_note}, analysis={analysis_mode}:{analysis_id or 'legacy'}).")

        def on_finished():
            elapsed = self._stop_task_timer("calc")
            self._log(f"Calculation finished in {elapsed:.2f}s.")
            self.workers.pop("calc", None)
            self._reset_run_controls_if_idle()

        def on_error(err: str):
            elapsed = self._stop_task_timer("calc")
            self._log(f"Calculation error after {elapsed:.2f}s: {err}")
            self.workers.pop("calc", None)
            self._reset_run_controls_if_idle()

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self.workers["calc"] = worker
        try:
            worker.start()
            self._wire_worker_output(worker)
        except Exception as exc:
            self._stop_task_timer("calc")
            self._log(f"Calculation error: {exc}")
            self.workers.pop("calc", None)
            self._reset_run_controls_if_idle()

    def stop_calc(self):
        worker = self.workers.get("calc")
        if worker:
            worker.stop()
            elapsed = self._stop_task_timer("calc")
            self._log(f"Calculation stopped after {elapsed:.2f}s.")
            self.workers.pop("calc", None)
            self._reset_run_controls_if_idle()

    def start_plot(self):
        if not self._ensure_no_active_pipeline_task("Plotting"):
            return
        dataset_paths = self._collect_dataset_paths()
        if not dataset_paths:
            QMessageBox.warning(self, "Plotting", "Please add at least one dataset.")
            return
        derivatives_dir = self.derivatives_dir.text().strip() or None
        n_jobs = self.plot_jobs.value()
        attempt_raw = self.plot_attempt.value()
        attempt = attempt_raw if attempt_raw > 0 else None
        input_tsv = self.plot_input_tsv.text().strip() or None
        output_report = self.plot_output_report.text().strip() or None
        qa_subject = self.chk_qa_subject.isChecked()
        qa_group = self.chk_qa_group.isChecked()
        qa_multisample = self.chk_qa_multisample.isChecked()
        qc_group = self.chk_qc_group.isChecked()
        qc_multisample = self.chk_qc_multisample.isChecked()
        analysis_mode, analysis_id, _cfg_policy, _sub_policy = self._collect_analysis_profile_settings()
        if not self._validate_analysis_selection(analysis_mode, analysis_id, cfg_policy=_cfg_policy, sub_policy=_sub_policy):
            return
        if analysis_mode == "new" and not analysis_id:
            QMessageBox.warning(self, "Plotting", "Mode 'new' requires a profile ID for plotting.\nUse mode 'reuse' with a profile ID, or 'latest'/'legacy'.")
            return
        if not self._validate_multisample_profile_compatibility(analysis_mode, analysis_id, qa_multisample=qa_multisample, qc_multisample=qc_multisample):
            return
        check_qa_subject = qa_subject
        if not any([qa_subject, qa_group, qa_multisample, qc_group, qc_multisample]):
            check_qa_subject = True
        try:
            validate_plot_request(dataset_paths=dataset_paths, qa_subject=check_qa_subject, qa_group=qa_group,
                                  qa_multisample=qa_multisample, qc_group=qc_group, qc_multisample=qc_multisample,
                                  input_tsv=input_tsv, output_report=output_report)
        except ValueError as exc:
            QMessageBox.warning(self, "Plotting", str(exc))
            return
        args = (
            dataset_paths, derivatives_dir, output_report, attempt, input_tsv, n_jobs,
            qa_subject, qa_group, qa_multisample, qc_group, qc_multisample,
            False, False, False, analysis_mode, analysis_id,
        )
        worker = Worker(run_plotting_dispatch, *args)
        self._set_active_run_task("plot")

        def on_started():
            self._start_task_timer("plot")
            self._log(f"Plotting started for {len(dataset_paths)} dataset(s) (analysis={analysis_mode}:{analysis_id or 'legacy'}).")

        def on_finished():
            elapsed = self._stop_task_timer("plot")
            self._log(f"Plotting finished in {elapsed:.2f}s.")
            self.workers.pop("plot", None)
            self._reset_run_controls_if_idle()

        def on_error(err: str):
            elapsed = self._stop_task_timer("plot")
            self._log(f"Plotting error after {elapsed:.2f}s: {err}")
            self.workers.pop("plot", None)
            self._reset_run_controls_if_idle()

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self.workers["plot"] = worker
        try:
            worker.start()
            self._wire_worker_output(worker)
        except Exception as exc:
            self._stop_task_timer("plot")
            self._log(f"Plotting error: {exc}")
            self.workers.pop("plot", None)
            self._reset_run_controls_if_idle()

    def stop_plot(self):
        worker = self.workers.get("plot")
        if worker:
            worker.stop()
            elapsed = self._stop_task_timer("plot")
            self._log(f"Plotting stopped after {elapsed:.2f}s.")
            self.workers.pop("plot", None)
            self._reset_run_controls_if_idle()

    def start_gqi(self):
        if not self._ensure_no_active_pipeline_task("GQI"):
            return
        dataset_paths = self._collect_dataset_paths()
        if not dataset_paths:
            QMessageBox.warning(self, "GQI", "Please add at least one dataset.")
            return
        derivatives_dir = self.derivatives_dir.text().strip() or None
        dataset_cfg_overrides = self._collect_dataset_config_overrides()
        analysis_mode, analysis_id, _cfg_policy, _sub_policy = self._collect_analysis_profile_settings()
        if not self._validate_analysis_selection(analysis_mode, analysis_id, cfg_policy=_cfg_policy, sub_policy=_sub_policy):
            return
        if analysis_mode == "new" and not analysis_id:
            QMessageBox.warning(self, "GQI", "Mode 'new' requires a profile ID for GQI recomputation.\nUse mode 'reuse' with a profile ID, or 'latest'/'legacy'.")
            return
        args = (
            dataset_paths, str(self.default_settings_path), derivatives_dir,
            self.global_config_path, dataset_cfg_overrides, analysis_mode, analysis_id,
        )
        worker = Worker(run_gqi_dispatch, *args)
        self._set_active_run_task("gqi")

        def on_started():
            self._start_task_timer("gqi")
            self._log(f"GQI started for {len(dataset_paths)} dataset(s) (analysis={analysis_mode}:{analysis_id or 'legacy'}).")

        def on_finished():
            elapsed = self._stop_task_timer("gqi")
            self._log(f"GQI finished in {elapsed:.2f}s.")
            self.workers.pop("gqi", None)
            self._reset_run_controls_if_idle()

        def on_error(err: str):
            elapsed = self._stop_task_timer("gqi")
            self._log(f"GQI error after {elapsed:.2f}s: {err}")
            self.workers.pop("gqi", None)
            self._reset_run_controls_if_idle()

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self.workers["gqi"] = worker
        try:
            worker.start()
            self._wire_worker_output(worker)
        except Exception as exc:
            self._stop_task_timer("gqi")
            self._log(f"GQI error: {exc}")
            self.workers.pop("gqi", None)
            self._reset_run_controls_if_idle()

    def stop_gqi(self):
        worker = self.workers.get("gqi")
        if worker:
            worker.stop()
            elapsed = self._stop_task_timer("gqi")
            self._log(f"GQI stopped after {elapsed:.2f}s.")
            self.workers.pop("gqi", None)
            self._reset_run_controls_if_idle()

    def start_all(self):
        if not self._ensure_no_active_pipeline_task("Run ALL"):
            return
        dataset_paths = self._collect_dataset_paths()
        if not dataset_paths:
            QMessageBox.warning(self, "Run ALL", "Please add at least one dataset.")
            return
        derivatives_dir = self.derivatives_dir.text().strip() or None
        dataset_subs = self._collect_dataset_sub_overrides()
        calc_jobs = self.jobs.value()
        plot_jobs = self.plot_jobs.value()
        dataset_njobs = self._collect_calc_njobs_overrides()
        dataset_cfg_overrides = self._collect_dataset_config_overrides()
        analysis_mode, analysis_id, cfg_policy, sub_policy = self._collect_analysis_profile_settings()
        if not self._validate_analysis_selection(analysis_mode, analysis_id, cfg_policy=cfg_policy, sub_policy=sub_policy):
            return
        qa_subject = self.chk_qa_subject.isChecked()
        qa_group = self.chk_qa_group.isChecked()
        qa_multisample = self.chk_qa_multisample.isChecked()
        qc_group = self.chk_qc_group.isChecked()
        qc_multisample = self.chk_qc_multisample.isChecked()
        multisample_requested = qa_multisample or qc_multisample
        if analysis_mode == "new" and multisample_requested and len(dataset_paths) > 1 and not analysis_id:
            analysis_id = self._generate_shared_analysis_id()
            self.edit_analysis_id.setText(analysis_id)
            self._log(f"Run ALL assigned shared analysis_id for multisample queue: {analysis_id}")
        if not self._validate_multisample_profile_compatibility(analysis_mode, analysis_id, qa_multisample=qa_multisample, qc_multisample=qc_multisample, require_existing_profiles=False):
            return
        requested_any_scope = any([qa_subject, qa_group, qa_multisample, qc_group, qc_multisample])
        args = (
            dataset_paths, str(self.default_settings_path), str(INTERNAL_PATH),
            "all", calc_jobs, plot_jobs, derivatives_dir, dataset_njobs, dataset_subs,
            self.global_config_path, dataset_cfg_overrides,
            analysis_mode, analysis_id, cfg_policy, sub_policy, False, False,
            qa_subject, qa_group, qa_multisample, qc_group, qc_multisample,
            False, False, not requested_any_scope,
        )
        worker = Worker(run_all_dispatch, *args)
        self._set_active_run_task("all")

        def on_started():
            self._start_task_timer("all")
            self._log(f"Run ALL started for {len(dataset_paths)} dataset(s) (analysis={analysis_mode}:{analysis_id or 'legacy'}).")

        def on_finished():
            elapsed = self._stop_task_timer("all")
            self._log(f"Run ALL finished in {elapsed:.2f}s.")
            self.workers.pop("all", None)
            self._reset_run_controls_if_idle()

        def on_error(err: str):
            elapsed = self._stop_task_timer("all")
            self._log(f"Run ALL error after {elapsed:.2f}s: {err}")
            self.workers.pop("all", None)
            self._reset_run_controls_if_idle()

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self.workers["all"] = worker
        try:
            worker.start()
            self._wire_worker_output(worker)
        except Exception as exc:
            self._stop_task_timer("all")
            self._log(f"Run ALL error: {exc}")
            self.workers.pop("all", None)
            self._reset_run_controls_if_idle()

    def stop_all(self):
        worker = self.workers.get("all")
        if worker:
            worker.stop()
            elapsed = self._stop_task_timer("all")
            self._log(f"Run ALL stopped after {elapsed:.2f}s.")
            self.workers.pop("all", None)
            self._reset_run_controls_if_idle()

    # ──────────────────────────────── #
    # QC Viewer launcher               #
    # ──────────────────────────────── #
    def _open_qc_viewer(self):
        """Open the integrated QC Viewer window."""
        try:
            from .qc_viewer import QCViewerWindow
            initial_dir = self.derivatives_dir.text() if self.derivatives_dir.text() else None
            self._qc_viewer_window = QCViewerWindow(
                parent=None,
                initial_dir=initial_dir,
            )
            self._qc_viewer_window.show()
        except Exception as e:
            QMessageBox.warning(
                self,
                "QC Viewer",
                f"Failed to open the QC Viewer:\n{e}",
            )

    # ──────────────────────────────── #
    # Live Output terminal             #
    # ──────────────────────────────── #
    def _open_live_terminal(self) -> None:
        """Show (or bring to front) the floating Live Output terminal window.

        The window accumulates all stdout/stderr emitted by active worker
        processes in real time. Existing workers are automatically wired up
        when the window opens; new workers are wired in each ``start_*`` method
        via ``_wire_worker_output``.
        """
        dialog = LiveTerminalDialog.get_instance(parent=None)
        # Wire any already-running workers that were started before the window
        # was opened (signal connections are de-duplicated by Qt).
        for name, w in self.workers.items():
            try:
                w.output_ready.connect(dialog.append_text, Qt.ConnectionType.UniqueConnection)
            except TypeError:
                pass  # already connected
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _wire_worker_output(self, worker: Worker) -> None:
        """Connect a new worker's output_ready signal to the Live Output dialog.

        If the dialog is not open, the signal simply has no receiver and the
        output is silently discarded — no error.
        """
        term = LiveTerminalDialog._instance
        if term is not None and term.isVisible():
            try:
                worker.output_ready.connect(term.append_text, Qt.ConnectionType.UniqueConnection)
            except TypeError:
                pass

    # ──────────────────────────────── #
    # ──────────────────────────────── #
    # Open CLI terminal                #
    # ──────────────────────────────── #
    def _open_cli_terminal(self) -> None:
        """Open a system terminal pre-configured with the MEGqc Python env.

        Delegates to ``output_monitoring.open_cli_terminal`` which handles
        macOS (Terminal.app), Windows (cmd.exe) and Linux (gnome-terminal /
        konsole / xfce4-terminal / xterm).
        """
        open_cli_terminal(log_callback=self._log)

    # ──────────────────────────────── #
    # generic worker wrapper           #
    # ──────────────────────────────── #
    def _run_task(self, key: str, func, *args):
        self.log.appendPlainText(f"Starting {key} …")
        worker = Worker(func, *args)
        worker.finished.connect(
            lambda k=key: self.log.appendPlainText(f"{k.capitalize()} finished")
        )
        worker.error.connect(
            lambda e, k=key: self.log.appendPlainText(f"{k.capitalize()} error: {e}")
        )
        worker.start()
        self.workers[key] = worker


def run_megqc_gui():
    """Entry point called by the ``megqc`` console script."""
    _ensure_xcb_cursor()  # no-op on non-Linux; auto-fixes missing libxcb-cursor0
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    # Fusion style is mandatory: it is the only Qt built-in style that fully
    # respects custom QPalette colours on macOS and Windows.  Without it,
    # the platform-native renderer ignores palette overrides and the themed
    # colours / button shapes look completely wrong.
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

