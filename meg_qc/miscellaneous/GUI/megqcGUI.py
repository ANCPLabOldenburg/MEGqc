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
import configparser
from pathlib import Path
from typing import Dict, Optional

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
    QMenu               #  ← add this entry
)
from PyQt6.QtCore import QObject, QProcess, pyqtSignal, Qt, QCoreApplication
from PyQt6.QtGui import QPixmap, QIcon, QPalette, QColor  # for loading images, setting icon, and theme toggling

# Core MEG QC pipeline functions
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary

# Use Qt5 widget integration and not the OS integrations (This will prevent incompatibilities)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)

# Locate bundled settings and logo files within the package
# Locate bundled settings and logo files within the package by filepath
# GUI_DIR es la carpeta donde está este mismo archivo
GUI_DIR = Path(__file__).parent
# PKG_ROOT apunta a la raíz del paquete meg_qc (dos niveles arriba)
PKG_ROOT = GUI_DIR.parent.parent

# Rutas a los ficheros
SETTINGS_PATH = PKG_ROOT / "settings" / "settings.ini"
INTERNAL_PATH = PKG_ROOT / "settings" / "settings_internal.ini"
# Logo e icon se quedan en la carpeta GUI
LOGO_PATH = GUI_DIR / "logo.png"
ICON_PATH = LOGO_PATH


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
        process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        process.started.connect(self._on_started)
        process.setProgram(sys.executable)
        process.setArguments(["-m", entry_module, "--func", func_path, "--args", payload])
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

        # ``kill`` complements the explicit tree teardown above.  When the
        # child already exited this is a no-op, otherwise it ensures the GUI
        # regains control even if the task ignores SIGTERM.
        self.process.kill()


class SettingsEditor(QWidget):
    """
    Scrollable editor for settings.ini.
    Comments above each key are shown as tooltips on the QLineEdit.
    """

    def __init__(self, config: configparser.ConfigParser, path: Path):
        super().__init__()
        self.config = config
        self.path = path
        self._comment_map = {}
        current_section = 'DEFAULT'
        pending = []
        try:
            with open(path, 'r') as f:
                for raw in f:
                    line = raw.rstrip('\n')
                    stripped = line.strip()
                    if stripped.startswith('[') and stripped.endswith(']'):
                        current_section = stripped[1:-1]
                        pending = []
                    elif stripped.startswith('#') or stripped.startswith(';'):
                        txt = stripped.lstrip('#; ').strip()
                        if txt:
                            pending.append(txt)
                    elif '=' in stripped and not stripped.startswith('#') and not stripped.startswith(';'):
                        key = stripped.split('=', 1)[0].strip()
                        comment = ' '.join(pending)
                        self._comment_map[(current_section, key)] = comment
                        pending = []
        except Exception:
            pass

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form_layout = QVBoxLayout(container)
        self.fields = {}

        # DEFAULT section
        defaults = config.defaults()
        if defaults:
            box = QGroupBox('DEFAULT')
            form = QFormLayout(box)
            for key, val in defaults.items():
                field = QLineEdit(val)
                if (tip := self._comment_map.get(('DEFAULT', key))):
                    field.setToolTip(tip)
                form.addRow(QLabel(key), field)
                self.fields[('DEFAULT', key)] = field
            form_layout.addWidget(box)
            sep = QFrame();
            sep.setFrameShape(QFrame.Shape.HLine)
            form_layout.addWidget(sep)

        # Other sections
        for section in config.sections():
            box = QGroupBox(section)
            form = QFormLayout(box)
            items = config._sections.get(section, {})
            for key, val in items.items():
                field = QLineEdit(val)
                if (tip := self._comment_map.get((section, key))):
                    field.setToolTip(tip)
                form.addRow(QLabel(key), field)
                self.fields[(section, key)] = field
            form_layout.addWidget(box)
            sep2 = QFrame();
            sep2.setFrameShape(QFrame.Shape.HLine)
            form_layout.addWidget(sep2)

        save = QPushButton('Save Settings')
        save.clicked.connect(self.save)
        form_layout.addWidget(save, alignment=Qt.AlignmentFlag.AlignCenter)

        scroll.setWidget(container)
        layout.addWidget(scroll)

    def save(self):
        # 1. recoge los valores nuevos en un dict {(section, key): str_value}
        new_vals = {(sec, key): w.text() for (sec, key), w in self.fields.items()}

        try:
            # 2. lee el archivo original completo
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            current = "DEFAULT"
            for i, raw in enumerate(lines):
                s = raw.strip()

                # detecta sección
                if s.startswith("[") and s.endswith("]"):
                    current = s[1:-1]

                # detecta línea clave = valor (no comentario)
                elif "=" in s and not s.startswith("#") and not s.startswith(";"):
                    key = s.split("=", 1)[0].strip()
                    if (current, key) in new_vals:
                        # 3. sustituye solo el valor, deja intacto todo lo demás (espacios, inline comments)
                        prefix, _ = raw.split("=", 1)
                        new_val = new_vals[(current, key)]
                        # conserva el texto que hubiera después del valor (p.ej. comentario al final de línea)
                        after = raw.split("=", 1)[1]
                        if "#" in after or ";" in after:
                            # si hay comentario inline, sepáralo
                            val_part, comment_part = after.split("#", 1) if "#" in after else after.split(";", 1)
                            lines[
                                i] = f"{prefix}= {new_val}  #{comment_part}" if "#" in after else f"{prefix}= {new_val}  ;{comment_part}"
                        else:
                            lines[i] = f"{prefix}= {new_val}\n"

            # 4. escribe de vuelta
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            QMessageBox.information(self, "Settings", "Settings saved successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")


class MainWindow(QMainWindow):
    """
    Main application window
    ─────────────────────────────────────────────────────────────
    • “Run” tab       – launch / stop calculation and plotting jobs
    • “Settings” tab  – edit settings.ini with mouse‑hover tooltips
    • Status bar      – CPU / RAM info + Theme selector (Dark / Light / Beige)
    """

    # ──────────────────────────────── #
    # constructor                      #
    # ──────────────────────────────── #
    def __init__(self):
        super().__init__()

        # -- Window basics -------------------------------------------------
        self.setWindowTitle("MEGqc")
        self.resize(500, 600)
        if ICON_PATH and ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        # -- Load settings (keep key case for tooltips) --------------------
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(SETTINGS_PATH)

        # -- Tabs: Run | Settings -----------------------------------------
        tabs = QTabWidget()
        tabs.addTab(self._create_run_tab(), "Run")
        tabs.addTab(SettingsEditor(self.config, SETTINGS_PATH), "Settings")

        # -- Central layout (optional logo + tabs) ------------------------
        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(5, 5, 5, 5)

        if LOGO_PATH and LOGO_PATH.exists():
            logo = QLabel()
            pix = QPixmap(str(LOGO_PATH))
            pix = pix.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
            logo.setPixmap(pix)
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vlay.addWidget(logo)

        vlay.addWidget(tabs)
        self.setCentralWidget(central)

        # -- Status bar: Theme menu + system info -------------------------
        self.themes = self._build_theme_dict()      # dict: name → palette
        # status‑bar theme selector
        self.theme_btn = QPushButton("🌓")  # half‑moon icon suggests theme switch
        self.theme_btn.setFixedWidth(60)  # smaller, now just an icon
        self.statusBar().addWidget(self.theme_btn)
        self.theme_btn.setStyleSheet("margin-left:15px;")

        theme_menu = QMenu(self)
        for name in self.themes.keys():
            act = theme_menu.addAction(name)
            act.triggered.connect(lambda _, n=name: self.apply_theme(n))
        self.theme_btn.setMenu(theme_menu)

        # System info (CPU / RAM)
        cpu_cnt = os.cpu_count() or 1
        total_bytes = psutil.virtual_memory().total if has_psutil else 0
        if not has_psutil:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_bytes = int(line.split()[1]) * 1024
                            break
            except Exception:
                total_bytes = 0
        total_gb = total_bytes / (1024**3) if total_bytes else 0

        sys_w = QWidget()
        sys_l = QVBoxLayout(sys_w)
        sys_l.setContentsMargins(0, 0, 0, 0)
        sys_l.addWidget(QLabel(f"CPUs: {cpu_cnt}"))
        sys_l.addWidget(QLabel(f"Total RAM: {total_gb:.1f} GB"))
        self.statusBar().addPermanentWidget(sys_w)

        # -- Workers dict + initial theme ---------------------------------
        self.workers: dict[str, Worker] = {}
        self.apply_theme("Ocean")                # set default palette


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
        dark.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
        dark.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
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
        themes["Palenight"] = pale

        return themes

    # ──────────────────────────────── #
    # apply selected theme             #
    # ──────────────────────────────── #
    def apply_theme(self, name: str):
        """Apply palette chosen from the Theme menu."""
        QApplication.instance().setPalette(self.themes[name])

    # ──────────────────────────────── #
    # build “Run” tab                  #
    # (identical to your original)     #
    # ──────────────────────────────── #
    def _create_run_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(12)

        # --- shared data directory ---
        shared = QFormLayout()

        self.data_dir = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self._browse(self.data_dir))
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.addWidget(self.data_dir)
        row_lay.addWidget(btn_browse)
        shared.addRow("Data directory:", row)

        self.jobs = QSpinBox()
        self.jobs.setRange(-1, os.cpu_count() or 1)
        self.jobs.setValue(-1)
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
            QMessageBox.information(self, 'Jobs Recommendation', info)

        btn_info.clicked.connect(show_jobs_info)

        row_jobs = QWidget()
        jobs_lay = QHBoxLayout(row_jobs)
        jobs_lay.setContentsMargins(0, 0, 0, 0)
        jobs_lay.addWidget(self.jobs)
        jobs_lay.addWidget(btn_info)
        shared.addRow("Jobs:", row_jobs)

        lay.addLayout(shared)

        # — Calculation section —
        calc_box = QGroupBox("Calculation")
        calc_form = QFormLayout(calc_box)

        self.calc_subs = QLineEdit()
        self.calc_subs.setPlaceholderText("all or IDs, e.g. 009,012")
        calc_form.addRow("Subjects:", self.calc_subs)

        self.btn_calc_run = QPushButton("Run Calculation")
        self.btn_calc_run.clicked.connect(self.start_calc)
        self.btn_calc_stop = QPushButton("Stop Calculation")
        self.btn_calc_stop.clicked.connect(self.stop_calc)
        row_btns = QWidget()
        btns_lay = QHBoxLayout(row_btns)
        btns_lay.setContentsMargins(0, 0, 0, 0)
        btns_lay.addWidget(self.btn_calc_run)
        btns_lay.addWidget(self.btn_calc_stop)
        calc_form.addRow("", row_btns)

        self.btn_gqi_run = QPushButton("Run GQI")
        self.btn_gqi_run.clicked.connect(self.start_gqi)
        self.btn_gqi_stop = QPushButton("Stop GQI")
        self.btn_gqi_stop.clicked.connect(self.stop_gqi)
        row_gqi = QWidget()
        gqi_lay = QHBoxLayout(row_gqi)
        gqi_lay.setContentsMargins(0, 0, 0, 0)
        gqi_lay.addWidget(self.btn_gqi_run)
        gqi_lay.addWidget(self.btn_gqi_stop)
        calc_form.addRow("", row_gqi)

        lay.addWidget(calc_box)

        # — Plotting section —
        plot_box = QGroupBox("Plotting")
        plot_form = QFormLayout(plot_box)

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

        lay.addWidget(plot_box)

        # — Log output —
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(QLabel("Log:"))
        lay.addWidget(self.log)

        return w

    # ──────────────────────────────── #
    # helper: browse directory         #
    # ──────────────────────────────── #
    def _browse(self, edit: QLineEdit):
        # Build options: only show directories, and force Qt's built-in dialog
        options = (
                QFileDialog.Option.ShowDirsOnly
                | QFileDialog.Option.DontUseNativeDialog
        )

        # Call Qt's folder picker
        #   - parent: self
        #   - window title: "Select Directory"
        #   - initial directory: use current text in the QLineEdit, or "" for home
        #   - options: as defined above
        start_dir = edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            start_dir,
            options
        )

        # If the user picked something, update the QLineEdit
        if path:
            edit.setText(path)

    # ──────────────────────────────── #
    # start / stop handlers            #
    # ──────────────────────────────── #
    def start_calc(self):
        """
        Slot for the “Run Calculation” button.
        1) Read user inputs from the Run tab
        2) Build the argument tuple for make_derivative_meg_qc
        3) Instantiate and configure a Worker
        4) Start the Worker and store it in self.workers
        """
        # 1) Gather inputs
        data_dir  = self.data_dir.text().strip()
        subs_raw  = self.calc_subs.text().strip()
        # If user typed “all” (case-insensitive) or left blank, use the string "all";
        # otherwise split by commas into a list of IDs.
        subs = (
            [s.strip() for s in subs_raw.split(",") if s.strip()]
            if subs_raw and subs_raw.lower() != "all"
            else "all"
        )
        n_jobs = self.jobs.value()

        # 2) Match the signature: (settings_path, internal_path, data_dir, subs, n_jobs)
        args = (
            str(SETTINGS_PATH),
            str(INTERNAL_PATH),
            data_dir,
            subs,
            n_jobs,
        )

        # 3) Create the Worker, hook up signals → log
        worker = Worker(make_derivative_meg_qc, *args)
        # Prevent launching the same task multiple times while it is already running.
        self.btn_calc_run.setEnabled(False)

        def on_started():
            self.log.appendPlainText("Calculation started")

        def on_finished():
            # Successful completion: log and re-enable button for future runs.
            self.log.appendPlainText("Calculation finished")
            self.btn_calc_run.setEnabled(True)
            self.workers.pop("calc", None)

        def on_error(err: str):
            # Error path: report and allow the user to retry.
            self.log.appendPlainText(f"Calculation error: {err}")
            self.btn_calc_run.setEnabled(True)
            self.workers.pop("calc", None)

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        # 4) Launch and save. Store the worker before starting to avoid race conditions
        # where the thread finishes extremely quickly.
        self.workers["calc"] = worker
        try:
            worker.start()
        except Exception as exc:
            # Starting the worker can fail if the OS denies process creation.
            self.log.appendPlainText(f"Calculation error: {exc}")
            self.btn_calc_run.setEnabled(True)
            self.workers.pop("calc", None)


    def stop_calc(self):
        """
        Slot for the “Stop Calculation” button.
        Delegates termination logic to Worker.stop(), which handles
        both Unix (killpg) and Windows (process.terminate()).
        """
        worker = self.workers.get("calc")
        if worker:
            worker.stop()
            self.log.appendPlainText("Calculation stopped")
            self.btn_calc_run.setEnabled(True)
            self.workers.pop("calc", None)


    def start_plot(self):
        """
        Slot for the “Run Plotting” button.
        Same pattern as start_calc:
        1) Gather inputs
        2) Build args tuple for make_plots_meg_qc
        3) Hook up signals → log
        4) Start Worker and store
        """
        # 1) Gather inputs
        data_dir = self.data_dir.text().strip()
        n_jobs = self.jobs.value()

        # 2) Build args tuple for make_plots_meg_qc
        # The plotting backend (full or lite) is selected inside
        # ``make_plots_meg_qc`` based on the 'full_html_reports' option in
        # settings.ini.
        args = (data_dir, n_jobs)

        # 3) Create Worker and wire signals
        worker = Worker(make_plots_meg_qc, *args)
        # Prevent duplicate plotting runs until the active one ends.
        self.btn_plot_run.setEnabled(False)

        def on_started():
            self.log.appendPlainText("Plotting started")

        def on_finished():
            self.log.appendPlainText("Plotting finished")
            self.btn_plot_run.setEnabled(True)
            self.workers.pop("plot", None)

        def on_error(err: str):
            self.log.appendPlainText(f"Plotting error: {err}")
            self.btn_plot_run.setEnabled(True)
            self.workers.pop("plot", None)

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        # 4) Launch and save
        self.workers["plot"] = worker
        try:
            worker.start()
        except Exception as exc:
            self.log.appendPlainText(f"Plotting error: {exc}")
            self.btn_plot_run.setEnabled(True)
            self.workers.pop("plot", None)


    def stop_plot(self):
        """
        Slot for the “Stop Plotting” button.
        Calls Worker.stop() to cleanly terminate the plotting process
        on both Unix and Windows.
        """
        worker = self.workers.get("plot")
        if worker:
            worker.stop()
            self.log.appendPlainText("Plotting stopped")
            self.btn_plot_run.setEnabled(True)
            self.workers.pop("plot", None)

    def start_gqi(self):
        """Run Global Quality Index calculation only."""
        data_dir = self.data_dir.text().strip()
        args = (data_dir, str(SETTINGS_PATH))
        worker = Worker(generate_gqi_summary, *args)
        # Prevent concurrent GQI runs; enable again once finished or stopped.
        self.btn_gqi_run.setEnabled(False)

        def on_started():
            self.log.appendPlainText("GQI started")

        def on_finished():
            self.log.appendPlainText("GQI finished")
            self.btn_gqi_run.setEnabled(True)
            self.workers.pop("gqi", None)

        def on_error(err: str):
            self.log.appendPlainText(f"GQI error: {err}")
            self.btn_gqi_run.setEnabled(True)
            self.workers.pop("gqi", None)

        worker.started.connect(on_started)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self.workers["gqi"] = worker
        try:
            worker.start()
        except Exception as exc:
            self.log.appendPlainText(f"GQI error: {exc}")
            self.btn_gqi_run.setEnabled(True)
            self.workers.pop("gqi", None)

    def stop_gqi(self):
        worker = self.workers.get("gqi")
        if worker:
            worker.stop()
            self.log.appendPlainText("GQI stopped")
            self.btn_gqi_run.setEnabled(True)
            self.workers.pop("gqi", None)


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
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_megqc_gui()

