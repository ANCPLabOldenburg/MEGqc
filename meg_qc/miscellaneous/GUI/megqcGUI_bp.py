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
import time
import os
import signal
import configparser
import multiprocessing
from pathlib import Path

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
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QIcon, QPalette, QColor  # for loading images, setting icon, and theme toggling

# Core MEG QC pipeline functions
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

# Locate bundled settings and logo files within the package
try:
    from importlib.resources import files

    SETTINGS_PATH = Path(files('meg_qc.settings').joinpath('settings.ini'))
    INTERNAL_PATH = Path(files('meg_qc.settings').joinpath('settings_internal.ini'))
    # Automatically locate bundled logo/icon in meg_qc.GUI.other
    LOGO_PATH = Path(files('meg_qc.GUI.other').joinpath('logo.png'))
    ICON_PATH = LOGO_PATH  # use same logo for window icon or change to icon.png if available
except Exception:
    from meg_qc.settings import settings as _pkg

    pkg_dir = Path(os.path.dirname(_pkg.__file__))
    SETTINGS_PATH = pkg_dir / 'settings.ini'
    INTERNAL_PATH = pkg_dir / 'settings_internal.ini'
    # Fallback for logo/icon location
    try:
        from meg_qc.GUI.other import logo as _logo_mod

        logo_dir = Path(os.path.dirname(_logo_mod.__file__))
        LOGO_PATH = logo_dir / 'logo.png'
        ICON_PATH = LOGO_PATH
    except Exception:
        LOGO_PATH = Path()
        ICON_PATH = Path()


class Worker(QThread):
    """
    Executes a blocking function in a separate OS process group.
    QThread is used purely for Qt signal integration.
    This allows us to send SIGTERM to the process group
    to cleanly kill joblib parallel children.
    """
    started = pyqtSignal()
    finished = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args
        self.process = None  # Will hold multiprocessing.Process

    def run(self):
        # Emit started signal for UI feedback
        self.started.emit()
        t0 = time.time()

        # Define subprocess target: sets its own session ID
        def target():
            # Start new session so killpg kills all children
            os.setsid()
            try:
                self.func(*self.args)
            except Exception:
                # Exit with code != 0 to signal error
                sys.exit(1)

        # Launch subprocess
        self.process = multiprocessing.Process(target=target)
        self.process.start()
        # Block until done or terminated
        self.process.join()

        # If terminated by SIGTERM, treat as user cancel without error
        if self.process.exitcode == -signal.SIGTERM:
            return
        # Non-zero exit codes (other than SIGTERM) are errors
        if self.process.exitcode != 0:
            self.error.emit(f"Process exited with code {self.process.exitcode}")
            return

        # On success, emit elapsed time
        elapsed = time.time() - t0
        self.finished.emit(elapsed)


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
        self.setWindowIcon(QIcon("path/to/your/icon.png"))  # change if needed

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
        self.apply_theme("Dark  🌙")                # set default palette

    # ──────────────────────────────── #
    # palette dictionary builder       #
    # ──────────────────────────────── #
    def _build_theme_dict(self) -> dict[str, QPalette]:
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
        themes["Dark  🌙"] = dark

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
        themes["Light 🔆"] = light

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
        themes["Beige  🏜"] = beige

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
        themes["Ocean  🌊"] = ocean

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
        themes["Contrast  🌓"] = hc

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
        themes["Solar  🌞"] = solar

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
        themes["Cyber  🕶"] = cyber

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

        # — Calculation section —
        calc_box = QGroupBox("Calculation")
        calc_form = QFormLayout(calc_box)

        self.calc_data = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self._browse(self.calc_data))
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.addWidget(self.calc_data)
        row_lay.addWidget(btn_browse)
        calc_form.addRow("Data directory:", row)

        self.calc_subs = QLineEdit()
        self.calc_subs.setPlaceholderText("all or IDs, e.g. 009,012")
        calc_form.addRow("Subjects:", self.calc_subs)

        self.calc_jobs = QSpinBox()
        self.calc_jobs.setRange(-1, os.cpu_count() or 1)
        self.calc_jobs.setValue(-1)
        btn_info = QPushButton("Info")
        btn_info.setToolTip("Parallel jobs info")

        def show_jobs_info():
            QMessageBox.information(
                self, "Jobs Recommendation",
                "Use -1 for all cores.  RAM ≳ 3.5 × cores is recommended."
            )
        btn_info.clicked.connect(show_jobs_info)

        row2 = QWidget()
        row2_lay = QHBoxLayout(row2)
        row2_lay.setContentsMargins(0, 0, 0, 0)
        row2_lay.addWidget(self.calc_jobs)
        row2_lay.addWidget(btn_info)
        calc_form.addRow("Jobs:", row2)

        btn_run = QPushButton("Run Calculation")
        btn_run.clicked.connect(self.start_calc)
        btn_stop = QPushButton("Stop Calculation")
        btn_stop.clicked.connect(self.stop_calc)
        row_btns = QWidget()
        btns_lay = QHBoxLayout(row_btns)
        btns_lay.setContentsMargins(0, 0, 0, 0)
        btns_lay.addWidget(btn_run)
        btns_lay.addWidget(btn_stop)
        calc_form.addRow("", row_btns)

        lay.addWidget(calc_box)

        # — Plotting section —
        plot_box = QGroupBox("Plotting")
        plot_form = QFormLayout(plot_box)

        self.plot_data = QLineEdit()
        btn_pbrowse = QPushButton("Browse")
        btn_pbrowse.clicked.connect(lambda: self._browse(self.plot_data))
        prow = QWidget()
        pl = QHBoxLayout(prow)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.plot_data)
        pl.addWidget(btn_pbrowse)
        plot_form.addRow("Data directory:", prow)

        btn_prun = QPushButton("Run Plotting")
        btn_prun.clicked.connect(self.start_plot)
        btn_pstop = QPushButton("Stop Plotting")
        btn_pstop.clicked.connect(self.stop_plot)
        prow2 = QWidget()
        pl2 = QHBoxLayout(prow2)
        pl2.setContentsMargins(0, 0, 0, 0)
        pl2.addWidget(btn_prun)
        pl2.addWidget(btn_pstop)
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
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            edit.setText(path)

    # ──────────────────────────────── #
    # start / stop handlers            #
    # ──────────────────────────────── #
    def start_calc(self):
        data_dir = self.calc_data.text().strip()
        subs_raw = self.calc_subs.text().strip()
        subs = (
            [s.strip() for s in subs_raw.split(",") if s.strip()]
            if subs_raw and subs_raw.lower() != "all"
            else "all"
        )
        n_jobs = self.calc_jobs.value()
        args = [str(SETTINGS_PATH), str(INTERNAL_PATH), data_dir, subs, n_jobs]
        self._run_task("calc", make_derivative_meg_qc, *args)

    def stop_calc(self):
        worker = self.workers.get("calc")
        if worker and worker.process and worker.process.is_alive():
            os.killpg(os.getpgid(worker.process.pid), signal.SIGTERM)
            self.log.appendPlainText("Calculation stopped")

    def start_plot(self):
        data_dir = self.plot_data.text().strip()
        self._run_task("plot", make_plots_meg_qc, data_dir)

    def stop_plot(self):
        worker = self.workers.get("plot")
        if worker and worker.process and worker.process.is_alive():
            os.killpg(os.getpgid(worker.process.pid), signal.SIGTERM)
            self.log.appendPlainText("Plotting stopped")

    # ──────────────────────────────── #
    # generic worker wrapper           #
    # ──────────────────────────────── #
    def _run_task(self, key: str, func, *args):
        self.log.appendPlainText(f"Starting {key} …")
        worker = Worker(func, *args)
        worker.finished.connect(
            lambda t, k=key: self.log.appendPlainText(f"{k.capitalize()} finished in {t:.2f}s")
        )
        worker.error.connect(
            lambda e, k=key: self.log.appendPlainText(f"{k.capitalize()} error: {e}")
        )
        worker.start()
        self.workers[key] = worker


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
