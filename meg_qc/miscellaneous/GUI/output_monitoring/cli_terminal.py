"""
cli_terminal.py – Open the host OS terminal with the MEGqc venv pre-activated.

Strategy
--------
* macOS  – writes a ``.command`` file and opens it with the ``open`` command.
           macOS always opens ``.command`` files in a **new** Terminal.app window;
           no AppleScript is required (and AppleScript ``make new window`` is
           unreliable when Terminal is not already running — error -10000).
* Windows – ``.bat`` file opened via ``cmd /k`` or Windows Terminal.
* Linux  – gnome-terminal / konsole / xfce4-terminal / xterm using
           ``bash --init-file``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional

from PyQt6.QtWidgets import QMessageBox


# ── banner lines (plain ASCII — no encoding issues in any shell) ───────────────
_BANNER_LINES = [
    "echo ''",
    "echo '======================================================'",
    "echo '   MEGqc CLI  -  activated environment'",
    "echo '======================================================'",
    "echo ''",
    "echo 'Entry points  (run each with --help for full options):'",
    "echo ''",
    "echo '  megqc'",
    "echo '      Launch the MEGqc graphical interface.'",
    "echo ''",
    "echo '  run-megqc --help'",
    "echo '      QA/QC calculation pipeline.'",
    "echo ''",
    "echo '  run-megqc-plotting --help'",
    "echo '      Report plotting module.'",
    "echo ''",
    "echo '  globalqualityindex --help'",
    "echo '      Global Quality Index (GQI) module.'",
    "echo ''",
    "echo '  get-megqc-config --help'",
    "echo '      Configuration / settings utility.'",
    "echo ''",
    "echo '======================================================'",
    "echo ''",
]


def _write_startup_script(activate_cmd: str, shell: str = "zsh",
                           suffix: str = ".sh") -> str:
    """Write activation + banner to a temp file and return its path.

    The script ends with ``exec <shell>`` which replaces the script process
    with a live interactive shell — keeping the terminal window open.
    The script deletes itself (``rm -f "$0"``) just before handing off so no
    temp files are left on disk.
    """
    python_bin = sys.executable
    lines = (
        [f"#!/usr/bin/env {shell}", activate_cmd, ""]
        + _BANNER_LINES
        + [
            f"echo 'Python: {python_bin}'",
            "echo ''",
            # Clean up this temp file, then replace process with interactive shell.
            'rm -f "$0"',
            f"exec {shell}",
            "",
        ]
    )
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="megqc_cli_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path


def open_cli_terminal(log_callback: Optional[Callable[[str], None]] = None) -> None:
    """Open a system terminal pre-configured with the MEGqc Python environment."""

    def _log(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)

    python_bin = sys.executable
    venv_dir = os.path.dirname(os.path.dirname(python_bin))

    try:
        # ── macOS ──────────────────────────────────────────────────────────
        if sys.platform == "darwin":
            activate_path = os.path.join(venv_dir, "bin", "activate")
            activate_cmd = (
                f"source '{activate_path}'"
                if os.path.isfile(activate_path)
                else f"export PATH='{os.path.dirname(python_bin)}:$PATH'"
            )
            # Use .command extension: macOS opens these in a NEW Terminal window
            # automatically without any AppleScript — completely avoids error -10000.
            tmp = _write_startup_script(activate_cmd, shell="zsh", suffix=".command")
            subprocess.Popen(["open", tmp])
            _log("Opened CLI terminal (Terminal.app via .command file).")

        # ── Windows ────────────────────────────────────────────────────────
        elif sys.platform == "win32":
            activate_bat = os.path.join(venv_dir, "Scripts", "activate.bat")
            bat_lines = ["@echo off"]
            if os.path.isfile(activate_bat):
                bat_lines.append(f'call "{activate_bat}"')
            else:
                bat_lines.append(f'set PATH={os.path.dirname(python_bin)};%PATH%')
            bat_lines += [
                "echo.",
                "echo ======================================================",
                "echo    MEGqc CLI  -  activated environment",
                "echo ======================================================",
                "echo.",
                "echo Entry points (run each with --help for full options):",
                "echo.",
                "echo   megqc                          Launch the GUI.",
                "echo   run-megqc --help               Calculation pipeline.",
                "echo   run-megqc-plotting --help      Plotting module.",
                "echo   globalqualityindex --help      GQI module.",
                "echo   get-megqc-config --help        Config utility.",
                "echo.",
                f"echo Python: {python_bin}",
                "echo.",
                "echo ======================================================",
                "echo.",
            ]
            fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="megqc_cli_")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\r\n".join(bat_lines) + "\r\n")
            launcher = "wt" if shutil.which("wt") else None
            if launcher:
                subprocess.Popen(f'wt cmd /k "{bat_path}"', shell=True)
            else:
                subprocess.Popen(f'cmd /k "{bat_path}"', shell=True)
            _log("Opened CLI terminal (Windows).")

        # ── Linux ──────────────────────────────────────────────────────────
        else:
            activate_path = os.path.join(venv_dir, "bin", "activate")
            activate_cmd = (
                f"source '{activate_path}'"
                if os.path.isfile(activate_path)
                else f"export PATH='{os.path.dirname(python_bin)}:$PATH'"
            )
            tmp = _write_startup_script(activate_cmd, shell="bash")
            # --init-file sources the script without printing the path; the
            # trailing 'exec bash' inside the script keeps the window alive.
            candidates = [
                ["gnome-terminal", "--", "bash", "--init-file", tmp],
                ["konsole", "-e", f"bash --init-file {tmp}"],
                ["xfce4-terminal", "-e", f"bash --init-file {tmp}"],
                ["xterm", "-e", f"bash --init-file {tmp}"],
            ]
            launched = False
            for cmd_parts in candidates:
                if shutil.which(cmd_parts[0]):
                    subprocess.Popen(cmd_parts)
                    _log(f"Opened CLI terminal ({cmd_parts[0]}).")
                    launched = True
                    break
            if not launched:
                _log("No supported terminal emulator found.")
                QMessageBox.warning(
                    None,
                    "Open CLI",
                    "Could not find a supported terminal emulator.\n"
                    "Activate the environment manually:\n\n"
                    f"  source {activate_path}\n"
                    "  run-megqc --help",
                )

    except Exception as exc:
        _log(f"CLI terminal launch failed: {exc}")
        QMessageBox.warning(None, "Open CLI", f"Failed to open terminal:\n{exc}")
