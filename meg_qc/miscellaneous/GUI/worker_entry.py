"""Entry point for subprocesses spawned by the MEGqc GUI.

Each GUI action launches a fresh Python interpreter using
``python -m meg_qc.miscellaneous.GUI.worker_entry`` so that joblib
workers live in a dedicated process tree.  The script imports the
requested callable and executes it with the JSON-encoded arguments.
It mirrors the robust termination model from the BIDS-Manager GUI.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import traceback
from typing import Any, List


def _configure_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to use UTF-8 on Windows.

    Windows consoles default to a narrow encoding (e.g. cp1252) that cannot
    represent many Unicode characters used in MEGqc messages (e.g. the
    non-breaking hyphen U+2011, warning sign U+26A0, arrows …).  Setting the
    environment variable *before* any subprocesses are spawned (joblib workers
    included) ensures they all inherit UTF-8 I/O.
    """
    if sys.platform == "win32":
        # Propagate to every child process spawned from here (joblib workers).
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        # Reconfigure the streams of *this* process.
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "buffer"):
                try:
                    new_stream = io.TextIOWrapper(
                        stream.buffer,
                        encoding="utf-8",
                        errors="replace",
                        line_buffering=True,
                    )
                    setattr(sys, stream_name, new_stream)
                except Exception:
                    pass  # Never crash the worker over I/O reconfiguration


def _become_session_leader() -> None:
    """Create a new POSIX session so ``killpg`` can terminate child workers."""

    try:
        os.setsid()
    except AttributeError:
        # ``setsid`` is not available on Windows; process groups are handled via
        # ``TerminateProcess`` when the GUI issues ``QProcess.kill``.
        pass


def _decode_args(payload: str) -> List[Any]:
    """Deserialize the JSON payload passed from the GUI."""

    data = json.loads(payload)
    if isinstance(data, list):
        return data
    # Accept any scalar for robustness and treat it as a single argument.
    return [data]


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="MEGqc GUI worker entry point")
    parser.add_argument("--func", required=True, help="module:function path to invoke")
    parser.add_argument("--args", required=True, help="JSON-encoded positional arguments")
    ns = parser.parse_args()

    _become_session_leader()

    try:
        module_path, func_name = ns.func.rsplit(":", 1)
    except ValueError:
        print(f"Invalid function path: {ns.func}", file=sys.stderr)
        return 2

    try:
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
    except Exception as exc:  # noqa: BLE001 - report import errors verbosely
        print(f"Failed to import {ns.func}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    try:
        args = _decode_args(ns.args)
    except json.JSONDecodeError as exc:
        print(f"Invalid argument payload: {exc}", file=sys.stderr)
        return 4

    try:
        func(*args)
    except Exception:  # noqa: BLE001 - propagate stack trace to stderr
        traceback.print_exc()
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
