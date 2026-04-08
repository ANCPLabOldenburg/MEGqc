"""
output_monitoring – Live terminal streaming and CLI launcher for MEGqc GUI.

Public API
----------
LiveTerminalDialog  – floating, minimisable window that streams worker output.
open_cli_terminal   – open the host OS terminal pre-configured with the venv.
"""

from .live_terminal import LiveTerminalDialog
from .cli_terminal import open_cli_terminal

__all__ = ["LiveTerminalDialog", "open_cli_terminal"]

