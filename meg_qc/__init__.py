from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meg_qc")
except PackageNotFoundError:
    # Source checkout fallback (kept in sync with pyproject.toml).
    __version__ = "0.6.7"
