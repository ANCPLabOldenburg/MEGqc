#!/usr/bin/env python
"""Run the MEEGqc test suite locally, end to end.

This is a convenience runner (not a pytest test module). It:

  1. downloads the cropped 10-second fixture bundle from the public URL and
     caches it next to this file (only the first time),
  2. runs pytest against it with the offscreen Qt platform,
  3. keeps everything it writes under ``tests/_local/`` (git-ignored), so the
     downloaded data and all test output stay at this directory's level and
     never pollute the repo or your datasets folder.

Included test modules
---------------------
  tests/realdata/test_megnet.py   - MEGnet pipeline integration (ECG/EOG)
  tests/realdata/test_calculation.py, test_output_completeness.py, etc.

Usage
-----
    python tests/run_local.py                              # the whole suite
    python tests/run_local.py tests/realdata/test_megnet.py  # MEGnet only
    python tests/run_local.py tests/gui -v                 # forward to pytest
    python tests/run_local.py -k ds007338                  # pick tests by id

Override the bundle URL with the MEEGQC_FIXTURES_URL environment variable.
Run it with the Python interpreter that has MEEGqc installed (``pip install -e .``).
"""
from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../MEEGqc/tests
REPO = HERE.parent                              # .../MEEGqc
LOCAL = HERE / "_local"                         # git-ignored (see .gitignore)
DATA = LOCAL / "meegqc_test_data"
OUTPUT = LOCAL / "output"

DEFAULT_URL = "https://cloud.uol.de/s/5JnCs6FxSnejjD6/download"
URL = os.environ.get("MEEGQC_FIXTURES_URL", DEFAULT_URL)


def _download(url: str, dest: Path) -> None:
    """Download a URL to *dest*. Prefer curl (uses system certs, avoids the
    macOS Python SSL-cert issue), fall back to urllib + certifi."""
    if shutil.which("curl"):
        subprocess.check_call(["curl", "-fSL", "-o", str(dest), url])
        return
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_data() -> None:
    if (DATA / "MANIFEST.json").is_file():
        print(f"fixtures: cached at {DATA}")
        return
    LOCAL.mkdir(parents=True, exist_ok=True)
    tgz = LOCAL / "fixtures.tar.gz"
    print(f"fixtures: downloading from {URL}")
    _download(URL, tgz)
    print("extracting...")
    with tarfile.open(tgz) as t:
        t.extractall(LOCAL)
    tgz.unlink()
    if not (DATA / "MANIFEST.json").is_file():
        sys.exit(f"ERROR: bundle did not contain meegqc_test_data/MANIFEST.json under {LOCAL}")
    print(f"fixtures: ready at {DATA}")


def main() -> int:
    ensure_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MEEGQC_TEST_DATA"] = str(DATA)
    env["QT_QPA_PLATFORM"] = "offscreen"
    # ensure the console scripts (run-meegqc, run-meegqc-plotting, ...) resolve
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    args = sys.argv[1:] or ["tests/"]
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
           f"--basetemp={OUTPUT}", *args]
    print(f"python  : {sys.version.split()[0]}")
    print(f"output  : {OUTPUT}")
    print("running :", " ".join(cmd))
    print("-" * 70)
    return subprocess.call(cmd, env=env, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())
