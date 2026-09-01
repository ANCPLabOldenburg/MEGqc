#!/usr/bin/env python
"""Verify that datalad datasets have been fully fetched (not just cloned).

After ``datalad clone``, data files are tiny LFS pointer stubs (~1 KB).
After ``datalad get``, they become real files (typically many MB).

Usage:
    python scripts/verify_datalad.py /path/to/ds003483
    python scripts/verify_datalad.py /path/to/datasets/   # checks all subdirs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Extensions of neuroimaging data files that should always be > 1 MB
# when fully fetched.  (JSON sidecars, .tsv, README etc. are legitimately small.)
DATA_EXTENSIONS = {
    ".fif", ".ds", ".edf", ".bdf", ".cnt", ".mat", ".sqd", ".kdf",
    ".meg4", ".hdr", ".eeg", ".vmrk", ".vhdr", ".set", ".fdt",
    ".nii", ".nii.gz", ".mgh", ".mgz",
}

MIN_SIZE_MB = 1
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def check_file(path: Path) -> str | None:
    """Return a warning string if *path* looks like a pointer stub, else None."""
    suffix = path.suffix.lower()
    # handle .nii.gz
    if path.name.endswith(".nii.gz"):
        suffix = ".nii.gz"

    if suffix not in DATA_EXTENSIONS:
        return None

    size = path.stat().st_size

    # Check 1: LFS pointer stub
    if size < 10_000:
        try:
            with open(path, "rb") as f:
                if f.read(len(LFS_MAGIC)) == LFS_MAGIC:
                    return f"  LFS POINTER STUB ({size:,} bytes) — needs datalad get"
        except (OSError, PermissionError):
            pass

    # Check 2: suspiciously small data file
    if size < MIN_SIZE_MB * 1024 * 1024:
        return f"  suspiciously small ({size:,} bytes / {size/1024:.1f} KB)"

    return None


def check_dataset(ds_path: Path) -> list[str]:
    """Walk a single dataset directory and return all warnings."""
    warnings: list[str] = []
    for f in sorted(ds_path.rglob("*")):
        if not f.is_file():
            continue
        msg = check_file(f)
        if msg:
            warnings.append(f"{f.relative_to(ds_path)}{msg}")
    return warnings


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1

    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"ERROR: {target} is not a directory")
        return 1

    # If the target itself looks like a dataset (has a participants.tsv or
    # sub-* dirs), check it directly.  Otherwise treat it as a parent folder
    # and check each child directory.
    has_subjects = any(target.glob("sub-*")) or (target / "participants.tsv").is_file()
    datasets = [target] if has_subjects else sorted(
        d for d in target.iterdir() if d.is_dir()
    )

    total_warnings = 0
    for ds in datasets:
        warnings = check_dataset(ds)
        if warnings:
            print(f"\n{ds.name}: {len(warnings)} incomplete file(s)")
            for w in warnings:
                print(w)
            total_warnings += len(warnings)
        else:
            print(f"{ds.name}: OK")

    if total_warnings:
        print(f"\n{total_warnings} incomplete file(s) found — run datalad get on the affected dataset(s)")
        return 1
    else:
        print("\nAll datasets verified.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
