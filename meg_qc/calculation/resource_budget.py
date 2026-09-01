"""Bound MEEGqc's parallelism by memory as well as by cores.

Two related concerns live here:

* **The memory guard (always on).** ``--n_jobs`` alone says nothing about RAM.
  Every recording processed concurrently is held in memory (MNE loads with
  ``preload=True`` and keeps filtered/resampled copies), so a large ``n_jobs``
  on big recordings will exhaust the machine and be killed by the OOM killer
  with no useful message. The guard turns "how many cores" into "how many
  recordings can we actually hold", and only ever *reduces* concurrency.

* **Hungry jobs (opt-in).** Subject-level workers fall idle once fewer subjects
  remain than workers. The slot registry lets a still-running subject notice
  that and spread its own recordings across the free capacity.

Everything is cross-platform: memory comes from ``psutil`` (already a
dependency) rather than ``/proc/meminfo``, and the slot registry uses an
atomic replace that behaves the same on Linux, macOS and Windows.
"""

from __future__ import annotations

import json
import time
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Measured on MEG FIF: a 1.92 GB recording produced a 7.7 GB resident worker,
# i.e. int32 on disk becomes float64 in memory plus filtered/resampled copies.
DEFAULT_EXPANSION = 4.0
DEFAULT_MEM_FRACTION = 0.5
DEFAULT_MEM_RESERVE_GB = 32.0

_GB = 1024 ** 3


# ── size measurement ────────────────────────────────────────────────────────
def recording_size_bytes(path) -> int:
    """On-disk bytes MNE will actually read for one recording.

    ``Path.stat().st_size`` is wrong for most of what MEEGqc reads:

    * a CTF recording is a **directory** - ``st_size`` reports the directory
      inode (a few hundred bytes) for a multi-GB recording;
    * in a datalad/git-annex dataset the data files are symlinks, so the size
      must be taken after dereferencing (``stat`` does this, ``lstat`` does not);
    * a FIF recording may be split across ``_meg.fif``, ``_meg-1.fif``, ... and
      MNE loads the whole chain;
    * an EEGLAB ``.set`` keeps its samples in a sibling ``.fdt``.

    Returns 0 when the path cannot be measured; callers treat that as "unknown"
    rather than "free", so an unmeasurable file never inflates the budget.
    """
    p = Path(path)
    try:
        if p.is_dir():
            total = 0
            for child in p.rglob("*"):
                try:
                    if child.is_file():
                        total += child.stat().st_size
                except OSError:
                    continue
            return total

        total = p.stat().st_size
        name = p.name
        if name.endswith(".fif"):
            stem = name[:-4]
            for sibling in p.parent.glob(f"{stem}-[0-9]*.fif"):
                try:
                    total += sibling.stat().st_size
                except OSError:
                    continue
        elif name.endswith(".set"):
            fdt = p.with_suffix(".fdt")
            try:
                if fdt.exists():
                    total += fdt.stat().st_size
            except OSError:
                pass
        return total
    except OSError:
        return 0


# ── memory probe ────────────────────────────────────────────────────────────
def memory_available_bytes() -> Optional[int]:
    """Currently available RAM, or None when it cannot be determined.

    Uses ``psutil`` so Linux, macOS and Windows behave identically. "Available"
    rather than "free" matters: on this machine free was 164 GB while available
    was 896 GB, the difference being reclaimable page cache. Sizing against
    "free" would throttle to nothing on an otherwise idle machine.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


# ── the budget ──────────────────────────────────────────────────────────────
def memory_budget(
    candidates: Sequence,
    *,
    mem_fraction: float = DEFAULT_MEM_FRACTION,
    mem_reserve_gb: float = DEFAULT_MEM_RESERVE_GB,
    expansion: float = DEFAULT_EXPANSION,
    sizes: Optional[Sequence[int]] = None,
) -> Tuple[int, str]:
    """How many of ``candidates`` can be held in RAM at once, right now.

    Packs greedily from the largest recording down, rather than assuming the
    mean: a single subject can mix a 517 MB and a 1 MB recording, and a
    mean-based estimate is optimistic exactly when it matters.

    Returns ``(count, explanation)``. ``count`` is never below 1 - the guard
    limits concurrency, it never refuses to make progress.
    """
    n = len(candidates)
    if n == 0:
        return 1, "no candidates"

    avail = memory_available_bytes()
    if avail is None:
        return n, "memory unknown (psutil unavailable) - not limiting"

    usable = max(0.0, avail - mem_reserve_gb * _GB) * float(mem_fraction)
    if sizes is None:
        sizes = [recording_size_bytes(c) for c in candidates]
    known = sorted((s for s in sizes if s > 0), reverse=True)
    if not known:
        return n, "sizes unknown - not limiting"

    # Unmeasurable files are charged the median known size rather than zero.
    median = known[len(known) // 2]
    full = sorted((s if s > 0 else median for s in sizes), reverse=True)

    total = 0.0
    take = 0
    for s in full:
        cost = s * float(expansion)
        if take >= 1 and total + cost > usable:
            break
        total += cost
        take += 1

    take = max(1, min(take, n))
    return take, (
        f"{avail / _GB:.0f} GB available, reserve {mem_reserve_gb:.0f} GB, "
        f"fraction {mem_fraction:.2f}, largest {full[0] / _GB:.2f} GB "
        f"x{expansion:.1f} -> {take} of {n}"
    )


def effective_jobs(
    requested_jobs: int,
    candidates: Sequence,
    *,
    mem_fraction: float = DEFAULT_MEM_FRACTION,
    mem_reserve_gb: float = DEFAULT_MEM_RESERVE_GB,
    expansion: float = DEFAULT_EXPANSION,
) -> Tuple[int, Optional[str]]:
    """Clamp a requested worker count to what memory allows.

    Returns ``(jobs, note)``; ``note`` is None when memory was not the limit, so
    callers only log when the guard actually did something.
    """
    if requested_jobs in (0, 1):
        return requested_jobs, None
    cpu = os.cpu_count() or 1
    want = cpu if requested_jobs < 0 else requested_jobs   # -1 means "all cores"

    allowed, why = memory_budget(
        candidates, mem_fraction=mem_fraction,
        mem_reserve_gb=mem_reserve_gb, expansion=expansion)
    # memory_budget caps at len(candidates); when everything fits, memory was
    # never the constraint and the guard should not claim credit for a
    # reduction that simply reflects how many subjects exist.
    if allowed >= len(candidates) or allowed >= want:
        return requested_jobs, None
    return allowed, (
        f"n_jobs {want} reduced to {allowed} by the memory guard ({why})")


