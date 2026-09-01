"""Per-recording progress record, so resume and error reporting work per run.

MEEGqc used to write a subject's derivatives once, after every one of its
recordings had been computed. Resume was therefore subject-granular:
``_already_processed_subjects`` called a subject done as soon as its calculation
folder held at least one file. That was reasonable when a subject carried a
handful of runs. It is not reasonable at 50+ runs per subject, where killing a
run discards hours of finished work.

Once derivatives are written per recording, the "at least one file" test becomes
actively wrong -- a subject interrupted after its first recording looks complete,
and ``--processed_subjects_policy skip`` would silently skip the rest. This
module supplies the record that replaces it.

Two things are deliberately folded into one file:

* **resume** -- which recordings already have derivatives on disk, and
* **error logging** -- what went wrong with each, written as it happens rather
  than accumulated in memory until the subject ends.

Keeping them together is what stops the log and the disk disagreeing. Under
per-recording writing a subject can die after writing 40 of 50 recordings; a
subject-level log would call that a total failure while 40 valid results sit on
disk. Here the log *is* derived from what was written.

Concurrency: recordings of one subject are computed by threads in a single
process, and each subject has its own file, so an in-process lock is sufficient.
No file locking is used, which keeps the behaviour identical on Linux, Windows
and macOS.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

#: Statuses that mean "derivatives for this recording are on disk".
WRITTEN_STATUSES = frozenset({"ok", "partial"})

#: Reserved recording id marking "every recording of this subject is done".
SUBJECT_COMPLETE = "__subject_complete__"


def recording_id(path: str) -> str:
    """Stable identifier for a recording.

    The raw file's base name is unique within a subject and survives moves of
    the dataset root, which a full path would not.  CTF recordings are
    directories (``...-meg.ds``); their name works the same way.
    """
    return os.path.basename(str(path).rstrip(os.sep).rstrip('/'))


class RecordingProgress:
    """Append-only record of which recordings of one subject are done.

    Stored as JSON Lines at ``<megqc_root>/.progress/sub-<label>.jsonl`` -- one
    line per recording, appended the moment that recording's derivatives reach
    disk.  Line-per-entry (rather than one rewritten JSON document) means an
    interrupted run leaves every completed entry intact.

    The directory sits beside ``calculation/`` rather than inside it because
    ``_already_processed_subjects`` treats every child of ``calculation/`` as a
    modality folder.
    """

    def __init__(self, megqc_root: str, sub: str):
        self.sub = str(sub)
        self.path = Path(megqc_root) / ".progress" / f"sub-{self.sub}.jsonl"
        self._lock = threading.Lock()
        self._entries: Optional[List[Dict]] = None

    # ------------------------------------------------------------------ read
    def _load(self) -> List[Dict]:
        if self._entries is not None:
            return self._entries
        entries: List[Dict] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        # A line torn by a kill mid-write. Everything before it
                        # is still good, so drop just this one rather than
                        # discarding the whole record.
                        continue
        except FileNotFoundError:
            pass
        self._entries = entries
        return entries

    def completed(self) -> Set[str]:
        """Recordings whose derivatives were written.

        Failed and skipped recordings are deliberately excluded: a rerun after
        fixing the cause should retry them, which matches how a wholly failed
        subject behaved before.
        """
        return {
            e.get("recording")
            for e in self._load()
            if e.get("status") in WRITTEN_STATUSES and e.get("recording")
        }

    def errors(self) -> List[Dict]:
        """Every error recorded for this subject, in the order written."""
        out: List[Dict] = []
        for entry in self._load():
            out.extend(entry.get("errors") or [])
        return out

    def is_complete(self, expected: Iterable[str]) -> bool:
        """True when every expected recording has been written.

        Takes the expected ids rather than a count: recordings are processed
        in parallel and finish out of order, so "how many are done" says
        nothing about *which*.
        """
        expected = {recording_id(e) for e in expected}
        return bool(expected) and expected.issubset(self.completed())

    # ----------------------------------------------------------------- write
    def record(self, recording: str, status: str, errors: Iterable[Dict] = ()) -> None:
        """Append one recording's outcome and flush it to disk.

        Called with the write lock already held by the caller in the pipeline,
        but locks independently as well so the class is safe to use on its own.
        """
        entry = {
            "recording": recording_id(recording),
            "status": status,
            "errors": list(errors or []),
            "t": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        line = json.dumps(entry) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode plus an explicit flush: a killed run keeps every entry
            # already written. fsync would be firmer but costs a disk round trip
            # per recording for a file that is cheap to rebuild by rerunning.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
            if self._entries is not None:
                self._entries.append(entry)

    def mark_complete(self) -> None:
        """Record that every recording of this subject has been written.

        The per-recording lines answer "which recordings are done"; this answers
        "is the subject done", which is what the subject-level skip policy asks.
        Deriving the latter from the former would need the expected file list,
        which the policy check does not have.
        """
        self.record(SUBJECT_COMPLETE, "complete", ())

    def subject_complete(self) -> bool:
        """True when :meth:`mark_complete` has been recorded."""
        return any(
            e.get("recording") == SUBJECT_COMPLETE and e.get("status") == "complete"
            for e in self._load()
        )

    def exists(self) -> bool:
        """True when this subject has a progress record at all.

        Runs written before per-recording progress existed have none, so the
        caller can fall back to the old filesystem test for them rather than
        reprocessing results that are already on disk.
        """
        return self.path.is_file()

    def reset(self) -> None:
        """Forget all progress for this subject (``--processed_subjects_policy rerun``)."""
        with self._lock:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self._entries = []
