"""Recording-level progress: resume must be per run, not per subject.

Derivatives are written as each recording finishes, so the old test -- "this
subject's folder holds at least one file, therefore the subject is done" -- would
skip the remaining 49 runs of a 50-run subject killed after its first. These
tests pin the record that replaces it.
"""
from __future__ import annotations

import json
import os

import pytest

from meg_qc.calculation.recording_progress import (
    SUBJECT_COMPLETE,
    RecordingProgress,
    recording_id,
)


def test_recording_id_is_the_basename():
    assert recording_id("/data/ds/sub-1/meg/sub-1_run-01_meg.fif") == "sub-1_run-01_meg.fif"


def test_recording_id_handles_ctf_directories():
    """CTF recordings are directories; a trailing separator must not swallow the name."""
    assert recording_id("/data/ds/sub-1/meg/sub-1_run-01_meg.ds/") == "sub-1_run-01_meg.ds"


def test_written_recordings_are_remembered(tmp_path):
    p = RecordingProgress(str(tmp_path), "01")
    p.record("sub-01_run-01_meg.fif", "ok")
    p.record("sub-01_run-02_meg.fif", "partial", [{"metric": "PSD"}])

    fresh = RecordingProgress(str(tmp_path), "01")     # re-read from disk
    assert fresh.completed() == {"sub-01_run-01_meg.fif", "sub-01_run-02_meg.fif"}


def test_failed_and_skipped_recordings_are_retried(tmp_path):
    """A failure must not be mistaken for completed work, or it is never retried."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("ok.fif", "ok")
    p.record("broken.fif", "failed", [{"metric": "write_derivative"}])
    p.record("missing.fif", "skipped")

    assert RecordingProgress(str(tmp_path), "01").completed() == {"ok.fif"}


def test_completion_uses_ids_not_counts(tmp_path):
    """Recordings finish out of order under --hungry-job, so a count says nothing."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("run-03.fif", "ok")
    p.record("run-01.fif", "ok")

    expected = ["run-01.fif", "run-02.fif", "run-03.fif"]
    assert not p.is_complete(expected)          # two of three, despite two entries
    p.record("run-02.fif", "ok")
    assert p.is_complete(expected)


def test_errors_are_readable_without_rerunning(tmp_path):
    """Errors reach disk per recording rather than accumulating until the end."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "partial", [{"metric": "PSD", "error_type": "ValueError"}])
    p.record("b.fif", "ok")
    p.record("c.fif", "failed", [{"metric": "write_derivative"}])

    metrics = [e["metric"] for e in RecordingProgress(str(tmp_path), "01").errors()]
    assert metrics == ["PSD", "write_derivative"]


def test_a_torn_final_line_does_not_lose_earlier_entries(tmp_path):
    """A kill mid-write must cost at most the recording being written."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "ok")
    p.record("b.fif", "ok")
    with open(p.path, "a", encoding="utf-8") as fh:
        fh.write('{"recording": "c.fif", "stat')      # truncated by a kill

    assert RecordingProgress(str(tmp_path), "01").completed() == {"a.fif", "b.fif"}


def test_subject_complete_marker(tmp_path):
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "ok")
    assert not p.subject_complete()
    p.mark_complete()
    assert RecordingProgress(str(tmp_path), "01").subject_complete()


def test_complete_marker_is_not_mistaken_for_a_recording(tmp_path):
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "ok")
    p.mark_complete()
    # The marker uses a reserved id; it must not be reported as written data.
    assert SUBJECT_COMPLETE not in p.completed()


def test_reset_clears_progress(tmp_path):
    """--processed_subjects_policy rerun must not inherit the previous record."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "ok")
    p.reset()
    assert RecordingProgress(str(tmp_path), "01").completed() == set()
    assert not p.exists()


def test_exists_distinguishes_legacy_results(tmp_path):
    """Runs predating the record have none, and must not be recomputed."""
    assert not RecordingProgress(str(tmp_path), "01").exists()
    RecordingProgress(str(tmp_path), "01").record("a.fif", "ok")
    assert RecordingProgress(str(tmp_path), "01").exists()


def test_concurrent_records_do_not_lose_entries(tmp_path):
    """Recordings register from worker threads; no entry may be dropped."""
    import threading

    p = RecordingProgress(str(tmp_path), "01")
    names = [f"run-{i:03d}.fif" for i in range(60)]

    def write(name):
        p.record(name, "ok")

    threads = [threading.Thread(target=write, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert RecordingProgress(str(tmp_path), "01").completed() == set(names)


def test_progress_lives_outside_the_calculation_tree(tmp_path):
    """calculation/'s children are treated as modality folders by the skip check."""
    p = RecordingProgress(str(tmp_path), "01")
    p.record("a.fif", "ok")
    assert p.path.parent.name == ".progress"
    assert "calculation" not in p.path.parts
