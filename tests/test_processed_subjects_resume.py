"""Subject-level skip must not swallow a partially written subject.

With derivatives written per recording, a subject's calculation folder is
non-empty as soon as its *first* recording lands. The old check called that
"processed", so ``--processed_subjects_policy skip`` would have skipped the
remaining runs. These tests pin the corrected behaviour, including the fallback
that keeps results produced before the progress record from being recomputed.
"""
from __future__ import annotations

import os

from meg_qc.calculation.meg_qc_pipeline import _already_processed_subjects
from meg_qc.calculation.recording_progress import RecordingProgress


def _output_file(root, sub, modality="meg"):
    """Simulate derivatives on disk for one subject."""
    d = os.path.join(root, "calculation", modality, f"sub-{sub}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"sub-{sub}_desc-STDs_meg.tsv"), "w") as fh:
        fh.write("x\n")


def test_partially_written_subject_is_not_skipped(tmp_path):
    root = str(tmp_path)
    _output_file(root, "01")                       # first recording landed
    RecordingProgress(root, "01").record("run-01.fif", "ok")   # ...and only that

    # Output exists, but the subject is not finished: it must be revisited so
    # its remaining recordings are processed.
    assert _already_processed_subjects(root) == set()


def test_completed_subject_is_skipped(tmp_path):
    root = str(tmp_path)
    _output_file(root, "01")
    p = RecordingProgress(root, "01")
    p.record("run-01.fif", "ok")
    p.mark_complete()

    assert _already_processed_subjects(root) == {"01"}


def test_legacy_results_without_progress_are_still_skipped(tmp_path):
    """835 GB of existing derivatives predate the record and must not recompute."""
    root = str(tmp_path)
    _output_file(root, "07")                       # no .progress entry at all

    assert _already_processed_subjects(root) == {"07"}


def test_legacy_and_new_subjects_coexist(tmp_path):
    root = str(tmp_path)
    _output_file(root, "01")                       # legacy: no manifest -> done
    _output_file(root, "02")                       # new, partial -> not done
    RecordingProgress(root, "02").record("run-01.fif", "ok")
    _output_file(root, "03")                       # new, complete -> done
    p3 = RecordingProgress(root, "03")
    p3.record("run-01.fif", "ok")
    p3.mark_complete()

    assert _already_processed_subjects(root) == {"01", "03"}


def test_missing_root_is_empty(tmp_path):
    assert _already_processed_subjects(str(tmp_path / "nope")) == set()


def test_eeg_and_meg_modalities_both_counted(tmp_path):
    root = str(tmp_path)
    _output_file(root, "01", modality="eeg")
    assert _already_processed_subjects(root) == {"01"}
