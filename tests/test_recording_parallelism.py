"""Work is parallelised per recording, not per subject.

Subject-level parallelism left workers idle whenever subjects had uneven
numbers of runs -- which is what the old --hungry-job flag patched over by
lending a subject its finished siblings' slots. Making the recording the unit
removes the problem rather than managing it, and these tests pin the properties
that depend on it.
"""
from __future__ import annotations

import os

import pytest

from meg_qc.calculation import meg_qc_pipeline as pipe
from meg_qc.calculation.recording_progress import RecordingProgress, recording_id


# ── the flag is gone as a behaviour ──────────────────────────────────────

def test_slot_machinery_is_gone():
    """The slot registry existed only to ration idle subject slots."""
    import meg_qc.calculation.resource_budget as rb
    for name in ("SlotRegistry", "inner_budget", "idle_slots", "_file_lock"):
        assert not hasattr(rb, name), f"{name} should have been removed"


def test_memory_guard_survives():
    """The guard is still there -- it just asks one question now."""
    import meg_qc.calculation.resource_budget as rb
    for name in ("effective_jobs", "memory_budget", "recording_size_bytes",
                 "memory_available_bytes"):
        assert hasattr(rb, name), f"{name} must stay"


def test_hungry_job_flag_is_gone():
    """It described subject-level scheduling that no longer exists."""
    import inspect
    assert "hungry_job" not in inspect.signature(pipe.make_derivative_meg_qc).parameters
    from meg_qc.test import run_calculation_dispatch
    assert "hungry_job" not in inspect.signature(run_calculation_dispatch).parameters


# ── worker-side caching ──────────────────────────────────────────────────

def test_worker_functions_are_module_level():
    """cloudpickle ships module-level functions by reference, so a worker keeps
    its globals between tasks. Nested functions would be sent by value and the
    dataset cache would reset on every task."""
    for fn in (pipe.process_one_recording, pipe.process_one_recording_safe,
               pipe._compute_one_recording, pipe._register_and_write,
               pipe._worker_dataset, pipe._worker_files):
        assert fn.__qualname__ == fn.__name__, (
            f"{fn.__name__} is nested ({fn.__qualname__}); it must be module level")


def test_dataset_is_loaded_once_per_worker(monkeypatch, tmp_path):
    """The dataset object is ~72 MB and ~3 s to serialise for ds004078; loading
    it per task would cost ~38 minutes on a 720-recording dataset."""
    calls = []

    def fake_load(path, *a, **k):
        calls.append(path)
        return f"dataset::{path}"

    monkeypatch.setattr(pipe.ancpbids, "load_dataset", fake_load)
    pipe._WORKER_DATASET_CACHE.clear()
    try:
        for _ in range(25):
            assert pipe._worker_dataset("/data/ds") == "dataset::/data/ds"
        assert len(calls) == 1, f"loaded {len(calls)} times, expected 1"
    finally:
        pipe._WORKER_DATASET_CACHE.clear()


def test_file_list_is_cached_per_subject(monkeypatch):
    calls = []

    def fake_files(sub, path, ds, m_or_g_chosen=None):
        calls.append(sub)
        return ([f"/d/sub-{sub}_run-0{i}_meg.fif" for i in range(3)], [{}, {}, {}])

    monkeypatch.setattr(pipe, "get_files_list", fake_files)
    monkeypatch.setattr(pipe, "_worker_dataset", lambda p: "ds")
    pipe._WORKER_FILES_CACHE.clear()
    try:
        for _ in range(10):
            pipe._worker_files("/d", "01", None)
            pipe._worker_files("/d", "02", None)
        assert calls == ["01", "02"], calls
    finally:
        pipe._WORKER_FILES_CACHE.clear()


# ── the work list ────────────────────────────────────────────────────────

def _build_work(subject_files, progress_root, policy="skip"):
    """Mirror of the dispatch loop's work-list construction."""
    work = []
    for sub, files in subject_files.items():
        prog = RecordingProgress(progress_root, sub)
        if policy == "rerun":
            prog.reset()
        done = prog.completed()
        for i, f in enumerate(files):
            if recording_id(f) not in done:
                work.append((sub, i, f))
    return work


def test_work_list_is_flat_across_subjects(tmp_path):
    """One queue for the whole dataset: a worker finishing a short recording
    takes the next one rather than waiting on its subject's slowest sibling."""
    subject_files = {
        "01": [f"/d/sub-01_run-{i:02d}_meg.fif" for i in range(50)],
        "02": [f"/d/sub-02_run-{i:02d}_meg.fif" for i in range(2)],
    }
    work = _build_work(subject_files, str(tmp_path))
    assert len(work) == 52
    assert {w[0] for w in work} == {"01", "02"}


def test_work_list_skips_already_written_recordings(tmp_path):
    files = [f"/d/sub-01_run-{i:02d}_meg.fif" for i in range(5)]
    p = RecordingProgress(str(tmp_path), "01")
    p.record(files[0], "ok")
    p.record(files[3], "partial", [{"metric": "PSD"}])

    work = _build_work({"01": files}, str(tmp_path))
    assert [w[2] for w in work] == [files[1], files[2], files[4]]


def test_failed_recordings_are_retried(tmp_path):
    files = [f"/d/sub-01_run-{i:02d}_meg.fif" for i in range(3)]
    p = RecordingProgress(str(tmp_path), "01")
    p.record(files[0], "ok")
    p.record(files[1], "failed", [{"metric": "write_derivative"}])
    p.record(files[2], "skipped")

    work = _build_work({"01": files}, str(tmp_path))
    assert [w[2] for w in work] == [files[1], files[2]]


def test_rerun_policy_reprocesses_everything(tmp_path):
    files = [f"/d/sub-01_run-{i:02d}_meg.fif" for i in range(4)]
    p = RecordingProgress(str(tmp_path), "01")
    for f in files:
        p.record(f, "ok")

    work = _build_work({"01": files}, str(tmp_path), policy="rerun")
    assert len(work) == 4


def test_index_mismatch_is_recovered_not_silently_wrong(monkeypatch, tmp_path):
    """The file index travels with the task; if the worker's listing differs,
    addressing by index would process the wrong recording."""
    files = ["/d/a_meg.fif", "/d/b_meg.fif", "/d/c_meg.fif"]
    monkeypatch.setattr(pipe, "_worker_files",
                        lambda p, s, m: (files, [{}, {}, {}]))
    monkeypatch.setattr(pipe, "_worker_dataset", lambda p: "ds")
    seen = {}

    def fake_compute(file_ind, data_file, **kw):
        seen["file"] = data_file
        return {"ok": False, "file_ind": file_ind,
                "skip": {"metric": "initial_processing", "file": data_file}}

    monkeypatch.setattr(pipe, "_compute_one_recording", fake_compute)

    # Index 0 in the parent, but the worker's list has it at 2.
    out = pipe.process_one_recording(
        sub="01", file_ind=0, expected_file="/d/c_meg.fif",
        dataset_path="/d", all_qc_params={}, internal_qc_params={},
        derivatives_root=str(tmp_path), output_root=str(tmp_path))
    assert seen["file"] == "/d/c_meg.fif", "addressed the wrong recording"
    assert out["status"] == "skipped"


def test_unknown_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(pipe, "_worker_files",
                        lambda p, s, m: (["/d/a_meg.fif"], [{}]))
    monkeypatch.setattr(pipe, "_worker_dataset", lambda p: "ds")
    out = pipe.process_one_recording(
        sub="01", file_ind=0, expected_file="/d/gone_meg.fif",
        dataset_path="/d", all_qc_params={}, internal_qc_params={},
        derivatives_root=str(tmp_path), output_root=str(tmp_path))
    assert out["status"] == "failed"
    assert out["errors"][0]["metric"] == "file_lookup"


def test_safe_wrapper_contains_a_crash(monkeypatch, tmp_path):
    """One bad recording must not take down the pool."""
    def boom(**kw):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(pipe, "process_one_recording", boom)
    out = pipe.process_one_recording_safe(
        sub="01", file_ind=0, expected_file="/d/a_meg.fif",
        dataset_path="/d", all_qc_params={}, internal_qc_params={},
        derivatives_root=str(tmp_path), output_root=str(tmp_path))
    assert out["status"] == "failed"
    assert "worker exploded" in out["errors"][0]["error_message"]
    # ...and it is recorded, so a rerun retries that recording.
    assert RecordingProgress(str(tmp_path), "01").completed() == set()
