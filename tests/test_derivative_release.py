"""The MEEGqc derivative node must not accumulate in the dataset graph.

``create_derivative`` ends with ``target_folder.folders.append(derivative)`` and
never looks for an existing node, so one call per subject leaves one ``MEEGqc``
folder behind each time. ``del derivative`` drops only the local name; the graph
keeps the node alive, and each node holds that subject's artifact tree including
the pandas objects captured by the ``content`` writer closures.

That is harmless under ``n_jobs >= 2`` (joblib's loky backend pickles a fresh
dataset per task) but not under ``n_jobs == 1``, where the sequential backend
runs every subject in-process against one shared object -- and ``--n_jobs``
defaults to 1.
"""
from __future__ import annotations

import gc
import weakref

import pytest

ancpbids = pytest.importorskip("ancpbids")

from meg_qc.calculation.meg_qc_pipeline import _release_derivative


class _Payload:
    """Weak-referenceable stand-in for the pandas object a writer closure holds."""
    __slots__ = ("rows", "__weakref__")

    def __init__(self):
        self.rows = list(range(1000))


def _minimal_bids(tmp_path):
    """Smallest tree ancpbids will load as a dataset."""
    (tmp_path / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.8.0"}'
    )
    sub = tmp_path / "sub-01" / "meg"
    sub.mkdir(parents=True)
    (sub / "sub-01_task-t_meg.fif").write_bytes(b"")
    return ancpbids.load_dataset(str(tmp_path))


def test_release_removes_the_node(tmp_path):
    ds = _minimal_bids(tmp_path)

    def meegqc_nodes():
        return sum(1 for f in ds.derivatives.folders if f.name == "MEEGqc")

    for _ in range(5):
        derivative = ds.create_derivative(name="MEEGqc")
        _release_derivative(ds, derivative)
        # One create + one release must leave nothing behind, however many
        # subjects the run processes.
        assert meegqc_nodes() == 0


def test_without_release_the_nodes_accumulate(tmp_path):
    """Pins the behaviour the fix exists for, so the fix cannot silently lapse."""
    ds = _minimal_bids(tmp_path)
    for _ in range(5):
        ds.create_derivative(name="MEEGqc")
    assert sum(1 for f in ds.derivatives.folders if f.name == "MEEGqc") == 5


def test_release_frees_artifact_payloads(tmp_path):
    """The point of the fix: artifact content must become collectable."""
    ds = _minimal_bids(tmp_path)
    refs = []

    for _ in range(3):
        derivative = ds.create_derivative(name="MEEGqc")
        folder = derivative.create_folder(name="calculation")
        payload = _Payload()                           # stands in for a DataFrame
        artifact = folder.create_artifact()
        artifact.add_entity("desc", "STDs")
        artifact.suffix = "meg"
        artifact.extension = ".tsv"
        artifact.content = lambda path, cont=payload: None   # captures payload
        refs.append(weakref.ref(payload))
        del payload, artifact, folder
        _release_derivative(ds, derivative)
        del derivative

    gc.collect()
    assert [r() for r in refs] == [None, None, None]


def test_release_is_safe_when_called_twice(tmp_path):
    """Runs in a ``finally``, so a second call after an error must not raise."""
    ds = _minimal_bids(tmp_path)
    derivative = ds.create_derivative(name="MEEGqc")
    _release_derivative(ds, derivative)
    _release_derivative(ds, derivative)   # must be a no-op, not ValueError
