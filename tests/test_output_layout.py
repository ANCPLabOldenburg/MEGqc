"""Tests for external MEEGqc output layouts."""

from __future__ import annotations

import json
from pathlib import Path

import ancpbids
import pytest

from meg_qc.output_paths import (
    dataset_derivatives_output,
    derivative_scope,
    derivative_write_context,
    resolve_output_roots,
)


def _make_dataset(path: Path, name: str = "source") -> Path:
    path.mkdir()
    (path / "dataset_description.json").write_text(
        json.dumps({"Name": name, "BIDSVersion": "1.8.0"}),
        encoding="utf-8",
    )
    return path


def _write_test_derivative(dataset_path: Path, output_root: Path, layout: str) -> None:
    dataset = ancpbids.load_dataset(str(dataset_path))
    derivative = dataset.create_derivative(name="MEEGqc")
    derivative.dataset_description.GeneratedBy.Name = "MEEGqc"

    folder = derivative.create_folder(name="calculation")
    artifact = folder.create_artifact()
    artifact.add_entity("desc", "layout")
    artifact.suffix = "test"
    artifact.extension = ".txt"
    artifact.content = lambda file_path: Path(file_path).write_text(
        "ok", encoding="utf-8"
    )

    original_base = dataset.base_dir_
    original_name = dataset.name
    original_parent = derivative.parent_object_
    with derivative_write_context(dataset, derivative, str(output_root), layout):
        ancpbids.write_derivative(dataset, derivative)

    assert dataset.base_dir_ == original_base
    assert dataset.name == original_name
    assert derivative.parent_object_ is original_parent


def test_default_output_stays_inside_dataset(tmp_path):
    dataset_path = _make_dataset(tmp_path / "dataset")

    output_root, derivatives_root = resolve_output_roots(
        str(dataset_path), None, output_layout="bids"
    )

    assert Path(output_root) == dataset_path
    assert Path(derivatives_root) == dataset_path / "derivatives"

    _write_test_derivative(dataset_path, Path(output_root), "bids")
    expected = dataset_path / "derivatives" / "MEEGqc" / "calculation"
    assert list(expected.glob("*_test.txt"))


def test_external_bids_layout_keeps_existing_tree(tmp_path):
    dataset_path = _make_dataset(tmp_path / "dataset", name="original")
    external_root = tmp_path / "external"

    output_root, derivatives_root = resolve_output_roots(
        str(dataset_path), str(external_root), output_layout="bids"
    )

    assert Path(output_root) == external_root / dataset_path.name
    assert Path(derivatives_root) == Path(output_root) / "derivatives"
    copied = json.loads(
        (Path(output_root) / "dataset_description.json").read_text(encoding="utf-8")
    )
    assert copied["Name"] == "original"

    _write_test_derivative(dataset_path, Path(output_root), "bids")
    expected = Path(derivatives_root) / "MEEGqc" / "calculation"
    assert list(expected.glob("*_test.txt"))


def test_literal_layout_uses_requested_folder(tmp_path):
    dataset_path = _make_dataset(tmp_path / "dataset")
    literal_root = tmp_path / "chosen-output"

    output_root, derivatives_root = resolve_output_roots(
        str(dataset_path), str(literal_root), output_layout="literal"
    )

    assert Path(output_root) == literal_root
    assert Path(derivatives_root) == literal_root
    assert not (literal_root / "dataset_description.json").exists()

    _write_test_derivative(dataset_path, literal_root, "literal")
    expected = literal_root / "MEEGqc" / "calculation"
    assert list(expected.glob("*_test.txt"))
    assert not (literal_root / "derivatives").exists()

    loaded = ancpbids.load_dataset(str(literal_root))
    files = loaded.query(
        scope=derivative_scope("literal", "calculation"),
        return_type="filename",
    )
    assert len(files) == 1
    assert Path(files[0]).name == "desc-layout_test.txt"


def test_literal_multi_dataset_outputs_are_separate(tmp_path):
    output_root = tmp_path / "external"
    first = _make_dataset(tmp_path / "dataset-one", name="first")
    second = _make_dataset(tmp_path / "dataset-two", name="second")

    first_output = dataset_derivatives_output(
        str(output_root), str(first), "literal", multiple_datasets=True
    )
    second_output = dataset_derivatives_output(
        str(output_root), str(second), "literal", multiple_datasets=True
    )

    assert Path(first_output) == output_root / "dataset-one"
    assert Path(second_output) == output_root / "dataset-two"
    assert first_output != second_output

    for dataset_path, dataset_output in (
        (first, first_output),
        (second, second_output),
    ):
        _, derivatives_root = resolve_output_roots(
            str(dataset_path), dataset_output, output_layout="literal"
        )
        _write_test_derivative(dataset_path, Path(derivatives_root), "literal")
        assert list(
            (Path(dataset_output) / "MEEGqc" / "calculation").glob("*_test.txt")
        )


def test_literal_layout_requires_external_output(tmp_path):
    with pytest.raises(ValueError, match="requires derivatives_output"):
        resolve_output_roots(str(tmp_path / "dataset"), None, "literal")


def test_unknown_output_layout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid output_layout"):
        resolve_output_roots(str(tmp_path / "dataset"), str(tmp_path), "flat")
