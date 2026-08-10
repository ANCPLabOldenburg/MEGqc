"""Output path handling shared by calculation and plotting."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager

OUTPUT_LAYOUTS = ("bids", "literal")


def normalize_output_layout(output_layout: str) -> str:
    """Return a validated output layout name."""
    layout = str(output_layout or "bids").strip().lower()
    if layout not in OUTPUT_LAYOUTS:
        supported = ", ".join(OUTPUT_LAYOUTS)
        raise ValueError(
            f"Invalid output_layout '{output_layout}'. Supported values: {supported}."
        )
    return layout


def dataset_derivatives_output(
    derivatives_output: str | None,
    dataset_path: str,
    output_layout: str,
    *,
    multiple_datasets: bool,
) -> str | None:
    """Return the external output argument to use for one dataset.

    Literal output for a multi-dataset run gets one dataset-named folder per
    input. The existing BIDS layout already adds that folder in
    :func:`resolve_output_roots`.
    """
    layout = normalize_output_layout(output_layout)
    if derivatives_output is None:
        if layout == "literal":
            raise ValueError("output_layout='literal' requires derivatives_output.")
        return None
    if layout == "literal" and multiple_datasets:
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        return os.path.join(derivatives_output, dataset_name)
    return derivatives_output


def resolve_output_roots(
    dataset_path: str,
    external_derivatives_root: str | None,
    output_layout: str = "bids",
) -> tuple[str, str]:
    """Return the output root and the folder that contains ``MEEGqc``.

    ``bids`` preserves the existing behavior. With an external output, it
    writes to ``<output>/<dataset>/derivatives/MEEGqc``.

    ``literal`` treats the supplied output as the derivatives folder itself,
    so the result is ``<output>/MEEGqc``. Multi-dataset callers pass a
    dataset-specific output produced by :func:`dataset_derivatives_output`.
    """
    layout = normalize_output_layout(output_layout)
    if external_derivatives_root is None:
        if layout == "literal":
            raise ValueError("output_layout='literal' requires derivatives_output.")
        output_root = dataset_path
        derivatives_root = os.path.join(output_root, "derivatives")
    elif layout == "bids":
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        output_root = os.path.join(external_derivatives_root, dataset_name)
        derivatives_root = os.path.join(output_root, "derivatives")
    else:
        output_root = external_derivatives_root
        derivatives_root = output_root

    os.makedirs(derivatives_root, exist_ok=True)

    # External BIDS layout is loaded as a dataset during report generation.
    # Literal layout is intentionally just an output folder and should not be
    # made to look like a BIDS dataset.
    if external_derivatives_root is not None and layout == "bids":
        _ensure_dataset_description(output_root, dataset_path)

    return output_root, derivatives_root


def derivative_scope(output_layout: str, *segments: str) -> str:
    """Return the ANCPBIDS query scope for the selected layout."""
    prefix = ["derivatives"] if normalize_output_layout(output_layout) == "bids" else []
    return os.path.join(*prefix, "MEEGqc", *segments)


@contextmanager
def derivative_write_context(
    dataset,
    derivative,
    output_root: str,
    output_layout: str,
) -> Iterator[None]:
    """Temporarily point an ANCPBIDS derivative at its selected output root."""
    layout = normalize_output_layout(output_layout)
    original_base = getattr(dataset, "base_dir_", None)
    original_name = getattr(dataset, "name", None)
    original_parent = getattr(derivative, "parent_object_", None)

    dataset.base_dir_ = output_root
    if layout == "literal":
        # ANCPBIDS normally places derivatives below a fixed ``derivatives``
        # parent. Reparenting only for the write makes ``MEEGqc`` land directly
        # below output_root while leaving the in-memory dataset graph unchanged.
        derivative.parent_object_ = dataset
        dataset.name = os.path.basename(os.path.normpath(output_root))

    try:
        yield
    finally:
        derivative.parent_object_ = original_parent
        dataset.base_dir_ = original_base
        dataset.name = original_name


def _ensure_dataset_description(output_root: str, dataset_path: str) -> None:
    """Seed an external BIDS output root with dataset metadata."""
    destination = os.path.join(output_root, "dataset_description.json")
    if os.path.exists(destination):
        return

    source = os.path.join(dataset_path, "dataset_description.json")
    if os.path.exists(source):
        try:
            shutil.copy2(source, destination)
            return
        except OSError:
            pass

    stub = {
        "Name": os.path.basename(os.path.normpath(output_root)),
        "BIDSVersion": "1.8.0",
    }
    try:
        with open(destination, "w", encoding="utf-8") as file:
            json.dump(stub, file, indent=2)
    except OSError:
        pass
