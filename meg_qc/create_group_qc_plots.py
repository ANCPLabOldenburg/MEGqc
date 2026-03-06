"""Command-line helper to run dataset-level QC group plotting."""

from __future__ import annotations

import argparse

from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots
from meg_qc.plotting.meg_qc_group_qc_plots import (
    make_group_qc_plots_meg_qc,
    make_group_qc_plots_multi_meg_qc,
)


def get_group_qc_plots() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MEGqc dataset-level QC plotting from Global Quality Index TSV: "
            "--inputdata <BIDS ds1> [<BIDS ds2> ...] [--input_tsv <path>] [--attempt <int>] "
            "[--derivatives_output <folder>] [--output_report <html path>]"
        )
    )
    parser.add_argument(
        "--inputdata",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the root of your BIDS MEG dataset(s)",
    )
    parser.add_argument(
        "--input_tsv",
        type=str,
        required=False,
        help=(
            "Optional explicit Global_Quality_Index_attempt_*.tsv file. "
            "If omitted, latest attempt is selected automatically."
        ),
    )
    parser.add_argument(
        "--attempt",
        type=int,
        required=False,
        help=(
            "Optional attempt number to select input TSV automatically "
            "when --input_tsv is not provided."
        ),
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional folder to store derivatives outside the BIDS dataset. "
            "A subfolder named after the dataset is used automatically."
        ),
    )
    parser.add_argument(
        "--output_report",
        type=str,
        required=False,
        help="Optional explicit output HTML path for the QC group report.",
    )
    args = parser.parse_args()

    data_directory = args.inputdata
    derivatives_base = args.derivatives_output

    for ds in data_directory:
        _, derivatives_root = resolve_output_roots(ds, derivatives_base)
        print(f"___MEGqc___: Reading derivatives from: {derivatives_root}")

    if len(data_directory) == 1:
        out = make_group_qc_plots_meg_qc(
            dataset_path=data_directory[0],
            input_tsv=args.input_tsv,
            output_html=args.output_report,
            attempt=args.attempt,
            derivatives_base=derivatives_base,
        )
    else:
        if args.input_tsv:
            print("___MEGqc___: --input_tsv is ignored in multi-dataset mode (automatic attempt/latest selection per dataset).")
        out = make_group_qc_plots_multi_meg_qc(
            dataset_paths=data_directory,
            output_html=args.output_report,
            attempt=args.attempt,
            derivatives_base=derivatives_base,
        )
    if out is None:
        print("___MEGqc___: No QC group report was generated.")


if __name__ == "__main__":
    get_group_qc_plots()
