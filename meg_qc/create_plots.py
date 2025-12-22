"""Command-line helper to run the plotting module."""

import argparse

from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc


def get_plots() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MEGqc plotting module: --inputdata <BIDS ds> [--derivatives_output <folder>]"
        )
    )
    parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help="Path to the root of your BIDS MEG dataset",
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional folder to store derivatives outside the BIDS dataset. "
            "A subfolder named after the dataset will be created automatically."
        ),
    )
    args = parser.parse_args()

    data_directory = args.inputdata
    derivatives_base = args.derivatives_output

    # Mirror the calculation pipeline: resolve the concrete derivatives root and
    # log it so users can verify exactly where plotting reads results from.
    _, derivatives_root = resolve_output_roots(data_directory, derivatives_base)
    print(f"___MEGqc___: Reading derivatives from: {derivatives_root}")

    make_plots_meg_qc(data_directory, derivatives_base=derivatives_base)


if __name__ == "__main__":
    get_plots()
