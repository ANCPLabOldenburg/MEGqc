"""Command-line helper to run dataset-level QA group plotting."""

import argparse

from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots
from meg_qc.plotting.meg_qc_group_plots import make_group_plots_meg_qc


def get_group_plots() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MEGqc dataset-level QA plotting: --inputdata <BIDS ds> "
            "[--derivatives_output <folder>]"
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
            "A subfolder named after the dataset will be used automatically."
        ),
    )
    parser.add_argument(
        "-njobs",
        "--njobs",
        type=int,
        default=1,
        required=False,
        help="Number of parallel workers for subject-level loading (1=sequential, -1=all cores).",
    )
    args = parser.parse_args()

    data_directory = args.inputdata
    derivatives_base = args.derivatives_output

    _, derivatives_root = resolve_output_roots(data_directory, derivatives_base)
    print(f"___MEGqc___: Reading derivatives from: {derivatives_root}")

    out = make_group_plots_meg_qc(
        data_directory,
        derivatives_base=derivatives_base,
        n_jobs=args.njobs,
    )
    if not out:
        print("___MEGqc___: No group QA reports were generated.")


if __name__ == "__main__":
    get_group_plots()
