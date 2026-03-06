"""Command-line helper to run MEGqc multi-sample QA comparison plotting."""

from __future__ import annotations

import argparse

from meg_qc.plotting.meg_qc_multi_sample_group_plots import make_multi_sample_group_plots_meg_qc


def get_multi_sample_group_plots() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MEGqc multi-sample QA plotting: "
            "--inputdata <BIDS ds 1> <BIDS ds 2> [more ...] "
            "[--derivatives_output <folder>] [--output_report <html path>]"
        )
    )
    parser.add_argument(
        "--inputdata",
        nargs="+",
        type=str,
        required=True,
        help="Paths to two or more BIDS datasets to compare",
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional external derivatives parent folder used for all datasets. "
            "If omitted, each dataset's in-place derivatives are used."
        ),
    )
    parser.add_argument(
        "--output_report",
        type=str,
        required=False,
        help="Optional explicit output HTML path for the multi-sample report",
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

    dataset_paths = args.inputdata
    derivatives_base = args.derivatives_output
    derivatives_bases = [derivatives_base] * len(dataset_paths) if derivatives_base else None

    out = make_multi_sample_group_plots_meg_qc(
        dataset_paths=dataset_paths,
        derivatives_bases=derivatives_bases,
        output_report_path=args.output_report,
        n_jobs=args.njobs,
    )
    if not out:
        print("___MEGqc___: No multi-sample QA report was generated.")


if __name__ == "__main__":
    get_multi_sample_group_plots()
