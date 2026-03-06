import argparse
import os
import shutil
import time
import datetime as dt
from typing import Callable, Dict, List, Optional, Sequence, Union


def hello_world():
    """
    Simple example function that prints the --subs argument from the command line.
    Not directly related to MEG QC, but provided as an example.
    """
    dataset_path_parser = argparse.ArgumentParser(description="parser for string to print")
    dataset_path_parser.add_argument("--subs", nargs="+", type=str, required=False, help="path to config file")
    args = dataset_path_parser.parse_args()
    print(args.subs)


def _timestamp_analysis_id() -> str:
    """Create a compact profile ID shared across multi-dataset runs."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


PROFILE_MODE_HELP = (
    "Analysis profile routing mode.\n"
    "  legacy: use classic path derivatives/Meg_QC (no profile).\n"
    "  new: create/use a profile under derivatives/Meg_QC/profiles/<analysis_id>.\n"
    "       If --analysis_id is omitted during calculation, one is auto-generated.\n"
    "       For plotting/GQI, 'new' is read-only only when --analysis_id is provided\n"
    "       (internally mapped to reuse).\n"
    "  reuse: use one explicit existing profile (--analysis_id required).\n"
    "  latest: resolve the most recently modified profile automatically."
)

PROFILE_ID_HELP = (
    "Analysis profile identifier.\n"
    "Required for analysis_mode='reuse'.\n"
    "Optional for analysis_mode='new' during calculation (auto-generated if omitted).\n"
    "Ignored in legacy mode."
)


def _megqc_installation_dir() -> str:
    return os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))


def _default_config_paths(install_dir: str) -> Dict[str, str]:
    internal = os.path.join(install_dir, "settings", "settings_internal.ini")
    user = os.path.join(install_dir, "settings", "settings.ini")
    return {"internal": os.path.normpath(internal), "user": os.path.normpath(user)}


def _normalize_dataset_paths(ds_paths: Union[Sequence[str], str]) -> List[str]:
    if isinstance(ds_paths, str):
        return [ds_paths]
    return [str(p) for p in ds_paths]


def _resolve_sub_list(subs: Optional[Sequence[str]]) -> Union[str, List[str]]:
    if not subs:
        return "all"
    cleaned = [s.strip() for s in subs if str(s).strip()]
    return cleaned if cleaned else "all"


def _parse_subs_token(token: str) -> Union[str, List[str]]:
    value = str(token).strip()
    if not value or value.lower() == "all":
        return "all"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts if parts else "all"


def _parse_subs_per_dataset_entries(
    entries: Optional[Sequence[str]],
) -> Optional[Dict[str, Union[str, List[str]]]]:
    if not entries:
        return None
    parsed: Dict[str, Union[str, List[str]]] = {}
    for raw in entries:
        if "::" not in str(raw):
            raise ValueError(
                "Each --subs_per_dataset entry must follow "
                "<dataset_path>::<all|sub1,sub2,...>"
            )
        ds_part, subs_part = str(raw).split("::", 1)
        ds_path = ds_part.strip()
        if not ds_path:
            raise ValueError("Dataset path in --subs_per_dataset cannot be empty.")
        parsed[os.path.normpath(ds_path)] = _parse_subs_token(subs_part)
    return parsed or None


def _parse_config_per_dataset_entries(
    entries: Optional[Sequence[str]],
) -> Optional[Dict[str, str]]:
    if not entries:
        return None
    parsed: Dict[str, str] = {}
    for raw in entries:
        if "::" not in str(raw):
            raise ValueError(
                "Each --config_per_dataset entry must follow "
                "<dataset_path>::<path/to/settings.ini>"
            )
        ds_part, cfg_part = str(raw).split("::", 1)
        ds_path = os.path.normpath(ds_part.strip())
        cfg_path = cfg_part.strip()
        if not ds_path:
            raise ValueError("Dataset path in --config_per_dataset cannot be empty.")
        if not cfg_path:
            raise ValueError("Config path in --config_per_dataset cannot be empty.")
        parsed[ds_path] = cfg_path
    return parsed or None


def _resolve_dataset_config_path(
    dataset_path: str,
    default_config_file_path: str,
    global_config_file_path: Optional[str] = None,
    config_per_dataset: Optional[Dict[str, str]] = None,
) -> str:
    if config_per_dataset:
        if dataset_path in config_per_dataset:
            return config_per_dataset[dataset_path]
        norm_key = os.path.normpath(dataset_path)
        if norm_key in config_per_dataset:
            return config_per_dataset[norm_key]
    if global_config_file_path:
        return global_config_file_path
    return default_config_file_path


def _resolve_dataset_sub_list(
    dataset_path: str,
    default_sub_list: Union[str, List[str]],
    dataset_subs: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Union[str, List[str]]:
    if not dataset_subs:
        return default_sub_list
    if dataset_path in dataset_subs:
        return dataset_subs[dataset_path]
    return dataset_subs.get(os.path.normpath(dataset_path), default_sub_list)


def _resolve_dataset_njobs(
    dataset_path: str,
    default_njobs: int,
    dataset_njobs: Optional[Dict[str, int]] = None,
) -> int:
    if not dataset_njobs:
        return int(default_njobs)
    if dataset_path in dataset_njobs:
        return int(dataset_njobs[dataset_path])
    norm_map = {os.path.normpath(k): int(v) for k, v in dataset_njobs.items()}
    return int(norm_map.get(os.path.normpath(dataset_path), default_njobs))


def run_calculation_dispatch(
    dataset_paths: Union[Sequence[str], str],
    config_file_path: str,
    internal_config_file_path: str,
    sub_list: Union[str, List[str]] = "all",
    n_jobs: int = 1,
    derivatives_output: Optional[str] = None,
    dataset_njobs: Optional[Dict[str, int]] = None,
    dataset_subs: Optional[Dict[str, Union[str, List[str]]]] = None,
    global_config_file_path: Optional[str] = None,
    config_per_dataset: Optional[Dict[str, str]] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
    existing_config_policy: str = "provided",
    processed_subjects_policy: str = "skip",
    interactive_prompts: bool = False,
    keep_temp_on_error: bool = False,
    logger: Callable[[str], None] = print,
) -> None:
    """
    Shared MEGqc calculation dispatcher used by CLI and GUI.

    Runs one or multiple datasets sequentially. Subject-level parallelism remains
    controlled by ``n_jobs`` (or ``dataset_njobs`` overrides when provided).
    """
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

    ds_list = _normalize_dataset_paths(dataset_paths)
    if not ds_list:
        raise ValueError("No dataset paths provided for calculation.")
    if analysis_mode == "legacy":
        logger(
            "[Calculation] analysis_mode='legacy' writes into derivatives/Meg_QC. "
            "Use profile mode (new/reuse/latest) for sensitivity-analysis runs."
        )

    # Share one generated profile ID across datasets for analysis_mode='new'.
    calc_analysis_id = analysis_id
    if analysis_mode == "new" and not calc_analysis_id:
        calc_analysis_id = _timestamp_analysis_id()
        logger(f"[Calculation] Generated shared analysis_id: {calc_analysis_id}")

    total_start = time.time()
    for idx, dataset_path in enumerate(ds_list, start=1):
        used_njobs = _resolve_dataset_njobs(dataset_path, n_jobs, dataset_njobs)
        used_subs = _resolve_dataset_sub_list(dataset_path, sub_list, dataset_subs)
        used_config = _resolve_dataset_config_path(
            dataset_path=dataset_path,
            default_config_file_path=config_file_path,
            global_config_file_path=global_config_file_path,
            config_per_dataset=config_per_dataset,
        )
        subs_label = "all" if used_subs == "all" else ",".join(used_subs)
        logger(
            f"[Calculation] Dataset {idx}/{len(ds_list)} started: {dataset_path} "
            f"(n_jobs={used_njobs}, subs={subs_label}, config={used_config})"
        )
        ds_start = time.time()
        make_derivative_meg_qc(
            default_config_file_path=used_config,
            internal_config_file_path=internal_config_file_path,
            ds_paths=dataset_path,
            sub_list=used_subs,
            n_jobs=used_njobs,
            derivatives_base=derivatives_output,
            analysis_mode=analysis_mode,
            analysis_id=calc_analysis_id,
            existing_config_policy=existing_config_policy,
            processed_subjects_policy=processed_subjects_policy,
            interactive_prompts=interactive_prompts,
            keep_temp_on_error=keep_temp_on_error,
        )
        ds_elapsed = time.time() - ds_start
        logger(
            f"[Calculation] Dataset {idx}/{len(ds_list)} finished in "
            f"{ds_elapsed:.2f}s: {dataset_path}"
        )

    total_elapsed = time.time() - total_start
    logger(f"[Calculation] All datasets finished in {total_elapsed:.2f}s.")


def _normalize_plot_modes(
    *,
    qa_subject: bool = False,
    qa_group: bool = False,
    qa_multisample: bool = False,
    qc_group: bool = False,
    qc_multisample: bool = False,
    qa_all: bool = False,
    qc_all: bool = False,
    all_modes: bool = False,
) -> Dict[str, bool]:
    if qa_all or all_modes:
        qa_subject = True
        qa_group = True
        qa_multisample = True
    if qc_all or all_modes:
        qc_group = True
        qc_multisample = True
    if not any([qa_subject, qa_group, qa_multisample, qc_group, qc_multisample]):
        qa_subject = True
    return {
        "qa_subject": bool(qa_subject),
        "qa_group": bool(qa_group),
        "qa_multisample": bool(qa_multisample),
        "qc_group": bool(qc_group),
        "qc_multisample": bool(qc_multisample),
    }


def validate_plot_request(
    *,
    dataset_paths: Sequence[str],
    qa_subject: bool,
    qa_group: bool,
    qa_multisample: bool,
    qc_group: bool,
    qc_multisample: bool,
    input_tsv: Optional[str],
    output_report: Optional[str],
) -> None:
    n_datasets = len(dataset_paths)
    if n_datasets < 1:
        raise ValueError("At least one dataset is required in --inputdata.")
    if qa_multisample and n_datasets < 2:
        raise ValueError("--qa-multisample requires at least two datasets in --inputdata.")
    if qc_multisample and n_datasets < 2:
        raise ValueError("--qc-multisample requires at least two datasets in --inputdata.")

    if input_tsv:
        if not qc_group:
            raise ValueError("--input_tsv is only valid with --qc-group.")
        if n_datasets != 1:
            raise ValueError("--input_tsv is only valid with --qc-group and a single dataset.")
        if qc_multisample:
            raise ValueError("--input_tsv cannot be combined with --qc-multisample.")

    if output_report:
        enabled_modes = [
            ("qa_subject", qa_subject),
            ("qa_group", qa_group),
            ("qa_multisample", qa_multisample),
            ("qc_group", qc_group),
            ("qc_multisample", qc_multisample),
        ]
        enabled_count = sum(1 for _, on in enabled_modes if on)
        if enabled_count != 1:
            raise ValueError(
                "--output_report is only allowed when exactly one plotting mode is selected."
            )
        if qa_subject or qa_group:
            raise ValueError(
                "--output_report is not used by --qa-subject/--qa-group. "
                "Use it with --qa-multisample, --qc-group, or --qc-multisample."
            )
        if qc_group and n_datasets != 1:
            raise ValueError(
                "--output_report with --qc-group is only supported for a single dataset."
            )


def run_plotting_dispatch(
    dataset_paths: Union[Sequence[str], str],
    derivatives_output: Optional[str] = None,
    output_report: Optional[str] = None,
    attempt: Optional[int] = None,
    input_tsv: Optional[str] = None,
    njobs: int = 1,
    qa_subject: bool = False,
    qa_group: bool = False,
    qa_multisample: bool = False,
    qc_group: bool = False,
    qc_multisample: bool = False,
    qa_all: bool = False,
    qc_all: bool = False,
    all_modes: bool = False,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
    logger: Callable[[str], None] = print,
) -> Dict[str, bool]:
    """
    Shared plotting dispatcher used by CLI and GUI.

    Returns the normalized mode dictionary used for execution.
    """
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
    from meg_qc.plotting.meg_qc_group_plots import make_group_plots_meg_qc
    from meg_qc.plotting.meg_qc_multi_sample_group_plots import make_multi_sample_group_plots_meg_qc
    from meg_qc.plotting.meg_qc_group_qc_plots import (
        make_group_qc_plots_meg_qc,
        make_group_qc_plots_multi_meg_qc,
    )
    from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root

    ds_list = _normalize_dataset_paths(dataset_paths)
    effective_mode = analysis_mode
    effective_id = analysis_id
    if effective_mode == "legacy":
        logger(
            "[Plotting] analysis_mode='legacy' selected. "
            "Profile mode is recommended for reproducible comparative runs."
        )
    # "new" is calculation/write mode. For plotting reads, map to reuse when
    # an explicit profile ID is provided.
    if effective_mode == "new":
        if effective_id:
            effective_mode = "reuse"
            logger("[Plotting] analysis_mode='new' mapped to 'reuse' for read-only plotting.")
        else:
            raise ValueError(
                "analysis_mode='new' requires --analysis_id for plotting. "
                "Use analysis_mode='reuse'|'latest'|'legacy'."
            )
    modes = _normalize_plot_modes(
        qa_subject=qa_subject,
        qa_group=qa_group,
        qa_multisample=qa_multisample,
        qc_group=qc_group,
        qc_multisample=qc_multisample,
        qa_all=qa_all,
        qc_all=qc_all,
        all_modes=all_modes,
    )
    # "all"/preset modes should run everything that is valid for current input.
    # For single-dataset runs, multisample modes are automatically disabled.
    if len(ds_list) < 2:
        if modes["qa_multisample"]:
            logger("Skipping QA multisample mode: at least two datasets are required.")
        if modes["qc_multisample"]:
            logger("Skipping QC multisample mode: at least two datasets are required.")
        modes["qa_multisample"] = False
        modes["qc_multisample"] = False
    validate_plot_request(
        dataset_paths=ds_list,
        qa_subject=modes["qa_subject"],
        qa_group=modes["qa_group"],
        qa_multisample=modes["qa_multisample"],
        qc_group=modes["qc_group"],
        qc_multisample=modes["qc_multisample"],
        input_tsv=input_tsv,
        output_report=output_report,
    )

    if not any([qa_subject, qa_group, qa_multisample, qc_group, qc_multisample, qa_all, qc_all, all_modes]):
        logger("No QA/QC plotting mode selected; defaulting to --qa-subject.")

    for ds in ds_list:
        _, derivatives_root, megqc_root, resolved_analysis_id, _segments = resolve_analysis_root(
            dataset_path=ds,
            external_derivatives_root=derivatives_output,
            analysis_mode=effective_mode,
            analysis_id=effective_id,
            create_if_missing=True,
        )
        logger(
            "Using derivatives from: "
            f"{derivatives_root} (analysis_root={megqc_root}, "
            f"analysis_id={resolved_analysis_id or 'legacy'})"
        )

    if modes["qa_subject"]:
        for ds in ds_list:
            logger(f"Running QA subject plotting for dataset: {ds}")
            make_plots_meg_qc(
                ds,
                n_jobs=njobs,
                derivatives_base=derivatives_output,
                analysis_mode=effective_mode,
                analysis_id=effective_id,
            )

    if modes["qa_group"]:
        for ds in ds_list:
            logger(f"Running QA group plotting for dataset: {ds}")
            make_group_plots_meg_qc(
                ds,
                derivatives_base=derivatives_output,
                n_jobs=njobs,
                analysis_mode=effective_mode,
                analysis_id=effective_id,
            )

    if modes["qa_multisample"]:
        logger("Running QA multisample plotting...")
        derivatives_bases = [derivatives_output] * len(ds_list) if derivatives_output else None
        make_multi_sample_group_plots_meg_qc(
            dataset_paths=ds_list,
            derivatives_bases=derivatives_bases,
            output_report_path=output_report,
            n_jobs=njobs,
            analysis_mode=effective_mode,
            analysis_id=effective_id,
        )

    if modes["qc_group"]:
        if len(ds_list) == 1:
            logger(f"Running QC group plotting for dataset: {ds_list[0]}")
            make_group_qc_plots_meg_qc(
                dataset_path=ds_list[0],
                input_tsv=input_tsv,
                output_html=output_report,
                attempt=attempt,
                derivatives_base=derivatives_output,
                analysis_mode=effective_mode,
                analysis_id=effective_id,
            )
        else:
            if input_tsv:
                raise ValueError(
                    "--input_tsv is not supported when --qc-group is used with multiple datasets."
                )
            for ds in ds_list:
                logger(f"Running QC group plotting for dataset: {ds}")
                make_group_qc_plots_meg_qc(
                    dataset_path=ds,
                    input_tsv=None,
                    output_html=None,
                    attempt=attempt,
                    derivatives_base=derivatives_output,
                    analysis_mode=effective_mode,
                    analysis_id=effective_id,
                )

    if modes["qc_multisample"]:
        logger("Running QC multisample plotting...")
        make_group_qc_plots_multi_meg_qc(
            dataset_paths=ds_list,
            output_html=output_report,
            attempt=attempt,
            derivatives_base=derivatives_output,
            analysis_mode=effective_mode,
            analysis_id=effective_id,
        )

    return modes


def run_megqc():
    """
    Main entry point for launching the MEG QC calculation pipeline from CLI.
    Supports one or multiple datasets via --inputdata.
    """
    dataset_path_parser = argparse.ArgumentParser(
        description=(
            "Command-line argument parser for MEGqc calculation.\n"
            "--inputdata (required): one or more BIDS dataset paths.\n"
            "--config (optional): path to a config file.\n"
            "--subs (optional): list of subject IDs (defaults to all).\n"
            "--n_jobs (optional): subject-level parallel workers.\n"
            "--derivatives_output (optional): external derivatives parent folder."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    dataset_path_parser.add_argument(
        "--inputdata",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Path(s) to the root of one or more BIDS MEG datasets.\n"
            "Examples:\n"
            "  --inputdata /path/to/dataset\n"
            "  --inputdata /path/to/ds1 /path/to/ds2"
        ),
    )

    dataset_path_parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to one global INI config file for all datasets.\n"
            "Optional: if omitted, package default settings.ini is used."
        ),
    )
    dataset_path_parser.add_argument(
        "--config_per_dataset",
        nargs="+",
        type=str,
        required=False,
        help=(
            "Optional per-dataset config overrides.\n"
            "Format per entry: <dataset_path>::<path/to/settings.ini>\n"
            "Entries override --config for the matching dataset."
        ),
    )

    dataset_path_parser.add_argument(
        "--subs",
        nargs="+",
        type=str,
        required=False,
        help=(
            "List of subject IDs to process.\n"
            "Optional: if omitted, all subjects are processed."
        ),
    )
    dataset_path_parser.add_argument(
        "--subs_per_dataset",
        nargs="+",
        type=str,
        required=False,
        help=(
            "Optional per-dataset subject overrides.\n"
            "Format per entry: <dataset_path>::<all|sub1,sub2,...>\n"
            "Example:\n"
            "  --subs_per_dataset \\\n"
            "    /path/ds1::001,002 \\\n"
            "    /path/ds2::all\n"
            "Unlisted datasets fallback to --subs (or all)."
        ),
    )

    dataset_path_parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help=(
            "Number of subject-level parallel jobs.\n"
            "Default is 1. Use -1 to utilize all available CPU cores.\n"
            "\n"
            "⚠️ Recommendation based on system memory:\n"
            "  - 8 GB RAM → up to 1 parallel job\n"
            "  - 16 GB RAM → up to 2 parallel jobs\n"
            "  - 32 GB RAM → up to 6 parallel jobs\n"
            "  - 64 GB RAM → up to 16 parallel jobs\n"
            "  - 128 GB RAM → up to 30 parallel jobs"
        ),
    )

    dataset_path_parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional external parent folder for derivatives. "
            "Per-dataset subfolders are created automatically."
        ),
    )
    dataset_path_parser.add_argument(
        "--analysis_mode",
        type=str,
        choices=["legacy", "new", "reuse", "latest"],
        default="legacy",
        help=PROFILE_MODE_HELP,
    )
    dataset_path_parser.add_argument(
        "--analysis_id",
        type=str,
        required=False,
        help=PROFILE_ID_HELP,
    )
    dataset_path_parser.add_argument(
        "--existing_config_policy",
        type=str,
        choices=["provided", "latest_saved", "fail"],
        default="provided",
        help=(
            "Policy when settings snapshots already exist in selected profile: "
            "provided (default), latest_saved, fail."
        ),
    )
    dataset_path_parser.add_argument(
        "--processed_subjects_policy",
        type=str,
        choices=["skip", "rerun", "fail"],
        default="skip",
        help=(
            "Policy for subjects already processed in selected profile/config: "
            "skip (default), rerun, fail."
        ),
    )
    dataset_path_parser.add_argument(
        "--keep-temp-on-error",
        action="store_true",
        help=(
            "Keep intermediate .tmp files when a dataset run errors, useful "
            "for debugging failed preprocessing."
        ),
    )
    dataset_path_parser.add_argument(
        "--run-all",
        action="store_true",
        dest="run_all_reports",
        help=(
            "After calculation finishes, run all QA and QC plotting modes in one shot "
            "(equivalent to run-megqc-plotting --all)."
        ),
    )
    # Backward-compatible alias retained for existing scripts.
    dataset_path_parser.add_argument(
        "--run_all_reports",
        action="store_true",
        dest="run_all_reports",
        help=argparse.SUPPRESS,
    )
    # Reuse plotting-scope semantics for --run-all.
    dataset_path_parser.add_argument("--qa-subject", action="store_true", help="Run subject-level QA plotting after --run-all.")
    dataset_path_parser.add_argument("--qa-group", action="store_true", help="Run dataset-level QA group plotting after --run-all.")
    dataset_path_parser.add_argument("--qa-multisample", action="store_true", help="Run QA multisample plotting after --run-all.")
    dataset_path_parser.add_argument("--qa-all", action="store_true", help="Enable all QA plotting scopes after --run-all.")
    dataset_path_parser.add_argument("--qc-group", action="store_true", help="Run QC group plotting after --run-all.")
    dataset_path_parser.add_argument("--qc-multisample", action="store_true", help="Run QC multisample plotting after --run-all.")
    dataset_path_parser.add_argument("--qc-all", action="store_true", help="Enable all QC plotting scopes after --run-all.")
    dataset_path_parser.add_argument("--all", action="store_true", help="Enable all QA+QC plotting scopes after --run-all.")

    args = dataset_path_parser.parse_args()
    if args.analysis_mode == "reuse" and not args.analysis_id:
        dataset_path_parser.error("analysis_mode='reuse' requires --analysis_id.")

    install_dir = _megqc_installation_dir()
    cfg_paths = _default_config_paths(install_dir)

    print("MEG QC installation directory:", install_dir)
    print("Datasets to process:", args.inputdata)

    sub_list = _resolve_sub_list(args.subs)
    if sub_list != "all":
        print("Subjects to process:", sub_list)
    try:
        subs_per_dataset = _parse_subs_per_dataset_entries(args.subs_per_dataset)
        config_per_dataset = _parse_config_per_dataset_entries(args.config_per_dataset)
    except ValueError as exc:
        dataset_path_parser.error(str(exc))
        return
    if subs_per_dataset:
        print("Per-dataset subject overrides enabled:")
        for ds_key, ds_subs in subs_per_dataset.items():
            label = "all" if ds_subs == "all" else ",".join(ds_subs)
            print(f"  - {ds_key} -> {label}")

    config_file_path = cfg_paths["user"]
    global_config_file_path = args.config
    if global_config_file_path:
        print(f"Global config selected: {global_config_file_path}")
    else:
        print(f"Using package default config: {config_file_path}")
    if config_per_dataset:
        print("Per-dataset config overrides enabled:")
        for ds_key, cfg_path in config_per_dataset.items():
            print(f"  - {ds_key} -> {cfg_path}")

    if args.run_all_reports:
        # If no explicit scope flags are provided, execute all QA+QC scopes.
        requested_any_scope = any([
            args.qa_subject,
            args.qa_group,
            args.qa_multisample,
            args.qc_group,
            args.qc_multisample,
            args.qa_all,
            args.qc_all,
            args.all,
        ])
        print(f"Running MEG QC calculation + plotting with n_jobs={args.n_jobs}")
        run_all_dispatch(
            dataset_paths=args.inputdata,
            config_file_path=config_file_path,
            internal_config_file_path=cfg_paths["internal"],
            sub_list=sub_list,
            calc_n_jobs=args.n_jobs,
            plot_njobs=args.n_jobs,
            derivatives_output=args.derivatives_output,
            dataset_subs=subs_per_dataset,
            global_config_file_path=global_config_file_path,
            config_per_dataset=config_per_dataset,
            analysis_mode=args.analysis_mode,
            analysis_id=args.analysis_id,
            existing_config_policy=args.existing_config_policy,
            processed_subjects_policy=args.processed_subjects_policy,
            interactive_prompts=False,
            keep_temp_on_error=args.keep_temp_on_error,
            qa_subject=args.qa_subject,
            qa_group=args.qa_group,
            qa_multisample=args.qa_multisample,
            qc_group=args.qc_group,
            qc_multisample=args.qc_multisample,
            qa_all=args.qa_all,
            qc_all=args.qc_all,
            all_modes=(args.all or not requested_any_scope),
            logger=print,
        )
        return

    print(f"Running MEG QC calculation with n_jobs={args.n_jobs}")
    run_calculation_dispatch(
        dataset_paths=args.inputdata,
        config_file_path=config_file_path,
        internal_config_file_path=cfg_paths["internal"],
        sub_list=sub_list,
        n_jobs=args.n_jobs,
        derivatives_output=args.derivatives_output,
        dataset_subs=subs_per_dataset,
        global_config_file_path=global_config_file_path,
        config_per_dataset=config_per_dataset,
        analysis_mode=args.analysis_mode,
        analysis_id=args.analysis_id,
        existing_config_policy=args.existing_config_policy,
        processed_subjects_policy=args.processed_subjects_policy,
        interactive_prompts=False,
        keep_temp_on_error=args.keep_temp_on_error,
        logger=print,
    )

    for dataset_path in _normalize_dataset_paths(args.inputdata):
        print(f"Results are available under: {dataset_path}/derivatives/Meg_QC/calculation")


def get_config():
    """
    Copies the default config file (settings.ini) to the user-specified target directory.
    Allows the user to customize the config before running MEG QC.
    """
    target_directory_parser = argparse.ArgumentParser(
        description="parser for MEGqc get_config: "
                    "--target_directory (mandatory) path/to/directory to store the config"
    )
    target_directory_parser.add_argument(
        "--target_directory",
        type=str,
        required=True,
        help="Path to which the default MEG QC config file (settings.ini) will be copied."
    )
    target_directory_parser.add_argument(
        "--filename",
        type=str,
        required=False,
        default="settings.ini",
        help="Optional filename for the copied config (default: settings.ini).",
    )
    args = target_directory_parser.parse_args()
    destination_directory = os.path.join(args.target_directory, args.filename)
    print("Destination directory for config:", destination_directory)

    path_to_megqc_installation = _megqc_installation_dir()
    print("MEG QC installation directory:", path_to_megqc_installation)

    config_file_path = os.path.join(path_to_megqc_installation, "settings", "settings.ini")
    print("Source of default config file:", config_file_path)

    shutil.copy(config_file_path, destination_directory)
    print("The config file has been copied to " + destination_directory)


def get_plots():
    """
    Unified plotting entry point for QA/QC report generation.

    This command supports explicit QA and QC modes with one-shot execution.
    It can run subject-level, group-level, and multi-sample reports depending
    on provided flags and number of datasets.
    """
    dataset_path_parser = argparse.ArgumentParser(
        description=(
            "MEGqc plotting dispatcher for QA/QC reports.\n\n"
            "Use explicit flags to choose report scope:\n"
            "  QA flags: --qa-subject, --qa-group, --qa-multisample, --qa-all\n"
            "  QC flags: --qc-group, --qc-multisample, --qc-all\n"
            "  Combined: --all (runs QA+QC in one shot)\n\n"
            "Examples:\n"
            "  Subject QA (single dataset):\n"
            "    run-megqc-plotting --inputdata /path/ds --qa-subject\n\n"
            "  Group QA (single dataset):\n"
            "    run-megqc-plotting --inputdata /path/ds --qa-group\n\n"
            "  Multi-sample QA (2+ datasets):\n"
            "    run-megqc-plotting --inputdata /path/ds1 /path/ds2 --qa-multisample\n\n"
            "  Group QC with selected attempt:\n"
            "    run-megqc-plotting --inputdata /path/ds --qc-group --attempt 2\n\n"
            "  All QA+QC reports (one shot):\n"
            "    run-megqc-plotting --inputdata /path/ds1 /path/ds2 --all"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    dataset_path_parser.add_argument(
        "--inputdata",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to one or more BIDS MEG datasets.",
    )
    dataset_path_parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help=(
            "Optional external derivatives parent folder. For single-dataset "
            "modes, dataset-specific derivatives are resolved from this root."
        ),
    )
    dataset_path_parser.add_argument(
        "--analysis_mode",
        type=str,
        choices=["legacy", "new", "reuse", "latest"],
        default="legacy",
        help=PROFILE_MODE_HELP,
    )
    dataset_path_parser.add_argument(
        "--analysis_id",
        type=str,
        required=False,
        help=PROFILE_ID_HELP,
    )
    dataset_path_parser.add_argument(
        "--output_report",
        type=str,
        required=False,
        help=(
            "Optional explicit output HTML path. Allowed only when exactly one "
            "single-report mode is selected: --qa-multisample, --qc-group, "
            "or --qc-multisample."
        ),
    )
    dataset_path_parser.add_argument(
        "--attempt",
        type=int,
        required=False,
        help=(
            "QC only: attempt number used to select "
            "Global_Quality_Index_attempt_<n>.tsv when --input_tsv is not given."
        ),
    )
    dataset_path_parser.add_argument(
        "--input_tsv",
        type=str,
        required=False,
        help=(
            "QC only: explicit Global_Quality_Index_attempt_*.tsv path. "
            "Allowed only with --qc-group and a single dataset."
        ),
    )
    dataset_path_parser.add_argument(
        "-njobs",
        "--njobs",
        type=int,
        default=1,
        required=False,
        help="Parallel workers for supported QA plotting tasks (1=sequential, -1=all cores).",
    )
    dataset_path_parser.add_argument(
        "--qa-subject",
        action="store_true",
        help="Run subject-level QA plotting for each selected dataset.",
    )
    dataset_path_parser.add_argument(
        "--qa-group",
        action="store_true",
        help="Run dataset-level QA group report for each selected dataset.",
    )
    dataset_path_parser.add_argument(
        "--qa-multisample",
        action="store_true",
        help="Run one multi-sample QA report across selected datasets (requires >=2 datasets).",
    )
    dataset_path_parser.add_argument(
        "--qa-all",
        action="store_true",
        help="Run all QA modes valid for the provided datasets.",
    )
    dataset_path_parser.add_argument(
        "--qc-group",
        action="store_true",
        help=(
            "Run dataset-level QC report(s). For one dataset, uses --input_tsv "
            "or --attempt when provided; otherwise selects latest attempt."
        ),
    )
    dataset_path_parser.add_argument(
        "--qc-multisample",
        action="store_true",
        help="Run one multi-sample QC report across selected datasets (requires >=2 datasets).",
    )
    dataset_path_parser.add_argument(
        "--qc-all",
        action="store_true",
        help="Run all QC modes valid for the provided datasets.",
    )
    dataset_path_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all QA and QC modes valid for the provided datasets in one shot.",
    )
    args = dataset_path_parser.parse_args()

    try:
        run_plotting_dispatch(
            dataset_paths=args.inputdata,
            derivatives_output=args.derivatives_output,
            output_report=args.output_report,
            attempt=args.attempt,
            input_tsv=args.input_tsv,
            njobs=args.njobs,
            qa_subject=args.qa_subject,
            qa_group=args.qa_group,
            qa_multisample=args.qa_multisample,
            qc_group=args.qc_group,
            qc_multisample=args.qc_multisample,
            qa_all=args.qa_all,
            qc_all=args.qc_all,
            all_modes=args.all,
            analysis_mode=args.analysis_mode,
            analysis_id=args.analysis_id,
            logger=print,
        )
    except ValueError as exc:
        dataset_path_parser.error(str(exc))


def run_gqi_dispatch(
    dataset_paths: Union[Sequence[str], str],
    default_config_file_path: str,
    derivatives_output: Optional[str] = None,
    global_config_file_path: Optional[str] = None,
    config_per_dataset: Optional[Dict[str, str]] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
    logger: Callable[[str], None] = print,
) -> None:
    """Shared dispatcher for GQI regeneration over one or multiple datasets."""
    from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary
    from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root

    ds_list = _normalize_dataset_paths(dataset_paths)
    if not ds_list:
        raise ValueError("No dataset paths provided for GQI.")
    effective_mode = analysis_mode
    effective_id = analysis_id
    if effective_mode == "new":
        if effective_id:
            effective_mode = "reuse"
            logger("[GQI] analysis_mode='new' mapped to 'reuse' for read-only GQI regeneration.")
        else:
            raise ValueError(
                "analysis_mode='new' requires --analysis_id for GQI. "
                "Use analysis_mode='reuse'|'latest'|'legacy'."
            )

    total_start = time.time()
    for idx, dataset_path in enumerate(ds_list, start=1):
        cfg_path = _resolve_dataset_config_path(
            dataset_path=dataset_path,
            default_config_file_path=default_config_file_path,
            global_config_file_path=global_config_file_path,
            config_per_dataset=config_per_dataset,
        )
        _, _derivatives_root, megqc_root, resolved_analysis_id, _segments = resolve_analysis_root(
            dataset_path=dataset_path,
            external_derivatives_root=derivatives_output,
            analysis_mode=effective_mode,
            analysis_id=effective_id,
            create_if_missing=True,
        )
        logger(
            f"[GQI] Dataset {idx}/{len(ds_list)} started: {dataset_path} "
            f"(config={cfg_path}, analysis_id={resolved_analysis_id or 'legacy'})"
        )
        ds_start = time.time()
        generate_gqi_summary(dataset_path, megqc_root, cfg_path)
        ds_elapsed = time.time() - ds_start
        logger(
            f"[GQI] Dataset {idx}/{len(ds_list)} finished in {ds_elapsed:.2f}s: "
            f"{dataset_path}"
        )
    total_elapsed = time.time() - total_start
    logger(f"[GQI] All datasets finished in {total_elapsed:.2f}s.")


def run_all_dispatch(
    dataset_paths: Union[Sequence[str], str],
    config_file_path: str,
    internal_config_file_path: str,
    sub_list: Union[str, List[str]] = "all",
    calc_n_jobs: int = 1,
    plot_njobs: int = 1,
    derivatives_output: Optional[str] = None,
    dataset_njobs: Optional[Dict[str, int]] = None,
    dataset_subs: Optional[Dict[str, Union[str, List[str]]]] = None,
    global_config_file_path: Optional[str] = None,
    config_per_dataset: Optional[Dict[str, str]] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
    existing_config_policy: str = "provided",
    processed_subjects_policy: str = "skip",
    interactive_prompts: bool = False,
    keep_temp_on_error: bool = False,
    qa_subject: bool = False,
    qa_group: bool = False,
    qa_multisample: bool = False,
    qc_group: bool = False,
    qc_multisample: bool = False,
    qa_all: bool = False,
    qc_all: bool = False,
    all_modes: bool = True,
    logger: Callable[[str], None] = print,
) -> None:
    """Run MEGqc calculation (incl. GQI) and then all QA/QC plotting modes.

    This helper is shared by CLI and GUI so that "run all" behavior is
    consistent and auditable from one entry point.
    """
    effective_analysis_id = analysis_id
    if analysis_mode == "new" and not effective_analysis_id:
        # Run-all needs one shared profile ID for both calculation and plotting.
        effective_analysis_id = _timestamp_analysis_id()
        logger(f"[Run ALL] Generated shared analysis_id: {effective_analysis_id}")
    plotting_mode = "reuse" if analysis_mode == "new" else analysis_mode

    run_calculation_dispatch(
        dataset_paths=dataset_paths,
        config_file_path=config_file_path,
        internal_config_file_path=internal_config_file_path,
        sub_list=sub_list,
        n_jobs=calc_n_jobs,
        derivatives_output=derivatives_output,
        dataset_njobs=dataset_njobs,
        dataset_subs=dataset_subs,
        global_config_file_path=global_config_file_path,
        config_per_dataset=config_per_dataset,
        analysis_mode=analysis_mode,
        analysis_id=effective_analysis_id,
        existing_config_policy=existing_config_policy,
        processed_subjects_policy=processed_subjects_policy,
        interactive_prompts=interactive_prompts,
        keep_temp_on_error=keep_temp_on_error,
        logger=logger,
    )
    run_plotting_dispatch(
        dataset_paths=dataset_paths,
        derivatives_output=derivatives_output,
        njobs=plot_njobs,
        qa_subject=qa_subject,
        qa_group=qa_group,
        qa_multisample=qa_multisample,
        qc_group=qc_group,
        qc_multisample=qc_multisample,
        qa_all=qa_all,
        qc_all=qc_all,
        all_modes=all_modes,
        analysis_mode=plotting_mode,
        analysis_id=effective_analysis_id,
        logger=logger,
    )


def run_gqi():
    """Recalculate Global Quality Index reports using existing metrics."""
    parser = argparse.ArgumentParser(
        description="Recompute Global Quality Index using previously calculated metrics"
    )
    parser.add_argument(
        "--inputdata",
        nargs="+",
        type=str,
        required=True,
        help="One or more BIDS dataset paths.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Optional global config file for all datasets.",
    )
    parser.add_argument(
        "--config_per_dataset",
        nargs="+",
        type=str,
        required=False,
        help=(
            "Optional per-dataset config overrides.\n"
            "Format per entry: <dataset_path>::<path/to/settings.ini>"
        ),
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help="Optional folder to store derivatives outside the BIDS dataset",
    )
    parser.add_argument(
        "--analysis_mode",
        type=str,
        choices=["legacy", "new", "reuse", "latest"],
        default="legacy",
        help=PROFILE_MODE_HELP,
    )
    parser.add_argument(
        "--analysis_id",
        type=str,
        required=False,
        help=PROFILE_ID_HELP,
    )
    args = parser.parse_args()
    if args.analysis_mode == "reuse" and not args.analysis_id:
        parser.error("analysis_mode='reuse' requires --analysis_id.")

    install_path = _megqc_installation_dir()
    default_config = os.path.join(install_path, "settings", "settings.ini")
    try:
        config_per_dataset = _parse_config_per_dataset_entries(args.config_per_dataset)
    except ValueError as exc:
        parser.error(str(exc))
        return

    run_gqi_dispatch(
        dataset_paths=args.inputdata,
        default_config_file_path=default_config,
        derivatives_output=args.derivatives_output,
        global_config_file_path=args.config,
        config_per_dataset=config_per_dataset,
        analysis_mode=args.analysis_mode,
        analysis_id=args.analysis_id,
        logger=print,
    )
