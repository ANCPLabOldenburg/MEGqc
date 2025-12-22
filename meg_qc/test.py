import argparse
import os
import sys
import shutil
from typing import List, Union

def hello_world():
    """
    Simple example function that prints the --subs argument from the command line.
    Not directly related to MEG QC, but provided as an example.
    """
    dataset_path_parser = argparse.ArgumentParser(description="parser for string to print")
    dataset_path_parser.add_argument("--subs", nargs='+', type=str, required=False, help="path to config file")
    args = dataset_path_parser.parse_args()
    print(args.subs)


def run_megqc():
    """
    Main entry point for launching the MEG QC pipeline from the command line.

    Command line usage example:
        run-megqc --inputdata /path/to/BIDS_dataset [--config /path/to/config.ini] [--subs 001 002] [--n_jobs 4]

    After parsing arguments, it calls make_derivative_meg_qc() with the chosen config,
    dataset path, subject list, and number of parallel jobs.
    """
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
    import time

    import argparse

    # Create an ArgumentParser for MEG QC
    dataset_path_parser = argparse.ArgumentParser(
        description=(
            "Command-line argument parser for MEGqc.\n"
            "--inputdata (required): path to a BIDS dataset.\n"
            "--config (optional): path to a config file.\n"
            "--subs (optional): list of subject IDs (defaults to all).\n"
            "--n_jobs (optional): number of parallel jobs (defaults to 1)."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    dataset_path_parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help=(
            "Path to the root of your BIDS MEG dataset.\n"
            "This is a required argument.\n"
            "Example: /path/to/dataset"
        )
    )

    dataset_path_parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to a INI config file with user-defined parameters.\n"
            "Optional: If not provided, default parameters are used.\n"
            "Example: /path/to/config.ini"
        )
    )

    dataset_path_parser.add_argument(
        "--subs",
        nargs='+',
        type=str,
        required=False,
        help=(
            "List of subject IDs to run the pipeline on.\n"
            "Optional: If not provided, the pipeline will run on all subjects.\n"
            "Example: --subs 009 012 013"
        )
    )

    dataset_path_parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help=(
            "Number of parallel jobs to use during processing.\n"
            "Default is 1. Use -1 to utilize all available CPU cores.\n"
            "\n"
            "⚠️ Recommendation based on system memory:\n"
            "  - 8 GB RAM → up to 1 parallel jobs (default)\n"
            "  - 16 GB RAM → up to 2 parallel jobs\n"
            "  - 32 GB RAM → up to 6 parallel jobs\n"
            "  - 64 GB RAM → up to 16 parallel jobs\n"
            "  - 128 GB RAM → up to 30 parallel jobs\n"
            "\n"
            "Using --n_jobs -1 will use all available CPU cores.\n"
            "Note: this may not always be optimal, especially when processing many subjects\n"
            "on systems with limited memory.\n"
            "⚠️ If you have many CPU cores but low RAM, this can lead to crashes.\n"
            "As a rule of thumb, your available RAM (in GB) should be at least\n"
            "3.5 times the number of CPUs. For example, using 16 CPUs\n"
            "requires at least 56 GB of total system memory (46 GB of available memory)."
        )
    )

    # Parse arguments
    args = dataset_path_parser.parse_args()

    # ----------------------------------------------------------------
    # Prepare internal and default user config file paths
    # ----------------------------------------------------------------
    path_to_megqc_installation = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir)
    )
    relative_path_to_internal_config = "settings/settings_internal.ini"
    relative_path_to_config = "settings/settings.ini"

    # Normalize both relative paths
    relative_path_to_internal_config = os.path.normpath(relative_path_to_internal_config)
    relative_path_to_config = os.path.normpath(relative_path_to_config)

    # Join paths to form absolute paths
    internal_config_file_path = os.path.join(
        path_to_megqc_installation,
        relative_path_to_internal_config
    )

    # Print for debug, showing which directory is in use
    print("MEG QC installation directory:", path_to_megqc_installation)

    data_directory = args.inputdata
    print("Data directory:", data_directory)

    # Check if --subs was provided
    if args.subs is None:
        sub_list = 'all'
    else:
        sub_list = args.subs
        print("Subjects to process:", sub_list)

    # Decide how to handle the config file
    if args.config is None:
        url_megqc_book = 'https://aaronreer.github.io/docker_workshop_setup/settings_explanation.html'
        text = 'The settings explanation section of our MEGqc User Jupyterbook'

        print(
            'You called the MEGqc pipeline without the optional\n\n'
            '--config <path/to/custom/config> argument.\n\n'
            'MEGqc will proceed with the default parameter settings.\n'
            'Detailed information on the user parameters in MEGqc and their default values '
            f'can be found here: \n\n\033]8;;{url_megqc_book}\033\\{text}\033]8;;\033\\\n\n'
        )
        user_confirm = input('Do you want to proceed with the default settings? (y/n): ').lower().strip() == 'y'
        if user_confirm:
            config_file_path = os.path.join(path_to_megqc_installation, relative_path_to_config)
        else:
            print(
                "Use the following command to copy the default config file:\n"
                "   get-megqc-config --target_directory <path/to/directory>\n\n"
                "Then edit the copied file (e.g., to adjust parameters) and run the pipeline again with:\n"
                "   run-megqc --inputdata <path> --config <path/to/modified_config.ini>\n\n"
            )
            return
    else:
        config_file_path = args.config

    # ----------------------------------------------------------------
    # Number of parallel jobs
    # ----------------------------------------------------------------
    n_jobs_used = args.n_jobs
    print(f"Running MEG QC in parallel with n_jobs={n_jobs_used}")

    # ----------------------------------------------------------------
    # Optionally measure time for the pipeline execution
    # ----------------------------------------------------------------
    start_time = time.time()

    # ----------------------------------------------------------------
    # Run the MEG QC pipeline
    # ----------------------------------------------------------------
    make_derivative_meg_qc(
        default_config_file_path=config_file_path,
        internal_config_file_path=internal_config_file_path,
        ds_paths=data_directory,
        sub_list=sub_list,
        n_jobs=n_jobs_used
    )

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print(f"MEGqc has completed the calculation of metrics in {elapsed_seconds:.2f} seconds.")
    print(
        f"Results can be found in {data_directory}/derivatives/Meg_QC/calculation"
    )

    # ----------------------------------------------------------------
    # Optionally prompt the user to run the plotting module
    # ----------------------------------------------------------------
    user_input = input('Do you want to run the MEGqc plotting module on the MEGqc results? (y/n): ').lower().strip() == 'y'
    if user_input:
        from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
        make_plots_meg_qc(data_directory)
        return
    else:
        return


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
    args = target_directory_parser.parse_args()
    destination_directory = args.target_directory + '/settings.ini'
    print("Destination directory for config:", destination_directory)

    path_to_megqc_installation = os.path.abspath(
        os.path.join(os.path.abspath(__file__), os.pardir)
    )
    print("MEG QC installation directory:", path_to_megqc_installation)

    config_file_path = os.path.join(path_to_megqc_installation, 'settings', 'settings.ini')
    print("Source of default config file:", config_file_path)

    shutil.copy(config_file_path, destination_directory)
    print('The config file has been copied to ' + destination_directory)

    return


def get_plots():
    """
    Launches the MEG QC plotting module, which generates visualizations
    from the pipeline results in the derivatives folder.
    """
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc
    from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots

    dataset_path_parser = argparse.ArgumentParser(
        description="parser for MEGqc: --inputdata (mandatory) path/to/BIDSds"
    )
    dataset_path_parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help="Path to the root of your BIDS MEG dataset"
    )
    dataset_path_parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help="Optional folder to store derivatives outside the BIDS dataset",
    )
    args = dataset_path_parser.parse_args()
    data_directory = args.inputdata

    # Mirror the calculation module: allow the derivatives to live outside the
    # dataset by resolving a custom output root when requested. The returned
    # path is logged to help users find the exact folder being read.
    _, derivatives_root = resolve_output_roots(data_directory, args.derivatives_output)
    print(f"Using derivatives from: {derivatives_root}")

    make_plots_meg_qc(data_directory, derivatives_base=args.derivatives_output)
    return


def run_gqi():
    """Recalculate Global Quality Index reports using existing metrics."""
    from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary
    from meg_qc.calculation.meg_qc_pipeline import resolve_output_roots

    parser = argparse.ArgumentParser(
        description="Recompute Global Quality Index using previously calculated metrics"
    )
    parser.add_argument(
        "--inputdata",
        type=str,
        required=True,
        help="Path to the root of your BIDS MEG dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to a config file with GQI parameters",
    )
    parser.add_argument(
        "--derivatives_output",
        type=str,
        required=False,
        help="Optional folder to store derivatives outside the BIDS dataset",
    )
    args = parser.parse_args()

    install_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    default_config = os.path.join(install_path, "settings", "settings.ini")
    cfg_path = args.config if args.config else default_config

    # Use the same resolver as the calculation module so that external
    # derivatives directories are handled consistently for GQI regeneration.
    _, derivatives_root = resolve_output_roots(args.inputdata, args.derivatives_output)

    generate_gqi_summary(args.inputdata, derivatives_root, cfg_path)
    return
