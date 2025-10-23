import sys
import time
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

# Parameters:
# ------------------------------------------------------------------
# Path to the root of your BIDS MEG dataset.
data_directory = "H:/_VIP/Python/MyWork/MEGqc/dataset/ds003483"

# List of subject IDs to run the pipeline on
# sub_list = 'all'
sub_list = ['009']
# sub_list = ['009','012','013','014','015']

# # Path to the root of your EEG datas
data_directory = "H:/Datos/MNI/BIDS_EEG/BIDS_Artifacts_Example"
# # List of subject IDs to run the pipeline on
# # sub_list = 'all'
sub_list = ['00000254']
# # sub_list = ['009','012','013','014','015']

# Path to a INI config file with user-defined parameters.
config_file_path = "H:/_VIP/Python/MyWork/megqcjul25/MEGqc/meg_qc/settings/settings.ini"
# Path to a INI config file of internal parameters.
internal_config_file_path = "H:/_VIP/Python/MyWork/megqcjul25/MEGqc/meg_qc/settings/settings_internal.ini"

# Number of CPU cores you want to use (for example, 4). Use -1 to utilize all available CPU cores:
n_jobs_to_use = 1
# Number of parallel jobs to use during processing.
# Default is 1. Use -1 to utilize all available CPU cores.
#
#  ⚠️ Recommendation based on system memory:
#     - 8 GB RAM → up to 1 parallel jobs (default)
#     - 16 GB RAM → up to 2 parallel jobs
#     - 32 GB RAM → up to 6 parallel jobs
#     - 64 GB RAM → up to 16 parallel jobs
#     - 128 GB RAM → up to 30 parallel jobs
#
#    Using --n_jobs -1 will use all available CPU cores.
#    Note: this may not always be optimal, especially when processing many subjects
#    on systems with limited memory.
#   ⚠️ If you have many CPU cores but low RAM, this can lead to crashes.
#    As a rule of thumb, your available RAM (in GB) should be at least
#    3.5 times the number of CPUs. For example, using 16 CPUs
#    requires at least 56 GB of total system memory (46 GB of available memory).


# ------------------------------------------------------------------

# RUN Calculation Module
# ------------------------------------------------------------------
start_time = time.time()

make_derivative_meg_qc(
    config_file_path,
    internal_config_file_path,
    data_directory,
    sub_list,
    n_jobs=n_jobs_to_use
)

end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Script finished. Elapsed time: {elapsed_seconds:.2f} seconds.")
# ------------------------------------------------------------------

