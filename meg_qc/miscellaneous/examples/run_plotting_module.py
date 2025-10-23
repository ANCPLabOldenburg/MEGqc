import sys
import time
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

# The plotting backend (full or lite) is controlled via the
# 'full_html_reports' option in settings.ini.

# Parameters:
# ------------------------------------------------------------------
# Path to the root of your BIDS MEG dataset.
data_directory = 'H:/_VIP/Python/MyWork/MEGqc/dataset/ds003483'
# # Path to the root of your EEG datas
data_directory = "H:/Datos/MNI/BIDS_EEG/BIDS_Artifacts_Example"
# Number of CPU cores you want to use (for example, 4). Use -1 to utilize all available CPU cores:
n_jobs_to_use = 1
# ------------------------------------------------------------------

# RUN plotting Module
# ------------------------------------------------------------------
start_time = time.time()

make_plots_meg_qc(data_directory,n_jobs_to_use)

end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Script finished. Elapsed time: {elapsed_seconds:.2f} seconds.")
# ------------------------------------------------------------------

