import sys
import time
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

# The plotting backend (full or lite) is controlled via the
# 'full_html_reports' option in settings.ini.

# Parameters:
# ------------------------------------------------------------------
# Path to the root of your BIDS MEG dataset.
data_directory = '/home/karelo/Desktop/Development/MEGQC_workshop/datasets/ds003483'
# Optional external derivatives root for plotting results
derivatives_output_path = None
# Number of CPU cores you want to use (for example, 4). Use -1 to utilize all available CPU cores:
n_jobs_to_use = 3
# ------------------------------------------------------------------

# RUN plotting Module
# ------------------------------------------------------------------
start_time = time.time()

make_plots_meg_qc(data_directory, n_jobs_to_use, derivatives_output_path)

end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Script finished. Elapsed time: {elapsed_seconds:.2f} seconds.")
# ------------------------------------------------------------------

