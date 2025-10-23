import sys
import os
import ancpbids
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style
from collections import defaultdict
import re
from typing import List
from pprint import pprint
import gc
from ancpbids import DatasetOptions
import configparser
from pathlib import Path

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
gradparent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(gradparent_dir)

from meg_qc.calculation.objects import QC_derivative

# Plotting backends (``universal_plots`` vs ``universal_plots_lite``) and the
# accompanying report helpers need to be available not only in the main process
# but also in the worker processes spawned by joblib.  Configure them at module
# import time so that all processes share the same setup.


def _load_plotting_backend():
    """Configure plotting backend and expose report helpers."""

    # If the backend has been loaded already, do nothing.
    if 'make_joined_report_mne' in globals():
        return

    cfg = configparser.ConfigParser()
    settings_path = (
        Path(__file__).resolve().parents[1] / 'settings' / 'settings.ini'
    )
    cfg.read(settings_path)
    use_full_reports = cfg['DEFAULT'].getboolean('full_html_reports', True)

    if use_full_reports:
        import meg_qc.plotting.universal_plots as _plots
    else:
        import meg_qc.plotting.universal_plots_lite as _plots

    # Make the chosen backend available under the expected module name so that
    # other modules (e.g. ``universal_html_report``) pick it up.
    sys.modules['meg_qc.plotting.universal_plots'] = _plots
    globals().update(
        {
            name: getattr(_plots, name)
            for name in dir(_plots)
            if not name.startswith('_')
        }
    )

    from meg_qc.plotting.universal_html_report import (
        make_joined_report_mne,
        make_summary_qc_report,
    )

    globals().update(
        {
            'make_joined_report_mne': make_joined_report_mne,
            'make_summary_qc_report': make_summary_qc_report,
        }
    )


_load_plotting_backend()

# IMPORTANT: keep this order of imports, first need to add parent dir to sys.path, then import from it.

# ____________________________

# How plotting in MEGqc works:
# During calculation save in the right folders the csvs with data for plotting
# During plotting step - read the csvs (find using ancpbids), plot them, save them as htmls in the right folders.


def create_categories_for_selector(entities: dict):

    """
    Create categories based on what metrics have already been calculated and detected as ancp bids as entities in MEGqc derivatives folder.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    categories : dict
        A dictionary of entities and their subcategories with modified names
    """

    # Create a copy of entities
    categories = entities.copy()

    # Rename 'description' to 'METRIC' and sort the values
    categories = {
        ('METRIC' if k == 'description' else k): sorted(v, key=str)
        for k, v in categories.items()
    }

    #From METRIC remove whatever is not metric.
    #Cos METRIC is originally a desc entity which can contain just anything:

    if 'METRIC' in categories:
        valid_metrics = ['_ALL_METRICS_', 'STDs', 'PSDs', 'PtPsManual', 'PtPsAuto', 'ECGs', 'EOGs', 'Head', 'Muscle']
        categories['METRIC'] = [x for x in categories['METRIC'] if x.lower() in [metric.lower() for metric in valid_metrics]]

    #add '_ALL_' to the beginning of the list for each category:

    for category, subcategories in categories.items():
        categories[category] = ['_ALL_'+category+'s_'] + subcategories

    # Add 'm_or_g' category
    categories['m_or_g'] = ['_ALL_sensors', 'mag', 'grad']

    return categories


def selector(entities: dict):

    """
    Creates a in-terminal visual selector for the user to choose the entities and settings for plotting.

    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    selected_entities : dict
        A dictionary of selected entities.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    """

    # SELECT ENTITIES and SETTINGS
    # Define the categories and subcategories
    categories = create_categories_for_selector(entities)

    selected = {}
    # Create a list of values with category titles
    for key, values in categories.items():
        results, quit_selector = select_subcategory(categories[key], key)

        print('___MEGqc___: select_subcategory: ', key, results)

        if quit_selector: # if user clicked cancel - stop:
            print('___MEGqc___: You clicked cancel. Please start over.')
            return None, None

        selected[key] = results


    # Separate into selected_entities and plot_settings
    selected_entities = {key: values for key, values in selected.items() if key != 'm_or_g'}
    plot_settings = {key: values for key, values in selected.items() if key == 'm_or_g'}

    return selected_entities, plot_settings


def select_subcategory(subcategories: List, category_title: str, window_title: str = "What would you like to plot? Click to select."):

    """
    Create a checkbox list dialog for the user to select subcategories.
    Example:
    sub: 009, 012, 013

    Parameters
    ----------
    subcategories : List
        A list of subcategories, such as: sub, ses, task, run, metric, mag/grad.
    category_title : str
        The title of the category.
    window_title : str
        The title of the checkbox list dialog, for visual.

    Returns
    -------
    results : List
        A list of selected subcategories.
    quit_selector : bool
        A boolean indicating whether the user clicked Cancel.

    """

    quit_selector = False

    # Create a list of values with category titles
    values = [(str(items), str(items)) for items in subcategories]

    while True:
        results = checkboxlist_dialog(
            title=window_title,
            text=category_title,
            values=values,
            style=Style.from_dict({
                'dialog': 'bg:#cdbbb3',
                'button': 'bg:#bf99a4',
                'checkbox': '#e8612c',
                'dialog.body': 'bg:#a9cfd0',
                'dialog shadow': 'bg:#c98982',
                'frame.label': '#fcaca3',
                'dialog.body label': '#fd8bb6',
            })
        ).run()

        # Set quit_selector to True if the user clicked Cancel (results is None)
        quit_selector = results is None

        if quit_selector or results:
            break
        else:
            print('___MEGqc___: Please select at least one subcategory or click Cancel.')


    # if '_ALL_' was chosen - choose all categories, except _ALL_ itself:
    if results: #if something was chosen
        for r in results:
            if '_ALL_' in r.upper():
                results = [str(category) for category in subcategories if '_ALL_' not in str(category).upper()]
                #Important! Keep ....if '_ALL_' not in str(category).upper() with underscores!
                #otherwise it will excude tasks like 'oddbALL' and such
                break

    return results, quit_selector


def get_ds_entities(dataset, calculated_derivs_folder: str):

    """
    Get the entities of the dataset using ancpbids, only get derivative entities, not all raw data.

    Parameters
    ----------
    dataset : ancpbids object
        The dataset object.
    calculated_derivs_folder : str
        The path to the calculated derivatives folder.

    Returns
    -------
    entities : dict
        A dictionary of entities and their subcategories.

    """

    try:
        entities = dataset.query_entities(scope=calculated_derivs_folder)
        print('___MEGqc___: ', 'Entities found in the dataset: ', entities)
        #we only get entities of calculated derivatives here, not entire raw ds.
    except:
        raise FileNotFoundError(f'___MEGqc___: No calculated derivatives found for this ds!')

    return entities


def csv_to_html_report(raw_info_path: str, metric: str, tsv_paths: List, report_str_path: str, plot_settings):

    """
    Create an HTML report from the CSV files.

    Parameters
    ----------
    raw_info_path : str
        The path to the raw info object.
    metric : str
        The metric to be plotted.
    tsv_paths : List
        A list of paths to the CSV files.
    report_str_path : str
        The path to the JSON file containing the report strings.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    Returns
    -------
    report_html_string : str
        The HTML report as a string.

    """

    m_or_g_chosen = plot_settings['m_or_g']

    time_series_derivs, sensors_derivs, ptp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, std_derivs, psd_derivs, muscle_derivs, head_derivs = [], [], [], [], [], [], [], [], [], []

    stim_derivs = []

    for tsv_path in tsv_paths: #if we got several tsvs for same metric, like for PSD:

        if "_eeg." in tsv_path:
            m_or_g_chosen = ['eeg']
        #get the final file name of tsv path:
        basename = os.path.basename(tsv_path)
        if 'desc-stimulus' in basename:
            stim_derivs = plot_stim_csv(tsv_path)

        if 'STD' in metric.upper():

            fig_std_epoch0 = []
            fig_std_epoch1 = []

            std_derivs += plot_sensors_3d_csv(tsv_path)

# bosch prueba
            for m_or_g in m_or_g_chosen:
                #fig_topomap = plot_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                #fig_topomap_3d = plot_3d_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='stds')

                #std_derivs += fig_topomap + fig_topomap_3d + fig_all_time + fig_std_epoch0 + fig_std_epoch1
                std_derivs += fig_all_time + fig_std_epoch0 + fig_std_epoch1

        if 'PTP' in metric.upper():

            fig_ptp_epoch0 = []
            fig_ptp_epoch1 = []

            ptp_manual_derivs += plot_sensors_3d_csv(tsv_path)

# bosch prueba
            for m_or_g in m_or_g_chosen:

                #fig_topomap = plot_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                #fig_topomap_3d = plot_3d_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='peaks')

                #ptp_manual_derivs += fig_topomap + fig_topomap_3d + fig_all_time + fig_ptp_epoch0 + fig_ptp_epoch1
                ptp_manual_derivs += fig_all_time + fig_ptp_epoch0 + fig_ptp_epoch1

        elif 'PSD' in metric.upper():

            method = 'welch'
            #is also preselected in internal_settings.ini Adjust here if change in calculation,
            # this module doesnt access internal settings

            psd_derivs += plot_sensors_3d_csv(tsv_path)

            for m_or_g in m_or_g_chosen:

                psd_derivs += Plot_psd_csv(m_or_g, tsv_path, method)

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'noise')

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'waves')

        elif 'ECG' in metric.upper():

            ecg_derivs += plot_sensors_3d_csv(tsv_path)

            ecg_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            ecg_derivs += plot_mean_rwave_csv(tsv_path, 'ECG')

            #TODO: add ch description like here? export it as separate report strings?
            #noisy_ch_derivs += [QC_derivative(fig, bad_ecg_eog[ecg_ch]+' '+ecg_ch, 'plotly', description_for_user = ecg_ch+' is '+ bad_ecg_eog[ecg_ch]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) tolerable number of breaks: '+str(ecg_eval[1])+', 3) tolerable number of bursts: '+str(ecg_eval[2]))]

            for m_or_g in m_or_g_chosen:
                ecg_derivs += plot_artif_per_ch_3_groups(tsv_path, m_or_g, 'ECG', flip_data=False)
                #ecg_derivs += plot_correlation_csv(tsv_path, 'ECG', m_or_g)

        elif 'EOG' in metric.upper():

            eog_derivs += plot_sensors_3d_csv(tsv_path)

            eog_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            eog_derivs += plot_mean_rwave_csv(tsv_path, 'EOG')

# bosch prueba
            #prueba for m_or_g in m_or_g_chosen:
                #prueba eog_derivs += plot_artif_per_ch_3_groups(tsv_path, m_or_g, 'EOG', flip_data=False)
                #prueba eog_derivs += plot_correlation_csv(tsv_path, 'EOG', m_or_g)


        elif 'MUSCLE' in metric.upper():

            muscle_derivs +=  plot_muscle_csv(tsv_path)


        elif 'HEAD' in metric.upper():

            head_pos_derivs, _ = plot_head_pos_csv(tsv_path)
            # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
            # head_pos_derivs += head_pos_derivs2
            head_derivs += head_pos_derivs

    QC_derivs = {
        'TIME_SERIES': time_series_derivs,
        'STIMULUS': stim_derivs,
        'SENSORS': sensors_derivs,
        'STD': std_derivs,
        'PSD': psd_derivs,
        'PTP_MANUAL': ptp_manual_derivs,
        'PTP_AUTO': pp_auto_derivs,
        'ECG': ecg_derivs,
        'EOG': eog_derivs,
        'HEAD': head_derivs,
        'MUSCLE': muscle_derivs,
        'REPORT_MNE': []
    }


    #Sort all based on fig_order of QC_derivative:
    #(To plot them in correct order in the report)
    for metric, values in QC_derivs.items():
        if values:
            QC_derivs[metric] = sorted(values, key=lambda x: x.fig_order)


    if not report_str_path: #if no report strings were saved. happens when mags/grads didnt run to make tsvs.
        report_strings = {
        'INITIAL_INFO': '',
        'TIME_SERIES': '',
        'STD': '',
        'PSD': '',
        'PTP_MANUAL': '',
        'PTP_AUTO': '',
        'ECG': '',
        'EOG': '',
        'HEAD': '',
        'MUSCLE': '',
        'SENSORS': '',
        'STIMULUS': ''
        }
    else:
        with open(report_str_path) as json_file:
            report_strings = json.load(json_file)


    report_html_string = make_joined_report_mne(raw_info_path, QC_derivs, report_strings)

    return report_html_string


def extract_raw_entities_from_obj(obj):

    """
    Function to create a key from the object excluding the 'desc' attribute

    Parameters
    ----------
    obj : ancpbids object
        An object from ancpbids.

    Returns
    -------
    tuple
        A tuple containing the name, extension, and suffix of the object.

    """
    # Remove the 'desc' part from the name, so we get the name of original raw that the deriv belongs to:
    raw_name = re.sub(r'_desc-[^_]+', '', obj.name)
    return (raw_name, obj.extension, obj.suffix)


def sort_tsvs_by_raw(tsvs_by_metric: dict):

    """
    For every metric, if we got same raw entitites, we can combine derivatives for the same raw into a list.
    Since we collected entities not from raw but from derivatives, we need to remove the desc part from the name.
    After that we combine files with the same 'name' in entity_val objects in 1 list:

    Parameters
    ----------
    tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files.

    Returns
    -------
    combined_tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files combined by raw entity

    """

    sorted_tsvs_by_metric_by_raw = {}

    for metric, obj_dict in tsvs_by_metric.items():
        combined_dict = defaultdict(list)

        for obj, tsv_path in obj_dict.items():
            raw_entities = extract_raw_entities_from_obj(obj)
            combined_dict[raw_entities].extend(tsv_path)

        # Convert keys back to original objects
        final_dict = {}
        for raw_entities, paths in combined_dict.items():
            # Find the first object with the same key
            for obj in obj_dict.keys():
                if extract_raw_entities_from_obj(obj) == raw_entities:
                    final_dict[obj] = paths
                    break

        sorted_tsvs_by_metric_by_raw[metric] = final_dict

    pprint('___MEGqc___: ', 'sorted_tsvs_by_metric_by_raw: ', sorted_tsvs_by_metric_by_raw)

    return sorted_tsvs_by_metric_by_raw

class Deriv_to_plot:

    """
    A class to represent the derivatives to be plotted.

    Attributes
    ----------
    path : str
        The path to the TSV file.
    metric : str
        The metric to be plotted.
    deriv_entity_obj : dict
        The entity object of the derivative created with ANCPBIDS.
    raw_entity_name : str
        The name of the raw entity.
    subject : str
        The subject ID.

    Methods
    -------
    __repr__()
        Return a string representation of the object.
    print_detailed_entities()
        Print the detailed entities of the object.
    find_raw_entity_name()
        Find the raw entity name from the deriv entity name.

    """

    def __init__(self, path: str, metric: str, deriv_entity_obj, raw_entity_name: str = None):

        self.path = path
        self.metric = metric
        self.deriv_entity_obj = deriv_entity_obj
        self.raw_entity_name = raw_entity_name

        # Find the subject ID using regex
        match = re.search(r'sub-\d+_', self.deriv_entity_obj['name'])
        if match:
            self.subject = match.group(0).replace('sub-', '').rstrip('_')
        else:
            self.subject = None  # or handle the case where the subject ID is not found

    def __repr__(self):

        return (
            f"Deriv_to_plot(\n"
            f"    subject={self.subject},\n"
            f"    path={self.path},\n"
            f"    metric={self.metric},\n"
            f"    deriv_entity_obj={self.deriv_entity_obj},\n"
            f"    raw_entity_name={self.raw_entity_name}\n"
            f")"
        )

    def print_detailed_entities(self):

        """
        Print the detailed entities of the object, cos in ANCP representation it s cut.
        Here skipping the last value, cos it s the contents.
        If u got a lot of contents, like html it ll print you a book XD.
        """

        keys = list(self.deriv_entity_obj.keys())
        for val in keys[:-1]:  # Iterate over all keys except the last one
            print('_Deriv_: ', val, self.deriv_entity_obj[val])

    def find_raw_entity_name(self):

        """
        Find the raw entity name from the deriv entity name
        """

        self.raw_entity_name = re.sub(r'_desc-.*', '', self.deriv_entity_obj['name'])


from joblib import Parallel, delayed


import json
import os
import sys
from typing import Any, Dict, List
import pandas as pd

def _df_from_records(records: Any) -> pd.DataFrame:
    """Return a DataFrame from a list of records (or an empty DF if None/empty)."""
    if isinstance(records, list) and len(records) > 0:
        return pd.DataFrame(records)
    return pd.DataFrame()


def create_GQIhtml_from_json(json_path: str, html_output: str) -> None:
    # ---- Load JSON ----
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # ---- Extract top-level values ----
    html_name = data.get("file_name", os.path.basename(json_path))
    GQI = data.get("GQI", "N/A")

    penalties: Dict[str, float] = data.get("GQI_penalties", {}) or {}

    params: Dict[str, Any] = data.get("parameters", {}) or {}
    std_lvl = params.get("std_lvl", "N/A")
    ptp_lvl = params.get("ptp_lvl", "N/A")
    std_epoch_lvl = params.get("std_epoch_lvl", "N/A")
    ptp_epoch_lvl = params.get("ptp_epoch_lvl", "N/A")

    # ---- Build DataFrames from lists of records ----
    general_df   = _df_from_records(data.get("STD_time_series"))
    ptp_df       = _df_from_records(data.get("PTP_time_series"))
    std_epoch_df = _df_from_records(data.get("STD_epoch_summary"))
    ptp_epoch_df = _df_from_records(data.get("PTP_epoch_summary"))
    ecg_df       = _df_from_records(data.get("ECG_correlation_summary"))
    eog_df       = _df_from_records(data.get("EOG_correlation_summary"))
    psd_df       = _df_from_records(data.get("PSD_noise_summary"))

    # ---- Muscle events -> DataFrame ----
    # JSON example: "Muscle_events": {"# Muscle Events": muscle_events, "total_number_of_events": total_events}
    muscle_obj = data.get("Muscle_events", {})
    if isinstance(muscle_obj, dict) and muscle_obj:
        # render as single-row summary table
        muscle_df = pd.DataFrame([muscle_obj])
    else:
        muscle_df = pd.DataFrame()

    # ---- Style block (as provided) ----
    style = """
        <style>
            body { font-family: Arial, sans-serif; margin: 10px; font-size: 16px; }
            h1 { color: #003366; font-size: 25px; margin-bottom: 6px; font-weight: bold; }
            h2 { color: #004d99; font-size: 19px; margin: 12px 0 6px 0; }
            table { border-collapse: collapse; margin: 0 0 8px 0; font-size: 16px; }
            th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            .table-flex { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; margin-bottom: 12px; }
            .table-box { flex: 1; min-width: 300px; }
            .file-label { font-size: 18px; font-weight: bold; margin: 0 0 2px 12px; }
            .subtitle { font-size: 19px; font-weight: bold; color: #222; margin: 0 0 12px 12px; }
            .header-grid { display: grid; grid-template-columns: 1fr 1fr; align-items: start; margin-bottom: 0; }
        </style>
    """

    # ---- Render HTML ----
    with open(html_output, "w", encoding="utf-8") as f:
        # (Removed a stray initial "</div></div>" which would break markup)
        f.write("<html><head><meta charset='UTF-8'>" + style + "</head><body>")

        # Header
        f.write("<div class='header-grid'>")
        f.write("<div><h1>MEGQC Global Quality Report</h1></div>")
        f.write(f"<div><div class='file-label'>File: {html_name}</div>")
        f.write(f"<div class='subtitle'>Global Quality Index (GQI): {GQI}</div></div></div>")

        # Penalties (only if present)
        if penalties:
            f.write("<h2>GQI Penalties</h2>")
            penalties_df = pd.DataFrame([
                {"Metric": "Bad Channels", "Penalty (%)": f"{penalties.get('ch', 0):.2f}"},
                {"Metric": "Correlation", "Penalty (%)": f"{penalties.get('corr', 0):.2f}"},
                {"Metric": "Muscle", "Penalty (%)": f"{penalties.get('mus', 0):.2f}"},
                {"Metric": "PSD Noise", "Penalty (%)": f"{penalties.get('psd', 0):.2f}"},
            ])
            f.write(penalties_df.to_html(index=False, escape=False))

        # Time series tables
        if not general_df.empty or not ptp_df.empty:
            f.write("<div class='table-flex'>")
            f.write(f"<div class='table-box'><h2>STD Time-Series (STD level: {std_lvl})</h2>")
            if not general_df.empty:
                f.write(general_df.to_html(index=False))
            else:
                f.write("<i>No data</i>")
            f.write("</div>")

            f.write(f"<div class='table-box'><h2>PTP Time-Series (STD level: {ptp_lvl})</h2>")
            if not ptp_df.empty:
                f.write(ptp_df.to_html(index=False))
            else:
                f.write("<i>No data</i>")
            f.write("</div></div>")

        # PSD Noise
        if not psd_df.empty:
            f.write("<h2>PSD Noise Summary</h2>")
            f.write(psd_df.to_html(index=False))

        # Epoch summaries
        if not std_epoch_df.empty or not ptp_epoch_df.empty:
            f.write("<div class='table-flex'>")
            f.write(f"<div class='table-box'><h2>STD Epoch Summary (STD level: {std_epoch_lvl})</h2>")
            if not std_epoch_df.empty:
                f.write(std_epoch_df.to_html(index=False))
            else:
                f.write("<i>No data</i>")
            f.write("</div>")

            f.write(f"<div class='table-box'><h2>PTP Epoch Summary (STD level: {ptp_epoch_lvl})</h2>")
            if not ptp_epoch_df.empty:
                f.write(ptp_epoch_df.to_html(index=False))
            else:
                f.write("<i>No data</i>")
            f.write("</div></div>")

        # Correlation summaries (ECG/EOG)
        if not ecg_df.empty or not eog_df.empty:
            f.write("<div class='table-flex'>")
            if not ecg_df.empty:
                f.write("<div class='table-box'><h2>ECG Correlation Summary</h2>")
                f.write(ecg_df.to_html(index=False))
                f.write("</div>")
            if not eog_df.empty:
                f.write("<div class='table-box'><h2>EOG Correlation Summary</h2>")
                f.write(eog_df.to_html(index=False))
                f.write("</div>")
            f.write("</div>")

        # Muscle events
        if not muscle_df.empty:
            f.write("<h2>Muscle Events Summary</h2>")
            f.write(muscle_df.to_html(index=False))
        else:
            f.write("<h2>Muscle Events Summary</h2><i>No data</i>")

        f.write("</body></html>")


def process_subject(
        sub: str,
        dataset,
        derivs_to_plot: list,
        chosen_entities: dict,
        plot_settings: dict,
):
    """Plot all metrics for a single subject."""

    derivative = dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"
    reports_folder = derivative.create_folder(name='reports')
    subject_folder = reports_folder.create_folder(name='sub-' + sub)

    existing_raws_per_sub = list(set(
        d.raw_entity_name for d in derivs_to_plot if d.subject == sub
    ))

    for raw_entity_name in existing_raws_per_sub:
        derivs_for_this_raw = [
            d for d in derivs_to_plot if d.raw_entity_name == raw_entity_name
        ]

        raw_entities_base = derivs_for_this_raw[0].deriv_entity_obj

        raw_info_path = None
        report_str_path = None
        simple_metrics_path = None
        for d in derivs_for_this_raw:
            if d.metric == 'RawInfo':
                raw_info_path = d.path
            elif d.metric == 'ReportStrings':
                report_str_path = d.path
            elif d.metric == 'SimpleMetrics':
                simple_metrics_path = d.path

        metrics_to_plot = [
            m for m in chosen_entities['METRIC']
            if m not in ['RawInfo', 'ReportStrings', 'SimpleMetrics']
        ]

        for metric in metrics_to_plot:
            tsv_paths = [d.path for d in derivs_for_this_raw if d.metric == metric]
            if not tsv_paths:
                print(f'___MEGqc___: No tsvs found for {metric} / subject {sub}')
                continue

            tsvs_for_this_raw = [d for d in derivs_for_this_raw if d.metric == metric]
            raw_entities_to_write = tsvs_for_this_raw[0].deriv_entity_obj

            html_report = csv_to_html_report(
                raw_info_path,
                metric,
                tsv_paths,
                report_str_path,
                plot_settings,
            )

            meg_artifact = subject_folder.create_artifact(raw=raw_entities_to_write)
            meg_artifact.add_entity('desc', metric)
            meg_artifact.suffix = 'meg'
            meg_artifact.extension = '.html'

            meg_artifact.content = lambda file_path, rep=html_report: rep.save(
                file_path, overwrite=True, open_browser=False
            )

        if report_str_path and simple_metrics_path:

            # Generate per-subject summary HTML only from the GQI JSON summary
            json_path = simple_metrics_path.replace('SimpleMetrics', 'GlobalSummaryReport')
            # Replace 'derivatives/Meg_QC/calculation' by 'derivatives/Meg_QC/reports'
            path_parts = simple_metrics_path.split(os.path.sep)
            for i in range(len(path_parts) - 2):  # -2 to ensure that is not going to produce an exception
                if (path_parts[i] == 'derivatives' and
                        path_parts[i + 1] == 'Meg_QC' and
                        path_parts[i + 2] == 'calculation'):
                    path_parts[i + 2] = 'reports'
            # Building the path
            modified_path = os.path.sep.join(path_parts)
            html_output = modified_path.replace('SimpleMetrics', 'GlobalSummaryReport')
            html_output = os.path.splitext(html_output)[0] + '.html'
            create_GQIhtml_from_json(json_path, html_output)

            summary_html = make_summary_qc_report(report_str_path, simple_metrics_path)
            meg_artifact = subject_folder.create_artifact(raw=raw_entities_base)
            meg_artifact.add_entity('desc', 'summary_qc_report')
            meg_artifact.suffix = 'meg'
            meg_artifact.extension = '.html'
            meg_artifact.content = (
                lambda file_path, cont=summary_html: open(file_path, "w", encoding="utf-8").write(cont)
            )

    ancpbids.write_derivative(dataset, derivative)
    return


def make_plots_meg_qc(dataset_path: str, n_jobs: int = 1):
    """
    Create plots for the MEG QC pipeline, but WITHOUT the interactive selector.
    Instead, we assume 'all' for every entity (subject, task, session, run, metric).
    """

    # Ensure plotting backend and report helpers are available
    _load_plotting_backend()

    try:
        dataset = ancpbids.load_dataset(dataset_path, DatasetOptions(lazy_loading=True))
        schema = dataset.get_schema()
    except Exception:
        print('___MEGqc___: ',
              'No data found in the given directory path! \nCheck directory path in config file and presence of data.')
        return

    # Make sure the derivatives folder exists:
    derivatives_path = os.path.join(dataset_path, 'derivatives')
    if not os.path.isdir(derivatives_path):
        os.mkdir(derivatives_path)
        print('___MEGqc___: Derivs folder was not found! Created new.')

    calculated_derivs_folder = os.path.join('derivatives', 'Meg_QC', 'calculation')

    # --------------------------------------------------------------------------------
    # REPLACE THE SELECTOR WITH A HARDCODED "ALL" CHOICE
    # --------------------------------------------------------------------------------
    # 1) Get all discovered entities from the derivatives scope
    entities_found = get_ds_entities(dataset, calculated_derivs_folder)

    # Suppose 'description' is the metric list
    all_metrics = entities_found.get('description', [])

    # If you want them deduplicated, do:
    all_metrics = list(set(all_metrics))

    # Collapse individual PSD descriptions into a single entry so that the
    # general PSD report (``PSDs``) can gather all derivatives at once.  This
    # prevents the loss of the ``PSDs`` report when only the noise/waves
    # derivatives are present in the dataset.
    psd_related = {'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad'}
    if psd_related.intersection(all_metrics):
        all_metrics = [m for m in all_metrics if m not in psd_related]
        if 'PSDs' not in all_metrics:
            all_metrics.append('PSDs')

    # Retain only recognised metrics and normalise some aliases. This prevents
    # intermediate derivatives like ``ECGchannel`` from being treated as
    # standalone metrics and generating separate HTML reports.
    valid_metrics = {
        'STDs': 'STDs',
        'STD': 'STDs',
        'PSDs': 'PSDs',
        'PtPsManual': 'PtPsManual',
        'PtPsAuto': 'PtPsAuto',
        'ECGs': 'ECGs',
        'EOGs': 'EOGs',
        'Head': 'Head',
        'Muscle': 'Muscle',
        'RawInfo': 'RawInfo',
        'ReportStrings': 'ReportStrings',
        'SimpleMetrics': 'SimpleMetrics',
    }
    all_metrics = [valid_metrics[m] for m in all_metrics if m in valid_metrics]
    # Preserve order while removing duplicates
    all_metrics = list(dict.fromkeys(all_metrics))

    # Now store it in chosen_entities as a list
    chosen_entities = {
        'subject': list(entities_found.get('subject', [])),
        'task': list(entities_found.get('task', [])),
        'session': list(entities_found.get('session', [])),
        'run': list(entities_found.get('run', [])),
        'METRIC': all_metrics
    }

    # And now you can append or pop, etc.
    chosen_entities['METRIC'].append('stimulus')
    chosen_entities['METRIC'].append('RawInfo')
    chosen_entities['METRIC'].append('ReportStrings')
    # Ensure SimpleMetrics is always present so that summary reports can be built
    chosen_entities['METRIC'].append('SimpleMetrics')

    # 5) Define a simple plot_settings. Example: always 'mag' and 'grad'
    plot_settings = {'m_or_g': ['mag', 'grad']} #at this time I do not know if it is MEG or EEG. I will later madofiy this value

    print('___MEGqc___: CHOSEN entities to plot:', chosen_entities)
    print('___MEGqc___: CHOSEN settings:', plot_settings)
    # --------------------------------------------------------------------------------

    # 2. Collect TSVs for each sub + metric
    tsvs_to_plot_by_metric = {}
    tsv_entities_by_metric = {}

    for metric in chosen_entities['METRIC']:
        query_args = {
            'subj': chosen_entities['subject'],
            'task': chosen_entities['task'],
            'suffix': 'meg',
            'extension': ['tsv', 'json', 'fif'],
            'return_type': 'filename',
            'desc': '',
            'scope': calculated_derivs_folder,
        }

        # If the user (now "all") had multiple possible descs for PSDs, ECGs, etc.
        if metric == 'PSDs':
            # Include all PSD derivatives (noise and waves) so the PSD report is
            # generated correctly.
            query_args['desc'] = ['PSDs', 'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad']
        elif metric == 'ECGs':
            query_args['desc'] = ['ECGchannel', 'ECGs']
        elif metric == 'EOGs':
            query_args['desc'] = ['EOGchannel', 'EOGs']
        else:
            query_args['desc'] = [metric]

        # Optional session/run
        if chosen_entities['session']:
            query_args['session'] = chosen_entities['session']
        if chosen_entities['run']:
            query_args['run'] = chosen_entities['run']

        tsv_paths = list(dataset.query(**query_args))
        tsvs_to_plot_by_metric[metric] = sorted(tsv_paths)

        if len(tsv_paths) == 0:
            query_args = {
                'subj': chosen_entities['subject'],
                'task': chosen_entities['task'],
                'suffix': 'eeg',
                'extension': ['tsv', 'json', 'fif'],
                'return_type': 'filename',
                'desc': '',
                'scope': calculated_derivs_folder,
            }

            # If the user (now "all") had multiple possible descs for PSDs, ECGs, etc.
            if metric == 'PSDs':
                # Include all PSD derivatives (noise and waves) so the PSD report is
                # generated correctly.
                query_args['desc'] = ['PSDs', 'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad']
            elif metric == 'ECGs':
                query_args['desc'] = ['ECGchannel', 'ECGs']
            elif metric == 'EOGs':
                query_args['desc'] = ['EOGchannel', 'EOGs']
            else:
                query_args['desc'] = [metric]

            # Optional session/run
            if chosen_entities['session']:
                query_args['session'] = chosen_entities['session']
            if chosen_entities['run']:
                query_args['run'] = chosen_entities['run']

            tsv_paths = list(dataset.query(**query_args))
            tsvs_to_plot_by_metric[metric] = sorted(tsv_paths)
            suffix = 'eeg'
        else:
            suffix = 'meg'

        # Now query object form for ancpbids entities
        query_args['return_type'] = 'object'
        entities_obj = sorted(list(dataset.query(**query_args)), key=lambda k: k['name'])
        tsv_entities_by_metric[metric] = entities_obj

    # Convert them into a list of Deriv_to_plot objects
    derivs_to_plot = []
    for (tsv_metric, tsv_paths), (entity_metric, entity_vals) in zip(
        tsvs_to_plot_by_metric.items(),
        tsv_entities_by_metric.items()
    ):
        if tsv_metric != entity_metric:
            raise ValueError('Different metrics in tsvs_to_plot_by_metric and entities_per_file')
        if len(tsv_paths) != len(entity_vals):
            raise ValueError(f'Different number of tsvs and entities for metric: {tsv_metric}')

        for tsv_path, deriv_entities in zip(tsv_paths, entity_vals):
            file_name_in_path = os.path.basename(tsv_path).split('_meg.')[0]
            file_name_in_obj = deriv_entities['name'].split('_meg.')[0]

            if file_name_in_obj not in file_name_in_path:
                raise ValueError('Different names in tsvs_to_plot_by_metric and entities_per_file')

            deriv = Deriv_to_plot(path=tsv_path, metric=tsv_metric, deriv_entity_obj=deriv_entities)
            deriv.find_raw_entity_name()
            derivs_to_plot.append(deriv)

    # Parallel execution per subject
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(
            sub=sub,
            dataset=dataset,
            derivs_to_plot=derivs_to_plot,
            chosen_entities=chosen_entities,
            plot_settings=plot_settings,
        )
        for sub in chosen_entities['subject']
    )

    return


# ____________________________
# RUN IT:

# make_plots_meg_qc(dataset_path='/data/areer/MEG_QC_stuff/data/openneuro/ds003483')

# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds003483')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds000117')
# make_plots_meg_qc(dataset_path='/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds004330')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/camcan')

# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000246')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000247')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds002761')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds004398')


# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/BIDS/ceegridCut')
