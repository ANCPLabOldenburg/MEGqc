import os
import gc
import ancpbids
from ancpbids.query import query_entities
from ancpbids import DatasetOptions
import time
import json
import sys
import mne
import shutil
import glob
from typing import List, Union
from joblib import Parallel, delayed

# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append(os.path.join('.'))
sys.path.append(os.path.join('.', 'meg_qc', 'calculation'))

# relative path for `make html` (docs)
sys.path.append(os.path.join('..', 'meg_qc', 'calculation'))

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested inside of another, need to add one more path level here:
sys.path.append(os.path.join('..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', '..', 'meg_qc', 'calculation'))

from meg_qc.calculation.initial_meg_qc import get_all_config_params, initial_processing, get_internal_config_params, \
    delete_temp_folder
# from meg_qc.plotting.universal_html_report import make_joined_report, make_joined_report_mne
from meg_qc.plotting.universal_plots import QC_derivative

from meg_qc.calculation.metrics.STD_meg_qc import STD_meg_qc
from meg_qc.calculation.metrics.PSD_meg_qc import PSD_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc import PP_manual_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc_numba import PP_manual_meg_qc_numba
from meg_qc.calculation.metrics.Peaks_auto_meg_qc import PP_auto_meg_qc
from meg_qc.calculation.metrics.ECG_EOG_meg_qc import ECG_meg_qc, EOG_meg_qc
from meg_qc.calculation.metrics.Head_meg_qc import HEAD_movement_meg_qc
from meg_qc.calculation.metrics.muscle_meg_qc import MUSCLE_meg_qc

import os
import json
import pandas as pd
from typing import Union, Optional, Dict

from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary

def ctf_workaround(dataset, sid):
    artifacts = dataset.query(suffix="meg", return_type="object", subj=sid, scope='raw')
    # convert to folders of found files
    folders = map(lambda a: a.get_parent().get_absolute_path(), artifacts)
    # remove duplicates
    folders = set(folders)
    # convert to liust before filtering
    folders = list(folders)

    # filter for folders which end with ".ds" (including os specific path separator)
    # folders = list(filter(lambda f: f.endswith(f"{os.sep}.ds"), folders))

    # Filter for folders which end with ".ds"
    filtered_folders = [f for f in folders if f.endswith('.ds')]

    return sorted(filtered_folders)


def get_files_list(sid: str, dataset_path: str, dataset):
    """
    Different ways for fif, ctf, etc...
    Using ancpbids to get the list of files for each subject in ds.

    Parameters
    ----------
    sid : str
        Subject ID to get the files for.
    dataset_path : str
        Path to the BIDS-conform data set to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.


    Returns
    -------
    list_of_files : list
        List of paths to the .fif files for each subject.
    entities_per_file : list
        List of entities for each file in list_of_files.
    """

    has_fif = False
    has_ctf = False
    probably_eeg = False

    for root, dirs, files in os.walk(dataset_path):

        # Exclude the 'derivatives' folder.
        # Because we will later save ds info as derivative with extension .fif
        # so if we work on this ds again it might see a ctf ds as fif.
        dirs[:] = [d for d in dirs if d != 'derivatives']

        # Check for .fif files
        if any(file.endswith('.fif') for file in files):
            has_fif = True

        # Check for folders ending with .ds
        if any(dir.endswith('.ds') for dir in dirs):
            has_ctf = True

        # If both are found, no need to continue walking
        if has_fif and has_ctf:
            raise ValueError('Both fif and ctf files found in the dataset. Can not define how to read the ds.')

        if any(file.endswith('.edf') for file in files):
            probably_eeg = True

    if has_fif:
        list_of_files = sorted(
            list(dataset.query(suffix='meg', extension='.fif', return_type='filename', subj=sid, scope='raw')))

        entities_per_file = dataset.query(subj=sid, suffix='meg', extension='.fif', scope='raw')
        # sort list_of_sub_jsons by name key to get same order as list_of_files
        entities_per_file = sorted(entities_per_file, key=lambda k: k['name'])

    elif has_ctf:
        list_of_files = ctf_workaround(dataset, sid)
        entities_per_file = dataset.query(subj=sid, suffix='meg', extension='.res4', scope='raw')

        # entities_per_file is a list of Artifact objects of ancpbids created from raw files. (fif for fif files and res4 for ctf files)
        # TODO: this assumes every .ds directory has a single corresponding .res4 file.
        # Is it always so?
        # Used because I cant get entities_per_file from .ds folders, ancpbids doesnt support folder query.
        # But we need entities_per_file to pass into subject_folder.create_artifact(),
        # so that it can add automatically all the entities to the new derivative on base of entities from raw file.

        # sort list_of_sub_jsons by name key to get same order as list_of_files
        entities_per_file = sorted(entities_per_file, key=lambda k: k['name'])

    elif probably_eeg:
        list_of_files = sorted(
            list(dataset.query(suffix='eeg', extension='.edf', return_type='filename', subj=sid, scope='raw')))

        entities_per_file = dataset.query(subj=sid, suffix='eeg', extension='.edf', scope='raw')
        # sort list_of_sub_jsons by name key to get same order as list_of_files
        entities_per_file = sorted(entities_per_file, key=lambda k: k['name'])

    else:
        list_of_files = []
        raise ValueError('No fif, ctf or eeg files found in the dataset.')

    # Find if we have crosstalk in list of files and entities_per_file, give notification that they will be skipped:
    # read about crosstalk files here: https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html
    crosstalk_files = [f for f in list_of_files if 'crosstalk' in f]
    if crosstalk_files:
        print('___MEGqc___: ', 'Crosstalk files found in the list of files. They will be skipped.')

    list_of_files = [f for f in list_of_files if 'crosstalk' not in f]
    entities_per_file = [e for e in entities_per_file if 'crosstalk' not in e['name']]

    # Check if the names in list_of_files and entities_per_file are the same:
    for i in range(len(list_of_files)):
        if probably_eeg:
            file_name_in_path = os.path.basename(list_of_files[i]).split('_eeg.')[0]
            file_name_in_obj = entities_per_file[i]['name'].split('_eeg.')[0]
        else:
            file_name_in_path = os.path.basename(list_of_files[i]).split('_meg.')[0]
            file_name_in_obj = entities_per_file[i]['name'].split('_meg.')[0]

        if file_name_in_obj not in file_name_in_path:
            raise ValueError('Different names in list_of_files and entities_per_file')

    # we can also check that final file of path in list of files is same as name in jsons

    return list_of_files, entities_per_file


def create_config_artifact(derivative, config_file_path: str, f_name_to_save: str, all_taken_raw_files: List[str]):
    """
    Save the config file used for this run as a derivative.

    Note: it is important the config and json to it have the exact same name except the extention!
    The code relies on it later on in add_raw_to_config_json() function.


    Parameters
    ----------
    derivative : ancpbids.Derivative
        Derivative object to save the config file.
    config_file_path : str
        Path to the config file used for this ds conversion
    f_name_to_save : str
        Name of the config file to save.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.

    """

    # get current time stamp for config file

    timestamp = time.strftime("Date%Y%m%dTime%H%M%S")

    f_name_to_save = f_name_to_save + str(timestamp)

    config_folder = derivative.create_folder(name='config')
    config_artifact = config_folder.create_artifact()

    config_artifact.content = lambda file_path, cont=config_file_path: shutil.copy(cont, file_path)
    config_artifact.add_entity('desc', f_name_to_save)  # file name
    config_artifact.suffix = 'meg'
    config_artifact.extension = '.ini'

    # Create a seconf json file with config name as key and all taken raw files as value
    # and prepare it to be save as derivative

    config_json = {f_name_to_save: all_taken_raw_files}

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont=config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', f_name_to_save)  # file name
    config_json_artifact.suffix = 'meg'
    config_json_artifact.extension = '.json'

    return


def ask_user_rerun_subs(reuse_config_file_path: str, sub_list: List[str]):
    """
    Ask the user if he wants to rerun the same subjects again or skip them.

    Parameters
    ----------
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.
    sub_list : list
        List of subjects to run the QC on.

    Returns
    -------
    sub_list : list
        Updated list of subjects to run the QC on.

    """

    list_of_files_json, _ = get_list_of_raws_for_config(reuse_config_file_path)
    if not list_of_files_json:
        return sub_list

    # find all 'sub-' in the file names to get the subject ID:
    json_subjects_to_skip = [f.split('sub-')[1].split('_')[0] for f in list_of_files_json]

    # keep unique subjects:
    json_subjects_to_skip = list(set(json_subjects_to_skip))

    # find subjects overlapping withing current list and the json file:
    subjects_to_skip = [sub for sub in sub_list if sub in json_subjects_to_skip]

    # ask the user if he wants to skip these subjects:
    print('___MEGqc___: ', 'These requested subjects were already processed before with this config file:',
          subjects_to_skip)
    while True:
        user_input = input(
            '___MEGqc___: Do you want to RERUN these subjects with the same config parameters? (Y/N): ').lower()
        if user_input == 'n':  # remove these subs
            print('___MEGqc___: ', 'Subjects to skip:', subjects_to_skip)
            sub_list = [sub for sub in sub_list if sub not in subjects_to_skip]
            print('___MEGqc___: ', 'Subjects to process:', sub_list)
            break
        elif user_input == 'y':  # keep these subs in all_taken_raw_files
            print('___MEGqc___: ', 'Subjects to process:', sub_list)
            break
        else:  # ask again if the input is not correct
            print('___MEGqc___: ', 'Wrong input. Please enter Y or N.')

    return sub_list


def get_list_of_raws_for_config(reuse_config_file_path: str):
    """
    Get the list of all raw files processed with the config file used before.

    Parameters
    ----------
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.

    Returns
    -------
    list_of_files : list
        List of all the raw files processed in this run, for this ds.
    config_desc : str
        Description entity of the config file used before.
    """

    # exchange ini to json:
    json_for_reused_config = reuse_config_file_path.replace('.ini', '.json')

    # check if the json file exists:
    if not os.path.isfile(json_for_reused_config):
        print('___MEGqc___: ',
              'No json file found for the config file used before. Can not add the new raw files to it.')
        return

    print('___MEGqc___: ', 'json_for_reused_config', json_for_reused_config)

    try:
        with open(json_for_reused_config, 'r') as file:
            config_json = json.load(file)
    except json.JSONDecodeError as e:
        with open(json_for_reused_config, 'r') as file:
            content = file.read()
        print(f"Error decoding JSON: {e}")
        print(f"File content:\n{content}")
        # Handle the error appropriately, e.g., set config_json to an empty dict or raise an error
        config_json = {}
        return

    # from file name get desc entity to use it as a key in the json file:
    # after desc- and before the underscores:
    file_name = os.path.basename(reuse_config_file_path).split('.')[0]
    config_desc = file_name.split('desc-')[1].split('_')[0]

    # get what files already were in the config file
    list_of_files = config_json[config_desc]

    return list_of_files, config_desc


def add_raw_to_config_json(derivative, reuse_config_file_path: str, all_taken_raw_files: List[str]):
    """
    Add the list of all taken raw files to the existing list of used settings in the config file.

    Expects that the config file .ini and the .json file (with the same name) are already saved as derivatives.

    To get corresponding json here use the easy way:
    just exchange ini to json in reuse file path (not using ANCPbids for it).
    The 'proper' way would be to:
    - query the desc entitiy of the reused config file
    - get the json file with the same desc entity
    This way will still assume that desc are exactly the same, so we use the easy way without ANCPbids d-tour.

    The function will also output the updated list of all taken raw files for this ds based on the users choice:
    rewrite or not the subjects that have already been processed with this config file.

    Parameters
    ----------
    derivative : ancpbids.Derivative
        Derivative object to save the config file.
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.

    Returns
    -------
    all_taken_raw_files : list
        Updated list of all the raw files processed in this run, for this ds.

    """

    list_of_files, config_desc = get_list_of_raws_for_config(reuse_config_file_path)

    # Continue to update the list with new files:
    list_of_files += all_taken_raw_files

    # sort and remove duplicates:
    list_of_files = sorted(list(set(list_of_files)))

    # overwrite the old json (premake ancp bids artifact):
    config_json = {config_desc: list_of_files}

    config_folder = derivative.create_folder(name='config')
    # TODO: we dont need to create config folder again, already got it, how to get it?

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont=config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', config_desc)  # file name
    config_json_artifact.suffix = 'meg'
    config_json_artifact.extension = '.json'

    return all_taken_raw_files


def check_ds_paths(ds_paths: Union[List[str], str]):
    """
    Check if the given paths to the data sets exist.

    Parameters
    ----------
    ds_paths : list or str
        List of paths to the BIDS-conform data sets to run the QC on.

    Returns
    -------
    ds_paths : list
        List of paths to the BIDS-conform data sets to run the QC on.
    """

    # has to be a list, even if there is just one path:
    if isinstance(ds_paths, str):
        ds_paths = [ds_paths]

    # make sure all directories in the list exist:
    for ds_path in ds_paths:
        if not os.path.isdir(ds_path):
            raise ValueError(f'Given path to the dataset does not exist. Path: {ds_path}')

    return ds_paths


def check_config_saved_ask_user(dataset):
    """
    Check if there is already config file used for this ds:
    If yes - ask the user if he wants to use it again. If not - use default one.
    When no config found or user doesnt want to reuse - will return None.
    otherwise will return the path to one config file used for this ds before to reuse now.

    Parameters
    ----------
    dataset : ancpbids.Dataset
        Dataset object to work with.

    Returns
    -------
    config_file_path : str
        Path to the config file used for this ds conversion.
    """

    # if os.path.isfile(os.path.join(derivatives_path, 'config', 'UsedSettings.ini')):
    #     print('___MEGqc___: ', 'There is already a config file used for this data set. Do you want to use it again?')
    #     #ask user if he wants to use the same config file again

    entities = query_entities(dataset, scope='derivatives')

    # print('___MEGqc___: ', 'entities', entities)

    # search if there is already a derivative with 'UsedSettings' in the name
    # if yes - ask the user if he wants to use it again. If not - use default one.
    used_settings_entity_list = []
    for key, entity_set in entities.items():
        if key == 'description':
            for ent in entity_set:
                if 'usedsettings' in ent.lower():
                    used_settings_entity_list.append(ent)

    used_setting_file_list = []
    for used_settings_entity in used_settings_entity_list:
        used_setting_file_list += sorted(list(
            dataset.query(suffix='meg', extension='.ini', desc=used_settings_entity, return_type='filename',
                          scope='derivatives')))

    reuse_config_file_path = None

    # Ask the user if he wants to use any of existing config files:
    if used_setting_file_list:
        print('___MEGqc___: ',
              'There are already config files used for this data set. Do you want to use any of them again?')
        print('___MEGqc___: ', 'List of the config files previously used for this data set:')
        for i, file in enumerate(used_setting_file_list):
            print('___MEGqc___: ', i, file)

        user_input = input(
            '___MEGqc___: Enter the number of the config file you want to use, or press Enter to use the default one: ')
        if user_input:
            reuse_config_file_path = used_setting_file_list[int(user_input)]
        else:
            print('___MEGqc___: ', 'You chose to use the default config file.')

    return reuse_config_file_path


def check_sub_list(sub_list: Union[List[str], str], dataset):
    """
    Check if the given subjects are in the data set.

    Parameters
    ----------
    sub_list : list or str
        List of subjects to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.

    Returns
    -------
    sub_list : list
        Updated list of subjects to run the QC on.

    """

    available_subs = sorted(list(dataset.query_entities(scope='raw')['subject']))
    if sub_list == 'all':
        sub_list = available_subs
    elif isinstance(sub_list, str) and sub_list != 'all':
        sub_list = [sub_list]
        # check if this sub is available:
        if sub_list[0] not in available_subs:
            print('___MEGqc___: ',
                  'The subject you want to run the QC on is not in your data set. Check the subject ID.')
            return
    elif isinstance(sub_list, list):
        # if they are given as str - IDs:
        if all(isinstance(sub, str) for sub in sub_list):
            sub_list_missing = [sub for sub in sub_list if sub not in available_subs]
            sub_list = [sub for sub in sub_list if sub in available_subs]
            if sub_list_missing:
                print('___MEGqc___: ', 'Could NOT find these subs in your data set. Check the subject IDs:',
                      sub_list_missing)
                print('___MEGqc___: ', 'Requested subjects found in your data set:', sub_list,
                      'Only these subjects will be processed.')

        # if they are given as int - indexes:
        elif all(isinstance(sub, int) for sub in sub_list):
            sub_list = [available_subs[i] for i in sub_list]

    print('___MEGqc___: ', 'Requested sub_list to process: ', sub_list)

    return sub_list


def process_one_subject(
        sub: str,
        dataset,
        dataset_path: str,
        all_qc_params: dict,
        internal_qc_params: dict
):
    """
    This function processes a single subject. It contains all the code that was
    originally inside the 'for sub in sub_list:' loop in 'make_derivative_meg_qc'.

    Parameters
    ----------
    sub : str
        Single subject ID string (e.g. '009').
    dataset : ancpbids.dataset
        BIDS-conform dataset loaded by ancpbids.
    dataset_path : str
        Path to the BIDS dataset.
    all_qc_params : dict
        QC parameters from user config file.
    internal_qc_params : dict
        Internal QC parameters that users do not change.
    """

    # We replicate everything that was inside the loop.

    # CREATE DERIVATIVE FOR THIS SUBJECT
    derivative = dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    print('___MEGqc___: ', 'Take SUB: ', sub)

    calculation_folder = derivative.create_folder(name='calculation')
    subject_folder = calculation_folder.create_folder(
        type_=dataset.get_schema().Subject,
        name='sub-' + sub
    )

    # GET FILE LIST FOR THIS SUBJECT
    list_of_files, entities_per_file = get_files_list(sub, dataset_path, dataset)

    if not list_of_files:
        print('___MEGqc___: ',
              'No files to work on. Check that given subjects are present in your data set.')
        return  # Stop if no files exist for this subject

    print('___MEGqc___: ', 'list_of_files to process:', list_of_files)
    print('___MEGqc___: ', 'entities_per_file to process:', entities_per_file)
    print('___MEGqc___: ', 'TOTAL files to process: ', len(list_of_files))

    # Keep track of all raw files processed for this subject (optional)
    all_taken_raw_files = [os.path.basename(f) for f in list_of_files]

    # Preassign in case nothing is processed
    raw = None

    # Counters, accumulators
    counter = 0
    avg_ecg = []
    avg_eog = []

    # LOOP OVER FIF FILES FOR THIS SUBJECT
    for file_ind, data_file in enumerate(list_of_files):  # e.g. [0:1] in your example

        print('___MEGqc___: ', 'Processing file: ', data_file)

        # Preassign strings with notes for the user (just as in your code)
        shielding_str, m_or_g_skipped_str, epoching_str = '', '', ''
        ecg_str, eog_str, head_str, muscle_str = '', '', '', ''
        pp_manual_str, pp_auto_str, std_str, psd_str = '', '', '', ''

        print('___MEGqc___: ', 'Starting initial processing...')
        start_time = time.time()

        # INITIAL PROCESSING
        (meg_system,
         dict_epochs_mg,
         chs_by_lobe,
         channels,
         raw_cropped_filtered,
         raw_cropped_filtered_resampled,
         raw_cropped,
         raw,
         info_derivs,
         stim_deriv,
         shielding_str,
         epoching_str,
         sensors_derivs,
         m_or_g_chosen,
         m_or_g_skipped_str,
         lobes_color_coding_str,
         resample_str,
         orig_meg_system
         ) = initial_processing(
            default_settings=all_qc_params['default'],
            filtering_settings=all_qc_params['Filtering'],
            epoching_params=all_qc_params['Epoching'],
            file_path=data_file,
            dataset_path=dataset_path
        )

        print('___MEGqc___: ',
              "Finished initial processing. --- Execution %s seconds ---"
              % (time.time() - start_time))

        # PREDEFINE VARIABLES FOR QC
        noisy_freqs_global = None
        std_derivs, psd_derivs = [], []
        pp_manual_derivs, pp_auto_derivs = [], []
        ecg_derivs, eog_derivs = [], []
        head_derivs, muscle_derivs = [], []
        simple_metrics_psd, simple_metrics_std = [], []
        simple_metrics_pp_manual, simple_metrics_pp_auto = [], []
        simple_metrics_ecg, simple_metrics_eog = [], []
        simple_metrics_head, simple_metrics_muscle = [], []

        # 1) STD
        if all_qc_params['default']['run_STD'] is True:
            print('___MEGqc___: ', 'Starting STD...')
            start_time = time.time()
            (std_derivs,
             simple_metrics_std,
             std_str) = STD_meg_qc(
                all_qc_params['STD'],
                channels,
                chs_by_lobe,
                dict_epochs_mg,
                raw_cropped_filtered_resampled,
                m_or_g_chosen
            )
            if orig_meg_system == 'EEG':
                # 1. Chanmge the value of measurement_unit_mag
                simple_metrics_std['measurement_unit_mag'] = 'microV'
                # 2. remove measurement_unit_grad
                del simple_metrics_std['measurement_unit_grad']

            print('___MEGqc___: ',
                  "Finished STD. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 2) PSD
        if all_qc_params['default']['run_PSD'] is True:
            print('___MEGqc___: ', 'Starting PSD...')
            start_time = time.time()
            (psd_derivs,
             simple_metrics_psd,
             psd_str,
             noisy_freqs_global) = PSD_meg_qc(
                all_qc_params['PSD'],
                internal_qc_params['PSD'],
                channels,
                chs_by_lobe,
                raw_cropped_filtered,
                m_or_g_chosen,
                helper_plots=False
            )
            if orig_meg_system == 'EEG':
                del noisy_freqs_global['grad']
                # 1. Chanmge the value of measurement_unit_mag
                simple_metrics_psd['measurement_unit_mag'] = '(microV)^2/Hz'
                # 2. remove measurement_unit_grad
                del simple_metrics_psd['measurement_unit_grad']

            print('___MEGqc___: ',
                  "Finished PSD. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 3) Peak‑to‑Peak manual
        if all_qc_params['default']['run_PTP_manual'] is True:
            start_time = time.time()

            # choose the implementation ----------------------------------
            if all_qc_params['PTP_manual']['numba_version'] is True:
                print('___MEGqc___: ', 'Starting Peak‑to‑Peak manual (Numba)...')
                func = PP_manual_meg_qc_numba  #  accelerated version
            else:
                print('___MEGqc___: ', 'Starting Peak‑to‑Peak manual...')
                func = PP_manual_meg_qc  # standard version
            # -------------------------------------------------------------

            (pp_manual_derivs,
             simple_metrics_pp_manual,
             pp_manual_str) = func(
                all_qc_params['PTP_manual'],
                channels,
                chs_by_lobe,
                dict_epochs_mg,
                raw_cropped_filtered_resampled,
                m_or_g_chosen
            )
            if orig_meg_system == 'EEG':
                # 1. Chanmge the value of measurement_unit_mag
                simple_metrics_pp_manual['measurement_unit_mag'] = 'microV'
                # 2. remove measurement_unit_grad
                del simple_metrics_pp_manual['measurement_unit_grad']

            print('___MEGqc___: ',
                  "Finished Peak‑to‑Peak manual. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 4) Peak-to-Peak auto from MNE
        if all_qc_params['default']['run_PTP_auto_mne'] is True:
            print('___MEGqc___: ', 'Starting Peak-to-Peak auto...')
            start_time = time.time()
            (pp_auto_derivs,
             bad_channels,
             pp_auto_str) = PP_auto_meg_qc(
                all_qc_params['PTP_auto'],
                channels,
                raw_cropped_filtered_resampled,
                m_or_g_chosen
            )
            print('___MEGqc___: ',
                  "Finished Peak-to-Peak auto. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 5) ECG
        if all_qc_params['default']['run_ECG'] is True:
            print('___MEGqc___: ', 'Starting ECG...')
            start_time = time.time()
            (ecg_derivs,
             simple_metrics_ecg,
             ecg_str,
             avg_objects_ecg) = ECG_meg_qc(
                all_qc_params['ECG'],
                internal_qc_params['ECG'],
                raw_cropped,
                channels,
                chs_by_lobe,
                m_or_g_chosen,
                orig_meg_system
            )
            print('___MEGqc___: ',
                  "Finished ECG. --- Execution %s seconds ---"
                  % (time.time() - start_time))

            avg_ecg += avg_objects_ecg

        # 6) EOG
        if all_qc_params['default']['run_EOG'] is True:
            print('___MEGqc___: ', 'Starting EOG...')
            start_time = time.time()
            (eog_derivs,
             simple_metrics_eog,
             eog_str,
             avg_objects_eog) = EOG_meg_qc(
                all_qc_params['EOG'],
                internal_qc_params['EOG'],
                raw_cropped,
                channels,
                chs_by_lobe,
                m_or_g_chosen,
                orig_meg_system
            )
            if orig_meg_system == 'EEG':
                # 2. remove measurement_unit_grad if present
                try:
                    del simple_metrics_eog['all_channels_ranked_by_EOG_contamination_level']['grad']
                except Exception:
                    None
            print('___MEGqc___: ',
                  "Finished EOG. --- Execution %s seconds ---"
                  % (time.time() - start_time))

            avg_eog += avg_objects_eog

        # 7) Head movement artifacts
        if all_qc_params['default']['run_Head'] is True:
            print('___MEGqc___: ', 'Starting Head movement calculation...')
            (head_derivs,
             simple_metrics_head,
             head_str,
             df_head_pos,
             head_pos) = HEAD_movement_meg_qc(raw_cropped)
            print('___MEGqc___: ',
                  "Finished Head movement calculation. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 8) Muscle artifacts
        if all_qc_params['default']['run_Muscle'] is True:
            print('___MEGqc___: ', 'Starting Muscle artifacts calculation...')
            start_time = time.time()
            (muscle_derivs,
             simple_metrics_muscle,
             muscle_str,
             scores_muscle_all3) = MUSCLE_meg_qc(
                all_qc_params['Muscle'],
                all_qc_params['PSD'],
                internal_qc_params['PSD'],
                channels,
                raw_cropped_filtered,
                noisy_freqs_global,
                m_or_g_chosen,
                dataset_path,
                attach_dummy=True,
                cut_dummy=True,
                orig_meg_type=orig_meg_system,
             )
            if orig_meg_system == 'EEG':
                # 1. Chanmge the value of measurement_unit_mag
                simple_metrics_muscle['muscle_calculated_using'] = 'electrode'
            # Store the total number of events analyzed so we can later express
            # the number of detected artifacts as a percentage.  The first
            # derivative contains a TSV table where each row corresponds to one
            # event that was evaluated during muscle detection.
            if muscle_derivs:
                total_events_for_muscle = muscle_derivs[0].content.shape[0]
                simple_metrics_muscle["total_number_of_events"] = int(
                    total_events_for_muscle
                )
            print('___MEGqc___: ',
                  "Finished Muscle artifacts calculation. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # REPORT STRINGS
        report_strings = {
            'INITIAL_INFO': (m_or_g_skipped_str + resample_str + epoching_str +
                             shielding_str + lobes_color_coding_str),
            'STD': std_str,
            'PSD': psd_str,
            'PTP_MANUAL': pp_manual_str,
            'PTP_AUTO': pp_auto_str,
            'ECG': ecg_str,
            'EOG': eog_str,
            'HEAD': head_str,
            'MUSCLE': muscle_str,
            'STIMULUS': 'If the data was cropped for this calculation, the stimulus data is also cropped.'
        }

        report_str_derivs = [QC_derivative(report_strings, 'ReportStrings', 'json')]

        # ORGANIZE QC DERIVATIVES
        QC_derivs = {
            'Raw info': info_derivs,
            'Stimulus channels': stim_deriv,
            'Report_strings': report_str_derivs,
            'Sensors locations': sensors_derivs,
            'Standard deviation of the data': std_derivs,
            'Frequency spectrum': psd_derivs,
            'Peak-to-Peak manual': pp_manual_derivs,
            'Peak-to-Peak auto from MNE': pp_auto_derivs,
            'ECG': ecg_derivs,
            'EOG': eog_derivs,
            'Head movement artifacts': head_derivs,
            'High frequency (Muscle) artifacts': muscle_derivs
        }

        QC_simple = {
            'STD': simple_metrics_std,
            'PSD': simple_metrics_psd,
            'PTP_MANUAL': simple_metrics_pp_manual,
            'PTP_AUTO': simple_metrics_pp_auto,
            'ECG': simple_metrics_ecg,
            'EOG': simple_metrics_eog,
            'HEAD': simple_metrics_head,
            'MUSCLE': simple_metrics_muscle
        }

        QC_derivs['Simple_metrics'] = [QC_derivative(QC_simple, 'SimpleMetrics', 'json')]

        # SAVE DERIVATIVES (EXCEPT MATPLOTLIB, PLOTLY, REPORT)
        for section in (sec for sec in QC_derivs.values() if sec):
            for deriv in (
                    d for d in section
                    if d.content_type not in ['matplotlib', 'plotly', 'report']
            ):
                meg_artifact = subject_folder.create_artifact(raw=entities_per_file[file_ind])
                counter += 1
                print('___MEGqc___: ', 'counter of subject_folder.create_artifact', counter)

                meg_artifact.add_entity('desc', deriv.name)  # file name
                if orig_meg_system == 'EEG':
                    meg_artifact.suffix = 'eeg'
                else:
                    meg_artifact.suffix = 'meg'
                meg_artifact.extension = '.html'

                if deriv.content_type == 'df':
                    meg_artifact.extension = '.tsv'
                    meg_artifact.content = lambda file_path, cont=deriv.content: cont.to_csv(
                        file_path, sep='\t'
                    )

                elif deriv.content_type == 'json':
                    meg_artifact.extension = '.json'

                    def json_writer(file_path, cont=deriv.content):
                        with open(file_path, "w") as file_wrapper:
                            json.dump(cont, file_wrapper, indent=4)

                    meg_artifact.content = json_writer

                elif deriv.content_type == 'info':
                    meg_artifact.extension = '.fif'
                    meg_artifact.content = lambda file_path, cont=deriv.content: mne.io.write_info(
                        file_path, cont
                    )
                else:
                    print('___MEGqc___: ', meg_artifact.name)
                    meg_artifact.content = 'dummy text'
                    meg_artifact.extension = '.txt'

        # CLEAN UP TEMP FILES
        try:
            os.remove(raw_cropped)
            os.remove(raw_cropped_filtered)
            os.remove(raw_cropped_filtered_resampled)

            del (meg_system, dict_epochs_mg, chs_by_lobe, channels,
                 raw_cropped_filtered, raw_cropped_filtered_resampled,
                 raw_cropped, info_derivs, stim_deriv, shielding_str,
                 epoching_str, sensors_derivs, m_or_g_chosen, m_or_g_skipped_str,
                 lobes_color_coding_str, resample_str)
            gc.collect()
            print('REMOVING TRASH: SUCCEEDED')
        except:
            print('REMOVING TRASH: FAILED')

    # WRITE DERIVATIVE
    ancpbids.write_derivative(dataset, derivative)



    # Removes intermediate trash objects
    del meg_artifact, derivative
    gc.collect()

    # Check if raw is None => means we never processed a file
    try:
        if raw is None:
            print('___MEGqc___: ', 'No data files could be processed for subject:', sub)
            return
    except:
        print('___MEGqc___: ', 'No data files could be processed for subject:', sub)

    # You can return whatever you want from here
    return all_taken_raw_files, orig_meg_system


def process_one_subject_safe(
        sub: str,
        dataset,
        dataset_path: str,
        all_qc_params: dict,
        internal_qc_params: dict):
    """Wrapper around :func:`process_one_subject` that catches errors.

    Parameters are identical to :func:`process_one_subject`.
    The function returns a tuple ``(sub, result)`` where ``result`` is
    ``None`` if the processing failed for this subject.
    """
#    try:
    result, orig_meg_system = process_one_subject(
        sub=sub,
        dataset=dataset,
        dataset_path=dataset_path,
        all_qc_params=all_qc_params,
        internal_qc_params=internal_qc_params,
    )
    return sub, result, orig_meg_system
    # except Exception as e:  # Catch any error so the parallel job continues
    #     print(f"___MEGqc___: Error processing subject {sub}: {e}")
    #     return sub, None


def _parse_count_percent(val: str):
    """Return ``(count, percent)`` from strings like ``"10 (5.0%)"``."""
    if not isinstance(val, str):
        return val, None
    try:
        if "(" in val and "%" in val:
            count_str, rest = val.split("(", 1)
            count = float(count_str.strip())
            percent = float(rest.strip().strip(")% "))
            return count, percent
        if val.endswith("%"):
            return None, float(val.strip("%"))
        return float(val), None
    except Exception:
        return None, None


def _parse_percent(val: str):
    """Parse a percentage string like ``"10.5%"``."""
    if isinstance(val, str):
        try:
            return float(val.strip().strip("%"))
        except Exception:
            return None
    return float(val)


def flatten_summary_metrics(js: dict) -> dict:
    """Flatten one GlobalSummaryReport JSON into numeric columns."""
    row = {}
    if js.get("GQI") is not None:
        row["GQI"] = js.get("GQI")

    for item in js.get("STD_time_series", []):
        metric = item.get("Metric", "").replace(" ", "_").lower()
        stype = list(item.keys())[1]
        if stype == 'ELECTRODES':
            num_mag, pct_mag = _parse_count_percent(item.get(stype, ""))
            num_grad = []
            pct_grad = []
        else:
            num_mag, pct_mag = _parse_count_percent(item.get("MAGNETOMETERS", ""))
            num_grad, pct_grad = _parse_count_percent(item.get("GRADIOMETERS", ""))
        row[f"STD_ts_{metric}_mag_num"] = num_mag
        row[f"STD_ts_{metric}_mag_percentage"] = pct_mag
        row[f"STD_ts_{metric}_grad_num"] = num_grad
        row[f"STD_ts_{metric}_grad_percentage"] = pct_grad

    for item in js.get("PTP_time_series", []):
        metric = item.get("Metric", "").replace(" ", "_").lower()
        if stype == 'ELECTRODES':
            num_mag, pct_mag = _parse_count_percent(item.get(stype, ""))
            num_grad = []
            pct_grad = []
        else:
            num_mag, pct_mag = _parse_count_percent(item.get("MAGNETOMETERS", ""))
            num_grad, pct_grad = _parse_count_percent(item.get("GRADIOMETERS", ""))
        row[f"PTP_ts_{metric}_mag_num"] = num_mag
        row[f"PTP_ts_{metric}_mag_percentage"] = pct_mag
        row[f"PTP_ts_{metric}_grad_num"] = num_grad
        row[f"PTP_ts_{metric}_grad_percentage"] = pct_grad

    for item in js.get("STD_epoch_summary", []):
        st = item.get("Sensor Type")
        if st is None:
            sensor = None
            num_noisy = []; pct_noisy = []; num_flat = []; pct_flat = []
        else:
            if st == "MAGNETOMETERS":
                nm = "mag"
            elif st == "GRADIOMETERS":
                nm = "grad"
            elif st == "ELECTRODES":
                nm = "eeg"
            else:
                nm = None
            sensor = nm
            num_noisy, pct_noisy = _parse_count_percent(item.get("Noisy Epochs", ""))
            num_flat, pct_flat = _parse_count_percent(item.get("Flat Epochs", ""))
        row[f"STD_ep_{sensor}_noisy_num"] = num_noisy
        row[f"STD_ep_{sensor}_noisy_percentage"] = pct_noisy
        row[f"STD_ep_{sensor}_flat_num"] = num_flat
        row[f"STD_ep_{sensor}_flat_percentage"] = pct_flat

    for item in js.get("PTP_epoch_summary", []):
        st = item.get("Sensor Type")
        if st == "MAGNETOMETERS":
            nm = "mag"
        elif st == "GRADIOMETERS":
            nm = "eeg"
        elif st == "ELECTRODES":
            nm = "eeg"
        else:
            nm = "None"
        sensor = nm
        num_noisy, pct_noisy = _parse_count_percent(item.get("Noisy Epochs", ""))
        num_flat, pct_flat = _parse_count_percent(item.get("Flat Epochs", ""))
        row[f"PTP_ep_{sensor}_noisy_num"] = num_noisy
        row[f"PTP_ep_{sensor}_noisy_percentage"] = pct_noisy
        row[f"PTP_ep_{sensor}_flat_num"] = num_flat
        row[f"PTP_ep_{sensor}_flat_percentage"] = pct_flat

    for item in js.get("ECG_correlation_summary", []):
        st = item.get("Sensor Type")
        if st == "MAGNETOMETERS":
            nm = "mag"
        elif st == "GRADIOMETERS":
            nm = "eeg"
        elif st == "ELECTRODES":
            nm = "eeg"
        else:
            nm = "None"
        sensor = nm
        num, pct = _parse_count_percent(item.get("# |High Correlations| > 0.8", ""))
        total = item.get("Total Channels")
        row[f"ECG_{sensor}_high_corr_num"] = num
        row[f"ECG_{sensor}_high_corr_percentage"] = pct
        row[f"ECG_{sensor}_total_channels"] = total

    for item in js.get("EOG_correlation_summary", []):
        st = item.get("Sensor Type")
        if st == "MAGNETOMETERS":
            nm = "mag"
        elif st == "GRADIOMETERS":
            nm = "eeg"
        elif st == "ELECTRODES":
            nm = "eeg"
        else:
            nm = "None"
        sensor = nm
        num, pct = _parse_count_percent(item.get("# |High Correlations| > 0.8", ""))
        total = item.get("Total Channels")
        row[f"EOG_{sensor}_high_corr_num"] = num
        row[f"EOG_{sensor}_high_corr_percentage"] = pct
        row[f"EOG_{sensor}_total_channels"] = total

    for item in js.get("PSD_noise_summary", []):
        st = item.get("Sensor Type")
        if st == "MAGNETOMETERS":
            nm = "mag"
        elif st == "GRADIOMETERS":
            nm = "eeg"
        elif st == "ELECTRODES":
            nm = "eeg"
        else:
            nm = "None"
        sensor = nm
        row["PSD_noise_mag_percentage"] = _parse_percent(item.get(nm, "0"))
        row["PSD_noise_grad_percentage"] = _parse_percent(item.get(nm, "0"))

    muscle = js.get("Muscle_events", {})
    row["Muscle_events_num"] = muscle.get("# Muscle Events")
    row["Muscle_events_total"] = muscle.get("total_number_of_events")

    for key, val in js.get("GQI_penalties", {}).items():
        row[f"GQI_penalty_{key}"] = val

    for key, val in js.get("GQI_metrics", {}).items():
        row[f"GQI_{key}"] = val

    for key, val in js.get("parameters", {}).items():
        row[f"param_{key}"] = val

    return row


def make_derivative_meg_qc(
        default_config_file_path: str,
        internal_config_file_path: str,
        ds_paths: Union[List[str], str],
        sub_list: Union[List[str], str] = 'all',
        n_jobs: int = 5  # Number of parallel jobs
):
    ds_paths = check_ds_paths(ds_paths)
    internal_qc_params = get_internal_config_params(internal_config_file_path)

    for dataset_path in ds_paths:
        print('___MEGqc___: ', 'DS path:', dataset_path)
        dataset = ancpbids.load_dataset(dataset_path, DatasetOptions(lazy_loading=True))
        schema = dataset.get_schema()

        derivatives_path = os.path.join(dataset_path, 'derivatives')
        if not os.path.isdir(derivatives_path):
            os.mkdir(derivatives_path)

        reuse_config_file_path = check_config_saved_ask_user(dataset)
        if reuse_config_file_path:
            config_file_path = reuse_config_file_path
        else:
            config_file_path = default_config_file_path
        print('___MEGqc___: ', 'Using config file: ', config_file_path)

        all_qc_params = get_all_config_params(config_file_path)
        if all_qc_params is None:
            return

        # Determine which subjects to run
        sub_list = check_sub_list(sub_list, dataset)
        if reuse_config_file_path:
            sub_list = ask_user_rerun_subs(reuse_config_file_path, sub_list)

        # Parallel execution over subjects
        # Each subject is processed by process_one_subject_safe() in parallel
        # with n_jobs specifying how many workers to run simultaneously
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_one_subject_safe)(
                sub=sub,
                dataset=dataset,
                dataset_path=dataset_path,
                all_qc_params=all_qc_params,
                internal_qc_params=internal_qc_params
            )
            for sub in sub_list
        )

        # for sub in sub_list:
        #     process_one_subject(
        #         sub=sub,
        #         dataset=dataset,
        #         dataset_path=dataset_path,
        #         all_qc_params=all_qc_params,
        #         internal_qc_params=internal_qc_params
        #     )
        # Optionally, you can handle the returned values here, e.g.,:
        # global_all_taken_raw_files = []
        # global_avg_ecg = []
        # global_avg_eog = []
        # for res in results:
        #     if res is not None:
        #         taken_files, ecg_data, eog_data, raw_obj = res
        #         global_all_taken_raw_files += taken_files
        #         global_avg_ecg += ecg_data
        #         global_avg_eog += eog_data

        dum1, dum2, data_type = results[0]
        results = [item[:2] for item in results]

        # Collect results and log subjects that failed
        excluded_subjects = [sub for sub, files in results if files is None]

        # Remove temporary folder of intermediate files
        delete_temp_folder(dataset_path)

        # Save config file used for this run as a derivative:
        all_subs_raw_files = []
        for sub, subj_files in results:
            if subj_files is not None:
                all_subs_raw_files.extend(subj_files)

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

        if reuse_config_file_path is None:
            # if no config file was used before, save the one used now
            create_config_artifact(derivative, config_file_path, 'UsedSettings', all_subs_raw_files)
        else:
            # otherwise - dont save config again, but add list of all taken raw files to the existing list of used settings:
            add_raw_to_config_json(derivative, reuse_config_file_path, all_subs_raw_files)

        # Write the pipeline-level derivative to disk
        ancpbids.write_derivative(dataset, derivative)

        # Save list of excluded subjects
        if excluded_subjects:
            excl_path = os.path.join(dataset_path, 'derivatives', 'Meg_QC', 'excluded_subjects')
            os.makedirs(os.path.dirname(excl_path), exist_ok=True)
            with open(excl_path, 'w', encoding='utf-8') as f:
                for sub in excluded_subjects:
                    f.write(str(sub) + '\n')

        # Generate Global Quality Index reports and group table
        try:
            generate_gqi_summary(dataset_path, config_file_path, data_type)
        except Exception as e:
            print("___MEGqc___: Failed to create global quality reports", e)
