import os
import re
import glob
import shutil
import gc
import mne
import configparser
import numpy as np
import pandas as pd
import random
import copy
import warnings
from typing import List
from meg_qc.calculation.objects import QC_derivative, MEG_channel


def get_all_config_params(config_file_path: str):
    """
    Parse all the parameters from config and put into a python dictionary
    divided by sections. Parsing approach can be changed here, which
    will not affect working of other fucntions.


    Parameters
    ----------
    config_file_path: str
        The path to the config file.

    Returns
    -------
    all_qc_params: dict
        A dictionary with all the parameters from the config file.

    """

    all_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_path)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['ch_types']
    m_or_g_chosen = [chosen.strip() for chosen in m_or_g_chosen.split(",")]
    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEGqc___: ', 'No channels to analyze. Check parameter ch_types in config file.')
        return None

    # TODO: save list of mags and grads here and use later everywhere? because for CTF types are messed up.

    run_STD = default_section.getboolean('STD')
    run_PSD = default_section.getboolean('PSD')
    run_PTP_manual = default_section.getboolean('PTP_manual')
    run_PTP_auto_mne = default_section.getboolean('PTP_auto_mne')
    run_ECG = default_section.getboolean('ECG')
    run_EOG = default_section.getboolean('EOG')
    run_Head = default_section.getboolean('Head')
    run_Muscle = default_section.getboolean('Muscle')

    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']
    try:
        if not tmin:
            tmin = 0
        else:
            tmin = float(tmin)
        if not tmax:
            tmax = None
        else:
            tmax = float(tmax)

        default_params = dict({
            'm_or_g_chosen': m_or_g_chosen,
            'run_STD': run_STD,
            'run_PSD': run_PSD,
            'run_PTP_manual': run_PTP_manual,
            'run_PTP_auto_mne': run_PTP_auto_mne,
            'run_ECG': run_ECG,
            'run_EOG': run_EOG,
            'run_Head': run_Head,
            'run_Muscle': run_Muscle,
            'plot_mne_butterfly': default_section.getboolean('plot_mne_butterfly'),
            'plot_interactive_time_series': default_section.getboolean('plot_interactive_time_series'),
            'plot_interactive_time_series_average': default_section.getboolean('plot_interactive_time_series_average'),
            'crop_tmin': tmin,
            'crop_tmax': tmax})
        all_qc_params['default'] = default_params

        filtering_section = config['Filtering']
        try:
            lfreq = filtering_section.getfloat('l_freq')
        except:
            lfreq = None

        try:
            hfreq = filtering_section.getfloat('h_freq')
        except:
            hfreq = None

        all_qc_params['Filtering'] = dict({
            'apply_filtering': filtering_section.getboolean('apply_filtering'),
            'l_freq': lfreq,
            'h_freq': hfreq,
            'method': filtering_section['method'],
            'downsample_to_hz': filtering_section.getint('downsample_to_hz')})

        epoching_section = config['Epoching']
        stim_channel = epoching_section['stim_channel']
        stim_channel = stim_channel.replace(" ", "")
        stim_channel = stim_channel.split(",")
        if stim_channel == ['']:
            stim_channel = None

        epoching_params = dict({
            'event_dur': epoching_section.getfloat('event_dur'),
            'epoch_tmin': epoching_section.getfloat('epoch_tmin'),
            'epoch_tmax': epoching_section.getfloat('epoch_tmax'),
            'stim_channel': stim_channel,
            'event_repeated': epoching_section['event_repeated']})
        all_qc_params['Epoching'] = epoching_params

        std_section = config['STD']
        all_qc_params['STD'] = dict({
            'std_lvl': std_section.getint('std_lvl'),
            'allow_percent_noisy_flat_epochs': std_section.getfloat('allow_percent_noisy_flat_epochs'),
            'noisy_channel_multiplier': std_section.getfloat('noisy_channel_multiplier'),
            'flat_multiplier': std_section.getfloat('flat_multiplier'), })

        psd_section = config['PSD']
        freq_min = psd_section['freq_min']
        freq_max = psd_section['freq_max']
        if not freq_min:
            freq_min = 0
        else:
            freq_min = float(freq_min)
        if not freq_max:
            freq_max = np.inf
        else:
            freq_max = float(freq_max)

        all_qc_params['PSD'] = dict({
            'freq_min': freq_min,
            'freq_max': freq_max,
            'psd_step_size': psd_section.getfloat('psd_step_size')})

        ptp_manual_section = config['PTP_manual']
        all_qc_params['PTP_manual'] = dict({
            'numba_version': ptp_manual_section.getboolean('numba_version'),
            'max_pair_dist_sec': ptp_manual_section.getfloat('max_pair_dist_sec'),
            'ptp_thresh_lvl': ptp_manual_section.getfloat('ptp_thresh_lvl'),
            'allow_percent_noisy_flat_epochs': ptp_manual_section.getfloat('allow_percent_noisy_flat_epochs'),
            'ptp_top_limit': ptp_manual_section.getfloat('ptp_top_limit'),
            'ptp_bottom_limit': ptp_manual_section.getfloat('ptp_bottom_limit'),
            'std_lvl': ptp_manual_section.getfloat('std_lvl'),
            'noisy_channel_multiplier': ptp_manual_section.getfloat('noisy_channel_multiplier'),
            'flat_multiplier': ptp_manual_section.getfloat('flat_multiplier')})

        ptp_mne_section = config['PTP_auto']
        all_qc_params['PTP_auto'] = dict({
            'peak_m': ptp_mne_section.getfloat('peak_m'),
            'flat_m': ptp_mne_section.getfloat('flat_m'),
            'peak_g': ptp_mne_section.getfloat('peak_g'),
            'flat_g': ptp_mne_section.getfloat('flat_g'),
            'bad_percent': ptp_mne_section.getint('bad_percent'),
            'min_duration': ptp_mne_section.getfloat('min_duration')})

        ecg_section = config['ECG']
        all_qc_params['ECG'] = dict({
            'drop_bad_ch': ecg_section.getboolean('drop_bad_ch'),
            'n_breaks_bursts_allowed_per_10min': ecg_section.getint('n_breaks_bursts_allowed_per_10min'),
            'allowed_range_of_peaks_stds': ecg_section.getfloat('allowed_range_of_peaks_stds'),
            'norm_lvl': ecg_section.getfloat('norm_lvl'),
            'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
            'thresh_lvl_peakfinder': ecg_section.getfloat('thresh_lvl_peakfinder'),
            'height_multiplier': ecg_section.getfloat('height_multiplier')})

        eog_section = config['EOG']
        all_qc_params['EOG'] = dict({
            'n_breaks_bursts_allowed_per_10min': eog_section.getint('n_breaks_bursts_allowed_per_10min'),
            'allowed_range_of_peaks_stds': eog_section.getfloat('allowed_range_of_peaks_stds'),
            'norm_lvl': eog_section.getfloat('norm_lvl'),
            'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
            'thresh_lvl_peakfinder': eog_section.getfloat('thresh_lvl_peakfinder'), })

        head_section = config['Head_movement']
        all_qc_params['Head'] = dict({})

        muscle_section = config['Muscle']
        list_thresholds = muscle_section['threshold_muscle']
        # separate values in list_thresholds based on coma, remove spaces and convert them to floats:
        list_thresholds = [float(i) for i in list_thresholds.split(',')]
        muscle_freqs = [float(i) for i in muscle_section['muscle_freqs'].split(',')]

        all_qc_params['Muscle'] = dict({
            'threshold_muscle': list_thresholds,
            'min_distance_between_different_muscle_events': muscle_section.getfloat(
                'min_distance_between_different_muscle_events'),
            'muscle_freqs': muscle_freqs,
            'min_length_good': muscle_section.getfloat('min_length_good')})

        gqi_section = config['GlobalQualityIndex']

        compute_gqi = gqi_section.getboolean('compute_gqi', fallback=True)
        include_corr = gqi_section.getboolean('include_ecg_eog', fallback=True)

        weights = {
            'ch': gqi_section.getfloat('bad_ch_weight'),
            'corr': gqi_section.getfloat('correlation_weight'),
            'mus': gqi_section.getfloat('muscle_weight'),
            'psd': gqi_section.getfloat('psd_noise_weight'),
        }
        total_w = sum(weights.values())
        if total_w == 0:
            total_w = 1
        weights = {k: v / total_w for k, v in weights.items()}
        all_qc_params['GlobalQualityIndex'] = {
            'compute_gqi': compute_gqi,
            'include_ecg_eog': include_corr,
            'ch':   {
                'start': gqi_section.getfloat('bad_ch_start'),
                'end': gqi_section.getfloat('bad_ch_end'),
                'weight': weights['ch']
            },
            'corr': {
                'start': gqi_section.getfloat('correlation_start'),
                'end': gqi_section.getfloat('correlation_end'),
                'weight': weights['corr']
            },
            'mus':  {
                'start': gqi_section.getfloat('muscle_start'),
                'end': gqi_section.getfloat('muscle_end'),
                'weight': weights['mus']
            },
            'psd':  {
                'start': gqi_section.getfloat('psd_noise_start'),
                'end': gqi_section.getfloat('psd_noise_end'),
                'weight': weights['psd']
            },
        }

    except:
        print('___MEGqc___: ',
              'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return all_qc_params


def get_internal_config_params(config_file_name: str):
    """
    Parse all the parameters from config and put into a python dictionary
    divided by sections. Parsing approach can be changed here, which
    will not affect working of other fucntions.
    These are interanl parameters, NOT to be changed by the user.


    Parameters
    ----------
    config_file_name: str
        The name of the config file.

    Returns
    -------
    internal_qc_params: dict
        A dictionary with all the parameters.

    """

    internal_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_name)

    ecg_section = config['ECG']
    internal_qc_params['ECG'] = dict({
        'max_n_peaks_allowed_for_ch': ecg_section.getint('max_n_peaks_allowed_for_ch'),
        'max_n_peaks_allowed_for_avg': ecg_section.getint('max_n_peaks_allowed_for_avg'),
        'ecg_epoch_tmin': ecg_section.getfloat('ecg_epoch_tmin'),
        'ecg_epoch_tmax': ecg_section.getfloat('ecg_epoch_tmax'),
        'before_t0': ecg_section.getfloat('before_t0'),
        'after_t0': ecg_section.getfloat('after_t0'),
        'window_size_for_mean_threshold_method': ecg_section.getfloat('window_size_for_mean_threshold_method')})

    eog_section = config['EOG']
    internal_qc_params['EOG'] = dict({
        'max_n_peaks_allowed_for_ch': eog_section.getint('max_n_peaks_allowed_for_ch'),
        'max_n_peaks_allowed_for_avg': eog_section.getint('max_n_peaks_allowed_for_avg'),
        'eog_epoch_tmin': eog_section.getfloat('eog_epoch_tmin'),
        'eog_epoch_tmax': eog_section.getfloat('eog_epoch_tmax'),
        'before_t0': eog_section.getfloat('before_t0'),
        'after_t0': eog_section.getfloat('after_t0'),
        'window_size_for_mean_threshold_method': eog_section.getfloat('window_size_for_mean_threshold_method')})

    psd_section = config['PSD']
    internal_qc_params['PSD'] = dict({
        'method': psd_section.get('method'),
        'prominence_lvl_pos_avg': psd_section.getint('prominence_lvl_pos_avg'),
        'prominence_lvl_pos_channels': psd_section.getint('prominence_lvl_pos_channels')})

    return internal_qc_params


def stim_data_to_df(raw: mne.io.Raw):
    """
    Extract stimulus data from MEG data and put it into a pandas DataFrame.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data.

    Returns
    -------
    stim_deriv : list
        List with QC_derivative object with stimulus data.

    """

    stim_channels = mne.pick_types(raw.info, stim=True)

    if len(stim_channels) == 0:
        print('___MEGqc___: ', 'No stimulus channels found.')
        stim_df = pd.DataFrame()
    else:
        stim_channel_names = [raw.info['ch_names'][ch] for ch in stim_channels]
        # Extract data for stimulus channels
        stim_data, times = raw[stim_channels, :]
        # Create a DataFrame with the stimulus data
        stim_df = pd.DataFrame(stim_data.T, columns=stim_channel_names)
        stim_df['time'] = times

    # save df as QC_derivative object
    stim_deriv = [QC_derivative(stim_df, 'stimulus', 'df')]

    return stim_deriv


def Epoch_meg(epoching_params, data: mne.io.Raw):
    """
    Epoch MEG data based on the parameters provided in the config file.

    Parameters
    ----------
    epoching_params : dict
        Dictionary with parameters for epoching.
    data : mne.io.Raw
        MEG data to be epoched.

    Returns
    -------
    dict_epochs_mg : dict
        Dictionary with epochs for each channel type: mag, grad.

    """

    event_dur = epoching_params['event_dur']
    epoch_tmin = epoching_params['epoch_tmin']
    epoch_tmax = epoching_params['epoch_tmax']
    stim_channel = epoching_params['stim_channel']

    if stim_channel is None:
        picks_stim = mne.pick_types(data.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(data.info['chs'][ch]['ch_name'])
    print('___MEGqc___: ', 'Stimulus channels detected:', stim_channel)

    picks_magn = data.copy().pick('mag').ch_names if 'mag' in data else None
    picks_grad = data.copy().pick('grad').ch_names if 'grad' in data else None

    if not stim_channel:
        print('___MEGqc___: ',
              'No stimulus channel detected. Setting stimulus channel to None to allow mne to detect events autamtically.')
        stim_channel = None
        # here for info on how None is handled by mne: https://mne.tools/stable/generated/mne.find_events.html
        # even if stim is None, mne will check once more when creating events.

    epochs_grad, epochs_mag = None, None

    try:
        events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)

        if len(events) < 1:
            print('___MEGqc___: ',
                  'No events with set minimum duration were found using all stimulus channels. No epoching can be done. Try different event duration in config file.')
        else:
            print('___MEGqc___: ', 'Events found:', len(events))
            epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True,
                                    baseline=None, event_repeated=epoching_params['event_repeated'])
            epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True,
                                     baseline=None, event_repeated=epoching_params['event_repeated'])

    except:  # case when we use stim_channel=None, mne checks once more,  finds no other stim ch and no events and throws error:
        print('___MEGqc___: ', 'No stim channels detected, no events found.')
        pass  # go to returning empty dict

    dict_epochs_mg = {
        'mag': epochs_mag,
        'grad': epochs_grad}

    return dict_epochs_mg


def check_chosen_ch_types(m_or_g_chosen: List, channels_objs: dict):
    """
    Check if the channels which the user gave in config file to analize actually present in the data set.

    Parameters
    ----------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad. These are theones the user chose.
    channels_objs : dict
        Dictionary with channel names for each channel type: mag, grad. These are the ones present in the data set.

    Returns
    -------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.

    """

    skipped_str = ''

    if not any(ch in m_or_g_chosen for ch in ['mag', 'grad']):
        skipped_str = "No channels to analyze. Check parameter ch_types in config file."
        raise ValueError(skipped_str)

    skipped_msgs = {
        'mag': "There are no magnetometers in this data set: check parameter ch_types in config file. Analysis will be done only for gradiometers.",
        'grad': "There are no gradiometers in this data set: check parameter ch_types in config file. Analysis will be done only for magnetometers."
    }

    for ch in ['mag', 'grad']:
        if len(channels_objs[ch]) == 0 and ch in m_or_g_chosen:
            skipped_str = skipped_msgs[ch]
            print(f'___MEGqc___: {skipped_str}')
            m_or_g_chosen.remove(ch)

    if not any(channels_objs[ch] for ch in ['mag', 'grad']):
        skipped_str = "There are no magnetometers nor gradiometers in this data set. Analysis will not be done."
        raise ValueError(skipped_str)

    # Now m_or_g_chosen contain only those channel types which are present in the data set and were chosen by the user.

    return m_or_g_chosen, skipped_str


def choose_channels(raw: mne.io.Raw):
    """
    Separate channels by 'mag' and 'grad'.
    Done this way, because pick() or pick_types() sometimes gets wrong results, especialy for CTF data.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data

    Returns
    -------
    channels : dict
        dict with ch names separated by mag and grad

    """

    channels = {'mag': [], 'grad': []}

    # Loop over all channel indexes
    for ch_idx, ch_name in enumerate(raw.info['ch_names']):
        ch_type = mne.channel_type(raw.info, ch_idx)
        if ch_type in channels:
            channels[ch_type].append(ch_name)

    return channels


def change_ch_type_CTF(raw, channels):
    """
    For CTF data channels types and units need to be chnaged from mag to grad.

    Parameters
    ----------
    channels : dict
        dict with ch names separated by mag and grad

    Returns
    -------
    channels : dict
        dict with ch names separated by mag and grad UPDATED

    """

    # Create a copy of the channels['mag'] list to iterate over
    mag_channels_copy = channels['mag'][:]

    for ch_name in mag_channels_copy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_channel_types({ch_name: 'grad'})
        channels['grad'].append(ch_name)
        # Remove from mag list
        channels['mag'].remove(ch_name)

    print('___MEGqc___: Types of channels changed from mag to grad for CTF data.')

    return channels, raw


def load_data(file_path):
    """
    Load MEG data from a file. It can be a CTF data or a FIF file.

    Parameters
    ----------
    file_path : str
        Path to the fif file with MEG data.

    Returns
    -------
    raw : mne.io.Raw
        MEG data.
    shielding_str : str
        String with information about active shielding.

    """

    shielding_str = ''

    meg_system = None

    def _resolve_split_fif_path(path: str) -> str:
        """Return the first available FIF split created by MNE.

        Large FIF files may be written in numbered chunks ("-1.fif", "-2.fif",
        etc.) or with BIDS-style split labels ("split-01"). When the requested
        path points to the unsuffixed name (or to a missing chunk), try to find
        the lowest-index split part so reading succeeds without manual cleanup.
        """

        if os.path.isfile(path):
            return path

        base_dir = os.path.dirname(path) or '.'
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        if ext.lower() != '.fif':
            return path

        candidates = glob.glob(os.path.join(base_dir, f"{name}*{ext}"))

        split_candidates = []
        for candidate in candidates:
            cand_base = os.path.basename(candidate)
            split_match = re.search(r'split-(\d+)', cand_base)
            numbered_match = re.search(r'-(\d+)\.fif$', cand_base)

            if split_match:
                split_candidates.append((int(split_match.group(1)), candidate))
            elif numbered_match:
                split_candidates.append((int(numbered_match.group(1)), candidate))

        if split_candidates:
            split_candidates.sort(key=lambda x: x[0])
            return split_candidates[0][1]

        return path

    if os.path.isdir(file_path) and file_path.endswith('.ds'):
        # It's a CTF data directory
        print("___MEGqc___: ", "Loading CTF data...")
        raw = mne.io.read_raw_ctf(file_path, preload=True, verbose='ERROR')
        meg_system = 'CTF'
        print(f"___MEGqc___: Recording duration: {raw.times[-1] / 60:.2f} min")

    elif file_path.endswith('.fif'):
        # It's a FIF file
        meg_system = 'Triux'

        print("___MEGqc___: ", "Loading FIF data...")
        try:
            resolved_path = _resolve_split_fif_path(file_path)
            if resolved_path != file_path:
                print(f"___MEGqc___: Using split FIF part: {resolved_path}")

            raw = mne.io.read_raw_fif(resolved_path, on_split_missing='ignore', verbose='ERROR')
            splits_detected = len(raw._raw_extras)
            recording_duration_min = raw.times[-1] / 60
            if splits_detected > 1:
                print(f"___MEGqc___: Split FIF detected with {splits_detected} parts; MNE has merged the splits.")
                print(f"___MEGqc___: Recording duration: {recording_duration_min:.2f} min")
            else:
                print(f"___MEGqc___: Recording duration: {recording_duration_min:.2f} min")
        except:
            resolved_path = _resolve_split_fif_path(file_path)
            if resolved_path != file_path:
                print(f"___MEGqc___: Using split FIF part: {resolved_path}")

            raw = mne.io.read_raw_fif(resolved_path, allow_maxshield=True, on_split_missing='ignore', verbose='ERROR')
            splits_detected = len(raw._raw_extras)
            recording_duration_min = raw.times[-1] / 60
            if splits_detected > 1:
                print(f"___MEGqc___: Split FIF detected with {splits_detected} parts; MNE has merged the splits.")
                print(f"___MEGqc___: Recording duration: {recording_duration_min:.2f} min")
            else:
                print(f"___MEGqc___: Recording duration: {recording_duration_min:.2f} min")
            shielding_str = ''' <p>This fif file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case environmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p>'''

    else:
        raise ValueError(
            "Unsupported file format or file does not exist. The pipeline works with CTF data directories and FIF files.")

    return raw, shielding_str, meg_system


def add_3d_ch_locations(raw, channels_objs):
    """
    Add channel locations to the MEG channels objects.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data.
    channels_objs : dict
        Dictionary with MEG channels.

    Returns
    -------
    channels_objs : dict
        Dictionary with MEG channels with added locations.

    """

    # Create a dictionary to store the channel locations
    ch_locs = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
    # why [:3]?  to Get only the x, y, z coordinates (first 3 values), theer are also rotations, etc storred in loc.

    # Iterate through the channel names and add the locations
    for key, value in channels_objs.items():
        for ch in value:
            ch.loc = ch_locs[ch.name]

    return channels_objs


def add_CTF_lobes(channels_objs):
    # Initialize dictionary to store channels by lobe and side
    lobes_ctf = {
        'Left Frontal': [],
        'Right Frontal': [],
        'Left Temporal': [],
        'Right Temporal': [],
        'Left Parietal': [],
        'Right Parietal': [],
        'Left Occipital': [],
        'Right Occipital': [],
        'Central': [],
        'Reference': [],
        'EEG/EOG/ECG': [],
        'Extra': []  # Add 'Extra' lobe
    }

    # Iterate through the channel names and categorize them
    for key, value in channels_objs.items():
        for ch in value:
            categorized = False  # Track if the channel is categorized
            # Magnetometers (assuming they start with 'M')
            # Even though they all have to be grads for CTF!!!
            if ch.name.startswith('MLF'):  # Left Frontal
                lobes_ctf['Left Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRF'):  # Right Frontal
                lobes_ctf['Right Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLT'):  # Left Temporal
                lobes_ctf['Left Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRT'):  # Right Temporal
                lobes_ctf['Right Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLP'):  # Left Parietal
                lobes_ctf['Left Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRP'):  # Right Parietal
                lobes_ctf['Right Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLO'):  # Left Occipital
                lobes_ctf['Left Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRO'):  # Right Occipital
                lobes_ctf['Right Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLC') or ch.name.startswith('MRC'):  # Central (Midline)
                lobes_ctf['Central'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MZ'):  # Reference Sensors
                lobes_ctf['Reference'].append(ch.name)
                categorized = True
            elif ch.name in ['Cz', 'Pz', 'ECG', 'VEOG', 'HEOG']:  # EEG/EOG/ECG channels
                lobes_ctf['EEG/EOG/ECG'].append(ch.name)
                categorized = True

            # Gradiometers (assuming they have a different prefix or suffix, such as 'G')
            elif ch.name.startswith('GLF'):  # Left Frontal Gradiometers
                lobes_ctf['Left Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRF'):  # Right Frontal Gradiometers
                lobes_ctf['Right Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLT'):  # Left Temporal Gradiometers
                lobes_ctf['Left Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRT'):  # Right Temporal Gradiometers
                lobes_ctf['Right Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLP'):  # Left Parietal Gradiometers
                lobes_ctf['Left Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRP'):  # Right Parietal Gradiometers
                lobes_ctf['Right Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLO'):  # Left Occipital Gradiometers
                lobes_ctf['Left Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRO'):  # Right Occipital Gradiometers
                lobes_ctf['Right Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLC') or ch.name.startswith('GRC'):  # Central (Midline) Gradiometers
                lobes_ctf['Central'].append(ch.name)
                categorized = True

            # If the channel was not categorized, add it to 'Extra'
            if not categorized:
                lobes_ctf['Extra'].append(ch.name)

    lobe_colors = {
        'Left Frontal': '#1f77b4',
        'Right Frontal': '#ff7f0e',
        'Left Temporal': '#2ca02c',
        'Right Temporal': '#9467bd',
        'Left Parietal': '#e377c2',
        'Right Parietal': '#d62728',
        'Left Occipital': '#bcbd22',
        'Right Occipital': '#17becf',
        'Central': '#8c564b',
        'Reference': '#7f7f7f',
        'EEG/EOG/ECG': '#bcbd22',
        'Extra': '#d3d3d3'
    }

    lobes_color_coding_str = 'Color coding by lobe is applied as per CTF system.'
    for key, value in channels_objs.items():
        for ch in value:
            for lobe in lobes_ctf.keys():
                if ch.name in lobes_ctf[lobe]:
                    ch.lobe = lobe
                    ch.lobe_color = lobe_colors[lobe]

    return channels_objs, lobes_color_coding_str


def add_Triux_lobes(channels_objs):
    lobes_treux = {
        'Left Frontal': ['MEG0621', 'MEG0622', 'MEG0623', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0121', 'MEG0122',
                         'MEG0123', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331',
                         'MEG0332', 'MEG0333', 'MEG0643', 'MEG0642', 'MEG0641', 'MEG0611', 'MEG0612', 'MEG0613',
                         'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0511', 'MEG0512',
                         'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533'],
        'Right Frontal': ['MEG0811', 'MEG0812', 'MEG0813', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922',
                          'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011',
                          'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033',
                          'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232',
                          'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1411', 'MEG1412', 'MEG1413'],
        'Left Temporal': ['MEG0111', 'MEG0112', 'MEG0113', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142',
                          'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231',
                          'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1511', 'MEG1512', 'MEG1513',
                          'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542',
                          'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623'],
        'Right Temporal': ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1421', 'MEG1422',
                           'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1341',
                           'MEG1342', 'MEG1343', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613',
                           'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642',
                           'MEG2643', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423'],
        'Left Parietal': ['MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432',
                          'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0741',
                          'MEG0742', 'MEG0743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823',
                          'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG0631', 'MEG0632',
                          'MEG0633', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG2011', 'MEG2012', 'MEG2013'],
        'Right Parietal': ['MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122',
                           'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG2233', 'MEG1141', 'MEG1142', 'MEG1143',
                           'MEG2243', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG2211',
                           'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233',
                           'MEG2241', 'MEG2242', 'MEG2243', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2441', 'MEG2442',
                           'MEG2443'],
        'Left Occipital': ['MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722',
                           'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1911',
                           'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933',
                           'MEG1941', 'MEG1942', 'MEG1943', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112',
                           'MEG2113', 'MEG2141', 'MEG2142', 'MEG2143'],
        'Right Occipital': ['MEG2031', 'MEG2032', 'MEG2033', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2311', 'MEG2312',
                            'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341',
                            'MEG2342', 'MEG2343', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523',
                            'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2431', 'MEG2432',
                            'MEG2433', 'MEG2131', 'MEG2132', 'MEG2133'],
        'Extra': []}  # Add 'Extra' lobe

    # These were just for Aarons presentation:
    # lobes_treux = {
    #         'Left Frontal': ['MEG0621', 'MEG0622', 'MEG0623', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331',  'MEG0332', 'MEG0333', 'MEG0643', 'MEG0642', 'MEG0641', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533'],
    #         'Right Frontal': ['MEG0811', 'MEG0812', 'MEG0813', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1411', 'MEG1412', 'MEG1413'],
    #         'Left Temporal': ['MEG0111', 'MEG0112', 'MEG0113', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623'],
    #         'Right Temporal': ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423'],
    #         'Left Parietal': ['MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0741', 'MEG0742', 'MEG0743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG2011', 'MEG2012', 'MEG2013'],
    #         'Right Parietal': ['MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG2233', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG2243', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2441', 'MEG2442', 'MEG2443'],
    #         'Left Occipital': ['MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2141', 'MEG2142', 'MEG2143', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2131', 'MEG2132', 'MEG2133'],
    #         'Right Occipital': ['MEG0611', 'MEG0612', 'MEG0613']}

    # #Now add to lobes_treux also the name of each channel with space in the middle:
    for lobe in lobes_treux.keys():
        lobes_treux[lobe] += [channel[:-4] + ' ' + channel[-4:] for channel in lobes_treux[lobe]]

    lobe_colors = {
        'Left Frontal': '#1f77b4',
        'Right Frontal': '#ff7f0e',
        'Left Temporal': '#2ca02c',
        'Right Temporal': '#9467bd',
        'Left Parietal': '#e377c2',
        'Right Parietal': '#d62728',
        'Left Occipital': '#bcbd22',
        'Right Occipital': '#17becf',
        'Extra': '#d3d3d3'}

    # These were just for Aarons presentation:
    # lobe_colors = {
    #     'Left Frontal': '#2ca02c',
    #     'Right Frontal': '#2ca02c',
    #     'Left Temporal': '#2ca02c',
    #     'Right Temporal': '#2ca02c',
    #     'Left Parietal': '#2ca02c',
    #     'Right Parietal': '#2ca02c',
    #     'Left Occipital': '#2ca02c',
    #     'Right Occipital': '#d62728'}

    # loop over all values in the dictionary:
    lobes_color_coding_str = 'Color coding by lobe is applied as per Treux system. Separation by lobes based on Y. Hu et al. "Partial Least Square Aided Beamforming Algorithm in Magnetoencephalography Source Imaging", 2018. '
    for key, value in channels_objs.items():
        for ch in value:
            categorized = False
            for lobe in lobes_treux.keys():
                if ch.name in lobes_treux[lobe]:
                    ch.lobe = lobe
                    ch.lobe_color = lobe_colors[lobe]
                    categorized = True
                    break
            # If the channel was not categorized, assign it to 'extra' lobe
            if not categorized:
                ch.lobe = 'Extra'
                ch.lobe_color = lobe_colors[lobe]

    return channels_objs, lobes_color_coding_str


def assign_channels_properties(channels_short: dict, meg_system: str):
    """
    Assign lobe area to each channel according to the lobe area dictionary + the color for plotting + channel location.

    Can later try to make this function a method of the MEG_channels class.
    At the moment not possible because it needs to know the total number of channels to figure which meg system to use for locations. And MEG_channels class is created for each channel separately.

    Parameters
    ----------
    channels : dict
        dict with channels names like: {'mag': [...], 'grad': [...]}
    meg_system: str
        CTF, Triux, None...

    Returns
    -------
    channels_objs : dict
        Dictionary with channel names for each channel type: mag, grad. Each channel has assigned lobe area and color for plotting + channel location.
    lobes_color_coding_str : str
        A string with information about the color coding of the lobes.

    """

    channels_full = copy.deepcopy(channels_short)

    # for understanding how the locations are obtained. They can be extracted as:
    # mag_locs = raw.copy().pick('mag').info['chs']
    # mag_pos = [ch['loc'][:3] for ch in mag_locs]
    # (XYZ locations are first 3 digit in the ch['loc']  where ch is 1 sensor in raw.info['chs'])

    # Assign lobe labels to the channels:

    if meg_system.upper() == 'TRIUX' and len(channels_full['mag']) == 102 and len(channels_full['grad']) == 204:
        # for 306 channel data in Elekta/Neuromag Treux system
        channels_full, lobes_color_coding_str = add_Triux_lobes(channels_full)

        # assign 'TRIUX' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'TRIUX'

    elif meg_system.upper() == 'CTF':
        channels_full, lobes_color_coding_str = add_CTF_lobes(channels_full)

        # assign 'CTF' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'CTF'

    else:
        lobes_color_coding_str = 'For MEG systems other than MEGIN Triux or CTF color coding by lobe is not applied.'
        lobe_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#d62728', '#bcbd22', '#17becf']
        print('___MEGqc___: ' + lobes_color_coding_str)

        for key, value in channels_full.items():
            for ch in value:
                ch.lobe = 'All channels'
                # take random color from lobe_colors:
                ch.lobe_color = random.choice(lobe_colors)
                ch.system = 'OTHER'

    # sort channels by name:
    for key, value in channels_full.items():
        channels_full[key] = sorted(value, key=lambda x: x.name)

    return channels_full, lobes_color_coding_str


def sort_channels_by_lobe(channels_objs: dict):
    """ Sorts channels by lobes.

    Parameters
    ----------
    channels_objs : dict
        A dictionary of channel objects.

    Returns
    -------
    chs_by_lobe : dict
        A dictionary of channels sorted by ch type and lobe.

    """
    chs_by_lobe = {}
    for m_or_g in channels_objs:

        # put all channels into separate lists based on their lobes:
        lobes_names = list(set([ch.lobe for ch in channels_objs[m_or_g]]))

        lobes_dict = {key: [] for key in lobes_names}
        # fill the dict with channels:
        for ch in channels_objs[m_or_g]:
            lobes_dict[ch.lobe].append(ch)

            # Sort the dictionary by lobes names (by the second word in the key, if it exists)
        chs_by_lobe[m_or_g] = dict(
            sorted(lobes_dict.items(), key=lambda x: x[0].split()[1] if len(x[0].split()) > 1 else ''))

    return chs_by_lobe




def save_meg_with_suffix(
    file_path: str,
    derivatives_root: str,
    raw,
    final_suffix: str = "FILTERED",
) -> str:
    """
    Save an MNE raw object alongside the derivatives with a custom suffix.

    The output directory is constructed as ``<derivatives_root>/temp/<subject>``
    where ``subject`` is inferred from the first path component starting with
    ``sub-`` in ``file_path``. Using ``derivatives_root`` allows callers to place
    temporary files outside the read-only BIDS directory if needed.
    """

    norm_path = os.path.normpath(file_path)
    components = norm_path.split(os.sep)

    subject = next((part for part in components if part.startswith('sub-')), None)
    if subject is None:
        raise ValueError("Unable to determine subject from file path for temporary output")

    output_dir = os.path.join(derivatives_root, 'temp', subject)
    output_dir = os.path.abspath(output_dir)
    print("Output directory:", output_dir)

    # Create the target folder if it does not exist already
    os.makedirs(output_dir, exist_ok=True)
    print("Directory created (or already exists):", output_dir)

    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() == '.ds':
        ext = '.fif'

    # Drop BIDS split tags so derivatives use the base recording name. When
    # MNE saves large files it may split them internally, but the resulting
    # derivatives should reference the unified recording rather than the
    # individual split chunk that happened to be loaded first.
    name = re.sub(r"_split-\d+", "", name)

    new_filename = f"{name}_{final_suffix}{ext}"
    new_file_path = os.path.join(output_dir, new_filename)
    print("New file path:", new_file_path)

    def _resolve_saved_split_path(path: str) -> str:
        """Return the first split chunk saved by MNE when splitting occurs."""

        if os.path.isfile(path):
            return path

        base_dir = os.path.dirname(path) or '.'
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        if ext.lower() != '.fif':
            return path

        candidates = glob.glob(os.path.join(base_dir, f"{name}*{ext}"))

        split_candidates = []
        for candidate in candidates:
            cand_base = os.path.basename(candidate)
            split_match = re.search(r'split-(\d+)', cand_base)
            numbered_match = re.search(r'-(\d+)\.fif$', cand_base)

            if split_match:
                split_candidates.append((int(split_match.group(1)), candidate))
            elif numbered_match:
                split_candidates.append((int(numbered_match.group(1)), candidate))

        if split_candidates:
            split_candidates.sort(key=lambda x: x[0])
            return split_candidates[0][1]

        return path

    raw.save(new_file_path, overwrite=True, verbose='ERROR')

    resolved_save_path = _resolve_saved_split_path(new_file_path)
    if resolved_save_path != new_file_path:
        print(f"___MEGqc___: Split FIF saved, first part: {resolved_save_path}")

    return resolved_save_path


def remove_fif_and_splits(path: str) -> None:
    """Remove a FIF file and any split parts created by MNE.

    MNE may split large FIF saves into multiple pieces using either
    ``split-XX`` or ``-1.fif`` style numbering. This function removes the
    specified file path as well as any adjacent split parts that share the
    same base name.
    """

    base_dir = os.path.dirname(path) or "."
    root, ext = os.path.splitext(os.path.basename(path))

    if ext.lower() != ".fif":
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return

    # Normalize the root to the original requested name (without split tags)
    normalized_root = re.sub(r"(?:_split-\d+|-\d+)$", "", root)
    patterns = [f"{normalized_root}*.fif"]

    for pattern in patterns:
        for candidate in glob.glob(os.path.join(base_dir, pattern)):
            try:
                os.remove(candidate)
            except FileNotFoundError:
                continue


def delete_temp_folder(derivatives_root: str) -> str:
    """
    Remove the temporary working directory used during preprocessing.

    Parameters
    ----------
    derivatives_root : str
         Absolute path to the dataset's derivatives directory (either inside
         the BIDS dataset or in an external location).
    """
    temp_dir = os.path.join(derivatives_root, 'temp')
    temp_dir = os.path.abspath(temp_dir)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        print("Removing directory:", temp_dir)

    return


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params: dict, file_path: str,
                       derivatives_root: str):
    """
    Here all the initial actions needed to analyse MEG data are done:

    - read fif file,
    - separate mags and grads names into 2 lists,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Parameters
    ----------
    default_settings : dict
        Dictionary with default settings for MEG QC.
    filtering_settings : dict
        Dictionary with parameters for filtering.
    epoching_params : dict
        Dictionary with parameters for epoching.
    file_path : str
        Path to the fif file with MEG data.

    Returns
    -------
    dict_epochs_mg : dict
        Dictionary with epochs for each channel type: mag, grad.
    chs_by_lobe : dict
        Dictionary with channel objects for each channel type: mag, grad. And by lobe. Each obj hold info about the channel name,
        lobe area and color code, locations and (in the future) pther info, like: if it has noise of any sort.
    channels : dict
        Dictionary with channel names for each channel type: mag, grad.
    raw_crop_filtered : mne.io.Raw
        Filtered and cropped MEG data.
    raw_crop_filtered_resampled : mne.io.Raw
        Filtered, cropped and resampled MEG data.
    raw_cropped : mne.io.Raw
        Cropped MEG data.
    raw : mne.io.Raw
        MEG data.
    info_derivs : list
        List with QC_derivative objects with MNE info object.
    shielding_str : str
        String with information about active shielding.
    epoching_str : str
        String with information about epoching.
    sensors_derivs : list
        List with data frames with sensors info.
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.
    lobes_color_coding_str : str
        String with information about color coding for lobes.
    resample_str : str
        String with information about resampling.

    """

    print('___MEGqc___: ', 'Reading data from file:', file_path)

    raw, shielding_str, meg_system = load_data(file_path)

    # Working with channels:
    channels = choose_channels(raw)

    if meg_system == 'CTF':  # ONLY FOR CTF we do this! Return raw with changed channel types.
        channels, raw = change_ch_type_CTF(raw, channels)

    # Turn channel names into objects:
    channels_objs = {key: [MEG_channel(name=ch_name, type=key) for ch_name in value] for key, value in channels.items()}

    # Assign channels properties:
    channels_objs, lobes_color_coding_str = assign_channels_properties(channels_objs, meg_system)

    # Add channel locations:
    channels_objs = add_3d_ch_locations(raw, channels_objs)

    # Check if there are channels to analyze according to info in config file:
    m_or_g_chosen, m_or_g_skipped_str = check_chosen_ch_types(m_or_g_chosen=default_settings['m_or_g_chosen'],
                                                              channels_objs=channels_objs)

    # Sort channels by lobe - this will be used often for plotting
    chs_by_lobe = sort_channels_by_lobe(channels_objs)
    print('___MEGqc___: ', 'Channels sorted by lobe.')

    info = raw.info
    info_derivs = [QC_derivative(content=info, name='RawInfo', content_type='info', fig_order=-1)]

    # crop the data to calculate faster:
    tmax_possible = raw.times[-1]
    tmax = default_settings['crop_tmax']
    if tmax is None or tmax > tmax_possible:
        tmax = tmax_possible
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)
    # When resampling for plotting, cropping or anything else you don't need permanent in raw inside any functions - always do raw_new=raw.copy() not just raw_new=raw. The last command doesn't create a new object, the whole raw will be changed and this will also be passed to other functions even if you don't return the raw.

    stim_deriv = stim_data_to_df(raw_cropped)

    # Data filtering:
    raw_cropped_filtered = raw_cropped.copy()
    if filtering_settings['apply_filtering'] is True:
        raw_cropped.load_data()  # Data has to be loaded into mememory before filetering:
        # Save raw_cropped
        raw_cropped_path = save_meg_with_suffix(file_path, derivatives_root, raw_cropped, final_suffix="CROPPED")

        raw_cropped_filtered = raw_cropped

        # if filtering_settings['h_freq'] is higher than the Nyquist frequency, set it to Nyquist frequency:
        if filtering_settings['h_freq'] > raw_cropped_filtered.info['sfreq'] / 2 - 1:
            filtering_settings['h_freq'] = raw_cropped_filtered.info['sfreq'] / 2 - 1
            print('___MEGqc___: ',
                  'High frequency for filtering is higher than Nyquist frequency. High frequency was set to Nyquist frequency:',
                  filtering_settings['h_freq'])

        raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'],
                                    picks='meg', method=filtering_settings['method'], iir_params=None)
        print('___MEGqc___: ', 'Data filtered from', filtering_settings['l_freq'], 'to', filtering_settings['h_freq'],
              'Hz.')

        # Save filtered signal
        raw_cropped_filtered_path = save_meg_with_suffix(file_path, derivatives_root, raw_cropped_filtered,
                                                         final_suffix="FILTERED")

        if filtering_settings['downsample_to_hz'] is False:
            raw_cropped_filtered_resampled = raw_cropped_filtered
            raw_cropped_filtered_resampled_path = raw_cropped_filtered_path
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)
        elif filtering_settings['downsample_to_hz'] >= filtering_settings['h_freq'] * 5:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['downsample_to_hz'])
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, derivatives_root,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
            print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['h_freq'] * 5)
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, derivatives_root,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            # frequency to resample is 5 times higher than the maximum chosen frequency of the function
            resample_str = 'Chosen "downsample_to_hz" value set was too low, it must be at least 5 time higher than the highest filer frequency. Data resampled to ' + str(
                filtering_settings['h_freq'] * 5) + ' Hz. '
            print('___MEGqc___: ', resample_str)


    else:
        print('___MEGqc___: ', 'Data not filtered.')
        # And downsample:
        if filtering_settings['downsample_to_hz'] is not False:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['downsample_to_hz'])
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, derivatives_root,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            if filtering_settings['downsample_to_hz'] < 500:
                resample_str = 'Data resampled to ' + str(filtering_settings[
                                                              'downsample_to_hz']) + ' Hz. Keep in mind: resampling to less than 500Hz is not recommended, since it might result in high frequency data loss (for example of the CHPI coils signal. '
                print('___MEGqc___: ', resample_str)
            else:
                resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
                print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, derivatives_root,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)

    del raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw
    gc.collect()

    # Load data
    raw_cropped_filtered, shielding_str, meg_system = load_data(raw_cropped_filtered_path)

    # Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching?
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered)
    epoching_str = ''
    if dict_epochs_mg['mag'] is None and dict_epochs_mg['grad'] is None:
        epoching_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''

    resample_str = '<p>' + resample_str + '</p>'

    # Extract chs_by_lobe into a data frame
    sensors_derivs = chs_dict_to_csv(chs_by_lobe, file_name_prefix='Sensors')

    raw_path = file_path

    return meg_system, dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered_path, raw_cropped_filtered_resampled_path, raw_cropped_path, raw_path, info_derivs, stim_deriv, shielding_str, epoching_str, sensors_derivs, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, resample_str


def chs_dict_to_csv(chs_by_lobe: dict, file_name_prefix: str):
    """
    Convert dictionary with channels objects to a data frame and save it as a csv file.

    Parameters
    ----------
    chs_by_lobe : dict
        Dictionary with channel objects for each channel type: mag, grad. And by lobe. Each obj hold info about the channel name,
        lobe area and color code, locations and (in the future) pther info, like: if it has noise of any sort.
    file_name_prefix : str
        Prefix for the file name. Example: 'Sensors' will result in file name 'Sensors.csv'.

    Returns
    -------
    df_deriv : list
        List with data frames with sensors info.

    """

    # Extract chs_by_lobe into a data frame
    chs_by_lobe_df = {k1: {k2: pd.concat([channel.to_df() for channel in v2]) for k2, v2 in v1.items()} for k1, v1 in
                      chs_by_lobe.items()}

    its = []
    for ch_type, content in chs_by_lobe_df.items():
        for lobe, items in content.items():
            its.append(items)

    df_fin = pd.concat(its)

    # if df already contains columns like 'STD epoch_' with numbers, 'STD epoch' needs to be removed from the data frame:
    if 'STD epoch' in df_fin and any(col.startswith('STD epoch_') and col[10:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='STD epoch')
    if 'PtP epoch' in df_fin and any(col.startswith('PtP epoch_') and col[10:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'PtP epoch' column
        df_fin = df_fin.drop(columns='PtP epoch')
    if 'PSD' in df_fin and any(col.startswith('PSD_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='PSD')
    if 'ECG' in df_fin and any(col.startswith('ECG_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='ECG')
    if 'EOG' in df_fin and any(col.startswith('EOG_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='EOG')

    df_deriv = [QC_derivative(content=df_fin, name=file_name_prefix, content_type='df')]

    return df_deriv
