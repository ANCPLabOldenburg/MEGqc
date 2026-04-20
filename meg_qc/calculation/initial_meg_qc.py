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
from meg_qc.calculation.objects import QC_derivative, MEG_channel, QC_channel

# Canonical set of channel types the package can process.
SUPPORTED_CH_TYPES = {'mag', 'grad', 'eeg'}


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

    # Backward/forward compatible handling:
    # older configs relied on DEFAULT, while newer public configs use [GENERAL].
    if "GENERAL" in config:
        default_section = config["GENERAL"]
    else:
        default_section = config["DEFAULT"]

    m_or_g_chosen = default_section['ch_types']
    m_or_g_chosen = [chosen.strip() for chosen in m_or_g_chosen.split(",")]
    if not any(ch in SUPPORTED_CH_TYPES for ch in m_or_g_chosen):
        print('___MEGqc___: ', f'No valid channel types to analyze. '
              f'Supported: {SUPPORTED_CH_TYPES}. Got: {m_or_g_chosen}')
        return None

    # TODO: save list of mags and grads here and use later everywhere? because for CTF types are messed up.

    run_STD = default_section.getboolean('STD')
    run_PSD = default_section.getboolean('PSD')
    run_PTP_manual = default_section.getboolean('PTP_manual')
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
            'run_ECG': run_ECG,
            'run_EOG': run_EOG,
            'run_Head': run_Head,
            'run_Muscle': run_Muscle,
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

        try:
            downsample_val = filtering_section.getint('downsample_to_hz')
        except (ValueError, TypeError):
            # Allows setting downsample_to_hz = False in the config to skip resampling
            raw_val = filtering_section.get('downsample_to_hz', '').strip().lower()
            downsample_val = False if raw_val in ('false', '0', 'no', 'none', '') else None
            if downsample_val is None:
                raise ValueError(
                    f"Invalid value for 'downsample_to_hz' in config: '{filtering_section.get('downsample_to_hz')}'. "
                    f"Expected an integer (e.g. 1000) or False to skip resampling."
                )

        all_qc_params['Filtering'] = dict({
            'apply_filtering': filtering_section.getboolean('apply_filtering'),
            'l_freq': lfreq,
            'h_freq': hfreq,
            'method': filtering_section['method'],
            'downsample_to_hz': downsample_val})

        epoching_section = config['Epoching']
        stim_channel = epoching_section['stim_channel']
        stim_channel = stim_channel.replace(" ", "")
        stim_channel = stim_channel.split(",")
        if stim_channel == ['']:
            stim_channel = None

        # preferred_stim_channels: combined trigger channels preferred over individual bit channels.
        # Split on comma, strip surrounding whitespace from each token, preserve internal spaces
        # (e.g. "STI 101" is a valid channel name with an internal space).
        preferred_raw = epoching_section.get('preferred_stim_channels', '')
        preferred_stim_channels = [c.strip() for c in preferred_raw.split(',') if c.strip()]

        # noisy_stim_channels: channels known to carry DC-offset noise rather than real events.
        noisy_raw = epoching_section.get('noisy_stim_channels', '')
        noisy_stim_channels = [c.strip() for c in noisy_raw.split(',') if c.strip()]

        epoching_params = dict({
            'event_dur': epoching_section.getfloat('event_dur'),
            'epoch_tmin': epoching_section.getfloat('epoch_tmin'),
            'epoch_tmax': epoching_section.getfloat('epoch_tmax'),
            'stim_channel': stim_channel,
            'preferred_stim_channels': preferred_stim_channels,
            'noisy_stim_channels': noisy_stim_channels,
            'event_repeated': epoching_section['event_repeated'],
            'use_fixed_length_epochs': epoching_section.getboolean('use_fixed_length_epochs'),
            'fixed_epoch_duration': epoching_section.getfloat('fixed_epoch_duration'),
            'fixed_epoch_overlap': epoching_section.getfloat('fixed_epoch_overlap')})
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


        ecg_section = config['ECG']
        ecg_fixed_ch = ecg_section.get('fixed_channel_names', '')
        ecg_fixed_ch = [name.strip() for name in ecg_fixed_ch.split(',') if name.strip()]
        all_qc_params['ECG'] = dict({
            'drop_bad_ch': ecg_section.getboolean('drop_bad_ch'),
            'n_breaks_bursts_allowed_per_10min': ecg_section.getint('n_breaks_bursts_allowed_per_10min'),
            'allowed_range_of_peaks_stds': ecg_section.getfloat('allowed_range_of_peaks_stds'),
            'norm_lvl': ecg_section.getfloat('norm_lvl'),
            'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
            'thresh_lvl_peakfinder': ecg_section.getfloat('thresh_lvl_peakfinder'),
            'height_multiplier': ecg_section.getfloat('height_multiplier'),
            'fixed_channel_names': ecg_fixed_ch})

        eog_section = config['EOG']
        eog_fixed_ch = eog_section.get('fixed_channel_names', '')
        eog_fixed_ch = [name.strip() for name in eog_fixed_ch.split(',') if name.strip()]
        all_qc_params['EOG'] = dict({
            'n_breaks_bursts_allowed_per_10min': eog_section.getint('n_breaks_bursts_allowed_per_10min'),
            'allowed_range_of_peaks_stds': eog_section.getfloat('allowed_range_of_peaks_stds'),
            'norm_lvl': eog_section.getfloat('norm_lvl'),
            'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
            'thresh_lvl_peakfinder': eog_section.getfloat('thresh_lvl_peakfinder'),
            'fixed_channel_names': eog_fixed_ch})

        head_section = config['Head_movement']
        all_qc_params['Head'] = dict({})

        muscle_section = config['Muscle']
        list_thresholds = muscle_section['threshold_muscle']
        # separate values in list_thresholds based on coma, remove spaces and convert them to floats:
        list_thresholds = [float(i) for i in list_thresholds.split(',')]
        muscle_freqs = [float(i) for i in muscle_section['muscle_freqs'].split(',')]
        muscle_freqs_eeg = [float(i) for i in muscle_section.get('muscle_freqs_eeg', '20, 100').split(',')]

        all_qc_params['Muscle'] = dict({
            'threshold_muscle': list_thresholds,
            'min_distance_between_different_muscle_events': muscle_section.getfloat(
                'min_distance_between_different_muscle_events'),
            'muscle_freqs': muscle_freqs,
            'muscle_freqs_eeg': muscle_freqs_eeg,
            'min_length_good': muscle_section.getfloat('min_length_good')})

        # ── EEG-specific settings (optional section) ───────────────────────
        eeg_settings = {'reference_method': 'average', 'montage': 'auto'}
        if 'EEG' in config:
            eeg_sec = config['EEG']
            eeg_settings['reference_method'] = eeg_sec.get('reference_method', 'average').strip()
            eeg_settings['montage'] = eeg_sec.get('montage', 'auto').strip()
        all_qc_params['EEG_settings'] = eeg_settings

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


def stim_data_to_df(raw: mne.io.Raw, epoching_params: dict = None):
    """
    Extract stimulus data from MEG data and put it into a pandas DataFrame.
    Also compute per-channel event counts using ``mne.find_events`` for later
    comparison with BIDS ``*_events.tsv``.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data.
    epoching_params : dict, optional
        Epoching parameters dict (used for ``event_dur``).  When *None*,
        a default minimum duration of 0.002 s is used.

    Returns
    -------
    stim_deriv : list
        List with QC_derivative object with stimulus time-series data.
    stim_channel_event_counts : dict
        ``{channel_name: {event_id: count, ...}, ...}`` for every stim channel.
    """

    event_dur = 0.002
    if epoching_params is not None:
        event_dur = epoching_params.get('event_dur', event_dur)

    stim_channels = mne.pick_types(raw.info, stim=True)
    stim_channel_event_counts = {}

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

        # Compute per-channel event counts (safe per channel)
        for ch_name in stim_channel_names:
            try:
                ev = mne.find_events(
                    raw, stim_channel=ch_name,
                    min_duration=event_dur, uint_cast=True, verbose=False,
                )
                ids, counts = np.unique(ev[:, 2], return_counts=True)
                stim_channel_event_counts[ch_name] = {
                    int(eid): int(cnt) for eid, cnt in zip(ids, counts)
                }
            except Exception:
                stim_channel_event_counts[ch_name] = {}

    # save df as QC_derivative object
    stim_deriv = [QC_derivative(stim_df, 'stimulus', 'df')]

    return stim_deriv, stim_channel_event_counts



def _read_bids_events_tsv(file_path: str, data: mne.io.Raw):
    """
    Look for a BIDS *_events.tsv file alongside the MEG file and convert it to
    an MNE events array (shape N×3: [abs_sample, 0, event_id]).

    Returns
    -------
    events : np.ndarray or None
    tsv_path : str or None
    id_to_trial_type : dict  {int event_id → str trial_type label} or {}
    """
    import re
    tsv_path = re.sub(
        r'(_(meg|eeg))?\.(fif|fif\.gz|raw|raw\.gz|set|edf|bdf\.gz|bdf|mff|nxe|ds|sqd|con|vhdr|cnt)$',
        '_events.tsv',
        file_path,
        flags=re.IGNORECASE,
    )
    if tsv_path == file_path or not os.path.exists(tsv_path):
        return None, None, {}

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'onset' not in df.columns:
            return None, None, {}

        sfreq = data.info['sfreq']
        first_samp = data.first_samp

        id_to_trial_type = {}

        # Event IDs: prefer numeric 'value', else encode 'trial_type' strings
        if 'value' in df.columns:
            event_vals = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int).values
            # Build id → trial_type from TSV if both columns present
            if 'trial_type' in df.columns:
                for val, tt in zip(event_vals, df['trial_type'].fillna('').values):
                    if val not in id_to_trial_type and tt:
                        id_to_trial_type[int(val)] = str(tt)
        elif 'trial_type' in df.columns:
            unique_types = {t: i + 1 for i, t in enumerate(df['trial_type'].dropna().unique())}
            event_vals = df['trial_type'].map(unique_types).fillna(0).astype(int).values
            id_to_trial_type = {v: k for k, v in unique_types.items()}
            print('___MEGqc___: BIDS events TSV trial_type → id mapping:', unique_types)
        else:
            event_vals = np.ones(len(df), dtype=int)

        onset_samples = (np.round(df['onset'].values * sfreq) + first_samp).astype(int)

        valid = (onset_samples >= first_samp) & (onset_samples <= data.last_samp)
        onset_samples = onset_samples[valid]
        event_vals = event_vals[valid]

        if len(onset_samples) == 0:
            return None, None, {}

        events = np.column_stack([
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),
            event_vals,
        ]).astype(int)

        print(f'___MEGqc___: BIDS events TSV: {tsv_path} → {len(events)} events, '
              f'IDs={np.unique(events[:, 2]).tolist()}')
        return events, tsv_path, id_to_trial_type

    except Exception as exc:
        print(f'___MEGqc___: Could not read BIDS events TSV ({tsv_path}): {exc}')
        return None, None, {}


def _build_epoch_summary_html(event_summary: dict) -> str:
    """
    Build an HTML summary of the epoching result from the event_summary dict
    returned by Epoch_meg.

    Parameters
    ----------
    event_summary : dict
        As returned by Epoch_meg — keys: source, id_to_trial_type,
        count_before_merge, count_after_merge, total_before_merge,
        total_after_merge, dropped_count, all_ids.

    Returns
    -------
    str
        HTML string ready to embed in the subject report.
    """
    if not event_summary:
        return ''

    source = event_summary.get('source', 'unknown')
    source_labels = {
        'bids_tsv':     'BIDS <code>*_events.tsv</code> file',
        'find_events':  'stimulus channel (mne.find_events)',
        'fixed_length': 'fixed-length segmentation (no event channel)',
    }
    source_label = source_labels.get(source, source)

    id_to_trial_type = event_summary.get('id_to_trial_type', {})
    count_before = event_summary.get('count_before_merge', {})
    count_after  = event_summary.get('count_after_merge',  {})
    all_ids      = event_summary.get('all_ids', [])
    total_before = event_summary.get('total_before_merge', 0)
    total_after  = event_summary.get('total_after_merge',  0)
    dropped      = event_summary.get('dropped_count', 0)

    html = f'<p><b>Event source:</b> {source_label}</p>'
    html += (
        f'<p><b>Events detected:</b> {total_before} &nbsp;&#8594;&nbsp; '
        f'<b>Epochs used:</b> {total_after}'
    )
    if dropped:
        html += (
            f' &nbsp; (<b>{dropped}</b> event(s) dropped or merged because of '
            f'overlapping onset times — see <code>event_repeated</code> setting)'
        )
    html += '</p>'

    # Per-ID table (skip for fixed-length where IDs are synthetic)
    if all_ids and source != 'fixed_length':
        table_style = (
            'border-collapse:collapse;font-size:0.9em;'
            'margin-top:4px;margin-bottom:8px;'
        )
        th_style = (
            'background:#f0f0f0;padding:4px 10px;'
            'border:1px solid #bbb;text-align:center;'
        )
        td_style = 'padding:3px 10px;border:1px solid #bbb;text-align:center;'

        html += f'<table style="{table_style}">'
        html += (
            f'<thead><tr>'
            f'<th style="{th_style}">Event ID</th>'
            f'<th style="{th_style}">Trial type / label</th>'
            f'<th style="{th_style}">Events detected</th>'
            f'<th style="{th_style}">Epochs used (after merge)</th>'
            f'</tr></thead><tbody>'
        )
        for eid in all_ids:
            # id_to_trial_type keys may be int or str depending on the source
            label = (id_to_trial_type.get(eid)
                     or id_to_trial_type.get(str(eid), ''))
            n_before = (count_before.get(eid)
                        or count_before.get(str(eid), 0))
            n_after  = (count_after.get(eid)
                        or count_after.get(str(eid), 0))
            html += (
                f'<tr>'
                f'<td style="{td_style}">{eid}</td>'
                f'<td style="{td_style}">{label}</td>'
                f'<td style="{td_style}">{n_before}</td>'
                f'<td style="{td_style}">{n_after}</td>'
                f'</tr>'
            )
        # Totals row
        html += (
            f'<tr style="font-weight:bold;background:#f7f7f7;">'
            f'<td style="{td_style}" colspan="2">Total</td>'
            f'<td style="{td_style}">{total_before}</td>'
            f'<td style="{td_style}">{total_after}</td>'
            f'</tr>'
        )
        html += '</tbody></table>'

    return html


def Epoch_meg(epoching_params, data: mne.io.Raw, file_path: str = None):
    """
    Epoch MEG data based on the parameters provided in the config file.

    Event detection priority:
    1. BIDS *_events.tsv alongside the MEG file (most reliable).
    2. mne.find_events on the best available stim channel
       (prefers combined channels like STI101; excludes known-noisy channels).
    3. Fixed-length epochs (fallback when use_fixed_length_epochs=True).

    Parameters
    ----------
    epoching_params : dict
        Dictionary with parameters for epoching.
    data : mne.io.Raw
        MEG data to be epoched.
    file_path : str, optional
        Path to the original MEG file, used to locate the BIDS events TSV.

    Returns
    -------
    dict_epochs_mg : dict
        Keys: 'mag', 'grad', 'epoching_mode', 'epoch_onset_times_s', 'event_summary'
    """
    from collections import Counter

    event_dur = epoching_params['event_dur']
    epoch_tmin = epoching_params['epoch_tmin']
    epoch_tmax = epoching_params['epoch_tmax']
    stim_channel = epoching_params['stim_channel']
    use_fixed_length_epochs = epoching_params['use_fixed_length_epochs']
    fixed_epoch_duration = epoching_params['fixed_epoch_duration']
    fixed_epoch_overlap = epoching_params['fixed_epoch_overlap']
    min_event_count = 2 if use_fixed_length_epochs else 1

    # Build case-insensitive sets from config lists (fall back to empty sets if keys are absent)
    _preferred = {ch.upper() for ch in epoching_params.get('preferred_stim_channels', [])}
    _noisy = {ch.upper() for ch in epoching_params.get('noisy_stim_channels', [])}

    user_specified_stim = stim_channel is not None

    if stim_channel is None:
        picks_stim = mne.pick_types(data.info, stim=True)
        stim_channel = [data.info['chs'][ch]['ch_name'] for ch in picks_stim]
    print('___MEGqc___: ', 'Stimulus channels available:', stim_channel)

    picks_magn = data.copy().pick('mag').ch_names if 'mag' in data else None
    picks_grad = data.copy().pick('grad').ch_names if 'grad' in data else None
    picks_eeg = data.copy().pick('eeg').ch_names if 'eeg' in data else None

    epochs_grad, epochs_mag, epochs_eeg = None, None, None
    epoching_mode = 'stim'
    # Track raw events array (before Epochs applies event_repeated / boundary drop)
    # so we can report "n events before merge" separately from "n epochs used".
    _raw_events = None
    _id_to_trial_type = {}

    def _make_fixed_length_epochs(picks, channel_type):
        if fixed_epoch_duration <= 0:
            print('___MEGqc___: Fixed-length epoch duration must be positive.')
            return None
        if fixed_epoch_overlap < 0:
            print('___MEGqc___: Fixed-length epoch overlap must be >= 0.')
            return None
        if fixed_epoch_overlap >= fixed_epoch_duration:
            print('___MEGqc___: Fixed-length epoch overlap must be smaller than duration.')
            return None
        epochs = mne.make_fixed_length_epochs(
            data, duration=fixed_epoch_duration, overlap=fixed_epoch_overlap, preload=True)
        if picks:
            epochs.pick(picks)
        print(f'___MEGqc___: Fixed-length epochs created for {channel_type}: {len(epochs)} epochs.')
        return epochs

    def _apply_fixed_length_fallback(reason):
        if not use_fixed_length_epochs:
            print('___MEGqc___: ', reason)
            return
        print(f'___MEGqc___: {reason} Falling back to fixed-length epoching.')
        nonlocal epoching_mode, epochs_mag, epochs_grad, epochs_eeg
        epoching_mode = 'fixed_length'
        if picks_magn:
            epochs_mag = _make_fixed_length_epochs(picks_magn, 'magnetometers')
        if picks_grad:
            epochs_grad = _make_fixed_length_epochs(picks_grad, 'gradiometers')
        if picks_eeg:
            epochs_eeg = _make_fixed_length_epochs(picks_eeg, 'EEG')

    def _make_stim_epochs(events):
        nonlocal epochs_mag, epochs_grad, epochs_eeg
        if picks_magn:
            epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax,
                                    preload=True, baseline=None,
                                    event_repeated=epoching_params['event_repeated'])
        if picks_grad:
            epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax,
                                     preload=True, baseline=None,
                                     event_repeated=epoching_params['event_repeated'])
        if picks_eeg:
            epochs_eeg = mne.Epochs(data, events, picks=picks_eeg, tmin=epoch_tmin, tmax=epoch_tmax,
                                    preload=True, baseline=None,
                                    event_repeated=epoching_params['event_repeated'])
        parts = []
        if epochs_mag is not None: parts.append(f'mag: {len(epochs_mag)}')
        if epochs_grad is not None: parts.append(f'grad: {len(epochs_grad)}')
        if epochs_eeg is not None: parts.append(f'eeg: {len(epochs_eeg)}')
        print(f'___MEGqc___: Epochs created — {", ".join(parts)}')

    # ── 1. Try BIDS events TSV (most reliable) ────────────────────────────────
    bids_events = None
    if file_path is not None and not user_specified_stim:
        bids_events, bids_tsv, _id_to_trial_type = _read_bids_events_tsv(file_path, data)

    if bids_events is not None and len(bids_events) >= min_event_count:
        print(f'___MEGqc___: Using BIDS events TSV ({len(bids_events)} events).')
        _raw_events = bids_events
        _make_stim_epochs(bids_events)
    else:
        # ── 2. mne.find_events fallback ───────────────────────────────────────
        if not stim_channel:
            _apply_fixed_length_fallback('No stimulus channels detected.')
        else:
            if not user_specified_stim:
                clean = [ch for ch in stim_channel if ch.upper() not in _noisy]
                preferred = [ch for ch in clean if ch.upper() in _preferred]
                if preferred:
                    stim_channel = preferred
                    print(f'___MEGqc___: Using preferred stim channel(s): {stim_channel}')
                elif clean:
                    stim_channel = clean
                    print(f'___MEGqc___: Using stim channel(s) (noise channels excluded): {stim_channel}')
                else:
                    print(f'___MEGqc___: Warning — only noisy stim channels found; trying anyway: {stim_channel}')
            try:
                events = mne.find_events(data, stim_channel=stim_channel,
                                         min_duration=event_dur, uint_cast=True, verbose=False)
                print(f'___MEGqc___: find_events → {len(events)} events, '
                      f'IDs={np.unique(events[:, 2]).tolist() if len(events) else []}')
                if len(events) < min_event_count:
                    _apply_fixed_length_fallback(
                        f'Only {len(events)} event(s) found (minimum required: {min_event_count}).')
                else:
                    _raw_events = events
                    _make_stim_epochs(events)
            except Exception as exc:
                _apply_fixed_length_fallback(f'mne.find_events failed ({exc}).')

    # ── Build per-ID event summary ─────────────────────────────────────────────
    # count_before_merge: from the raw events array before mne.Epochs drops anything
    # count_after_merge: from epochs.events after boundary-drop + event_repeated handling
    ep_ref = epochs_mag if epochs_mag is not None else (epochs_grad if epochs_grad is not None else epochs_eeg)
    if _raw_events is not None and len(_raw_events):
        count_before = dict(Counter(int(x) for x in _raw_events[:, 2]))
    else:
        count_before = {}

    if ep_ref is not None and hasattr(ep_ref, 'events') and len(ep_ref.events):
        count_after = dict(Counter(int(x) for x in ep_ref.events[:, 2]))
    elif epoching_mode == 'fixed_length' and ep_ref is not None:
        # Fixed-length epochs have one synthetic event ID (0 or 1); represent as total
        n = len(ep_ref)
        count_after = {0: n}
        count_before = {0: n}
    else:
        count_after = {}

    all_ids = sorted(set(list(count_before.keys()) + list(count_after.keys())))
    total_before = sum(count_before.values())
    total_after = sum(count_after.values())

    event_summary = {
        'source': 'bids_tsv' if (bids_events is not None and len(bids_events) >= min_event_count)
                  else ('find_events' if _raw_events is not None else 'fixed_length'),
        'id_to_trial_type': _id_to_trial_type,
        'count_before_merge': count_before,
        'count_after_merge': count_after,
        'total_before_merge': total_before,
        'total_after_merge': total_after,
        'dropped_count': total_before - total_after,
        'all_ids': all_ids,
    }
    if total_before != total_after:
        print(f'___MEGqc___: Events before merge/drop: {total_before}, '
              f'used in epochs: {total_after} '
              f'(dropped/merged: {total_before - total_after})')
    else:
        print(f'___MEGqc___: Events: {total_before}, all used in epochs.')

    # ── Assemble return dict ───────────────────────────────────────────────────
    first_samp = getattr(data, 'first_samp', 0)
    epoch_onset_times_s = None
    for ep_obj in [epochs_mag, epochs_grad, epochs_eeg]:
        if ep_obj is not None and len(ep_obj.events):
            epoch_onset_times_s = (
                (ep_obj.events[:, 0] - first_samp) / data.info['sfreq']
            ).tolist()
            break

    return {
        'mag': epochs_mag,
        'grad': epochs_grad,
        'eeg': epochs_eeg,
        'epoching_mode': epoching_mode,
        'epoch_onset_times_s': epoch_onset_times_s,
        'event_summary': event_summary,
    }




def check_chosen_ch_types(m_or_g_chosen: List, channels_objs: dict):
    """
    Check if the channels which the user gave in config file to analize actually present in the data set.

    IMPORTANT: This function works on a **copy** of ``m_or_g_chosen`` so
    that the caller's original list (e.g. the config dict) is never mutated.
    This is critical when the pipeline processes both MEG and EEG files in the
    same subject loop — mutating the shared config list would permanently
    remove channel types discovered absent in one file but present in another.

    Parameters
    ----------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad, eeg. These are the ones the user chose.
    channels_objs : dict
        Dictionary with channel names for each channel type: mag, grad, eeg. These are the ones present in the data set.

    Returns
    -------
    m_or_g_chosen : list
        **New** list with only those channel types that are both requested
        and present in the data.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.

    """

    # Work on a copy so the caller's list is never mutated.
    m_or_g_chosen = list(m_or_g_chosen)

    skipped_str = ''

    if not any(ch in m_or_g_chosen for ch in SUPPORTED_CH_TYPES):
        skipped_str = "No valid channel types to analyze. Check parameter ch_types in config file."
        raise ValueError(skipped_str)

    type_labels = {'mag': 'magnetometers', 'grad': 'gradiometers', 'eeg': 'EEG channels'}

    for ch in list(m_or_g_chosen):  # iterate copy since we may modify the list
        if ch in channels_objs and len(channels_objs[ch]) == 0:
            remaining = [t for t in m_or_g_chosen if t != ch and t in channels_objs and len(channels_objs[t]) > 0]
            label = type_labels.get(ch, ch)
            if remaining:
                skipped_str = (f"There are no {label} in this data set: check parameter ch_types in config file. "
                               f"Analysis will be done for: {', '.join(type_labels.get(r, r) for r in remaining)}.")
            else:
                skipped_str = f"There are no {label} in this data set."
            print(f'___MEGqc___: {skipped_str}')
            m_or_g_chosen.remove(ch)

    if not any(channels_objs.get(ch) for ch in SUPPORTED_CH_TYPES):
        skipped_str = "There are no magnetometers, gradiometers, nor EEG channels in this data set. Analysis will not be done."
        raise ValueError(skipped_str)

    # Now m_or_g_chosen contain only those channel types which are present in the data set and were chosen by the user.

    return m_or_g_chosen, skipped_str


def choose_channels(raw: mne.io.Raw):
    """
    Separate channels by 'mag', 'grad', and 'eeg'.
    Done this way, because pick() or pick_types() sometimes gets wrong results, especialy for CTF data.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG or EEG data

    Returns
    -------
    channels : dict
        dict with ch names separated by mag, grad, and eeg

    """

    channels = {'mag': [], 'grad': [], 'eeg': []}

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
    Load MEG or EEG data from a file.

    Parameters
    ----------
    file_path : str
        Path to the data file (FIF, CTF .ds, EDF, BDF, BrainVision, EEGLAB, EGI, Neuroscan).

    Returns
    -------
    raw : mne.io.Raw
        Loaded data.
    shielding_str : str
        String with information about active shielding (empty for EEG).
    meg_system : str
        Identifier: 'Triux', 'CTF', 'EEG', or 'OTHER'.
    modality : str
        'meg' or 'eeg'.

    """

    shielding_str = ''
    meg_system = None
    modality = 'meg'

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

    # Normalise for extension checks
    fp_lower = file_path.lower()

    if os.path.isdir(file_path) and file_path.endswith('.ds'):
        # It's a CTF data directory.
        # BIDS suffix is the primary modality classifier (same logic as FIF):
        # files named *_eeg.ds are independent EEG recordings; *_meg.ds are MEG.
        _bids_suffix_is_eeg_ctf = '_eeg.' in os.path.basename(file_path).lower()
        if _bids_suffix_is_eeg_ctf:
            meg_system = 'EEG'
            modality = 'eeg'
        else:
            meg_system = 'CTF'
            modality = 'meg'

        print("___MEGqc___: ", "Loading CTF data...")
        raw = mne.io.read_raw_ctf(file_path, preload=True, verbose='ERROR')
        print(f"___MEGqc___: Recording duration: {raw.times[-1] / 60:.2f} min")

        # ── Safeguard: reclassify CTF .ds with BIDS suffix _meg but 0 MEG chs ─
        # Same logic as the FIF safeguard: only fires when a *_meg.ds file has
        # literally zero mag/grad channels. Normal CTF MEG files that carry a
        # few EEG electrodes alongside the MEG sensors stay as MEG.
        if modality == 'meg':
            _has_any_meg_ch = any(
                mne.channel_type(raw.info, idx) in ('mag', 'grad')
                for idx in range(len(raw.info['ch_names']))
            )
            if not _has_any_meg_ch:
                meg_system = 'EEG'
                modality = 'eeg'
                print('___MEGqc___: CTF .ds file has suffix _meg but contains no MEG channels — '
                      'reclassifying as independent EEG recording (safeguard).')

        if _bids_suffix_is_eeg_ctf:
            print(f'___MEGqc___: CTF .ds file classified as independent EEG recording (BIDS suffix _eeg).')

    elif file_path.endswith('.fif'):
        # It's a FIF file.  The BIDS suffix in the filename is the primary
        # modality classifier: files named ``*_meg.fif`` are MEG recordings
        # (even if they contain a few EEG electrodes), while files named
        # ``*_eeg.fif`` are independent EEG recordings stored in FIF format.
        _bids_suffix_is_eeg = '_eeg.' in os.path.basename(file_path).lower()
        if _bids_suffix_is_eeg:
            meg_system = 'EEG'
            modality = 'eeg'
        else:
            meg_system = 'Triux'
            modality = 'meg'

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

        # ── Safeguard: reclassify FIF with BIDS suffix _meg but no MEG chs ─
        # Extremely rare edge case: a file named *_meg.fif that has zero
        # mag/grad channels.  We fall back to EEG only when there are truly
        # no MEG channels.  This never fires for normal MEG files that
        # happen to carry a few EEG electrodes alongside the MEG sensors.
        if modality == 'meg':
            _has_any_meg_ch = any(
                mne.channel_type(raw.info, idx) in ('mag', 'grad')
                for idx in range(len(raw.info['ch_names']))
            )
            if not _has_any_meg_ch:
                meg_system = 'EEG'
                modality = 'eeg'
                print('___MEGqc___: FIF file has suffix _meg but contains no MEG channels — '
                      'reclassifying as independent EEG recording (safeguard).')

        if _bids_suffix_is_eeg:
            print(f'___MEGqc___: FIF file classified as independent EEG recording (BIDS suffix _eeg).')

    # ──── EEG FORMATS ──────────────────────────────────────────────────────
    elif fp_lower.endswith('.edf'):
        print("___MEGqc___: Loading EDF data...")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'

    elif fp_lower.endswith('.bdf'):
        print("___MEGqc___: Loading BDF data...")
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'

    elif fp_lower.endswith('.vhdr'):
        print("___MEGqc___: Loading BrainVision data...")
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'

    elif fp_lower.endswith('.set'):
        print("___MEGqc___: Loading EEGLAB data...")
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'

    elif os.path.isdir(file_path) and fp_lower.endswith('.mff'):
        print("___MEGqc___: Loading EGI/MFF data...")
        raw = mne.io.read_raw_egi(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'

    elif fp_lower.endswith('.cnt'):
        print("___MEGqc___: Loading Neuroscan CNT data...")
        raw = mne.io.read_raw_cnt(file_path, preload=True, verbose='ERROR')
        meg_system = 'EEG'
        modality = 'eeg'
    # ──── END EEG FORMATS ──────────────────────────────────────────────────

    else:
        raise ValueError(
            f"Unsupported file format or file does not exist: {file_path}. "
            "Supported formats: .fif, .ds (CTF), .edf, .bdf, .vhdr, .set, .mff, .cnt")

    if modality == 'eeg':
        print(f"___MEGqc___: Recording duration: {raw.times[-1] / 60:.2f} min")

    return raw, shielding_str, meg_system, modality


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


# ──── EEG-specific helpers ─────────────────────────────────────────────────

def _apply_bids_channel_types(raw, file_path):
    """Read the companion BIDS ``*_channels.tsv`` and correct channel types.

    Many EEG recordings contain non-EEG channels (EOG, EMG, ECG, MISC) that
    MNE may auto-detect as EEG.  The BIDS ``channels.tsv`` sidecar explicitly
    labels each channel, so we use it to set the correct types in the raw
    object *before* any downstream channel separation.

    Parameters
    ----------
    raw : mne.io.Raw
        Loaded raw data (modified in-place).
    file_path : str
        Path to the data file.  The function derives the ``*_channels.tsv``
        path by replacing the file extension and suffix.

    Returns
    -------
    raw : mne.io.Raw
        Data with corrected channel types.
    correction_str : str
        Description of corrections applied (empty if none).
    """
    import re as _re

    correction_str = ''

    # Derive channels.tsv path from data file path
    # BIDS pattern: sub-XX_task-YY_..._eeg.edf  →  sub-XX_task-YY_..._channels.tsv
    base_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    # Remove extension
    name_no_ext = os.path.splitext(base_name)[0]
    # Remove suffix (_eeg, _meg, etc.) — last _word segment
    name_no_suffix = _re.sub(r'_[a-zA-Z]+$', '', name_no_ext)
    channels_tsv_path = os.path.join(base_dir, name_no_suffix + '_channels.tsv')

    if not os.path.isfile(channels_tsv_path):
        print(f'___MEGqc___: No channels.tsv found at {channels_tsv_path} — skipping channel type correction.')
        return raw, correction_str

    try:
        channels_df = pd.read_csv(channels_tsv_path, sep='\t')
    except Exception as exc:
        print(f'___MEGqc___: Could not read channels.tsv: {exc}')
        return raw, correction_str

    if 'name' not in channels_df.columns or 'type' not in channels_df.columns:
        print('___MEGqc___: channels.tsv missing required columns (name, type) — skipping.')
        return raw, correction_str

    # BIDS type → MNE type mapping
    bids_to_mne = {
        'EEG': 'eeg',
        'EOG': 'eog',
        'VEOG': 'eog',
        'HEOG': 'eog',
        'EMG': 'emg',
        'ECG': 'ecg',
        'MISC': 'misc',
        'STIM': 'stim',
        'REF': 'eeg',      # Reference EEG channels stay as EEG
        'TRIG': 'stim',
        'BIO': 'bio',
        'RESP': 'resp',
    }

    corrections = {}
    raw_ch_names = set(raw.ch_names)

    for _, row in channels_df.iterrows():
        ch_name = str(row['name']).strip()
        bids_type = str(row['type']).strip().upper()

        if ch_name not in raw_ch_names:
            continue

        mne_target_type = bids_to_mne.get(bids_type)
        if mne_target_type is None:
            continue

        # Get current MNE type
        ch_idx = raw.ch_names.index(ch_name)
        current_type = mne.channel_type(raw.info, ch_idx)

        if current_type != mne_target_type:
            corrections[ch_name] = mne_target_type

    if corrections:
        try:
            raw.set_channel_types(corrections)
            correction_str = (
                f'Corrected channel types from BIDS channels.tsv for '
                f'{len(corrections)} channel(s): '
                f'{", ".join(f"{k}→{v}" for k, v in list(corrections.items())[:10])}'
                + ('...' if len(corrections) > 10 else ''))
            print(f'___MEGqc___: {correction_str}')
        except RuntimeError:
            # Likely a projector conflict — remove projectors containing the
            # channels we want to retype, then retry.
            projs_to_keep = []
            correction_ch_set = set(corrections.keys())
            for proj in raw.info['projs']:
                col_names = set(proj['data'].get('col_names', []))
                if col_names & correction_ch_set:
                    print(f'___MEGqc___: Removing projector "{proj["desc"]}" '
                          f'that conflicts with channel type correction.')
                else:
                    projs_to_keep.append(proj)
            raw.info['projs'] = projs_to_keep
            try:
                raw.set_channel_types(corrections)
                correction_str = (
                    f'Corrected channel types from BIDS channels.tsv for '
                    f'{len(corrections)} channel(s) (after removing conflicting projectors): '
                    f'{", ".join(f"{k}→{v}" for k, v in list(corrections.items())[:10])}'
                    + ('...' if len(corrections) > 10 else ''))
                print(f'___MEGqc___: {correction_str}')
            except Exception as exc:
                correction_str = f'Failed to apply BIDS channel type corrections: {exc}'
                print(f'___MEGqc___: WARNING: {correction_str}')
        except Exception as exc:
            correction_str = f'Failed to apply BIDS channel type corrections: {exc}'
            print(f'___MEGqc___: WARNING: {correction_str}')
    else:
        print('___MEGqc___: BIDS channels.tsv found — all channel types already match MNE detection.')

    return raw, correction_str


def _strip_eeg_reference_suffix(name):
    """Strip common EEG reference suffixes from a channel name.

    Referential montage recordings often name channels as ``Fp1-M2``,
    ``F3-A1``, ``C3-Ref`` etc.  Standard montages only contain the
    electrode name (``Fp1``, ``F3``, ``C3``).  This helper removes the
    reference part so that montage matching succeeds.

    Returns the stripped name, or the original if no known suffix is found.
    """
    import re
    return re.sub(r'[-_](?:M[12]|A[12]|Ref|LE|RE|AVG|REF|ref|le|re|avg)$', '', name, flags=re.IGNORECASE)


def apply_eeg_montage(raw, eeg_settings):
    """
    Detect and apply a standard EEG montage for topomap plotting.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    eeg_settings : dict
        Settings from config: {'montage': 'auto'|'standard_1020'|..., ...}

    Returns
    -------
    raw : mne.io.Raw
        Data with montage applied (in-place).
    montage_str : str
        Description of montage applied.
    """
    montage_name = eeg_settings.get('montage', 'auto')
    montage_str = ''

    # Collect EEG channel names
    eeg_ch_names = [ch for idx, ch in enumerate(raw.ch_names)
                    if mne.channel_type(raw.info, idx) == 'eeg']

    if not eeg_ch_names:
        return raw, 'No EEG channels found — montage not applied.'

    if montage_name == 'auto':
        # --- Pass 1: try matching raw channel names directly ---
        for candidate in ['standard_1005', 'standard_1010', 'standard_1020']:
            try:
                montage = mne.channels.make_standard_montage(candidate)
                montage_ch_names = set(montage.ch_names)
                overlap = set(eeg_ch_names) & montage_ch_names
                if len(overlap) >= len(eeg_ch_names) * 0.5:  # 50% match threshold
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw.set_montage(montage, on_missing='warn')
                    montage_str = (f'Auto-detected montage: {candidate} '
                                   f'({len(overlap)}/{len(eeg_ch_names)} channels matched)')
                    print(f'___MEGqc___: {montage_str}')
                    return raw, montage_str
            except Exception:
                continue

        # --- Pass 2: strip reference suffixes (e.g. Fp1-M2 → Fp1) and retry ---
        stripped_map = {}  # old_name → new_name  (only if actually changed)
        for ch in eeg_ch_names:
            stripped = _strip_eeg_reference_suffix(ch)
            if stripped != ch:
                stripped_map[ch] = stripped

        if stripped_map:
            print(f'___MEGqc___: Attempting montage match after stripping '
                  f'reference suffixes from {len(stripped_map)} channel(s): '
                  f'{list(stripped_map.items())[:5]}...')
            stripped_names = [stripped_map.get(ch, ch) for ch in eeg_ch_names]

            for candidate in ['standard_1005', 'standard_1010', 'standard_1020']:
                try:
                    montage = mne.channels.make_standard_montage(candidate)
                    montage_ch_names = set(montage.ch_names)
                    overlap = set(stripped_names) & montage_ch_names
                    if len(overlap) >= len(eeg_ch_names) * 0.5:
                        # Rename channels in the raw object
                        raw.rename_channels(stripped_map)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            raw.set_montage(montage, on_missing='warn')
                        montage_str = (
                            f'Auto-detected montage: {candidate} '
                            f'({len(overlap)}/{len(eeg_ch_names)} channels matched '
                            f'after stripping reference suffixes from '
                            f'{len(stripped_map)} channel(s))')
                        print(f'___MEGqc___: {montage_str}')
                        return raw, montage_str
                except Exception:
                    continue

        montage_str = ('Could not auto-detect EEG montage from channel names. '
                       'Topomaps may not be available.')
        print(f'___MEGqc___: WARNING: {montage_str}')
    else:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage(montage, on_missing='warn')
            montage_str = f'Applied montage: {montage_name}'
            print(f'___MEGqc___: {montage_str}')
        except Exception as e:
            # If explicit montage fails, try stripping reference suffixes
            eeg_ch_stripped = {}
            for ch in eeg_ch_names:
                stripped = _strip_eeg_reference_suffix(ch)
                if stripped != ch:
                    eeg_ch_stripped[ch] = stripped
            if eeg_ch_stripped:
                try:
                    raw.rename_channels(eeg_ch_stripped)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw.set_montage(montage, on_missing='warn')
                    montage_str = (f'Applied montage: {montage_name} '
                                   f'(after stripping reference suffixes from '
                                   f'{len(eeg_ch_stripped)} channel(s))')
                    print(f'___MEGqc___: {montage_str}')
                    return raw, montage_str
                except Exception as e2:
                    montage_str = f'Failed to apply montage {montage_name}: {e2}'
                    print(f'___MEGqc___: WARNING: {montage_str}')
            else:
                montage_str = f'Failed to apply montage {montage_name}: {e}'
                print(f'___MEGqc___: WARNING: {montage_str}')

    return raw, montage_str


def apply_eeg_reference(raw, eeg_settings):
    """
    Apply EEG re-referencing.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    eeg_settings : dict
        Settings with 'reference_method'.

    Returns
    -------
    raw : mne.io.Raw
        Re-referenced data (in-place).
    ref_str : str
        Description.
    """
    ref_method = eeg_settings.get('reference_method', 'average')
    ref_str = ''

    if ref_method == 'none':
        ref_str = 'EEG reference: original (no re-referencing applied).'
    elif ref_method == 'average':
        try:
            raw.set_eeg_reference('average', projection=True)
            ref_str = 'EEG reference: common average reference applied.'
        except Exception as e:
            ref_str = f'EEG average reference failed ({e}), keeping original reference.'
    elif ref_method.upper() == 'REST':
        try:
            raw.set_eeg_reference('REST')
            ref_str = 'EEG reference: REST (Reference Electrode Standardization Technique) applied.'
        except Exception as e:
            try:
                raw.set_eeg_reference('average', projection=True)
                ref_str = f'REST reference failed ({e}), fell back to average reference.'
            except Exception as e2:
                ref_str = f'EEG reference failed ({e2}), keeping original reference.'
    else:
        ref_str = f'Unknown reference method: {ref_method}, skipping re-referencing.'

    print(f'___MEGqc___: {ref_str}')
    return raw, ref_str


def add_EEG_lobes(channels_objs):
    """
    Assign brain region (lobe) labels to EEG channels based on standard
    10-20 / 10-10 / 10-05 naming conventions.

    Electrode naming rules:
    - First 1-3 letters = region prefix (Fp, AF, F, FC, FT, C, CP, T, TP, P, PO, O, I)
    - Trailing digit: odd = left hemisphere, even = right hemisphere
    - Trailing 'z' or 'Z' = midline

    Parameters
    ----------
    channels_objs : dict
        Dictionary with channel objects for each channel type.

    Returns
    -------
    channels_objs : dict
        Updated with lobe and lobe_color set on each channel.
    lobes_color_coding_str : str
        Description string.
    """
    # Prefix → sub-region mapping (longest prefixes first for matching)
    _prefix_to_region = [
        ('Fp', 'Frontal'),
        ('AF', 'Frontal'),
        ('FC', 'Frontal'),
        ('FT', 'Temporal'),
        ('F',  'Frontal'),
        ('CP', 'Parietal'),
        ('C',  'Central'),
        ('TP', 'Temporal'),
        ('T',  'Temporal'),
        ('PO', 'Occipital'),
        ('P',  'Parietal'),
        ('O',  'Occipital'),
        ('I',  'Occipital'),
    ]

    # Explicit named channels
    _explicit = {
        'Cz': ('Central', ''), 'Fz': ('Frontal', ''), 'Pz': ('Parietal', ''),
        'Oz': ('Occipital', ''), 'Fpz': ('Frontal', ''), 'FCz': ('Frontal', ''),
        'CPz': ('Parietal', ''), 'POz': ('Occipital', ''), 'Iz': ('Occipital', ''),
        'A1': ('Reference', 'Left'), 'A2': ('Reference', 'Right'),
        'M1': ('Reference', 'Left'), 'M2': ('Reference', 'Right'),
        'Nz': ('Reference', ''),
        # Old 10-20 naming: T3/T4 = temporal, T5/T6 = temporal
        'T3': ('Temporal', 'Left'), 'T4': ('Temporal', 'Right'),
        'T5': ('Temporal', 'Left'), 'T6': ('Temporal', 'Right'),
        'T7': ('Temporal', 'Left'), 'T8': ('Temporal', 'Right'),
    }

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
        'Frontal': '#1f77b4',
        'Temporal': '#2ca02c',
        'Parietal': '#e377c2',
        'Occipital': '#bcbd22',
        'Reference': '#7f7f7f',
        'Extra': '#d3d3d3',
    }

    def _classify(name):
        """Return (region, laterality) for a channel name."""
        if name in _explicit:
            return _explicit[name]

        # Try prefix match (longest first, handled by ordering)
        region = None
        for prefix, reg in _prefix_to_region:
            if name.startswith(prefix):
                region = reg
                break
        if region is None:
            return ('Extra', '')

        # Determine laterality from trailing characters
        suffix = name.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        if suffix:
            try:
                num = int(suffix)
                return (region, 'Left' if num % 2 == 1 else 'Right')
            except ValueError:
                pass
        if name.endswith('z') or name.endswith('Z'):
            return (region, '')
        return (region, '')

    for key, value in channels_objs.items():
        for ch in value:
            region, laterality = _classify(ch.name)
            if laterality and region not in ('Central', 'Reference', 'Extra'):
                full_lobe = f'{laterality} {region}'
            else:
                full_lobe = region
            ch.lobe = full_lobe
            ch.lobe_color = lobe_colors.get(full_lobe, lobe_colors.get(region, '#d3d3d3'))

    lobes_color_coding_str = ('Color coding by brain region based on standard '
                              '10-20/10-10 EEG electrode nomenclature.')
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

    elif meg_system.upper() == 'EEG':
        channels_full, lobes_color_coding_str = add_EEG_lobes(channels_full)

        # assign 'EEG' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'EEG'

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

    The output directory is constructed as ``<derivatives_root>/.tmp/<subject>``
    where ``subject`` is inferred from the first path component starting with
    ``sub-`` in ``file_path``. Using ``derivatives_root`` allows callers to place
    temporary files outside the read-only BIDS directory if needed.
    """

    norm_path = os.path.normpath(file_path)
    components = norm_path.split(os.sep)

    subject = next((part for part in components if part.startswith('sub-')), None)
    if subject is None:
        raise ValueError("Unable to determine subject from file path for temporary output")

    # Profile-scoped temporary intermediates are stored in ".tmp" to make
    # their transient nature explicit for users inspecting derivatives.
    output_dir = os.path.join(derivatives_root, '.tmp', subject)
    output_dir = os.path.abspath(output_dir)
    print("Output directory:", output_dir)

    # Create the target folder if it does not exist already
    os.makedirs(output_dir, exist_ok=True)
    print("Directory created (or already exists):", output_dir)

    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() == '.ds':
        ext = '.fif'

    # EEG formats cannot be saved back in their original format via MNE;
    # convert to FIF for intermediate storage.
    _eeg_extensions = {'.edf', '.bdf', '.vhdr', '.set', '.cnt', '.mff'}
    if ext.lower() in _eeg_extensions:
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
    # Prefer modern ".tmp" folder and also clean legacy "temp" if present.
    for folder_name in ('.tmp', 'temp'):
        temp_dir = os.path.join(derivatives_root, folder_name)
        temp_dir = os.path.abspath(temp_dir)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            print("Removing directory:", temp_dir)

    return


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params: dict, file_path: str,
                       derivatives_root: str, eeg_settings: dict = None):
    """
    Here all the initial actions needed to analyse MEG or EEG data are done:

    - read data file,
    - separate channel types (mag / grad / eeg),
    - apply EEG montage and reference if applicable,
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
        Path to the data file.
    derivatives_root : str
        Path to the derivatives directory.
    eeg_settings : dict, optional
        EEG-specific settings (montage, reference_method). Only used when modality is EEG.

    Returns
    -------
    tuple
        (meg_system, dict_epochs_mg, chs_by_lobe, channels,
         raw_cropped_filtered_path, raw_cropped_filtered_resampled_path,
         raw_cropped_path, raw_path, info_derivs, stim_deriv,
         event_summary_deriv,
         shielding_str, epoching_str, sensors_derivs, m_or_g_chosen,
         m_or_g_skipped_str, lobes_color_coding_str, resample_str)
    """

    print('___MEGqc___: ', 'Reading data from file:', file_path)

    raw, shielding_str, meg_system, modality = load_data(file_path)

    # ── Correct channel types from BIDS channels.tsv (if available) ─────
    # This must happen before choose_channels() so that non-EEG channels
    # (EOG, EMG, ECG, MISC) are not misclassified as EEG by MNE.
    raw, bids_correction_str = _apply_bids_channel_types(raw, file_path)

    # Working with channels:
    channels = choose_channels(raw)

    if meg_system == 'CTF':  # ONLY FOR CTF we do this! Return raw with changed channel types.
        channels, raw = change_ch_type_CTF(raw, channels)

    # EEG-specific: apply montage and reference
    if meg_system == 'EEG' and eeg_settings:
        raw, montage_str = apply_eeg_montage(raw, eeg_settings)
        raw, ref_str = apply_eeg_reference(raw, eeg_settings)
        shielding_str = '<p>' + montage_str + '</p><p>' + ref_str + '</p>'
        if bids_correction_str:
            shielding_str += '<p>' + bids_correction_str + '</p>'
        # Re-run choose_channels in case montage renamed channels
        channels = choose_channels(raw)

    # Turn channel names into objects:
    channels_objs = {key: [QC_channel(name=ch_name, type=key) for ch_name in value] for key, value in channels.items()}

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

    stim_deriv, stim_channel_event_counts = stim_data_to_df(raw_cropped, epoching_params)

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
                                    picks='eeg' if meg_system == 'EEG' else 'meg',
                                    method=filtering_settings['method'], iir_params=None)
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
        # Load data into memory before any saving or resampling
        raw_cropped_filtered.load_data()
        # Save the unfiltered cropped data; both CROPPED and FILTERED paths point
        # to the same file because no filtering was applied.
        raw_cropped_path = save_meg_with_suffix(file_path, derivatives_root,
                                                raw_cropped_filtered, final_suffix="CROPPED")
        raw_cropped_filtered_path = raw_cropped_path  # identical data — no filtering done

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
            raw_cropped_filtered_resampled_path = raw_cropped_filtered_path  # same file, no resampling
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)

    del raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw
    gc.collect()

    # Load data
    raw_cropped_filtered, shielding_str, meg_system, _modality = load_data(raw_cropped_filtered_path)

    # Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching?
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered, file_path=file_path)
    event_summary = dict_epochs_mg.get('event_summary', {})
    epoch_onset_times_s = dict_epochs_mg.get('epoch_onset_times_s')

    # ── Build BIDS events.tsv comparison data ──────────────────────────────
    # Always attempt to read the BIDS events.tsv (even if Epoch_meg already
    # used it) so we can store per-trial-type info for the plotting module.
    bids_events_info = {}
    try:
        bids_events, bids_tsv_path, bids_id_to_tt = _read_bids_events_tsv(
            file_path, raw_cropped_filtered
        )
        if bids_events is not None and bids_tsv_path is not None:
            # Per-trial-type counts
            from collections import Counter
            bids_id_counts = dict(Counter(int(x) for x in bids_events[:, 2]))
            bids_events_info = {
                'tsv_path': bids_tsv_path,
                'total_events': int(len(bids_events)),
                'id_counts': {int(k): int(v) for k, v in bids_id_counts.items()},
                'id_to_trial_type': {int(k): str(v) for k, v in bids_id_to_tt.items()},
                # Store per-event onsets (seconds) and trial_type for timeline plot
                # Limit to 2000 events to keep JSON manageable
                'event_onsets_s': [
                    float((s - raw_cropped_filtered.first_samp) / raw_cropped_filtered.info['sfreq'])
                    for s in bids_events[:2000, 0]
                ],
                'event_ids': [int(x) for x in bids_events[:2000, 2]],
            }
    except Exception as exc:
        print(f'___MEGqc___: Could not build BIDS events comparison: {exc}')

    # ── Assemble EventSummary JSON derivative ──────────────────────────────
    event_summary_payload = {
        'event_summary': event_summary,
        'epoch_onset_times_s': epoch_onset_times_s,
        'sfreq': float(raw_cropped_filtered.info['sfreq']),
        'stim_channel_event_counts': stim_channel_event_counts,
        'bids_events_info': bids_events_info,
    }
    event_summary_deriv = [
        QC_derivative(event_summary_payload, 'EventSummary', 'json')
    ]

    epoching_str = ''
    if dict_epochs_mg.get('epoching_mode') == 'fixed_length':
        epoching_str = (
            '<p>No stimulus channels were detected or too few events found. '
            f'The data was epoched into fixed-length segments with duration '
            f'{epoching_params["fixed_epoch_duration"]} s and overlap '
            f'{epoching_params["fixed_epoch_overlap"]} s.</p>'
        )
        epoching_str += _build_epoch_summary_html(event_summary)
        epoching_str += '<br></br>'
    elif dict_epochs_mg['mag'] is None and dict_epochs_mg['grad'] is None and dict_epochs_mg.get('eeg') is None:
        epoching_str = (
            '<p>No epoching could be done in this data set: no events found. '
            'Quality measurements were only performed on the entire time series. '
            'If this was not expected, try: 1) checking the presence of a stimulus '
            'channel in the data set, 2) setting the stimulus channel explicitly in '
            'the config file, 3) setting a different event duration in the config '
            'file.</p><br></br>'
        )
    else:
        # Successful event-based epoching — show full per-ID summary
        epoching_str = _build_epoch_summary_html(event_summary)
        epoching_str += '<br></br>'

    resample_str = '<p>' + resample_str + '</p>'

    # Extract chs_by_lobe into a data frame
    sensors_derivs = chs_dict_to_csv(chs_by_lobe, file_name_prefix='Sensors')

    raw_path = file_path

    return (meg_system, dict_epochs_mg, chs_by_lobe, channels,
            raw_cropped_filtered_path, raw_cropped_filtered_resampled_path,
            raw_cropped_path, raw_path, info_derivs, stim_deriv,
            event_summary_deriv,
            shielding_str, epoching_str, sensors_derivs, m_or_g_chosen,
            m_or_g_skipped_str, lobes_color_coding_str, resample_str)


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
