# -*- coding: utf-8 -*-
"""
Module: meg_qc_numba.py

Numba-accelerated version of peak-to-peak amplitude calculations for MEG QC.
"""

import numpy as np
import pandas as pd
import mne
from typing import List
from scipy.signal import find_peaks
from numba import njit, prange

# Imports from existing meg_qc package
from meg_qc.plotting.universal_plots import assign_epoched_std_ptp_to_channels
from meg_qc.plotting.universal_html_report import simple_metric_basic
from meg_qc.calculation.metrics.STD_meg_qc import (
    make_dict_global_std_ptp,
    make_dict_local_std_ptp,
    get_big_small_std_ptp_all_data,
    get_noisy_flat_std_ptp_epochs,
)
from meg_qc.calculation.initial_meg_qc import chs_dict_to_csv, load_data
import copy


# --- Numba-accelerated peak detection ---
@njit(cache=True)
def detect_peaks_numba(x: np.ndarray, thresh: float):
    """
    Detects simple positive and negative peaks based on threshold and neighbor comparison.
    Returns arrays of positive and negative peak indices.
    """
    n = x.size
    pos_list = []
    neg_list = []
    for i in range(1, n-1):
        v = x[i]
        # Positive peak
        if v > thresh and v > x[i-1] and v > x[i+1]:
            pos_list.append(i)
        # Negative peak
        if -v > thresh and -v > -x[i-1] and -v > -x[i+1]:
            neg_list.append(i)
    pos_arr = np.empty(len(pos_list), np.int64)
    neg_arr = np.empty(len(neg_list), np.int64)
    for i in range(len(pos_list)):
        pos_arr[i] = pos_list[i]
    for i in range(len(neg_list)):
        neg_arr[i] = neg_list[i]
    return pos_arr, neg_arr


# --- Numba-accelerated pairing logic ---
@njit(cache=True)
def neighbour_peak_amplitude_numba(
    max_pair_dist_sec: float,
    sfreq: int,
    pos_locs: np.ndarray,
    neg_locs: np.ndarray,
    pos_mags: np.ndarray,
    neg_mags: np.ndarray
):
    """
    Pair each positive peak to the closest negative peak within max distance.
    Returns mean amplitude and array of amplitudes.
    """
    if pos_locs.size == 0 or neg_locs.size == 0:
        return 0.0, np.empty(0, np.float64)

    max_dist = (max_pair_dist_sec * sfreq) / 2.0
    amps_list = []
    n_neg = neg_locs.size

    for i in range(pos_locs.size):
        p = pos_locs[i]
        best_dist = max_dist + 1.0
        best_mag = 0.0
        # scan all negative peaks to find nearest
        for j in range(n_neg):
            d = abs(p - neg_locs[j])
            if d < best_dist:
                best_dist = d
                best_mag = neg_mags[j]
        if best_dist <= max_dist:
            amps_list.append(pos_mags[i] - best_mag)

    if len(amps_list) == 0:
        fb = pos_mags.max() - neg_mags.min()
        return fb, np.array([fb], np.float64)

    arr = np.empty(len(amps_list), np.float64)
    for i in range(len(amps_list)):
        arr[i] = amps_list[i]
    return arr.mean(), arr


# --- Batch computation over epochs with parallelism ---
@njit(cache=True, parallel=True)
def compute_ptp_epochs_numba(
    data: np.ndarray,
    sfreq: int,
    ptp_thresh_lvl: float,
    max_pair_dist_sec: float
):
    """
    Computes PtP amplitudes for epoched data in parallel.
    data shape: (n_epochs, n_channels, n_times)
    Returns array of shape (n_channels, n_epochs).
    """
    n_epochs, n_channels, _ = data.shape
    out = np.empty((n_channels, n_epochs), np.float64)

    for ep in prange(n_epochs):
        for ch in range(n_channels):
            x = data[ep, ch, :]
            thr = (x.max() - x.min()) / ptp_thresh_lvl
            pos, neg = detect_peaks_numba(x, thr)
            mean_amp, _ = neighbour_peak_amplitude_numba(
                max_pair_dist_sec, sfreq, pos, neg, x[pos], x[neg]
            )
            out[ch, ep] = mean_amp
    return out


# --- Adapted get_ptp_epochs using Numba ---
def get_ptp_epochs(
    channels: List,
    epochs_mg: mne.Epochs,
    sfreq: int,
    ptp_thresh_lvl: float,
    max_pair_dist_sec: float
) -> pd.DataFrame:
    """
    Peak-to-peak amplitude for each channel/epoch using Numba acceleration.
    """
    data = epochs_mg.get_data(picks=channels)  # (n_epochs, n_channels, n_times)
    ptp_arr = compute_ptp_epochs_numba(data, sfreq, ptp_thresh_lvl, max_pair_dist_sec)
    return pd.DataFrame(ptp_arr, index=channels)


# --- Adapted get_ptp_all_data using Numba peak pairing ---
def get_ptp_all_data(
    data: mne.io.Raw,
    channels: List,
    sfreq: int,
    ptp_thresh_lvl: float,
    max_pair_dist_sec: float
) -> dict:
    """
    Calculates PtP amplitude across entire recording for each channel.
    Uses Numba pairing logic for speedup.
    """
    data_channels = data.get_data(picks=channels)
    peak_ampl = []
    for one_ch_data in data_channels:
        thresh = (one_ch_data.max() - one_ch_data.min()) / ptp_thresh_lvl
        pos, neg = detect_peaks_numba(one_ch_data, thresh)
        pos_mags = one_ch_data[pos]
        neg_mags = one_ch_data[neg]
        mean_amp, _ = neighbour_peak_amplitude_numba(
            max_pair_dist_sec, sfreq, pos, neg, pos_mags, neg_mags
        )
        peak_ampl.append(mean_amp)
    return {ch: amp for ch, amp in zip(channels, peak_ampl)}


# --- Rest of pipeline unchanged ---
def make_simple_metric_ptp_manual(
    ptp_manual_params: dict,
    big_ptp_with_value_all_data: dict,
    small_ptp_with_value_all_data: dict,
    channels: dict,
    deriv_epoch_ptp: dict,
    metric_local: bool,
    m_or_g_chosen: List
) -> dict:
    """
    Create a simple metric for peak-to-peak amplitude (global and local).
    """
    metric_global_name = 'ptp_manual_all'
    metric_global_description = (
        'Peak-to-peak deviation of the data over the entire time series (not epoched): '  
        '...'
    )
    metric_local_name = 'ptp_manual_epoch'
    if metric_local:
        metric_local_description = (
            'Peak-to-peak deviation per epoch: ...'
        )
    else:
        metric_local_description = 'Not calculated. No epochs found'

    metric_global_content = {'mag': None, 'grad': None, 'eeg': None}
    metric_local_content  = {'mag': None, 'grad': None, 'eeg': None}

    for m_or_g in m_or_g_chosen:
        metric_global_content[m_or_g] = make_dict_global_std_ptp(
            ptp_manual_params,
            big_ptp_with_value_all_data[m_or_g],
            small_ptp_with_value_all_data[m_or_g],
            channels[m_or_g],
            'ptp'
        )
        if metric_local:
            metric_local_content[m_or_g] = make_dict_local_std_ptp(
                ptp_manual_params,
                deriv_epoch_ptp[m_or_g][1].content,
                deriv_epoch_ptp[m_or_g][2].content,
                percent_noisy_flat_allowed=ptp_manual_params['allow_percent_noisy_flat_epochs']
            )
        else:
            metric_local_content[m_or_g] = None

    simple_metric = simple_metric_basic(
        metric_global_name,
        metric_global_description,
        metric_global_content['mag'],
        metric_global_content['grad'],
        metric_local_name,
        metric_local_description,
        metric_local_content['mag'],
        metric_local_content['grad'],
        metric_global_content_eeg=metric_global_content.get('eeg'),
        metric_local_content_eeg=metric_local_content.get('eeg'),
    )
    return simple_metric


def PP_manual_meg_qc_numba(
    ptp_manual_params: dict,
    channels: dict,
    chs_by_lobe: dict,
    dict_epochs_mg: dict,
    data_path: str,
    m_or_g_chosen: List
):
    """
    Main PtP QC function: global and per-epoch peak-to-peak amplitudes.
    """
    # Load data
    data, shielding_str, meg_system, _modality = load_data(data_path)
    sfreq = data.info['sfreq']

    big_ptp_with_value_all_data = {}
    small_ptp_with_value_all_data = {}
    derivs_ptp = []
    derivs_list = []
    peak_ampl = {}
    noisy_flat_epochs_derivs = {}
    chs_by_lobe_ptp = copy.deepcopy(chs_by_lobe)

    # Global PtP per channel
    for m_or_g in m_or_g_chosen:
        peak_ampl[m_or_g] = get_ptp_all_data(
            data,
            channels[m_or_g],
            sfreq,
            ptp_thresh_lvl=ptp_manual_params['ptp_thresh_lvl'],
            max_pair_dist_sec=ptp_manual_params['max_pair_dist_sec']
        )
        # Annotate channels
        for lobe in chs_by_lobe_ptp[m_or_g]:
            for ch in chs_by_lobe_ptp[m_or_g][lobe]:
                ch.ptp_overall = peak_ampl[m_or_g][ch.name]
        big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g] = (
            get_big_small_std_ptp_all_data(
                peak_ampl[m_or_g],
                channels[m_or_g],
                ptp_manual_params['std_lvl']
            )
        )

    # Local PtP per epoch, if available
    metric_local = False
    if dict_epochs_mg.get('mag') is not None or dict_epochs_mg.get('grad') is not None or dict_epochs_mg.get('eeg') is not None:
        metric_local = True
        for m_or_g in m_or_g_chosen:
            df_ptp = get_ptp_epochs(
                channels[m_or_g],
                dict_epochs_mg[m_or_g],
                sfreq,
                ptp_manual_params['ptp_thresh_lvl'],
                ptp_manual_params['max_pair_dist_sec']
            )
            # Assign epoched metrics
            chs_by_lobe_ptp[m_or_g] = assign_epoched_std_ptp_to_channels(
                what_data='peaks',
                chs_by_lobe=chs_by_lobe_ptp[m_or_g],
                df_std_ptp=df_ptp
            )
            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_std_ptp_epochs(
                df_ptp,
                m_or_g,
                'ptp',
                ptp_manual_params['noisy_channel_multiplier'],
                ptp_manual_params['flat_multiplier'],
                ptp_manual_params['allow_percent_noisy_flat_epochs']
            )
            derivs_list += noisy_flat_epochs_derivs[m_or_g]
        pp_manual_str = ''
    else:
        pp_manual_str = (
            'Peak-to-Peak amplitude per epoch cannot be calculated because no events are present.'
        )
        print('___MEGqc___:', pp_manual_str)

    # Compile results
    simple_metric_ptp_manual = make_simple_metric_ptp_manual(
        ptp_manual_params,
        big_ptp_with_value_all_data,
        small_ptp_with_value_all_data,
        channels,
        noisy_flat_epochs_derivs,
        metric_local,
        m_or_g_chosen
    )

    df_deriv = chs_dict_to_csv(chs_by_lobe_ptp, file_name_prefix='PtPsManual')
    derivs_ptp = derivs_list + df_deriv

    return derivs_ptp, simple_metric_ptp_manual, pp_manual_str
