import numpy as np
import pandas as pd
import mne
from typing import List
from scipy.signal import find_peaks
from meg_qc.plotting.universal_plots import assign_epoched_std_ptp_to_channels
from meg_qc.plotting.universal_html_report import simple_metric_basic
from meg_qc.calculation.metrics.STD_meg_qc import make_dict_global_std_ptp, make_dict_local_std_ptp, \
    get_big_small_std_ptp_all_data, get_noisy_flat_std_ptp_epochs
from meg_qc.calculation.initial_meg_qc import chs_dict_to_csv
from meg_qc.calculation.initial_meg_qc import chs_dict_to_csv, load_data
import copy


# The manual PtP version.

def neighbour_peak_amplitude(max_pair_dist_sec: float, sfreq: int, pos_peak_locs: np.ndarray, neg_peak_locs: np.ndarray,
                             pos_peak_magnitudes: np.ndarray, neg_peak_magnitudes: np.ndarray):
    """
    Find a pair: postive + negative peak and calculates the amplitude between them. 
    If no neighbour is found withing given distance - this peak is skipped. 
    If several neighbours are found - several pairs are created. 
    As the result a mean peak-to-peak distance is calculated over all detected pairs for given chunck of data

    Parameters:
    -----------
    max_pair_dist_sec : float
        Maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair
    sfreq : int
        Sampling frequency of data. Attention to which data is used! original or resampled.
    pos_peak_locs : np.ndarray
        Output of peak_finder function - positions of detected Positive peaks
    neg_peak_locs : np.ndarray
        Output of peak_finder function - positions of detected Negative peaks
    pos_peak_magnitudes : np.ndarray
        Output of peak_finder function - magnitudes of detected Positive peaks
    neg_peak_magnitudes : np.ndarray
        Output of peak_finder function - magnitudes of detected Negative peaks

    Returns:
    --------
    mean_amplitude : float
        Mean value over all detected peak pairs for this chunck of data.
    amplitude : np.ndarray
        Array of all detected peak pairs for this chunck of data.

    """

    if len(pos_peak_locs) < 1 or len(neg_peak_locs) < 1:
        return 0, None

    pair_dist = max_pair_dist_sec * sfreq
    pairs_magnitudes = []
    pairs_locs = []

    # Looping over all positive peaks
    for posit_peak_ind, posit_peak_loc in enumerate(pos_peak_locs):

        # Finding the value in neg_peak_locs which is closest to posit_peak_loc
        closest_negative_peak_index = np.abs(neg_peak_locs - posit_peak_loc).argmin()

        # Check if the closest negative peak is within the given distance
        if np.abs(neg_peak_locs[closest_negative_peak_index] - posit_peak_loc) <= pair_dist / 2:
            pairs_locs.append([pos_peak_locs[posit_peak_ind], neg_peak_locs[closest_negative_peak_index]])
            pairs_magnitudes.append(
                [pos_peak_magnitudes[posit_peak_ind], neg_peak_magnitudes[closest_negative_peak_index]])

    # if no positive+negative pairs were fould (no corresponding peaks at given distamce to each other) ->
    # - give the difference between min and max value of the data + a note that no pairs were found

    if len(pairs_magnitudes) == 0:
        pairs_magnitudes.append([max(pos_peak_magnitudes), min(neg_peak_magnitudes)])
        print('___MEGqc___: ',
              'No pairs found with the given distance between peaks. The amplitude is calculated as the difference between the max and min value of the entire data. \nConsider changing the distance between peaks in the config file.')

    amplitude = np.zeros(len(pairs_magnitudes), )
    # print('___MEGqc___: ', 'Number of peaks pairs used for for PtP calculation: ', len(pairs_magnitudes))
    # TODO: think of: sometimes we get only a few pairs, like 1-2-3, this is not enough for an accurate estimation of the mean amplitude.
    # Set minimum of pairs or another approach?

    for i, pair in enumerate(pairs_magnitudes):
        amplitude[i] = pair[0] - pair[1]

    return np.mean(amplitude), amplitude


def get_ptp_all_data(data: mne.io.Raw, channels: List, sfreq: int, ptp_thresh_lvl: float, max_pair_dist_sec: float):
    """
    Calculate peak-to-peak amplitude for all channels over whole data (not epoched).

    Parameters:
    -----------
    data : mne.io.Raw
        Raw data
    channels : List
        List of channel names to be used for peak-to-peak amplitude calculation
    sfreq : int
        Sampling frequency of data. Attention to which data is used! original or resampled.
    ptp_thresh_lvl : float
        The level definig how the PtP threshold will be scaled. Higher number will result in more peaks detected.
        The threshold is calculated as (max - min) / ptp_thresh_lvl
    max_pair_dist_sec : float
        Maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair

    Returns:
    --------
    peak_ampl_channels : dict
        Peak-to-peak amplitude values for each channel.

    """

    data_channels = data.get_data(picks=channels)

    peak_ampl_channels = []
    for one_ch_data in data_channels:
        thresh = (max(one_ch_data) - min(one_ch_data)) / ptp_thresh_lvl
        # can also change the whole thresh to a single number setting

        # mne.preprocessing.peak_finder() gives error if there are no peaks detected. We use scipy.signal.find_peaks() instead here:
        pos_peak_locs, _ = find_peaks(one_ch_data,
                                      prominence=thresh)  # assume there are no peaks within 0.5 seconds from each other.
        pos_peak_magnitudes = one_ch_data[pos_peak_locs]

        neg_peak_locs, _ = find_peaks(-one_ch_data,
                                      prominence=thresh)  # assume there are no peaks within 0.5 seconds from each other.
        neg_peak_magnitudes = one_ch_data[neg_peak_locs]

        pp_ampl, _ = neighbour_peak_amplitude(max_pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs,
                                              pos_peak_magnitudes, neg_peak_magnitudes)
        peak_ampl_channels.append(pp_ampl)

    # add channel name for every std value:
    peak_ampl_channels_named = {}
    for i, ch in enumerate(channels):
        peak_ampl_channels_named[ch] = peak_ampl_channels[i]

    return peak_ampl_channels_named


def get_ptp_epochs(channels: List, epochs_mg: mne.Epochs, sfreq: int, ptp_thresh_lvl: float, max_pair_dist_sec: float):
    """
    Calculate peak-to-peak amplitude for every epoch and every channel (mag or grad).

    Parameters:
    -----------
    channels : List
        List of channel names to be used for peak-to-peak amplitude calculation
    epochs_mg : mne.Epochs
        Epochs data
    sfreq : int
        Sampling frequency of data. Attention to which data is used! original or resampled.
    ptp_thresh_lvl : float  
        The level definig how the PtP threshold will be scaled. Higher number will result in more peaks detected.
        The threshold is calculated as (max - min) / ptp_thresh_lvl
    max_pair_dist_sec : float
        Maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair

    Returns:
    --------
    pd.DataFrame
        Dataframe containing the mean peak-to-peak aplitude for each epoch for each channel

    """
    dict_ep = {}

    # get 1 epoch, 1 channel and calculate PtP on its data:
    for ep in range(0, len(epochs_mg)):
        peak_ampl_epoch = []
        for ch_name in channels:
            data_ch_epoch = epochs_mg[ep].get_data(picks=ch_name)[0][0]
            # [0][0] is because get_data creats array in array in array, it expects several epochs, several channels, but we only need  one.

            thresh = (max(data_ch_epoch) - min(data_ch_epoch)) / ptp_thresh_lvl
            # can also change the whole thresh to a single number setting

            pos_peak_locs, _ = find_peaks(data_ch_epoch,
                                          prominence=thresh)  # assume there are no peaks within 0.5 seconds from each other.
            pos_peak_magnitudes = data_ch_epoch[pos_peak_locs]

            neg_peak_locs, _ = find_peaks(-data_ch_epoch,
                                          prominence=thresh)  # assume there are no peaks within 0.5 seconds from each other.
            neg_peak_magnitudes = data_ch_epoch[neg_peak_locs]

            pp_ampl, _ = neighbour_peak_amplitude(max_pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs,
                                                  pos_peak_magnitudes, neg_peak_magnitudes)
            peak_ampl_epoch.append(pp_ampl)

        dict_ep[ep] = peak_ampl_epoch

    return pd.DataFrame(dict_ep, index=channels)


def make_simple_metric_ptp_manual(ptp_manual_params: dict, big_ptp_with_value_all_data: dict,
                                  small_ptp_with_value_all_data: dict, channels: dict, deriv_epoch_ptp: dict,
                                  metric_local: bool, m_or_g_chosen: List, onset_times: list=None):
    """
    Create a simple metric for peak-to-peak amplitude. 
    Global: The metric is calculated for all data (not epoched) and 
    Local: for each epoch.

    Parameters:
    -----------
    ptp_manual_params : dict
        Dictionary containing the parameters for the metric
    big_ptp_with_value_all_data : dict
        Dict (mag, grad) with channels with peak-to-peak amplitude higher than the threshold + the value of the peak-to-peak amplitude
    small_ptp_with_value_all_data : dict
        Dict (mag, grad) with channels with peak-to-peak amplitude lower than the threshold + the value of the peak-to-peak amplitude
    channels : dict
        Dict (mag, grad) with all channel names
    deriv_epoch_ptp : dict
        Dict (mag, grad) with peak-to-peak amplitude for each epoch for each channel
    metric_local : bool
        If True, the local metric was calculated and will be added to the simple metric
    m_or_g_chosen : List
        'mag' or 'grad' or both, chosen by user in config file

    Returns:
    --------
    simple_metric : dict
        Dict (mag, grad) with the simple metric for peak-to-peak manual amplitude

    """

    metric_global_name = 'ptp_manual_all'
    metric_global_description = 'Peak-to-peak deviation of the data over the entire time series (not epoched): ... The ptp_lvl is the peak-to-peak threshold level set by the user. Threshold = ... The channel where data is higher than this threshod is considered as noisy. Same: if the std of some channel is lower than -threshold, this channel is considered as flat. In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'ptp_manual_epoch'
    if metric_local == True:
        metric_local_description = 'Peak-to-peak deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take Peak-to-Peak amplitude (PtP) of data of THIS channel in THIS epoch. 2) Take PtP of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_channel_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat. '
    else:
        metric_local_description = 'Not calculated. Ne epochs found'

    metric_global_content = {'mag': None, 'grad': None, 'eeg': None}
    metric_local_content = {'mag': None, 'grad': None, 'eeg': None}
    for m_or_g in m_or_g_chosen:
        metric_global_content[m_or_g] = make_dict_global_std_ptp(ptp_manual_params, big_ptp_with_value_all_data[m_or_g],
                                                                 small_ptp_with_value_all_data[m_or_g],
                                                                 channels[m_or_g], 'ptp')

        if metric_local is True:
            metric_local_content[m_or_g] = make_dict_local_std_ptp(ptp_manual_params,
                                                                   deriv_epoch_ptp[m_or_g][1].content,
                                                                   deriv_epoch_ptp[m_or_g][2].content,
                                                                   percent_noisy_flat_allowed=ptp_manual_params[
                                                                       'allow_percent_noisy_flat_epochs'],
                                                                   onset_times=onset_times)
            # deriv_epoch_std[m_or_g][1].content is df with big std(noisy), df_epoch_std[m_or_g][2].content is df with small std(flat)
        else:
            metric_local_content[m_or_g] = None

    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'],
                                        metric_global_content['grad'], metric_local_name, metric_local_description,
                                        metric_local_content['mag'], metric_local_content['grad'],
                                        metric_global_content_eeg=metric_global_content.get('eeg'),
                                        metric_local_content_eeg=metric_local_content.get('eeg'))

    return simple_metric


def PP_manual_meg_qc(
    ptp_manual_params: dict,
    channels: dict,
    chs_by_lobe: dict,
    dict_epochs_mg: dict,
    data_path: str,
    m_or_g_chosen: List
):
    """
    Main Peak to peak amplitude function. Calculates:

    - Peak to peak amplitudes (PtP) of data for each channel over all time series.
    - Channels with big PtP (noisy) and small PtP (flat) over all time series.
    - PtP of data for each channel  in each epoch.
    - Epochs with big PtP (noisy) and small PtP (flat).

    PtP is calculated as the average amplitude between the positive and negative peaks,
    which are located at a certain distance from each other. The distance is set by the user in config file.

    Parameters:
    -----------
    ptp_manual_params : dict
        Dictionary containing the parameters for the metric
    channels : dict
        Dict (mag, grad) with all channel names
    chs_by_lobe : dict
        dictionary with channels grouped first by ch type and then by lobe: chs_by_lobe['mag']['Left Occipital'] or chs_by_lobe['grad']['Left Occipital']
    dict_epochs_mg : dict
        Dict (mag, grad) with epochs for each channel. Should be the same for both channels. Used only to check if epochs are present.
    data : mne.io.Raw
        Raw data
    m_or_g_chosen : List
        'mag' or 'grad' or both, chosen by user in config file.

    Returns:
    --------
    derivs_ptp : List
        List with QC_deriv objects for peak-to-peak amplitude (figures and csv files)
    simple_metric_ptp_manual : dict
        Simple metric for peak-to-peak amplitude
    pp+manual_str : str
        String with notes about PtP manual for report

    """
    data, shielding_str, meg_system, _modality = load_data(data_path)

    sfreq = data.info['sfreq']

    big_ptp_with_value_all_data = {}
    small_ptp_with_value_all_data = {}
    derivs_ptp = []
    derivs_list = []
    peak_ampl = {}
    noisy_flat_epochs_derivs = {}

    chs_by_lobe_ptp = copy.deepcopy(chs_by_lobe)
    # copy here, because want to keep original dict unchanged. 
    # In principal it s good to collect all data about channel metrics there BUT if the metrics are run in parallel this might produce conflicts 
    # (without copying  dict can be chanaged both inside+outside this function even when it is not returned.)

    for m_or_g in m_or_g_chosen:

        peak_ampl[m_or_g] = get_ptp_all_data(data, channels[m_or_g], sfreq,
                                             ptp_thresh_lvl=ptp_manual_params['ptp_thresh_lvl'],
                                             max_pair_dist_sec=ptp_manual_params['max_pair_dist_sec'])

        # Add ptp data into channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe_ptp[m_or_g]:
            for ch in chs_by_lobe_ptp[m_or_g][lobe]:
                ch.ptp_overall = peak_ampl[m_or_g][ch.name]

        big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(
            peak_ampl[m_or_g], channels[m_or_g], ptp_manual_params['std_lvl'])

    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None or dict_epochs_mg.get('eeg') is not None:  # if epochs are present
        for m_or_g in m_or_g_chosen:
            df_ptp = get_ptp_epochs(channels[m_or_g], dict_epochs_mg[m_or_g], sfreq,
                                    ptp_manual_params['ptp_thresh_lvl'], ptp_manual_params['max_pair_dist_sec'])

            chs_by_lobe_ptp[m_or_g] = assign_epoched_std_ptp_to_channels(what_data='peaks',
                                                                         chs_by_lobe=chs_by_lobe_ptp[m_or_g],
                                                                         df_std_ptp=df_ptp)  # for easier plotting

            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_std_ptp_epochs(df_ptp, m_or_g, 'ptp', ptp_manual_params[
                'noisy_channel_multiplier'], ptp_manual_params['flat_multiplier'], ptp_manual_params[
                                                                                 'allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

            metric_local = True
        pp_manual_str = ''

    else:
        metric_local = False
        pp_manual_str = 'Peak-to-Peak amplitude per epoch can not be calculated because no events are present. Check stimulus channel.'
        print('___MEGqc___: ', pp_manual_str)

    simple_metric_ptp_manual = make_simple_metric_ptp_manual(ptp_manual_params, big_ptp_with_value_all_data,
                                                             small_ptp_with_value_all_data, channels,
                                                             noisy_flat_epochs_derivs, metric_local, m_or_g_chosen,
                                                             onset_times=dict_epochs_mg.get('epoch_onset_times_s'))

    # Extract chs_by_lobe into a data frame
    df_deriv = chs_dict_to_csv(chs_by_lobe_ptp, file_name_prefix='PtPsManual')

    derivs_ptp += derivs_list + df_deriv

    # each deriv saved into a separate list and only at the end put together because this way they keep the right order:
    # first everything about mags, then everything about grads. - in this order they ll be added to repot.
    # TODO: Can use fig_order parameter of QC_derivative to adjust figure order in the report, if got nothing better to do XD.

    return derivs_ptp, simple_metric_ptp_manual, pp_manual_str