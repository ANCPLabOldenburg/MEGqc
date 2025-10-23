
# # Annotate muscle artifacts
# 
# Explanation from MNE:
# Muscle contractions produce high frequency activity that can mask brain signal
# of interest. Muscle artifacts can be produced when clenching the jaw,
# swallowing, or twitching a cranial muscle. Muscle artifacts are most
# noticeable in the range of 110-140 Hz.
# 
# This code uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
# segments where muscle activity is likely present. This is done by band-pass
# filtering the data in the 110-140 Hz range. Then, the envelope is taken using
# the hilbert analytical signal to only consider the absolute amplitude and not
# the phase of the high frequency signal. The envelope is z-scored and summed
# across channels and divided by the square root of the number of channels.
# Because muscle artifacts last several hundred milliseconds, a low-pass filter
# is applied on the averaged z-scores at 4 Hz, to remove transient peaks.
# Segments above a set threshold are annotated as ``BAD_muscle``. In addition,
# the ``min_length_good`` parameter determines the cutoff for whether short
# spans of "good data" in between muscle artifacts are included in the
# surrounding "BAD" annotation.

import time
import os
import gc
import mne
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
# from mne.preprocessing import annotate_muscle_zscore
from meg_qc.miscellaneous.optimizations.artifact_detection_ancp import annotate_muscle_zscore
from typing import List
from meg_qc.plotting.universal_plots import QC_derivative
from meg_qc.calculation.initial_meg_qc import (load_data, save_meg_with_suffix)

def find_powerline_noise_short(raw, psd_params, psd_params_internal, m_or_g_chosen, channels):

    """
    Find powerline noise in the data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    psd_params : dict
        The parameters for PSD calculation originally defined in the config file.
    psd_params_internal : dict
        The parameters for PSD calculation originally defined in the internal config file. 
    m_or_g_chosen : List
        The channel types chosen for the analysis: 'mag' or 'grad'.
    
    Returns
    -------
    noisy_freqs : dict
        The noisy frequencies found in the data separated by channel type.
    
    """

    method = psd_params_internal['method']
    prominence_lvl_pos_avg = psd_params_internal['prominence_lvl_pos_avg']
    psd_step_size = psd_params['psd_step_size']
    sfreq=raw.info['sfreq']
    nfft=int(sfreq/psd_step_size)
    nperseg=int(sfreq/psd_step_size)


    noisy_freqs = {}
    for m_or_g in m_or_g_chosen:

        psds, freqs = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=channels[m_or_g], n_jobs=1, n_fft=nfft, n_per_seg=nperseg).get_data(return_freqs=True)
        avg_psd=np.mean(psds,axis=0) # average psd over all channels
        prominence_pos=(max(avg_psd) - min(avg_psd)) / prominence_lvl_pos_avg

        noisy_freqs_indexes, _ = find_peaks(avg_psd, prominence=prominence_pos)
        noisy_freqs [m_or_g] = freqs[noisy_freqs_indexes]

    return noisy_freqs


def make_simple_metric_muscle(m_or_g_decided: str, z_scores_dict: dict, muscle_str_joined: str):

    """
    Make a simple metric dict for muscle events.
    
    Parameters
    ----------
    m_or_g_decided : str
        The channel type used for muscle detection: 'mag' or 'grad'.
    z_scores_dict : dict
        The z-score thresholds used for muscle detection.
    muscle_str_joined : str
        Notes about muscle detection to use as description.
        
    Returns
    -------
    simple_metric : dict
        A simple metric dict for muscle events.
        
    """

    #if the string contains <p> or </p> - remove it:
    muscle_str_joined = muscle_str_joined.replace("<p>", "").replace("</p>", "")


    simple_metric = {
    'description': muscle_str_joined+'Data below shows detected high frequency (muscle) events.',
    'muscle_calculated_using': m_or_g_decided,
    'unit_muscle_evet_times': 'seconds',
    'unit_muscle_event_zscore': 'z-score',
    'zscore_thresholds': z_scores_dict}

    return simple_metric


def filter_noise_before_muscle_detection(raw: mne.io.Raw, noisy_freqs_global: dict, muscle_freqs: List = [110, 140]):

    """
    Filter out power line noise and other noisy freqs in range of muscle artifacts before muscle artifact detection.
    MNE advices to filter power line noise. We also filter here noisy frequencies in range of muscle artifacts.
    List of noisy frequencies for filtering come from PSD artifact detection function. If any noise peaks were found there for mags or grads 
    they will all be passed here and checked if they are in range of muscle artifacts.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    noisy_freqs_global : dict
        The noisy frequencies found in PSD artifact detection function.
    muscle_freqs : List
        The frequencies of muscle artifacts, usually 110 and 140 Hz.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw data with filtered noise or not filtered if no noise was found.
        
    """

    #Find out if the data contains powerline noise freqs or other noisy in range of muscle artifacts - notch filter them before muscle artifact detection:

    # - collect all values in moisy_freqs_global into one list:
    noisy_freqs=[]
    for key in noisy_freqs_global.keys():
        noisy_freqs.extend(np.round(noisy_freqs_global[key], 1))
    
    
    # - detect power line freqs and their harmonics
    powerline=[50, 60]

    #Were the power line freqs found in this data?
    powerline_found = [x for x in powerline if x in noisy_freqs]

    # add harmonics of powerline freqs to the list of noisy freqs IF they are in range of muscle artifacts [110-140Hz]:
    for freq in powerline_found:
        for i in range(1, 3):
            if freq*i not in powerline_found and muscle_freqs[0]<freq*i<muscle_freqs[1]:
                powerline_found.append(freq*i)


    noisy_freqs_all = powerline_found

    #(issue almost never happens, but might):
    # find Nyquist frequncy for this data to check if the noisy freqs are not higher than it (otherwise filter will fail):
    noisy_freqs_all = [x for x in noisy_freqs_all if x<raw.info['sfreq']/2 - 1]


    # - notch filter the data (it has to be preloaded before. done in the parent function):
    if noisy_freqs_all==[]:
        print('___MEGqc___: ', 'No powerline noise found in data or PSD artifacts detection was not performed. Notch filtering skipped.')
    elif (len(noisy_freqs_all))>0:
        print('___MEGqc___: ', 'Powerline noise was found in data. Notch filtering at: ', noisy_freqs_all, ' Hz')
        raw.notch_filter(noisy_freqs_all)
    else:
        print('Something went wrong with powerline frequencies. Notch filtering skipped. Check parameter noisy_freqs_all')

    return raw


def attach_dummy_data(raw: mne.io.Raw, attach_seconds: int = 5):

    """
    Attach a dummy start and end to the data to avoid filtering artifacts at the beginning/end of the recording.
    Dummy data is mirrored data: take beginning of real data, mirror it and attach to the start of the recording.
    Same for the end of the recording.
    It will be cut off after filtering.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    attach_seconds : int
        The number of seconds to attach to the start and end of the recording.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw data with dummy start attached.
        
    """
    
    print('Duration original: ', raw.n_times / raw.info['sfreq'])
    # Attach a dummy start to the data to avoid filtering artifacts at the beginning of the recording:
    raw_dummy_start=raw.copy()
    raw_dummy_start_data = raw_dummy_start.crop(tmin=0, tmax=attach_seconds-1/raw.info['sfreq']).get_data()
    inverted_data_start = np.flip(raw_dummy_start_data, axis=1) # Invert the data

    # Attach a dummy end to the data to avoid filtering artifacts at the end of the recording:
    raw_dummy_end=raw.copy()
    raw_dummy_end_data = raw_dummy_end.crop(tmin=raw_dummy_end.times[int(-attach_seconds*raw.info['sfreq']-1/raw.info['sfreq'])], tmax=raw_dummy_end.times[-1]).get_data()
    inverted_data_end = np.flip(raw_dummy_end_data, axis=1) # Invert the data

    # Update the raw object with the inverted data
    raw_dummy_start._data = inverted_data_start
    raw_dummy_end._data = inverted_data_end
    # print('Duration of start attached: ', raw_dummy_start.n_times / raw.info['sfreq'])
    # print('Duration of end attached: ', raw_dummy_end.n_times / raw.info['sfreq'])

    # Concatenate the inverted data with the original data
    raw = mne.concatenate_raws([raw_dummy_start, raw, raw_dummy_end])
    # print('Duration after attaching dummy data: ', raw.n_times / raw.info['sfreq'])

    return raw



def calculate_muscle_NO_threshold(raw_muscle_path, m_or_g_decided, muscle_params, threshold_muscle, muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, muscle_str_joined, dataset_path, orig_meg_type):

    """
    Calculate muscle artifacts without thresholding by user.

    We still have to use threshold_muscle here, even if we do not want to use it:
    annotate_muscle_zscore() requires threshold_muscle so define a minimal one here: 5 z-score.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    m_or_g_decided : List
        The channel types chosen for the analysis: 'mag' or 'grad'.
    muscle_params : dict
        The parameters for muscle artifact detection originally defined in the config file.
    threshold_muscle : float
        The z-score threshold for muscle detection.
    muscle_freqs : List
        The frequencies of muscle artifacts, usually 110 and 140 Hz.
    cut_dummy : bool
        Whether to cut the dummy data after filtering. Dummy data is attached to the start and end of the recording to avoid filtering artifacts. Default is True.
    attach_sec : int
        The number of seconds to attach to the start and end of the recording.
    min_distance_between_different_muscle_events : int
        The minimum distance between different muscle events in seconds.
    muscle_str_joined : str
        Notes about muscle detection to use as description.
    
    Returns
    -------
    simple_metric : dict
        A simple metric dict for muscle events.
    scores_muscle : np.ndarray
        The muscle scores.
    df_deriv : List
        A list of QC_derivative objects for muscle events containing figures.
    
    """

    for m_or_g in m_or_g_decided: #generally no need for loop, we will use just 1 type here. Left in case we change the principle.

        z_scores_dict={}

        z_score_details={}

        annot_muscle, scores_muscle = annotate_muscle_zscore(
        raw_muscle_path,dataset_path, ch_type=m_or_g, threshold=threshold_muscle, min_length_good=muscle_params['min_length_good'],
        filter_freq=muscle_freqs,orig_meg_type=orig_meg_type)
        gc.collect()

        # Load raw muscle stage signal
        raw, shielding_str, meg_system = load_data(raw_muscle_path)
        # raw.load_data()

        #cut attached beginning and end from annot_muscle, scores_muscle:
        if cut_dummy is True:
            # annot_muscle = annot_muscle[annot_muscle['onset']>attach_sec]
            # annot_muscle['onset'] = annot_muscle['onset']-attach_sec
            # annot_muscle['duration'] = annot_muscle['duration']-attach_sec
            scores_muscle = scores_muscle[int(attach_sec*raw.info['sfreq']): int(-attach_sec*raw.info['sfreq'])]
            raw = raw.crop(tmin=attach_sec, tmax=raw.times[int(-attach_sec*raw.info['sfreq'])])

        # Plot muscle z-scores across recording
        peak_locs_pos, _ = find_peaks(scores_muscle, height=threshold_muscle, distance=raw.info['sfreq']*min_distance_between_different_muscle_events)

        muscle_times = raw.times[peak_locs_pos]
        high_scores_muscle=scores_muscle[peak_locs_pos]

        df_deriv = save_muscle_to_csv('Muscle', raw, scores_muscle, muscle_times, high_scores_muscle, m_or_g_decided[0])

        # Clean raw
        del raw
        gc.collect()

        # collect all details for simple metric:
        z_score_details['muscle_event_times'] = muscle_times.tolist()
        z_score_details['muscle_event_zscore'] = high_scores_muscle.tolist()
        z_scores_dict = {
            'number_muscle_events': len(muscle_times), 
            'Details': z_score_details}
            
        simple_metric = make_simple_metric_muscle(m_or_g_decided[0], z_scores_dict, muscle_str_joined)

    return simple_metric, scores_muscle, df_deriv


def save_muscle_to_csv(file_name_prefix: str, raw: mne.io.Raw, scores_muscle: np.ndarray, high_scores_muscle_times: np.ndarray, high_scores_muscle: np.ndarray, m_or_g: str):

    """
    Save muscle artifacts to a CSV file.

    Parameters
    ----------
    file_name_prefix : str
        The prefix for the file name. Example: 'Muscle'.
    raw : mne.io.Raw
        The raw data.
    scores_muscle : np.ndarray
        The muscle scores.
    high_scores_muscle_times : np.ndarray
        The times of the high muscle scores.
    high_scores_muscle : np.ndarray
        The high muscle scores.
    m_or_g : str
        The channel type chosen for the analysis: 'mag' or 'grad'.
    
    Returns
    -------
    df_deriv : List
        A list of QC_derivative objects for muscle events containing figures.
    
    """

    data_times = raw.times
    m_or_g_combined = ' '.join(m_or_g)
    data = [data_times, scores_muscle, high_scores_muscle_times, high_scores_muscle, [m_or_g]]

    ind = ['data_times', 'scores_muscle', 'high_scores_muscle_times', 'high_scores_muscle', 'ch_type']

    df = pd.DataFrame(data=data, index=ind, columns=[c for c in range(len(data_times))])
    df=df.transpose()

    df_deriv = [QC_derivative(content = df, name = file_name_prefix, content_type = 'df')]

    return df_deriv


def MUSCLE_meg_qc(muscle_params: dict, psd_params: dict, psd_params_internal: dict, channels: dict, data_path: str, noisy_freqs_global: dict, m_or_g_chosen:list, dataset_path: str, attach_dummy:bool = True, cut_dummy:bool = True, orig_meg_type=None):

    """
    Detect muscle artifacts in MEG data. 
    Gives the number of muscle artifacts based on the set z score threshold: artifact time + artifact z score.
    Threshold  is set by the user in the config file. Several thresholds can be used on the loop.

    Notes
    -----
    The data has to first be notch filtered at powerline frequencies as suggested by mne.


    Parameters
    ----------
    
    muscle_params : dict
        The parameters for muscle artifact detection originally defined in the config file.
    psd_params : dict
        The parameters for PSD calculation originally defined in the config file. This in only needed to calculate powerline noise in case PSD was not calculated before.
    psd_params_internal : dict
        The parameters for PSD calculation originally defined in the internal config file. 
    channels : dict
        Dictionary with channels names separated by mag/grad
    raw_orig : mne.io.Raw
        The raw data.
    noisy_freqs_global : List
        The powerline frequencies found in the data by previously running PSD_meg_qc.
    m_or_g_chosen : List
        The channel types chosen for the analysis: 'mag' or 'grad'.
    attach_dummy : bool
        Whether to attach dummy data to the start and end of the recording to avoid filtering artifacts. Default is True.
    cut_dummy : bool
        Whether to cut the dummy data after filtering. Default is True.

    Returns
    -------
    muscle_derivs : List
        A list of QC_derivative objects for muscle events containing figures.
    simple_metric : dict
        A simple metric dict for muscle events.
    muscle_str_joined : str
        String with notes about muscle artifacts for report
    scores_muscle : np.ndarray
        The muscle scores.
    raw : mne.io.Raw
        The raw data with filtered noise or not filtered if no noise was found.

    """

    # Load data
    raw_orig, shielding_str, meg_system = load_data(data_path)


    if noisy_freqs_global is None: # if PSD was not calculated before, calculate noise frequencies now:
        noisy_freqs_global = find_powerline_noise_short(raw_orig, psd_params, psd_params_internal, m_or_g_chosen, channels)
        print('___MEGqc___: ', 'Noisy frequencies found in data at (HZ): ', noisy_freqs_global)
    else: # if PSD was calculated before, use the frequencies from the PSD step:
        pass


    muscle_freqs = muscle_params['muscle_freqs']
   
    raw = raw_orig # make a copy of the raw data, to make sure the original data is not changed while filtering for this metric.

    if muscle_freqs[1] > raw.info['sfreq']/2-5:
        muscle_freqs[1] = raw.info['sfreq'] / 2 - 5

    if 'mag' in m_or_g_chosen:
        m_or_g_decided=['mag']
        muscle_str = 'For this data file artifact detection was performed on magnetometers, they are more sensitive to muscle activity than gradiometers. '
        print('___MEGqc___: ', muscle_str)
    elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
        m_or_g_decided=['grad']
        muscle_str = 'For this data file artifact detection was performed on gradiometers, they are less sensitive to muscle activity than magnetometers. '
        print('___MEGqc___: ', muscle_str)
    else:
        print('___MEGqc___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')
        return [], []
    
    muscle_note = "This metric shows high frequency artifacts in range between 110-140 Hz. High power in this frequency band compared to the rest of the signal is strongly correlated with muscles artifacts, as suggested by MNE. However, high frequency oscillations may also occure in this range for reasons other than muscle activity (for example, in an empty room recording). "
    muscle_str_joined=muscle_note+"<p>"+muscle_str+"</p>"

    attach_sec = 3 # seconds

    if attach_dummy is True:
        raw = attach_dummy_data(raw, attach_sec) #attach dummy data to avoid filtering artifacts at the beginning and end of the recording.  

    raw.load_data() #need to preload data for filtering both in notch filter and in annotate_muscle_zscore

    # Filter out power line noise and other noisy freqs in range of muscle artifacts before muscle artifact detection.
    raw = filter_noise_before_muscle_detection(raw, noisy_freqs_global, muscle_freqs)

    # Save filtered
    raw_muscle_path = save_meg_with_suffix(data_path, dataset_path, raw, final_suffix="MUSCLE_FILTERED")
    # Clean filtered
    del raw
    del raw_orig
    gc.collect()

    # Loop through different thresholds for muscle artifact detection:
    threshold_muscle_list = muscle_params['threshold_muscle']  # z-score
    min_distance_between_different_muscle_events = muscle_params['min_distance_between_different_muscle_events']  # seconds

    time.sleep(3)
    simple_metric, scores_muscle, df_deriv = calculate_muscle_NO_threshold(raw_muscle_path, m_or_g_decided, muscle_params, threshold_muscle_list[0], muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, muscle_str_joined,dataset_path, orig_meg_type)

    os.remove(raw_muscle_path)
    return df_deriv, simple_metric, muscle_str_joined, scores_muscle