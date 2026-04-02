import numpy as np
import pandas as pd
import mne
import copy
from typing import List
from meg_qc.plotting.universal_plots import QC_derivative, assign_epoched_std_ptp_to_channels
from meg_qc.plotting.universal_html_report import simple_metric_basic
from meg_qc.calculation.initial_meg_qc import (chs_dict_to_csv,load_data)


def get_std_all_data(data: mne.io.Raw, channels: List):

    """
    Calculate std for each channel - for the entire time duration.

    Parameters
    ----------
    data : mne.io.Raw
        raw data 
    channels : List 
        list of channel names

    Returns
    -------
    std_channels_named : dict
        dictionary with channel names and their std values
    
    """

    data_channels = data.get_data(picks=channels)
    std_channels = np.std(data_channels, axis=1)

    # Add channel name for every std value:
    std_channels_named = {ch: std for ch, std in zip(channels, std_channels)}

    return std_channels_named


def get_big_small_std_ptp_all_data(ptp_or_std_channels_named: dict, channels: List, std_multiplier: float):

    """
    Function calculates peak-to-peak amplitude or STDs over the entire data set for each channel.
    Threshold for noisy = mean + multiplier*std, above it - noisy,
    Threshold for flat = mean - multiplier*std, below it - flat,
    where:

    - mean is mean stds/ptp values over all channels
    - std is standard deviation of stds values over all channels
    - multiplier is a parameter set in config, defines how many stds/ptp above or below mean should be taken as threshold.


    Parameters
    ----------
    ptp_or_std_channels_named : dict
        peak-to-peak amplitude or std for each channel
    channels : List
        list of channel names
    std_multiplier : float
        multipliar for std, used to define thresholds for noisy and flat channels

    Returns
    -------
    noisy_channels : dict
        dictionary with channel names and their stds/ptp values. Noisy channels.
    flat_channels : dict
        dictionary with channel names and their stds/ptp values. Flat channels.

    """

    # Put all values in 1 array from the dictionsry:
    ptp_or_std_channels = np.array(list(ptp_or_std_channels_named.values()))

    ## Check if channel data is within std level of PtP/std.
    std_of_measure_channels=np.std(ptp_or_std_channels)
    mean_of_measure_channels=np.mean(ptp_or_std_channels)

    print('___MEGqc___: ', mean_of_measure_channels + std_multiplier*std_of_measure_channels, ' threshold for NOISY. ')
    print('___MEGqc___: ', mean_of_measure_channels - std_multiplier*std_of_measure_channels, ' threshold for FLAT. ')

    # Find the index of channels with biggest and smallest std:
    ch_ind_big_measure = [index for (index, item) in enumerate(ptp_or_std_channels) if item > mean_of_measure_channels + std_multiplier*std_of_measure_channels] #find index with bigst std
    ch_ind_small_measure = [index for (index, item) in enumerate(ptp_or_std_channels) if item < mean_of_measure_channels - std_multiplier*std_of_measure_channels] #find index with smallest std

    #make dictionaries with channel names and their std values:
    noisy_channels = {}
    flat_channels = {}

    for index in ch_ind_big_measure:
        ch_name = np.array(channels)[index]
        noisy_channels[ch_name] = ptp_or_std_channels[index]

    for index in ch_ind_small_measure:
        ch_name = np.array(channels)[index]
        flat_channels[ch_name] = ptp_or_std_channels[index]

    return noisy_channels, flat_channels

#%%
def get_std_epochs(channels: List, epochs_mg: mne.Epochs):

    """ 
    Calculate std for multiple epochs for a list of channels.
    Used as internal function in std_meg_epoch()

    Parameters
    ----------
    channels : List
        list of channel names
    epochs_mg : mne.Epochs
        epochs data as mne.Epochs object

    Returns
    -------
    pd.DataFrame
        dataframe with std values for each channel and each epoch
    """

    # Get the data of all epochs and channels at once
    data_epochs = epochs_mg.get_data(picks=channels)  # shape: (n_epochs, n_channels, n_times)

    # Compute the standard deviation along the time axis (axis=2) for each channel in every epoch
    std_array = np.std(data_epochs, axis=2)  # shape: (n_epochs, n_channels)

    # Transpose so that rows represent channels and columns represent epochs, matching the original version
    return pd.DataFrame(std_array.T, index=channels)



def get_noisy_flat_std_ptp_epochs(df_std: pd.DataFrame, ch_type: str, std_or_ptp: str, noisy_channel_multiplier: float, flat_multiplier: float, percent_noisy_flat_allowed: float):
    
    """
    1. Define if the channels data inside the epoch is noisy or flat:
    Compare the std of this channel for this epoch (df_std) to the mean STD of this particular channel or over all epchs.

    - If std of this channel for this epoch is over the mean std of this channel for all epochs together * flat multiplyer, then  the data for this channel in this epoch is noisy.
    - If std of this channel for this epoch is under the mean std of this channel for all epochs together * noisy multiplyer, then this the data for this channel in this epoch is flat.
    
    Multiplyer is set by user in the config file.

    2. Count how many channels are noisy/flat in each epoch. 
    If more than percent_noisy_flat_allowed of channels are noisy/flat, then this epoch is noisy/flat.
    Percent is set by user in the config file.

    3. Create MEG_QC_derivative as 3 dfs:

    - df_epoch_vs_mean: ratio of std of this channel for this epoch to the mean std of this channel over all epochs together
    - df_noisy_epoch: df with True/False values for each channel in each epoch, True if this channel is noisy in this epoch
    - df_flat_epoch: df with True/False values for each channel in each epoch, True if this channel is flat in this epoch

    Parameters
    ----------
    df_std : pd.DataFrame
        dataframe with std/ptp values for each channel and each epoch
    ch_type : str
        channel type, 'mag', 'grad'
    std_or_ptp : str
        'std' or 'ptp' - to use std or peak to peak amplitude as a metric
    noisy_channel_multiplier : float
        multiplier to define noisy channel, if std of this channel for this epoch is over (the mean std of this channel for all epochs together*multipliar), then this channel is noisy
        set by user in the config file
    flat_multiplier : float
        multiplier to define flat channel, if std of this channel for this epoch is under (the mean std of this channel for all epochs together*multipliar), then this channel is flat
        set by user in the config file
    percent_noisy_flat_allowed : float
        percent of noisy/flat channels allowed in each epoch, if more than this percent, then this epoch is noisy/flat. Example: 70
        Means that if more than 70% of channels are noisy/flat in this epoch, then this epoch is noisy/flat.

    Returns
    -------
    list
        list of 3 MEG_QC_derivative objects:
        - df_epoch_vs_mean: ratio of std of this channel for this epoch to the mean std of this channel over all epochs together
        - df_noisy_epoch: df with True/False values for each channel in each epoch, True if this channel is noisy in this epoch
        - df_flat_epoch: df with True/False values for each channel in each epoch, True if this channel is flat in this epoch
    
    """

    epochs = [int(ep) for ep in df_std.columns.tolist()] #get epoch numbers

    # Save true channel count BEFORE adding summary rows so the denominator
    # in percentage / threshold calculations is always correct.
    n_channels = len(df_std.index)

    # Make a separate DataFrame and calculate the mean of stds for each channel over all epochs together
    df_std_with_mean = df_std.copy()
    df_std_with_mean['mean'] = df_std_with_mean.mean(axis=1)
    # Avoid division by zero when the channel is constant by treating the mean
    # of zero as missing data. Subsequent comparisons will skip these entries.
    mean_series = df_std_with_mean['mean'].replace(0, np.nan)

    #TODO: check mwhy all mean tuen into 1 in tsv???


    # Compare mean std of each channel to std of this channel for every epoch and convert to boolean type
    df_noisy_epoch = df_std_with_mean.copy().astype(bool)
    df_flat_epoch = df_std_with_mean.copy().astype(bool)
    df_epoch_vs_mean = df_std_with_mean.copy()

    # Now see which channles in epoch are over std_level or under -std_level:
    
    #append raws to df_noisy_epoch to hold the % of noisy/flat channels in each epoch:
    # NOTE: use the SAME label string here as is written in the loop and read in
    # make_dict_local_std_ptp, otherwise pandas creates a duplicate "ghost" row
    # which inflates len(df) and corrupts iloc[:-3] slicing.
    df_noisy_epoch.loc['number noisy channels'] = np.nan
    df_noisy_epoch.loc['% noisy channels'] = np.nan
    df_noisy_epoch.loc['noisy < %s perc' % percent_noisy_flat_allowed] = np.nan

    df_flat_epoch.loc['number flat channels'] = np.nan
    df_flat_epoch.loc['% flat channels'] = np.nan
    df_flat_epoch.loc['flat < %s perc' % percent_noisy_flat_allowed] = np.nan


    for ep in epochs:  

        ratio = df_std_with_mean.iloc[:, ep] / mean_series
        # ratio is NaN for channels with zero mean; these are marked not noisy/flat
        # by replacing NaN with False in the boolean comparison results
        df_epoch_vs_mean.iloc[:, ep] = ratio
        # Align to original channel index to avoid broadcasting issues
        noisy_mask = (ratio > noisy_channel_multiplier).fillna(False)
        flat_mask = (ratio < flat_multiplier).fillna(False)
        df_noisy_epoch.loc[df_std.index, ep] = noisy_mask.values
        df_flat_epoch.loc[df_std.index, ep] = flat_mask.values

        # Calculate the number of noisy/flat channels in this epoch:
        df_noisy_epoch.loc['number noisy channels', ep] = df_noisy_epoch.iloc[:-3,ep].sum()
        df_flat_epoch.loc['number flat channels', ep] = df_flat_epoch.iloc[:-3,ep].sum()

        # Calculate percent of noisy/flat channels in this epoch.
        # Use n_channels (saved before summary rows were added) as the denominator
        # so that the 3 summary rows are never counted in the percentage.
        df_noisy_epoch.loc['% noisy channels', ep] = round(df_noisy_epoch.iloc[:-3,ep].sum() / n_channels * 100, 1)
        df_flat_epoch.loc['% flat channels', ep] = round(df_flat_epoch.iloc[:-3,ep].sum() / n_channels * 100, 1)

        # Now check if the epoch has over percent_noisy_flat_allowed% of noisy/flat channels:
        df_noisy_epoch.loc['noisy < %s perc' % percent_noisy_flat_allowed, ep] = df_noisy_epoch.iloc[:-3,ep].sum() > n_channels * percent_noisy_flat_allowed / 100
        df_flat_epoch.loc['flat < %s perc' % percent_noisy_flat_allowed, ep] = df_flat_epoch.iloc[:-3,ep].sum() > n_channels * percent_noisy_flat_allowed / 100


    # Create derivatives:
    noisy_flat_epochs_derivs = [
        QC_derivative(df_epoch_vs_mean, std_or_ptp+'_per_epoch_vs_mean_ratio_'+ch_type, 'df'),
        QC_derivative(df_noisy_epoch, 'Noisy_epochs_on_'+std_or_ptp+'_base_'+ch_type, 'df'),
        QC_derivative(df_flat_epoch, 'Flat_epochs_on_'+std_or_ptp+'_base_'+ch_type, 'df')]

    return noisy_flat_epochs_derivs

#%% All about simple metric jsons:

def make_dict_global_std_ptp(std_ptp_params: dict, big_std_with_value_all_data: List[dict], small_std_with_value_all_data: List[dict], channels: List[str], std_or_ptp: str):


    """Make a dictionary with global metric content for std or ptp metric.
    Global means that it is calculated over entire data series, not over epochs.
    
    Parameters
    ----------
    std_ptp_params : dict
        dictionary with parameters for std or ptp metric
    big_std_with_value_all_data : List
        list of dictionaries (channel_name: value) for channels with big std or ptp
    small_std_with_value_all_data : List
        list of dictionaries (channel_name: value) for channels with small std or ptp
    channels : List
        list of channel names
    std_or_ptp : str
        'std' or 'ptp': use STD or Peak-to-peak metric

    Returns
    -------
    metric_global_content : dict
        dictionary with global metric content for std or ptp metric
    """

    global_details = {
        'noisy_ch': big_std_with_value_all_data,
        'flat_ch': small_std_with_value_all_data}

    metric_global_content = {
        'number_of_noisy_ch': len(big_std_with_value_all_data),
        'percent_of_noisy_ch': round(len(big_std_with_value_all_data)/len(channels)*100, 1), 
        'number_of_flat_ch': len(small_std_with_value_all_data),
        'percent_of_flat_ch': round(len(small_std_with_value_all_data)/len(channels)*100, 1), 
        std_or_ptp+'_lvl': std_ptp_params['std_lvl'],
        'details': global_details}

    return metric_global_content


def make_dict_local_std_ptp(std_ptp_params: dict, noisy_epochs_df: pd.DataFrame, flat_epochs_df: pd.DataFrame, percent_noisy_flat_allowed: float=70, onset_times: list=None):

    """
    Make a dictionary with local metric content for std or ptp metric.
    Local means that it is calculated over epochs.

    Parameters
    ----------
    std_ptp_params : dict
        dictionary with parameters for std or ptp metric, originally from config file
    noisy_epochs_df : pd.DataFrame
        dataframe with True/False values for noisy channels in each epoch
    flat_epochs_df : pd.DataFrame
        dataframe with True/False values for flat channels in each epoch
    percent_noisy_flat_allowed : float
        percent of noisy/flat channels allowed in each epoch, if more than this percent, then this epoch is noisy/flat. Example: 70
    
    Returns
    -------
    metric_local_content : dict
        dictionary with local metric content for std or ptp metric

    """
        
    epochs = noisy_epochs_df.columns.tolist()
    epochs = [int(ep) for ep in epochs[:-1]]

    epochs_details = []
    for i, ep in enumerate(epochs):
        onset_s = round(float(onset_times[i]), 4) if (onset_times is not None and i < len(onset_times)) else None
        epochs_details += [{'epoch': ep, 'onset_time_s': onset_s, 'number_of_noisy_ch': int(noisy_epochs_df.loc['number noisy channels',ep]), 'perc_of_noisy_ch': float(noisy_epochs_df.loc['% noisy channels',ep]), 'epoch_too_noisy': noisy_epochs_df.loc['noisy < %s perc' % percent_noisy_flat_allowed, ep], 'number_of_flat_ch': int(flat_epochs_df.loc['number flat channels', ep]), 'perc_of_flat_ch': float(flat_epochs_df.loc['% flat channels', ep]), 'epoch_too_flat': flat_epochs_df.loc['flat < %s perc' % percent_noisy_flat_allowed,ep]}]

    total_num_noisy_ep=len([ep for ep in epochs_details if ep['epoch_too_noisy'] is True])
    total_perc_noisy_ep=round(total_num_noisy_ep/len(epochs)*100)

    total_num_flat_ep=len([ep for ep in epochs_details if ep['epoch_too_flat'] is True])
    total_perc_flat_ep=round(total_num_flat_ep/len(epochs)*100)

    metric_local_content={
        'allow_percent_noisy_flat_epochs': std_ptp_params['allow_percent_noisy_flat_epochs'],
        'noisy_channel_multiplier': std_ptp_params['noisy_channel_multiplier'],
        'flat_multiplier': std_ptp_params['flat_multiplier'],
        'total_num_noisy_ep': total_num_noisy_ep, 
        'total_perc_noisy_ep': total_perc_noisy_ep, 
        'total_num_flat_ep': total_num_flat_ep,
        'total_perc_flat_ep': total_perc_flat_ep,
        'details': epochs_details}

    return metric_local_content



def make_simple_metric_std(std_params:  dict, big_std_with_value_all_data: List[dict], small_std_with_value_all_data: List[dict], channels: List[str], deriv_epoch_std: dict, metric_local_present: bool, m_or_g_chosen: List[dict], onset_times: list=None):

    """
    Make simple metric for STD.

    Parameters
    ----------
    std_params : dict
        dictionary with parameters for std metric, originally from config file
    big_std_with_value_all_data : List
        list of dictionaries (channel_name: value) for channels with big std
    small_std_with_value_all_data : List
        list of dictionaries (channel_name: value) for channels with small std
    channels : List
        list of channel names
    deriv_epoch_std : dict
        dictionary with QC_derivative objects containing data frames. 
        Used only data frame 1 and 2. 
        1: contains True/False values for noisy channels in each epoch. 
        2: contains True/False values for flat channels in each epoch.
    metric_local_present : bool
        True if local metric was calculated (epochs present). False if not calculated (epochs were not detected).
    m_or_g_chosen : List
        list of strings with channel types chosen by user: ['mag', 'grad'] or ['mag'] or ['grad']

    Returns
    -------
    dict
        dictionary with simple metric for std/ptp

    """

    metric_global_name = 'STD_all_time_series'
    metric_global_description = 'Standard deviation of the data over the entire time series (not epoched): the number of noisy channels depends on the std of the data over all channels. The std level is set by the user. Noisy channel: The channel where std of data is higher than threshod: mean_over_all_stds_channel + (std_of_all_channels*std_lvl). Flat: where std of data is lower than threshld: mean_over_all_stds_channel - (std_of_all_channels*std_lvl). In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'STD_epoch'
    if metric_local_present==True:
        metric_local_description = 'Standard deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take std of data of THIS channel in THIS epoch. 2) Take std of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_channel_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat.'
    else:
        metric_local_description = 'Not calculated. No epochs found'

    metric_global_content={'mag': None, 'grad': None, 'eeg': None}
    metric_local_content={'mag': None, 'grad': None, 'eeg': None}
    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_std_ptp(std_params, big_std_with_value_all_data[m_or_g], small_std_with_value_all_data[m_or_g], channels[m_or_g], 'std')
        
        if metric_local_present is True:
            metric_local_content[m_or_g]=make_dict_local_std_ptp(std_params, deriv_epoch_std[m_or_g][1].content, deriv_epoch_std[m_or_g][2].content, percent_noisy_flat_allowed=std_params['allow_percent_noisy_flat_epochs'], onset_times=onset_times)
            #deriv_epoch_std[m_or_g][1].content is df with big std(noisy), df_epoch_std[m_or_g][2].content is df with small std(flat)
        else:
            metric_local_content[m_or_g]=None
    
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'], metric_global_content_eeg=metric_global_content.get('eeg'), metric_local_content_eeg=metric_local_content.get('eeg'))

    return simple_metric

#%%
def STD_meg_qc(std_params: dict, channels: dict, chs_by_lobe: dict, dict_epochs_mg: dict, data_path:str, m_or_g_chosen: List):

    """
    Main STD function. Calculates:
    - Std of data for each channel over all time series.
    - Channels with big std (noisy) and small std (flat) over all time series.
    - Std of data for each channel  in each epoch.
    - Epochs with big std (noisy) and small std (flat).

    Parameters
    ----------
    std_params : dict
        dictionary with parameters for std metric, originally from config file
    channels : dict
        dictionary with channel names for each channel type: channels['mag'] or channels['grad']
    chs_by_lobe : dict
        dictionary with channels grouped first by ch type and then by lobe: chs_by_lobe['mag']['Left Occipital'] or chs_by_lobe['grad']['Left Occipital']
    dict_epochs_mg : dict
        dictionary with epochs for each channel type: dict_epochs_mg['mag'] or dict_epochs_mg['grad']
    data : mne.io.Raw
        raw data
    m_or_g_chosen : List
        list of strings with channel types chosen by user: ['mag', 'grad'] or ['mag'] or ['grad']

    Returns
    -------
    derivs_std : List
        list of QC_derivative objects containing data frames and figures for std metric.
    simple_metric_std : dict
        dictionary with simple metric for std/ptp.
    std_str : str
        String with notes about STD for report
    
    """
    # Load data
    data, shielding_str, meg_system, _modality = load_data(data_path)

    # data.load_data()

    big_std_with_value_all_data = {}
    small_std_with_value_all_data = {}
    std_all_data = {}
    derivs_std = []
    derivs_list = []
    noisy_flat_epochs_derivs={}

    chs_by_lobe_std=copy.deepcopy(chs_by_lobe)
    # copy here, because want to keep original dict unchanged. 
    # In principal it s good to collect all data about channel metrics there BUT if the metrics are run in parallel this might produce conflicts 
    # (without copying  dict can be chanaged both inside+outside this function even when it is not returned.)

    for m_or_g in m_or_g_chosen:

        std_all_data[m_or_g] = get_std_all_data(data, channels[m_or_g])

        #Add std data into channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe_std[m_or_g]:
            for ch in chs_by_lobe_std[m_or_g][lobe]:
                ch.std_overall = std_all_data[m_or_g][ch.name]
                #print(ch.__dict__) #will print all the info saved in the object, more than just simply printing the object

        big_std_with_value_all_data[m_or_g], small_std_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(std_all_data[m_or_g], channels[m_or_g], std_params['std_lvl'])

    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None or dict_epochs_mg.get('eeg') is not None: #If epochs are present
        for m_or_g in m_or_g_chosen:
            df_std = get_std_epochs(channels[m_or_g], dict_epochs_mg[m_or_g])

            chs_by_lobe_std[m_or_g] = assign_epoched_std_ptp_to_channels(what_data='stds', chs_by_lobe=chs_by_lobe_std[m_or_g], df_std_ptp=df_std) #for easier plotting

            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_std_ptp_epochs(df_std, m_or_g, 'std', std_params['noisy_channel_multiplier'], std_params['flat_multiplier'], std_params['allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

        metric_local=True
        std_str = ''
    else:
        metric_local=False
        std_str = 'STD per epoch can not be calculated because no events are present. Check stimulus channel.'
        print('___MEGqc___: ', std_str)


    simple_metric_std = make_simple_metric_std(std_params, big_std_with_value_all_data, small_std_with_value_all_data, channels, noisy_flat_epochs_derivs, metric_local, m_or_g_chosen, onset_times=dict_epochs_mg.get('epoch_onset_times_s'))

    #Extract chs_by_lobe into a data frame
    df_deriv = chs_dict_to_csv(chs_by_lobe_std,  file_name_prefix = 'STDs')

    derivs_std += derivs_list + df_deriv

    #each deriv saved into a separate list and only at the end put together because this way they keep the right order: 
    #first everything about mags, then everything about grads. - in this ordr they ll be added to repot. 
    # TODO: Can use fig_order parameter of QC_derivative to adjust figure order in the report, if got nothing better to do XD.


    return derivs_std, simple_metric_std, std_str
