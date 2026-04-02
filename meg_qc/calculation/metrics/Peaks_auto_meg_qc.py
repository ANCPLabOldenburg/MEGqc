import mne
from typing import List
from meg_qc.plotting.universal_plots import QC_derivative

# This module is not used in the final version of the pipeline. 
# We use the manual one. But this one is left in case we still want to bring it back.

def get_amplitude_annots_per_channel(raw: mne.io.Raw, peak: float, flat: float, channels: List, bad_percent:  int, min_duration: float):
    
    """
    Create peak-to-peak amplitude annotations for every channel separately
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    peak : float
        Peak value.
    flat : float
        Flat value.
    channels : List
        list of channel names.
    bad_percent : int
        Percent of bad data allowed to still cound channels as good.
    min_duration : float
        Minimum duration of bad data to be considered as bad? (check this)
    
    Returns
    -------
    df_ptp_amlitude_annot : pd.DataFrame
        Dataframe with peak-to-peak amplitude annotations.
    bad_channels : List
        list of bad channels.
    """
    
    amplit_annot_with_ch_names=mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time) #initialize 
    bad_channels=[]

    for channel in channels:
        #get annotation object:
        amplit_annot=mne.preprocessing.annotate_amplitude(raw, peak=peak, flat=flat , bad_percent=bad_percent, min_duration=min_duration, picks=[channel], verbose=False)
        bad_channels.append(amplit_annot[1]) #Can later add these into annotation as well.

        if len(amplit_annot[0])>0:
            #create new annot obj and add there all data + channel name:
            amplit_annot_with_ch_names.append(onset=amplit_annot[0][0]['onset'], duration=amplit_annot[0][0]['duration'], description=amplit_annot[0][0]['description'], ch_names=[[channel]])

    df_ptp_amlitude_annot=amplit_annot_with_ch_names.to_data_frame()
    
    return df_ptp_amlitude_annot, bad_channels


def PP_auto_meg_qc(ptp_auto_params: dict, channels:list, data: mne.io.Raw, m_or_g_chosen: List):
    
    """
    Calculates peak-to-peak amplitude annotations for every channel using MNE built-in approach.
    
    Parameters
    ----------
    ptp_auto_params : dict
        Dictionary with parameters for peak-to-peak amplitude annotations.
    channels : List
        list of channels.
    data : mne.io.Raw
        Raw data.
    m_or_g_chosen : List
        list of channels types.
        
    Returns
    -------
    deriv_ptp_auto : List
        list of QC_derivative objects containing dataframes with peak-to-peak amplitude annotations.
    bad_channels : List
        list of bad channels.
    pp_auto_str : str
        string with notes about PtP auto for report
        
    """

    peaks = {'grad': ptp_auto_params['peak_g'], 'mag': ptp_auto_params['peak_m'],
             'eeg': ptp_auto_params.get('peak_eeg', 200e-6)}
    flats = {'grad': ptp_auto_params['flat_g'], 'mag': ptp_auto_params['flat_m'],
             'eeg': ptp_auto_params.get('flat_eeg', 1e-6)}
    bad_channels = {}

    deriv_ptp_auto= []
    for  m_or_g in m_or_g_chosen:
        dfs_ptp_amlitude_annot, bad_channels[m_or_g] = get_amplitude_annots_per_channel(data, peaks[m_or_g], flats[m_or_g], channels[m_or_g], bad_percent=ptp_auto_params['bad_percent'], min_duration= ptp_auto_params['min_duration'])
        deriv_ptp_auto += [QC_derivative(dfs_ptp_amlitude_annot,'ptp_amplitude_annots_'+m_or_g, 'df')]

    pp_auto_str = 'Peak-to-peak amplitude annotations were calculated automatically using mne function annotate_amplitude. See csv files for details.'
    
    return deriv_ptp_auto, bad_channels, pp_auto_str
