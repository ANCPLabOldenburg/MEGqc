import numpy as np
import pandas as pd
import mne
import plotly.graph_objects as go
from scipy.integrate import simpson
from scipy.signal import find_peaks, peak_widths
import copy
import re   
from typing import List, Union

from meg_qc.plotting.universal_plots import QC_derivative, get_tit_and_unit
from meg_qc.calculation.initial_meg_qc import (chs_dict_to_csv,load_data)
from meg_qc.plotting.universal_html_report import simple_metric_basic


#%%
def add_log_buttons(fig: go.Figure):

    """
    Add buttons to switch scale between log and linear. For some reason only swithcing the Y scale works so far.

    Parameters
    ----------
    fig : go.Figure
        The figure to be modified withot buttons
        
    Returns
    -------
    fig : go.Figure
        The modified figure with the buttons
        
    """

    updatemenus = [
    {
        "buttons": [
            {
                "args": [{"xaxis.type": "linear"}],
                "label": "X linear",
                "method": "relayout"
            },
            {
                "args": [{"xaxis.type": "log"}],
                "label": "X log",
                "method": "relayout"
            }
        ],
        "direction": "right",
        "showactive": True,
        "type": "buttons",
        "x": 0.15,
        "y": -0.1
    },
    {
        "buttons": [
            {
                "args": [{"yaxis.type": "linear"}],
                "label": "Y linear",
                "method": "relayout"
            },
            {
                "args": [{"yaxis.type": "log"}],
                "label": "Y log",
                "method": "relayout"
            }
        ],
        "direction": "right",
        "showactive": True,
        "type": "buttons",
        "x": 1,
        "y": -0.1
    }]

    fig.update_layout(updatemenus=updatemenus)

    return fig


def get_bands_amplitude_per_ch(freq_bands: List, freqs: List, psds: Union[List, np.ndarray], channels: List, bands_names: List = None):

    """
    Calculate the area under the curve of frequency bands per channel.
    Adopted from: https://raphaelvallat.com/bandpower.html
    Take mean over all channels (mags/grads).

    
    Parameters
    ----------
    freq_bands : List
        list of lists of frequencies. Expects list of lists: [[f_low, f_high], [f_low, f_high], ...]
    freqs : List
        list of frequencies.
    psds : np.ndarray or list
        numpy array (or list) of power spectrum dencities. Expects array of arrays: channels*psds. (or list of lists)
        Will not work properly if 1 dimentional array given. In this case do: np.array([your_1d_array])
    channels : List
        list of channel names. Expects list of strings: ['MEG 0111', 'MEG 0112', ...] 
        If only one channel is given, it should be a list of one string: ['Average]
    bands_names : List, optional
        list of names of the bands. If None, the names will be generated automatically, by default None
        Important! these names are in the same order as in freq_bands. 

        
    Returns
    -------
    band_ampl_df : pd.DataFrame
        Dataframe of amplitudes of each frequency band like: [abs_power_of_delta, abs_power_of_gamma, etc...] - in absolute values
    band_ampl_relative_to_signal_df : pd.DataFrame
        Dataframe of amplitudes of each frequency band divided by the total amplitudeof the signal for this channel. 
        Shows how much amplitude this particular band takes in the entire signal.
    ampl_by_Nfreq_per_ch_list_df : pd.DataFrame
        Dataframe of amplitudes of each frequency band divided by the number of frequencies in the band.
        (This is done to compare with RMSE later. But not used any more).
    total_signal_amplitude : List
        list of total signal amplitude for each channel.


    """

    #Check that freq_bands correspond to band names corectly:
    if bands_names is None:
        bands_as_str=[str(band[0])+'-'+str(band[1])+'Hz' for band in freq_bands]
    else:
        bands_as_str = bands_names #assume everything is correct. then check:
        # Iterate over wave_bands and bands_names
        for band, name in zip(freq_bands, bands_names):
            # Extract the frequency range from the name
            freq_range = re.search(r'\((.*?) Hz\)', name).group(1).split('-')

            freq_range = [float(freq) for freq in freq_range]

            # If the frequency range doesn't match the band, correct bands_as_str
            if freq_range != band:
                bands_as_str = [str(band[0])+'-'+str(band[1])+'Hz' for band in freq_bands]
                break


    freq_res = freqs[1] - freqs[0]

    total_signal_amplitude = []

    band_ampl_df = pd.DataFrame(index=channels, columns=bands_as_str)
    band_ampl_relative_to_signal_df = pd.DataFrame(index=channels, columns=bands_as_str)
    ampl_by_Nfreq_per_ch_list_df = pd.DataFrame(index=channels, columns=bands_as_str)

    for ch_n, _ in enumerate(psds):
        total_signal_amplitude.append(simpson(psds[ch_n], dx=freq_res)) #amplitudeof all bands 

        for band_n, band in enumerate(freq_bands):

            idx_band = np.logical_and(freqs >= band[0], freqs <= band[-1]) 
            # idx_band is a list of booleans, where True means that the frequency is in the band and False means it is not.

            # Compute the absolute amplitude of the band by approximating the area under the curve:

            band_ampl = simpson(psds[ch_n][idx_band], dx=freq_res) #amplitude of chosen band

            band_ampl_df.iloc[ch_n, band_n] = band_ampl

            #Calculate how much of the total amplitude of the average signal goes into each of the noise freqs:
            band_ampl_relative_to_signal_df.iloc[ch_n, band_n] = band_ampl / total_signal_amplitude[ch_n] # relative amplitude: % of this band in the total bands amplitude for this channel:

            #devide the amplitude of band by the  number of frequencies in the band, to compare with RMSE later:
            ampl_by_Nfreq_per_ch_list_df.iloc[ch_n, band_n] = band_ampl/sum(idx_band)


    return band_ampl_df, band_ampl_relative_to_signal_df, ampl_by_Nfreq_per_ch_list_df, total_signal_amplitude


    
# In[53]:

def get_ampl_of_brain_waves(channels: List, m_or_g: str, freqs: np.ndarray, psds: np.ndarray, avg_psd: np.ndarray):

    """
    Calculate area under the curve of frequency bands (alpha, beta, gamma, etc) for all channels.
    get_bands_amplitude_per_ch() is used to calculate the amplitude of each frequency band in each channel.

    Parameters
    ----------
    channels : List
        list of channel names
    m_or_g : str
        'mag' or 'grad' - to choose which channels to calculate amplitude for.
    freqs : np.ndarray
        numpy array of frequencies for mag  or grad
    psds : np.ndarray
        numpy array of power spectrum dencities for mag or grad
    avg_psd : np.ndarray
        numpy array of average power spectrum dencities for mag or grad

    Returns
    -------
    waves_pie_df_deriv : QC_derivative object or empty list
        If plotflag is True, returns one QC_derivative object with data for pie plot.
        If plotflag is False, returns empty list.
    dfs_with_name : List
        list of dataframes with amplitude of each frequency band in each channel
        (dfs: absolute amplitude, relative amplitude, amplitude divided by number of frequencies in the band)
    mean_brain_waves_dict : dict
        Dictionary of mean amplitude of each frequency band (used for simple metric json)

    """
    
    # Calculate the band amplitude:
    wave_bands=[[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 100]]
    bands_names = ["delta (0.5-4 Hz)", "theta (4-8 Hz)", "alpha (8-12 Hz)", "beta (12-30 Hz)", "gamma (30-100 Hz)"]

    abs_band_ampl_df, relative_band_ampl_df, ampl_by_Nfreq_per_ch_list_df, _ = get_bands_amplitude_per_ch(wave_bands, freqs, psds, channels, bands_names)

    dfs_with_name = [
        QC_derivative(abs_band_ampl_df,'abs_ampl_'+m_or_g, 'df'),
        QC_derivative(relative_band_ampl_df, 'relative_ampl_'+m_or_g, 'df'),
        QC_derivative(ampl_by_Nfreq_per_ch_list_df, 'ampl_by_Nfreq_'+m_or_g, 'df')]

    # Calculate the mean amplitude of each band over all channels:
    abs_mean_band_ampl_df, relative_mean_band_ampl_df, _, total_ampl = get_bands_amplitude_per_ch(wave_bands, freqs, [avg_psd], ['Average PSD'], bands_names)
    
    # Merge the dataframes and with 'absolute' and 'relative' raw names:
    brain_bands_df = pd.concat([abs_mean_band_ampl_df, relative_mean_band_ampl_df], axis=0)  
    brain_bands_df.index = ['absolute_'+m_or_g, 'relative_'+m_or_g]

    #add total_ampl as a new column in 'Absolute' raw and None in 'Relative' raw:
    brain_bands_df['total_amplitude'] = [total_ampl[0], 1]

    waves_pie_df_deriv = [QC_derivative(content=brain_bands_df, name='PSDwaves'+m_or_g.capitalize(), content_type = 'df')]
    
    #convert results to a list:

    mean_brain_waves_abs=abs_mean_band_ampl_df.iloc[0, :].values.tolist()
    mean_brain_waves_relative=relative_mean_band_ampl_df.iloc[0, :].values.tolist()

    mean_brain_waves_dict= {bands_names[i]: {'mean_brain_waves_relative': np.round(mean_brain_waves_relative[i]*100, 2), 'mean_brain_waves_abs': mean_brain_waves_abs[i]} for i in range(len(bands_names))}

    return waves_pie_df_deriv, dfs_with_name, mean_brain_waves_dict


def split_blended_freqs_at_the_lowest_point(noisy_bands_indexes:List[List], one_psd:List[dict], noisy_freqs_indexes:List[dict]):

    """
    If there are 2 bands that are blended together, split them at the lowest point between 2 central noise frequencies.
    
    Parameters
    ----------
    noisy_bands_indexes : List[list]
        list of lists with indexes of noisy bands. Indexes! not frequency bands themselves. Index is defined by fequency/freq_resolution.
    one_psd : List
        vector if psd values for 1 channel (or 1 average over all channels)
    noisy_freqs_indexes : List
        list of indexes of noisy frequencies. Indexes! not frequencies themselves. Index is defined by fequency/freq_resolution.

    Returns
    -------
    noisy_bands_final_indexes : List[list]
        list of lists with indexes of noisy bands After the split.
        Indexes! not frequency bands themselves. Index is defined by fequency/freq_resolution.
    split_indexes : List
        list of indexes at which the bands were split (used later for plotting only).
    
    """

    noisy_bands_final_indexes = noisy_bands_indexes.copy()
    split_indexes = []

    if len(noisy_bands_indexes)>1: #if there are at least 2 bands
        for i, _ in enumerate(noisy_bands_indexes[:-1]):
            #if bands overlap - SPLIT them:
            if noisy_bands_final_indexes[i+1][0]<=noisy_bands_final_indexes[i][1]: #if the end of the previous band is after the start of the current band
                
                split_ind=np.argmin(one_psd[noisy_freqs_indexes[i]:noisy_freqs_indexes[i+1]])
                split_ind=noisy_freqs_indexes[i]+split_ind
                #here need to sum them, because argmin above found the index counted from the start of the noisy_freqs_indexes[iter-1] band, not from the start of the freqs array
                #print('split at the lowest point between 2 peaks', split_point)

                noisy_bands_final_indexes[i][1]=split_ind #assign end of the previous band
                noisy_bands_final_indexes[i+1][0]=split_ind #assigne beginnning of the current band
                split_indexes.append(int(split_ind))

    #print('split_indexes', split_indexes, 'noisy_bands_final_indexes', noisy_bands_final_indexes)

    return noisy_bands_final_indexes, split_indexes


def cut_the_noise_from_psd(noisy_bands_indexes: List[dict], freqs: List, one_psd: List, helper_plots: bool, ch_name: str ='', noisy_freqs_indexes: List =[], unit: str =''):

    """
    CURRENTLY NOT USED
    Cut the noise peaks out of PSD curve. 
    If turned on, in the next steps, the area under the curve will be calculated only for the cut out peaks.
    
    This was one of the earlier approaches, but we cant just cut the noise peaks out of psd.
    In reality we can not define, which part of the 'noisy' frequency is signal and which is noise. 
    In case later, during preprocessing this noise will be filtered out, it will be done completely: both the peak and the main psd area.

    Process:

    1. Find the height of the noise peaks. For this take the average between the height of the start and end of this noise bend.
    2. Cut the noise peaks out of PSD curve at the found height.
    3. Baseline the peaks: all the peaks are brought to 0 level.

    Function also can prodece helper plot to demonstrate the process.

    Parameters
    ----------
    noisy_bands_indexes : List[list]
        list of lists with indexes of noisy bands. Indexes! Not frequency bands themselves. Index is defined by fequency/freq_resolution.
    freqs : List
        vector of frequencies
    one_psd : List
        vector if psd values for 1 channel (or 1 average over all channels)
    helper_plots : bool
        if True, helper plots will be produced
    ch_name : str, optional
        channel name, by default '', used for plot display
    noisy_freqs_indexes : List, optional
        list of indexes of noisy frequencies. Indexes! not frequencies themselves. Index is defined by fequency/freq_resolution., 
        by default [] because we might have no noisy frequencies at all. Used for plot display.
    unit : str, optional
        unit of the psd values, by default '', used for plot display

    Returns
    -------
    psd_only_peaks_baselined : List
        vector of psd values for 1 channel (or 1 average over all channels) with the noise peaks cut out and baselined to 0 level.
        Later used to calculate area under the curve for the noise peaks only.
    
    """



    #band height will be chosen as average between the height of the limits of this bend.
    peak_heights = []
    for band_indexes in noisy_bands_indexes:
        peak_heights.append(np.mean([one_psd[band_indexes[0]], one_psd[band_indexes[-1]]]))

    psd_only_signal=one_psd.copy()
    psd_only_peaks=one_psd.copy()
    psd_only_peaks[:]=None
    psd_only_peaks_baselined=one_psd.copy()
    psd_only_peaks_baselined[:]=0
    
    for fr_n, fr_b in enumerate(noisy_bands_indexes):
        #turn fr_b into a range from start to end of it:
        fr_b=[i for i in range(fr_b[0], fr_b[1]+1)]
        
        psd_only_signal[fr_b]=None #keep only main psd, remove noise bands, just for visual
        psd_only_peaks[fr_b]=one_psd[fr_b].copy() #keep only noise bands, remove psd, again for visual
        psd_only_peaks_baselined[fr_b]=one_psd[fr_b].copy()-[peak_heights[fr_n]]*len(psd_only_peaks[fr_b])
        #keep only noise bands and baseline them to 0 (remove the signal which is under the noise line)

        # clip the values to 0 if they are negative, they might appear in the beginning of psd curve
        psd_only_peaks_baselined=np.array(psd_only_peaks_baselined) 
        psd_only_peaks_baselined = np.clip(psd_only_peaks_baselined, 0, None) 

    #Plot psd before and after cutting the noise:
    if helper_plots is True:
        fig1 = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig1.update_layout(title=ch_name+' Original noncut PSD')
        fig2 = plot_one_psd(ch_name, freqs, psd_only_peaks, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig2.update_layout(title=ch_name+' PSD with noise peaks only')
        fig3 = plot_one_psd(ch_name, freqs, psd_only_signal, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig3.update_layout(title=ch_name+' PSD with signal only')
        fig4 = plot_one_psd(ch_name, freqs, psd_only_peaks_baselined, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig4.update_layout(title=ch_name+' PSD with noise peaks only, baselined to 0')

        #put all 4 figures in one figure as subplots:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=(ch_name+' Original noncut PSD', ch_name+' PSD with noise peaks only', ch_name+' PSD with signal only', ch_name+' PSD with noise peaks only, baselined to 0'))
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig1.data[1], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)
        fig.add_trace(fig2.data[1], row=1, col=2)
        fig.add_trace(fig3.data[0], row=2, col=1)
        fig.add_trace(fig3.data[1], row=2, col=1)
        fig.add_trace(fig4.data[0], row=2, col=2)
        fig.add_trace(fig4.data[1], row=2, col=2)
        #add rectagles to every subplot:
        for i in range(len(noisy_bands_indexes)):
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=1, col=1)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=1, col=2)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=2, col=1)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=2, col=2)

        fig.update_layout(height=800, width=1300, title_text=ch_name+' PSD before and after cutting the noise')

        fig.show()
        #or show each figure separately:
        # fig1.show()
        # fig2.show()
        # fig3.show()
        # fig4.show()

    return psd_only_peaks_baselined


def plot_one_psd(ch_name: str, freqs: List, one_psd: List, peak_indexes: List, noisy_freq_bands_indexes: List[list], unit: str):
    
    """
    Plot PSD for one channels or for the average over multiple channels with noise peaks and split points using plotly.
    Used in helper plots for debugging.
    
    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'
    freqs : List
        list of frequencies
    one_psd : List
        list of psd values for one channels or for the average over multiple channels
    peak_indexes : List
        list of indexes of the noise peaks in the psd
    noisy_freq_bands_indexes : List[list]
        list of lists of indexes of the noisy frequency bands in the psd. Indexes! Not frequency bands themselves. Index is defined by fequency/freq_resolution.
    unit : str
        unit of the psd values. For example 'T/Hz'

    Returns
    -------
    fig
        plotly figure of the psd with noise peaks and bands around them.

    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=one_psd, name=ch_name+' psd'))
    fig.add_trace(go.Scatter(x=freqs[peak_indexes], y=one_psd[peak_indexes], mode='markers', name='peaks'))
    #plot split points as vertical lines and noise bands as red rectangles:

    noisy_freq_bands = [[freqs[noisy_freq_bands_indexes[i][0]], freqs[noisy_freq_bands_indexes[i][1]]] for i in range(len(noisy_freq_bands_indexes))]

    for fr_b in noisy_freq_bands:
        fig.add_vrect(x0=fr_b[0], x1=fr_b[-1], line_width=1, fillcolor="red", opacity=0.2, layer="below")
    
    fig.update_layout(title=ch_name+' PSD with noise peaks and split edges', xaxis_title='Frequency', yaxis_title='Amplitude ('+unit+')',
        yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'))
    
    #Add buttons to switch scale between log and linear:
    fig = add_log_buttons(fig)
    
    return fig


def find_noisy_freq_bands_complex(ch_name: str, freqs: List, one_psd: List, helper_plots: bool, m_or_g: str, prominence_lvl_pos: int):

    """
    Detect the frequency band around the noise peaks.
    Complex approach: This function is trying to detect the actual start and end of peaks.

    1. Bands around the noise frequencies are created based on detected peak_width.
    2. If the found bands overlap, they are cut at the lowest point between 2 neighbouring noise peaks on PSD curve.

    This function is not used by default, becausesuch a complex approach, even though can accurately find start and end of the noise bands, 
    is not very robust. It can sometimes take too much of the area around the noise peak, leading to a large part of the signel folsely counted as noise.
    By default, the more simple approach is used. See find_noisy_freq_bands_simple() function.

    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'. For plotting purposes only.
    freqs : List
        list of frequencies
    one_psd : List
        list of psd values for one channels or for the average over multiple channels
    helper_plots : bool
        if True, helper plots will be shown
    m_or_g : str
        'mag' or 'grad' - for plotting purposes only - to get the unit of the psd values
    prominence_lvl_pos : int
        prominence level for peak detection. The higher the value, the more peaks will be detected. 


    Returns
    -------
    noisy_freqs : List
        list of noisy frequencies
    noisy_freqs_indexes : List
        list of indexes of noisy frequencies in the psd
    noisy_bands_final : List[list]
        list of lists of noisy frequency bands. Each list contains 2 values: start and end of the band.
    noisy_bands_final_indexes : List[list]
        list of lists of indexes of noisy frequency bands. Each list contains 2 values: start and end of the band.
    split_indexes : List
        list of indexes of the split points in the psd
    
    """

    _, unit = get_tit_and_unit(m_or_g, True)
    # Run peak detection on psd -> get number of noise freqs, define freq bands around them
     
    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)

    if noisy_freqs_indexes.size==0: #if no noise found

        if helper_plots is True: #visual
            _, unit = get_tit_and_unit(m_or_g, True)
            fig = plot_one_psd(ch_name, freqs, one_psd, [], [], unit)
            fig.show()

        return [], [], [], [], [], []


    noisy_freqs=freqs[noisy_freqs_indexes]

    # Make frequency bands around noise frequencies on base of the detected width of the peaks:
    _, _, left_ips, right_ips = peak_widths(one_psd, noisy_freqs_indexes, rel_height=1)

    noisy_bands_indexes=[]
    for ip_n, _ in enumerate(noisy_freqs_indexes):
        #+1 here because I  will use these values as range,and range in python is usually "up to the value but not including", this should fix it to the right rang
        noisy_bands_indexes.append([round(left_ips[ip_n]), round(right_ips[ip_n])+1])


    # Split the blended freqs at the lowest point between 2 peaks 
    noisy_bands_final_indexes, split_indexes = split_blended_freqs_at_the_lowest_point(noisy_bands_indexes, one_psd, noisy_freqs_indexes)
    #print(ch_name, 'LOWEST POINT ', 'noisy_bands_final_indexes: ', noisy_bands_final_indexes, 'split_indexes: ', split_indexes)

    if helper_plots is True: #visual of the split
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_final_indexes, unit)
        fig.show()

    #Get actual freq bands from their indexes:
    noisy_bands_final=[]
    for fr_b in noisy_bands_final_indexes:
        noisy_bands_final.append([freqs[fr_b][0], freqs[fr_b][1]])

    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_final_indexes, split_indexes


def find_noisy_freq_bands_simple(ch_name: str, freqs: List, one_psd: List, helper_plots: bool, m_or_g: str, prominence_lvl_pos: int, band_half_length: float):
    
    """
    Form a frequency band around the noise peaks.
    Simple approach: used by default.

    1. Create frequency band around central noise frequency just by adding -x...+x Hz around.
    2. If the found bands overlap, they are cut at the lowest point between 2 neighbouring noise peaks on PSD curve.

    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'. For plotting purposes only.
    freqs : List
        list of frequencies
    one_psd : List
        list of psd values for one channels or for the average over multiple channels
    helper_plots : bool
        if True, helper plots will be shown
    m_or_g : str
        'mag' or 'grad' - for plotting purposes only - to get the unit of the psd values
    prominence_lvl_pos : int
        prominence level for peak detection. The higher the value, the more peaks will be detected. 
    band_half_length : float
        length of the frequency band before and after the noise peak in Hz. The band will be created by adding -band_half_length...+band_half_length Hz around the noise peak.

    Returns
    -------
    noisy_freqs : List
        list of noisy frequencies
    noisy_freqs_indexes : List
        list of indexes of noisy frequencies in the psd
    noisy_bands_final : List[list]
        list of lists of noisy frequency bands. Each list contains 2 values: start and end of the band.
    noisy_bands_final_indexes : List[list]
        list of lists of indexes of noisy frequency bands. Each list contains 2 values: start and end of the band.
    split_indexes : List
        list of indexes of the split points in the psd
    
    """

    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)


    if noisy_freqs_indexes.size==0:

        if helper_plots is True: #visual check
            _, unit = get_tit_and_unit(m_or_g, True)
            fig = plot_one_psd(ch_name, freqs, one_psd, [], [], unit)
            fig.show()

        return [], [], [], [], []

    #make frequency bands around the central noise frequency (for example, -1...+1 Hz band around the peak):
    freq_res = freqs[1] - freqs[0]
    noisy_bands_indexes=[]
    for i, _ in enumerate(noisy_freqs_indexes):
        bend_start_ind = round(noisy_freqs_indexes[i] - band_half_length/freq_res)
        #need to round the indexes. because freq_res has sometimes many digits after coma, 
        # like 0.506686867543 instead of 0.5, so the indexes might become floats if we dont round.
        if bend_start_ind < 0: #index cant be negative
            bend_start_ind = 0
        bend_end_ind = round(noisy_freqs_indexes[i] + band_half_length/freq_res)
        if bend_end_ind > len(freqs)-1: #index cant go over the limit of freq range
            bend_end_ind = len(freqs)-1
        noisy_bands_indexes.append([bend_start_ind, bend_end_ind])
        
    
    # Split the blended freqs if their bands cross:
    noisy_bands_final_indexes, split_indexes = split_blended_freqs_at_the_lowest_point(noisy_bands_indexes, one_psd, noisy_freqs_indexes)
    if helper_plots is True: #visual of the split
        _, unit = get_tit_and_unit(m_or_g, True)
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_final_indexes, unit)
        fig.show()

    noisy_freqs = freqs[noisy_freqs_indexes]

    noisy_bands_final = [[freqs[noisy_bands_final_indexes[i][0]], freqs[noisy_bands_final_indexes[i][1]]] for i in range(len(noisy_bands_final_indexes))]

    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_final_indexes, split_indexes


def find_number_and_ampl_of_noise_freqs(ch_name: str, freqs: List, one_psd: List, helper_plots: bool, m_or_g: str, cut_noise_from_psd: bool, prominence_lvl_pos: int, simple_or_complex: str = 'simple'):

    """
    The function finds the number and amplitude of noisy frequencies in PSD function in these steps:

    Calculate area under the curve for each noisy peak (amplitude of the noise)): 
        - if cut_noise_from_psd is True:: area is limited to where noise band crosses the fitted curve. - count from there.
        - If not (default): area is limited to the whole area under the noise band, including the psd of the signal.

    Calculate what ration of noise and signal.

    
    Parameters
    ----------
    ch_name : str
        name of the channel or 'average'
    freqs : List
        list of frequencies
    one_psd : List
        list of psd values for one channel or average psd
    helper_plots : bool
        if True, plot the helper plots (will show the noise bands, how they are split and how the peaks are cut from the psd if this is activated).
    m_or_g : str
        'mag' or 'grad'
    cut_noise_from_psd : bool
        if True, cut the noise peaks at the point they start and baseline them to 0. Optional. By default not used
    prominence_lvl_pos : int
        prominence level for peak detection (central frequencies of noise bands). The higher the value, the more peaks will be detected. 
        prominence_lvl will be different for average psd and psd of 1 channel, because average has small peaks smoothed.
    simple_or_complex : str
        'simple' or 'complex' approach to create the bands around the noise peaks. Simple by default. See functions above for details.

    Returns
    -------
    noise_pie_derivative : List
        list with QC_derivative object containing the pie chart with the noise amplitude and signal amplitude
    noise_ampl : List
        list of noise amplitudes for each noisy frequency band
    noise_ampl_relative_to_signal : List
        list of noise amplitudes relative to the signal amplitude for each noisy frequency band
    noisy_freqs : List
        list of noisy frequencies
    
    
    """

    _, unit = get_tit_and_unit(m_or_g, True)

    #Total amplitude of the signal together with noise:
    freq_res = freqs[1] - freqs[0]
    total_amplitude = simpson(one_psd, dx=freq_res) 

    if simple_or_complex == 'simple':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_indexes = find_noisy_freq_bands_simple(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos, band_half_length=1)
        # band_half_length is set to 1. Means we go 1Hz left and 1 Hz right from the central freq to create a band.
    elif simple_or_complex == 'complex':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_indexes = find_noisy_freq_bands_complex(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos)
        # complex is currently not used.
    else:
        print('___MEGqc___: ','simple_or_complex should be either "simple" or "complex"')
        return

    #*. Cut the noise peaks at the point they start and baseline them to 0.
    # Fit a curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0. Optional. By default not used
    if cut_noise_from_psd is True:
        psd_noise_final = cut_the_noise_from_psd(noisy_bands_indexes_final, freqs, one_psd, helper_plots, ch_name, noisy_freqs_indexes, unit)
    else:
        psd_noise_final = one_psd


    # Calculate area under the curve for each noisy peak: 
    # if cut the noise -> area is limited to where amplitude crosses the fitted curve. - count from there to the peak amplitude.
    # if dont cut the noise -> area is calculated from 0 to the peak amplitude. (WE USE THIS).
    

    if noisy_bands_final: #if not empty

        noise_ampl_df, noise_ampl_relative_to_signal_df, _, _ = get_bands_amplitude_per_ch(noisy_bands_final, freqs, [psd_noise_final], [ch_name])
        #convert results to a list:

        noise_ampl = noise_ampl_df.iloc[0, :].values.tolist() #take the first and only raw, because there is only one channel calculated by this fucntion
        noise_ampl_relative_to_signal=noise_ampl_relative_to_signal_df.iloc[0, :].values.tolist()
    else:
        noise_ampl = []
        noise_ampl_relative_to_signal = []

    if ch_name.lower() == 'average':

        # Replace empty lists with [0] in case no noise was found:
        noisy_freqs = noisy_freqs if len(noisy_freqs) else [0]
        noise_ampl = noise_ampl if len(noise_ampl) else [0]
        noise_ampl_relative_to_signal = noise_ampl_relative_to_signal if len(noise_ampl_relative_to_signal) else [0]
        # will put 0 in the pie chart if no noise was found (if arrays/lists are empty and have 0 length)

        # Create a DataFrame
        df = pd.DataFrame({
            'noisy_freqs_'+m_or_g: noisy_freqs,
            'noise_ampl_'+m_or_g: noise_ampl,
            'noise_ampl_relative_to_signal_'+m_or_g: noise_ampl_relative_to_signal,
            'total_amplitude_'+m_or_g: [total_amplitude] + [np.nan] * (len(noisy_freqs) - len([total_amplitude]))})  # Add total_amplitude only once, rest fill with none

        noise_pie_df_deriv = QC_derivative(content=df, name='PSDnoise'+m_or_g.capitalize(), content_type = 'df')

    else:
        noise_pie_df_deriv = []

    return noise_pie_df_deriv, noise_ampl, noise_ampl_relative_to_signal, noisy_freqs



def get_ampl_of_noisy_freqs(channels: List, freqs: List, avg_psd: List, psds: List, m_or_g: str, helper_plots: bool = False, cut_noise_from_psd: bool = False, prominence_lvl_pos_avg: int = 50, prominence_lvl_pos_channels: int = 15, simple_or_complex: str = 'simple'):

    """
    Find noisy frequencies, their absolute and relative amplitude for averages over all channel (mag or grad) PSD and for each separate channel.

    Parameters
    ----------
    channels : List
        list of channel names
    freqs : List
        list of frequencies
    avg_psd : List
        list of average PSD values over all channels
    psds : List
        list of PSD values for each channel
    m_or_g : str
        'mag' or 'grad'
    helper_plots : bool
        if True, plot helper plots
    cut_noise_from_psd : bool
        if True, cut the noise peaks at the point they start and baseline them to 0.
    prominence_lvl_pos_avg : int
        prominence level of peak detection for finding noisy frequencies in the average PSD
    prominence_lvl_pos_channels : int
        prominence level of peak detection for finding noisy frequencies in the PSD of each channel
    simple_or_complex : str
        'simple' or 'complex' - method of finding noisy frequencies. see find_number_and_ampl_of_noise_freqs() for details

    Returns
    -------
    noise_pie_derivative : QC_derivative object or empty list 
        QC_derivative containig a pie chart of SNR
    noise_ampl_global : dict
        dictionary for simple metric with info about noisy frequencies in the average PSD, absolute values for bands
    noise_ampl_relative_to_all_signal_global : dict
        dictionary for simple metric with info about noisy frequencies in the average PSD, relative values for bands
    noisy_freqs_global : dict
        dictionary for simple metric with info about noisy frequencies in the average PSD, central frequencies
    noise_ampl_local_all_ch : dict
        dictionary for simple metric with info about noisy frequencies in the PSD of each channel, absolute values for bands
    noise_ampl_relative_to_all_signal_local_all_ch : dict
        dictionary for simple metric with info about noisy frequencies in the PSD of each channel, relative values for bands
    noisy_freqs_local_all_ch : dict
        dictionary for simple metric with info about noisy frequencies in the PSD of each channel, central frequencies
    
    
    """

    #Calculate noise freqs globally: on the average psd curve over all channels together:
    noise_pie_derivative, noise_ampl_global, noise_ampl_relative_to_all_signal_global, noisy_freqs_global = find_number_and_ampl_of_noise_freqs('Average', freqs, avg_psd, helper_plots, m_or_g, cut_noise_from_psd, prominence_lvl_pos_avg, simple_or_complex)


    #Calculate noise freqs locally: on the psd curve of each channel separately:
    noise_ampl_local_all_ch={}
    noise_ampl_relative_to_all_signal_local_all_ch={}
    noisy_freqs_local_all_ch={}

    for ch_n, ch in enumerate(channels): #for each channel separately
        _, noise_ampl_local_all_ch[ch], noise_ampl_relative_to_all_signal_local_all_ch[ch], noisy_freqs_local_all_ch[ch] = find_number_and_ampl_of_noise_freqs(ch, freqs, psds[ch_n,:], helper_plots, m_or_g, cut_noise_from_psd, prominence_lvl_pos_channels, simple_or_complex)

    return noise_pie_derivative, noise_ampl_global, noise_ampl_relative_to_all_signal_global, noisy_freqs_global, noise_ampl_local_all_ch, noise_ampl_relative_to_all_signal_local_all_ch, noisy_freqs_local_all_ch


def make_dict_global_psd(mean_brain_waves_dict: dict, noisy_freqs_global: List, noise_ampl_global: List, noise_ampl_relative_to_all_signal_global: List):

    """
    Create a dictionary for the global part of psd simple metrics. Global: overall part of noise in the signal (all channels averaged).

    Parameters
    ----------
    mean_brain_waves_dict : dict
        dictionary with the mean brain waves (alpha, beta, etc) metrics in the form: {wave band name: {mean realtive value, mean absolute value}, ...}
    noisy_freqs_global : List
        list of noisy frequencies
    noise_ampl_global : List
        list of noise amplitudes for each noisy frequency band
    noise_ampl_relative_to_all_signal_global : List
        list of noise amplitudes relative to the total signal amplitude for each noisy frequency band
    
    Returns
    -------
    dict_global : dict
        dictionary with the global part of psd simple metrics

    """
        
    noisy_freqs_dict={}
    for fr_n, fr in enumerate(noisy_freqs_global):
        noisy_freqs_dict[fr]={'noise_ampl_global': float(noise_ampl_global[fr_n]), 'percent_of_this_noise_ampl_relative_to_all_signal_global': round(float(noise_ampl_relative_to_all_signal_global[fr_n]*100), 2)}


    dict_global = {
        "mean_brain_waves: ": mean_brain_waves_dict,
        "noisy_frequencies_count: ": len(noisy_freqs_global),
        "details": noisy_freqs_dict}

    return dict_global


def make_dict_local_psd(noisy_freqs_local: dict, noise_ampl_local: dict, noise_ampl_relative_to_all_signal_local: dict, channels: List):

    """
    Create a dictionary for the local part of psd simple metrics. Local: part of noise in the signal for each channel separately.
    
    Parameters
    ----------
    noisy_freqs_local : dict
        dictionary with noisy frequencies for each channel
    noise_ampl_local : dict
        dictionary with noise amplitudes for each noisy frequency band for each channel
    noise_ampl_relative_to_all_signal_local : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band for each channel
        
    Returns
    -------
    dict_local : dict
        dictionary with the local part of psd simple metrics
        
    """

    noisy_freqs_dict_all_ch={}
    for ch in channels:
        central_freqs=noisy_freqs_local[ch]
        noisy_freqs_dict={}     
        for fr_n, fr in enumerate(central_freqs):
            noisy_freqs_dict[fr]={'noise_ampl_local': float(noise_ampl_local[ch][fr_n]), 'percent_of_ths_noise_ampl_relative_to_all_signal_local':  round(float(noise_ampl_relative_to_all_signal_local[ch][fr_n]*100), 2)}
        noisy_freqs_dict_all_ch[ch]=noisy_freqs_dict

    dict_local = {"details": noisy_freqs_dict_all_ch}

    return dict_local


def make_simple_metric_psd(mean_brain_waves_dict: dict, noise_ampl_global:dict, noise_ampl_relative_to_all_signal_global:dict, noisy_freqs_global:dict, noise_ampl_local:dict, noise_ampl_relative_to_all_signal_local:dict, noisy_freqs_local:dict, m_or_g_chosen:list, freqs:dict, channels: dict):

    """
    Create a dictionary for the psd simple metrics.

    Parameters
    ----------
    mean_brain_waves_dict : dict
        dictionary with mean brain waves (alpha, beta, etc) for each channel type. Inside each channel type: dictionary in the form: {wave band name: {mean realtive value, mean absolute value}, ...}
    noise_ampl_global : dict
        dictionary with noise amplitudes for each noisy frequency band 
    noise_ampl_relative_to_all_signal_global : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band 
    noisy_freqs_global : dict
        dictionary with noisy frequencies
    noise_ampl_local : dict
        dictionary with noise amplitudes for each noisy frequency band for each channel
    noise_ampl_relative_to_all_signal_local : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band for each channel
    noisy_freqs_local : dict
        dictionary with noisy frequencies for each channel
    m_or_g_chosen : List
        list with chosen channel types: 'mag' or/and 'grad'

    Returns
    -------
    simple_metric : dict
        dictionary with the psd simple metrics

    """

    metric_global_name = 'PSD_global'
    metric_global_description = 'Noise frequencies detected globally (based on average over all channels in this data file). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude. Brain wave bands mean: amplitudes (area under the curve) of functionally distinct frequency bands. mean_brain_waves_relative in %, mean_brain_waves_abs in mag/grad units.'
    metric_local_name = 'PSD_local'
    metric_local_description = 'Noise frequencies detected locally (present only on individual channels). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude. Brain wave bands per every channel - see csv files.'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}

    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_psd(mean_brain_waves_dict[m_or_g], noisy_freqs_global[m_or_g], noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g])
        metric_local_content[m_or_g]=make_dict_local_psd(noisy_freqs_local[m_or_g], noise_ampl_local[m_or_g], noise_ampl_relative_to_all_signal_local[m_or_g], channels[m_or_g])
        
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'], psd=True)

    return simple_metric


def get_nfft_nperseg(raw: mne.io.Raw, psd_step_size: float):
    
    """
    Get nfft and nperseg parameters for Welch psd function. 
    Allowes to always have the step size in psd which is chosen by the user. Recommended 0.5 Hz.
    
    Parameters
    ----------
    raw : mne.io.Raw
        raw data
    psd_step_size : float
        step size for PSD chosen by user, recommended 0.5 Hz
        
    Returns
    -------
    nfft : int
        Number of points for fft. Used in welch psd function from mne.
        The length of FFT used, must be >= n_per_seg (default: 256). The segments will be zero-padded if n_fft > n_per_seg. 
        If n_per_seg is None, n_fft must be <= number of time points in the data.
    nperseg : int
        Number of points for each segment. Used in welch psd function from mne.
        Length of each Welch segment (windowed with a Hamming window). Defaults to None, which sets n_per_seg equal to n_fft.

    """

    sfreq=raw.info['sfreq']
    nfft=int(sfreq/psd_step_size)
    nperseg=int(sfreq/psd_step_size)

    return nfft, nperseg


def assign_psds_to_channels(chs_by_lobe: dict, freqs: List, psds: List):

    """

    Assign std or ptp values of each epoch as list to each channel. 
    This is done for extraction into TSV later and plotting. 
    
    Parameters
    ----------
    chs_by_lobe : dict
        dictionary with channel objects sorted by ch type and lobe
    freqs : List
        list of frequencies
    psds : List
        list of psd values for each channel

    Returns
    -------
    chs_by_lobe : dict
        dictionary with channel objects sorted by ch type and lobe with added psd and freq values

    """

    for lobe in chs_by_lobe:
        for ch_n, ch in enumerate(chs_by_lobe[lobe]):
            ch.psd = psds[ch_n]
            ch.freq = freqs

    return chs_by_lobe

#%%
def PSD_meg_qc(psd_params: dict, psd_params_internal: dict, channels:dict, chs_by_lobe: dict, data_path:str, m_or_g_chosen: List, helper_plots: bool):
    
    """
    Main psd function. Calculates:

    - PSD for each channel
    - amplitudes (area under the curve) of functionally distinct frequency bands, such as 
        delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), and gamma (30-100 Hz) for each channel 
        and average amplitude of band over all channels
    - average psd over all channels
    - noise frequencies for average psd + creates a band around them
    - noise frequencies for each channel + creates a band around them
    - noise amplitudes (area under the curve) for each noisy frequency band for average psd
    - noise amplitudes (area under the curve) for each noisy frequency band for each channel.

    
    Frequency spectrum peaks we can often see:
    
    - Hz 50, 100, 150 - powerline EU
    - Hz 60, 120, 180 - powerline US
    - Hz 6 - noise of shielding chambers 
    - Hz 44 - MEG device noise
    - Hz 17 - train station 
    - Hz 10 - specific for MEG device in Nessy 
    - Hz 1 - highpass filter.
    - flat frequency spectrum is white noise process. Has same energy in every frequency (starts around 50Hz or even below)

    
    Parameters
    ----------
    psd_params : dict
        dictionary with psd parameters originating from config file
    psd_params_internal : dict
        dictionary with internal psd parameters
    channels : dict
        dictionary with channel names for each channel type: 'mag' or/and 'grad'
    chs_by_lobe : dict
        dictionary with channel objects sorted by ch type and lobe
    raw_orig : mne.io.Raw
        raw data
    m_or_g_chosen : List
        list with chosen channel types: 'mag' or/and 'grad'
    helper_plots : bool
        if True, plots with noisy freq bands for average PSD + for 3 different channels will be created (but not added to report).

    Returns
    -------
    derivs_psd : List
        list with the psd derivatives as QC_derivative objects (figures)
    simple_metric : dict
        dictionary with the psd simple metrics
    psd_str : str
        string with notes about PSD for report
    noisy_freqs_global : dict
        dictionary with noisy frequencies for average psd - used in Muscle artifact detection

    """
    # Load data
    raw, shielding_str, meg_system = load_data(data_path)

    # raw.load_data()

    # these parameters will be saved into a dictionary. this allowes to calculate for mag or grad or both:
    freqs = {}
    psds = {}
    derivs_psd = []
    mean_brain_waves_dict = {'mag':{}, 'grad':{}}
    noise_ampl_global={'mag':[], 'grad':[]}
    noise_ampl_relative_to_all_signal_global={'mag':[], 'grad':[]}
    noisy_freqs_global={'mag':[], 'grad':[]}
    noise_ampl_local={'mag':[], 'grad':[]}
    noise_ampl_relative_to_all_signal_local={'mag':[], 'grad':[]}
    noisy_freqs_local={'mag':[], 'grad':[]}

    method = psd_params_internal['method']
    prominence_lvl_pos_avg = psd_params_internal['prominence_lvl_pos_avg']
    prominence_lvl_pos_channels = psd_params_internal['prominence_lvl_pos_channels']

    nfft, nperseg = get_nfft_nperseg(raw, psd_params['psd_step_size'])

    chs_by_lobe_psd=copy.deepcopy(chs_by_lobe)

    if psd_params['freq_max'] > raw.info['sfreq'] / 2 - 5:
        psd_params['freq_max'] = raw.info['sfreq'] / 2 - 5

    for m_or_g in m_or_g_chosen:

        psds[m_or_g], freqs[m_or_g] = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=channels[m_or_g], n_fft=nfft, n_per_seg=nperseg).get_data(return_freqs=True)
        psds[m_or_g]=np.sqrt(psds[m_or_g]) # amplitude of the noise in this band. without sqrt it is power.

        # Add psds and freqs into chs_by_lobe dict:
        chs_by_lobe_psd[m_or_g] = assign_psds_to_channels(chs_by_lobe_psd[m_or_g], freqs[m_or_g], psds[m_or_g])

        avg_psd=np.mean(psds[m_or_g],axis=0) # average psd over all channels
        
        #Calculate the amplitude of alpha, beta, etc bands for each channel + average over all channels:
        bands_pie_df_deriv, dfs_wave_bands_ampl, mean_brain_waves_dict[m_or_g] = get_ampl_of_brain_waves(channels=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], avg_psd=avg_psd)

        # #Calculate noise freqs for each channel + on the average psd curve over all channels together:
        noise_pie_derivative, noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g], noisy_freqs_global[m_or_g], noise_ampl_local[m_or_g], noise_ampl_relative_to_all_signal_local[m_or_g], noisy_freqs_local[m_or_g] = get_ampl_of_noisy_freqs(channels[m_or_g], freqs[m_or_g], avg_psd, psds[m_or_g], m_or_g, helper_plots=helper_plots, cut_noise_from_psd=False, prominence_lvl_pos_avg=prominence_lvl_pos_avg, prominence_lvl_pos_channels=prominence_lvl_pos_channels, simple_or_complex='simple')
        
        derivs_psd += dfs_wave_bands_ampl +[noise_pie_derivative] + bands_pie_df_deriv


    # Make a simple metric for PSD:
    simple_metric=make_simple_metric_psd(mean_brain_waves_dict, noise_ampl_global, noise_ampl_relative_to_all_signal_global, noisy_freqs_global, noise_ampl_local, noise_ampl_relative_to_all_signal_local, noisy_freqs_local, m_or_g_chosen, freqs, channels)

    psd_str = '' #blank for now. maybe wil need to add notes later.

    #Extract chs_by_lobe into a data frame
    df_deriv = chs_dict_to_csv(chs_by_lobe_psd,  file_name_prefix = 'PSDs')

    derivs_psd += df_deriv

    return derivs_psd, simple_metric, psd_str, noisy_freqs_global
