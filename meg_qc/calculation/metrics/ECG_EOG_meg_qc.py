import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
#import matplotlib #this is in case we will need to suppress mne matplotlib plots
import copy
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from typing import List, Union
from meg_qc.plotting.universal_html_report import simple_metric_basic
from meg_qc.plotting.universal_plots import QC_derivative, get_tit_and_unit
from meg_qc.calculation.initial_meg_qc import (chs_dict_to_csv,load_data)


def check_3_conditions(ch_data: Union[List, np.ndarray], fs: int, ecg_or_eog: str, n_breaks_bursts_allowed_per_10min: int, allowed_range_of_peaks_stds: float, height_multiplier: float):

    """
    Check if the ECG/EOG channel is not corrupted using 3 conditions:
    - peaks have similar amplitude
    - no breaks longer than normal max distance between peaks of hear beats
    - no bursts: too short intervals between peaks

    Parameters
    ----------
    ch_data : List or np.ndarray
        Data of the channel to check
    fs : int
        Sampling frequency of the data
    ecg_or_eog : str
        'ECG' or 'EOG'
    n_breaks_bursts_allowed_per_10min : int, optional
        Number of breaks allowed per 10 minutes of recording, by default 3. Can also set to 0, but then it can falsely detect a break/burst if the peak detection was not perfect.
    allowed_range_of_peaks_stds : float, optional
        Allowed range of standard deviations of peak amplitudes, by default 0.05. Works for ECG channel, but not good for EOG channel.
    height_multiplier: float
        Will define how high the peaks on the ECG channel should be to be counted as peaks. Higher value - higher the peak need to be, hense less peaks will be found.
    
    Returns
    -------
    ecg_eval : dict
        Dictionary with 3 booleans, indicating if the channel is good or bad according to 3 conditions:
        - similar_ampl: True if peaks have similar amplitudes, False otherwise
        - no_breaks: True if there are no breaks longer than normal max distance between peaks of hear beats, False otherwise
        - no_bursts: True if there are no bursts: too short intervals between peaks, False otherwise
    peaks : List
        List of peaks locations

    """

    # 1. Check if R peaks (or EOG peaks)  have similar amplitude. If not - data is too noisy:
    # Find R peaks (or peaks of EOG wave) using find_peaks
    height = np.mean(ch_data) + height_multiplier * np.std(ch_data)
    peaks, _ = find_peaks(ch_data, height=height, distance=round(0.5 * fs)) #assume there are no peaks within 0.5 seconds from each other.


    # scale ecg data between 0 and 1: here we dont care about the absolute values. important is the pattern: 
    # are the peak magnitudes the same on average or not? Since absolute values and hence mean and std 
    # can be different for different data sets, we can just scale everything between 0 and 1 and then
    # compare the peak magnitudes
    ch_data_scaled = (ch_data - np.min(ch_data))/(np.max(ch_data) - np.min(ch_data))
    peak_amplitudes = ch_data_scaled[peaks]

    amplitude_std = np.std(peak_amplitudes)

    if amplitude_std <= allowed_range_of_peaks_stds: 
        similar_ampl = True
        print("___MEGqc___: Peaks have similar amplitudes, amplitude std: ", amplitude_std)
    else:
        similar_ampl = False
        print("___MEGqc___: Peaks do not have similar amplitudes, amplitude std: ", amplitude_std)


    # 2. Calculate RR intervals (time differences between consecutive R peaks)
    rr_intervals = np.diff(peaks) / fs

    if ecg_or_eog.upper() == 'ECG':
        rr_dist_allowed = [0.6, 1.6] #take possible pulse rate of 100-40 bpm (hense distance between peaks is 0.6-1.6 seconds)
    elif ecg_or_eog.upper() == 'EOG':
        rr_dist_allowed = [1, 10] #take possible blink rate of 60-5 per minute (hense distance between peaks is 1-10 seconds). Yes, 60 is a very high rate, but I see this in some data sets often.


    #Count how many segment there are in rr_intervals with breaks or bursts:
    n_breaks = 0
    n_bursts = 0
    for i in range(len(rr_intervals)):
        if rr_intervals[i] > rr_dist_allowed[1]:
            n_breaks += 1
        if rr_intervals[i] < rr_dist_allowed[0]:
            n_bursts += 1

    no_breaks, no_bursts = True, True
    #Check if there are too many breaks:
    if n_breaks > len(rr_intervals)/60*10/n_breaks_bursts_allowed_per_10min:
        print("___MEGqc___: There are more than 2 breaks in the data, number: ", n_breaks)
        no_breaks = False
    if n_bursts > len(rr_intervals)/60*10/n_breaks_bursts_allowed_per_10min:
        print("___MEGqc___: There are more than 2 bursts in the data, number: ", n_bursts)
        no_bursts = False

    ecg_eval = {'similar_ampl': similar_ampl, 'no_breaks': no_breaks, 'no_bursts': no_bursts}
    
    return ecg_eval, peaks


def detect_noisy_ecg(raw: mne.io.Raw, ecg_ch: str,  ecg_or_eog: str, n_breaks_bursts_allowed_per_10min: int, allowed_range_of_peaks_stds: float, height_multiplier: float):
    
    """
    Detects noisy ECG or EOG channels.

    The channel is noisy when:

    1. The distance between the peaks of ECG/EOG signal is too large (events are not frequent enoigh for a human) or too small (events are too frequent for a human).
    2. There are too many breaks in the data (indicating lack of heartbeats or blinks for a too long period) - corrupted channel or dustructed recording
    3. Peaks are of significantly different amplitudes (indicating that the channel is noisy).

    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    ecg_ch : str
        ECG channel names to be checked.
    ecg_or_eog : str
        'ECG' or 'EOG'
    n_breaks_bursts_allowed_per_10min : int
        Number of breaks allowed per 10 minutes of recording. The default is 3.
    allowed_range_of_peaks_stds : float
        Allowed range of peaks standard deviations. The default is 0.05.

        - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
        - The peaks will be detected on the scaled data
        - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy
    
    height_multiplier: float
        Defines how high the peaks on the ECG channel should be to be counted as peaks. Higher value - higher the peak need to be, hense less peaks will be found.

        
    Returns
    -------
    bad_ecg_eog : dict
        Dictionary with channel names as keys and 'good' or 'bad' as values.
    ch_data : List
        Data of the ECG channel recorded.
    peaks : List
        List of peaks locations.
    ecg_eval_str : str
        String indicating if the channel is good or bad according to 3 conditions:
        - similar_ampl: True if peaks have similar amplitudes, False otherwise
        - no_breaks: True if there are no breaks longer than normal max distance between peaks of hear beats, False otherwise
        - no_bursts: True if there are no bursts: too short intervals between peaks, False otherwise

    """

    sfreq=raw.info['sfreq']

    bad_ecg_eog = {}
    peaks = []

    ch_data = raw.get_data(picks=ecg_ch)[0] #here ch_data will be the RAW DATA
    # get_data creates list inside of a list because expects to create a list for each channel. this is why [0]

    ecg_eval, peaks = check_3_conditions(ch_data, sfreq, ecg_or_eog, n_breaks_bursts_allowed_per_10min, allowed_range_of_peaks_stds, height_multiplier)
    print(f'___MEGqc___: {ecg_ch} satisfied conditions for a good channel: ', ecg_eval)

    #If all values in dict are true:
    if all(value == True for value in ecg_eval.values()):
        bad_ecg_eog[ecg_ch] = 'good'
    else:
        bad_ecg_eog[ecg_ch] = 'bad'

    ecg_eval_str = 'Overall ' + bad_ecg_eog[ecg_ch] + ' ' + ecg_or_eog + ' channel: ' + ecg_ch + ': \n - Peaks have similar amplitude: ' + str(ecg_eval["similar_ampl"]) + ' \n - No breaks (too long distances between peaks): ' + str(ecg_eval["no_breaks"]) + ' \n - No bursts (too short distances between peaks): ' + str(ecg_eval["no_bursts"]) + '\n'
    print(f'___MEGqc___: ', ecg_eval_str)

    return bad_ecg_eog, ch_data, peaks, ecg_eval_str


def find_epoch_peaks(ch_data: np.ndarray, thresh_lvl_peakfinder: float):
    
    """
    Find the peaks in the epoch data using the peakfinder algorithm.

    Parameters
    ----------
    ch_data : np.ndarray
        The data of the channel.
    thresh_lvl_peakfinder : float
        The threshold for the peakfinder algorithm.

    Returns
    -------
    peak_locs_pos : np.ndarray
        The locations of the positive peaks.
    peak_locs_neg : np.ndarray
        The locations of the negative peaks.
    peak_magnitudes_pos : np.ndarray
        The magnitudes of the positive peaks.
    peak_magnitudes_neg : np.ndarray
        The magnitudes of the negative peaks.

        
    """


    thresh_mean=(max(ch_data) - min(ch_data)) / thresh_lvl_peakfinder
    peak_locs_pos, _ = find_peaks(ch_data, prominence=thresh_mean)
    peak_locs_neg, _ = find_peaks(-ch_data, prominence=thresh_mean)

    try:
        peak_magnitudes_pos=ch_data[peak_locs_pos]
    except:
        peak_magnitudes_pos=np.empty(0)

    try:
        peak_magnitudes_neg=ch_data[peak_locs_neg]
    except:
        peak_magnitudes_neg=np.empty(0)

    return peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg


class Avg_artif:
    
    """ 
    Instance of this class:

    - contains average ECG/EOG epoch for a particular channel,
    - calculates its main peak (location and magnitude), possibe on both smoothed and non smoothed data.
    - evaluates if this epoch is concidered as artifact or not based on the main peak amplitude.
    

    Attributes
    ----------
    name : str
        name of the channel
    artif_data : List
        list of floats, average ecg epoch for a particular channel
    peak_loc : int
        locations of peaks inside the artifact epoch
    peak_magnitude : float
        magnitudes of peaks inside the artifact epoch
    wave_shape : bool
        True if the average epoch has typical wave shape, False otherwise. R wave shape  - for ECG or just a wave shape for EOG.
    artif_over_threshold : bool
        True if the main peak is concidered as artifact, False otherwise. True if artifact sas magnitude over the threshold
    main_peak_loc : int
        location of the main peak inside the artifact epoch
    main_peak_magnitude : float
        magnitude of the main peak inside the artifact epoch
    artif_data_smoothed : List
        list of floats, average ecg epoch for a particular channel, smoothed usig Gaussian filter
    peak_loc_smoothed : int
        locations of peaks inside the artifact epoch calculated on smoothed data
    peak_magnitude_smoothed : float
        magnitudes of peaks inside the artifact epoch calculated on smoothed data
    wave_shape_smoothed : bool
        True if the average epoch has typical wave shape, False otherwise. R wave shape  - for ECG or just a wave shape for EOG. Calculated on smoothed data
    artif_over_threshold_smoothed : bool
        True if the main peak is concidered as artifact, False otherwise. True if artifact sas magnitude over the threshold. Calculated on smoothed data
    main_peak_loc_smoothed : int
        location of the main peak inside the artifact epoch. Calculated on smoothed data
    main_peak_magnitude_smoothed : float
        magnitude of the main peak inside the artifact epoch. Calculated on smoothed data
    corr_coef : float
        correlation coefficient between the ECG/EOG channels data and average data of this mag/grad channel
    p_value : float
        p-value of the correlation coefficient between the ECG/EOG channels data and average data of this mag/grad channel
    amplitude_ratio: float
        relation of the amplitude of a particular channel to all other channels
    similarity_score: float
        similarity score of the mean ecg/eog data of this channel to refernce ecg/eog data comprised of both correlation and amplitude like: similarity_score = corr_coef * amplitude_ratio
    lobe: str
        which lobe his channel belongs to
    color: str
        color code for this channel according to the lobe it belongs to
    

    Methods
    -------
    __init__(self, name: str, artif_data:list, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None)
        Constructor
    __repr__(self)
        Returns a string representation of the object

    """

    def __init__(self, name: str, artif_data: List, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None, artif_data_smoothed: Union[List, None] = None, peak_loc_smoothed=None, peak_magnitude_smoothed=None, wave_shape_smoothed:bool=None, artif_over_threshold_smoothed:bool=None, main_peak_loc_smoothed: int=None, main_peak_magnitude_smoothed: float=None, corr_coef: float = None, p_value: float = None, amplitude_ratio: float = None, similarity_score: float = None, lobe: str = None, color: str = None):
        """Constructor"""
        
        self.name =  name
        self.artif_data = artif_data
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.wave_shape =  wave_shape
        self.artif_over_threshold = artif_over_threshold
        self.main_peak_loc = main_peak_loc
        self.main_peak_magnitude = main_peak_magnitude
        self.artif_data_smoothed = artif_data_smoothed
        self.peak_loc_smoothed = peak_loc_smoothed
        self.peak_magnitude_smoothed = peak_magnitude_smoothed
        self.wave_shape_smoothed =  wave_shape_smoothed
        self.artif_over_threshold_smoothed = artif_over_threshold_smoothed
        self.main_peak_loc_smoothed = main_peak_loc_smoothed
        self.main_peak_magnitude_smoothed = main_peak_magnitude_smoothed
        self.corr_coef = corr_coef
        self.p_value = p_value
        self.amplitude_ratio = amplitude_ratio
        self.similarity_score = similarity_score
        self.lobe = lobe
        self.color = color


    def __repr__(self):
        """
        Returns a string representation of the object
        
        """

        return 'Mean artifact for: ' + str(self.name) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) +'\n - main_peak_loc: '+ str(self.main_peak_loc) +'\n - main_peak_magnitude: '+str(self.main_peak_magnitude)+'\n - wave_shape: '+ str(self.wave_shape) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n - corr_coef: ' + str(self.corr_coef) + '\n - p_value: ' + str(self.p_value) + '\n - amplitude_ratio: ' + str(self.amplitude_ratio) + '\n - similarity_score: ' + str(self.similarity_score) + '\n - lobe: ' + str(self.lobe) + '\n - color: ' + str(self.color) + '\n - peak_loc_smoothed: ' + str(self.peak_loc_smoothed) + '\n - peak_magnitude_smoothed: ' + str(self.peak_magnitude_smoothed) + '\n - wave_shape_smoothed: ' + str(self.wave_shape_smoothed) + '\n - artif_over_threshold_smoothed: ' + str(self.artif_over_threshold_smoothed) + '\n - main_peak_loc_smoothed: ' + str(self.main_peak_loc_smoothed) + '\n - main_peak_magnitude_smoothed: ' + str(self.main_peak_magnitude_smoothed) + '\n'
    


    def get_peaks_wave(self, max_n_peaks_allowed: int, thresh_lvl_peakfinder: float):

        """
        Find peaks in the average artifact epoch and decide if the epoch has wave shape: 
        few peaks (different number allowed for ECG and EOG) - wave shape, many or no peaks - not.
        Function for non smoothed data.
        
        Parameters
        ----------
        max_n_peaks_allowed : int
            maximum number of peaks allowed in the average artifact epoch
        thresh_lvl_peakfinder : float
            threshold for peakfinder function.
        
            
        """

        peak_locs_pos_orig, peak_locs_neg_orig, peak_magnitudes_pos_orig, peak_magnitudes_neg_orig = find_epoch_peaks(ch_data=self.artif_data, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc=np.concatenate((peak_locs_pos_orig, peak_locs_neg_orig), axis=None)
        self.peak_magnitude=np.concatenate((peak_magnitudes_pos_orig, peak_magnitudes_neg_orig), axis=None)

        if np.size(self.peak_loc)==0: #no peaks found
            self.wave_shape=False
        elif 1<=len(self.peak_loc)<=max_n_peaks_allowed:
            self.wave_shape=True
        elif len(self.peak_loc)>max_n_peaks_allowed:
            self.wave_shape=False
        else:
            print('Something went wrong with peak detection')


    def get_peaks_wave_smoothed(self, gaussian_sigma: int, max_n_peaks_allowed: int, thresh_lvl_peakfinder: float):

        """
        Find peaks in the average artifact epoch and decide if the epoch has wave shape: 
        few peaks (different number allowed for ECG and EOG) - wave shape, many or no peaks - not.
        Function for smoothed data. If it was not smoothed yet - it will be smoothed inside this function using gaussian filter.
        
        Parameters
        ----------
        gaussian_sigma : int
            sigma for gaussian smoothing
        max_n_peaks_allowed : int
            maximum number of peaks allowed in the average artifact epoch
        thresh_lvl_peakfinder : float
            threshold for peakfinder function.

        """

        if self.artif_data_smoothed is None: #if no smoothed data available yet
            self.smooth_artif(gaussian_sigma) 

        peak_locs_pos_smoothed, peak_locs_neg_smoothed, peak_magnitudes_pos_smoothed, peak_magnitudes_neg_smoothed = find_epoch_peaks(ch_data=self.artif_data_smoothed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc_smoothed=np.concatenate((peak_locs_pos_smoothed, peak_locs_neg_smoothed), axis=None)
        self.peak_magnitude_smoothed=np.concatenate((peak_magnitudes_pos_smoothed, peak_magnitudes_neg_smoothed), axis=None)

        if np.size(self.peak_loc_smoothed)==0:
            self.wave_shape_smoothed=False
        elif 1<=len(self.peak_loc_smoothed)<=max_n_peaks_allowed:
            self.wave_shape_smoothed=True
        elif len(self.peak_loc_smoothed)>max_n_peaks_allowed:
            self.wave_shape_smoothed=False
        else:
            print('Something went wrong with peak detection')


    def get_highest_peak(self, t: np.ndarray, before_t0: float, after_t0: float):

        """
        Find the highest peak of the artifact epoch inside the given time window. 
        Time window is centered around the t0 of the ecg/eog event and limited by before_t0 and after_t0.
        

        Parameters
        ----------
        t : List
            time vector
        before_t0 : float
            before time limit for the peak
        after_t0 : float
            after time limit for the peak
            
        Returns
        -------
        self.main_peak_loc : int
            location of the main peak
        self.main_peak_magnitude : float
            magnitude of the main peak
        
        """

        if self.peak_loc is None: #if no peaks were found on original data:
            self.main_peak_magnitude=None
            self.main_peak_loc=None
        elif self.peak_loc is not None: #if peaks were found on original data:
            self.main_peak_magnitude = -1000
            for peak_loc in self.peak_loc:
                if before_t0<t[peak_loc]<after_t0: #if peak is inside the before_t0 and after_t0 was found:
                    if self.artif_data[peak_loc] > self.main_peak_magnitude: #if this peak is higher than the previous one:
                        self.main_peak_magnitude=self.artif_data[peak_loc]
                        self.main_peak_loc=peak_loc 
    
            if self.main_peak_magnitude == -1000: #if no peak was found inside the before_t0 and after_t0:
                self.main_peak_magnitude=None
                self.main_peak_loc=None
        else:
            self.main_peak_loc, self.main_peak_magnitude = None, None


        return self.main_peak_loc, self.main_peak_magnitude
    

    def get_highest_peak_smoothed(self, t: np.ndarray, before_t0: float, after_max: float):

        """
        Find the highest peak of the artifact epoch inside the give time window on SMOOTHED data.
        Time window is centered around the t0 of the ecg/eog event and limited by before_t0 and after_t0.
        
        
        Parameters
        ----------
        t : List
            time vector
        before_t0 : float
            before time limit for the peak
        after_t0 : float
            after time limit for the peak
            maximum time limit for the peak
            
        Returns
        -------
        self.main_peak_magnitude_smoothed : float
            magnitude of the main peak on smoothed data
        self.main_peak_loc_smoothed : int
            location of the main peak on smoothed data
        
        
        """

        if self.peak_loc_smoothed is None:
            self.main_peak_magnitude_smoothed=None
            self.main_peak_loc_smoothed=None
        elif self.peak_loc_smoothed is not None:
            self.main_peak_magnitude_smoothed = -1000
            for peak_loc in self.peak_loc_smoothed:
                if before_t0<t[peak_loc]<after_max:
                    if self.artif_data_smoothed[peak_loc] > self.main_peak_magnitude_smoothed:
                        self.main_peak_magnitude_smoothed=self.artif_data_smoothed[peak_loc]
                        self.main_peak_loc_smoothed=peak_loc 
    
            if self.main_peak_magnitude_smoothed == -1000:
                self.main_peak_magnitude_smoothed=None
                self.main_peak_loc_smoothed=None

        else:    
            self.main_peak_loc_smoothed, self.main_peak_magnitude_smoothed = None, None


        return self.main_peak_loc_smoothed, self.main_peak_magnitude_smoothed
    
    
    def smooth_artif(self, gauss_sigma: int):

        """ 
        Smooth the artifact epoch using gaussian filter.
        This is done do detect the wave shape in presence of noise. 
        Usually EOG are more noisy than ECG, so need different sigma for these 2 kinds.
        
        Parameters
        ----------
        gauss_sigma : int
            sigma of the gaussian filter
            
        Returns
        -------
        self
            Avg_artif object with smoothed artifact epoch in self.artif_data_smoothed
        
        """

        data_copy=copy.deepcopy(self.artif_data)
        self.artif_data_smoothed = gaussian_filter(data_copy, gauss_sigma)

        return self
    

    def flip_artif(self):
            
        """
        Flip the artifact epoch upside down on original (non smoothed) data.
        This is only done if the need to flip was detected in flip_channels() function.
        
        Returns
        -------
        self
            Avg_artif object with flipped artifact epoch in self.artif_data and self.peak_magnitude
        
        """

        if self.artif_data is not None:
            self.artif_data = -self.artif_data
        if self.peak_magnitude is not None:
            self.peak_magnitude = -self.peak_magnitude

        return self
    

    def flip_artif_smoothed(self):
            
        """
        Flip the SMOOTHED artifact epoch upside down.
        This is only done if the need to flip was detected in flip_channels() function.
        
        Returns
        -------
        self
            Avg_artif object with flipped smoothed artifact epoch in self.artif_data_smoothed and self.peak_magnitude_smoothed
        
        """

        if self.artif_data_smoothed is not None:
            self.artif_data_smoothed = -self.artif_data_smoothed
        
        if self.peak_magnitude_smoothed is not None:
            self.peak_magnitude_smoothed = -self.peak_magnitude_smoothed

        return self

    def detect_artif_above_threshold(self, artif_threshold_lvl: float, t: np.ndarray, before_t0: float, after_t0: float):

        """
        Detect if the highest peak of the artifact epoch is above a given threshold.
        Time window is centered around the t0 of the ecg/eog event and limited by before_t0 and after_t0.

        Parameters
        ----------
        artif_threshold_lvl : float
            threshold level
        t : List
            time vector
        before_t0 : float
            minimum time limit for the peak
        after_t0 : float
            maximum time limit for the peak

        Returns
        -------
        self.artif_over_threshold : bool
            True if the highest peak is above the threshold, False otherwise

        """

        if self.artif_data is not None:
            #find the highest peak inside the time frame:
            _, main_peak_magnitude_orig = self.get_highest_peak(t=t, before_t0=before_t0, after_t0=after_t0)
            if main_peak_magnitude_orig is not None:
                if main_peak_magnitude_orig>abs(artif_threshold_lvl) and self.wave_shape is True:
                    self.artif_over_threshold=True
                else:
                    self.artif_over_threshold=False
            else:
                self.artif_over_threshold=False
        
        return self.artif_over_threshold


    def detect_artif_above_threshold_smoothed(self, artif_threshold_lvl: float, t: np.ndarray, before_t0: float, after_t0: float):

        """
        Detect if the highest peak of the artifact epoch is above a given threshold for SMOOTHED data.
        Time window is centered around the t0 of the ecg/eog event and limited by before_t0 and after_t0.

        Parameters
        ----------
        artif_threshold_lvl : float
            threshold level
        t : List
            time vector
        before_t0 : float
            minimum time limit for the peak
        after_t0 : float
            maximum time limit for the peak

        Returns
        -------
        self.artif_over_threshold : bool
            True if the highest peak is above the threshold, False otherwise

        """

        if self.artif_data_smoothed is not None:
            #find the highest peak inside the before_t0 and after_t0:
            _, main_peak_magnitude_smoothed = self.get_highest_peak(t=t, before_t0=before_t0, after_t0=after_t0)
            if main_peak_magnitude_smoothed is not None:
                if main_peak_magnitude_smoothed>abs(artif_threshold_lvl) and self.wave_shape_smoothed is True:
                    self.artif_over_threshold_smoothed=True
                else:
                    self.artif_over_threshold_smoothed=False
            else:
                self.artif_over_threshold_smoothed=False

        return self.artif_over_threshold_smoothed


def detect_channels_above_norm(norm_lvl: float, list_mean_artif_epochs: List, mean_magnitude_peak: float, t: np.ndarray, t0_actual: float, window_size_for_mean_threshold_method: float, mean_magnitude_peak_smoothed: float = None, t0_actual_smoothed: float = None):

    """
    Find the channels which got average artifact amplitude higher than the average over all channels*norm_lvl.
    
    Parameters
    ----------
    norm_lvl : float
        The norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold.
    list_mean_artif_epochs : List
        List of MeanArtifactEpoch objects, each hold the information about mean artifact for one channel.
    mean_magnitude_peak : float
        The magnitude the mean artifact amplitude over all channels.
    t : np.ndarray
        Time vector.
    t0_actual : float
        The time of the ecg/eog event.
    window_size_for_mean_threshold_method: float
        this value will be taken before and after the t0_actual. It defines the time window in which the peak of artifact on the channel has to present 
        to be counted as artifact peak and compared t the threshold. Unit: seconds
    mean_magnitude_peak_smoothed : float, optional
        The magnitude the mean artifact amplitude over all channels for SMOOTHED data. The default is None.
    t0_actual_smoothed : float, optional
        The time of the ecg/eog event for SMOOTHED data. The default is None.

    Returns
    -------
    affected_orig : List
        List of channels which got average artifact amplitude higher than the average over all channels*norm_lvl.
    not_affected_orig : List
        List of channels which got average artifact amplitude lower than the average over all channels*norm_lvl.
    artif_threshold_lvl : float
        The threshold level for the artifact amplitude.
    affected_smoothed : List
        List of channels which got average artifact amplitude higher than the average over all channels*norm_lvl for SMOOTHED data.
    not_affected_smoothed : List 
        List of channels which got average artifact amplitude lower than the average over all channels*norm_lvl for SMOOTHED data.
    artif_threshold_lvl_smoothed : float
        The threshold level for the artifact amplitude for SMOOTHED data.
    
    """

    before_t0=-window_size_for_mean_threshold_method+t0_actual
    after_t0=window_size_for_mean_threshold_method+t0_actual


    #Find the channels which got peaks over this mean:
    affected_orig=[]
    not_affected_orig=[]
    affected_smoothed=[]
    not_affected_smoothed=[]

    artif_threshold_lvl=mean_magnitude_peak/norm_lvl #data over this level will be counted as artifact contaminated

    if mean_magnitude_peak_smoothed is None or t0_actual_smoothed is None:
        print('___MEGqc___: ', 'mean_magnitude_peak_smoothed and t0_actual_smoothed should be provided')
    else:
        artifact_lvl_smoothed=mean_magnitude_peak_smoothed/norm_lvl  #SO WHEN USING SMOOTHED CHANNELS - USE SMOOTHED AVERAGE TOO!
        before_t0_smoothed=-window_size_for_mean_threshold_method+t0_actual_smoothed
        after_t0_smoothed=window_size_for_mean_threshold_method+t0_actual_smoothed

    # Detect which channels are affected by the artifact based on the threshold:
    for potentially_affected in list_mean_artif_epochs:

        result = potentially_affected.detect_artif_above_threshold(artif_threshold_lvl, t, before_t0, after_t0)
        if result is True:
            affected_orig.append(potentially_affected)
        else:
            not_affected_orig.append(potentially_affected)
        
        result_smoothed = potentially_affected.detect_artif_above_threshold_smoothed(artifact_lvl_smoothed, t, before_t0_smoothed, after_t0_smoothed)
        if result_smoothed is True:
            affected_smoothed.append(potentially_affected)
        else:
            not_affected_smoothed.append(potentially_affected)

    return affected_orig, not_affected_orig, artif_threshold_lvl, affected_smoothed, not_affected_smoothed, artifact_lvl_smoothed


def flip_channels(artif_per_ch_nonflipped: List, tmin: float, tmax: float, sfreq: int, params_internal: dict):

    """
    Flip the channels if the peak of the artifact is negative and located close to the estimated t0.

    Flip approach: 

    - define  a window around the ecg/eog event deteceted by mne. This is not the real t0, but  an approximation. 
        The size of the window defines by how large on average the error of mne is when mne algorythm estimates even time. 
        So for example if mne is off by 0.05s on average, then the window should be -0.05 to 0.05s. 
    - take 5 channels with the largest peak in this window - assume these peaks are the actual artifact.
    - find the average of these 5 peaks - this is the new estimated_t0 (but still not the real t0)
    - create a new window around this new t0 - in this time window all the artifact wave shapes should be located on all channels.
    - flip the channels, if they have a peak inside of this new window, but the peak is negative and it is the closest peak to estimated t0. 
        if the peak is positive - do not flip.
    - collect all final flipped+unflipped eppochs of these channels 

    
    Parameters
    ----------
    avg_artif_nonflipped : List
        List of Avg_artif objects with not flipped data.
    tmin : float
        time in sec before the peak of the artifact (negative number).
    tmax : float
        time in sec after the peak of the artifact (positive number).
    sfreq : int
        Sampling frequency.
    params_internal : dict
        Dictionary with internal parameters.


    Returns
    -------
    artifacts_flipped : List
        The list of the ecg epochs.
    artif_time_vector : np.ndarray
        The time vector for the ecg epoch (for plotting further).


    """

    artif_time_vector = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round

    _, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end = estimate_t0(artif_per_ch_nonflipped, artif_time_vector, params_internal)

    artifacts_flipped=[]

    for ch_artif in artif_per_ch_nonflipped: #for each channel:

        if ch_artif.peak_loc.size>0: #if there are any peaks - find peak_locs which is located the closest to t0_estimated_ind:
            peak_loc_closest_to_t0=ch_artif.peak_loc[np.argmin(np.abs(ch_artif.peak_loc-t0_estimated_ind))]

            #if peak_loc_closest_t0 is negative and is located in the estimated time window of the wave - flip the data:
            if (ch_artif.artif_data[peak_loc_closest_to_t0]<0) & (peak_loc_closest_to_t0>t0_estimated_ind_start) & (peak_loc_closest_to_t0<t0_estimated_ind_end):
                ch_artif.flip_artif()
                if ch_artif.artif_data_smoothed is not None: #if there is also smoothed data present - flip it as well:
                    ch_artif.flip_artif_smoothed()
            else:
                pass
        else:
            pass

        artifacts_flipped.append(ch_artif)

    return artifacts_flipped, artif_time_vector


def estimate_t0(artif_per_ch_nonflipped: List, t: np.ndarray, params_internal: dict):
    
    """ 
    Estimate t0 for the artifact. MNE has it s own estimation of t0, but it is often not accurate.
    t0 will be the point of the maximal amplitude of the artifact.
    Steps:

    1. find maxima on all channels (absolute values) in time frame, for example: -0.02<t[peak_loc]<0.012 
        (here R wave is typically detected by mne - for ecg), 
        for eog it is usually: -0.1<t[peak_loc]<0.2)
        But these are set in settings_internal.ini file.
    2. take 5 channels with most prominent peak 
    3. find estimated average t0 for all 5 channels, set it as new t0.
    

    Parameters
    ----------
    ecg_or_eog : str
        The type of the artifact: 'ECG' or 'EOG'.
    avg_ecg_epoch_data_nonflipped : np.ndarray
        The data of the channels.
    t : np.ndarray
        The time vector.
    params_internal : dict
        Dictionary with internal parameters.
        
    Returns
    -------
    t0_estimated_ind : int
        The index of the estimated t0.
    t0_estimated : float
        The estimated t0.
    t0_estimated_ind_start : int
        The start index of the time window for the estimated t0.
    t0_estimated_ind_end : int
        The end index of the time window for the estimated t0.
    
        
    """

    window_size_for_mean_threshold_method=params_internal['window_size_for_mean_threshold_method']
    before_t0 = params_internal['before_t0']
    after_t0 = params_internal['after_t0']

    #collect artif data for each channel into nd array:
    avg_ecg_epoch_data_nonflipped = np.array([ch.artif_data for ch in artif_per_ch_nonflipped]) 

    #find indexes of t where t is between before_t0 and after_t0 (limits where R wave typically is detected by mne):
    t_event_ind=np.argwhere((t>before_t0) & (t<after_t0))

    # cut the data of each channel to the time interval where wave is expected to be:
    avg_ecg_epoch_data_nonflipped_limited_to_event=avg_ecg_epoch_data_nonflipped[:,t_event_ind[0][0]:t_event_ind[-1][0]]

    #find 5 channels with max values in the time interval where wave is expected to be:
    max_values=np.max(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event), axis=1)
    max_values_ind=np.argsort(max_values)[::-1]
    max_values_ind=max_values_ind[:5]

    # find the index of max value for each of these 5 channels:
    max_values_ind_in_avg_ecg_epoch_data_nonflipped=np.argmax(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event[max_values_ind]), axis=1)
    
    #find average index of max value for these 5 channels, then derive t0_estimated:
    t0_estimated_average=int(np.round(np.mean(max_values_ind_in_avg_ecg_epoch_data_nonflipped)))
    #limited to event means that the index is limited to the time interval where R wave is expected to be.
    #Now need to get back to actual time interval of the whole epoch:

    #find t0_estimated to use as the point where peak of each ch data should be:
    t0_estimated_ind=t_event_ind[0][0]+t0_estimated_average #sum because time window was cut from the beginning of the epoch previously
    t0_estimated=t[t0_estimated_ind]

    # window around t0_estimated where the peak on different channels should be detected:
    t0_estimated_ind_start=np.argwhere(t==round(t0_estimated-window_size_for_mean_threshold_method, 3))[0][0] 
    t0_estimated_ind_end=np.argwhere(t==round(t0_estimated+window_size_for_mean_threshold_method, 3))[0][0]
    #yes you have to round it here because the numbers stored in in memery like 0.010000003 even when it looks like 0.01, hence np.where cant find the target float in t vector
    
    return t0_estimated, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end



def calculate_artifacts_on_channels(artif_epochs: mne.Epochs, channels: List, chs_by_lobe: dict, thresh_lvl_peakfinder: float, tmin: float, tmax: float, params_internal: dict, gaussian_sigma: int):

    """
    Find channels that are affected by ECG or EOG events.
    The function calculates average ECG epoch for each channel and then finds the peak of the wave on each channel.
   

    Parameters
    ----------
    artif_epochs : mne.Epochs
        ECG epochs.
    channels : List
        List of channels to use.
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe
    thresh_lvl_peakfinder : float
        Threshold level for peakfinder.
    tmin : float
        Start time.
    tmax : float
        End time.
    params_internal : dict
        Dictionary with internal parameters.
    gaussian_sigma : int, optional
        Sigma for gaussian filter. The default is 6. Usually for EOG need higher (6-7), t s more noisy, for ECG - lower (4-5).

        
    Returns 
    -------
    all_artifs_nonflipped : List
        List of channels with Avg_artif objects, data in these is not flipped yet.
        
    """

    max_n_peaks_allowed_for_ch = params_internal['max_n_peaks_allowed_for_ch']

    max_n_peaks_allowed=round(((abs(tmin)+abs(tmax))/0.1)*max_n_peaks_allowed_for_ch)
    print('___MEGqc___: ', 'max_n_peaks_allowed_for_ch: '+str(max_n_peaks_allowed))

    #1.:
    #averaging the ECG epochs together:
    avg_epochs = artif_epochs.average(picks=channels)#.apply_baseline((-0.5, -0.2))
    #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
    #The data in an Evoked object are stored in an array of shape (n_channels, n_times)

    # 1. find maxima on all channels (absolute values) in time frame around -0.02<t[peak_loc]<0.012 (here R wave is typicaly detected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
    # 2. take 5 channels with most prominent peak 
    # 3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate

    avg_artif_data_nonflipped=avg_epochs.data #shape (n_channels, n_times)

    # 4. detect peaks on channels 
    all_artifs_nonflipped = []
    for i, ch_data in enumerate(avg_artif_data_nonflipped):  # find peaks and estimate detect wave shape on all channels
        artif_nonflipped = Avg_artif(name=channels[i], artif_data=ch_data)
        artif_nonflipped.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        artif_nonflipped.get_peaks_wave_smoothed(gaussian_sigma = gaussian_sigma, max_n_peaks_allowed=max_n_peaks_allowed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        all_artifs_nonflipped.append(artif_nonflipped)

    # assign lobe to each channel right away (for plotting)
    all_artifs_nonflipped = assign_lobe_to_artifacts(all_artifs_nonflipped, chs_by_lobe)

    return all_artifs_nonflipped


def find_mean_rwave_blink(ch_data: Union[List, np.ndarray], event_indexes: np.ndarray, tmin: float, tmax: float, sfreq: int):

    """
    Calculate mean R wave on the data of either original ECG channel or reconstructed ECG channel.
    In some cases (for reconstructed) there are no events, so mean Rwave cant be estimated.
    This usually does not happen for real ECG channel. Because real ECG channel passes the check even earlier in the code. (see check_3_conditions())

    Parameters
    ----------
    ch_data : np.ndarray
        Data of the channel (real or reconstructed).
    event_indexes : array
        Array of event indexes (R wave peaks).
    tmin : float
        Start time of ECG epoch (negative value).
    tmax : float
        End time of ECG epoch (positive value).
    sfreq : int
        Sampling frequency.

    Returns
    -------
    mean_rwave : np.ndarray
        Mean R wave (1 dimentional).
    
    """

    # Initialize an empty array to store the extracted epochs
    epochs = np.zeros((len(event_indexes), int((tmax-tmin)*sfreq)+1))

    # Loop through each ECG event and extract the corresponding epoch
    for i, event in enumerate(event_indexes):
        start = np.round(event + tmin*sfreq).astype(int)
        end = np.round(event + tmax*sfreq).astype(int)+1

        if start < 0:
            continue

        if end > len(ch_data):
            continue

        epochs[i, :] = ch_data[start:end]

    #average all epochs:
    mean_rwave=np.mean(epochs, axis=0)

    return mean_rwave


def assign_lobe_to_artifacts(artif_per_ch: List, chs_by_lobe: dict):

    """ Loop over all channels in artif_per_ch and assign lobe and lobe color to each channel for plotting purposes.

    Parameters
    ----------
    artif_per_ch : List
        List of channels with Avg_artif objects.
    chs_by_lobe : dict
        Dictionary of channels grouped by lobe with color codes.

    Returns
    -------
    artif_per_ch : List
        List of channels with Avg_artif objects, now with assigned lobe and color for plotting. 

    """
    
    for lobe,  ch_list in chs_by_lobe.items(): #loop over dict of channels for plotting
        for ch_for_plot in ch_list: #same, level deeper
            for ch_artif in artif_per_ch: #loop over list of instances of Avg_artif class
                if ch_artif.name == ch_for_plot.name:
                    ch_artif.lobe = ch_for_plot.lobe
                    ch_artif.color = ch_for_plot.lobe_color
                    break

    #Check that all channels have been assigned a lobe:
    for ch_artif in artif_per_ch:
        if ch_artif.lobe is None or ch_artif.color is None:
            print('___MEGqc___: ', 'Channel ', ch_artif.name, ' has not been assigned a lobe or color for plotting. Check assign_lobe_to_artifacts().')

    return artif_per_ch

def align_artif_data(ch_wave, mean_rwave):

    """
    Align the channel data with the mean R wave by finding the time shift that maximizes the correlation between the two signals.
    
    Parameters
    ----------
    ch_wave : np.ndarray
        Channel data.
    mean_rwave : np.ndarray
        Mean R wave.

    Returns
    -------
    best_aligned_ch_wave : np.ndarray
        Channel data aligned with the mean R wave.
    best_time_shift : int
        Time shift that maximizes the correlation.
    best_correlation : float
        Correlation between the mean R wave and the aligned channel data in the best alignment.
        
    """

    # Find peaks in mean_rwave
    peaks1, _ = find_peaks(mean_rwave)

    # Initialize variables for best alignment
    best_time_shift = 0
    best_correlation = -np.inf
    best_aligned_ch_wave = None

    # Try aligning ch_wave in both orientations
    for flip in [False, True]:
        # Flip ch_wave if needed
        #aligned_ch_wave = np.flip(ch_wave) if flip else ch_wave
        aligned_ch_wave = -ch_wave if flip else ch_wave

        # Find peaks in aligned_ch_wave
        peaks2, _ = find_peaks(aligned_ch_wave)

        # Calculate the time shift based on the peak positions
        time_shift = peaks1[0] - peaks2[0]

        # Shift aligned_ch_wave to align with mean_rwave
        aligned_ch_wave = np.roll(aligned_ch_wave, time_shift)

        # Calculate the correlation between mean_rwave and aligned_ch_wave
        correlation = np.corrcoef(mean_rwave, aligned_ch_wave)[0, 1]

        # Update the best alignment if the correlation is higher
        if correlation > best_correlation:
            best_correlation = correlation
            best_time_shift = time_shift
            best_aligned_ch_wave = aligned_ch_wave
        
    return best_aligned_ch_wave, best_time_shift, best_correlation


def find_affected_by_correlation(mean_rwave: np.ndarray, artif_per_ch: List):

    """
    Calculate correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    Higher correlation coefficient means that the channel is more likely to be affected by ECG artifact.

    Here we assume that both vectors have sme length! these are defined by tmin and tmax which are set in config and propageted in this script. 
    Keep in mind if changing anything with tmin and tmax
    
    Parameters
    ----------
    mean_rwave : np.ndarray
        Mean R wave (1 dimentional).
    artif_per_ch : List
        List of channels with Avg_artif objects.

    Returns
    -------
    artif_per_ch : List
        List of channels with Avg_artif objects, now with assigned correlation coefficient and p-value.
    
    """

    
    if len(mean_rwave) != len(artif_per_ch[0].artif_data):
        print('___MEGqc___: ', 'mean_rwave and artif_per_ch.artif_data have different length! Both are defined by tmin and tmax in config.py and are use to cut the data. Keep in mind if changing anything with tmin and tmax')
        print('len(mean_rwave): ', len(mean_rwave), 'len(artif_per_ch[0].artif_data): ', len(artif_per_ch[0].artif_data))
        return

    for ch in artif_per_ch:
        ch.corr_coef, ch.p_value = pearsonr(ch.artif_data_smoothed, mean_rwave)
    
    return artif_per_ch


def rms_amplitude(wave):

    """
    Function to calculate the Root Mean Square (RMS) amplitude

    Parameters
    ----------
    wave : np.ndarray
        Waveform data.
    
    Returns
    -------
    rms : float
        RMS amplitude of the waveform.
    """

    return np.sqrt(np.mean(np.square(wave)))


def minmax_amplitude(wave):

    """
    Calculate the amplitude between the maximum and minimum values of the waveform to see the full amplitude of the wave.
    
    Parameters
    ----------
    wave : np.ndarray
        Waveform data.
        
    Returns
    -------
    amplitude : float
        Amplitude between the maximum and minimum values of the waveform.
        
    """

    return np.max(wave) - np.min(wave)


def find_affected_by_amplitude_ratio(artif_per_ch: List):

    """
    Calculate the amplitude ratio for each channel.

    1. Calculate the mean peak-to-peak amplitude across all channels:

    First computes the peak-to-peak amplitude (simply difference between the maximum and minimum values) for each channel's smoothed artifact data.
    The mean of these peak-to-peak amplitudes is then calculated (ptp_all_comp_waves).
    
    2. Calculate the amplitude ratio for each channel:

    For each channel, calculate the ratio between the channel's peak-to-peak amplitude (minmax_amplitude(ch.artif_data_smoothed)) and 
    the mean peak-to-peak amplitude across all channels (ptp_all_comp_waves).
    
    This ratio is stored as an attribute (amplitude_ratio) in the Avg_artif object for each channel.
    

    Parameters
    ----------
    artif_per_ch : List
        List of channels with Avg_artif objects.
        
    Returns
    -------
    artif_per_ch : List
        List of channels with Avg_artif objects, now with assigned amplitude ratio.
        
    """

    #Find MEAN peak to peak amlitude over all waves we have:
    ptp_all_comp_waves = np.mean([minmax_amplitude(ch.artif_data_smoothed) for ch in artif_per_ch])    

    # Find amplitude ratio for each channel: 
    # dibvide the peak to peak amplitude of the channel by the MEAN peak to peak amplitude of all channels
    # So we see which of the channles have higher amplutude than the average over all channels
    for ch in artif_per_ch:
        #ch.amplitude_ratio = rms_amplitude(ch.artif_data_smoothed) / rms_all_comp_waves
        ch.amplitude_ratio = minmax_amplitude(ch.artif_data_smoothed) / ptp_all_comp_waves


    #TODO: tried to normalize here, but maybe we dont need that? cos without it will give raw results
    #That can be used for in-betweeen-data sets camprarison

    #scale all amplitude ratios from 0 to 1, where 1 is the highes over all channels 
    # max_amplitude_ratio = max([ch.amplitude_ratio for ch in artif_per_ch])
    # for ch in artif_per_ch:
    #     ch.amplitude_ratio = ch.amplitude_ratio / max_amplitude_ratio

    return artif_per_ch


def find_affected_by_similarity_score(artif_per_ch: List):

    """
    Combine the two metrics like: similarity_score = correlation * amplitude_ratio

    Parameters
    ----------
    artif_per_ch : List
        List of channels with Avg_artif objects.

    Returns
    -------
    artif_per_ch : List
        List of channels with Avg_artif objects, now with assigned similarity score.

    """

    for ch in artif_per_ch:
        ch.similarity_score = abs(ch.corr_coef) * abs(ch.amplitude_ratio)

    return artif_per_ch


def split_correlated_artifacts_into_3_groups(artif_per_ch):

    """
    Collect artif_per_ch into 3 lists - for plotting:
    - a third of all channels that are the most correlated with mean_rwave
    - a third of all channels that are the least correlated with mean_rwave
    - a third of all channels that are in the middle of the correlation with mean_rwave

    Parameters
    ----------
    artif_per_ch : List
        List of objects of class Avg_artif

    Returns
    -------
    artif_per_ch : List
        List of objects of class Avg_artif, ranked by correlation coefficient
    most_correlated : List
        List of objects of class Avg_artif that are the most correlated with mean_rwave
    least_correlated : List
        List of objects of class Avg_artif that are the least correlated with mean_rwave
    middle_correlated : List
        List of objects of class Avg_artif that are in the middle of the correlation with mean_rwave
    corr_val_of_last_least_correlated : float
        Correlation value of the last channel in the list of the least correlated channels
    corr_val_of_last_middle_correlated : float
        Correlation value of the last channel in the list of the middle correlated channels
    corr_val_of_last_most_correlated : float
        Correlation value of the last channel in the list of the most correlated channels


    """

    #sort by correlation coef. Take abs of the corr coeff, because the channels might be just flipped due to their location against magnetic field::
    artif_per_ch.sort(key=lambda x: abs(x.corr_coef), reverse=True)

    most_correlated = artif_per_ch[:int(len(artif_per_ch)/3)]
    least_correlated = artif_per_ch[-int(len(artif_per_ch)/3):]
    middle_correlated = artif_per_ch[int(len(artif_per_ch)/3):-int(len(artif_per_ch)/3)]

    #get correlation values of all most correlated channels:
    all_most_correlated = [abs(ch.corr_coef) for ch in most_correlated]
    all_middle_correlated = [abs(ch.corr_coef) for ch in middle_correlated]
    all_least_correlated = [abs(ch.corr_coef) for ch in least_correlated]

    #find the correlation value of the last channel in the list of the most correlated channels:
    # this is needed for plotting correlation values, to know where to put separation rectangles.
    corr_val_of_last_most_correlated = max(all_most_correlated)
    corr_val_of_last_middle_correlated = max(all_middle_correlated)
    corr_val_of_last_least_correlated = max(all_least_correlated)

    return most_correlated, middle_correlated, least_correlated, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated


def find_affected_over_mean(artif_per_ch: List, ecg_or_eog: str, params_internal: dict, thresh_lvl_peakfinder: float, m_or_g: str, norm_lvl: float, flip_data: bool, gaussian_sigma: float, artif_time_vector: np.ndarray):
    
    """
    1. Calculate average ECG epoch on the epochs from all channels. Check if average has a wave shape. 
    If no wave shape - no need to check for affected channels further.
    If it has - check further

    2. Set a threshold which defines a high amplitude of ECG event. (All above this threshold counted as potential ECG peak.)
    Threshold is the magnitude of the peak of the average ECG/EOG epoch multiplued by norm_lvl. 
    norl_lvl is chosen by user in config file
    
    3. Find all peaks above this threshold.
    Finding approach:

    - again, set t0 actual as the time point of the peak of an average artifact (over all channels)
    - again, set a window around t0_actual. this new window is defined by how long the wave of the artifact normally is. 
        The window is centered around t0 and for ECG it will be -0.-02 to 0.02s, for EOG it will be -0.1 to 0.1s.
    - find one main peak of the epoch for each channel which would be inside this window and closest to t0.
    - if this peaks magnitude is over the threshold - this channels is considered to be affected by ECG or EOG. Otherwise - not affected.
        (The epoch has to have a wave shape).

    4. Affected and non affected channels will be plotted and outputted as lists for adding tothe json on the next step.

    Parameters
    ----------
    artif_per_ch : List 
        list of Avg_artif objects
    ecg_or_eog : str
        'ECG' or 'EOG'
    params_internal : dict
        dictionary with parameters from setings_internal file
    thresh_lvl_peakfinder : float
        threshold for peakfinder. Defines the magnitude of the peak of the average ECG/EOG epoch multiplued by norm_lvl.
    m_or_g : str
        'mag' or 'grad'
    chs_by_lobe : dict
        dictionary with channels grouped by lobes
    norm_lvl : float
        defines the threshold for peakfinder. Threshold = mean overall artifact poch magnitude * norm_lvl
    flip_data : bool
        ifo for plotting. If data was flipped - only upper threshold will be shown on the plot, if not - both upper and lower
    gaussian_sigma : float
        sigma for gaussian smoothing
    artif_time_vector : np.ndarray
        time vector for the artifact epoch

    Returns
    -------
    affected_channels: List
        list of affected channels
    affected_derivs: List
        list of QC_derivative objects with figures for affected and not affected channels (smoothed and not soothed versions)
    bad_avg_str : str
        string about the average artifact: if it was not considered to be a wave shape
    avg_overall_obj : Avg_artif
        Avg_artif object with the average artifact

    """

    max_n_peaks_allowed_for_avg = params_internal['max_n_peaks_allowed_for_avg']
    window_size_for_mean_threshold_method = params_internal['window_size_for_mean_threshold_method']

    artif_per_ch_only_data = [ch.artif_data for ch in artif_per_ch] # USE NON SMOOTHED data. If needed, can be changed to smoothed data
    avg_overall=np.mean(artif_per_ch_only_data, axis=0) 
    # will show if there is ecg artifact present  on average. should have wave shape if yes. 
    # otherwise - it was not picked up/reconstructed correctly

    avg_overall_obj=Avg_artif(name='Mean_'+ecg_or_eog+'_overall',artif_data=avg_overall)

    #detect peaks and wave for the average overall artifact:
    avg_overall_obj.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
    avg_overall_obj.get_peaks_wave_smoothed(gaussian_sigma = gaussian_sigma, max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)

    affected_derivs=[]
    affected_channels = []

    if avg_overall_obj.wave_shape is True or avg_overall_obj.wave_shape_smoothed is True: #if the average ecg artifact is good - do steps 2 and 3:

        mean_magnitude_peak=np.max(avg_overall_obj.peak_magnitude)
        mean_ecg_loc_peak = avg_overall_obj.peak_loc[np.argmax(avg_overall_obj.peak_magnitude)]
        t0_actual=artif_time_vector[mean_ecg_loc_peak]
        #set t0_actual as the time of the peak of the average ecg artifact
        
        if avg_overall_obj.wave_shape_smoothed is not None: #if smoothed average and its peaks were also calculated:
            mean_magnitude_peak_smoothed=np.max(avg_overall_obj.peak_magnitude_smoothed)
            mean_ecg_loc_peak_smoothed = avg_overall_obj.peak_loc_smoothed[np.argmax(avg_overall_obj.peak_magnitude_smoothed)]
            t0_actual_smoothed=artif_time_vector[mean_ecg_loc_peak_smoothed]
        else:
            mean_magnitude_peak_smoothed=None
            t0_actual_smoothed=None
            
        
        tit, _ = get_tit_and_unit(m_or_g)
        
        if avg_overall_obj.wave_shape is True:
            avg_artif_description1 = tit+": (original) GOOD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg)."
        else:
            avg_artif_description1 = tit+": (original) BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."

        if avg_overall_obj.wave_shape_smoothed is True:
            avg_artif_description2 =  tit+": (smoothed) GOOD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude_smoothed)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg)."
        else:
            avg_artif_description2 = tit+": (smoothed) BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude_smoothed)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."

        avg_artif_description = avg_artif_description1 + "<p></p>" + avg_artif_description2

        print('___MEGqc___: ', avg_artif_description1)
        print('___MEGqc___: ', avg_artif_description2)

        bad_avg_str = avg_artif_description + ''

        # detect channels which are over the threshold defined by mean_magnitude_peak (average overall artifact) and norm_lvl (set in config):
        affected_channels, not_affected_channels, artifact_lvl, affected_channels_smoothed, not_affected_channels_smoothed, artifact_lvl_smoothed = detect_channels_above_norm(norm_lvl=norm_lvl, list_mean_artif_epochs=artif_per_ch, mean_magnitude_peak=mean_magnitude_peak, t=artif_time_vector, t0_actual=t0_actual, window_size_for_mean_threshold_method=window_size_for_mean_threshold_method, mean_magnitude_peak_smoothed=mean_magnitude_peak_smoothed, t0_actual_smoothed=t0_actual_smoothed)

    else: #if the average artifact is bad - end processing
        tit, _ = get_tit_and_unit(m_or_g)
        avg_artif_description = tit+": BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."
        bad_avg_str = avg_artif_description + tit+": "+ ecg_or_eog+ " signal detection/reconstruction did not produce reliable results. Affected channels can not be estimated."
        print('___MEGqc___: ', bad_avg_str)

    return affected_channels, affected_derivs, bad_avg_str, avg_overall_obj


#%%
def make_dict_global_ECG_EOG(channels_ranked: List, use_method: str):
    """
    Make a dictionary for the global part of simple metrics for ECG/EOG artifacts.
    For ECG/EOG no local metrics are calculated, so global is the only one.
    
    Parameters
    ----------
    channels_ranked : List
        List of all affected channels
    use_method : str
        Method used for detection of ECG/EOG artifacts: correlation_recorded, correlation_reconstructed or mean_threshold.
        Depending in this the dictionary will have difefrent structure and descriptions.
        
    Returns
    -------
    metric_global_content : dict
        Dictionary with simple metrics for ECG/EOG artifacts.
   
    """

    # sort all_affected_channels by main_peak_magnitude:
    if use_method == 'mean_threshold':
        if channels_ranked:
            all_affected_channels_sorted = sorted(channels_ranked, key=lambda ch: ch.main_peak_magnitude, reverse=True)
            affected_chs = {ch.name: ch.main_peak_magnitude for ch in all_affected_channels_sorted}
            metric_global_content = {'details':  affected_chs}
        else:
            metric_global_content = {'details':  None}
    elif use_method == 'correlation_recorded' or use_method == 'correlation_reconstructed':
        all_affected_channels_sorted = sorted(channels_ranked, key=lambda ch: abs(ch.corr_coef), reverse=True)
        affected_chs = {ch.name: [ch.corr_coef, ch.p_value] for ch in all_affected_channels_sorted}
        metric_global_content = {'details':  affected_chs}
    else:
        raise ValueError('Unknown method_used: ', use_method)

    return metric_global_content


def make_simple_metric_ECG_EOG(channels_ranked: dict, m_or_g_chosen: List, ecg_or_eog: str, avg_artif_str: dict, use_method: str):
    
    """
    Make simple metric for ECG/EOG artifacts as a dictionary, which will further be converted into json file.
    
    Parameters
    ----------
    channels_ranked : dict
        Dictionary with lists of channels.
    m_or_g_chosen : List
        List of channel types chosen for the analysis. 
    ecg_or_eog : str
        String 'ecg' or 'eog' depending on the artifact type.
    avg_artif_str : dict
        Dict with strings with info about the ECG/EOG channel and average artifact.
    use_method : str
        Method used for detection of ECG/EOG artifacts: correlation_recorded, correlation_reconstructed or mean_threshold.
        Depending in this the dictionary will have difefrent structure and descriptions.
        
    Returns
    -------
    simple_metric : dict
        Dictionary with simple metrics for ECG/EOG artifacts.
        
    """

    metric_global_name = 'all_channels_ranked_by_'+ecg_or_eog+'_contamination_level'
    metric_global_content = {'mag': None, 'grad': None}

    if use_method == 'mean_threshold':
        metric_global_description = 'Here presented the channels with average (over '+ecg_or_eog+' epochs of this channel) ' +ecg_or_eog+ ' artifact above the threshold. Channels are listed here in order from the highest to lowest artifact amplitude. Non affected channels are not listed. Threshld is defined as average '+ecg_or_eog+' artifact peak magnitude over al channels * norm_lvl. norm_lvl is defined in the config file. Channels are presented in the form: ch.name: ch.main_peak_magnitude.'
    elif use_method == 'correlation_recorded' or use_method == 'correlation_reconstructed':
        metric_global_description = 'Here the channels are ranked by correlation coefficient between the channel and the averaged '+ecg_or_eog+' channel (recorded or reconstructed). Channels are listed here in order from the highest to lowest correlation coefficient. Channels are presented in the form: ch.name: [ch.corr_coef, ch.p_value]. Sign of the correlation value is kept to reflect the position of the channel toward the magnetic fild omly, it does not reflect the level of contamination (absolute value should be considered for this).'

    for m_or_g in m_or_g_chosen:
        if channels_ranked[m_or_g]: #if there are affected channels for this channel type
            metric_global_content[m_or_g]= make_dict_global_ECG_EOG(channels_ranked[m_or_g], use_method)
        else:
            metric_global_content[m_or_g]= avg_artif_str[m_or_g]

    if use_method == 'mean_threshold':
        measurement_units = True
    else:
        measurement_units = False

    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], display_only_global=True, measurement_units = measurement_units)

    return simple_metric

import re


def get_ECG_data_choose_method(raw: mne.io.Raw, ecg_params: dict, orig_meg_system):

    """
    Choose the method of finding affected channels based on the presense and quality of ECG channel.

    Options:
    - Channel present and good: correlation with ECG channel
    - Channel present and bad or missing:correlation with reconstructed channel
    - Use mean ECG artifact as threshold (currrently not used)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    ecg_params : dict
        Dictionary with ECG parameters originating from config file.
    
        
    Returns
    -------
    use_method : str
        String with the method chosen for the analysis.
    ecg_str : str
        String with info about the ECG channel presense.
    noisy_ch_derivs : List
        List of QC_derivative objects with plot of the ECG channel
    ecg_data:
        ECG channel data.
    event_indexes:
        Indexes of the ECG events.

    """
    picks_ECG = mne.pick_types(raw.info, ecg=True)

    if len(picks_ECG) == 0:
        pattern = re.compile(r'EKG|ECG', re.IGNORECASE)
        picks_ECG = [i for i, ch_name in enumerate(raw.info['ch_names']) if pattern.search(ch_name)]

    ecg_ch_name = [raw.info['chs'][name]['ch_name'] for name in picks_ECG]

    print ('___MEGqc___: ECG channel names:', ecg_ch_name)

    if len(ecg_ch_name)>=1: #ecg channel present

        if len(ecg_ch_name)>1: #more than 1 ecg channel present
            ecg_str = 'More than 1 ECG channel found. The first one is used to identify hearbeats. '

        ecg_ch_name = ecg_ch_name[0]

        bad_ecg_eog, ecg_data, event_indexes, ecg_eval_str = detect_noisy_ecg(raw, ecg_ch_name,  ecg_or_eog = 'ECG', n_breaks_bursts_allowed_per_10min = ecg_params['n_breaks_bursts_allowed_per_10min'], allowed_range_of_peaks_stds = ecg_params['allowed_range_of_peaks_stds'], height_multiplier = ecg_params['height_multiplier'])

        if orig_meg_system == 'EEG':
            bad_ecg_eog[ecg_ch_name] = 'good'

        if bad_ecg_eog[ecg_ch_name] == 'bad': #ecg channel present but noisy:
            ecg_str = 'ECG channel data is too noisy, cardio artifacts were reconstructed. ECG channel was dropped from the analysis. Consider checking the quality of ECG channel on your recording device. \n'
            print('___MEGqc___: ', ecg_str)
            raw.drop_channels(ecg_ch_name)
            use_method = 'correlation_reconstructed'
            # TODO: here in case the recorded ECG was bad - we try to reconstruct. Think of this logic.
            # It might be better to not even try, because reconstructed is rarely better than recorded.
            # However we might have a broken ecg ch - then there is a chance that reconstruction works somewhat better.


        elif bad_ecg_eog[ecg_ch_name] == 'good': #ecg channel present and good - use it
            ecg_str = ecg_ch_name + ' is used to identify hearbeats. \n'
            use_method = 'correlation_recorded'

    else: #no ecg channel present

        ecg_str = 'No ECG channel found. The signal is reconstructed based on magnetometers data. \n'
        use_method = 'correlation_reconstructed'

        # _, _, _, ecg_data = mne.preprocessing.find_ecg_events(raw, return_ecg=True)
        # # here the RECONSTRUCTED ecg data will be outputted (based on magnetometers), and only if u set return_ecg=True and no real ec channel present).
        # ecg_data = ecg_data[0]

    if use_method == 'correlation_reconstructed':
        ecg_ch_name, bad_ecg_eog, ecg_data, event_indexes, ecg_eval_str = reconstruct_ecg_and_check(raw, ecg_params['n_breaks_bursts_allowed_per_10min'], ecg_params['allowed_range_of_peaks_stds'], ecg_params['height_multiplier'])
        
        if bad_ecg_eog[ecg_ch_name] == 'bad':
            use_method = 'reconstructed-bad'
            #pass here, cos we dont need to do anything with the data if it is bad
            
    print('___MEGqc___: ', ecg_str)

    ecg_str_total = ecg_str + ecg_eval_str
    #Replace all \n with <br> for the html report:
    ecg_str_total = ecg_str_total.replace('\n', '<br>')

    return use_method, ecg_str_total, ecg_ch_name, ecg_data, event_indexes

def get_EOG_data(raw: mne.io.Raw, orig_meg_system):

    """
    Find if the EOG channel is present anfd get its data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    
        
    Returns
    -------
    eog_str : str
        String with info about the EOG channel presense.
    eog_data:
        EOG channel data.
    event_indexes:
        Indexes of the ECG events.
    eog_ch_name: str
        Name of the EOG channel.

    """

    
    # Find EOG events in your data and get the name of the EOG channel

    # Select the EOG channels
    eog_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True)

    if len(eog_channels) == 0:
        pattern = re.compile(r'EOG|ROC|LOC', re.IGNORECASE)
        eog_channels = [i for i, ch_name in enumerate(raw.info['ch_names']) if pattern.search(ch_name)]



    # Get the names of the EOG channels
    eog_channel_names = [raw.ch_names[ch] for ch in eog_channels]

    print('___MEGqc___: EOG channel names:', eog_channel_names)


    #TODO: WHY AM I DOING THIS CHECK??
    try:
        eog_events, eog_channel_names = find_eog_events_custom(raw, eog_channel_names)
        #eog_events = mne.preprocessing.find_eog_events(raw)
        #eog_events_times  = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq']

        #even if 2 EOG channels are present, MNE can only detect blinks!
    except:
        noisy_ch_derivs, eog_data, event_indexes = [], [], []
        eog_str = 'No EOG channels found is this data set - EOG artifacts can not be detected.'
        print('___MEGqc___: ', eog_str)
        return eog_str, noisy_ch_derivs, eog_data, event_indexes

    # Get the data of the EOG channel as an array. MNE only sees blinks, not saccades.
    eog_data = raw.get_data(picks=eog_channel_names)

    eog_str = ', '.join(eog_channel_names)+' was used to identify eye blinks. '

    height = np.mean(eog_data) + 1 * np.std(eog_data)
    fs=raw.info['sfreq']

    event_indexes_all = []
    for ch in eog_data:
        event_indexes, _ = find_peaks(ch, height=height, distance=round(0.5 * fs)) #assume there are no peaks within 0.5 seconds from each other.
        event_indexes_all += [event_indexes.tolist()]

    return eog_str, eog_data, event_indexes_all, eog_channel_names

def find_eog_events_custom(raw, eog_channel_names):
    # try with each EOG channels
    for eog_ch in eog_channel_names:
        try:
            print(f"Extracting EOG events from : {eog_ch}")
            eog_events = mne.preprocessing.find_eog_events(raw, ch_name=eog_ch)
            print(f"✓ Found  {len(eog_events)} EOG events with {eog_ch}")
            return eog_events, eog_ch

        except Exception as e:
            print(f"✗ Error with {eog_ch}: {e}")
            continue
    raise ValueError("No EOG events found in any channel")


def reconstruct_ecg_and_check(raw: mne.io.Raw, n_breaks_bursts_allowed_per_10min: int, allowed_range_of_peaks_stds: float, height_multiplier: float):

    """
    Reconstruct ECG channel based on magnetometers data and check if it is good.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    n_breaks_bursts_allowed_per_10min : int
        Number of breaks and bursts allowed per 10 minutes.
    allowed_range_of_peaks_stds : float
        Allowed range of peaks in standard deviations.
    height_multiplier : float
        Multiplier for the height of the peaks.
    
    
    Returns
    -------
    bad_ecg_eog: dict
        Dictionary with info about the ECG channel quality.
    ecg_data:
        ECG channel data.
    peaks: np.ndarray
        Peaks of the ECG channel.
    ecg_eval_str: str
        String with info about the ECG channel quality.
    ecg_ch: str
        Name of the ECG channel: 'Reconstructed_ECG_ch'.
        
    """

    ecg_ch = 'Reconstructed_ECG_ch'
    sfreq = raw.info['sfreq']

    _, _, _, ecg_data = mne.preprocessing.find_ecg_events(raw, return_ecg=True)
    # here the RECONSTRUCTED ecg data will be outputted (based on magnetometers), 
    # and only if u set return_ecg=True and no real ecg channel present).
    ecg_data = ecg_data[0]
    
    ecg_eval, peaks = check_3_conditions(ecg_data, sfreq, 'ECG', n_breaks_bursts_allowed_per_10min, allowed_range_of_peaks_stds, height_multiplier)
    print(f'___MEGqc___: {ecg_ch} satisfied conditions for a good channel: ', ecg_eval)

    bad_ecg_eog = {}
    #If all values in disct are true:
    if all(value == True for value in ecg_eval.values()):
        bad_ecg_eog[ecg_ch] = 'good'
    else:
        bad_ecg_eog[ecg_ch] = 'bad'

    ecg_eval_str = 'Overall ' + bad_ecg_eog[ecg_ch] + ' '  + ecg_ch + ' : \n - Peaks have similar amplitude: ' + str(ecg_eval["similar_ampl"]) + ' \n - No breaks (too long distances between peaks): ' + str(ecg_eval["no_breaks"]) + ' \n - No bursts (too short distances between peaks): ' + str(ecg_eval["no_bursts"]) + '\n'
    print(f'___MEGqc___: ', ecg_eval_str)

    return ecg_ch, bad_ecg_eog, ecg_data, peaks, ecg_eval_str



def check_mean_wave(ecg_data: np.ndarray, ecg_or_eog: str, event_indexes: np.ndarray, tmin: float, tmax: float, sfreq: int, params_internal: dict, thresh_lvl_peakfinder: float):

    """
    Calculate mean R wave based on either real ECG channel data or on reconstructed data (depends on the method used) 
    and check if it has an R wave shape.
    Plot Rwave with peaks.
    
    Parameters
    ----------
    use_method : str
        String with the method chosen for the analysis.
    ecg_data: np.ndarray
        ECG channel data. If it s empty, it will be reconstructed here
    event_indexes:
        Indexes of the ECG events.
    tmin : float
        Epoch start time before event (negative value)
    tmax : float
        Epoch end time after event (positive value)
    sfreq : float
        Sampling frequency
    params_internal : dict
        Dictionary with internal parameters originating from settings_internal.
    thresh_lvl_peakfinder : float
        Threshold level for peakfinder function.
    
    Returns
    -------
    mean_rwave_obj.wave_shape: bool
        True if the mean R wave shape is good, False if not.
    ecg_str_checked: str
        String with info about the ECG channel quality (after checking)
    mean_rwave: np.array
        Mean R wave (1 dimentional).
    mean_rwave_time: np.array
        Time vector for the mean R wave.
    
    """

    max_n_peaks_allowed_for_avg=params_internal['max_n_peaks_allowed_for_avg']

    #Calculate average over the whole reconstrcted channels and check if it has an R wave shape:

    if len(event_indexes) <1:
        ecg_str_checked = 'No expected wave shape was detected in the averaged event of '+ecg_or_eog+' channel.'
        print('___MEGqc___: ', ecg_str_checked)

        return False, ecg_str_checked, np.empty((0, 0)), np.empty((0, 0))

    mean_rwave = find_mean_rwave_blink(ecg_data, event_indexes, tmin, tmax, sfreq)  

    mean_rwave_obj=Avg_artif(name='Mean_rwave', artif_data=mean_rwave)

    #detect peaks and wave for the average overall artifact:
    mean_rwave_obj.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)

    if mean_rwave_obj.wave_shape is True:
        ecg_str_checked = 'Mean event of '+ecg_or_eog+' channel has expected shape.'
        print('___MEGqc___: ', ecg_str_checked)
    else:
        ecg_str_checked = 'Mean events of '+ecg_or_eog+' channel does not have expected shape. Artifact detection was not performed.'
        print('___MEGqc___: ', ecg_str_checked)

    if mean_rwave.size > 0:
        mean_rwave_time = np.linspace(tmin, tmax, len(mean_rwave))
    else:
        #empty array
        mean_rwave_time = np.empty((0, 0))

    return mean_rwave_obj.wave_shape, ecg_str_checked, mean_rwave, mean_rwave_time


# ___________ Functions for alignment of ECG with meg channels:

def find_t0_mean(ch_data: Union[List, np.ndarray]):

    """
    Find all t0 options for the mean ECG wave.

    Parameters
    ----------
    ch_data : np.ndarray or list
        averaged ECG channel data.
    
    Returns
    -------
    potential_t0: List
        List with all potential t0 options for the mean ECG wave.
        Will be used to get all possible option for shifting the ECG wave to align it with the MEG channels.
    """

    
    prominence=(max(ch_data) - min(ch_data)) / 8
    #run peak detection:
    peaks_pos_loc, _ = find_peaks(ch_data, prominence=prominence)
    peaks_neg_loc, _ = find_peaks(-ch_data, prominence=prominence)
    
    #put all these together and sort by which comes first:
    if len(peaks_pos_loc) == 0:
        peaks_pos_loc = [None]
    if len(peaks_neg_loc) == 0:
        peaks_neg_loc = [None]

    potential_t0 = list(peaks_pos_loc) + list(peaks_neg_loc)
    potential_t0 = [item for item in potential_t0 if item is not None]
    potential_t0 = sorted(potential_t0)

    if len(potential_t0) == 0: #if no peaks were found - just take the max of ch_data:
        potential_t0 = [np.argmax(ch_data)]

    return potential_t0

def find_t0_highest(ch_data: np.ndarray):

    """
    Find the t0 as the largest in absolute amplitude peak of the ECG artifact on ONE channel.
    This function is looped over all channels to find the t0 for all channels.

    Parameters
    ----------
    ch_data : np.ndarray or list
        the data for average ECG artifact on meg channel.

    Returns
    -------
    t0: int
        t0 for the channel (index, not the seconds!).
    """
    
    prominence=(max(ch_data) - min(ch_data)) / 8
    #run peak detection:
    peaks_pos_loc, _ = find_peaks(ch_data, prominence=prominence)
    if len(peaks_pos_loc) == 0:
        peaks_pos_loc = None
    else:
        peaks_pos_magn = ch_data[peaks_pos_loc]
        # find peak with highest magnitude:
        max_peak_pos_loc = peaks_pos_loc[np.argmax(peaks_pos_magn)]


    peaks_neg_loc, _ = find_peaks(-ch_data, prominence=prominence)
    if len(peaks_neg_loc) == 0:
        peaks_neg_loc = None
    else:
        peaks_neg_magn = ch_data[peaks_neg_loc]
        min_peak_neg_loc = peaks_neg_loc[np.argmin(peaks_neg_magn)]

    if peaks_pos_loc is None and peaks_neg_loc is None:
        t0 = None
    elif peaks_pos_loc is None:
        t0 = min_peak_neg_loc
    elif peaks_neg_loc is None:
        t0 = max_peak_pos_loc
    else:
        #choose the one with highest absolute magnitude:
        if abs(ch_data[max_peak_pos_loc]) > abs(ch_data[min_peak_neg_loc]):
            t0 = max_peak_pos_loc
        else:
            t0 = min_peak_neg_loc

    return t0


def find_t0_channels(artif_per_ch: List, tmin: float, tmax: float):

    """ 
    Run peak detection on all channels and find the 10 channels with the highest peaks.
    Then find the t0 for each of these channels and take the mean of these t0s as the final t0.
    It is also possible that t0 of these 10 channels dont concentrate around the same point, but around 1 points.
    For this reason theer is a check on how far the time points are from each other. If over 0.01, then they probabably 
    concentrate around 2 points and then just the 1 highes magnitude (not the mean) as taken as the final t0.

    Parameters
    ----------
    artif_per_ch : List
        List of Avg_artif objects, one for each channel.
    tmin : float
        Start time of epoch.
    tmax : float
        End time of epoch.  

    Returns
    -------
    t0_channels : int
        The final t0 (index, not the seconds!) that will be used for all channels as a refernce point. 
        To this point the Average ECG will be aligned.

    """
    
    chosen_t0 = []
    chosen_t0_magnitudes = []

    for ch in artif_per_ch:
        data = ch.artif_data_smoothed
        
        #potential_t0 = find_t0_1ch(data)
        ch_t0 = find_t0_highest(data)
        
        if ch_t0 is not None:
            chosen_t0.append(ch_t0)
            chosen_t0_magnitudes.append(abs(data[ch_t0]))
            #take absolute value of magnitudes because we don't care if it's positive or negative

    #CHECK IF ABS IS ACTUALLY BETTER THAN NOT ABS

    #find the 10 channels with the highest magnitudes:
    chosen_t0_magnitudes = np.array(chosen_t0_magnitudes)
    chosen_t0 = np.array(chosen_t0)
    chosen_t0_sorted = chosen_t0[np.argsort(chosen_t0_magnitudes)]
    chosen_t0_sorted = chosen_t0_sorted[-10:]


    #find the distance between 10 chosen peaks:
    t = np.linspace(tmin, tmax, len(artif_per_ch[0].artif_data_smoothed))
    time_max = t[np.max(chosen_t0_sorted)]
    time_min = t[np.min(chosen_t0_sorted)]

    #if the values of 10 highest peaks are close together, take the mean of them:
    if abs(time_max - time_min) < 0.01:
        #find the average location of the highest peak over the 10 channels:
        t0_channels = int(np.mean(chosen_t0_sorted))
    else: 
        #if not close - this is the rare case when half of channels have first of phase of r wave stronger 
        #and second half has second part of r wave stronger. And these 2 are almost the same amplitude.
        #so if we would take mean - they will cancel out and we will get the middle lowest point of rwave instead of a peak.
        # so we take the highest peak instead:
        t0_channels = int(np.max(chosen_t0_sorted))

    return t0_channels


def shift_mean_wave(mean_rwave: np.ndarray, ind_t0_channels: int, ind_t0_mean: int):

    """
    Shifts the mean ECG wave to align with the ECG artifacts found on meg channels.
    np.roll is used to shift. meaning: for example when shift to the right: 
    the end of array will be attached in the beginning to the left.
    Usually ok, but it may cause issues if the array was originally very short or very strongly shifted, 
    then it may split the wave shape in half and the shifted wave will look completely unusable.
    Therefore, dont limit tmin and tmax too tight in config file (default is good). 
    Or come up with other way insted of np.roll.

    Parameters
    ----------
    mean_rwave : np.ndarray
        The mean ECG wave, not shifted yet.
    t0_channels : int
        The location of the peak of ECG artifact on the MEG channels. (This is not seconds! This is index).
    t0_mean : int
        The location of the peak of the mean ECG wave on the ECG channel. (This is not seconds! This is index).
    
    Returns
    -------
    mean_rwave_shifted : np.ndarray
        The mean ECG wave shifted to align with the ECG artifacts found on meg channels.
    
    """

    t0_shift = ind_t0_channels - ind_t0_mean
    mean_rwave_shifted = np.roll(mean_rwave, t0_shift)

    return mean_rwave_shifted


def align_mean_rwave(mean_rwave: np.ndarray, artif_per_ch: List, tmin: float, tmax: float):

    """ Aligns the mean ECG wave with the ECG artifacts found on meg channels.
    1) The average highest point of 10 most prominent meg channels is used as refernce.
    The ECG artifact is shifted multiple times, 
    2) each time the correlation of ECG channel with
    meg channel artofacts is calculated, then the aligment versiob which shows highes correlation 
    is chosen as final.
    Part 1) is done inside this function, part 2) is done inside the main ECG_meg_qc function, 
    but they are both the part of one algorithm.

    Parameters
    ----------
    mean_rwave : np.array
        The mean ECG wave (resulting from recorded or recosntructed ECG signal).
    artif_per_ch : List
        List of Avg_artif objects, each of which contains the ECG artifact from one MEG channel.
    tmin : float
        The start time of the ECG artifact, set in config
    tmax : float
        The end time of the ECG artifact, set in config
    
    Returns
    -------
    mean_rwave_shifted_variations : List
        List of arrays. Every array is a variation of he mean ECG wave shifted 
        to align with the ECG artifacts found on meg channels.
    
    """

    t0_channels = find_t0_channels(artif_per_ch, tmin, tmax)

    t = np.linspace(tmin, tmax, len(mean_rwave))
    
    t0_mean = find_t0_mean(mean_rwave)
    

    mean_rwave_shifted_variations = []
    for t0_m in t0_mean:
        mean_rwave_shifted_variations.append(shift_mean_wave(mean_rwave, t0_channels, t0_m))
    

    #plot every variation, useful for debugging

    # t0_time_channels = t[t0_channels] #for plotting
    # t0_time_mean = t[t0_mean] #for plottting

    # for i, mean_rwave_shifted in enumerate(mean_rwave_shifted_variations):
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter (x=t, y=mean_rwave_shifted, mode='lines', name='mean_rwave_shifted'))
    #     fig.add_trace(go.Scatter (x=t, y=mean_rwave, mode='lines', name='mean_rwave'))
    #     fig.add_vline(x=t0_time_channels, line_dash="dash", line_color="red", name='t0_channels')
    #     fig.add_vline(x=t0_time_mean, line_dash="dash", line_color="blue", name='t0_mean')
    #     fig.update_layout(title='Mean R wave shifted to align with the ECG artifacts found on meg channels')
    #     fig.show()

    return mean_rwave_shifted_variations


#%%
def ECG_meg_qc(ecg_params: dict, ecg_params_internal: dict, data_path:str, channels: List, chs_by_lobe_orig: dict, m_or_g_chosen: List, orig_meg_system):

    """
    Main ECG function. Calculates average ECG artifact and finds affected channels.
    
    Parameters
    ----------
    ecg_params : dict
        Dictionary with ECG parameters originating from config file.
    ecg_params_internal : dict
        Dictionary with ECG parameters originating from config file preset, not to be changed by user.
    raw : mne.io.Raw
        Raw data.
    channels : dict
        Dictionary with listds of channels for each channel type (mag and grad).
    chs_by_lobe : dict
        Dictionary with lists of channels by lobe.
    m_or_g_chosen : List
        List of channel types chosen for the analysis.
        
    Returns
    -------
    ecg_derivs : List
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_ECG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    ecg_str : str
        String with information about ECG channel used in the final report.
    avg_objects_ecg : List
        List of Avg_artif objects, each of which contains the ECG artifact from one MEG channel.
        

    """

    # Load data
    raw, shielding_str, meg_system = load_data(data_path)

    chs_by_lobe = copy.deepcopy(chs_by_lobe_orig) 
    #in case we will change this variable in any way. If not copied it might introduce errors in parallel processing. 
    # This variable is used in all modules

    sfreq=raw.info['sfreq']
    tmin=ecg_params_internal['ecg_epoch_tmin']
    tmax=ecg_params_internal['ecg_epoch_tmax']

    #WROTE THIS BEFORE, BUT ACTUALLY NEED TO CHECK IF IT S STILL TRUE OR THE PROBLEM WAS SOLVED FOR THRESHOLD METHOD:
    #tmin, tmax can be anything from -0.1/0.1 to -0.04/0.04. for CORRELATION method. But if we do mean and threshold - time best has to be -0.04/0.04. 
    # For this method number of peaks in particular time frame is calculated and based on that good/bad rwave is decided.
    norm_lvl=ecg_params['norm_lvl']
    gaussian_sigma=ecg_params['gaussian_sigma']
    thresh_lvl_peakfinder=ecg_params['thresh_lvl_peakfinder']

    ecg_derivs = []
    use_method, ecg_str, ecg_ch_name, ecg_data, event_indexes = get_ECG_data_choose_method(raw, ecg_params, orig_meg_system)

    n_events = len(event_indexes)
    minutes_in_data = len(ecg_data) / sfreq / 60
    events_rate_per_min = round(n_events / minutes_in_data, 1)
    print('___MEGqc___: ', 'ECG events detected: ', n_events)
    n_events_str = '<br><br>ECG events detected: ' + str(n_events)
    print('___MEGqc___: ', 'Heart beats per minute: ', events_rate_per_min)
    n_events_str += '<br>Heart beats per minute: ' + str(events_rate_per_min)

    
    if use_method == 'reconstructed-bad': 
        # data was reconstricted and is bad - dont continue
        simple_metric_ECG = {'description': ecg_str}
        ecg_str += n_events_str

        #Still Create df to be exported to tsv, with some missing data:
        #(We dont calculate mean ECG when the data is bad - cant find the properly looking mean wave)

        ecg_ch_df = pd.DataFrame({
            ecg_ch_name: ecg_data,
            'event_indexes': event_indexes.tolist() + [None] * (len(ecg_data) - len(event_indexes)),
            'fs': [raw.info['sfreq']] + [None] * (len(ecg_data) - 1),
            'mean_rwave': [None] * len(ecg_data),
            'mean_rwave_time': [None] * len(ecg_data),
            'recorded_or_reconstructed': [use_method] + [None] * (len(ecg_data) - 1),
            'mean_rwave_shifted' : [None] * len(ecg_data),
            'n_events' : [n_events] + [None] * (len(ecg_data) - 1),
            'events_rate_per_min' : [events_rate_per_min] + [None] * (len(ecg_data) - 1)})

        ecg_derivs += [QC_derivative(content=ecg_ch_df, name='ECGchannel', content_type = 'df')]

        return ecg_derivs, simple_metric_ECG, ecg_str, []

    mean_good, ecg_str_checked, mean_rwave, mean_rwave_time = check_mean_wave(ecg_data, 'ECG', event_indexes, tmin, tmax, sfreq, ecg_params_internal, thresh_lvl_peakfinder)

    ecg_str += ecg_str_checked + n_events_str

    if mean_good is False: 
        #mean ECG wave calculsted but bad - dont continue
        simple_metric_ECG = {'description': ecg_str}

        #Still Create df to be exported to tsv, except mean_rwave_shifted:

        ecg_ch_df = pd.DataFrame({
            ecg_ch_name: ecg_data,
            'event_indexes': event_indexes.tolist() + [None] * (len(ecg_data) - len(event_indexes)),
            'fs': [raw.info['sfreq']] + [None] * (len(ecg_data) - 1),
            'mean_rwave': mean_rwave.tolist() + [None] * (len(ecg_data) - len(mean_rwave)),
            'mean_rwave_time': mean_rwave_time.tolist() + [None] * (len(ecg_data) - len(mean_rwave_time)),
            'recorded_or_reconstructed': [use_method] + [None] * (len(ecg_data) - 1),
            'mean_rwave_shifted' : [None] * len(ecg_data),
            'n_events' : [n_events] + [None] * (len(ecg_data) - 1),
            'events_rate_per_min' : [events_rate_per_min] + [None] * (len(ecg_data) - 1)})

        ecg_derivs += [QC_derivative(content=ecg_ch_df, name='ECGchannel', content_type = 'df')]

        return ecg_derivs, simple_metric_ECG, ecg_str, []

    
    affected_channels={}
    best_affected_channels={}
    bad_avg_str = {}
    avg_objects_ecg =[]

    for m_or_g  in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        # ecg_derivs += plot_ecg_eog_mne(ecg_epochs, m_or_g, tmin, tmax)

        artif_per_ch = calculate_artifacts_on_channels(ecg_epochs, channels[m_or_g], chs_by_lobe=chs_by_lobe[m_or_g], thresh_lvl_peakfinder=thresh_lvl_peakfinder, tmin=tmin, tmax=tmax, params_internal=ecg_params_internal, gaussian_sigma=gaussian_sigma)

        #use_method = 'mean_threshold' 

        #2 options:
        #1. find channels with peaks above threshold defined by average over all channels+multiplier set by user
        #2. find channels that have highest Pearson correlation with average R wave shape (if the ECG channel is present)

        if use_method == 'mean_threshold':
            artif_per_ch, artif_time_vector = flip_channels(artif_per_ch, tmin, tmax, sfreq, ecg_params_internal)
            affected_channels[m_or_g], affected_derivs, bad_avg_str[m_or_g], avg_overall_obj = find_affected_over_mean(artif_per_ch, 'ECG', ecg_params_internal, thresh_lvl_peakfinder, m_or_g=m_or_g, norm_lvl=norm_lvl, flip_data=True, gaussian_sigma=gaussian_sigma, artif_time_vector=artif_time_vector)


        elif use_method == 'correlation_recorded' or use_method == 'correlation_reconstructed':

            #align the mean ECG wave with the ECG artifacts found on meg channels:
            #Find the correlation value between all variations of alignment the mean ECG wave with the ECG artifacts found on meg channels.
            #The alignment with the highest correlation is chosen as the final one.
            #Our target is best_affected_channels[m_or_g] - the channels that are most correlated with the mean ECG wave, after all variations of alignemnt were checked.
            #We also get best_mean_corr and best_mean_shifted - mostely useful if we wanna plot them.

            mean_rwave_shifted_variations = align_mean_rwave(mean_rwave, artif_per_ch, tmin, tmax)
            
            #preassign default value:
            best_mean_corr = 0
            
            for mean_shifted in mean_rwave_shifted_variations:
                affected_channels[m_or_g] = find_affected_by_correlation(mean_shifted, artif_per_ch)


                #collect all correlation values for all channels:
                all_corr_values = [abs(ch.corr_coef) for ch in affected_channels[m_or_g]]
                #get 10 highest correlations:
                all_corr_values.sort(reverse=True)
                print(all_corr_values)
                mean_corr = np.nanmean(all_corr_values[0:10]) #[0:10]
                # here use nanmean, not just nan, because in rare cases 
                # pearson calculates nan which can mess up all further calculations.
                # this can happen if all values of the mave are the same, flat channel 
                # or if they contain naan values.
                
                #if mean corr is better than the previous one - save it:
                if mean_corr > best_mean_corr:
                    best_mean_corr = mean_corr
                    best_mean_rwave_shifted = mean_shifted
                    best_affected_channels[m_or_g] = copy.deepcopy(affected_channels[m_or_g])

            # Now that we found best correlation values, next step is to calculate magnitude ratios of every channel
            # Then, calculate similarity value comprised of correlation and magnitude ratio:

            best_affected_channels[m_or_g] = find_affected_by_amplitude_ratio(best_affected_channels[m_or_g])

            best_affected_channels[m_or_g] = find_affected_by_similarity_score(best_affected_channels[m_or_g])


            bad_avg_str[m_or_g] = ''
            avg_overall_obj = None

        else:
            raise ValueError('use_method should be either mean_threshold or correlation_recorded or correlation_reconstructed')
        

        #Create FULL df to be exported to tsv:
        ecg_ch_df = pd.DataFrame({
            ecg_ch_name: ecg_data,
            'event_indexes': event_indexes.tolist() + [None] * (len(ecg_data) - len(event_indexes)),
            'fs': [raw.info['sfreq']] + [None] * (len(ecg_data) - 1),
            'mean_rwave': mean_rwave.tolist() + [None] * (len(ecg_data) - len(mean_rwave)),
            'mean_rwave_time': mean_rwave_time.tolist() + [None] * (len(ecg_data) - len(mean_rwave_time)),
            'recorded_or_reconstructed': [use_method] + [None] * (len(ecg_data) - 1),
            'mean_rwave_shifted' : best_mean_rwave_shifted.tolist() + [None] * (len(ecg_data) - len(mean_rwave)),
            'n_events' : [n_events] + [None] * (len(ecg_data) - 1),
            'events_rate_per_min' : [events_rate_per_min] + [None] * (len(ecg_data) - 1)})

        ecg_derivs += [QC_derivative(content=ecg_ch_df, name='ECGchannel', content_type = 'df')]

        #higher thresh_lvl_peakfinder - more peaks will be found on the eog artifact for both separate channels and average overall. As a result, average overll may change completely, since it is centered around the peaks of 5 most prominent channels.
        avg_objects_ecg.append(avg_overall_obj)


    simple_metric_ECG = make_simple_metric_ECG_EOG(best_affected_channels, m_or_g_chosen, 'ECG', bad_avg_str, use_method)

    #Extract chs_by_lobe into a data frame
    artif_time_vector = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round
    #TODO: above we always use tmin, tmax, sfreq to create time vector in every fuction. here it s done again, maybe change above?

    for m_or_g  in m_or_g_chosen:
        for lobe, lobe_channels in chs_by_lobe[m_or_g].items():
            for lobe_ch in lobe_channels:
                lobe_ch.add_ecg_info(best_affected_channels[m_or_g], artif_time_vector)


    ecg_csv_deriv = chs_dict_to_csv(chs_by_lobe,  file_name_prefix = 'ECGs')

    ecg_derivs += ecg_csv_deriv

    return ecg_derivs, simple_metric_ECG, ecg_str, avg_objects_ecg


#%%
def EOG_meg_qc(eog_params: dict, eog_params_internal: dict, data_path: str, channels: dict, chs_by_lobe_orig: dict, m_or_g_chosen: List, orig_meg_system):
    
    """
    Main EOG function. Calculates average EOG artifact and finds affected channels.
    
    Parameters
    ----------
    eog_params : dict
        Dictionary with EOG parameters originating from the config file.
    eog_params_internal : dict
        Dictionary with EOG parameters originating from the config file - preset for internal use, not to be changed by the user.
    raw : mne.io.Raw
        Raw MEG data.
    channels : dict
        Dictionary with listds of channels for each channel type (mag and grad).
    chs_by_lobe : dict
        Dictionary with lists of channels separated by lobe.
    m_or_g_chosen : List
        List of channel types chosen for the analysis.
        
    Returns
    -------
    eog_derivs : List
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_EOG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    eog_str : str
        String with information about EOG channel used in the final report.
    avg_objects_eog : List
        List of Avg_artif objects, each of which contains the EOG artifact from one MEG channel.
    
    """
    # Load data
    raw, shielding_str, meg_system = load_data(data_path)

    chs_by_lobe = copy.deepcopy(chs_by_lobe_orig) 
    #in case we will change this variable in any way. If not copied it might introduce errors in parallel processing. 
    # This variable is used in all modules

    sfreq=raw.info['sfreq']
    tmin=eog_params_internal['eog_epoch_tmin']
    tmax=eog_params_internal['eog_epoch_tmax']

    norm_lvl=eog_params['norm_lvl']
    gaussian_sigma=eog_params['gaussian_sigma']
    thresh_lvl_peakfinder=eog_params['thresh_lvl_peakfinder']

    eog_str, eog_data, event_indexes, eog_ch_name = get_EOG_data(raw, orig_meg_system)

    eog_derivs = []
    if len(eog_data) == 0:
        simple_metric_EOG = {'description': eog_str}
        # For EOG we dont create any df cos there is no data: no reconstructed channel possible.
        return eog_derivs, simple_metric_EOG, eog_str, []
    

    # Now choose the channel with blinks only (if there are several):
    #(TODO: NEED TO FIGURE OUT HOW. For now just take the first one)
    eog_data = eog_data[0]
    if orig_meg_system != 'EEG':
        eog_ch_name = eog_ch_name[0]
    event_indexes = event_indexes[0]
    print('___MEG_QC___: Blinks will be detected based on channel: ', eog_ch_name)

    n_events = len(event_indexes)
    minutes_in_data = len(eog_data) / sfreq / 60
    events_rate_per_min = round(n_events / minutes_in_data, 1)
    print('___MEGqc___: ', 'EOG events detected: ', n_events)
    n_events_str = '<br><br>EOG events detected: ' + str(n_events)
    print('___MEGqc___: ', 'Blink rate per minute: ', events_rate_per_min)
    n_events_str += '<br>Blink rate per minute: ' + str(events_rate_per_min)

    use_method = 'correlation_recorded' #'mean_threshold' 
    #no need to choose method in EOG because we cant reconstruct channel, always correlaion on recorded ch (if channel present) or fail.

    
    mean_good, eog_str_checked, mean_blink, mean_rwave_time = check_mean_wave(eog_data, 'EOG', event_indexes, tmin, tmax, sfreq, eog_params_internal, thresh_lvl_peakfinder)
    eog_str += eog_str_checked + n_events_str


    #save to df:
    eog_ch_df = pd.DataFrame({
        eog_ch_name: eog_data,
        'event_indexes': event_indexes + [None] * (len(eog_data) - len(event_indexes)),
        'fs': [raw.info['sfreq']] + [None] * (len(eog_data) - 1),
        'mean_rwave': mean_blink.tolist() + [None] * (len(eog_data) - len(mean_blink)),
        'mean_rwave_time': mean_rwave_time.tolist() + [None] * (len(eog_data) - len(mean_rwave_time)),
        'recorded_or_reconstructed': [use_method] + [None] * (len(eog_data) - 1),
        'n_events': [n_events] + [None] * (len(eog_data) - 1),
        'events_rate_per_min': [events_rate_per_min] + [None] * (len(eog_data) - 1)})
    
    eog_derivs += [QC_derivative(content=eog_ch_df, name='EOGchannel', content_type = 'df')]
    

    if mean_good is False:
        simple_metric_EOG = {'description': eog_str}
        return eog_derivs, simple_metric_EOG, eog_str, []


    affected_channels={}
    bad_avg_str = {}
    avg_objects_eog=[]
    best_affected_channels={}
    
    for m_or_g  in m_or_g_chosen:

        if orig_meg_system == 'EEG':
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_ch_name, picks=channels[m_or_g], tmin=tmin, tmax=tmax)
        else:
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        # eog_derivs += plot_ecg_eog_mne(eog_epochs, m_or_g, tmin, tmax)

        artif_per_ch = calculate_artifacts_on_channels(eog_epochs, channels[m_or_g], chs_by_lobe=chs_by_lobe[m_or_g], thresh_lvl_peakfinder=thresh_lvl_peakfinder, tmin=tmin, tmax=tmax, params_internal=eog_params_internal, gaussian_sigma=gaussian_sigma)


        #2 options:
        #1. find channels with peaks above threshold defined by average over all channels+multiplier set by user
        #2. find channels that have highest Pearson correlation with average R wave shape (if the ECG channel is present)

        if use_method == 'mean_threshold':
            artif_per_ch, artif_time_vector = flip_channels(artif_per_ch, tmin, tmax, sfreq, eog_params_internal)
            affected_channels[m_or_g], affected_derivs, bad_avg_str[m_or_g], avg_overall_obj = find_affected_over_mean(artif_per_ch, 'EOG', eog_params_internal, thresh_lvl_peakfinder, m_or_g=m_or_g, norm_lvl=norm_lvl, flip_data=True, gaussian_sigma=gaussian_sigma, artif_time_vector=artif_time_vector)
            correlation_derivs = []

        elif use_method == 'correlation_recorded' or use_method == 'correlation_reconstructed':
            
            affected_channels[m_or_g] = find_affected_by_correlation(mean_blink, artif_per_ch)
            bad_avg_str[m_or_g] = ''
            avg_overall_obj = None

            #ADDED:

            best_affected_channels[m_or_g] = copy.deepcopy(affected_channels[m_or_g])

            # Now that we found best correlation values, next step is to calculate magnitude ratios of every channel
            # Then, calculate similarity value comprised of correlation and magnitude ratio:

            best_affected_channels[m_or_g] = find_affected_by_amplitude_ratio(best_affected_channels[m_or_g])

            best_affected_channels[m_or_g] = find_affected_by_similarity_score(best_affected_channels[m_or_g])



        else:
            raise ValueError('use_method should be either mean_threshold or correlation')
        
        #higher thresh_lvl_peakfinder - more peaks will be found on the eog artifact for both separate channels and average overall. As a result, average overll may change completely, since it is centered around the peaks of 5 most prominent channels.
        avg_objects_eog.append(avg_overall_obj)


    simple_metric_EOG = make_simple_metric_ECG_EOG(best_affected_channels, m_or_g_chosen, 'EOG', bad_avg_str, use_method)

    #Extract chs_by_lobe into a data frame
    artif_time_vector = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round
    #TODO: above we always use tmin, tmax, sfreq to create time vector in every fuction. here it s done again, maybe change above?

    for m_or_g  in m_or_g_chosen:
        for lobe, lobe_channels in chs_by_lobe[m_or_g].items():
            for lobe_ch in lobe_channels:
                lobe_ch.add_eog_info(best_affected_channels[m_or_g], artif_time_vector)

    eog_csv_deriv = chs_dict_to_csv(chs_by_lobe,  file_name_prefix = 'EOGs')

    eog_derivs += eog_csv_deriv

    return eog_derivs, simple_metric_EOG, eog_str, avg_objects_eog
