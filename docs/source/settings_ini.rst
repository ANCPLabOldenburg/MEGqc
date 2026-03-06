Pipeline Settings
=================

The pipeline settings are stored in the ``settings.ini`` file. 

Settings are divided into sections, each corresponding to a different pipeline step.

As the result of analysis will be produced (for each data file (.fif)):

- html report for all metrics
- csv file with the results of the analysis for some of metrics
- machine readable json file with the results of the analysis for all metrics

In the html report: 

- all the plots produced by MEQ-QC are interactive, they can be scrolled through and enlarged. 
- a few plots from MNE (in ECG and EOG sections) are not interactive.

Default settings [DEFAULT]
--------------------------
- **ch_types** (str) : which channels to process. Enter 1 or both values separated by coma. Default: *mag, grad*
- **subjects** (str) - all or list of subjects separated by coma - which subjects to process. If all chosen - all subjects in data set will be processed. Default: *all*

- Metrics to run:
    - **STD** (bool) : Standard deviation. Default: *True*
    - **PSD** (bool) : Power spectral density. Default: *True*
    - **PTP_manual** (bool) : Peak-to-peak amplitude manual. Default: *True*
    - **PTP_auto_mne** (bool) : Peak-to-peak amplitude automatic (MNE). Default: *True*
    - **ECG** (bool) : ECG: heart beat artifacts. Default: *True*
    - **EOG** (bool) : EOG: oculart artifacs. Default: *True*
    - **Head** (bool) : Head movement artifacts. Default: *True*
    - **Muscle** (bool) : High frequency (110-140Hz) noise strongy correlated with (but not limited to) Muscle artifacts. Default: *True*

- **data_crop_tmin** (int) & **data_crop_tmax** (int) : Settings for data crop. If no cropping needed, leave blank. Unit: seconds. Default: *blank*
- **data_directory** (str) : **Absolute** path to the data directory. Possible to enter several dataset paths separated by comma. Example: *user/path/to/my/data/ds000000, user/path/to/my/data/ds000001*

Filtering [Filtering]
---------------------
- **apply_filtering** (bool): if True, the data will be filtered. Default: *True*
- **downsample_to_hz** (int or float) - frequency to downsample to. Must be: 1. at least 5 times higher than h_freq to avoid misrepresentation. 2. Low value may lead to misrepresentation of chpi coils signal. They send signals in frequencies in higher than 100 hz. When downsampled this information may disappear. Recommended optimal value: 1000. Recommended minimum value: 500 Unit: Hz. Can be also set to False to avoid downsampling.
- **l_freq** (int or float) : lower frequency for bandpass filter. Unit: Hz. Default: *0*
- **h_freq** (int or float) : higher frequency for bandpass filter. Unit: Hz. Default: *140*. Reason: output of PSD can be used for filtering the data before muscle artifact detection. Musce artifacts are usually around 110-140 Hz, so this setting allows to see if there are extra frequencies which would need to be filtered out
- **method** (str) : method of filtering. Default: *iir*. Or turn off filtering completely by setting apply_filtering = False. Parameters in this case dont matter.


Epoching [Epoching]
-------------------
- **event_dur** (float) : duration of the event in seconds. Unit: sec. Default: *0.2*
- **epoch_tmin** (float) : time in seconds before the event. Unit: sec. Default: *-0.2*
- **epoch_tmax** (float) : time in seconds after the event. Unit: sec. Default: 1
- **stim_channel** (str) : leave blank if want it to be detected automatically or write explicitely like *STI101*. Default:  * blank *. 
- **event_repeated** (str) : how to handle duplicates in events[:, 0]. Can be 'error' to raise an error, ‘drop’ to only retain the row occurring first in the events, or 'merge' to combine the coinciding events (=duplicates) into a new event (see Notes for details). Default: *merge*
- **use_fixed_length_epochs** (bool) : if True and no stimulus channels are detected or fewer than 2 events are found, epoch the data into fixed-length segments. Default: *True*
- **fixed_epoch_duration** (float) : length of fixed epochs in seconds. Used only when fixed-length epoching is enabled. Default: *2.0*
- **fixed_epoch_overlap** (float) : overlap between consecutive fixed epochs in seconds. Must be smaller than duration. Default: *0.0*

Standard deviation [STD]
------------------------
- **std_lvl** (int) : set like std_lvl = 1 or std_lvl = 3, etc. Defines how many std from the mean to use for the threshold. Default: *1*
- **allow_percent_noisy_flat_epochs** (int) : Defines how many percent of epochs can be noisy or flat. Higher than this  - epoch is marked as noisy/flat. Unit: percent. Default: *70*
- **noisy_channel_multiplier** (float or int) : Multiplier to define noisy channel, if std of this channel for this epoch is over** (the mean std of this channel for all epochs together*multipliar), then this channel is noisy. Higher value - less channels are marked as noisy. Default: *1.2*
- **flat_multiplier** (float or int) : Multiplier to define flat channel, if std of this channel for this epoch is under** (the mean std of this channel for all epochs together*multipliar), then this channel is flat. Default: *0.5*

Power spectral density [PSD]
----------------------------
- **freq_min** (int or float) : lower frequency for PSD calculation. Unit: seconds. Default: *0.5*
- **freq_max** (int or float) : higher frequency for PSD calculation. Unit: seconds. Default: *140*. Reason: output of PSD can be used for filtering the data before muscle artifact detection. Musce artifacts are usually around 110-140 Hz, so this setting allows to see if there are extra frequencies which would need to be filtered out
- **psd_step_size** (float or int) : frequency resolution of the PSD. Unit: Hz. Default: *0.5*


Peak-to-peak amplitude manual [PTP_manual]
------------------------------------------
- **max_pair_dist_sec** (float) : will hard code it when decide on best value after trying out different data sets. might be different for mag and grad. Unit: seconds. Default: *20*
- **thresh_lvl** (int) : scaling factor for threshold. The higher this vaues is - the more peaks will be detected. Default: *10*
- **allow_percent_noisy_flat_epochs** (int) : defines how many percent of epochs can be noisy or flat. Over this number - epoch is marged as noisy/flat. Unit: percent. Default: *70*
- **std_lvl** (int) : set like std_lvl = 1 or std_lvl = 3, etc. Defines how many std from the mean to use for the threshold. Default: *1*
- **noisy_channel_multiplier** (float or int) : multiplier to define noisy channel, if std of this channel for this epoch is over** (the mean std of this channel for all epochs together*multipliar), then this channel is noisy. Default: *1.2*
- **flat_multiplier** (float or int) : multiplier to define flat channel, if std of this channel for this epoch is under** (the mean std of this channel for all epochs together*multipliar), then this channel is flat. Default: *0.5*
- **ptp_top_limit & ptp_bottom_limit** (float or int) : these 2 are not used now. done in case we want to limit by exact number not by std level. 


Peak-to-peak amplitude auto (based on MNE annotatons) [PTP_auto]
----------------------------------------------------------------
- **peak_m** (float or int) : minimal PTP amplitude to count as peak for magnetometers. Unit: Tesla or Tesla/meter depending on channel type. Default: *4e-14*
- **peak_g** (float or int) : minimal PTP amplitude to count as peak for gradiometers. Unit: Tesla or Tesla/meter depending on channel type. Default: *4e-14*
- **flat_m** (float or int) : max PTP amplitude to count as flat for magnetometers. Unit: Tesla or Tesla/meter depending on channel type. Default: *3e-14*
- **flat_g** (float or int) : max PTP amplitude to count as flat for gradiometers. Unit: Tesla or Tesla/meter depending on channel type. Default: *3e-14*
- **bad_percent** (int) : percentage of the time a channel can be above or below thresholds. Below this percentage, Annotations are created. Above this percentage, the channel involved is return in bads. Note the returned bads are not automatically added to info['bads']. Unit: percent. Default: *5*
- **min_duration** (float) : minimum duration required by consecutives samples to be above peak or below flat thresholds to be considered. to consider as above or below threshold. For some systems, adjacent time samples with exactly the same value are not totally uncommon. Unit: seconds. Default: *0.002*


Heart beat artifacts [ECG]
--------------------------
- **drop_bad_ch** (bool) - if True - will drop the bad ECG channel from the data and attempt to reconstruct ECG data on base of magnetometers. If False - will not drop the bad ECG channel and will attempt to calculate ECG events on base of the bad ECG channel. Default: *True*
- **n_breaks_bursts_allowed_per_10min** (int) - number of breaks in ECG channel allowed per 10 minutes of recording. (This setting is for ECG channel only, not for any other channels Used to detect a noisy ECG channel). Default: *3*
- **allowed_range_of_peaks_stds** (float) - the allowed range of peaks in standard deviations. (This setting is for ECG channel only, not for any other channels Used to detect a noisy ECG channel). Defaault: *0.05* (experimentally chosen value). How the setting is used:
    
    - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
    - The peaks will be detected on the scaled data
    - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy
    
    Unit: arbitrary (the data using this setting is always scaled between 0 and 1). Default: *0.05*

- **ecg_epoch_tmin** (float) : time in seconds before the event. Unit: seconds. Dont set smaller than -0.03. Default: *-0.04*
- **ecg_epoch_tmax** (float) : time in seconds after the event. Unit: seconds. Dont set smaller than 0.03. Default: *0.04*
- **norm_lvl** (int) : The norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold. Default: *1*
- **gaussian_sigma** (int) - The sigma of the gaussian kernel used to smooth the data. The higher the sigma, the more smoothing. Typically ECG data is less noisy than EOG nd requires smaller sigma. Default: 4
- **fixed_channel_names** (str) - Optional comma-separated list of channel names to treat as ECG (e.g., *EEG059*). Leave blank to use automatic ECG detection via the raw object.


Eye movement artifacts [EOG]
----------------------------
- **n_breaks_bursts_allowed_per_10min** (int) - number of breaks in ECG channel allowed per 10 minutes of recording. (This setting is for EOG channel only, not for any other channels Used to detect a noisy EOG channel). Default: *3*
- **allowed_range_of_peaks_stds** (float) - the allowed range of peaks in standard deviations. (This setting is for EOG channel only, not for any other channels Used to detect a noisy EOG channel). Default: *0.15* (experimentally chosen value). How the setting is used:
    
    - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
    - The peaks will be detected on the scaled data
    - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy

    Unit: arbitrary (the data using this setting is always scaled between 0 and 1). Default: *0.05*

- **eog_epoch_tmin** (float) : time in seconds before the event. Unit: seconds. Default: *-0.2*
- **eog_epoch_tmax** (float) : time in seconds after the event. Unit: seconds. Default: *0.4*
- **norm_lvl** (int) : the norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold. Default: *1*
- **gaussian_sigma** (int) - The sigma of the gaussian kernel used to smooth the data. The higher the sigma, the more smoothing. Typically EOG data is more noisy than EG nd requires larger sigma. Default: 6
- **fixed_channel_names** (str) - Optional comma-separated list of channel names to treat as EOG (e.g., *EEG057, EEG058*). Leave blank to use automatic EOG detection via the raw object.

Head_movement artifacts [Head_movement]
---------------------------------------
No available settings


Muscle artifacts [Muscle]
-------------------------
- **muscle_freqs** (2 ints or 2 float) : defines the frequency band for detecting muscle activity. Unit: Hz. Default: 110, 140
- **threshold_muscle** (int or float) : threshold for muscle detection. Zscores detected above this threshold will be considered as muscle artifacts. Unit: z-score.  Default: *5, 10*
- **min_length_good** (int or float) : The shortest allowed duration of "good data"** (in seconds) between adjacent muscle annotations; shorter segments will be incorporated into the surrounding annotations. Unit: seconds. Default: *0.2*
- **min_distance_between_different_muscle_events** (int or float) : minimum distance between different muscle events in seconds. If events happen closer to each other they will all be counted as one event and the time will be assigned as the first peak. Unit: seconds. Default: *1*  

Difference between last 2 settings: **min_length_good** - used to detect ALL muscle events, **min_distance_between_different_muscle_events** - used to detect evets with z-score higher than the threshold on base of ALL muscle events
