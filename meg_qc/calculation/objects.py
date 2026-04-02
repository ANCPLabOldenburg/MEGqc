import plotly
from io import BytesIO
import base64
import warnings
import numpy as np
import pandas as pd
from typing import List, Union

class QC_channel:

    """ 
    Channel with calculated data, used also for plotting.
    Supports MEG channels (mag, grad) and EEG channels.
    """

    def __init__(self, name: str, type: str, lobe: str = None, lobe_color: str = None, system: str = None, loc: List = None, time_series: Union[List, np.ndarray] = None, std_overall: float = None, std_epoch: Union[List, np.ndarray] = None, ptp_overall: float = None, ptp_epoch: Union[List, np.ndarray] = None, psd: Union[List, np.ndarray] = None, freq: Union[List, np.ndarray] = None, mean_ecg: Union[List, np.ndarray] = None, mean_ecg_smoothed: Union[List, np.ndarray] = None, mean_eog: Union[List, np.ndarray] = None, mean_eog_smoothed: Union[List, np.ndarray] = None, ecg_time = None, eog_time = None, ecg_corr_coeff = None, ecg_pval = None, ecg_amplitude_ratio = None, ecg_similarity_score = None, eog_corr_coeff = None, eog_pval = None, eog_amplitude_ratio = None, eog_similarity_score = None, muscle = None, head = None, muscle_time = None, head_time = None):

        """
        Constructor method
        
        Parameters
        ----------
        name : str
            The name of the channel.
        type : str
            The type of the channel: 'mag', 'grad', or 'eeg'
        lobe : str
            The lobe area of the channel: 'left frontal', 'right frontal', 'left temporal', 'right temporal', 'left parietal', 'right parietal', 'left occipital', 'right occipital', 'central', 'subcortical', 'unknown'.
        lobe_color : str
            The color code for plotting with plotly according to the lobe area of the channel.
        system : str
            The system of the channel: 'CTF', 'TRIUX', 'OTHER'
        loc : List
            The location of the channel on the helmet.
        time_series : array
            The time series of the channel.
        std_overall : float
            The standard deviation of the channel time series.
        std_epoch : array
            The standard deviation of the channel time series per epochs.
        ptp_overall : float
            The peak-to-peak amplitude of the channel time series.
        ptp_epoch : array
            The peak-to-peak amplitude of the channel time series per epochs.
        psd : array
            The power spectral density of the channel.
        freq: array
            Frequencies for psd.
        mean_ecg : float
            The mean ECG artifact of the channel.
        mean_eog : float
            The mean EOG artifact of the channel.
        mean_ecg_smoothed : float
            The mean ECG artifact of the channel smoothed.
        mean_eog_smoothed : float
            The mean EOG artifact of the channel smoothed.
        ecg_corr_coeff : float
            The correlation coefficient of the channel with ECG.
        ecg_pval : float
            The p-value of the correlation coefficient of the channel with ECG.
        ecg_amplitude_ratio : float
            relation of the amplitude of a particular channel to all other channels for ECG contamination.
        ecg_similarity_score : float
            similarity score of the mean ecg data of this channel to refernce ecg/eog data comprised of both correlation and amplitude like: similarity_score = corr_coef * amplitude_ratio
        eog_corr_coeff : float
            The correlation coefficient of the channel with EOG.
        eog_pval : float
            The p-value of the correlation coefficient of the channel with EOG.
        eog_amplitude_ratio : float
            relation of the amplitude of a particular channel to all other channels for EOG contamination.
        eog_similarity_score : float
            similarity score of the mean eog data of this channel to refernce ecg/eog data comprised of both correlation and amplitude like: similarity_score = corr_coef * amplitude_ratio
        ecg_time : float
            The time vector of the ECG artifact.
        eog_time : float
            The time vector of the EOG artifact.
        muscle : float
            The muscle artifact data of the channel.
        head : float
            The head movement artifact data of the channel.
        muscle_time : float
            The time vector of the muscle artifact.
        head_time : float
            The time vector of the head movement artifact.
        

        """

        self.name = name
        self.type = type
        self.lobe = lobe
        self.lobe_color = lobe_color
        self.system = system
        self.loc = loc
        self.time_series = time_series
        self.std_overall = std_overall
        self.std_epoch = std_epoch
        self.ptp_overall = ptp_overall
        self.ptp_epoch = ptp_epoch
        self.psd = psd
        self.freq = freq
        self.mean_ecg = mean_ecg
        self.mean_ecg_smoothed = mean_ecg_smoothed
        self.mean_eog = mean_eog
        self.mean_eog_smoothed = mean_eog_smoothed
        self.ecg_corr_coeff = ecg_corr_coeff
        self.ecg_pval = ecg_pval
        self.ecg_amplitude_ratio = ecg_amplitude_ratio
        self.ecg_similarity_score = ecg_similarity_score
        self.eog_corr_coeff = eog_corr_coeff
        self.eog_pval = eog_pval
        self.eog_amplitude_ratio = eog_amplitude_ratio
        self.eog_similarity_score = eog_similarity_score
        self.ecg_time = ecg_time
        self.eog_time = eog_time
        self.muscle = muscle
        self.head = head
        self.muscle_time = muscle_time
        self.head_time = head_time


    def __repr__(self):

        """
        Returns the string representation of the object.

        """

        all_metrics = [self.std_overall, self.std_epoch, self.ptp_overall, self.ptp_epoch, self.psd, self.mean_ecg, self.mean_eog, self.ecg_corr_coeff, self.ecg_pval, self.ecg_amplitude_ratio, self.ecg_similarity_score, self.eog_corr_coeff, self.eog_pval, self.eog_amplitude_ratio, self.eog_similarity_score, self.muscle, self.head]
        all_metrics_names= ['std_overall', 'std_epoch', 'ptp_overall', 'ptp_epoch', 'psd', 'mean_ecg', 'mean_eog', 'ecg_corr_coeff', 'ecg_pval', 'ecg_amplitude_ratio', 'ecg_similarity_score', 'eog_corr_coeff', 'eog_pval', 'eog_amplitude_ratio', 'eog_similarity_score', 'muscle', 'head']
        non_none_indexes = [i for i, item in enumerate(all_metrics) if item is not None]

        return self.name + f' (type: {self.type}, lobe area: {self.lobe}, color code: {self.lobe_color}, location: {self.loc}, metrics_assigned: {", ".join([all_metrics_names[i] for i in non_none_indexes])}, | ecg_corr_coeff {self.ecg_corr_coeff}, eog_corr_coeff {self.eog_corr_coeff}, ecg_amplitude_ratio {self.ecg_amplitude_ratio}, eog_amplitude_ratio {self.eog_amplitude_ratio}, ecg_similarity_score {self.ecg_similarity_score}, eog_similarity_score {self.eog_similarity_score})'
    

    def to_df(self):

        """
        Returns the object as a pandas DataFrame. To be later exported into a tsv file.
        """
        
        data_dict = {}
        attr_to_column = {
            'name': 'Name', 'type': 'Type', 'lobe': 'Lobe', 'lobe_color': 'Lobe Color', 'system': 'System', 'loc': 'Sensor_location',
            'time_series': 'Time series', 'std_overall': 'STD all', 'std_epoch': 'STD epoch', 'ptp_overall': 'PtP all', 'ptp_epoch': 'PtP epoch',
            'psd': 'PSD', 'freq': 'Freq', 'mean_ecg': 'mean_ecg', 'mean_ecg_smoothed': 'smoothed_mean_ecg', 'mean_eog': 'mean_eog',
            'mean_eog_smoothed': 'smoothed_mean_eog', 'ecg_corr_coeff': 'ecg_corr_coeff', 'ecg_pval': 'ecg_pval', 'ecg_amplitude_ratio': 'ecg_amplitude_ratio',
            'ecg_similarity_score': 'ecg_similarity_score', 'eog_corr_coeff': 'eog_corr_coeff', 'eog_pval': 'eog_pval', 'eog_amplitude_ratio': 'eog_amplitude_ratio',
            'eog_similarity_score': 'eog_similarity_score', 'muscle': 'Muscle', 'head': 'Head'
        }

        for attr, column_name in attr_to_column.items():
            value = getattr(self, attr)
            if isinstance(value, (list, np.ndarray)):
                if attr.lower() == 'psd':
                    freqs = getattr(self, 'freq')
                    data_dict.update({f'{column_name}_Hz_{freqs[i]}': [v] for i, v in enumerate(value)})
                # elif attr.lower() in ['mean_ecg', 'mean_eog', 'muscle', 'head']:
                #     times = getattr(self, f'{attr.split("_")[-1]}_time') #will take part of the string before _time
                #     data_dict.update({f'{column_name}_sec_{times[i]}': [v] for i, v in enumerate(value)})

                elif 'mean_ecg' in attr or 'mean_eog' in attr or 'muscle' == attr or 'head' == attr:
                    if attr == 'mean_ecg':
                        times = getattr(self, 'ecg_time') #attr can be 'mean_ecg', etc
                    elif attr == 'mean_eog':
                        times = getattr(self, 'eog_time') #attr can be 'mean_ecg', etc
                    elif attr == 'head':
                        times = getattr(self, 'head_time') #attr can be 'mean_ecg', etc
                    elif attr == 'muscle':
                        times = getattr(self, 'muscle_time') #attr can be 'mean_ecg', etc
                    
                    for i, v in enumerate(value):
                        t = times[i]
                        data_dict[f'{column_name}_sec_{t}'] = [v]

                else: #TODO: here maybe change to elif std/ptp?
                    for i, v in enumerate(value):
                        data_dict[f'{column_name}_{i}'] = [v]

            else:
                data_dict[column_name] = [value]

        return pd.DataFrame(data_dict)
    

    def add_ecg_info(self, Avg_artif_list: List, artif_time_vector: List):

        """
        Adds ECG artifact info to the channel object.

        Parameters
        ----------
        Avg_artif_list : List
            List of the average artifact objects.
        artif_time_vector : List
            Time vector of the artifact.

        """

        for artif_ch in Avg_artif_list:
            if artif_ch.name == self.name:
                self.mean_ecg = artif_ch.artif_data
                self.mean_ecg_smoothed = artif_ch.artif_data_smoothed
                self.ecg_time = artif_time_vector
                self.ecg_corr_coeff = artif_ch.corr_coef
                self.ecg_pval = artif_ch.p_value
                self.ecg_amplitude_ratio = artif_ch.amplitude_ratio
                self.ecg_similarity_score = artif_ch.similarity_score
                

    def add_eog_info(self, Avg_artif_list: List, artif_time_vector: List):

        """
        Adds EOG artifact info to the channel object.

        Parameters
        ----------
        Avg_artif_list : List
            List of the average artifact objects.
        artif_time_vector : List
            Time vector of the artifact.

        """

        for artif_ch in Avg_artif_list:
            if artif_ch.name == self.name:
                self.mean_eog = artif_ch.artif_data
                self.mean_eog_smoothed = artif_ch.artif_data_smoothed
                self.eog_time = artif_time_vector
                self.eog_corr_coeff = artif_ch.corr_coef
                self.eog_pval = artif_ch.p_value
                self.eog_amplitude_ratio = artif_ch.amplitude_ratio
                self.eog_similarity_score = artif_ch.similarity_score

                #Attention: here time_vector, corr_coeff, p_val and everything get assigned to ecg or eog, 
                # but artif_ch doesnt have this separation to ecg/eog. 
                # Need to just make sure that the function is called in the right place.


# Backward-compatibility alias — existing code that imports MEG_channel keeps working.
MEG_channel = QC_channel


class QC_derivative:

    """ 
    Derivative of a QC measurement, main content of which is figure, data frame (saved later as csv) or html string.

    Attributes
    ----------
    content : figure, pd.DataFrame or str
        The main content of the derivative.
    name : str
        The name of the derivative (used to save in to file system)
    content_type : str
        The type of the content: 'plotly', 'matplotlib', 'csv', 'report' or 'mne_report'.
        Used to choose the right way to save the derivative in main function.
    description_for_user : str, optional
        The description of the derivative, by default 'Add measurement description for a user...'
        Used in the report to describe the derivative.
    

    """

    def __init__(self, content, name: str, content_type: str, description_for_user: str = '', fig_order: float = 0):

        """
        Constructor method
        
        Parameters
        ----------
        content : figure, pd.DataFrame or str
            The main content of the derivative.
        name : str
            The name of the derivative (used to save in to file system)
        content_type : str
            The type of the content: 'plotly', 'matplotlib', 'df', 'report' or 'mne_report'.
            Used to choose the right way to save the derivative in main function.
        description_for_user : str, optional
            The description of the derivative, by default 'Add measurement description for a user...'
            Used in the report to describe the derivative.
        fig_order : int, optional
            The order of the figure in the report, by default 0. Used for sorting.
        

        """

        self.content =  content
        self.name = name
        self.content_type = content_type
        self.description_for_user = description_for_user
        self.fig_order = fig_order

    def __repr__(self):

        """
        Returns the string representation of the object.
        """

        return 'MEG QC derivative: \n content: ' + str(type(self.content)) + '\n name: ' + self.name + '\n type: ' + self.content_type + '\n description for user: ' + self.description_for_user + '\n '

    def convert_fig_to_html(self):

        """
        Converts figure to html string.
        
        Returns
        -------
        html : str or None
            Html string or None if content_type is not 'plotly' or 'matplotlib'.

        """

        if self.content_type == 'plotly':
            return plotly.io.to_html(self.content, full_html=False)
        elif self.content_type == 'matplotlib':
            tmpfile = BytesIO()
            self.content.savefig(tmpfile, format='png', dpi=130) #writing image into a temporary file
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            return html
            # return mpld3.fig_to_html(self.content)
        elif not self.content_type:
            warnings.warn("Empty content_type of this QC_derivative instance")
        else:
            return None

    def convert_fig_to_html_add_description(self):

        """
        Converts figure to html string and adds description.

        Returns
        -------
        html : str or None
            Html string: fig + description or None + description if content_type is not 'plotly' or 'matplotlib'.

        """

        figure_report = self.convert_fig_to_html()

        return """<br></br>"""+ figure_report + """<p>"""+self.description_for_user+"""</p>"""