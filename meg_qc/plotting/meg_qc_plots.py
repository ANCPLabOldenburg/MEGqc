import sys
import os
import ancpbids
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style
from collections import defaultdict
import re
from typing import List
from pprint import pprint

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
gradparent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(gradparent_dir)

from meg_qc.plotting.universal_plots import *
from meg_qc.plotting.universal_html_report import make_joined_report_mne

# IMPORTANT: keep this order of imports, first need to add parent dir to sys.path, then import from it.

# ____________________________

# How plotting in MEGqc works:
# During calculation save in the right folders the csvs with data for plotting
# During plotting step - read the csvs (find using ancpbids), plot them, save them as htmls in the right folders.


def create_categories_for_selector(entities: dict):

    """
    Create categories based on what metrics have already been calculated and detected as ancp bids as entities in MEGqc derivatives folder.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.
    
    Returns
    -------
    categories : dict
        A dictionary of entities and their subcategories with modified names
    """

    # Create a copy of entities
    categories = entities.copy()

    # Rename 'description' to 'METRIC' and sort the values
    categories = {
        ('METRIC' if k == 'description' else k): sorted(v, key=str)
        for k, v in categories.items()
    }

    #From METRIC remove whatever is not metric. 
    #Cos METRIC is originally a desc entity which can contain just anything:
                
    if 'METRIC' in categories:
        valid_metrics = ['_ALL_METRICS_', 'STDs', 'PSDs', 'PtPsManual', 'PtPsAuto', 'ECGs', 'EOGs', 'Head', 'Muscle']
        categories['METRIC'] = [x for x in categories['METRIC'] if x.lower() in [metric.lower() for metric in valid_metrics]]

    #add '_ALL_' to the beginning of the list for each category:

    for category, subcategories in categories.items():
        categories[category] = ['_ALL_'+category+'s_'] + subcategories

    # Add 'm_or_g' category
    categories['m_or_g'] = ['_ALL_sensors', 'mag', 'grad']

    return categories


def selector(entities: dict):

    """
    Creates a in-terminal visual selector for the user to choose the entities and settings for plotting.

    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    selected_entities : dict
        A dictionary of selected entities.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    """

    # SELECT ENTITIES and SETTINGS
    # Define the categories and subcategories
    categories = create_categories_for_selector(entities)

    selected = {}
    # Create a list of values with category titles
    for key, values in categories.items():
        results, quit_selector = select_subcategory(categories[key], key)

        print('___MEGqc___: select_subcategory: ', key, results)

        if quit_selector: # if user clicked cancel - stop:
            print('___MEGqc___: You clicked cancel. Please start over.')
            return None, None
        
        selected[key] = results


    # Separate into selected_entities and plot_settings
    selected_entities = {key: values for key, values in selected.items() if key != 'm_or_g'}
    plot_settings = {key: values for key, values in selected.items() if key == 'm_or_g'}

    return selected_entities, plot_settings


def select_subcategory(subcategories: List, category_title: str, window_title: str = "What would you like to plot? Click to select."):

    """
    Create a checkbox list dialog for the user to select subcategories.
    Example:
    sub: 009, 012, 013

    Parameters
    ----------
    subcategories : List
        A list of subcategories, such as: sub, ses, task, run, metric, mag/grad.
    category_title : str
        The title of the category.
    window_title : str
        The title of the checkbox list dialog, for visual.

    Returns
    -------
    results : List
        A list of selected subcategories.
    quit_selector : bool
        A boolean indicating whether the user clicked Cancel.

    """

    quit_selector = False

    # Create a list of values with category titles
    values = [(str(items), str(items)) for items in subcategories]

    while True:
        results = checkboxlist_dialog(
            title=window_title,
            text=category_title,
            values=values,
            style=Style.from_dict({
                'dialog': 'bg:#cdbbb3',
                'button': 'bg:#bf99a4',
                'checkbox': '#e8612c',
                'dialog.body': 'bg:#a9cfd0',
                'dialog shadow': 'bg:#c98982',
                'frame.label': '#fcaca3',
                'dialog.body label': '#fd8bb6',
            })
        ).run()

        # Set quit_selector to True if the user clicked Cancel (results is None)
        quit_selector = results is None

        if quit_selector or results:
            break
        else:
            print('___MEGqc___: Please select at least one subcategory or click Cancel.')


    # if '_ALL_' was chosen - choose all categories, except _ALL_ itself:
    if results: #if something was chosen
        for r in results:
            if '_ALL_' in r.upper():
                results = [str(category) for category in subcategories if '_ALL_' not in str(category).upper()]
                #Important! Keep ....if '_ALL_' not in str(category).upper() with underscores!
                #otherwise it will excude tasks like 'oddbALL' and such
                break

    return results, quit_selector


def get_ds_entities(dataset, calculated_derivs_folder: str):

    """
    Get the entities of the dataset using ancpbids, only get derivative entities, not all raw data.

    Parameters
    ----------
    dataset : ancpbids object
        The dataset object.
    calculated_derivs_folder : str
        The path to the calculated derivatives folder.
    
    Returns
    -------
    entities : dict
        A dictionary of entities and their subcategories.

    """

    try: 
        entities = dataset.query_entities(scope=calculated_derivs_folder)
        print('___MEGqc___: ', 'Entities found in the dataset: ', entities)
        #we only get entities of calculated derivatives here, not entire raw ds.
    except:
        raise FileNotFoundError(f'___MEGqc___: No calculated derivatives found for this ds!')
    
    return entities


def csv_to_html_report(raw_info_path: str, metric: str, tsv_paths: List, report_str_path: str, plot_settings):

    """
    Create an HTML report from the CSV files.

    Parameters
    ----------
    raw_info_path : str
        The path to the raw info object.
    metric : str
        The metric to be plotted.
    tsv_paths : List
        A list of paths to the CSV files.
    report_str_path : str
        The path to the JSON file containing the report strings.
    plot_settings : dict
        A dictionary of selected settings for plotting.
    
    Returns
    -------
    report_html_string : str
        The HTML report as a string.
    
    """

    m_or_g_chosen = plot_settings['m_or_g'] 

    time_series_derivs, sensors_derivs, ptp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, std_derivs, psd_derivs, muscle_derivs, head_derivs = [], [], [], [], [], [], [], [], [], []

    stim_derivs = []
    
    for tsv_path in tsv_paths: #if we got several tsvs for same metric, like for PSD:

        #get the final file name of tsv path:
        basename = os.path.basename(tsv_path)
        if 'desc-stimulus' in basename:
            stim_derivs = plot_stim_csv(tsv_path) 

        if 'STD' in metric.upper():

            fig_std_epoch0 = []
            fig_std_epoch1 = []

            std_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_topomap = plot_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_topomap_3d = plot_3d_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='stds')

                std_derivs += fig_topomap + fig_topomap_3d + fig_all_time + fig_std_epoch0 + fig_std_epoch1

        if 'PTP' in metric.upper():

            fig_ptp_epoch0 = []
            fig_ptp_epoch1 = []

            ptp_manual_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_topomap = plot_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_topomap_3d = plot_3d_topomap_std_ptp_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='peaks')

                ptp_manual_derivs += fig_topomap + fig_topomap_3d + fig_all_time + fig_ptp_epoch0 + fig_ptp_epoch1

        elif 'PSD' in metric.upper():

            method = 'welch' #is also hard coded in PSD_meg_qc() for now

            psd_derivs += plot_sensors_3d_csv(tsv_path)

            for m_or_g in m_or_g_chosen:

                psd_derivs += Plot_psd_csv(m_or_g, tsv_path, method)

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'noise')

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'waves')

        elif 'ECG' in metric.upper():

            ecg_derivs += plot_sensors_3d_csv(tsv_path)

            ecg_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            ecg_derivs += plot_mean_rwave_csv(tsv_path, 'ECG')

            #TODO: add ch description like here? export it as separate report strings?
            #noisy_ch_derivs += [QC_derivative(fig, bad_ecg_eog[ecg_ch]+' '+ecg_ch, 'plotly', description_for_user = ecg_ch+' is '+ bad_ecg_eog[ecg_ch]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) tolerable number of breaks: '+str(ecg_eval[1])+', 3) tolerable number of bursts: '+str(ecg_eval[2]))]

            for m_or_g in m_or_g_chosen:
                ecg_derivs += plot_artif_per_ch_3_groups(tsv_path, m_or_g, 'ECG', flip_data=False)
                #ecg_derivs += plot_correlation_csv(tsv_path, 'ECG', m_or_g)

        elif 'EOG' in metric.upper():

            eog_derivs += plot_sensors_3d_csv(tsv_path)

            eog_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            eog_derivs += plot_mean_rwave_csv(tsv_path, 'EOG')
                
            for m_or_g in m_or_g_chosen:
                eog_derivs += plot_artif_per_ch_3_groups(tsv_path, m_or_g, 'EOG', flip_data=False)
                #eog_derivs += plot_correlation_csv(tsv_path, 'EOG', m_or_g)

            
        elif 'MUSCLE' in metric.upper():

            muscle_derivs +=  plot_muscle_csv(tsv_path)

            
        elif 'HEAD' in metric.upper():
                
            head_pos_derivs, _ = plot_head_pos_csv(tsv_path)
            # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
            # head_pos_derivs += head_pos_derivs2
            head_derivs += head_pos_derivs

    QC_derivs = {
        'TIME_SERIES': time_series_derivs,
        'STIMULUS': stim_derivs,
        'SENSORS': sensors_derivs,
        'STD': std_derivs,
        'PSD': psd_derivs,
        'PTP_MANUAL': ptp_manual_derivs,
        'PTP_AUTO': pp_auto_derivs,
        'ECG': ecg_derivs,
        'EOG': eog_derivs,
        'HEAD': head_derivs,
        'MUSCLE': muscle_derivs,
        'REPORT_MNE': []
    }


    #Sort all based on fig_order of QC_derivative:
    #(To plot them in correct order in the report)
    for metric, values in QC_derivs.items():
        if values:
            QC_derivs[metric] = sorted(values, key=lambda x: x.fig_order)


    if not report_str_path: #if no report strings were saved. happens when mags/grads didnt run to make tsvs.
        report_strings = {
        'INITIAL_INFO': '',
        'TIME_SERIES': '',
        'STD': '',
        'PSD': '',
        'PTP_MANUAL': '',
        'PTP_AUTO': '',
        'ECG': '',
        'EOG': '',
        'HEAD': '',
        'MUSCLE': '',
        'SENSORS': '',
        'STIMULUS': ''
        }
    else:
        with open(report_str_path) as json_file:
            report_strings = json.load(json_file)


    report_html_string = make_joined_report_mne(raw_info_path, QC_derivs, report_strings)

    return report_html_string 


def extract_raw_entities_from_obj(obj):

    """
    Function to create a key from the object excluding the 'desc' attribute
    
    Parameters
    ----------
    obj : ancpbids object
        An object from ancpbids.
    
    Returns
    -------
    tuple
        A tuple containing the name, extension, and suffix of the object.

    """
    # Remove the 'desc' part from the name, so we get the name of original raw that the deriv belongs to:
    raw_name = re.sub(r'_desc-[^_]+', '', obj.name)
    return (raw_name, obj.extension, obj.suffix)


def sort_tsvs_by_raw(tsvs_by_metric: dict):

    """
    For every metric, if we got same raw entitites, we can combine derivatives for the same raw into a list.
    Since we collected entities not from raw but from derivatives, we need to remove the desc part from the name.
    After that we combine files with the same 'name' in entity_val objects in 1 list:

    Parameters
    ----------
    tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files.
    
    Returns
    -------
    combined_tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files combined by raw entity

    """

    sorted_tsvs_by_metric_by_raw = {}

    for metric, obj_dict in tsvs_by_metric.items():
        combined_dict = defaultdict(list)
        
        for obj, tsv_path in obj_dict.items():
            raw_entities = extract_raw_entities_from_obj(obj)
            combined_dict[raw_entities].extend(tsv_path)
        
        # Convert keys back to original objects
        final_dict = {}
        for raw_entities, paths in combined_dict.items():
            # Find the first object with the same key
            for obj in obj_dict.keys():
                if extract_raw_entities_from_obj(obj) == raw_entities:
                    final_dict[obj] = paths
                    break
    
        sorted_tsvs_by_metric_by_raw[metric] = final_dict

    pprint('___MEGqc___: ', 'sorted_tsvs_by_metric_by_raw: ', sorted_tsvs_by_metric_by_raw)

    return sorted_tsvs_by_metric_by_raw

class Deriv_to_plot:

    def __init__(self, path: str, metric: str, deriv_entity_obj, raw_entity_name: str = None):

        self.path = path
        self.metric = metric
        self.deriv_entity_obj = deriv_entity_obj
        self.raw_entity_name = raw_entity_name

        # Find the subject ID using regex
        match = re.search(r'sub-\d+_', self.deriv_entity_obj['name'])
        if match:
            self.subject = match.group(0).replace('sub-', '').rstrip('_')
        else:
            self.subject = None  # or handle the case where the subject ID is not found

    def __repr__(self):
        return (
            f"Deriv_to_plot(\n"
            f"    subject={self.subject},\n"
            f"    path={self.path},\n"
            f"    metric={self.metric},\n"
            f"    deriv_entity_obj={self.deriv_entity_obj},\n"
            f"    raw_entity_name={self.raw_entity_name}\n"
            f")"
        )

    def print_detailed_entities(self):
        keys = list(self.deriv_entity_obj.keys())
        for val in keys[:-1]:  # Iterate over all keys except the last one
            print('_Deriv_: ', val, self.deriv_entity_obj[val])

    def find_raw_entity_name(self):
        #find the raw entity name from the deriv entity name:
        self.raw_entity_name = re.sub(r'_desc-.*', '', self.deriv_entity_obj['name'])


def make_plots_meg_qc(dataset_path: str):

    """
    Create plots for the MEG QC pipeline.

    Parameters
    ----------
    dataset_path : str
        A list of paths to the datasets.
    
    Returns
    -------
    tsvs_to_plot_by_metric : dict
        A dictionary of metrics and their corresponding TSV files.
    
    """

    #1. ____Collect what we got in the derivatives and give user selector to choose what to plot in report____

    try:
        dataset = ancpbids.load_dataset(dataset_path)
        schema = dataset.get_schema()
    except:
        print('___MEGqc___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
        return

    #make sure the derivatives folder exists (it must! otherwise what do we plot from?):
    derivatives_path = os.path.join(dataset_path, 'derivatives')
    if not os.path.isdir(derivatives_path):
        os.mkdir(derivatives_path)
        print('___MEGqc___: ', 'Derivs folder was not found! Created new.')

    calculated_derivs_folder = os.path.join('derivatives', 'Meg_QC', 'calculation')

    entities = get_ds_entities(dataset, calculated_derivs_folder) #get entities of the dataset using ancpbids

    chosen_entities, plot_settings = selector(entities)
    if not chosen_entities:
        return

    #check that 'task' and 'subject' entities are not empty, because they are REQUIRED: 
    if 'task' not in chosen_entities or not chosen_entities['task']:
        print('___MEGqc___: ', 'Task entity is required! Please start over and select a task.')
        return
    
    if 'subject' not in chosen_entities or not chosen_entities['subject']:
        print('___MEGqc___: ', 'Subject entity is required! Please start over and select a subject.')
        return
    
    # Ensure 'run' and 'session' are in chosen_entities, set to None if missing.
    # None can be ignored by ancpbids later, but empty list can raise errors
    for key in ['run', 'session']:
        chosen_entities.setdefault(key, None)

    #Add stimulus, raw info obj and report strings.
    chosen_entities['METRIC'].append('stimulus')
    chosen_entities['METRIC'].append('RawInfo')
    chosen_entities['METRIC'].append('ReportStrings')

    print('___MEGqc___: CHOSEN entities to plot: ', chosen_entities)
    print('___MEGqc___: CHOSEN settings: ', plot_settings)


    # 2. ____Here we collect tsvs to be plotted in the report for each sub, metric based on selection:____

    calculated_derivs_folder = os.path.join('derivatives', 'Meg_QC', 'calculation')

    tsvs_to_plot_by_metric = {}
    tsv_entities_by_metric = {}

    for metric in chosen_entities['METRIC']:
        # Creating the full list of files for each combination of chosen entities:

        # We call query with entities that always must present + entities that might present, might not:
        # This is how the call would look if we had all entities:
        # tsv_path = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['session'], task = chosen_entities['task'], run = chosen_entities['run'], desc = desc, scope=calculated_derivs_folder)))

        #required entities:
        entities = {
            'subj': chosen_entities['subject'],
            'task': chosen_entities['task'],
            'suffix': 'meg',
            'extension': ['tsv', 'json', 'fif'], 
            #tsv is for all the figures, json is for report strings, fif is for raw info obj.
            'return_type': 'filename',
            'desc': '',
            'scope': calculated_derivs_folder,
        }

        #add desc based on metric:
        if metric == 'PSDs':
            entities['desc'] = ['PSDs', 'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad']
        elif metric == 'ECGs':
            entities['desc'] = ['ECGchannel', 'ECGs']
        elif metric == 'EOGs':
            entities['desc'] = ['EOGchannel', 'EOGs']
        else:
            entities['desc'] = [metric]


        #optional entities:
        if 'session' in chosen_entities and chosen_entities['session']:
            entities['session'] = chosen_entities['session']

        if 'run' in chosen_entities and chosen_entities['run']:
            entities['run'] = chosen_entities['run']

        # Query tsv derivs and get the tsv paths:
        tsv_paths = list(dataset.query(**entities))
        tsvs_to_plot_by_metric[metric] = sorted(tsv_paths)

        # Query same tsv derivs and get the tsv entities to later use them to save report with same entities by ancpbids:
        entities['return_type'] = 'object'
        entities_obj = sorted(list(dataset.query(**entities)), key=lambda k: k['name'])
        tsv_entities_by_metric[metric] = entities_obj


    # Collect all derivs into a list of Deriv_to_plot objects, combining tsv path and its entities:
    derivs_to_plot = []
    for (tsv_metric, tsv_paths), (entity_metric, entity_vals) in zip(tsvs_to_plot_by_metric.items(), tsv_entities_by_metric.items()):

        # 1) Check:
        if tsv_metric != entity_metric:
            raise ValueError('Different metrics in tsvs_to_plot_by_metric and entities_per_file')
        if len(tsv_paths) != len(entity_vals):
            raise ValueError('Different number of tsvs and entities for metric: ', tsv_metric)
        
        for tsv_paths, deriv_entities in zip(tsv_paths, entity_vals):
        #check that every metric_value is same as file_value:
            file_name_in_path = os.path.basename(tsv_paths).split('_meg.')[0]
            file_name_in_obj = deriv_entities['name'].split('_meg.')[0]

            if file_name_in_obj not in file_name_in_path:
                raise ValueError('Different names in tsvs_to_plot_by_metric and entities_per_file')
            
            #2) Collect:
            deriv = Deriv_to_plot(path = tsv_paths, metric = tsv_metric, deriv_entity_obj = deriv_entities)
            deriv.find_raw_entity_name()

            derivs_to_plot.append(deriv)

    pprint('_________________________end part2_________________________')
    for d in derivs_to_plot:
        print(d)
        d.print_detailed_entities()
        pprint(' ')


    # 3. ___Create the derivatives for each metric and save them to the dataset:___

    # Create a folder for the reports
    derivative = dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    reports_folder = derivative.create_folder(name='reports')

    #for each sub and each raw_entity_name derivs_to_plot find RawInfo, ReportStrings and all tsvs for this raw:

    for sub in chosen_entities['subject']:
        subject_folder = reports_folder.create_folder(name='sub-'+sub)

        #find existing raws for this subject:
        existing_raws_per_sub = list(set([deriv.raw_entity_name for deriv in derivs_to_plot if deriv.subject == sub]))

        print('___MEGqc___: ', 'existing_raws_per_sub: ', existing_raws_per_sub)
        

        for raw_entity_name in existing_raws_per_sub:
            #for each raw entity name, find all derivs that belong to this raw:
            derivs_for_this_raw = [deriv for deriv in derivs_to_plot if deriv.raw_entity_name == raw_entity_name]

            pprint('___________Part3: derivs_for_this_raw______________')
            for d in derivs_for_this_raw:
                print(d)
                d.print_detailed_entities()
                pprint(' ')

            #print('___MEGqc___: ', 'derivs_for_this_raw: ', derivs_for_this_raw)

            #find RawInfo and ReportStrings for this raw in derivs_for_this_raw:
            raw_info_path = None
            report_str_path = None
            tsv_paths = []

            for deriv in derivs_for_this_raw:
                if deriv.metric == 'RawInfo':
                    raw_info_path = deriv.path
                elif deriv.metric == 'ReportStrings':
                    report_str_path = deriv.path


            # Now create a separate report for each METRIC for this raw file:
            metrics_to_plot = [metric for metric in chosen_entities['METRIC'] if metric not in ['RawInfo', 'ReportStrings']]

            counter = 0
            for metric in metrics_to_plot:

                #collect tsvs for this metric:
                tsv_paths = [deriv.path for deriv in derivs_for_this_raw if deriv.metric == metric]
                if not tsv_paths:
                    print(f'___MEGqc___: No tsvs found for this metric ({metric}) and subject ({sub})')
                    continue
                    

                tsvs_for_this_raw = [deriv for deriv in derivs_for_this_raw if deriv.metric == metric]

                print('______NEW VERSION_____')
                print('___MEGqc___: ', 'metric: ', metric)
                print('___MEGqc___: ', 'tsv_paths: ', tsv_paths)
                print('___MEGqc___: ', 'tsvs_for_this_raw: ', tsvs_for_this_raw)

                raw_entities_to_write = tsvs_for_this_raw[0].deriv_entity_obj
                # take any file belonging to this raw and metric, we only need basic ancpbids entities, 
                # desc and extention add later.
                # But if we take one randow file for this raw, but not this metric, everything gets messed up,
                # some ancp bids magic, talk to developers about it. XD

               
                print('___MEGqc___: ', 'raw_entities_to_write: ', raw_entities_to_write)
                keys = list(raw_entities_to_write.keys())
                for val in keys[:-1]:  # Iterate over all keys except the last one
                    print('___val___: ', val, raw_entities_to_write[val])
                print('___MEGqc___: ', 'raw_info_path: ', raw_info_path)
                print(' ')
                

                # Now prepare the derivative to be written:
                meg_artifact = subject_folder.create_artifact(raw=raw_entities_to_write) 
                #meg_artifact = subject_folder.create_artifact() 
                # create artifact, take entities from entities of the previously calculated tsv derivative

                meg_artifact.add_entity('desc', metric) #add metric to entities
                meg_artifact.suffix = 'meg'
                meg_artifact.extension = '.html'

                deriv = csv_to_html_report(raw_info_path, metric, tsv_paths, report_str_path, plot_settings)

                print('___MEGqc___: ', 'deriv: ', deriv)

                #define method how the derivative will be written to file system:
                meg_artifact.content = lambda file_path, cont=deriv: cont.save(file_path, overwrite=True, open_browser=False)

                counter += 1
                print('counter: ', counter)


    ancpbids.write_derivative(dataset, derivative) 

    return 


# ____________________________
# RUN IT:

# THIS IS NEW VERSION

# make_plots_meg_qc(dataset_path='/data/areer/MEG_QC_stuff/data/openneuro/ds003483')

make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds003483')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds000117')
# make_plots_meg_qc(dataset_path='/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds004330')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/camcan')

# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000246')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000247')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds002761')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds004398')


# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/BIDS/ceegridCut')
