import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import mne
import random
import copy
import os
from typing import List
import matplotlib.pyplot as plt
from mne.preprocessing import compute_average_dev_head_t
from meg_qc.calculation.objects import QC_derivative, MEG_channel
import matplotlib #this is in case we will need to suppress mne matplotlib plots


# mne.viz.set_browser_backend('matplotlib')
# matplotlib.use('Agg') 
#this command will suppress showing matplotlib figures produced by mne. They will still be saved for use in report but not shown when running the pipeline


def get_tit_and_unit(m_or_g: str, psd: bool = False):

    """
    Return title and unit for a given type of data (magnetometers or gradiometers) and type of plot (psd or not)
    
    Parameters
    ----------
    m_or_g : str
        'mag' or 'grad'
    psd : bool, optional
        True if psd plot, False if not, by default False

    Returns
    -------
    m_or_g_tit : str
        'Magnetometers' or 'Gradiometers'
    unit : str
        'T' or 'T/m' or 'T/Hz' or 'T/m / Hz'

    """
    
    if m_or_g=='mag':
        m_or_g_tit='Magnetometers'
        if psd is False:
            unit='Tesla'
        elif psd is True:
            unit='Tesla/Hz'
    elif m_or_g=='grad':
        m_or_g_tit='Gradiometers'
        if psd is False:
            unit='Tesla/m'
        elif psd is True:
            unit='Tesla/m / Hz'
    elif m_or_g == 'eeg':
        m_or_g_tit = 'EEG'
        if psd is False:
            unit = 'microV'
        elif psd is True:
            unit = '(microV)^2/Hz'
    elif m_or_g == 'ECG':
        m_or_g_tit = 'ECG channel'
        unit = 'V'
    elif m_or_g == 'EOG':
        m_or_g_tit = 'EOG channel'
        unit = 'V'
    else:
        m_or_g_tit = '?'
        unit='?'

    return m_or_g_tit, unit


def plot_stim_csv_simple(f_path: str) -> List[QC_derivative]:
    """
    Plot stimulus channels.

    Parameters
    ----------
    f_path : str
        Path to the tsv file with PSD data.

    Returns
    -------
    List[QC_derivative]
        List of QC_derivative objects with plotly figures as content
    """

    df = pd.read_csv(f_path, sep='\t')

    # Check if the first column is just indexes and remove it if necessary
    if df.columns[0] == df.index.name or df.iloc[:, 0].equals(pd.Series(df.index)):
        df = df.drop(df.columns[0], axis=1)

    # Extract the 'time' column for the x-axis
    time = df['time']

    qc_derivatives = []

    # Loop over each column (excluding 'time') and create a separate figure for each
    for i, col in enumerate(df.columns):
        if col != 'time':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=df[col], mode='lines', name=col))
            fig.update_layout(
                title=col,
                title_x=0.5,  # Center the title
                xaxis_title='Time (s)',
                yaxis_title=col,
                showlegend=False
            )

            # Apply description only to the last figure
            qc_derivative = QC_derivative(content=fig, name=f'Stimulus - {col}', content_type='plotly')
            qc_derivatives.append(qc_derivative)

    return qc_derivatives


def plot_stim_csv_colored_leveled(f_path: str) -> List[QC_derivative]:
    """
    Plot stimulus channels.

    Parameters
    ----------
    f_path : str
        Path to the tsv file with PSD data.

    Returns
    -------
    List[QC_derivative]
        List of QC_derivative objects with plotly figures as content
    """

    df = pd.read_csv(f_path, sep='\t')

    # Check if the first column is just indexes and remove it if necessary
    if df.columns[0] == df.index.name or df.iloc[:, 0].equals(pd.Series(df.index)):
        df = df.drop(df.columns[0], axis=1)

    # Extract the 'time' column for the x-axis
    time = df['time']

    qc_derivatives = []

    # Loop over each column (excluding 'time') and create a separate figure for each
    for i, col in enumerate(df.columns):
        if col != 'time':
            y_data = df[col]

            # Check if there are repeating values and exclude 0 values
            unique_values = y_data[y_data > 0].unique()
            if 1 < len(unique_values) <= 30:
                fig = go.Figure()

                # Plot the entire line first
                fig.add_trace(go.Scatter(
                    x=time, y=y_data, mode='lines', name=col,
                    line=dict(color='grey'),  # Default color for the entire line
                    hoverinfo='text',
                    text=[f'Value-{y}, time-{t}s' for y, t in zip(y_data, time)]
                ))

                # Group repeated values and assign colors
                group_ids = {value: idx for idx, value in enumerate(unique_values)}
                for value, group_id in group_ids.items():
                    indices = y_data == value
                    fig.add_trace(go.Scatter(
                        x=time[indices], y=y_data[indices], mode='markers', name=f'ID-{int(value)}',
                        marker=dict(color=f'rgba({group_id * 50 % 255}, {group_id * 100 % 255}, {group_id * 150 % 255}, 1)'),
                        hoverinfo='text',
                        text=[f'ID-{int(value)}, time-{t}s' for t in time[indices]]
                    ))

                fig.update_layout(
                    title=col,
                    title_x=0.5,  # Center the title
                    xaxis_title='Time (s)',
                    yaxis_title='Stimulus ID',
                    showlegend=True,
                    legend=dict(title='Groups', x=1, y=1)
                )
            else:
                # Create the figure as originally
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=y_data, mode='lines', name=col))
                fig.update_layout(
                    title=col,
                    title_x=0.5,  # Center the title
                    xaxis_title='Time (s)',
                    yaxis_title='Stimulus ID',
                    showlegend=False
                )

            # Apply description only to the last figure
            qc_derivative = QC_derivative(content=fig, name=f'Stimulus - {col}', content_type='plotly')
            qc_derivatives.append(qc_derivative)

    return qc_derivatives


def plot_stim_csv(f_path: str) -> List[QC_derivative]:
    """
    Plot stimulus channels.

    Parameters
    ----------
    f_path : str
        Path to the tsv file with PSD data.

    Returns
    -------
    List[QC_derivative]
        List of QC_derivative objects with plotly figures as content
    """

    df = pd.read_csv(f_path, sep='\t')

    #Check if there was stimulus present
    if df.empty:
        return []

    # Check if the first column is just indexes and remove it if necessary
    if df.columns[0] == df.index.name or df.iloc[:, 0].equals(pd.Series(df.index)):
        df = df.drop(df.columns[0], axis=1)

    # Extract the 'time' column for the x-axis
    time = df['time']

    qc_derivatives = []

    # Define a set of bright and appealing colors
    colors = [
        'rgba(255, 99, 71, 1)', 'rgba(135, 206, 250, 1)', 'rgba(255, 215, 0, 1)',
        'rgba(0, 128, 0, 1)', 'rgba(148, 0, 211, 1)', 'rgba(255, 140, 0, 1)',
        'rgba(255, 20, 147, 1)', 'rgba(0, 191, 255, 1)', 'rgba(255, 69, 0, 1)',
        'rgba(50, 205, 50, 1)', 'rgba(138, 43, 226, 1)', 'rgba(255, 105, 180, 1)',
        'rgba(0, 255, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 255, 0, 1)',
        'rgba(75, 0, 130, 1)', 'rgba(255, 165, 0, 1)', 'rgba(255, 0, 255, 1)',
        'rgba(0, 0, 255, 1)', 'rgba(0, 128, 128, 1)', 'rgba(255, 99, 71, 1)',
        'rgba(135, 206, 250, 1)', 'rgba(255, 215, 0, 1)', 'rgba(0, 128, 0, 1)',
        'rgba(148, 0, 211, 1)', 'rgba(255, 140, 0, 1)', 'rgba(255, 20, 147, 1)',
        'rgba(0, 191, 255, 1)', 'rgba(255, 69, 0, 1)', 'rgba(50, 205, 50, 1)'
    ]

    # Loop over each column (excluding 'time') and create a separate figure for each
    for i, col in enumerate(df.columns):
        if col != 'time':
            y_data = df[col]

            # Check if there are repeating values and exclude 0 values
            unique_values = y_data[y_data > 0].unique()
            if 1 < len(unique_values) <= 30:
                fig = go.Figure()

                # Transform y values to 0 (no stimulus) and 1 (all other stimulus IDs)
                transformed_y = y_data.apply(lambda y: 0 if y == 0 else 1)

                # Plot the entire line first
                fig.add_trace(go.Scatter(
                    x=time, y=transformed_y, mode='lines', name=col,
                    line=dict(color='grey'),  # Default color for the entire line
                    hoverinfo='text',
                    text=[f'Value-{y}, time-{t}s' for y, t in zip(y_data, time)]
                ))

                # Group repeated values and assign colors
                group_ids = {value: idx for idx, value in enumerate(unique_values)}
                for value, group_id in group_ids.items():
                    indices = y_data == value
                    fig.add_trace(go.Scatter(
                        x=time[indices], y=transformed_y[indices], mode='markers', name=f'ID-{int(value)}',
                        marker=dict(color=colors[group_id % len(colors)]),
                        hoverinfo='text',
                        text=[f'ID-{int(value)}, time-{t}s' for t in time[indices]]
                    ))

                fig.update_layout(
                    title=col,
                    title_x=0.5,  # Center the title
                    xaxis_title='Time (s)',
                    showlegend=True,
                    legend=dict(title='Stim IDs', x=1, y=1)
                )
            else:
                # Create the figure as originally
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=y_data, mode='lines', name=col))
                fig.update_layout(
                    title=col,
                    title_x=0.5,  # Center the title
                    xaxis_title='Time (s)',
                    showlegend=False
                )

            # Apply description only to the last figure
            qc_derivative = QC_derivative(content=fig, name=f'Stimulus - {col}', content_type='plotly')
            qc_derivatives.append(qc_derivative)

    return qc_derivatives

def plot_ch_df_as_lines_by_lobe_csv(f_path: str, metric: str, x_values, m_or_g, df=None):

    """
    Plots data from a data frame as lines, each lobe has own color.
    Data is taken from previously saved tsv file.

    Parameters
    ----------
    f_path : str
        Path to the csv file with the data to plot.
    metric : str
        The metric of the data to plot: 'psd', 'ecg', 'eog', 'smoothed_ecg', 'smoothed_eog'.
    x_values : List
        List of x values for the plot.
    m_or_g : str
        'mag' or 'grad'.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure.

    """
    if f_path is not None:
        df = pd.read_csv(f_path, sep='\t') #TODO: maybe remove reading csv and pass directly the df here?
    else:
        df = df


    fig = go.Figure()
    traces_lobes=[]
    traces_chs=[]

    add_scores = False #in most cases except ecg/eog we dont add scores to the plot
    if metric.lower() == 'psd':
        col_prefix = 'PSD_Hz_'
    elif metric.lower() == 'ecg':
        col_prefix = 'mean_ecg_sec_'
    elif metric.lower() == 'eog':
        col_prefix = 'mean_eog_sec_'
    elif metric.lower() == 'smoothed_ecg' or metric.lower() == 'ecg_smoothed':
        col_prefix = 'smoothed_mean_ecg_sec_'
        #Need to check if all 3 columns exist in df, are not empty and are not none - if so, add scores to hovertemplate:
        ecg_eog_scores = ['ecg_corr_coeff', 'ecg_pval', 'ecg_amplitude_ratio', 'ecg_similarity_score']
        add_scores = all(column_name in df.columns and not df[column_name].empty and df[column_name].notnull().any() for column_name in ecg_eog_scores)
    elif metric.lower() == 'smoothed_eog' or metric.lower() == 'eog_smoothed':
        col_prefix = 'smoothed_mean_eog_sec_'
        #Need to check if all 3 columns exist in df, are not empty and are not none - if so, add scores to hovertemplate:
        ecg_eog_scores = ['eog_corr_coeff', 'eog_pval', 'eog_amplitude_ratio', 'eog_similarity_score']
        add_scores = all(column_name in df.columns and not df[column_name].empty and df[column_name].notnull().any() for column_name in ecg_eog_scores)
    else:
        print('No proper column in df! Check the metric!')

   
    for index, row in df.iterrows():

        if row['Type'] == m_or_g: #plot only mag/grad
            ch_data = []
            for col in df.columns:
                if col_prefix in col:

                    #ch_data = row[col] #or maybe 
                    ch_data.append(row[col])

                    # normally color must be same for all channels in lobe, so we could assign it before the loop as the color of the first channel,
                    # but here it is done explicitly for every channel so that if there is any color error in chs_by_lobe, it will be visible
            
            color = row['Lobe Color']

            #traces_chs += [go.Scatter(x=x_values, y=ch_data, line=dict(color=color), name=row['Name'] , legendgroup=row['Lobe'] , legendgrouptitle=dict(text=row['Lobe'].upper(), font=dict(color=color)))]


            if add_scores:

                traces_chs += [go.Scatter(
                    x=x_values, 
                    y=ch_data, 
                    line=dict(color=color), 
                    name=row['Name'],
                    legendgroup=row['Lobe'],
                    legendgrouptitle=dict(text=row['Lobe'].upper(), font=dict(color=color)),

                    hovertemplate = (
                    '<b>'+row['Name']+'</b><br>' +
                    'time: %{x} s<br>'+
                    'magnitude: %{y} T<br>' +
                    '<i>corr_coeff: </i>'+'{:.2f}'.format(row[ecg_eog_scores[0]])+'<br>' +
                    '<i>p-value: </i>'+str(row[ecg_eog_scores[1]])+'<br>' +
                    '<i>amplitude_ratio: </i>'+'{:.2f}'.format(row[ecg_eog_scores[2]])+'<br>' +
                    '<i>similarity_score: </i>'+'{:.2f}'.format(row[ecg_eog_scores[3]])+'<br>'
                ))]
            else:
                traces_chs += [go.Scatter(
                    x=x_values, 
                    y=ch_data, 
                    line=dict(color=color), 
                    name=row['Name'],
                    legendgroup=row['Lobe'],
                    legendgrouptitle=dict(text=row['Lobe'].upper(), font=dict(color=color))
                )]
               
    # sort traces in random order: WHY?
    # When you plot traves right away in the order of the lobes, all the traces of one color lay on top of each other and yu can't see them all.
    # This is why they are not plotted in the loop. So we sort them in random order, so that traces of different colors are mixed.
    traces = traces_lobes + sorted(traces_chs, key=lambda x: random.random())

    if not traces:
        return None

    # Now first add these traces to the figure and only after that update the layout to make sure that the legend is grouped by lobe.
    fig = go.Figure(data=traces)

    fig.update_layout(legend_traceorder='grouped', legend_tracegroupgap=12, legend_groupclick='toggleitem')
    #You can make it so when you click on lobe title or any channel in lobe you activate/hide all related channels if u set legend_groupclick='togglegroup'.
    #But then you cant see individual channels, it turn on/off the whole group. There is no option to tun group off by clicking on group title. Grup title and group items behave the same.

    #to see the legend: there is really nothing to sort here. The legend is sorted by default by the order of the traces in the figure. The onl way is to group the traces by lobe.
    #print(fig['layout'])

    #https://plotly.com/python/reference/?_ga=2.140286640.2070772584.1683497503-1784993506.1683497503#layout-legend-traceorder
    

    return fig


def switch_names_on_off(fig: go.Figure):

    """
    Switches between showing channel names when hovering and always showing channel names.
    
    Parameters
    ----------
    fig : go.Figure
        The figure to be modified.
        
    Returns
    -------
    fig : go.Figure
        The modified figure.
        
    """

    # Define the buttons
    buttons = [
    dict(label='Show channels names on hover',
         method='update',
         args=[{'mode': 'markers'}]),
    dict(label='Always show channels names',
         method='update',
         args=[{'mode': 'markers+text'}])
    ]

    # Add the buttons to the layout
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        showactive=True,
                                        buttons=buttons)])

    return fig


def keep_unique_locs(ch_list: List):

    """
    Combines channel names that have the same location and returns the unique locations and combined channel names for 3D plotting.

    Parameters
    ----------
    ch_list : List
        A list of channel objects.

    Returns
    -------
    new_locations : List
        A list of unique locations.
    new_names : List
        A list of combined channel names.
    new_colors : List
        A list of colors for each unique location.
    new_lobes : List
        A list of lobes for each unique location.

    """


    channel_names = [ch.name for ch in ch_list]
    channel_locations = [ch.loc for ch in ch_list]
    channel_colors = [ch.lobe_color for ch in ch_list]
    channel_lobes = [ch.lobe for ch in ch_list]

    # Create dictionaries to store unique locations and combined channel names
    unique_locations = {}
    combined_names = {}
    unique_colors = {}
    unique_lobes = {}

    # Loop through each channel and its location
    for i, (name, loc, color, lobe) in enumerate(zip(channel_names, channel_locations, channel_colors, channel_lobes)):
        # Convert location to a tuple for use as a dictionary key
        loc_key = tuple(loc)
        
        # If location is already in the unique_locations dictionary, add channel name to combined_names
        if loc_key in unique_locations:
            combined_names[unique_locations[loc_key]].append(name)
        # Otherwise, add location to unique_locations and channel name to combined_names
        else:
            unique_locations[loc_key] = i
            combined_names[i] = [name]
            unique_colors[i] = color
            unique_lobes[i] = lobe

    # Create new lists of unique locations and combined channel names
    new_locations = list(unique_locations.keys())
    new_names = [' & '.join(combined_names[i]) for i in combined_names]
    new_colors = [unique_colors[i] for i in unique_colors]
    new_lobes = [unique_lobes[i] for i in unique_lobes]

    return new_locations, new_names, new_colors, new_lobes 


def make_3d_sensors_trace(d3_locs: List, names: List, color: str, textsize: int, legend_category: str = 'channels', symbol: str = 'circle', textposition: str = 'top right'):

    """ 
    Makes traces for sensors in 1 lobe, one color. Names and locations are combined if the sonsors have same coordinates.
    This func already gets them combined from keep_unique_locs function.
    (Since grads have 2 sensors located in the same spot - need to put their names together to make pretty plot label. 
    Mags are located aproxximately in the same place).

    Parameters
    ----------
    d3_locs : List
        A list of 3D locations of the sensors.
    names : List
        A list of names of the sensors.
    color : str
        A color of the sensors.
    textsize : int
        A size of the text.
    ch_type : str
        A type of the channels.
    symbol : str
        A symbol of the sensors.
    textposition : str
        A position of the text.
    
    Returns
    -------
    trace : plotly.graph_objs._scatter3d.Scatter3d
        A trace of the sensors.
    
    
    """
    if not d3_locs:
        print('__MEGqc__: No sensors locations to plot!')
        return None
    
    trace = go.Scatter3d(
    x=[loc[0] for loc in d3_locs],
    y=[loc[1] for loc in d3_locs],
    z=[loc[2] for loc in d3_locs],
    mode='markers',
    marker=dict(
        color=color,
        size=8,
        symbol=symbol,
    ),
    text=names,
    hoverinfo='text',
    name=legend_category,
    textposition=textposition,
    textfont=dict(size=textsize, color=color))

    return trace


def get_meg_system(sensors_df):

    """
    Get which meg system we work with from the df. Make sure there is only 1 system.
    
    """
    
    # Get unique values, avoiding NaNs and empty strings
    system = sensors_df['System'].dropna().unique().tolist()
    system = [s for s in system if s != '']

    # Check the number of unique values
    if len(system) == 1:
        result = system[0]
    else:
        result = 'OTHER'

    return result

def plot_sensors_3d_csv(sensors_csv_path: str):

    """
    Plots the 3D locations of the sensors in the raw file. 
    Plot both mags and grads (if both present) in 1 figure. 
    Can turn mags/grads visialisation on and off.
    Separete channels into brain areas by color coding.

    Plot is made on base of the tsv file with sensors locations.


    Parameters
    ----------
    sensors_csv_path : str
        Path to the tsv file with the sensors locations.
    
    Returns
    -------
    qc_derivative : List
        A list of QC_derivative objects containing the plotly figures with the sensor locations.

    """
    file_name = os.path.basename(sensors_csv_path)
    if 'ecgchannel' in file_name.lower() or 'eogchannel' in file_name.lower():
        return []
    #we will get tsv representing ECG/EOG channel itself landed here. We dont need to plot it with this func.

    df = pd.read_csv(sensors_csv_path, sep='\t')

    #double check: if there are no lobes in df - skip this plot, it s not the right df:
    if 'Lobe' not in df.columns or 'System' not in df.columns:
        return []

    system = get_meg_system(df)

    if system.upper() == 'TRIUX':
        fig_desc = "Magnetometers names end with '1' like 'MEG0111'. Gradiometers names end with '2' and '3' like 'MEG0112', 'MEG0113'."
    elif system.upper() == 'EEG':
        fig_desc = "EEG Channels"
    else:
        fig_desc = ""

    #to not rewrite the whole func, just turn the df back into dic of MEG_channel:
    
    unique_lobes = df['Lobe'].unique().tolist()

    lobes_dict={}
    for lobe in unique_lobes:
        lobes_dict[lobe] = []
        for index, row in df.iterrows():
            if row['Lobe'] == lobe:
                locs = [float(row[col]) for col in df.columns if 'Sensor_location' in col]
                lobes_dict[lobe].append(MEG_channel(name = row['Name'], type = row['Type'], lobe = row['Lobe'], lobe_color = row['Lobe Color'], system = row ['System'], loc = locs))

    traces = []

    #system = df['System'].unique().tolist()

    if len(lobes_dict)>1: 
        #if there are lobes - we use color coding: one color per each lobe
        for lobe in lobes_dict:
            ch_locs, ch_names, ch_color, ch_lobe = keep_unique_locs(lobes_dict[lobe])
            traces.append(make_3d_sensors_trace(ch_locs, ch_names, ch_color[0], 10, ch_lobe[0], 'circle', 'top left'))
            #here color and lobe must be identical for all channels in 1 trace, thi is why we take the first element of the list
            # TEXT SIZE set to 10. This works for the "Always show names" option but not for "Show names on hover" option

    else: 
        #if there are no lobes - the dict will only have one lobe as the key and all channels inside of it
        # we use random colors previously assigned to channels, channel names will be used instead of lobe names in make_3d_trace function
        ch_locs, ch_names, ch_color, ch_lobe = keep_unique_locs(lobes_dict[lobe])
        for i, _ in enumerate(ch_locs):
            traces.append(make_3d_sensors_trace([ch_locs[i]], ch_names[i], ch_color[i], 10, ch_names[i], 'circle', 'top left'))

    if not traces:
        return []
    
    fig = go.Figure(data=traces)

    fig.update_layout(
        width=900, height=900,
        title={
        'text': 'Sensors positions',
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    
    fig.update_layout(
        scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )


    # Add the button to have names show up on hover or always:
    fig = switch_names_on_off(fig)

    fig.update_traces(hoverlabel=dict(font=dict(size=10))) #TEXT SIZE set to 10 again. This works for the "Show names on hover" option, but not for "Always show names" option
    
    qc_derivative = [QC_derivative(content=fig, name='Sensors_positions', content_type='plotly', description_for_user=fig_desc, fig_order=-1)]

    return qc_derivative 


def boxplot_epochs(df_mg: pd.DataFrame, ch_type: str, what_data: str, x_axis_boxes: str):

    """
    Creates representation of calculated data as multiple boxplots. Used in STD and PtP_manual measurements. 

    - If x_axis_boxes is 'channels', each box represents 1 epoch, each dot is std of 1 channel for this epoch
    - If x_axis_boxes is 'epochs', each box represents 1 channel, each dot is std of 1 epoch for this channel

    
    Parameters
    ----------
    df_mg : pd.DataFrame
        Data frame with std or peak-to-peak values for each channel and epoch. Columns are epochs, rows are channels.
    ch_type : str
        Type of the channel: 'mag', 'grad'
    what_data : str
        Type of the data: 'peaks' or 'stds'
    x_axis_boxes : str
        What to plot as boxplot on x axis: 'channels' or 'epochs'

    Returns
    -------
    fig_deriv : QC_derivative 
        derivative containing plotly figure
    
    """

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_tit
    else:
        print('what_data should be either peaks or stds')

    if x_axis_boxes=='channels':
        #transpose the data to plot channels on x axes
        df_mg = df_mg.T
        legend_title = ''
        hovertemplate='Epoch: %{text}<br>'+hover_tit+': %{y: .2e}'
    elif x_axis_boxes=='epochs':
        legend_title = 'Epochs'
        hovertemplate='%{text}<br>'+hover_tit+': %{y: .2e}'
    else:
        print('x_axis_boxes should be either channels or epochs')

    #collect all names of original df into a list to use as tick labels:
    boxes_names = df_mg.columns.tolist() #list of channel names or epoch names
    #boxes_names=list(df_mg) 

    fig = go.Figure()

    for col in df_mg:
        fig.add_trace(go.Box(y=df_mg[col].values, 
        name=str(df_mg[col].name), 
        opacity=0.7, 
        boxpoints="all", 
        pointpos=0,
        marker_size=3,
        line_width=1,
        text=df_mg[col].index,
        ))
        fig.update_traces(hovertemplate=hovertemplate)

    
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(boxes_names))],
            ticktext = boxes_names,
            rangeslider=dict(visible=True)
        ),
        xaxis_title='Experimental epochs',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title=legend_title)
        

    fig_deriv = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return fig_deriv


def boxplot_epoched_xaxis_channels_csv(std_csv_path: str, ch_type: str, what_data: str):

    """
    Creates representation of calculated data as multiple boxplots. Used in STD and PtP_manual measurements. 
    Color tagged channels by lobes. 
    One box is one channel, boxes are on x axis. Epoch are inside as dots. Y axis shows the STD/PtP value.

    On base of the data from tsv file.
    
    Parameters
    ----------
    std_csv_path: str
        Path to the tsv file with std data
    ch_type : str
        Type of the channel: 'mag', 'grad'
    what_data : str
        Type of the data: 'peaks' or 'stds'

    Returns
    -------
    fig_deriv : QC_derivative 
        derivative containing plotly figure
    
    """
    df = pd.read_csv(std_csv_path, sep='\t')

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PtP Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_tit
        data_prefix = 'PtP epoch_'
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_tit
        data_prefix = 'STD epoch_'
    else:
        print('what_data should be either peaks or stds')
        return []


    #Check if df has relevant data for plotting:
    #find columns with epochs:
    relevant_columns = [col for col in df.columns if data_prefix in col]

    # Filter rows where 'Type' is the one we need: mag, grad
    filtered_df = df[df['Type'] == ch_type]

    # Check if all relevant cells are empty
    all_empty = filtered_df[relevant_columns].isnull().all().all()

    if all_empty:
        return []


    # Figure column names:
    # Create a list of columns that start with 'STD epoch_'
    epoch_columns = [col for col in df.columns if col.startswith('STD epoch_') or col.startswith('PtP epoch_')]

    # Get the number of these columns
    num_epoch_columns = len(epoch_columns)

    # Create a list of numbers from 0 to that length
    epochs_names = [i for i in range(num_epoch_columns)]


    x_axis_boxes = 'channels'
    if x_axis_boxes=='channels':
        hovertemplate='Epoch: %{text}<br>'+hover_tit+': %{y: .2e}'
    elif x_axis_boxes=='epochs':
        #legend_title = 'Epochs'
        hovertemplate='%{text}<br>'+hover_tit+': %{y: .2e}'
    else:
        print('x_axis_boxes should be either channels or epochs')
        return []


    fig = go.Figure()

    #Here each trace is 1 box representing 1 channel. Epochs inside the box are automatically plotted given argument boxpoints="all":
    #Boxes are groupped by lobe. So first each channel fo lobe 1 is plotted, then each of lobe 2, etc..
    boxes_names = []


    for index, row in df.iterrows():
        if row['Type'] == ch_type: #plot only mag/grad
            
            data = [row[data_prefix+str(n)] for n in epochs_names]

            boxes_names += [row['Name']]

            fig.add_trace(go.Box(y=data, 
            name=row['Name'], 
            opacity=0.7, 
            boxpoints="all", 
            pointpos=0,
            marker_color=row['Lobe Color'],
            marker_size=3,
            legendgroup=row['Lobe'], 
            legendgrouptitle=dict(text=row['Lobe'].upper()),
            line_width=0.8,
            line_color=row['Lobe Color'],
            text=epochs_names))

    
    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(boxes_names))],
            ticktext = boxes_names,
            rangeslider=dict(visible=True)),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},)
        #legend_title=legend_title)

    fig_deriv = [QC_derivative(content=fig, name=fig_name, content_type='plotly')]

    return fig_deriv


def plot_topomap_std_ptp_csv(std_csv_path: str, ch_type: str, what_data: str):

    """
    
    Plot STD using mne.viz.plot_topomap(data, pos, *, ch_type='mag', sensors=True, names=None) 
    
    For every channel we take STD/PtP value and plot as topomap
    """

    #First, convert scv back into dict with MEG_channel objects:

    df = pd.read_csv(std_csv_path, sep='\t')

    if "_eeg." in std_csv_path:
        ch_type = ['eeg']

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_all_data_Topomap_'+ch_tit
    elif what_data=='stds':
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_all_data_Topomap_'+ch_tit
    else:
        raise ValueError('what_data must be set to "stds" or "peaks"')

    
    data = []
    names = []
    pos = np.empty((0, 2))

    for index, row in df.iterrows():

        if row['Type'] == ch_type: #plot only mag/grad

            if what_data == 'stds':
                data += [row['STD all']]
            elif what_data == 'peaks':
                data += [row['PtP all']]
                # data: array, shape (n_chan,) The data values to plot.


            # Filter columns containing 'Sensor_location'
            sensor_location_cols = [col for col in df.columns if 'Sensor_location' in col]
            #pos = np.array([[float(row[col]) for col in sensor_location_cols[:2]] for _, row in df.iterrows()])
            new_pos = np.array([[float(row[col]) for col in sensor_location_cols[:2]]])  # Ensure 2D shape
            # Append the new row to pos
            pos = np.vstack([pos, new_pos])

            names += [row['Name']]

    #convert data to array:
    data = np.array(data)
    mask = np.array([True for i in range(len(names))])

    mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',
        linewidth=0, markersize=3)
    # mask is to change marker look of channels. applied to everything here, 
    # but in principle it is used to highlight particular channels

    fig, ax = plt.subplots(figsize=(8, 6))  # Create a Matplotlib figure and axes

    mne.viz.plot_topomap(
    data, pos, ch_type=ch_type, names=names, size=6, mask=mask,
    mask_params=mask_params, show=False, axes=ax, sphere=(0., 0., 0., 0.1)  
    )

    # It will give warning; 'invalid value encountered in divide'
    # This is most likely because for grads positions are the same, 
    # so the division by 0 occurs. But this is normal. 
    # Can add some jitter to posoitions if this bothers a lot.


    ax.set_title(f'{y_ax_and_fig_title} Topomap for {ch_tit}')

    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='matplotlib')]
                 
    return qc_derivative


def plot_3d_topomap_std_ptp_csv(sensors_csv_path: str, ch_type: str, what_data: str):

    """
    Plot the topomap of STP/PtP values (values take over all time, not epoched).
    One dot reperesnt 1 channel. Dots are colored from blue (lowest std/ptp) to red (highest).
    Plots is intereactive 3d with hovering labels.
    If we got gradiometers - 2 channels usually have same locations - values will be combined.
    See comemnts in the code below for this case.

    Parameters
    ----------
    sensors_csv_path : str
        Path to the tsv file with the sensors locations.
    ch_type : str
        Type of the channels: mag or grad
    what_data : str
        'peaks' or 'stds'
    
    Returns
    -------
    qc_derivative : List
        A list of QC_derivative objects containing the plotly figures with the sensor locations.

    """
    
    df = pd.read_csv(sensors_csv_path, sep='\t')

    #take only those channels that are of right type:
    df = df[df['Type'] == ch_type]

    ch_tit, unit = get_tit_and_unit(ch_type)


    if what_data=='peaks':
        fig_name='PP_manual_all_data_Topomap_'+ch_tit
        metric = 'PtP'
        metric_column = 'PtP all'
    elif what_data=='stds':
        fig_name='STD_epoch_all_data_Topomap_'+ch_tit
        metric = 'STD'
        metric_column = 'STD all'
    else:
        raise ValueError('what_data must be set to "stds" or "peaks"')
    

    # Create a DataFrame with sensor locations as columns
    sensor_df = df[['Sensor_location_0', 'Sensor_location_1', 'Sensor_location_2', metric_column, 'Name']]

    # Group by sensor locations, this is done in case we got GRADIOMETERS,
    # cos they have 2 sensors located in the same spot.
    # We calculate mean value for each group: the mean of 'STD all' or 'PtP all' columns for 2 channels,
    # this mean value will be used to define the color. 
    # We assume that means std/ptp of physically close to each other gardiomeeters is also close in value.
    # Here also create groupped names, later used in hover 
    grouped = sensor_df.groupby(['Sensor_location_0', 'Sensor_location_1', 'Sensor_location_2']).agg({
        metric_column: 'mean',
        'Name': lambda x: ', '.join([f"{name} - {metric}: {std:.2e} {unit}" for name, std in zip(x, sensor_df.loc[x.index, metric_column])])
    }).reset_index()

    # Extract the grouped sensor locations and mean metric values
    grouped_sensor_locations = grouped[['Sensor_location_0', 'Sensor_location_1', 'Sensor_location_2']].values
    mean_metric_values = grouped[metric_column].values
    grouped_names = grouped['Name'].values

    # Create the 3D scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=grouped_sensor_locations[:, 0],
        y=grouped_sensor_locations[:, 1],
        z=grouped_sensor_locations[:, 2],
        mode='markers',
        marker=dict(
            size=13,
            color=mean_metric_values,  # Use the mean metric values for the color scale
            colorscale='Bluered',  # Use the 'Bluered' colorscale
            colorbar=dict(
                title=dict(
                    text=f'{metric}, {unit}',  # el texto del título
                    side='right'  # 'right' / 'top' / 'bottom'
                ),
                tickmode='array',
                tickvals=[np.min(mean_metric_values), np.max(mean_metric_values)],
                ticktext=[f'{np.min(mean_metric_values):.2e}', f'{np.max(mean_metric_values):.2e}'],
                ticks='outside'
            ),
            opacity=0.8
        ),
        text=grouped_names,  # Use channel names and formatted mean metric values as hover text
        hoverinfo='text'
    ))

    # Set plot layout
    fig.update_layout(
        width=900, height=900,
        title={
        'text': metric + ' topomap: '+ ch_tit,
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    
    fig.update_layout(
        scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )

    fig.update_traces(hoverlabel=dict(font=dict(size=10))) #TEXT SIZE set to 10 again. This works for the "Show names on hover" option, but not for "Always show names" option

    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='plotly')]

    return qc_derivative


# ______________________PSD__________________________

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


def figure_x_axis(df, metric):

    """
    Figure out the x axis for plotting based on the metric.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame with the data to be plotted.
    metric : str
        The metric of the data: 'psd', 'eog', 'ecg', 'muscle', 'head'.

    Returns
    -------
    freqs : np.array
        Array of frequencies for the PSD data.
    time_vec : np.array
        Array of time values for the EOG, ECG, muscle, or head data.
    
    """
     
    metric_lower = metric.lower()

    if metric_lower == 'psd':
        # Figure out frequencies:
        freq_cols = [column for column in df if column.startswith('PSD_Hz_')]
        freqs = np.array([float(x.replace('PSD_Hz_', '')) for x in freq_cols])
        return freqs

    prefix_map = {
        'ecg': 'mean_ecg_sec_',
        'smoothed_ecg': 'smoothed_mean_ecg_sec_',
        'smoothed_eog': 'smoothed_mean_eog_sec_',
        'eog': 'mean_eog_sec_',
        'muscle': 'Muscle_sec_',
        'head': 'Head_sec_'
    }

    if metric_lower in prefix_map:
        prefix = prefix_map[metric_lower]
        time_cols = [column for column in df if column.startswith(prefix)]
        time_vec = np.array([float(x.replace(prefix, '')) for x in time_cols])
        return time_vec

    print('Wrong metric! Cant figure out xaxis for plotting.')

    return None


def Plot_psd_csv(m_or_g:str, f_path: str, method: str):

    """
    Plotting Power Spectral Density for all channels based on dtaa from tsv file.

    Parameters
    ----------
    m_or_g : str
        'mag' or 'grad'
    f_path : str
        Path to the tsv file with PSD data.
    method : str
        'welch' or 'multitaper' or other method

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content
        
    """

    # First, get the epochs from csv and convert back into object.
    df = pd.read_csv(f_path, sep='\t') 

    if 'Name' not in df.columns:
        return []

    # Figure out frequencies:
    freqs = figure_x_axis(df, metric='psd')

    #TODO: DF with freqs still has redundand columns with names of frequencies like column.startswith('Freq_')
    # Remove them!

    channels = []
    for index, row in df.iterrows():
        channels.append(row['Name'])

    fig = plot_ch_df_as_lines_by_lobe_csv(f_path, 'psd', freqs, m_or_g)

    if fig is None:
        return []

    tit, unit = get_tit_and_unit(m_or_g)
    fig.update_layout(
    title={
    'text': method[0].upper()+method[1:]+" periodogram for all "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    yaxis_title="Amplitude, "+unit,
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'),
    xaxis_title="Frequency (Hz)")

    fig.update_traces(hovertemplate='Frequency: %{x} Hz<br>Amplitude: %{y: .2e} T/Hz')

    #Add buttons to switch scale between log and linear:
    fig = add_log_buttons(fig)
    
    fig_name='PSD_all_data_'+tit

    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='plotly')]

    return qc_derivative



def edit_legend_pie_SNR(noisy_freqs: List, noise_ampl: List, total_amplitude: float, noise_ampl_relative_to_signal: List):

    """
    Edit the legend for pie chart of signal to noise ratio.

    Parameters
    __________

    noisy_freqs: List
        list of noisy frequencies
    noise_ampl: List
        list of their amplitudes
    total_amplitude: float
        Total amplitude of all frequencies
    noise_ampl_relative_to_signal: List
        list of relative (to entire signal) values of noise freq's amplitude

    Returns
    -------
    noise_and_signal_ampl:
        list of amplitudes of noise freqs + total signal amplitude
    noise_ampl_relative_to_signal:
        list of relative noise freqs + amplitude of clean signal
    bands_names:
        names of freq bands 
    
    """
    

    #Legend for the pie chart:

    bands_names=[]
    if noisy_freqs == [0]:
        noisy_freqs, noise_ampl, noise_ampl_relative_to_signal = [], [], []
        #empty lists so they dont show up on pie chart
    else:
        for fr_n, fr in enumerate(noisy_freqs):
            bands_names.append(str(round(fr,1))+' Hz noise')

    bands_names.append('Main signal')
    
    noise_and_signal_ampl = noise_ampl.copy()
    noise_and_signal_ampl.append(total_amplitude-sum(noise_ampl)) #adding main signal ampl in the list

    noise_ampl_relative_to_signal.append(1-sum(noise_ampl_relative_to_signal)) #adding main signal relative ampl in the list

    return  noise_and_signal_ampl, noise_ampl_relative_to_signal, bands_names


def plot_pie_chart_freq_csv(tsv_pie_path: str, m_or_g: str, noise_or_waves: str):
    
    """
    Plot pie chart representation of relative amplitude of each frequency band over the entire 
    times series of mags or grads, not separated by individual channels.

    Parameters
    ----------
    tsv_pie_path: str
        Path to the tsv file with pie chart data
    m_or_g : str
        'mag' or 'grad'
    noise_or_waves: str
        do we plot SNR or brain waves percentage (alpha, beta, etc)
    
    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """

    #if it s not the right ch kind in the file
    base_name = os.path.basename(tsv_pie_path) #name of the final file
    
    if m_or_g not in base_name.lower():
        return []
    
    # Read the data from the TSV file into a DataFrame
    df = pd.read_csv(tsv_pie_path, sep='\t')

    if m_or_g == 'eeg':
        stype = 'mag'
    else:
        stype = m_or_g

    if noise_or_waves == 'noise' and 'PSDnoise' in base_name:
        #check that we input tsv file with the right data

        fig_tit = "Ratio of signal and noise in the data: " 
        fig_name = 'PSD_SNR_all_channels_'

        # Extract the data
        total_amplitude = df['total_amplitude_'+stype].dropna().iloc[0]  # Get the first non-null value
        noisy_freqs = df['noisy_freqs_'+stype].tolist()

        noise_ampl = df['noise_ampl_'+stype].tolist()
        amplitudes_relative = df['noise_ampl_relative_to_signal_'+stype].tolist()

        amplitudes_abs, amplitudes_relative, bands_names = edit_legend_pie_SNR(noisy_freqs, noise_ampl, total_amplitude, amplitudes_relative)

    elif noise_or_waves == 'waves' and 'PSDwaves' in base_name:

        fig_tit = "Relative area under the amplitude spectrum: " 
        fig_name = 'PSD_Relative_band_amplitude_all_channels_'


        # Set the first column as the index
        df.set_index(df.columns[0], inplace=True)

        # Extract total_amplitude into a separate variable
        total_amplitude = df['total_amplitude'].loc['absolute_'+stype]

        #drop total ampl:
        df_no_total = copy.deepcopy(df.drop('total_amplitude', axis=1))

        # Extract rows into lists
        amplitudes_abs = df_no_total.loc['absolute_'+stype].tolist()
        amplitudes_relative = df_no_total.loc['relative_'+stype].tolist()

        # Extract column names into a separate list
        bands_names = df_no_total.columns.tolist()

    else:
        return []

    all_bands_names=bands_names.copy() 
    #the lists change in this function and this change is tranfered outside the fuction even when these lists are not returned explicitly. 
    #To keep them in original state outside the function, they are copied here.
    all_mean_abs_values=amplitudes_abs.copy()
    ch_type_tit, unit = get_tit_and_unit(stype, psd=True)

    #If mean relative percentages dont sum up into 100%, add the 'unknown' part.
    all_mean_relative_values=[v * 100 for v in amplitudes_relative]  #in percentage
    relative_unknown=100-(sum(amplitudes_relative))*100

    if relative_unknown>0:
        all_mean_relative_values.append(relative_unknown)
        all_bands_names.append('other frequencies')
        all_mean_abs_values.append(total_amplitude - sum(all_mean_abs_values))


    if not all_mean_relative_values:
        return []
    
    labels=[None]*len(all_bands_names)
    for n, name in enumerate(all_bands_names):
        labels[n]=name + ': ' + str("%.2e" % all_mean_abs_values[n]) + ' ' + unit # "%.2e" % removes too many digits after coma

        #if some of the all_mean_abs_values are zero - they should not be shown in pie chart:

    fig = go.Figure(data=[go.Pie(labels=labels, values=all_mean_relative_values)])
    fig.update_layout(
    title={
    'text': fig_tit + ch_type_tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

    fig_name=fig_name+ch_type_tit

    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='plotly')]

    return qc_derivative


def assign_epoched_std_ptp_to_channels(what_data, chs_by_lobe, df_std_ptp):

    """
    Assign std or ptp values of each epoch as list to each channel. 
    This is done for easier plotting when need to plot epochs per channel and also color coded by lobes.
    
    Parameters
    ----------
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe.
    df_std_ptp : pd.DataFrame
        Data Frame containing std or ptp value for each chnnel and each epoch
    
        
    Returns
    -------
    chs_by_lobe : dict
        updated dictionary with channel objects sorted by lobe - with info about std or ptp of epochs.
    """

    if what_data=='peaks':
        #Add the data about std of each epoch (as a list, 1 std for 1 epoch) into each channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe:
            for ch in chs_by_lobe[lobe]:
                ch.ptp_epoch = df_std_ptp.loc[ch.name].values
    elif what_data=='stds':
        for lobe in chs_by_lobe:
            for ch in chs_by_lobe[lobe]:
                ch.std_epoch = df_std_ptp.loc[ch.name].values
    else:
        print('what_data should be either peaks or stds')

    return chs_by_lobe


def boxplot_epoched_xaxis_epochs_csv(std_csv_path: str, ch_type: str, what_data: str):

    """
    Represent std of epochs for each channel as box plots, where each box on x axis is 1 epoch. Dots inside the box are channels.
    On base of the data from tsv file
    
    Process: 
    Each box need to be plotted as a separate trace first.
    Each channels inside each box has to be plottted as separate trace to allow diffrenet color coding
    
    For each box_representing_epoch:
        box trace
        For each color coded lobe:
            For each dot_representing_channel in lobe:
                dot trace

    Add all traces to plotly figure


    Parameters
    ----------
    std_csv_path: str
        Path to the tsv file with std data
    ch_type : str
        'mag' or 'grad'
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """

    # First, get the epochs from csv and convert back into object.
    df = pd.read_csv(std_csv_path, sep='\t')  

    # Figure column names:
    # Create a list of columns that start with 'STD epoch_'
    epoch_columns = [col for col in df.columns if col.startswith('STD epoch_') or col.startswith('PtP epoch_')]

    # Extract the actual epoch names from the column names
    epochs_names = [int(col.split('_')[-1]) for col in epoch_columns]

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PtP Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_2_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_2_'+ch_tit
    else:
        print('what_data should be either peaks or stds')


    boxwidth=0.5 #the area around which the data dots are scattered depends on the width of the box.

    # For this plot have to separately create a box (no data points plotted) as 1 trace
    # Then separately create for each cannel (dot) a separate trace. It s the only way to make them all different lobe colors.
    # Additionally, the dots are scattered along the x axis inside each box, this is done for visualisation only, x position does not hold information.
    
    # Put all data dots in a list of traces groupped by lobe:
    
    dot_traces = []
    box_traces = []    

    for ep in epochs_names:
        dots_in_1_box=[]
        for index, row in df.iterrows():

            if row['Type'] == ch_type: #plot only mag/grad

                if what_data == 'stds':
                    data = row['STD epoch_' + str(ep)]
                elif what_data == 'peaks':
                    data = row['PtP epoch_'+ str(ep)]
                else:
                    raise ValueError('what_data should be either peaks or stds')    

                dots_in_1_box += [data]

                x = ep + random.uniform(-0.2*boxwidth, 0.2*boxwidth) 
                #here create random y values for data dots, they dont have a meaning, just used so that dots are scattered around the box plot and not in 1 line.
                
                dot_traces += [go.Scatter(x=[x], y=[data], mode='markers', marker=dict(size=4, color=row['Lobe Color']), opacity=0.8, name=row['Name'], text=str(ep), legendgroup=row['Lobe'], legendgrouptitle=dict(text=row['Lobe'].upper()), hovertemplate='Epoch: '+str(ep)+'<br>'+hover_tit+': %{y: .2e}')]

        # create box plot trace
        box_traces += [go.Box(x0=ep, y=dots_in_1_box, orientation='v', name=ep, line_width=1.8, opacity=0.8, boxpoints=False, width=boxwidth, showlegend=False)]
    
    #Collect all traces and add them to the figure:

    all_traces = box_traces+dot_traces

    if not dot_traces:
        return []
    
    fig = go.Figure(data=all_traces)
        
    #more settings:
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(epochs_names))],
            ticktext = epochs_names,
            rangeslider=dict(visible=True)
        ),
        xaxis_title='Experimental epochs',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_groupclick='togglegroup') #this setting allowes to select the whole group when clicking on 1 element of the group. But then you can not select only 1 element.

    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='plotly')]

    return qc_derivative


def boxplot_all_time_csv(std_csv_path: str, ch_type: str, what_data: str):

    """
    Create representation of calculated std data as a boxplot over the whoe time series, not epoched.
    (box contains magnetometers or gradiomneters, not together): 
    each dot represents 1 channel (std value over whole data of this channel). Too high/low stds are outliers.

    On base of the data from tsv file.

    Parameters
    ----------
    std_csv_path: str
        Path to the tsv file with std data.
    ch_type : str
        'mag' or 'grad'
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """

    #First, convert scv back into dict with MEG_channel objects:

    df = pd.read_csv(std_csv_path, sep='\t')  

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PP_Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_all_data_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_all_data_'+ch_tit
    else:
        raise ValueError('what_data must be set to "stds" or "peaks"')

    boxwidth=0.4 #the area around which the data dots are scattered depends on the width of the box.

    # For this plot have to separately create a box (no data points plotted) as 1 trace
    # Then separately create for each cannel (dot) a separate trace. It s the only way to make them all different lobe colors.
    # Additionally, the dots are scattered along the y axis, this is done for visualisation only, y position does not hold information.
    
    # Put all data dots in a list of traces groupped by lobe:
    values_all=[]
    traces = []

    for index, row in df.iterrows():

        if row['Type'] == ch_type: #plot only mag/grad

            if what_data == 'stds':
                data = row['STD all']
            elif what_data == 'peaks':
                data = row['PtP all']

            values_all += [data]

            y = random.uniform(-0.2*boxwidth, 0.2*boxwidth) 
            #here create random y values for data dots, they dont have a meaning, just used so that dots are scattered around the box plot and not in 1 line.
            
            traces += [go.Scatter(x=[data], y=[y], mode='markers', marker=dict(size=5, color=row['Lobe Color']), name=row['Name'], legendgroup=row['Lobe'], legendgrouptitle=dict(text=row['Lobe'].upper()))]


    # create box plot trace
    box_trace = go.Box(x=values_all, y0=0, orientation='h', name='box', line_width=1, opacity=0.7, boxpoints=False, width=boxwidth, showlegend=False)
    
    #Colllect all traces and add them to the figure:
    all_traces = [box_trace]+traces

    if not traces:
        return []
    
    fig = go.Figure(data=all_traces)

    #Add hover text to the dots, remove too many digits after coma.
    fig.update_traces(hovertemplate=hover_tit+': %{x: .2e}')
        
    #more settings:
    fig.update_layout(
        yaxis_range=[-0.5,0.5],
        yaxis={'visible': False, 'showticklabels': False},
        xaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'),
        xaxis_title=y_ax_and_fig_title+" in "+unit,
        title={
        'text': y_ax_and_fig_title+' of the data for '+ch_tit+' over the entire time series',
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend_groupclick='togglegroup') #this setting allowes to select the whole group when clicking on 1 element of the group. But then you can not select only 1 element.


    description_for_user = 'Positions of points on the Y axis do not hold information, made for visialisation only.'
    qc_derivative = [QC_derivative(content=fig, name=fig_name, content_type='plotly', description_for_user = description_for_user)]

    return qc_derivative


def plot_muscle_csv(f_path: str):

    """
    Plot the muscle events with the z-scores and the threshold.
    On base of the data from tsv file.
    
    Parameters
    ----------
    f_path: str
        Path to tsv file with data.
    
        
    Returns
    -------
    fig_derivs : List
        A list of QC_derivative objects with plotly figures for muscle events.

    """

    df = pd.read_csv(f_path, sep='\t')  

    if df['scores_muscle'].empty or df['scores_muscle'].isna().all():
        return []
    
    m_or_g = df['ch_type'][0]

    fig_derivs = []

    fig=go.Figure()
    tit, _ = get_tit_and_unit(m_or_g)
    # fig.add_trace(go.Scatter(x=raw.times, y=scores_muscle, mode='lines', name='high freq (muscle scores)'))
    # fig.add_trace(go.Scatter(x=muscle_times, y=high_scores_muscle, mode='markers', name='high freq (muscle) events'))
    
    fig.add_trace(go.Scatter(x=df['data_times'], y=df['scores_muscle'], mode='lines', name='high freq (muscle scores)'))
    fig.add_trace(go.Scatter(x=df['high_scores_muscle_times'], y=df['high_scores_muscle'], mode='markers', name='high freq (muscle) events'))
    
    # #removed threshold, so this one is not plotted now:
    #fig.add_trace(go.Scatter(x=raw.times, y=[threshold_muscle]*len(raw.times), mode='lines', name='z score threshold: '+str(threshold_muscle)))
    fig.update_layout(xaxis_title='time, (s)', yaxis_title='zscore', title={
    'text': "Muscle z scores (high fequency artifacts) over time based on "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

    fig_derivs += [QC_derivative(fig, 'muscle_z_scores_over_time_based_on_'+tit, 'plotly', 'Calculation is done using MNE function annotate_muscle_zscore(). It requires a z-score threshold, which can be changed in the settings file. (by defaults 5). Values over this threshold are marked in red.')]
    
    return fig_derivs



def plot_muscle_annotations_mne(raw: mne.io.Raw, m_or_g: str, annot_muscle: mne.Annotations = None, interactive_matplot:bool = False):

    '''
    Currently not used since cant be added into HTML report

    '''
    # View the annotations (interactive_matplot)

    tit, _ = get_tit_and_unit(m_or_g)
    fig_derivs = []
    if interactive_matplot is True:
        order = np.arange(144, 164)
        raw.set_annotations(annot_muscle)
        fig2=raw.plot(start=5, duration=20, order=order)
        #Change settings to show all channels!

        # No suppressing of plots should be done here. This one is matplotlib interactive plot, so it ll only work with %matplotlib qt.
        # Makes no sense to suppress it. Also, adding to QC_derivative is just formal, cos whe extracting to html it s not interactive any more. 
        # Should not be added to report. Kept here in case mne will allow to extract interactive later.

        fig_derivs += [QC_derivative(fig2, 'muscle_annotations_'+tit, 'matplotlib')]
    
    return fig_derivs

    
def plot_head_pos_csv(f_path: str):

    """ 
    Plot positions and rotations of the head. On base of data from tsv file.
    
    Parameters
    ----------
    f_path: str
        Path to a file with data.
        
    Returns
    -------
    head_derivs : List 
        List of QC_derivative objects containing figures with head positions and rotations.
    head_pos_baselined : np.ndarray
        Head positions and rotations starting from 0 instead of the mne detected starting point. Can be used for plotting.
    """

    head_pos = pd.read_csv(f_path, sep='\t') 

    #drop first column. cos index is being created as an extra column when transforming from csv back to df:
    head_pos.drop(columns=head_pos.columns[0], axis=1, inplace=True)

    # Check if all specified columns are empty or contain only NaN values
    columns_to_check = ['x', 'y', 'z', 'q1', 'q2', 'q3']
    if head_pos[columns_to_check].isna().all().all() or head_pos[columns_to_check].empty:
        return [],[]

    #plot head_pos using PLOTLY:

    # First, for each head position subtract the first point from all the other points to make it always deviate from 0:
    head_pos_baselined=head_pos.copy()
    #head_pos_baselined=head_pos_degrees.copy()
    for column in ['x', 'y', 'z', 'q1', 'q2', 'q3']:
        head_pos_baselined[column] -= head_pos_baselined[column][0]

    t = head_pos['t']

    fig1p = make_subplots(rows=3, cols=2, subplot_titles=("Position (mm)", "Rotation (quat)"))

    names_pos=['x', 'y', 'z']
    names_rot=['q1', 'q2', 'q3']
    for counter in [0, 1, 2]:
        position=1000*-head_pos[names_pos[counter]]
        #position=1000*-head_pos_baselined[names_pos[counter]]
        fig1p.add_trace(go.Scatter(x=t, y=position, mode='lines', name=names_pos[counter]), row=counter+1, col=1)
        fig1p.update_yaxes(title_text=names_pos[counter], row=counter+1, col=1)
        rotation=head_pos[names_rot[counter]]
        #rotation=head_pos_baselined[names_rot[counter]]
        fig1p.add_trace(go.Scatter(x=t, y=rotation, mode='lines', name=names_rot[counter]), row=counter+1, col=2)
        fig1p.update_yaxes(title_text=names_rot[counter], row=counter+1, col=2)

    fig1p.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig1p.update_xaxes(title_text='Time (s)', row=3, col=2)

    head_derivs = [QC_derivative(fig1p, 'Head_position_rotation_average_plotly', 'plotly', description_for_user = 'The green horizontal lines - original head position. Red lines - the new head position averaged over all the time points.')]

    return head_derivs, head_pos_baselined


def make_head_pos_plot_mne(raw: mne.io.Raw, head_pos: np.ndarray):

    """
    Currently not used if we wanna plot solely from csv. 
    This function requires also raw as input and cant be only from csv.

    TODO: but we can calculate these inputs earlier and add them to csv as well.

    """

    original_head_dev_t = mne.transforms.invert_transform(
        raw.info['dev_head_t'])
    average_head_dev_t = mne.transforms.invert_transform(
        compute_average_dev_head_t(raw, head_pos))
    
    matplotlib.use('Agg') #this command will suppress showing matplotlib figures produced by mne. They will still be saved for use in report but not shown when running the pipeline

    #plot using MNE:
    fig1 = mne.viz.plot_head_positions(head_pos, mode='traces')
    #fig1 = mne.viz.plot_head_positions(head_pos_degrees)
    for ax, val, val_ori in zip(fig1.axes[::2], average_head_dev_t['trans'][:3, 3],
                        original_head_dev_t['trans'][:3, 3]):
        ax.axhline(1000*val, color='r')
        ax.axhline(1000*val_ori, color='g')
        #print('___MEGqc___: ', 'val', val, 'val_ori', val_ori)
    # The green horizontal lines represent the original head position, whereas the
    # Red lines are the new head position averaged over all the time points.


    head_derivs = [QC_derivative(fig1, 'Head_position_rotation_average_mne', 'matplotlib', description_for_user = 'The green horizontal lines - original head position. Red lines - the new head position averaged over all the time points.')]

    return head_derivs


def make_head_annots_plot(raw: mne.io.Raw, head_pos: np.ndarray):

    """
    Plot raw data with annotated head movement. Currently not used.

    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    head_pos : np.ndarray
        Head positions and rotations.
        
    Returns
    -------
    head_derivs : List
        List of QC derivatives with annotated figures.
        
    """

    head_derivs = []

    mean_distance_limit = 0.0015  # in meters
    annotation_movement, hpi_disp = annotate_movement(
        raw, head_pos, mean_distance_limit=mean_distance_limit)
    raw.set_annotations(annotation_movement)
    fig2=raw.plot(n_channels=100, duration=20)
    head_derivs += [QC_derivative(fig2, 'Head_position_annot', 'matplotlib')]

    return head_derivs

#__________ECG/EOG__________#

def plot_ECG_EOG_channel_csv(f_path):

    """
    Plot the ECG channel data and detected peaks
    
    Parameters
    ----------
    f_path : str
        Path to the tsv file with the derivs to plot
        
    Returns
    -------
    ch_deriv : List
        List of QC_derivative objects with plotly figures of the ECG/EOG channels
        
    """

    #if its not the right file, skip:
    base_name = os.path.basename(f_path) #name of the fimal file
    
    if 'ecgchannel' not in base_name.lower() and 'eogchannel' not in base_name.lower():
        return []

    df = pd.read_csv(f_path, sep='\t', dtype={6: str})

    # Find the column containing the ECG/EOG data. Depending on how the TSV was
    # written, the first column may be an unnamed index column.  Hence, search
    # for a column name starting with ``ECG`` or ``EOG`` and fall back to the
    # first column if none is found.
    channel_cols = [
        col for col in df.columns
        if col.lower().startswith('ecg') or col.lower().startswith('eog')
    ]
    ch_name = channel_cols[0] if channel_cols else df.columns[0]
    ch_data = df[ch_name].values

    if not ch_data.any():  # Check if all values are falsy (0, False, or empty)
        return []
    
    peaks = df['event_indexes'].dropna()
    peaks = [int(x) for x in peaks]
    fs = int(df['fs'].dropna().iloc[0])

    time = np.arange(len(ch_data))/fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=ch_data, mode='lines', name=ch_name,
                             hovertemplate='Time: %{x} s<br>Amplitude: %{y} V<br>'))
    fig.add_trace(go.Scatter(x=time[peaks], y=ch_data[peaks], mode='markers', name='peak',
                             hovertemplate='Time: %{x} s<br>Amplitude: %{y} V<br>'))
    fig.update_layout(xaxis_title='time, s', 
                yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
                yaxis_title='Amplitude, V',
                title={
                'text': ch_name,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
    
    ch_deriv = [QC_derivative(fig, ch_name, 'plotly', fig_order = 1)]

    return ch_deriv


def figure_x_axis(df, metric):

    ''''
    Get the x axis for the plot based on the metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame with the data.
    metric : str
        The metric for which the x axis is needed. Can be 'PSD', 'ECG', 'EOG', 'Muscle', 'Head'.

    Returns
    -------
    freqs : np.ndarray
        Frequencies for the PSD plot.
    time_vec : np.ndarray
        Time vector for the ECG, EOG, Muscle, Head plots.
    
    '''
     
    if metric.lower() == 'psd':
        # Figure out frequencies:
        freq_cols = [column for column in df if column.startswith('PSD_Hz_')]
        freqs = np.array([float(x.replace('PSD_Hz_', '')) for x in freq_cols])
        return freqs
    
    elif metric.lower() == 'eog' or metric.lower() == 'ecg' or metric.lower() == 'muscle' or metric.lower() == 'head':
        if metric.lower() == 'ecg':
            prefix = 'mean_ecg_sec_'
        elif metric.lower() == 'eog': 
            prefix = 'mean_eog_sec_'
        elif metric.lower() == 'smoothed_ecg' or metric.lower() == 'ecg_smoothed':
            prefix = 'smoothed_mean_ecg_sec_'
        elif metric.lower() == 'smoothed_eog' or metric.lower() == 'eog_smoothed':
            prefix = 'smoothed_mean_eog_sec_'
        elif metric.lower() == 'muscle':
            prefix = 'Muscle_sec_'
        elif metric.lower() == 'head':
            prefix = 'Head_sec_'
        
        time_cols = [column for column in df if column.startswith(prefix)]
        time_vec = np.array([float(x.replace(prefix, '')) for x in time_cols])

        return time_vec
    
    else:
        print('Oh well IDK! figure_x_axis()')
        return None
    

def split_affected_into_3_groups_csv(df: pd.DataFrame, metric: str, split_by: str = 'similarity_score' or 'corr_coeff'):

    """
    Collect artif_per_ch into 3 lists - for plotting:
    - a third of all channels that are the most correlated with mean_rwave
    - a third of all channels that are the least correlated with mean_rwave
    - a third of all channels that are in the middle of the correlation with mean_rwave

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with the data.
    metric : str
        The metric for which the x axis is needed. Can be 'ECG' or 'EOG'.
    split_by : str
        The metric by which the channels will be split. Can be 'corr_coeff' or 'similarity_score'.

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

    if metric.lower() != 'ecg' and metric.lower() != 'eog':
        print('Wrong metric in split_affected_into_3_groups_csv()')

    #sort the data frame by the correlation coefficient or similarity score and split into 3 groups:
    df_sorted = df.reindex(df[metric.lower()+'_'+split_by].abs().sort_values(ascending=False).index)

    total_rows = len(df_sorted)
    third = total_rows // 3

    most_affected = df_sorted.copy()[:third]
    middle_affected = df_sorted.copy()[third:2*third]
    least_affected = df_sorted.copy()[2*third:]

    #find the correlation value of the last channel in the list of the most correlated channels:
    # this is needed for plotting correlation values, to know where to put separation rectangles.
    val_of_last_most_affected = max(most_affected[metric.lower()+'_'+split_by].abs().tolist())
    val_of_last_middle_affected = max(middle_affected[metric.lower()+'_'+split_by].abs().tolist())
    val_of_last_least_affected = max(least_affected[metric.lower()+'_'+split_by].abs().tolist())

    return most_affected, middle_affected, least_affected, val_of_last_most_affected, val_of_last_middle_affected, val_of_last_least_affected


def plot_affected_channels_csv(df, artifact_lvl: float, t: np.ndarray, m_or_g: str, ecg_or_eog: str, title: str, flip_data: bool or str = 'flip', smoothed: bool = False):

    """
    Plot the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.
    Based on the data from tsv file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame with the data.
    artifact_lvl : float
        The threshold for the artifact amplitude.
    t : np.ndarray
        Time vector.
    m_or_g : str
        Either 'mag' or 'grad'.
    ecg_or_eog : str
        Either 'ECG' or 'EOG'.
    title : str
        The title of the figure.
    flip_data : bool
        If True, the absolute value of the data will be used for the calculation of the mean artifact amplitude. Default to 'flip'. 
        'flip' means that the data will be flipped if the peak of the artifact is negative. 
        This is donr to get the same sign of the artifact for all channels, then to get the mean artifact amplitude over all channels and the threshold for the artifact amplitude onbase of this mean
        And also for the reasons of visualization: the artifact amplitude is always positive.
    smoothed: bool
        Plot smoothed data (true) or nonrmal (false)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure with the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.

        
    """

    fig_tit=ecg_or_eog.upper()+title

    if df is not None:
        if smoothed is True:
            metric = ecg_or_eog+'_smoothed'
        elif smoothed is False:
            metric = ecg_or_eog
        fig = plot_ch_df_as_lines_by_lobe_csv(None, metric, t, m_or_g, df)

        if fig is None:
            return go.Figure()

        #decorate the plot:
        ch_type_tit, unit = get_tit_and_unit(m_or_g)
        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Mean magnitude in '+unit,
            title={
                'text': fig_tit+str(len(df))+' '+ch_type_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


    else:
        fig=go.Figure()
        ch_type_tit, _ = get_tit_and_unit(m_or_g)
        title=fig_tit+'0 ' +ch_type_tit
        fig.update_layout(
            title={
            'text': title,
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top'})
        
    #in any case - add the threshold on the plot
    #TODO: remove threshold?
    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), line=dict(color='red'), name='Thres=mean_peak/norm_lvl')) #add threshold level

    if flip_data is False and artifact_lvl is not None: 
        fig.add_trace(go.Scatter(x=t, y=[(-artifact_lvl)]*len(t), line=dict(color='black'), name='-Thres=mean_peak/norm_lvl'))

    return fig


def plot_mean_rwave_csv(f_path: str, ecg_or_eog: str):

    """
    Plon mean rwave(ECG) or mean blink (EOG) from data in CSV file.


    Parameters
    ----------
    f_path: str
        Path to csv file
    ecg_or_eog: str
        plot ECG or EOG data

    Returns
    -------
    fig_derivs : List
        list with one QC_derivative object, which contains the plot.


    """

    #if it s not the right ch kind in the file
    base_name = os.path.basename(f_path) #name of the final file
    if ecg_or_eog.lower() + 'channel' not in base_name.lower():
        return []

    # Load the data from the .tsv file into a DataFrame
    df = pd.read_csv(f_path, sep='\t', dtype={6: str})

    if df['mean_rwave'].empty or df['mean_rwave'].isna().all():
        return []

    # Set the plot's title and labels
    if 'recorded' in df['recorded_or_reconstructed'][0].lower():
        which = ' recorded'
    elif 'reconstructed' in df['recorded_or_reconstructed'][0].lower():
        which = ' reconstructed'
    else:
        which = ''
    
    #TODO: can there be the case that no shift was done and column is empty? should not be...
    # Create a scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter (x=df['mean_rwave_time'], y=df['mean_rwave'], mode='lines', name='Original '+ ecg_or_eog.upper(),
        hovertemplate='Time: %{x} s<br>Amplitude: %{y} V<br>'))
    if ecg_or_eog.lower() == 'ecg':
        fig.add_trace(go.Scatter (x=df['mean_rwave_time'], y=df['mean_rwave_shifted'], mode='lines', name='Shifted ' + ecg_or_eog.upper(),
        hovertemplate='Time: %{x} s<br>Amplitude: %{y} V<br>'))

    if ecg_or_eog.lower() == 'ecg':
        plot_tit = 'Mean' + which + ' R wave was shifted to align with the ' + ecg_or_eog.upper() + ' signal found on MEG channels.'
        annot_text = "The alignment is necessary for performing Pearson correlation between ECG signal found in each channel and reference mean signal of the ECG recording."
    elif ecg_or_eog.lower() == 'eog':
        plot_tit = 'Mean' + which + ' blink signal'
        annot_text = ""

    fig.update_layout(
            xaxis_title='Time, s',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Amplitude, V',
            title={
                'text': plot_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            annotations=[
                dict(
                x=0.5,
                y=-0.25,
                showarrow=False,
                text=annot_text,
                xref="paper",
                yref="paper",
                font=dict(size=12),
                align="center"
        )])

    mean_ecg_eog_ch_deriv = [QC_derivative(fig, ecg_or_eog+'mean_ch_data', 'plotly', fig_order = 2)]

    return mean_ecg_eog_ch_deriv


def plot_artif_per_ch_3_groups(f_path: str, m_or_g: str, ecg_or_eog: str, flip_data: bool):

    """
    This is the final function.
    Plot average artifact for each channel, colored by lobe, 
    channels are split into 3 separate plots, based on their correlation with mean_rwave: equal number of channels in each group.
    Based on the data from tsv file.

    Parameters
    ----------
    f_path : str
        Path to the tsv file with data.
    m_or_g : str
        Type of the channel: mag or grad
    ecg_or_eog : str
        Type of the artifact: ECG or EOG
    flip_data : bool
        Use True or False, doesnt matter here. It is only passed into the plotting function and influences the threshold presentation. But since treshold is not used in correlation method, this is not used.

    Returns
    -------
    artif_per_ch : List
        List of objects of class Avg_artif
    affected_derivs : List
        List of objects of class QC_derivative (plots)
    

    """

    #if its not the right file, skip:
    base_name = os.path.basename(f_path) #name of the fimal file
    
    if 'desc-ecgs' not in base_name.lower() and 'desc-eogs' not in base_name.lower():
        return []


    ecg_or_eog = ecg_or_eog.lower()

    df = pd.read_csv(f_path, sep='\t') #TODO: maybe remove reading csv and pass directly the df here?
    
    df = df.drop(df[df['Type'] != m_or_g].index) #remove non needed channel kind

    artif_time_vector = figure_x_axis(df, metric=ecg_or_eog)

    most_similar, mid_similar, least_similar, _, _, _ = split_affected_into_3_groups_csv(df, ecg_or_eog, split_by='similarity_score')

    smoothed = True
    fig_most_affected = plot_affected_channels_csv(most_similar, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' most affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed)
    fig_middle_affected = plot_affected_channels_csv(mid_similar, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' moderately affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed)
    fig_least_affected = plot_affected_channels_csv(least_similar, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' least affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed)


    #set the same Y axis limits for all 3 figures for clear comparison:

    if ecg_or_eog.lower() == 'ecg' and smoothed is False:
        prefix = 'mean_ecg_sec_'
    elif ecg_or_eog.lower() == 'ecg' and smoothed is True:
        prefix = 'smoothed_mean_ecg_sec_'
    elif ecg_or_eog.lower() == 'eog' and smoothed is False:
        prefix = 'mean_eog_sec_'
    elif ecg_or_eog.lower() == 'eog' and smoothed is True:
        prefix = 'smoothed_mean_eog_sec_'

    cols = [column for column in df if column.startswith(prefix)]
    cols = ['Name']+cols

    limits_df = df[cols]

    ymax = limits_df.loc[:, limits_df.columns != 'Name'].max().max()
    ymin = limits_df.loc[:, limits_df.columns != 'Name'].min().min()

    ylim = [ymin*.95, ymax*1.05]

    # update the layout of all three figures with the same y-axis limits
    fig_most_affected.update_layout(yaxis_range=ylim)
    fig_middle_affected.update_layout(yaxis_range=ylim)
    fig_least_affected.update_layout(yaxis_range=ylim)
    
    m_or_g_order = 0.1 if m_or_g == 'mag' else 0.2
    affected_derivs = []
    affected_derivs += [QC_derivative(fig_most_affected, ecg_or_eog+'most_affected_channels_'+m_or_g, 'plotly', fig_order = 3.01+m_or_g_order)] #for exaple for mage we get: 3.11
    affected_derivs += [QC_derivative(fig_middle_affected, ecg_or_eog+'middle_affected_channels_'+m_or_g, 'plotly', fig_order = 3.02+m_or_g_order)]
    affected_derivs += [QC_derivative(fig_least_affected, ecg_or_eog+'least_affected_channels_'+m_or_g, 'plotly', fig_order = 3.03+m_or_g_order)]

   
    return affected_derivs


def plot_correlation_csv(f_path: str, ecg_or_eog: str, m_or_g: str):

    """
    Plot correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    Based on the data from tsv file.

    Parameters
    ----------
    f_path : str
        Path to the tsv file with data.
    ecg_or_eog : str
        Either 'ECG' or 'EOG'.
    m_or_g : str
        Either 'mag' or 'grad'.

    Returns
    -------
    corr_derivs : List
        List with 1 QC_derivative instance: Figure with correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    
    """

    #if its not the right file, skip:
    base_name = os.path.basename(f_path) #name of the fimal file
    
    if 'desc-ecgs' not in base_name.lower() and 'desc-eogs' not in base_name.lower():
        return []

    ecg_or_eog = ecg_or_eog.lower()

    df = pd.read_csv(f_path, sep='\t') #TODO: maybe remove reading csv and pass directly the df here?
    df = df.drop(df[df['Type'] != m_or_g].index) #remove non needed channel kind

    _, _, _, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated = split_affected_into_3_groups_csv(df, ecg_or_eog, split_by='corr_coeff')

    traces = []

    tit, _ = get_tit_and_unit(m_or_g)

    # for index, row in df.iterrows():
    #     traces += [go.Scatter(x=[abs(row[ecg_or_eog.lower()+'_corr_coeff'])], y=[row[ecg_or_eog.lower()+'_pval']], mode='markers', marker=dict(size=5, color=row['Lobe Color']), name=row['Name'], legendgroup=row['Lobe Color'], legendgrouptitle=dict(text=row['Lobe'].upper()), hovertemplate='Corr coeff: '+str(row[ecg_or_eog.lower()+'_corr_coeff'])+'<br>p-value: '+str(abs(row[ecg_or_eog.lower()+'_pval'])))]


    for index, row in df.iterrows():
        traces += [go.Scatter(x=[abs(row[ecg_or_eog.lower()+'_corr_coeff'])], y=[row[ecg_or_eog.lower()+'_pval']], mode='markers', marker=dict(size=5, color=row['Lobe Color']), name=row['Name'], legendgroup=row['Lobe Color'], legendgrouptitle=dict(text=row['Lobe'].upper()), hovertemplate='Corr coeff: '+str(row[ecg_or_eog.lower()+'_corr_coeff'])+'<br>p-value: '+str(abs(row[ecg_or_eog.lower()+'_pval'])))]

    if not traces:
        return []
    
    # Create the figure with the traces
    fig = go.Figure(data=traces)

    # # Reverse the x and y axes
    # fig.update_xaxes(autorange="reversed")
    # fig.update_yaxes(autorange="reversed")


    fig.add_shape(type="rect", xref="x", yref="y", x0=0, y0=-0.1, x1=corr_val_of_last_least_correlated, y1=1.1, line=dict(color="Green", width=2), fillcolor="Green", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_least_correlated, y0=-0.1, x1=corr_val_of_last_middle_correlated, y1=1.1, line=dict(color="Yellow", width=2), fillcolor="Yellow", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_middle_correlated, y0=-0.1, x1=1, y1=1.1, line=dict(color="Red", width=2), fillcolor="Red", opacity=0.1)

    fig.update_layout(
        title={
            'text': tit+': Pearson correlation between reference '+ecg_or_eog.upper()+' epoch and '+ecg_or_eog.upper()+' epoch in each channel',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title='Correlation coefficient',
        yaxis_title = 'P-value')

    m_or_g_order = 0.1 if m_or_g == 'mag' else 0.2
    corr_derivs = [QC_derivative(fig, 'Corr_values_'+ecg_or_eog, 'plotly', description_for_user='Absolute value of the correlation coefficient is shown here. The sign would only represent the position of the channel towards magnetic field. <p>- Green: 33% of all channels that have the weakest correlation with mean ' +ecg_or_eog +'; </p> <p>- Yellow: 33% of all channels that have mild correlation with mean ' +ecg_or_eog +';</p> <p>- Red: 33% of all channels that have the stronges correlation with mean ' +ecg_or_eog +'. </p>', fig_order = 4+m_or_g_order)]

    return corr_derivs


def plot_mean_rwave_shifted(mean_rwave_shifted: np.ndarray, mean_rwave: np.ndarray, ecg_or_eog: str, tmin: float, tmax: float):
    
    """
    Only for demonstartion while running the pipeline. Dpesnt go into final report.

    Plots the mean ECG wave and the mean ECG wave shifted to align with the ECG artifacts found on meg channels.
    Probably will not be included into the report. Just for algorythm demosntration.
    The already shifted mean ECG wave is plotted in the report.

    Parameters
    ----------
    mean_rwave_shifted : np.ndarray
        The mean ECG wave shifted to align with the ECG artifacts found on meg channels.
    mean_rwave : np.ndarray
        The mean ECG wave, not shifted, original.
    ecg_or_eog : str
        'ECG' or 'EOG'
    tmin : float
        The start time of the epoch.
    tmax : float
        The end time of the epoch.

    Returns
    -------
    fig_derivs : List
        list with one QC_derivative object, which contains the plot. (in case want to input intot he report)
    
    """

    t = np.linspace(tmin, tmax, len(mean_rwave_shifted))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=mean_rwave_shifted, mode='lines', name='mean_rwave_shifted'))
    fig.add_trace(go.Scatter(x=t, y=mean_rwave, mode='lines', name='mean_rwave'))

    fig.show()

    #fig_derivs = [QC_derivative(fig, 'Mean_artifact_'+ecg_or_eog+'_shifted', 'plotly')] 
    # #activate is you want to output the shift demonstration to the report, normally dont'
    
    fig_derivs = []

    return fig_derivs


def plot_ecg_eog_mne(channels: dict, ecg_epochs: mne.Epochs, m_or_g: str, tmin: float, tmax: float):

    """
    Plot ECG/EOG artifact with topomap and average over epochs (MNE plots based on matplotlib)

    NOT USED NOW

    Parameters
    ----------
    channels : dict
        Dictionary  with ch names divided by mag/grad
    ecg_epochs : mne.Epochs
        ECG/EOG epochs.
    m_or_g : str
        String 'mag' or 'grad' depending on the channel type.
    tmin : float
        Start time of the epoch.
    tmax : float
        End time of the epoch.
    
    Returns
    -------
    mne_ecg_derivs : List
        List of QC_derivative objects with MNE plots.
    
    
    """

    mne_ecg_derivs = []
    fig_ecg = ecg_epochs.plot_image(combine='mean', picks = channels[m_or_g])[0] #plot averageg over ecg epochs artifact
    # [0] is to plot only 1 figure. the function by default is trying to plot both mag and grad, but here we want 
    # to do them saparetely depending on what was chosen for analysis
    mne_ecg_derivs += [QC_derivative(fig_ecg, 'mean_ECG_epoch_'+m_or_g, 'matplotlib')]

    #averaging the ECG epochs together:
    avg_ecg_epochs = ecg_epochs.average() #.apply_baseline((-0.5, -0.2))
    # about baseline see here: https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py

    #plot average artifact with topomap
    fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[tmin-tmin/100, tmin/2, 0, tmax/2, tmax-tmax/100], picks = channels[m_or_g])
    # tmin+tmin/10 and tmax-tmax/10 is done because mne sometimes has a plotting issue, probably connected tosamplig rate: 
    # for example tmin is  set to -0.05 to 0.02, but it  can only plot between -0.0496 and 0.02.

    mne_ecg_derivs += [QC_derivative(fig_ecg_sensors, 'ECG_field_pattern_sensors_'+m_or_g, 'matplotlib')]

    return mne_ecg_derivs