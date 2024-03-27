cd 
import mne

#from source.universal_plots import get_tit_and_unit #for terminal
#from universal_plots import get_tit_and_unit #for notebook

#To avaoid import issues just added the func here from universal_plots:
#TODO: remover when import issues arer fixed.

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


def make_html_section(derivs_section: list, section_name: str, report_strings: dict):

    """
    Create 1 section of html report. 1 section describes 1 metric like "ECG" or "EOG", "Head position" or "Muscle"...
    Functions does:

    - Add section title
    - Add user notification if needed (for example: head positions not calculated)
    - Loop over list of derivs belonging to 1 section, keep only figures
    - Put figures one after another with description under. Description should be set inside of the QC_derivative object.

    Parameters
    ----------
    derivs_section : list
        A list of QC_derivative objects belonging to 1 section.
    section_name : str
        The name of the section like "ECG" or "EOG", "Head position" or "Muscle"...
    report_strings : dict
        A dictionary with strings to be added to the report: general notes + notes about every measurement (when it was not calculated, for example). 
        This is not a detailed description of the measurement.

    Returns
    -------
    html_section_str : str
        The html string of 1 section of the report.
    """

    fig_derivs_section = keep_fig_derivs(derivs_section)
    if 'report' in section_name:
        text_section_content=report_strings['INITIAL_INFO']
    elif 'Time series' in section_name:
        text_section_content="""<p>"""+report_strings['TIME_SERIES']+"""</p>"""
    elif 'ECG' in section_name:
        text_section_content="""<p>"""+report_strings['ECG']+"""</p>"""
    elif 'EOG' in section_name:
        text_section_content="""<p>"""+report_strings['EOG']+"""</p>"""
    elif 'Head' in section_name:
        text_section_content="""<p>"""+report_strings['HEAD']+"""</p>"""
    elif 'Muscle' in section_name:
        text_section_content="""<p>"""+report_strings['MUSCLE']+"""</p>"""
    elif 'Standard deviation' in section_name or 'STD' in section_name:
        text_section_content="""<p>"""+report_strings['STD']+"""</p>"""
    elif 'Frequency' in section_name or 'PSD' in section_name:
        text_section_content="""<p>"""+report_strings['PSD']+"""</p>"""
    elif 'Peak-to-Peak manual' in section_name or 'PtP_manual' in section_name :
        text_section_content="""<p>"""+report_strings['PTP_MANUAL']+"""</p>"""
    elif 'Peak-to-Peak auto' in section_name or 'PtP_auto' in section_name:
        text_section_content="""<p>"""+report_strings['PTP_AUTO']+"""</p>"""
    elif derivs_section and not fig_derivs_section:
        text_section_content="""<p>This measurement has no figures. Please see csv files.</p>"""
    elif 'Sensors' in section_name: #TODO: check if all works fine after this change
        text_section_content="""<p>""""""</p>"""
    else:
        text_section_content="""<p>""""""</p>"""

    if fig_derivs_section:
        for f in range(0, len(fig_derivs_section)):
            text_section_content += fig_derivs_section[f].convert_fig_to_html_add_description()

    html_section_str="""
        <!-- *** Section *** --->
        <center>
        <h2>"""+section_name+"""</h2>
        """ + text_section_content+"""
        <br></br>
        <br></br>
        </center>"""

    # The way to get figures if need to open them from saved files:
    # figures = {}
    # figures_report= {}
    # for x in range(0, len(fig_derivs_section)):
    #     with open(fig_derivs_section[x], 'r') as figures["f{0}".format(x)]:
    #         figures_report["f{0}".format(x)] = figures["f{0}".format(x)].read()

    return html_section_str


def keep_fig_derivs(derivs_section:list):

    """
    Loop over list of derivs belonging to 1 section, keep only figures to add to report.
    
    Parameters
    ----------
    derivs_section : list
        A list of QC_derivative objects belonging to 1 section.
        
    Returns
    -------
    fig_derivs_section : list
        A list of QC_derivative objects belonging to 1 section with only figures."""
    
    fig_derivs_section=[]
    for d in derivs_section:
        if d.content_type == 'plotly' or d.content_type == 'matplotlib':
            fig_derivs_section.append(d)

    return fig_derivs_section


def make_joined_report(sections: dict, report_strings: dict):

    """
    Create report as html string with all sections. Currently make_joined_report_mne is used.

    Parameters
    ----------
    sections : dict
        A dictionary with section names as keys and lists of QC_derivative objects as values.
    sreport_strings : dict
        A dictionary with strings to be added to the report: general notes + notes about every measurement (when it was not calculated, for example). 
        This is not a detailed description of the measurement.
    

    Returns
    -------
    html_string : str
        The html whole string of the report.
    
    """


    header_html_string = """
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>MEG QC report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            """+ report_strings['SHIELDING'] + report_strings['M_OR_G_SKIPPED'] + report_strings['EPOCHING']

    main_html_string = ''
    for key in sections:

        html_section_str = make_html_section(derivs_section = sections[key], section_name = key, report_strings = report_strings)
        main_html_string += html_section_str


    end_string = """
                     </center>
            </body>
        </html>"""


    html_string = header_html_string + main_html_string + end_string

    return html_string


def make_joined_report_mne(raw, sections:dict, report_strings: dict, default_settings: dict):

    """
    Create report as html string with all sections and embed the sections into MNE report object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.
    sections : dict
        A dictionary with section names as keys and lists of QC_derivative objects as values.
    report_strings : dict
        A dictionary with strings to be added to the report: general notes + notes about every measurement (when it was not calculated, for example). 
        This is not a detailed description of the measurement.
    default_settings : dict
        A dictionary with default settings.
    

    Returns
    -------
    report : mne.Report
        The MNE report object with all sections.
    
    """

    report = mne.Report(title=' MEG QC Report')
    # This method also accepts a path, e.g., raw=raw_path
    if raw: #if raw s not empty
        if default_settings['plot_mne_butterfly'] is True:
            report.add_raw(raw=raw, title='Raw info from MNE', psd=False, butterfly=True)  
        else:
            report.add_raw(raw=raw, title='Raw info from MNE', psd=False, butterfly=False)
        # omit PSD plot. Butterfly sets the mne plot of butterfly time series, stim channel, etc...

    for key, values in sections.items():
        if values and key != 'Report' and key != 'Report MNE' and key != 'Simple_metrics':
            html_section_str = make_html_section(derivs_section = sections[key], section_name = key, report_strings = report_strings)
            report.add_html(html_section_str, title=key)

    return report


def simple_metric_basic(metric_global_name: str, metric_global_description: str, metric_global_content_mag: dict, metric_global_content_grad: dict, metric_local_name: str =None, metric_local_description: str =None, metric_local_content_mag: dict =None, metric_local_content_grad: dict =None, display_only_global: bool =False, psd: bool=False, measurement_units: bool = True):
    
    """
    Basic structure of simple metric for all measurements.
    
    Parameters
    ----------
    metric_global_name : str
        Name of the global metric.
    metric_global_description : str
        Description of the global metric.
    metric_global_content_mag : dict
        Content of the global metric for the magnitometers as a dictionary.
        Content is created inside of the module for corresponding measurement.
    metric_global_content_grad : dict
        Content of the global metric for the gradiometers as a dictionary.
        Content is created inside of the module for corresponding measurement.
    metric_local_name : str, optional
        Name of the local metric, by default None (in case of no local metric is calculated)
    metric_local_description : str, optional
        Description of the local metric, by default None (in case of no local metric is calculated)
    metric_local_content_mag : dict, optional 
        Content of the local metric for the magnitometers as a dictionary, by default None (in case of no local metric is calculated)
        Content is created inside of the module for corresponding measurement.
    metric_local_content_grad : dict, optional
        Content of the local metric for the gradiometers as a dictionary, by default None (in case of no local metric is calculated)
        Content is created inside of the module for corresponding measurement.
    display_only_global : bool, optional
        If True, only global metric is displayed, by default False
        This parameter is set to True in case we dont need to display any info about local metric at all. For example for muscle artifacts.
        In case we want to display some notification about local metric, but not the actual metric (for example it failed to calculate for a reason), 
        this parameter is set to False and metric_local_description should contain that notification and metric_local_name - the name of missing local metric.
    psd : bool, optional
        If True, the metric is done for PSD and the units are changed accordingly, by default False
    measurement_units : bool, optional
        If True, the measurement units are added to the metric, by default True

    Returns
    -------
    simple_metric : dict
        Dictionary with the whole simple metric to be converted into json in main script.
        """
    
    _, unit_mag = get_tit_and_unit('mag', psd=psd)
    _, unit_grad = get_tit_and_unit('grad', psd=psd)

    if display_only_global is False:
       m_local = {metric_local_name: {
            "description": metric_local_description,
            "mag": metric_local_content_mag,
            "grad": metric_local_content_grad}}
    else:
        m_local = {}


    if measurement_units is True:

        simple_metric={
            'measurement_unit_mag': unit_mag,
            'measurement_unit_grad': unit_grad,
            metric_global_name: {
                'description': metric_global_description,
                "mag": metric_global_content_mag,
                "grad": metric_global_content_grad}
            }
    else:
        simple_metric={
            metric_global_name: {
                'description': metric_global_description,
                "mag": metric_global_content_mag,
                "grad": metric_global_content_grad}
            }

    #merge local and global metrics:
    simple_metric.update(m_local)

    return simple_metric