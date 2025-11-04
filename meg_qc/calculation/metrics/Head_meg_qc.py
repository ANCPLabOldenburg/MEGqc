import numpy as np
import pandas as pd
import mne
import time
from typing import List
from meg_qc.plotting.universal_plots import QC_derivative
from meg_qc.calculation.initial_meg_qc import load_data


def compute_head_pos_std_and_max_rotation_movement(head_pos: np.ndarray):
    """
    Compute the standard deviation of the movement of the head over time and the maximum rotation and movement in 3 directions.

    Parameters
    ----------
    head_pos : np.ndarray
        Head positions as numpy array calculated by MNE. The shape of the array should be (n_timepoints, 10).

    Returns
    -------
    std_head_pos : float
        Standard deviation of the movement of the head over time: X, Y, Z coordinates are calculated using Pythagorean theorem to get 1 float value.
    std_head_rotations : float
        Standard deviation of the rotation of the head over time: Q1, Q2, Q3 coordinates are calculated using Pythagorean theorem to get 1 float value.
    max_movement_xyz : List
        Maximum movement amplitude in 3 directions: X, Y, Z coordinates.
    max_rotation_q : List
        Maximum rotation amplitude in 3 directions: Q1, Q2, Q3 coordinates.
    df_head_pos : pandas dataframe
        Head positions as pandas dataframe just for visualization and check.

    """

    # head positions as data frame just for visualization and check:
    df_head_pos = pd.DataFrame(head_pos, columns=['t', 'q1', 'q2', 'q3', 'x', 'y', 'z', 'gof', 'err',
                                                  'v'])  # ..., goodness of fit, error, velocity

    # get the head position in xyz coordinates:
    head_pos_transposed = head_pos.transpose()

    xyz_coords = np.array(
        [[x, y, z] for x, y, z in zip(head_pos_transposed[4], head_pos_transposed[5], head_pos_transposed[6])])
    q1q2q3_coords = np.array(
        [[q1, q2, q3] for q1, q2, q3 in zip(head_pos_transposed[1], head_pos_transposed[2], head_pos_transposed[3])])

    # Translate rotations into degrees: (360/2pi)*value . Not sure if the formila is correct.
    # q1q2q3_coords=360/(2*np.pi)*q1q2q3_coords

    # Calculate the maximum movement in 3 directions:
    max_movement_x = (np.max(xyz_coords[:, 0]) - np.min(xyz_coords[:, 0]))
    max_movement_y = (np.max(xyz_coords[:, 1]) - np.min(xyz_coords[:, 1]))
    max_movement_z = (np.max(xyz_coords[:, 2]) - np.min(xyz_coords[:, 2]))

    # Calculate the maximum rotation in 3 directions:
    rotation_coords = np.array(
        [[q1, q2, q3] for q1, q2, q3 in zip(head_pos_transposed[1], head_pos_transposed[2], head_pos_transposed[3])])
    max_rotation_q1 = (np.max(rotation_coords[:, 0]) - np.min(rotation_coords[:, 0]))
    max_rotation_q2 = (np.max(rotation_coords[:, 1]) - np.min(rotation_coords[:, 1]))
    max_rotation_q3 = (np.max(rotation_coords[:, 2]) - np.min(rotation_coords[:, 2]))
    # max_rotation_q1 = (df_head_pos['q1'].max()-df_head_pos['q1'].min()) #or like this using dataframes

    # Calculate the standard deviation of the movement of the head over time:
    # 1. Calculate the distances between each consecutive pair of coordinates. Like x2-x1, y2-y1, z2-z1
    # Use Pythagorean theorem: the distance between two points (x1, y1, z1) and (x2, y2, z2) in 3D space 
    # is the square root of (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2.
    # 2. Then calculate the standard deviation of the distances: σ = √(Σ(x_i - mean)^2 / n)

    # 1. Calculate the distances between each consecutive pair of coordinates
    distances_xyz = np.sqrt(np.sum((xyz_coords[1:] - xyz_coords[:-1]) ** 2, axis=1))
    distances_q = np.sqrt(np.sum((q1q2q3_coords[1:] - q1q2q3_coords[:-1]) ** 2, axis=1))

    # 2. Calculate the standard deviation
    std_head_pos = np.std(distances_xyz)
    std_head_rotations = np.std(distances_q)

    return std_head_pos, std_head_rotations, [max_movement_x, max_movement_y, max_movement_z], [max_rotation_q1,
                                                                                                max_rotation_q2,
                                                                                                max_rotation_q3], df_head_pos


def make_simple_metric_head(std_head_pos: float, std_head_rotations: float, max_movement_xyz: List,
                            max_rotation_q: List):
    """
    Make simple metric for head positions.

    Parameters
    ----------
    std_head_pos : float
        Standard deviation of the movement of the head over time.
    std_head_rotations : float
        Standard deviation of the rotation of the head over time.
    max_movement_xyz : List
        Maximum movement amplitude in 3 directions: X, Y, Z coordinates.
    max_rotation_q : List
        Maximum rotation amplitude in 3 directions: Q1, Q2, Q3 coordinates.

    Returns
    -------
    simple_metric : dict
        Simple metric for head positions."""

    simple_metric_details = {
        'movement_amplitude_X': max_movement_xyz[0] * 1000,
        'movement_amplitude_Y': max_movement_xyz[1] * 1000,
        'movement_amplitude_Z': max_movement_xyz[2] * 1000,
        'rotation_amplitude_Q1': max_rotation_q[0],
        'rotation_amplitude_Q2': max_rotation_q[1],
        'rotation_amplitude_Q3': max_rotation_q[2]}

    simple_metric = {
        'description': 'Head movement and rotation + their standard deviations calculated on base of 3 coordinates for each time point using Pythagorean theorem.',
        'unit_movement_xyz': 'mm',
        'unit_rotation_q1q2q3': 'quads',
        'std_movement_xyz': std_head_pos,
        'std_rotation_q1q2q3': std_head_rotations,
        'details': simple_metric_details}

    return simple_metric


def head_pos_to_csv(file_name_prefix, head_pos):
    """
    Save head positions to csv file for future visualization.

    Parameters
    ----------
    file_name_prefix : str
        Prefix for the file name. Example: 'Head'.
    head_pos : np.ndarray
        Head positions as numpy array calculated by MNE. The shape of the array should be (n_timepoints, 10).

    Returns 
    -------
    df_deriv : QC_derivative
        QC derivative with head positions.
    """

    names = ['t', 'q1', 'q2', 'q3', 'x', 'y', 'z', 'gof', 'err', 'v']
    df = pd.DataFrame(data=head_pos, columns=names)

    df_deriv = [QC_derivative(content=df, name=file_name_prefix, content_type='df')]

    return df_deriv


def get_head_positions(raw: mne.io.Raw):
    """
    Get head positions and rotations using MNE

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.

    Returns
    -------
    head_pos: np.ndarray
        Head positions and rotations calculated by MNE.
    no_head_pos_str: str
        String with information about head positions if they were not calculated, otherwise empty.

    """

    no_head_pos_str = ''
    head_pos = np.empty([0])

    try:
        # for Neuromag use (3 steps):
        chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
        # We can use mne.chpi.get_chpi_info to retrieve the coil frequencies, 
        # the index of the channel indicating when which coil was switched on, 
        # and the respective “event codes” associated with each coil’s activity.
        # Output:
        # - The frequency used for each individual cHPI coil.
        # - The index of the STIM channel containing information about when which cHPI coils were switched on.
        # - The values coding for the “on” state of each individual cHPI coil.

        print('___MEGqc___: ', f'cHPI coil frequencies extracted from raw: {chpi_freqs} Hz')

        # Estimating continuous head position
        print('___MEGqc___: ', 'Start Computing cHPI amplitudes and locations...')
        start_time = time.time()
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        print('___MEGqc___: ', "Finished. --- Execution %s seconds ---" % (time.time() - start_time))
        # print('___MEGqc___: ', 'chpi_locs:', chpi_locs)

    except:
        print('___MEGqc___: ', 'Neuromag approach to compute Head positions failed. Trying CTF approach...')
        try:
            # for CTF use:
            chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
        except:
            print('___MEGqc___: ', 'Also CTF appriach to compute Head positions failed. Trying KIT approach...')
            try:
                # for KIT use:
                chpi_locs = mne.chpi.extract_chpi_locs_kit(raw)
            except:
                print('___MEGqc___: ',
                      'Also KIT appriach to compute Head positions failed. Head positions can not be computed')
                no_head_pos_str = 'Head positions can not be computed. They can only be calculated if they have been continuously recorded during the session.'
                return head_pos, no_head_pos_str

    # Next steps - for all systems:
    print('___MEGqc___: ', 'Start computing head positions...')
    start_time = time.time()
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    print('___MEGqc___: ',
          "Finished computing head positions. --- Execution %s seconds ---" % (time.time() - start_time))
    # print('___MEGqc___: ', 'Head positions:', head_pos)

    return head_pos, no_head_pos_str


def HEAD_movement_meg_qc(data_path: str):
    """
    Main function for head movement. Calculates:

    - head positions (x, y, z) and rotations (q1, q2, q3)
    - maximum amplitude of positions and rotations
    - std of positions and rotations over whole time series: 
        1) calculate 1 value for positions and 1 value for rotations using Pythagorean theorem - for each time point.
        2) calculate std of these values and get 1 std for positions and 1 std for rotations over whole time series.


    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.

    Returns
    -------
    head_derivs : List
        List of QC derivatives with figures.
    simple_metrics_head : dict
        Dictionary with simple metrics for head movement.
    head_str : str
        String with information about head positions if they were not calculated, otherwise empty. For report
    df_head_pos : pandas dataframe
        Head positions as pandas dataframe just for visualization and check.
    head_pos : np.ndarray
        Head positions and rotations calculated by MNE.

    """
    raw, shielding_str, meg_system = load_data(data_path)

    # Compute head positions using mne:
    head_pos, head_str = get_head_positions(raw)
    if head_pos.size == 0:
        head_str = 'Head positions can not be computed. They can only be calculated if they have been continuously recorded during the session.'
        print('___MEGqc___: ', head_str)
        simple_metric_head = {'description': 'Head positions could not be computed.'}
        return [], simple_metric_head, head_str, None, None

    # Optional! translate rotation columns [1:4] in head_pos.T into degrees: (360/2pi)*value: 
    # (we assume they are in radients. But in the plot it says they are in quat! 
    # see: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)

    head_pos_degrees = head_pos.T.copy()
    for q in range(1, 4):
        head_pos_degrees[q] = 360 / (2 * np.pi) * head_pos_degrees[q]
    head_pos_degrees = head_pos_degrees.transpose()
    # Currently we dont use these.

    head_derivs = head_pos_to_csv('Head', head_pos)

    # Calculate the standard deviation of the movement of the head over time:
    std_head_pos, std_head_rotations, max_movement_xyz, max_rotation_q, df_head_pos = compute_head_pos_std_and_max_rotation_movement(
        head_pos)

    print('___MEGqc___: ', 'Std of head positions in mm: ', std_head_pos * 1000)
    print('___MEGqc___: ', 'Std of head rotations in quat: ', std_head_rotations)
    print('___MEGqc___: ', 'Max movement (x, y, z) in mm: ', [m * 1000 for m in max_movement_xyz])
    print('___MEGqc___: ', 'Max rotation (q1, q2, q3) in quat: ', max_rotation_q)

    # Make a simple metric:
    simple_metrics_head = make_simple_metric_head(std_head_pos, std_head_rotations, max_movement_xyz, max_rotation_q)

    return head_derivs, simple_metrics_head, head_str, df_head_pos, head_pos


