# -*- coding: utf-8 -*-
"""
Last modified on Feb 24, 2025

@author: Hermann Zeyen <hermann.zeyen@gmail.com>
         Université Paris-Saclay
"""

import numpy as np
from .dialog import dialog


def get_geometry(file, h1=None, h2=None, dispo=None, dec=None, dx=None,
                 topo=False, title=""):
    """
    Asks for measurement geometry information

    Parameters
    ----------
    file : str
        File for which geometry information is required.
    h1 : float
        Proposal for height of sensor 1 above ground.
        If None, this input is not asked for.
    h2 : float
        Proposal for height of sensor 2 above ground.
        If None, this input is not asked for.
    dispo : int
        If 0, sensors are vertically placed one above the other. If 1,
        horizontal disposition of sensors
        If None, this input is not asked for.
    dec : float
        Proposal for direction of Y axis with respect to North [degrees].
        If None, this input is not asked for.
    dx : float
        Proposal for distance between flight lines (BRGM format only)
    topo : bool
        If True, ask whether topography should be used

    """
    labels = [file]
    types = ["l"]
    values = ["None"]
    if h1 is not None:
        if h2 is not None:
            labels.append("Height of sensor 1 above ground [m]")
        else:
            labels.append("Height of sensor above ground [m]")
        types.append("e")
        values.append(h1)
    if h2 is not None:
        labels.append("Height of sensor 2 above ground [m]")
        types.append("e")
        values.append(h2)
    if dispo is not None:
        labels.append(["Vertical disposition", "Horizontal disposition"])
        types.append("r")
        values.append(dispo)
    if dec is not None:
        labels.append("Direction of Y-axis [degrees from N to E]")
        types.append("e")
        values.append(dec)
    if dx is not None:
        labels.append("Average flight line distance [m]")
        types.append("e")
        values.append(dx)
    if topo:
        labels.append("Use topography")
        types.append("c")
        values.append(1)
    labels.append("Title text")
    types.append("e")
    values.append(title)
    results, ok_button = dialog(labels, types, values,
                                title="Geometry parameters")
    if not ok_button:
        ret = [False]
        for _ in types:
            ret.append(None)
        return ret
    ret = [True]
    i = 0
    if h1 is not None:
        i += 1
        ret.append(float(results[i]))
    if h2 is not None:
        i += 1
        ret.append(float(results[i]))
    if dispo is not None:
        i += 1
        ret.append(int(results[i]))
    if dec is not None:
        i += 1
        ret.append(float(results[i]))
    if dx is not None:
        i += 1
        ret.append(float(results[i]))
    if topo:
        i += 1
        ret.append(int(results[i]) > -1)
    i += 1
    ret.append(results[i])
    return ret


def get_time_correction():
    """
    Get time shift in seconds to correct for wrong instrument time

    Returns
    -------
    float
        Time correction in seconds or None if dialogue aborted

    """
    results, okButton = dialog(
        ["GPS time (hh:mm:ss)", "Magnetometer time"],
        ["e", "e"], ["00:00:00.0", "00:00:00.0"],
        "Times")
    if not okButton:
        print("Time correction aborted")
        return None
    GPS_time = results[0].split(":")
    mag_time = results[1].split(":")
    GPS_seconds = (int(GPS_time[0])*60.+int(GPS_time[1]))*60.\
        + float(GPS_time[2])
    mag_seconds = (int(mag_time[0])*60.+int(mag_time[1]))*60.\
        + float(mag_time[2])
    return GPS_seconds-mag_seconds


def get_justify_indices(glob=False):
    """
    Get parameters for reduction of directional effects

    Parameters
    ----------
    glob : bool, optional, default. False
        If set, ask for global or local adjustment in Gauss-transform

    Returns
    -------
    None.

    """
    labels = ["Modify median of", ["Odd lines", "Even lines"]]
    types = ["l", "r"]
    values = ["None", 0]
    if glob:
        labels.append(["Global adjustment", "Line by line"])
        types.append("r")
        values.append(1)

    results, okButton = dialog(labels, types, values,
                               "Justification parameters")
    if okButton:
        ret = [True, results[1]]
        if glob:
            ret.append(int(results[2]))
    else:
        ret = [False, None]
        if glob:
            ret.append(None)
    return ret


def clip_parameters():
    """
    Ask for parameters for data clipping

    Returns
    -------
    min_fix : float
        Clip all data below this value
    max_fix : float
        Clip all data above this value
    percent_down : float
        Clip all data below the given percentile (values between 0 and 1)
    percent_up : float
        Clip all data abovz the given percentile (values between 0 and 1)
    histo : bool
        If true, limits are chosen interactively on histogram

    """
    results, okButton = dialog(
        ["Lower fixed clip value", "Upper fixed clip value",
         "Lower percentile", "upper percentile", "histogram"],
        ["e", "e", "e", "e", "c"], [None, None, 0.01, None, None],
        "Clipping parameters")
    histo = False
    if okButton:
        min_fix = None
        max_fix = None
        percent_down = None
        percent_up = None
        if results[0] != "None":
            min_fix = float(results[0])
        if results[1] != "None":
            max_fix = float(results[1])
        if results[2] != "None":
            percent_down = float(results[2])
        if results[3] != "None":
            percent_up = float(results[3])
        if min_fix:
            percent_down = None
        if max_fix:
            percent_up = None
# If extreme values should be chosen manually in histogram, do this now
        if results[4] == 0:
            histo = True
        return True, min_fix, max_fix, percent_down, percent_up, histo
    return False, None, None, None, None, None


def get_spector1D(direction, max_len):
    """
    Ask for parameters of 1D depth determination using spectral analysis

    Parameters
    ----------
    direction : int
        If 0: lines are taken in Y direction, else in X direction
    max_len : int
        Maximum length of window for spectral analysis. Actually not used.

    Returns
    -------
    result : bool
        True if OK button pressed, False if Cancel button was pressed
    direction : int
        Same as direction of input parameters
    half_width : int
        Half_width for determination of local maximum
    """
    results, okButton = dialog(
        ["Direction of analysis",
         ["N-S", "E-W"],
         "Half width for maxima determination",
         f"Window length (not yet used)\nmax in X: {max_len[1]} in "
         + f"Y: {max_len[0]}"],
        ["l", "r", "e", "e"],
        [None, direction + 1, 1, max_len[direction]])
    if okButton:
        return True, int(results[1]), int(results[2])
    return False, None, None


def get_spector2D(window_len, step, n_Nys):
    """
    Ask for parameters of 2 depth determination using spectral analysis

    Parameters
    ----------
    window_len : float
        Length of window for spectral analysis in meters
    step : float
        Step size of gliding window in meters
    n_Nys : list of two int
        Number of FFT coefficients available in Y direction (n_Ny[0]) and
        X direction (n_Ny[1])

    Returns
    -------
    result : bool
        True if OK button pressed, False if Cancel button was pressed
    window_len : float
        Same as window_len of input parameters
    step : float
        Same as step of input parameters
    n_Ny : int
        Number of FFT coefficients to be used for analysis
    """
    n_Ny = np.min(n_Nys)
    results, okButton = dialog(
        ["Window length [m]",
         "  Attention: there must be at least 16 points per window "
         + "length\n"
         + "             see below: Nr of FFT coefficients >= 8!",
         "Step size [m]",
         "Half width for maxima determination",
         f"Number of FFT coefficients\nmax in X: {n_Nys[1]}, Y: {n_Nys[0]}"],
        ["e", "l", "e", "e", "e"],
        [window_len, None, step, 1, n_Ny],
        "2D FFT parameters")
    if okButton:
        return True, float(results[0]), float(results[2]), int(results[3]), \
            int(results[4])
    return False, None, None, None, None

