# -*- coding: utf-8 -*-
"""
Last modified on Feb 21, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Université Paris-Saclay, France

Contains the following utility functions for program PyMaGra:

    - extract : Mark data segments for treatment as function of odd or even
        line number
    - data_flatten : Copy dictionary data into Numpy 1D array
    - clean_data : Mute (set to nan) data values outside a user-defined range
    - julian2date : Transform julian day number to calender date
    - date2julian : Transform calender date to julian day number
    - time2data : Convert seconds from start of year into julian day and time
    - diurnal_variation : Estimate diurnal variations from line medians
    - diurnal_correction : Correct diurnal variations of magnetic data
    - interpol_line : Interpolate data along a line onto a regular grid
    - interpol_2D : Interpolate data onto a regular 2D grid
    - extrapolate : Extrapolate data to fille a full rectangle
    - justify_lines_median : Eliminate directional effect by median adjustment
    - gauss_transform : Calculate Gaussian statistics parameters
    - justify_lines_gaussian : Eliminate directional effect by adjusting
        Gaussian statistics
    - next_minute : Calculate next full minute after a given time
    - add_time : Add a time to a given one (in hour, minute, second)
    - fit2lines : Fit two straight lines to a data series
    - min_max : Calculate positions of local maxima and minima in a data series
    - min_max2D : Calculate position of local maxima and minima on a 2D grid
    - get_date_string : Transform actual date and time into a string
    - file_name : Create a file name containing the actual date and time

"""

import os
from copy import deepcopy
from datetime import datetime
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression as LR
from PyQt5 import QtWidgets


def extract(data, choice):
    """
    Mark data segments for treatment as function of odd or even line number

    Parameters
    ----------
    choice : tuple of str
        may be:

        - "all" (choose all lines)
        - "odd" (choose odd lines, natural counting)
        - "even" (choose even line numbers, natural counting)
        - "N", "S", "W" or "E"

    """
    if choice == "all":
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            val["mask"] = True
        data.line_choice = "all"
    else:
        data.line_choice = choice
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            if choice == "odd":
                if key % 2 == 1:
                    val["mask"] = True
                else:
                    val["mask"] = False
            elif choice == "even":
                if key % 2 == 0:
                    val["mask"] = True
                else:
                    val["mask"] = False


def data_flatten(data):
    """
    Data that are stored in dictionary data with one entrance per line are
    concatenated into one 1D numpy array

    Parameters
    ----------
    data : dictionary with keys equal to line number
        contains for every line itself a dictionary with the following
        keys:

        - "s1" : Data of sensor 1
        - "s2" : Data of sensor 2
        - "x", "y", "z": Coordinates of data

    Returns
    -------
    s1 : Numpy 1D float array
         Concatenated data of sensor 1
    s2 : Numpy 1D float array
         Concatenated data of sensor 2
    x  : Numpy 1D float array
         Concatenated x-coordinates of all data
    y  : Numpy 1D float array
         Concatenated y-coordinates of all data
    z  : Numpy 1D float array
         Concatenated z-coordinates of all data
    """
    s1 = []
    s2 = []
    x = []
    y = []
    z = []
    for key, val in data.items():
        if isinstance(key, (str)):
            break
        s1 += list(val["s1"])
        s2 += list(val["s2"])
        x += list(val["x"])
        y += list(val["y"])
        z += list(val["z"])
    return s1, s2, x, y, z


def clean_data(data, min_fix=None, max_fix=None, percent_down=None,
               percent_up=None):
    """
    Set data to np.nan under certain conditions which may be:

    Parameters
    ----------
    data : object of class data
        Contains data to be cleaned (see io.get_line).
    min_fix : float
        All data below this value are set to nan.
    max_fix : float
        All data above this value are set to nan.
    percent_down : float
        The lowermost percentile values are set to nan
        A value of 0.01 means that all values lower than the
        1% quantile are set to None.
    percent_up : float
        The uppermost percentile values are set to nan
        A value of 0.01 means that all values higher than the
        99% quantile are set to None.

    Returns
    -------
    data : dictionary
        Same structure as input data, but with with data outside defined
        limits set to np.nan
    """
    print("\n")
    grad_data = data.grad_data
    s1 = data.sensor1
    s2 = data.sensor2
    if min_fix:
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            v = np.copy(val["s1"])
            v[v < min_fix] = np.nan
            val["s1"] = np.copy(v)
            if grad_data:
                v = np.copy(val["s2"])
                v[v < min_fix] = np.nan
                val["s2"] = np.copy(v)
        print(f"Clip below {np.round(min_fix, 1)}")
    if max_fix:
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            v = np.copy(val["s1"])
            v[v > max_fix] = np.nan
            val["s1"] = np.copy(v)
            if grad_data:
                v = np.copy(val["s2"])
                v[v > max_fix] = np.nan
                val["s2"] = np.copy(v)
        print(f"Clip above {np.round(max_fix, 1)}")
    if percent_down:
        vmin1 = np.nanquantile(s1, percent_down)
        if grad_data:
            vmin2 = np.nanquantile(s2, percent_down)
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            v = np.copy(val["s1"])
            v[v < vmin1] = np.nan
            val["s1"] = np.copy(v)
            if grad_data:
                v = np.copy(val["s2"])
                v[v < vmin2] = np.nan
                val["s2"] = np.copy(v)
        if grad_data:
            print(f"Clip below {np.round(vmin1, 1)} for sensor 1 and "
                  + f"{np.round(vmin2, 1)} for sensor 2")
        else:
            print(f"Clip below {np.round(vmin1, 1)}")
    if percent_up:
        vmax1 = np.nanquantile(s1, 1.0 - percent_up)
        if grad_data:
            vmax2 = np.nanquantile(s2, 1.0 - percent_up)
        for key, val in data.data.items():
            if isinstance(key, (str)):
                break
            v = np.copy(val["s1"])
            v[v > vmax1] = np.nan
            val["s1"] = np.copy(v)
            if grad_data:
                v = np.copy(val["s2"])
                v[v > vmax2] = np.nan
                val["s2"] = np.copy(v)
        if grad_data:
            print(f"Clip above {np.round(vmax1, 1)} for sensor 1 and "
                  + f"{np.round(vmax2, 1)} for sensor 2")
        else:
            print(f"Clip above {np.round(vmax1, 1)}")
    data.sensor1, data.sensor2, _, _, _ = data_flatten(data.data)


def julian2date(j_day, year):
    """
    Function translates Julian day number to standard date.
    1st of January is Julian day number 1.

    Parameters
    ----------
    j_day : int
        Number of Julian day
    year : int
        Year in which to do the calculation (important to know whether
        it is a leap year). May be 2 or 4 ciphers

    Returns
    -------
    day: int
        Day of month
    month: int
        Month in year
    """
    day_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
                          334])
    if year % 4 == 0:
        day_month[2:] += 1
    month = np.where(day_month >= j_day)[0][0]
    day = j_day - day_month[month - 1]
    return day, month


def date2julian(day, month, year):
    """
    Function translates month and day of month to Julian day of year.
    1st of January is Julian day number 1.

    Parameters
    ----------
    day : int
        Day of month (natural counting, starts at 1)
    month : int
        Month of year (natural counting, starts at 1 for January)
    year : int
        Year in which to do the calculation (important to know whether
        it is a leap year). May be 2 or 4 ciphers

    Returns
    -------
    j_day: int
        Julian day of year
    """
    day_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
                          334])
    if year % 4 == 0:
        day_month[2:] += 1
    return day_month[month - 1] + day


def time2data(time, year):
    """
    Convert seconds into julian day of year, hour, minute and second

    Parameters
    ----------
    time : may be a single float or a 1D numpy array
        Time to be converted [s], based on julian day of year, i.e.
        time = julian_day*86000+hour*3600+minute*60+second.

    Returns
    -------
    All returned parameters have the same shape as time.

    month :  int
        month of year
    day : int
        day of month.
    h : int
        Hour.
    m : int
        minute.
    s : float
        second.

    """
    try:
        _ = len(time)
        d = np.array(time / 86400.0, dtype=int)
        t = time - d * 86400.0
        h = np.array(t / 3600.0, dtype=int)
        t -= h * 3600.0
        m = np.array(t / 60.0, dtype=int)
        s = t - m * 60.0
        day = np.zeros_like(d)
        month = np.zeros_like(d)
        for i, dd in enumerate(d):
            day[i], month[i] = julian2date(dd, year)
    except TypeError:
        day = int(time / 86400.0)
        t = time - day * 86400.0
        h = int(t / 3600.0)
        t -= h * 3600.0
        m = int(t / 60.0)
        s = t - m * 60.0
        day, month = julian2date(day, year)
    return month, day, h, m, s


def diurnal_variation(times, data, degree=3):
    """
    Calculate a fit of degree "degree" to the data of lines "lines" which
    will be used for correction of diurnal variations if not base station
    data exist

    Parameters
    ----------
    times : 1D numpy float array
        Times for the data to be fitted in seconds
    data : 1D numpy float array
        Data to be fitted (usually a series of line medians)
    degree : int, optional
        Degree of polynom to be fitted to data. The default is 3.

    Returns
    -------
    Polynome coefficients (1D numpy array of size degree+1)
        The polynome is calculated as
        P[degree]+P[degree-1]*x+P[degree-2]*x**2...+P[0]*x**degree
        If multiple blocks are fitted together, P contains a polynome for
        each block, i.e. len(P) = (degree+1)*number_of_blocks
    tmn : float
        For the stability of polynome fit, times (given in seconds) are
        reduced such that the minimum time is zero. tmn is this minumum
        time. To apply the coefficients to data, their time must be
        transformed to time-tmn before applying the polynome coefficients.

    """
    tmn = times.min()
    return np.polyfit(times - tmn, data, deg=degree), tmn


def diurnal_correction(data_c, base, base_flag=True, degree=5):
    """
    Apply diurnal corrections
    The diurnal variations may come from base station data (function
    read_base) or, if no base station data exist, they are calculated in
    function diurnal_variation by fitting a polynomial of degree "degree"
    to the measured data. Data of different days are then fitted
    independently.

    Base station data (measured or calculated) are interpolated onto
    measurement times and simply subtracted from data. The process is done
    in situ, i.e. the values of arrays self.sensor1 and self.sensor2 are
    modified. If you need to keep the original data, you must copy them to
    other arrays before applying diurnal_correction

    Parameters
    ----------
    data : Member of class data
        Contains data to be corrected
    base : Member of class Geometrics
        Constains base station data if there are or stores extimated ones
    base_flag : bool, optional; Default: True
        If True, base station data were read from file. Else, estimate
        diurnal variations from line medians
    degree : int, optional, Default: 3
        Degree of polynom to be fitted to data.
        This parameter is only used if no base station data exist

    """
# Test whether base station data have already been read
    data = deepcopy(data_c.data)
# If no base station data exist, use median values of sensor 1 of every
#    line asindicator for diurnal variations.
# Add median value at the time of the beginning and the end of every line
    if not base_flag:
        month_base = []
        day_base = []
        year_base = []
        hour_base = []
        minute_base = []
        second_base = []
        time_base = []
        jday_base = []
        base_d = np.array([])
        year = data["year"]
        t = np.array([])
        v = np.array([])
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            t = np.concatenate((t, val["time"]))
            v = np.concatenate((v, val["s1"]))
        index = np.where(np.isfinite(v))[0]
        t = t[index]
        v = v[index]
        days = np.array(t / 86400.0, dtype=int)
        days_u = np.unique(days)
        params = []
        tm = []
        tdiurnal = np.array([])
        for d in days_u:
            index = np.where(days == d)
            fit_parameters, tmn = diurnal_variation(t[index], v[index],
                                                    degree=degree)
            params.append(fit_parameters)
            tm.append(tmn)
            t0 = t[index].min()
            t1 = t[index].max()
            tt = np.arange(t0 - 60.0, t1 + 100.0, 60.0) - tmn
            npar = len(fit_parameters)
            vv = np.zeros(len(tt))
            for i in range(npar):
                vv += fit_parameters[i] * tt ** (degree - i)
            tdiurnal = np.concatenate((tdiurnal, tt + tmn))
            base_d = np.concatenate((base_d, vv))
        for tim in tdiurnal:
            mon, da, ho, mi, se = time2data(tim, year)
            year_base.append(year)
            month_base.append(mon)
            day_base.append(da)
            hour_base.append(ho)
            minute_base.append(mi)
            second_base.append(se)
            time_base.append(tim)
            jday_base.append(date2julian(int(da), int(mon), int(year)))
        month_base = np.array(month_base, dtype=int)
        day_base = np.array(day_base, dtype=int)
        jday_base = np.array(jday_base, dtype=int)
        year_base = np.array(year_base, dtype=int)
        hour_base = np.array(hour_base, dtype=int)
        minute_base = np.array(minute_base, dtype=int)
        second_base = np.array(second_base)
        time_base = np.array(time_base)
        with open("temp.stn", "w", encoding="utf-8") as fo:
            for i, b in enumerate(base_d):
                if np.isnan(b):
                    continue
                fo.write(f"*  0 {jday_base[i]:3d} {hour_base[i]:02d}"
                         + f"{minute_base[i]:02d}{int(second_base[i]):02d}"
                         + f"{i:5d}{int(b*10):7d}\n")
        base.read_base("temp.stn", year)
        os.remove("temp.stn")
    else:
        time1 = []
        time2 = []
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            time1.append(min(val["time"][0], val["time"][-1]))
            time2.append(max(val["time"][0], val["time"][-1]))
        time1 = np.array(time1)
        time2 = np.array(time2)
# Check whether base station data cover all measurements.
# If not, set base station data to zero and avoid in this way base
# corrections
        if base.time_base.min() > time1.min() or \
                base.time_base.max() < time2.max():
            print("\nWARNING in function  diurnal_correction:")
            print("   Base station data do not cover all measurements.")
            print("           No base station corrections effectuated.\n")
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "Function utilities.diurnal_correction:\n\n"
                + "Base station data do not cover all measurements\n"
                + f"Base station between day {base.time_base.min()/86400.} "
                + f"and {base.time_base.max()/86400.}\n"
                + "Instrument data between day "
                + f"{time1.min()/86400.} and {time2.max()/86400.}\n\n"
                + "No base station corrections effectuated.",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return False
# interpolate base station values at measurement times
    tb, ind = np.unique(base.time_base, return_index=True)
    b = base.base[ind]
    f = interpolate.interp1d(tb, b, kind=1)
    for key, val in data.items():
        if isinstance(key, (str)):
            break
        diurnal = f(val["time"])
# Subtract base station data from field data
        val["s1"] -= diurnal
        val["median1"] = np.nanmedian(val["s1"])
        if data["grad_data"]:
            val["s2"] -= diurnal
            val["median2"] = np.nanmedian(val["s2"])
    data_c.data = deepcopy(data)
    return True


def interpol_line(data, nsensor, i_line=0, dx=0.2, xmin=0.0, xmax=0.0, k=3):
    """
    interpolate data of one line onto a regular grid

    Parameters
    ----------
    data : Member of class data
        data to be interpolated
    i_line : int, optional. Default is 0.
        Number of line to be interpolated (counting starts at 0).
    dx : float, optional
        Sampling step in meters for interpolated data. The default is 0.2.
    xmin : float, optional. Default is 0.
        Position of first sample along self.direction in meters.
    xmax : float, optional. Default is 0.
        Position of last sample along self.direction in meters.
    k : int, optional
        Degree of spline used for interpolation. The default is 3.
        See scipy.interpolate.interp1d for more information.
        Only splines are used. Correspondance between k and "kind" of
        scipy.interpolate.interp1d:

        - k=0: kind="zero"
        - k=1: kind="slinear"
        - k=2: kind="quadratic"
        - k=3: kind="cubic"

    If xmin == xmax, the starting and end points are calculated
    automatically. For this,the starting point is placed at the nearest
    multiple of dx for the coordinate of self.direction (see function
    read_stn)

    Returns
    -------
    sensor_inter: numpy float array
        Interpolated data
    x_inter: numpy float array
        Interpolated X-coordinates
    y_inter: numpy float array
        Interpolated Y-coordinates
    dmin: float
        Position of first interpolated point within line [m]
    dmax: float
        Position of last interpolated point within line [m]

    """
    kind = ["zero", "slinear", "quadratic", "cubic"]
    if k > 3 or k < 0:
        print("\nWARNING Function interpol_line:\n"
              f"      Given k ({k}) is not allowed. k is set to 3 (cubic)")
        k = 3
# Extract data
    data = data.data
    if nsensor == 1:
        s1 = data[i_line]["s1"]
    else:
        s1 = data[i_line]["s2"]
    index = np.isfinite(s1)
    s1 = s1[index]
    xdat = data[i_line]["x"]
    xdat = xdat[index]
    ydat = data[i_line]["y"]
    ydat = ydat[index]
# Define coordinates in principal direction along which to interpolate data
    if data["direction"] in ("N", "S", 0.0, 180.0):
        ddat = np.copy(xdat)
    else:
        ddat = np.copy(ydat)
# Define starting and end points
    if xmin == xmax:
        dmin = np.ceil(np.round(ddat.min(), 2) / dx) * dx
        dmax = np.floor(np.round(ddat.max(), 2) / dx) * dx
    else:
        dmin = xmin
        dmax = xmax
# Calculate number of interpolated data and their positions along the principal
#   direction
    nx = np.int((dmax - dmin) / dx + 1)
    d_inter = dx * np.arange(nx) + dmin
# Do interpolation for first sensor
    f = interpolate.interp1d(ddat, s1, kind=kind[k], fill_value="extrapolate")
    sensor_inter = f(d_inter)
# Do interpolation for X-coordinates
    f = interpolate.interp1d(ddat, xdat, kind=kind[k],
                             fill_value="extrapolate")
    x_inter = f(d_inter)
# Do interpolation for Y-coordinates
    f = interpolate.interp1d(ddat, ydat, kind=kind[k],
                             fill_value="extrapolate")
    y_inter = f(d_inter)
    return sensor_inter, x_inter, y_inter, dmin, dmax


def interpol_2D(data_c, dx=0.2, dy=0.2, fill_hole=False):
    """
    Routine interpolates data on all lines onto a regular grid. No
    extrapolation is done, i.e. if at the beginning or the end of a line
    data are missing (the line starts later than others or stops earlier),
    the interpolated array will contain nans
    The interpolation method used is
    scipy.interpolate.CloughTocher2DInterpolator

    Parameters
    ----------
    data_c : member of class data
        data to be interpolated
    dx : float, optional
        Sampling step in meters in x-direction. The default is 0.2.
    dy : float, optional
        Sampling step in meters in y-direction. The default is 0.2.
    fill_hole : bool, optional. Default False
        If True, internal missing grid points are interpolated, however not
        external ones (misisng points at the edge of the grid).
        If False, only small holes (up to 3 grid points) are interpolated,
        others are associated np.nan

    Returns
    -------
    sensor1_inter : 2D numpy float array
        Contains gridded data of sensor 1
    sensor2_inter : 2D numpy float array
        Contains gridded data of sensor 2
    grad_inter :2D numpy float array
        contains the vertical gradient

    The shape of the arrays depends on the principal direction of the
    lines:
    - If self.direction == 1, shape is (number_of_data_points_per_line,
      number_of_lines)
    - else: (number_of_lines, number_of_data_points_per_line)

    x_inter : 1D numpy float array
        x_coordinates of the columns of s1_inter and s2_inter
    y_inter : 1D numpy float array
        y_coordinates of the rows of s1_inter and s2_inter
    topo_inter

    """
    if fill_hole:
        n_hole = 1000000
    else:
        n_hole = 3
    data = data_c.data
    if data_c.topo_flag:
        topo = data_c.topo
        z = data_c.z
    keys = []
    for key in data.keys():
        if isinstance(key, (str)):
            break
        keys.append(int(key))
    keys = np.array(keys)
    key1 = keys[0]
    direction = data[key1]["direction"]
    if direction in ("N", "S", 0.0, 180.0):
        direction = "N"
    else:
        direction = "E"
    if data_c.grad_data:
        nsensor = 2
    else:
        nsensor = 1
# search all line positions. If different blocks were joint, it is possible
# that pieces of one the same line were measured in different blocks and these
# pieces should be joint before interpolation
    pos_l = []
    for k in keys:
        if direction == "N":
            pos_l.append(np.nanmedian(data[k]["x"]))
        else:
            pos_l.append(np.nanmedian(data[k]["y"]))
    pos_l = np.array(pos_l)
    line_positions = np.unique(pos_l)
    nl = len(line_positions)
# Define grid area covering the whole measured area
    xmin = data[key1]["x"].min()
    xmax = data[key1]["x"].max()
    ymin = data[key1]["y"].min()
    ymax = data[key1]["y"].max()
    for key, val in data.items():
        if isinstance(key, (str)):
            break
        xmin = min(val["x"].min(), xmin)
        xmax = max(val["x"].max(), xmax)
        ymin = min(val["y"].min(), ymin)
        ymax = max(val["y"].max(), ymax)
    xmin = np.ceil(xmin / dx) * dx
    xmax = np.floor(xmax / dx) * dx
    ymin = np.ceil(ymin / dy) * dy
    ymax = np.floor(ymax / dy) * dy
    x_inter = xmin + np.arange(int((xmax - xmin) / dx) + 1) * dx
    y_inter = ymin + np.arange(int((ymax - ymin) / dy) + 1) * dy
    xi, yi = np.meshgrid(x_inter, y_inter)
    ny, nx = xi.shape
# set arrays for interpolation and predefine certain values to nan
    if direction == "N":
        posx = np.zeros((ny, nl))
        p_inter = y_inter
        p2_inter = x_inter
        nin = ny
        ncross = nx
        dp = dy
        dp2 = dx
    else:
        posx = np.zeros((nx, nl))
        p_inter = x_inter
        p2_inter = y_inter
        nin = nx
        ncross = ny
        dp = dx
        dp2 = dy
    zint = np.zeros_like(posx)
    s1int = np.zeros_like(posx)
    s2int = np.zeros_like(posx)
    tim = np.zeros_like(posx)
    tint = np.zeros_like(posx)
    posx[:, :] = np.nan
    zint[:, :] = np.nan
    tint[:, :] = np.nan
    s1int[:, :] = np.nan
    s2int[:, :] = np.nan
    for iline, lpos in enumerate(line_positions):
        x = pos = z = t = s1 = s2 = topo = np.array([])
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            if direction == "N":
                if not np.isclose(np.nanmedian(val["x"]), lpos):
                    continue
                x = np.concatenate((x, val["x"]))
                pos = np.concatenate((pos, val["y"]))
            else:
                if not np.isclose(np.nanmedian(val["y"]), lpos):
                    continue
                x = np.concatenate((x, val["y"]))
                pos = np.concatenate((pos, val["x"]))
            z = np.concatenate((z, val["z"]))
            t = np.concatenate((t, val["time"]))
            s1 = np.concatenate((s1, val["s1"]))
            topo = np.concatenate((topo, val["topo"]))
            if nsensor == 2:
                s2 = np.concatenate((s2, val["s2"]))
        pos, pos_ind = np.unique(pos, return_index=True)
        x = x[pos_ind]
        z = z[pos_ind]
        s1 = s1[pos_ind]
        t = t[pos_ind]
        topo = topo[pos_ind]
        index = np.where(np.isfinite(s1))[0]
        pi = pos[index]
        pm = pos.min()
        n1 = np.where(p_inter >= pm)[0][0]
        pm = pos.max()
        n2 = np.where(p_inter <= pm)[0][-1] + 1
        f = interpolate.interp1d(pi, s1[index], kind="linear")
        s1int[n1:n2, iline] = f(p_inter[n1:n2])
        f = interpolate.interp1d(pi, x[index], kind="linear")
        posx[n1:n2, iline] = f(p_inter[n1:n2])
        f = interpolate.interp1d(pi, z[index], kind="linear")
        zint[n1:n2, iline] = f(p_inter[n1:n2])
        f = interpolate.interp1d(pi, t[index], kind="linear")
        tim[n1:n2, iline] = f(p_inter[n1:n2])
        f = interpolate.interp1d(pi, topo[index], kind="linear")
        tint[n1:n2, iline] = f(p_inter[n1:n2])

# Check whether there are big holes in data along the line (>10*step size)
        hole = np.where(abs(pi[1:] - pi[:-1]) > 10.0 * dp)[0]
        if len(hole > 0):
            for h in hole:
                n1 = int((pi[h] - p_inter[0]) / dp)
                n2 = int((pi[h + 1] - p_inter[0]) / dp) + 1
                s1int[n1:n2, iline] = np.nan
                posx[n1:n2, iline] = np.nan
                zint[n1:n2, iline] = np.nan
                tint[n1:n2, iline] = np.nan
        if nsensor == 2:
            s2 = s2[pos_ind]
            index = np.where(np.isfinite(s2))[0]
            pi = pos[index]
            pm = pi.min()
            n1 = np.where(p_inter >= pm)[0][0]
            pm = pi.max()
            n2 = np.where(p_inter <= pm)[0][-1] + 1
            f = interpolate.interp1d(pi, s2[index], kind="linear")
            s2int[n1:n2, iline] = f(p_inter[n1:n2])
            hole = np.where(abs(pi[1:] - pi[:-1]) > 10.0 * dp)[0]
            if len(hole > 0):
                for h in hole:
                    n1 = int((pi[h] - p_inter[0]) / dp)
                    n2 = int((pi[h + 1] - p_inter[0]) / dp) + 1
                    s2int[n1:n2, iline] = np.nan
    z_inter = np.zeros((ncross, nin))
    z_inter[:, :] = np.nan
    s1_inter = np.zeros_like(z_inter)
    s1_inter[:, :] = np.nan
    s2_inter = np.zeros_like(z_inter)
    s2_inter[:, :] = np.nan
    t_inter = np.zeros_like(z_inter)
    topo_inter = np.zeros_like(z_inter)
    topo_inter[:, :] = np.nan
    for i in range(len(p_inter)):
        pos, pos_ind = np.unique(posx[i, :], return_index=True)
        s1 = s1int[i, :][pos_ind]
        t = tim[i, :][pos_ind]
        index = np.where(np.isfinite(s1))[0]
        if len(index) < 1:
            continue
        i1 = index[0]
        i2 = index[-1] + 1
        pi = pos[i1:i2]
        pm = pi.min()
        n1 = np.where(p2_inter >= pm)[0][0]
        pm = pi.max()
        n2 = np.where(p2_inter <= pm)[0][-1] + 1
        pi = pi[index]
        s1 = s1[index]
        n = i2 - i1
        if n < 3:
            f = interpolate.interp1d(pi, s1[i1:i2], kind="linear")
        else:
            f = interpolate.interp1d(pi, s1[i1:i2], kind="linear")
        s1_inter[n1:n2, i] = f(p2_inter[n1:n2])
        z = zint[i, :]
        z = z[pos_ind]
        if n < 3:
            f = interpolate.interp1d(pi, z[i1:i2], kind="linear")
        else:
            f = interpolate.interp1d(pi, z[i1:i2], kind="linear")
        z_inter[n1:n2, i] = f(p2_inter[n1:n2])
        if n < 3:
            f = interpolate.interp1d(pi, t[i1:i2], kind="linear")
        else:
            f = interpolate.interp1d(pi, t[i1:i2], kind="linear")
        t_inter[n1:n2, i] = f(p2_inter[n1:n2])
        top = tint[i, :]
        top = top[pos_ind]
        if n < 3:
            f = interpolate.interp1d(pi, top[i1:i2], kind="linear")
        else:
            f = interpolate.interp1d(pi, top[i1:i2], kind="linear")
        topo_inter[n1:n2, i] = f(p2_inter[n1:n2])
# Check whether there are big holes in data along the line (>3*line distance)
        if len(index) > 1:
            hole = np.where(pos_ind[1:] - pos_ind[:-1] > n_hole)[0]
            if len(hole > 0):
                for ih, h in enumerate(hole):
                    n1 = int((pos[pos_ind[h]] - p2_inter[0]) / dp2) + 1
                    n2 = int((pos[pos_ind[h] + 1] - p2_inter[0]) / dp2)
                    s1_inter[n1:n2, i] = np.nan
                    z_inter[n1:n2, i] = np.nan
                    topo_inter[n1:n2, i] = np.nan
        if nsensor == 2:
            s2 = s2int[i, :][pos_ind]
            index = np.where(np.isfinite(s2))[0]
            i1 = index[0]
            i2 = index[-1] + 1
            pi = pos[i1:i2]
            pm = pi.min()
            n1 = np.where(p2_inter >= pm)[0][0]
            pm = pos[i1:i2].max()
            n2 = np.where(p2_inter <= pm)[0][-1] + 1
            pi = pi[index]
            s2 = s2[index]
            if n < 3:
                f = interpolate.interp1d(pi, s2[i1:i2], kind="linear")
            else:
                f = interpolate.interp1d(pi, s2[i1:i2], kind="linear")
            s2_inter[n1:n2, i] = f(p2_inter[n1:n2])
            if len(index) > 1:
                hole = np.where(pos_ind[1:] - pos_ind[:-1] > n_hole)[0]
                if len(hole > 0):
                    for ih, h in enumerate(hole):
                        n1 = int((pos[pos_ind[hole[0]]]-p2_inter[0])/dp2)
                        n2 = int((pos[pos_ind[hole[0]]+1]-p2_inter[0])/dp2)
                        s2_inter[n1+1:n2, i] = np.nan
    if data[key1]["direction"] in ("N", "S", 0.0, 180.0):
        s1_inter = s1_inter.T
        z_inter = z_inter.T
        t_inter = t_inter.T
        topo_inter = topo_inter.T
        if nsensor == 2:
            s2_inter = s2_inter.T
    if nsensor == 2:
        grad_inter = (s2_inter - s1_inter) / data["d_sensor"]
        return (s1_inter, s2_inter, grad_inter, x_inter, y_inter, z_inter,
                t_inter, topo_inter)
    dum = 0
    return s1_inter, dum, dum, x_inter, y_inter, z_inter, t_inter, topo_inter


def extrapolate(d, x, y):
    """
    Routine fills nans on an interpolated grid.
    For this, it searches first for every line and column the first and
    last existing (non-nan) points. Then, for every non-defined point, it
    searches the "n_nearest" nearest points (see first command line) and
    associates a weighted average value. The weight is calculated as
    1/distance**2

    Parameters
    ----------
    d : 2D numpy array, shape: (ny,nx)
           Contains data on regular grid. NaN for inexistant data.
    x : 1D numpy array
        Coordiantes of the columns of data
    y : 1D numpy array
        Coordiantes of the rows of data

    Returns
    -------
    data: 2D numpy array with the same shape as input data.
        Contains full regular grid of data

    """
    n_nearest = 8
    data = np.copy(d)
    XX, YY = np.meshgrid(x, y)
# Search all points at the beginning and the end of all rows
    zlim = []
    xlim = []
    ylim = []
    for i in range(len(x)):
        j1 = len(y)
        j2 = 0
        for j in range(len(y)):
            if np.isfinite(data[j, i]):
                j1 = j
                break
        for j in range(len(y) - 1, -1, -1):
            if np.isfinite(data[j, i]):
                j2 = j
                break
        if j1 <= j2:
            zlim.append(data[j1, i])
            xlim.append(XX[j1, i])
            ylim.append(YY[j1, i])
            zlim.append(data[j2, i])
            xlim.append(XX[j2, i])
            ylim.append(YY[j2, i])
# Search all points at the beginning and the end of all columns
    for j in range(len(y)):
        i1 = len(x)
        i2 = 0
        for i in range(len(x)):
            if np.isfinite(data[j, i]):
                i1 = i
                break
        for i in range(len(x) - 1, -1, -1):
            if np.isfinite(data[j, i]):
                i2 = i
                break
        if i1 <= i2:
            zlim.append(data[j, i1])
            xlim.append(XX[j, i1])
            ylim.append(YY[j, i1])
            zlim.append(data[j, i2])
            xlim.append(XX[j, i2])
            ylim.append(YY[j, i2])
    zlim = np.array(zlim)
    xlim = np.array(xlim)
    ylim = np.array(ylim)
# Do extrapolation
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            if np.isnan(data[j, i]):
                dist = 1 / ((xx - xlim) ** 2 + (yy - ylim) ** 2)
                ind = np.argsort(dist)
                data[j, i] = np.dot(
                    dist[ind[-n_nearest:]], zlim[ind[-n_nearest:]]
                ) / np.sum(dist[ind[-n_nearest:]])
    return data


def justify_lines_median(data_c, just=0, inplace=True):
    """
    Often the measurment direction has an influence on magnetic data due to
    uncorrected effects of acquisition instrumentation.
    The function calculates the median values of every line and adjusts the
    one of every second line to the average median of the neighbouring
    lines

    Parameters
    ----------
    just : int, optional. Default is 0.
        If 0: Leave medians of even line (python counting, i.e. first line
        is even) untouched, justify odd lines to medians of even lines.
        If 1: Do the reverse
    inplace : bool, optional. Default is True
        if True, justified data are back-copied to self.sensorN_inter and
        True is returned. If not, new arrays are returned.

    Returns
    -------
    s1_justified : 1D numpy array with justified data of first sensor
    s2_justified : 1D numpy array with justified data of second sensor

    """
    data = deepcopy(data_c.data)
    keys = []
    for key in data.keys():
        if isinstance(key, str):
            break
        keys.append(key)
    keys = np.array(keys)
    max_key = keys.max()
    if just == 0:
        d_change = data[1]["direction"]
        d_keep = data[0]["direction"]
    else:
        d_change = data[0]["direction"]
        d_keep = data[1]["direction"]
        # for day in self.lines_per_day.keys():
    for key, val in data.items():
        k = keys[1]
        k1 = keys[1]
        k2 = keys[1]
        if isinstance(key, str):
            break
        if val["direction"] == d_change:
            if key == 0:
                for k in keys[1:]:
                    if data[k]["direction"] == d_keep:
                        break
                dm1 = data[key]["median1"] - data[k]["median1"]
                if data["grad_data"]:
                    dm2 = data[key]["median2"] - data[k]["median2"]
            elif key == max_key:
                for k in keys[-1::-1] == d_keep:
                    break
                dm1 = data[key]["median1"] - data[k]["median1"]
                if data["grad_data"]:
                    dm2 = data[key]["median2"] - data[k]["median2"]
            else:
                for k1 in keys[key+1:]:
                    if data[k1]["direction"] == d_keep:
                        break
                for k2 in keys[key-1::-1]:
                    if data[k2]["direction"] == d_keep:
                        break
                dm1 = (data[key]["median1"]
                       - (data[k1]["median1"] + data[k2]["median1"]) / 2.0)
                if data["grad_data"]:
                    dm2 = (data[key]["median2"]
                           - (data[k1]["median2"] + data[k2]["median2"]) / 2.0)
            val["s1"] -= dm1
            val["median1"] -= dm1
            if data["grad_data"]:
                val["s2"] -= dm2
                val["median2"] -= dm2
    if inplace:
        data_c.data = deepcopy(data)
        return True
    return data


def gauss_transform(data_fix, data_move):
    """
    Transforms data sets to gaussian distribution does a projection
    of the second data set onto the distribution of the first and returns
    the back-transformed modified second data set

    Parameters
    ----------
    data_fix : numpy 1D array
        Reference data set.
    data_move : numpy 1D array
        Data set to be projected onto the gaussian distribution of
        data_fix.

    Returns
    -------
    numpy 1D array
        Modified data_move array.

    """
# For the number of quantiles take the number of data of the smaller data set
    n = min(len(data_fix), len(data_move))
# It seems that the number of quantiles in
# sklearn.preprocessing.QuantileTransformer is limited to 10000.
    n = min(n, 10000)
# Do the Gauss-transform of the reference data set
    GT_fix = QuantileTransformer(n_quantiles=n, output_distribution="normal")
    _ = GT_fix.fit_transform(data_fix)[:, 0]
# Do the Gauss-transform of the data set to be modified
    GT_move = QuantileTransformer(n_quantiles=n, output_distribution="normal")
    v_move = GT_move.fit_transform(data_move)
# Project data_move onto the Gauss distribution of data_fix and return the
#     back-transformed data.
    return GT_fix.inverse_transform(v_move)[:, 0]


def justify_lines_gaussian(data, just=0, local=1, inplace=True):
    """
    Often the measurment direction has an influence on magnetic data due to
    uncorrected effects of acquisition instrumentation.
    The function calculates the median values of every line and adjusts the
    one of every second line to the average median of the neighbouring
    lines

    Parameters
    ----------
    just : int, optional. Default is 0.
        If 0: Leve medians of even line (python counting, i.e. first line
        is even) untouched, justify odd lines to medians of even lines
        If 1: Do the reverse
    local : int, optional. Default is 1
        If 0: apply gaussian transform to the whole data set
        If 1: apply gaussian transform only to neighboring lines
    inplace : bool, optional. default is True
        If True, justified data are back-copied to self.sensorN_inter and
        True is returned. If not, new arrays are returned

    Returns
    -------
    s1_justified : 2D numpy float array
        Justified data of first sensor
    s2_justified : 2D numpy float array
        Justified data of second sensor

    """
    s1_justified = np.copy(data.sensor1_inter)
    s2_justified = np.copy(data.sensor2_inter)

    if local:
        if just == 0:
            if data.direction == 0:
                nlines = data.sensor1_inter.shape[1]
                for i in range(1, nlines, 2):
                    data_fix = data.sensor1_inter[:, i-1:i+2:2].reshape(-1, 1)
                    data_move = data.sensor1_inter[:, i].reshape(-1, 1)
                    s1_justified[:, i] = gauss_transform(data_fix, data_move)
                    data_fix = data.sensor2_inter[:, i-1:i+2:2].reshape(-1, 1)
                    data_move = data.sensor2_inter[:, i].reshape(-1, 1)
                    s2_justified[:, i] = gauss_transform(data_fix, data_move)
            else:
                nlines = data.sensor1_inter.shape[0]
                for i in range(1, nlines, 2):
                    data_fix = data.sensor1_inter[i-1:i+2:2, :].reshape(-1, 1)
                    data_move = data.sensor1_inter[i, :].reshape(-1, 1)
                    s1_justified[i, :] = gauss_transform(data_fix, data_move)
                    data_fix = data.sensor2_inter[i-1:i+2:2, :].reshape(-1, 1)
                    data_move = data.sensor2_inter[i, :].reshape(-1, 1)
                    s2_justified[i, :] = gauss_transform(data_fix, data_move)
        else:
            if data.direction == 0:
                nlines = data.sensor1_inter.shape[1]
                data_fix = data.sensor1_inter[:, 1].reshape(-1, 1)
                data_move = data.sensor1_inter[:, 0].reshape(-1, 1)
                s1_justified[:, 0] = gauss_transform(data_fix, data_move)
                data_fix = data.sensor2_inter[:, 1].reshape(-1, 1)
                data_move = data.sensor2_inter[:, 0].reshape(-1, 1)
                s2_justified[:, 0] = gauss_transform(data_fix, data_move)
                for i in range(2, nlines, 2):
                    data_fix = data.sensor1_inter[:, i-1:i+2:2].reshape(-1, 1)
                    data_move = data.sensor1_inter[:, i].reshape(-1, 1)
                    s1_justified[:, i] = gauss_transform(data_fix, data_move)
                    data_fix = data.sensor2_inter[:, i-1:i+2:2].reshape(-1, 1)
                    data_move = data.sensor2_inter[:, i].reshape(-1, 1)
                    s2_justified[:, i] = gauss_transform(data_fix, data_move)
            else:
                nlines = data.sensor1_inter.shape[0]
                data_fix = data.sensor1_inter[1, :].reshape(-1, 1)
                data_move = data.sensor1_inter[0, :].reshape(-1, 1)
                s1_justified[0, :] = gauss_transform(data_fix, data_move)
                data_fix = data.sensor2_inter[1, :].reshape(-1, 1)
                data_move = data.sensor2_inter[0, :].reshape(-1, 1)
                s2_justified[0, :] = gauss_transform(data_fix, data_move)
                for i in range(2, nlines, 2):
                    data_fix = data.sensor1_inter[i-1:i+2:2, :].reshape(-1, 1)
                    data_move = data.sensor1_inter[i, :].reshape(-1, 1)
                    s1_justified[i, :] = gauss_transform(data_fix, data_move)
                    data_fix = data.sensor2_inter[i-1:i+2:2, :].reshape(-1, 1)
                    data_move = data.sensor2_inter[i, :].reshape(-1, 1)
                    s2_justified[i, :] = gauss_transform(data_fix, data_move)
    else:
        if just == 0:
            if data.direction == 0:
                s = gauss_transform(data.sensor1_inter[:, 0::2].reshape(-1, 1),
                                    data.sensor1_inter[:, 1::2].reshape(-1, 1))
                s1_justified[:, 1::2] = \
                    s.reshape(data.sensor1_inter[:, 1::2].shape)
                s = gauss_transform(data.sensor2_inter[:, 0::2].reshape(-1, 1),
                                    data.sensor2_inter[:, 1::2].reshape(-1, 1))
                s2_justified[:, 1::2] = \
                    s.reshape(data.sensor2_inter[:, 1::2].shape)
            else:
                s = gauss_transform(data.sensor1_inter[0::2, :].reshape(-1, 1),
                                    data.sensor1_inter[1::2, :].reshape(-1, 1))
                s1_justified[1::2, :] = \
                    s.reshape(data.sensor1_inter[1::2, :].shape)
                s = gauss_transform(data.sensor2_inter[0::2, :].reshape(-1, 1),
                                    data.sensor2_inter[1::2, :].reshape(-1, 1))
                s2_justified[1::2, :] = \
                    s.reshape(data.sensor1_inter[1::2, :].shape)
        else:
            if data.direction == 0:
                s = gauss_transform(data.sensor1_inter[:, 1::2].reshape(-1, 1),
                                    data.sensor1_inter[:, 0::2].reshape(-1, 1))
                s1_justified[:, 0::2] = \
                    s.reshape(data.sensor1_inter[:, 1::2].shape)
                s = gauss_transform(data.sensor2_inter[:, 1::2].reshape(-1, 1),
                                    data.sensor2_inter[:, 0::2].reshape(-1, 1))
                s2_justified[:, 0::2] = \
                    s.reshape(data.sensor2_inter[:, 1::2].shape)
            else:
                s = gauss_transform(data.sensor1_inter[1::2, :].reshape(-1, 1),
                                    data.sensor1_inter[0::2, :].reshape(-1, 1))
                s1_justified[0::2, :] = \
                    s.reshape(data.sensor1_inter[1::2, :].shape)
                s = gauss_transform(data.sensor2_inter[1::2, :].reshape(-1, 1),
                                    data.sensor2_inter[0::2, :].reshape(-1, 1))
                s2_justified[0::2, :] = \
                    s.reshape(data.sensor1_inter[1::2, :].shape)

    if inplace:
        data.sensor1_inter = np.copy(s1_justified)
        data.sensor2_inter = np.copy(s2_justified)
        return True
    return s1_justified, s2_justified


def next_minute(hour, minute, second):
    """
    Get next full minute after actual time

    Parameters
    ----------
    hour : int
        Actual hour
    minute : int
        Actual minute
    second : float
        Actual second

    Returns
    -------
    hour : int
        hour of next minute
    minute : int
        next minute
    second : float
        0.
    """
    second = 0.0
    minute += 1
    if minute == 60:
        hour += 1
        minute = 0
    return hour, minute, second


def add_time(hour, minute, second, dt):
    """
    Add a time step to actual time

    Parameters
    ----------
    hour : int
        Actual hour
    minute : int
        Actual minute
    second : float
        Actual second
    dt : Time to be added in seconds

    Returns
    -------
    hour : int
        hour of new time
    minute : int
        minute of new time
    second : float
        second of new time
    """
    second += dt
    if second == 60.0:
        minute += 1
        second = 0.0
        if minute == 60:
            hour += 1
            minute = 0
    return hour, minute, second


def fit2lines(x, y, n0, n1, n2, n0_flag):
    """
    Fit two regression lines to a data set.
    Find the point where a break in slope gives the best fit between the
    n1th point and the n2th point

    Parameters
    ----------
    x : numpy 1D array, float
        x-coordinates of the data series.
    y : numpy 1D array, float
        y-coordinates of the data series.
    n0 : int
        first point of the series to be considered.
    n1 : int
        First point in series considered for possible slope break.
    n2 : int
        Last point in series considered for possible slope break.
    n0_flag : bool
        if True : first line must pass through point n0

    Returns
    -------
    regression coefficients for first slope
    regression coefficients for second slope
    int: position of slope beak
    float: misfit

    """
    r1_best = None
    r2_best = None
    qual = 1.0e20
    n3 = len(x)
    isplit = n1
    slopes1 = []
    slopes2 = []
    inter1 = []
    inter2 = []
    fits = []
    isplits = []
# Fit two regression lines to data. For this, search breaking point between
#     third and 11th data point for which the fit is best
    for i in range(n1, n2):
        k1 = x[n0:i+1].reshape(-1, 1)
        k2 = x[i:].reshape(-1, 1)
        if n0_flag:
            reg1 = LR(fit_intercept=False).fit(k1 - k1[0], y[n0:i+1]-y[n0])
            reg1.intercept_ = y[n0]-k1[0][0]*reg1.coef_[0]
        else:
            reg1 = LR(fit_intercept=True).fit(k1, y[n0:i+1])
        reg2 = LR(fit_intercept=True).fit(k2, y[i:])
        yy = np.zeros(n3)
        yy[n0:i+1] = k1.flatten()*reg1.coef_[0]+reg1.intercept_-y[n0:i+1]
        yy[i:] = k2.flatten()*reg2.coef_[0]+reg2.intercept_-y[i:]
        fit = np.sum(yy**2)
# If fit is better than earlier ones, calculate depths from both slopes
        slopes1.append(reg1.coef_[0])
        slopes2.append(reg2.coef_[0])
        inter1.append(reg1.intercept_)
        inter2.append(reg2.intercept_)
        fits.append(fit)
        isplits.append(i)
        if np.isfinite(fit) and fit < qual:
            qual = fit
            r1_best = reg1
            r2_best = reg2
            isplit = i
    return (r1_best, r2_best, isplit, qual, slopes1, slopes2, inter1, inter2,
            fits, isplits)


def min_max(data, half_width=3):
    """
    Find all relative minima and maxima in a data vector.

    A maximum is found if a value at position i of the vector is larger
    than or equal to all other values in a range [i-half_width:i+half_with]
    and at the same time strictly larger than all values of one side.
    Sometimes, e.g. if seismograms are saturated, a slope exists on one
    side, but the values are constant on the other side. The one-sided test
    is to avoid that a flat curve with 2*half_width constant values is also
    considered as local maximum. Equivalent scheme for definition of a
    minimum. In addition, the function reduces possible local maxima and
    minima such that a maximum is always followed by a minimum and vice
    versa. If several local maxima follow each other (i.e. no wide enough
    local minimum exists between them), the program searches the strongest
    one of the subsequent maxima or, if several equal maximum values exist,
    it takes as position of the maximum the center point between those
    multiple maxima (again for saturated curves).

    Parameters
    ----------
    data : 1D numpy float array
        Data vrctor to be analysed
    half_width : int, optional (default: 3)
        Number of samples analyzed to all ides of every data sample.

    Returns
    -------
    max_pos: 1D numpy int array
        All position numbers where there is a relative maximum in vector
        "data".
    max_val: 1D numpy float array
        Values at these positions
    min_pos: 1D numpy int array
        All position numbers where there is a relative minimum in vector
        "data"
    min_val: 1D numpy float array
        Values at these positions
    """
    N = len(data)
    NN = np.arange(N, dtype="int")
# extreme_pos (extreme_neg) will contain the sum of all values <= (>=) the
#   central value
# half will contain the maximum of the number of
#   values < (>) the central value on the left and the right side
# A maximum (minimum) is found if extreme_xxx[i]==(2*half_width+1) and if
#   half_extreme_xxx[i]==half_width.
    extreme_pos = np.zeros(N, dtype=bool)
    extreme_neg = np.zeros(N, dtype=bool)
# Start loop over data points
# Sum of neigbouring points for which value[i] <= (>=) value[test_point]
    for k in range(N):
        dn0 = min(half_width, k)
        dn1 = min(half_width, N - k - 1) + 1
        width = dn0 + dn1
        ext_pos = sum(data[k]-data[k-dn0:k+dn1] >= 0) == width
# Sum of neighbouring values to the left (half1) and right (half2) < value
# [test_point]
        half1 = sum(data[k]-data[k-dn0:k] > 0)
        half2 = sum(data[k]-data[k+1:k+dn1] > 0)
        half = max(half1, half2) == half_width
        extreme_pos[k] = ext_pos and half

# Sum of neighbouring values to the left (half1) or right (half2) > value
# [test_point]
        ext_neg = sum(data[k]-data[k-dn0:k+dn1] <= 0) == width
        half1 = sum(data[k]-data[k-dn0:k] < 0)
        half2 = sum(data[k]-data[k+1:k+dn1] < 0)
        half = max(half1, half2) == half_width
        extreme_neg[k] = ext_neg and half
# Search all points that fulfill the criteria for local maximum and minimum
#        max_pos = NN[(extreme_pos==width) & (half_extreme_pos==half_width)]
    max_pos = NN[extreme_pos]
    max_val = data[max_pos]
    min_pos = NN[extreme_neg]
    min_val = data[min_pos]
    del extreme_pos, extreme_neg
# mx_sig is a vector with length equal to number of found maxima with +1
# mn_sig is a vector with length equal to number of found maxima with -1
#   These vectors will be used to know which position is a maximum, which one
#   a minimum, once all extrema are concatenated in a single vector in order to
#   intercalate maxima and minima and to find places where multiple maxima or
#   minima follow each other
    mx_sig = np.ones(len(max_pos))
    mn_sig = -np.ones(len(min_pos))
# Concatenate positions, values and signs of maxima and minima into a single
#   vector for each of them
    signs = np.concatenate((mx_sig, mn_sig))
    positions = np.concatenate((max_pos, min_pos))
    values = np.concatenate((max_val, min_val))
# Order the concatenated vectors by positions
    iord = np.argsort(positions)
    pord = positions[iord]
    vord = values[iord]
    sord = signs[iord]
    ls = len(sord)
# Prepare lists that will contain positions, values and signs of alternating
#   extreme values (avoiding having several maxima (minima) following each
#   other without a minumum (maximum) between them).
    pos = []
    val = []
    sig = []
    i = 1
# Start loop over concatenated extreme positions
# If sign of position [i] is different from position [i-1] accept position
#   [i-1] into a new list
    while i < ls:
        if sord[i] != sord[i - 1]:
            pos.append(pord[i - 1])
            val.append(vord[i - 1])
            sig.append(sord[i - 1])
        if i == ls - 1:
            if sord[i] != sord[i - 1]:
                pos.append(pord[i])
                val.append(vord[i])
                sig.append(sord[i])
                i += 1
            break
# if sign of position i is the same as the one of position i-1 search for next
#   position where sign changes
        i1 = i + 1
        for i1 in range(i + 1, ls):
            if sord[i] != sord[i1]:
                break
        if i1 < i:
            break
# Search maximum values of the positions having the same sign
#   the chosen position is the average position of all equal maximum (minimum)
#   values. If one of the relative maxima (minima) has the strongest value, its
#   position and value will be copied into the new list.
        if sord[i] > 0:
            mx = np.where(vord[i:i1] == max(vord[i:i1]))
            mpos = int(np.mean(pord[i:i1][mx]))
            pos.append(mpos)
            val.append(max(vord[i:i1]))
            sig.append(sord[i])
        else:
            mx = np.where(vord[i:i1] == min(vord[i:i1]))
            mpos = int(np.mean(pord[i:i1][mx]))
            pos.append(mpos)
            val.append(min(vord[i:i1]))
            sig.append(sord[i])
        i = i1 + 1
    del max_pos, max_val, min_pos, min_val, iord, pord, vord, sord
# Transform lists to numpy arrays
    pos = np.array(pos)
    val = np.array(val)
    sig = np.array(sig)
# Separate again relative maxima from relative minima
    max_val = val[sig > 0]
    max_pos = pos[sig > 0]
    min_val = val[sig < 0]
    min_pos = pos[sig < 0]
    del pos, val, sig, positions, signs, values
    return max_pos, max_val, min_pos, min_val


def min_max2D(data, half_width=3):
    """
    Find all relative minima and maxima in a 2D data matrix.

    A maximum is found if a value at position i of the vector is larger
    than or equal to all other values in a range [i-half_width:i+half_with]
    in both directions independently and at the same time strictly larger
    than all values of one side.

    The function searches first all relative extrema alng every column,
    then along every row. The returned positions are the combination of
    points found in both directions. In this way ridges parallel to the
    x axis and parallel to the y axes are detected.

    Parameters
    ----------
    data : 2D numpy float array
        Data matrix to be analysed
    half_width : int, optional (default: 3)
        Number of samples analyzed to all ides of every data sample.

    Returns
    -------
    maxima : List of two 1D numpy int arrays
        All position numbers where there is a relative maximum in array
        "data".
        maxima[0]: vector of row numbers, maxima[1]: vector of column
        numberss
    minima: List of two 1D numpy int arrays
        All position numbers where there is a relative minimum in array
        "data"
        minima[0]: vector of row numbers, minima[1]: vector of column
        numbers
    """
    ny, nx = data.shape
    extreme_pos = np.zeros((ny, nx), dtype=bool)
    extreme_neg = np.zeros((ny, nx), dtype=bool)
# Loop over columns
    for k in range(nx):
        max_pos, _, min_pos, _ = min_max(data[:, k], half_width=half_width)
        extreme_pos[max_pos, k] = True
        extreme_neg[min_pos, k] = True
# Loop over columns
    for k in range(ny):
        max_pos, _, min_pos, _ = min_max(data[k, :], half_width=half_width)
        extreme_pos[k, max_pos] = True
        extreme_neg[k, min_pos] = True
    return np.where(extreme_pos), np.where(extreme_neg)


# def zero_xing2D(self, data):
#     """
#     Search zero crossings in a 2D data set

#     The zero crossing is marked in the cell to the left or below the zero
#     crossing if the value itself is not zero.

#     Parameters
#     ----------
#     data : 2D numpy float array
#         data to be analyzed

#     Returns
#     -------
#     zeroes : List of two 1D numpy int arrays
#         All position numbers where there is a zero crossing in array
#         "data". zeroes[0]: vector of row numbers, zeroes[1]: vector of
#         column numbers

#     """
#     ny, nx = data.shape
#     xing = np.zeros((ny, nx), dtype=bool)
#     for k in range(nx-1):
#         for i in range(ny-1):
#             if data[i, k] == 0.:
#                 xing[i, k] = True
#                 continue
#             if data[i, k]*data[i+1, k] < 0. or\
#                     data[i, k]*data[i, k+1] < 0.:
#                 xing[i, k] = True
#     return np.where(xing)


def get_date_string():
    """
    Gets date and time of execution of the function

    Returns
    -------
    date and time: str
        Date and time are stored as YYYY-MM-DD_hh-mm-ss

    Is used to define output names where the text is standard but time of
    creation allows distinguishing.

    """
    now = datetime.now()
    c_time = now.strftime("%H-%M-%S")
    d = now.strftime("%Y-%m-%d")
    return f"{d}_{c_time}"


def file_name(base, ext):
    """
    Creates a filename composed of a header, the date string and an
    extension : base+"yyy-mm-dd_hh-mm-ss"+ext

    Parameters
    ----------
    base : str
        Header of file name
    ext : str
        Extension of file name. Must contain the dot. Utsually, the
        extension is simply a standard extension like ".dat", but it may
        contain more text (e.g., "corrected.dat")

    Returns
    -------
    file_name : str
        Composed file name

    """
    txt = get_date_string()
    return f"{base}_{txt}_{ext}"
